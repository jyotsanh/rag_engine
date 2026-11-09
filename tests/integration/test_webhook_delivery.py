"""Completion webhook delivery in real MongoDB, over real HTTP to a local receiver through
the SSRF-safe client: signing, retries and backoff, permanent failures, the delivery log,
the atomic "webhook pending" marker, and the recovery sweep for lost webhooks."""

import json
import socket
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from rag_engine.config import Settings
from rag_engine.domain import (
    MAX_WEBHOOK_LOG,
    CrawlConfig,
    CrawlJob,
    JobError,
    JobStatus,
    Tenant,
    WebhookAttempt,
    WebhookStatus,
    new_id,
    utc_now,
)
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import JobRepository, TenantRepository
from rag_engine.security import SecretBox
from rag_engine.webhooks import WebhookSecretService, verify_webhook
from rag_engine.webhooks.delivery import DeliveryOutcome, WebhookDeliverer, event_id
from rag_engine.worker.recovery import JobRecovery
from tests.conftest import OverrideSettings
from tests.support.fake_receiver import Answer, FakeReceiver

_KB = "kb_" + "1" * 32


@pytest.fixture(scope="module")
def receiver() -> Iterator[FakeReceiver]:
    server = FakeReceiver().start()
    yield server
    server.stop()


@dataclass
class _Env:
    mongo: MongoConnection
    box: SecretBox
    config: Settings
    receiver: FakeReceiver
    tenant: Tenant
    secret: str
    sent: list[str] = field(default_factory=list)

    @property
    def jobs(self) -> JobRepository:
        return JobRepository(self.mongo.db)

    def deliverer(self) -> WebhookDeliverer:
        return WebhookDeliverer(self.mongo.db, self.box, config=self.config, rng=lambda: 0.5)

    async def finished(
        self, *, url: str | None = "", status: JobStatus = JobStatus.SUCCEEDED
    ) -> CrawlJob:
        """A job of the tenant that finished with `status` (webhook to the receiver)."""
        webhook_url = self.receiver.url() if url == "" else url
        job = await self.jobs.insert(
            CrawlJob(tenant_id=self.tenant.id, kb_id=_KB, source_id=new_id("src"),
                     seed_url="https://acme.example/",
                     config=CrawlConfig(webhook_url=webhook_url))
        )  # fmt: skip
        if status is JobStatus.CANCELLED:
            return await self.jobs.request_cancel(self.tenant.id, job.id)
        if status is JobStatus.FAILED:
            return await self.jobs.fail(job.id, JobError(code="boom", message="$oops"))
        await self.jobs.start(job.id, "run-1")
        await self.jobs.begin_finalizing(job.id, "run-1")
        return await self.jobs.succeed(job.id, "run-1")


@pytest.fixture
async def env(
    integration_settings: Settings, override_settings: OverrideSettings, receiver: FakeReceiver
) -> AsyncIterator[_Env]:
    config = override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True,  # the receiver listens on loopback
        WEBHOOK_TIMEOUT_SECONDS=1,
        WEBHOOK_MAX_ATTEMPTS=3,
        WEBHOOK_RETRY_BASE_SECONDS=30,
        WEBHOOK_RETRY_MAX_SECONDS=3600,
    )
    receiver.reset()
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Tenant, CrawlJob])
    box = SecretBox.from_settings(config)
    tenants = TenantRepository(mongo.db)
    tenant = await tenants.create(Tenant(name=new_id("t")))
    tenant, secret = await WebhookSecretService(tenants, box).rotate(tenant.id)
    yield _Env(mongo, box, config, receiver, tenant, secret)
    await mongo.close()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---- #
# Tests
# ---- #


class TestWebhookPendingMarker_S83:
    """S83 — a job with a `webhook_url` owes its webhook from the terminal transition on."""

    @pytest.mark.parametrize("status", [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED])
    async def test_every_terminal_transition_marks_it_pending(
        self, env: _Env, status: JobStatus
    ) -> None:
        job = await env.finished(status=status)

        assert job.status is status and job.webhook is not None
        assert job.webhook.status is WebhookStatus.PENDING
        assert (job.webhook.attempts, job.webhook.log) == (0, [])
        assert job.webhook.touched_at == job.finished_at

    async def test_a_dead_runs_cancel_marks_it_too(self, env: _Env) -> None:
        job = await env.jobs.insert(
            CrawlJob(tenant_id=env.tenant.id, kb_id=_KB, source_id=new_id("src"),
                     seed_url="https://acme.example/",
                     config=CrawlConfig(webhook_url=env.receiver.url()))
        )  # fmt: skip
        await env.jobs.start(job.id, "run-1")
        await env.jobs.request_cancel(env.tenant.id, job.id)

        cancelled = await env.jobs.requeue(job.id, stale_run_id="run-1")

        assert cancelled.status is JobStatus.CANCELLED and cancelled.webhook is not None

    async def test_a_job_without_a_url_owes_nothing(self, env: _Env) -> None:
        job = await env.finished(url=None)

        assert job.webhook is None

    async def test_values_are_stored_literally(self, env: _Env) -> None:
        job = await env.finished(status=JobStatus.FAILED)

        assert job.error == JobError(code="boom", message="$oops")  # not a field path


class TestWebhookDelivery_S83:
    """S83 — `WebhookDeliverer.attempt`: signed POST, classification, backoff, the log."""

    async def test_a_delivery_is_signed_and_logged(self, env: _Env) -> None:
        job = await env.finished()

        result = await env.deliverer().attempt(job.id)

        [request] = env.receiver.received
        timestamp = verify_webhook(request.body, request.headers["x-ragengine-signature"],
                                   env.secret)  # fmt: skip
        payload = json.loads(request.body)
        done = await env.jobs.get_or_raise(job.id)
        assert result.outcome is DeliveryOutcome.DELIVERED and result.status_code == 204
        assert request.method == "POST" and request.path == "/hook"
        assert request.headers["content-type"] == "application/json"
        assert request.headers["user-agent"] == env.config.WEBHOOK_USER_AGENT
        assert request.headers["x-ragengine-event"] == "job.succeeded"
        assert request.headers["x-ragengine-delivery"] == event_id(job) == payload["id"]
        assert request.headers["x-ragengine-attempt"] == "1"
        assert abs(timestamp - utc_now().timestamp()) < 60
        assert payload["type"] == "job.succeeded"
        assert payload["data"]["job"]["id"] == job.id
        assert payload["data"]["job"]["status"] == "succeeded"
        assert set(payload["data"]["job"]) >= {"counters", "usage", "error", "report"}
        assert done.webhook is not None and done.webhook.status is WebhookStatus.DELIVERED
        assert done.webhook.delivered_at is not None and done.webhook.attempts == 1
        [entry] = done.webhook.log
        assert (entry.attempt, entry.status_code, entry.error) == (1, 204, None)

    async def test_a_transient_failure_retries_with_a_fresh_signature(self, env: _Env) -> None:
        job = await env.finished()
        env.receiver.answer(Answer(500))

        first = await env.deliverer().attempt(job.id)
        middle = await env.jobs.get_or_raise(job.id)
        second = await env.deliverer().attempt(job.id)

        one, two = env.receiver.received
        assert first.outcome is DeliveryOutcome.RETRY and first.status_code == 500
        assert first.retry_in == pytest.approx(22.5)  # 30 s base, equal jitter at 0.5
        assert middle.webhook is not None and middle.webhook.status is WebhookStatus.RETRYING
        assert second.outcome is DeliveryOutcome.DELIVERED and second.attempt == 2
        assert one.headers["x-ragengine-delivery"] == two.headers["x-ragengine-delivery"]
        assert (one.headers["x-ragengine-attempt"], two.headers["x-ragengine-attempt"]) == (
            "1", "2",
        )  # fmt: skip
        assert one.body == two.body
        for request in (one, two):  # each attempt is signed and verifies on its own
            verify_webhook(request.body, request.headers["x-ragengine-signature"], env.secret)
        done = await env.jobs.get_or_raise(job.id)
        assert done.webhook is not None
        assert [(e.attempt, e.status_code) for e in done.webhook.log] == [(1, 500), (2, 204)]

    async def test_retry_after_is_honoured(self, env: _Env) -> None:
        job = await env.finished()
        env.receiver.answer(Answer(429, headers={"Retry-After": "120"}))

        result = await env.deliverer().attempt(job.id)

        assert result.outcome is DeliveryOutcome.RETRY and result.retry_in == 120

    async def test_out_of_attempts_it_fails(self, env: _Env) -> None:
        job = await env.finished()
        env.receiver.answer(default=Answer(503))

        outcomes = [(await env.deliverer().attempt(job.id)).outcome for _ in range(3)]

        done = await env.jobs.get_or_raise(job.id)
        assert outcomes == [DeliveryOutcome.RETRY, DeliveryOutcome.RETRY, DeliveryOutcome.FAILED]
        assert done.webhook is not None and done.webhook.status is WebhookStatus.FAILED
        assert "gave up after 3 attempts" in (done.webhook.log[-1].detail or "")
        assert (await env.deliverer().attempt(job.id)).outcome is DeliveryOutcome.SKIPPED
        assert len(env.receiver.received) == 3

    @pytest.mark.parametrize(("status", "error"), [(400, "http_error"), (410, "http_error"),
                                                   (301, "redirect_not_followed")])  # fmt: skip
    async def test_permanent_answers_fail_at_once(self, env: _Env, status: int, error: str) -> None:
        job = await env.finished()
        env.receiver.answer(Answer(status, headers={"Location": env.receiver.url("/elsewhere")}))

        result = await env.deliverer().attempt(job.id)

        assert result.outcome is DeliveryOutcome.FAILED and result.error == error
        assert [r.path for r in env.receiver.received] == ["/hook"]  # redirect not followed

    async def test_a_timeout_retries(self, env: _Env) -> None:
        job = await env.finished()
        env.receiver.answer(Answer(204, delay=2.5))  # WEBHOOK_TIMEOUT_SECONDS = 1

        result = await env.deliverer().attempt(job.id)

        assert result.outcome is DeliveryOutcome.RETRY and result.error == "timeout"

    async def test_a_refused_connection_retries(self, env: _Env) -> None:
        job = await env.finished(url=f"http://127.0.0.1:{_closed_port()}/hook")

        result = await env.deliverer().attempt(job.id)

        assert result.outcome is DeliveryOutcome.RETRY and result.error == "network_error"
        assert result.status_code is None

    async def test_an_ssrf_blocked_receiver_fails_for_good(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        job = await env.finished()
        override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=False)

        result = await env.deliverer().attempt(job.id)

        assert result.outcome is DeliveryOutcome.FAILED and result.error == "ssrf_blocked"
        assert env.receiver.received == []

    async def test_a_tenant_without_a_secret_fails(self, env: _Env) -> None:
        tenants = TenantRepository(env.mongo.db)
        bare = await tenants.create(Tenant(name=new_id("t")))
        env.tenant = bare
        job = await env.finished()

        result = await env.deliverer().attempt(job.id)

        assert result.outcome is DeliveryOutcome.FAILED
        assert result.error == "webhook_secret_missing" and env.receiver.received == []

    async def test_nothing_to_send_is_skipped(self, env: _Env) -> None:
        without = await env.finished(url=None)
        delivered = await env.finished()
        await env.deliverer().attempt(delivered.id)

        results = [await env.deliverer().attempt(job_id)
                   for job_id in (without.id, delivered.id, "job_" + "f" * 32)]  # fmt: skip

        assert {r.outcome for r in results} == {DeliveryOutcome.SKIPPED}
        assert len(env.receiver.received) == 1

    async def test_a_duplicated_attempt_is_recorded_once(self, env: _Env) -> None:
        job = await env.finished()
        attempt = WebhookAttempt(attempt=1, at=utc_now(), status_code=500, error="http_error")

        first = await env.jobs.record_webhook_attempt(job.id, attempt, WebhookStatus.RETRYING)
        again = await env.jobs.record_webhook_attempt(job.id, attempt, WebhookStatus.RETRYING)

        assert first is not None and again is None

    async def test_the_log_keeps_the_latest_attempts(self, env: _Env) -> None:
        job = await env.finished()

        for n in range(1, 26):
            await env.jobs.record_webhook_attempt(
                job.id, WebhookAttempt(attempt=n, at=utc_now(), status_code=503),
                WebhookStatus.RETRYING,
            )  # fmt: skip

        done = await env.jobs.get_or_raise(job.id)
        assert done.webhook is not None and done.webhook.attempts == 25
        assert [e.attempt for e in done.webhook.log] == list(range(6, 26))
        assert len(done.webhook.log) == MAX_WEBHOOK_LOG


class TestLostWebhooks_S83:
    """S83 — the recovery sweep re-sends webhooks whose delivery task was lost."""

    def _recovery(self, env: _Env, *, after: float) -> JobRecovery:
        later = utc_now() + timedelta(seconds=after)
        return JobRecovery(
            env.mongo.db,
            enqueue=lambda job_id, delay: None,
            enqueue_cleanup=lambda *request: None,
            enqueue_webhook=env.sent.append,
            config=env.config,
            clock=lambda: later,
        )

    async def test_pending_and_stale_retrying_webhooks_are_resent(self, env: _Env) -> None:
        pending = await env.finished()
        retrying = await env.finished()
        env.receiver.answer(Answer(503))
        await env.deliverer().attempt(retrying.id)
        delivered = await env.finished()
        await env.deliverer().attempt(delivered.id)
        grace, stale = env.config.WEBHOOK_PENDING_GRACE_SECONDS, env.config.WEBHOOK_STALE_SECONDS

        early = await self._recovery(env, after=grace - 30).run()
        after_grace = await self._recovery(env, after=grace + 1).run()
        repeat = await self._recovery(env, after=grace + 1).run()  # just touched
        after_stale = await self._recovery(env, after=stale + 1).run()

        assert early.webhooks == []
        assert after_grace.webhooks == [pending.id]  # the retrying one waits on its backoff
        assert repeat.webhooks == []
        assert retrying.id in after_stale.webhooks and delivered.id not in after_stale.webhooks
        assert env.sent.count(pending.id) >= 1 and delivered.id not in env.sent

    async def test_recovery_sends_the_webhook_of_a_job_it_ends(self, env: _Env) -> None:
        job = await env.jobs.insert(
            CrawlJob(tenant_id=env.tenant.id, kb_id=_KB, source_id=new_id("src"),
                     seed_url="https://acme.example/",
                     config=CrawlConfig(webhook_url=env.receiver.url()))
        )  # fmt: skip
        for attempt in range(1, env.config.CRAWL_JOB_MAX_ATTEMPTS + 1):
            await env.jobs.start(job.id, f"run-{attempt}")
            if attempt < env.config.CRAWL_JOB_MAX_ATTEMPTS:
                await env.jobs.requeue(job.id, stale_run_id=f"run-{attempt}")

        report = await self._recovery(env, after=env.config.CRAWL_STALE_HEARTBEAT_SECONDS + 1).run()

        assert report.failed == [job.id] and env.sent == [job.id]
        failed = await env.jobs.get_or_raise(job.id)
        assert failed.webhook is not None and failed.webhook.status is WebhookStatus.PENDING


class TestDeliverWebhookTask_S83:
    """S83 — the `deliver_job_webhook` Celery task, in-process, to the local receiver."""

    def test_the_task_delivers(
        self,
        integration_settings: Settings,
        override_settings: OverrideSettings,
        receiver: FakeReceiver,
    ) -> None:
        from rag_engine.worker.context import worker_context
        from rag_engine.worker.tasks.webhooks import deliver_job_webhook

        config = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
        receiver.reset()
        try:
            db = worker_context.runtime.mongo.db

            async def arrange() -> tuple[CrawlJob, str]:
                tenants, jobs = TenantRepository(db), JobRepository(db)
                tenant = await tenants.create(Tenant(name=new_id("t")))
                _, secret = await WebhookSecretService(
                    tenants, SecretBox.from_settings(config)
                ).rotate(tenant.id)
                job = await jobs.insert(
                    CrawlJob(tenant_id=tenant.id, kb_id=_KB, source_id=new_id("src"),
                             seed_url="https://acme.example/",
                             config=CrawlConfig(webhook_url=receiver.url()))
                )  # fmt: skip
                return await jobs.request_cancel(tenant.id, job.id), secret

            job, secret = worker_context.run(arrange())

            result = deliver_job_webhook.apply(args=[job.id]).get(timeout=60)

            assert result["outcome"] == "delivered" and result["status_code"] == 204
            [request] = receiver.received
            assert request.headers["x-ragengine-event"] == "job.cancelled"
            verify_webhook(request.body, request.headers["x-ragengine-signature"], secret)
        finally:
            worker_context.shutdown()
