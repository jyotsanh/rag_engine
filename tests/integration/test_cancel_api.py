"""`DELETE /jobs/{id}` through the real app, MongoDB and Redis: queued jobs end at once,
running ones are flagged for their worker, finished ones are refused, the old version
keeps serving, and a cancelled job's webhook is sent."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from rag_engine.api import create_app
from rag_engine.api.deps import get_cancel_flags, get_task_queue
from rag_engine.config import Settings
from rag_engine.crawling.cancellation import RedisCancelFlags
from rag_engine.domain import CrawlJob, JobError, JobStatus, SourceStats
from rag_engine.infrastructure import (
    AppInfrastructure,
    MongoConnection,
    RedisConnection,
    RedisUnavailableError,
)
from rag_engine.repositories import JobRepository
from rag_engine.repositories.sources import SourceRepository
from tests.conftest import APP_STARTUP_TIMEOUT, OverrideSettings
from tests.support.fakes import RecordingTaskQueue

API = "/api/v1"


@dataclass
class _Tenant:
    id: str
    headers: dict[str, str]


@dataclass
class _World:
    app: Any
    client: httpx.AsyncClient
    admin: dict[str, str]
    mongo: MongoConnection
    flags: RedisCancelFlags
    tasks: RecordingTaskQueue
    acme: _Tenant
    globex: _Tenant

    @property
    def jobs(self) -> JobRepository:
        return JobRepository(self.mongo.db)

    async def job(self, path: str = "/docs/", **body: Any) -> CrawlJob:
        kb = await self.client.post(f"{API}/knowledge-bases", json={"name": path.strip("/")},
                                    headers=self.acme.headers)  # fmt: skip
        created = await self.client.post(
            f"{API}/knowledge-bases/{kb.json()['id']}/crawls",
            json={"url": f"http://127.0.0.1:8080{path}", **body}, headers=self.acme.headers,
        )  # fmt: skip
        assert created.status_code == 202, created.text
        return await self.jobs.get_or_raise(created.json()["id"])

    async def cancel(self, job_id: str, tenant: _Tenant | None = None) -> httpx.Response:
        return await self.client.delete(
            f"{API}/jobs/{job_id}", headers=(tenant or self.acme).headers
        )


async def _bootstrap(client: httpx.AsyncClient, admin: dict[str, str], name: str) -> _Tenant:
    tenant = (await client.post(f"{API}/admin/tenants", json={"name": name}, headers=admin)).json()
    issued = await client.post(f"{API}/admin/tenants/{tenant['id']}/keys", json={}, headers=admin)
    await client.patch(f"{API}/admin/tenants/{tenant['id']}",
                       json={"quotas": {"max_concurrent_jobs": 20}}, headers=admin)  # fmt: skip
    return _Tenant(tenant["id"], {"Authorization": f"Bearer {issued.json()['api_key']}"})


@pytest.fixture
async def world(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[_World]:
    override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    app = create_app(infrastructure_factory=AppInfrastructure)
    tasks = RecordingTaskQueue()
    app.dependency_overrides[get_task_queue] = lambda: tasks
    admin = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    mongo, redis = MongoConnection(integration_settings), RedisConnection(integration_settings)
    await mongo.connect()
    await redis.connect()
    async with (
        LifespanManager(app, startup_timeout=APP_STARTUP_TIMEOUT),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
    ):
        yield _World(app, client, admin, mongo, RedisCancelFlags(redis.client, redis.keys),
                     tasks, await _bootstrap(client, admin, "Acme"),
                     await _bootstrap(client, admin, "Globex"))  # fmt: skip
    await redis.close()
    await mongo.close()


class _DownFlags:
    async def request(self, job_id: str) -> None:
        raise RedisUnavailableError()


# ---- #
# Tests
# ---- #


class TestCancelJob_S87:
    """S87 — `DELETE /jobs/{id}`: queued → 200 cancelled, running → 202 flagged, idempotent,
    finished → 409, tenant-scoped; the old version keeps serving."""

    async def test_a_queued_job_is_cancelled_at_once(self, world: _World) -> None:
        job = await world.job()

        response = await world.cancel(job.id)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "cancelled" and body["id"] == job.id
        assert body["finished_at"] is not None and body["cancel_requested_at"] is not None
        assert await world.flags.is_cancelled(job.id) is False  # nothing to stop: no flag
        assert ("deliver_job_webhook", job.id) not in world.tasks.sent  # no webhook_url

    async def test_a_running_job_is_flagged_for_its_worker(self, world: _World) -> None:
        job = await world.job()
        await world.jobs.start(job.id, "run-1")

        first = await world.cancel(job.id)
        again = await world.cancel(job.id)

        assert first.status_code == again.status_code == 202
        assert first.json()["status"] == "running"
        stamped = first.json()["cancel_requested_at"]
        assert stamped is not None and again.json()["cancel_requested_at"] == stamped  # first kept
        assert await world.flags.is_cancelled(job.id) is True  # the engine polls this
        assert (await world.jobs.heartbeat(job.id, "run-1")).cancel_requested_at is not None

    async def test_once_the_worker_stopped_it_is_200(self, world: _World) -> None:
        job = await world.job()
        await world.jobs.start(job.id, "run-1")
        await world.cancel(job.id)
        await world.jobs.cancel(job.id, "run-1")  # what the worker does after draining

        response = await world.cancel(job.id)

        assert response.status_code == 200 and response.json()["status"] == "cancelled"

    async def test_cancelling_twice_is_safe(self, world: _World) -> None:
        job = await world.job()

        first, second = await world.cancel(job.id), await world.cancel(job.id)

        assert first.status_code == second.status_code == 200
        assert first.json()["finished_at"] == second.json()["finished_at"]

    @pytest.mark.parametrize("status", ["finalizing", "succeeded", "failed"])
    async def test_finalizing_and_finished_jobs_are_409(self, world: _World, status: str) -> None:
        job = await world.job()
        if status == "failed":
            await world.jobs.fail(job.id, JobError(code="boom", message="x"))
        else:
            await world.jobs.start(job.id, "run-1")
            await world.jobs.begin_finalizing(job.id, "run-1")
            if status == "succeeded":
                await world.jobs.succeed(job.id, "run-1")

        response = await world.cancel(job.id)

        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "invalid_job_transition"
        assert error["details"]["from"] == status and error["details"]["to"] == "cancelled"
        assert (await world.jobs.get_or_raise(job.id)).status.value == status

    async def test_the_previous_version_keeps_serving(self, world: _World) -> None:
        job = await world.job()
        sources = SourceRepository(world.mongo.db)
        previous = "job_" + "a" * 32  # an earlier crawl's version, live
        await sources.activate_version(job.source_id, previous, stats=SourceStats(pages=5))

        await world.cancel(job.id)

        source = await sources.get_or_raise(job.source_id)
        assert source.active_version == previous and source.stats.pages == 5

    async def test_a_cancelled_queued_jobs_webhook_is_sent_once(self, world: _World) -> None:
        await world.client.post(f"{API}/admin/tenants/{world.acme.id}/webhook-secret",
                                headers=world.admin)  # fmt: skip
        job = await world.job(webhook_url="http://127.0.0.1:9000/hook")

        await world.cancel(job.id)
        await world.cancel(job.id)  # already cancelled: not sent again

        assert world.tasks.sent.count(("deliver_job_webhook", job.id)) == 1
        cancelled = await world.jobs.get_or_raise(job.id)
        assert cancelled.webhook is not None and cancelled.webhook.status.value == "pending"

    async def test_a_running_jobs_webhook_waits_for_its_worker(self, world: _World) -> None:
        await world.client.post(f"{API}/admin/tenants/{world.acme.id}/webhook-secret",
                                headers=world.admin)  # fmt: skip
        job = await world.job(webhook_url="http://127.0.0.1:9000/hook")
        await world.jobs.start(job.id, "run-1")

        await world.cancel(job.id)

        assert ("deliver_job_webhook", job.id) not in world.tasks.sent  # the runner sends it

    async def test_a_redis_outage_still_cancels(self, world: _World) -> None:
        job = await world.job()
        await world.jobs.start(job.id, "run-1")
        world.app.dependency_overrides[get_cancel_flags] = _DownFlags

        response = await world.cancel(job.id)

        assert response.status_code == 202
        # the durable stamp in MongoDB still reaches the worker through its heartbeat
        assert (await world.jobs.get_or_raise(job.id)).cancel_requested_at is not None

    async def test_tenant_scoping_and_ids(self, world: _World) -> None:
        job = await world.job()

        foreign = await world.cancel(job.id, tenant=world.globex)
        unknown = await world.cancel("job_" + "f" * 32)
        malformed = await world.cancel("kb_" + "1" * 32)
        anonymous = await world.client.delete(f"{API}/jobs/{job.id}")

        assert foreign.status_code == unknown.status_code == 404
        assert foreign.json()["error"]["code"] == "job_not_found"
        assert malformed.status_code == 422 and anonymous.status_code == 401
        assert (await world.jobs.get_or_raise(job.id)).status is JobStatus.QUEUED

    async def test_openapi_documents_the_outcomes(self, world: _World) -> None:
        spec = (await world.client.get(f"{API}/openapi.json")).json()

        operation = spec["paths"]["/api/v1/jobs/{job_id}"]["delete"]
        assert {"200", "202", "404", "409"} <= set(operation["responses"])
