"""`GET /jobs/{id}` and `GET /jobs` through the real app and MongoDB: live counters, usage,
the crawl report, the webhook's delivery, filters, pagination and tenant isolation."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from rag_engine.api import create_app
from rag_engine.api.deps import get_task_queue
from rag_engine.config import Settings
from rag_engine.domain import (
    CrawlJob,
    CrawlReport,
    JobError,
    PageReason,
    PageSample,
    PageStatus,
    ReasonCount,
    WebhookAttempt,
    WebhookStatus,
    utc_now,
)
from rag_engine.infrastructure import AppInfrastructure, MongoConnection
from rag_engine.repositories import JobRepository
from tests.conftest import APP_STARTUP_TIMEOUT, OverrideSettings
from tests.support.fakes import RecordingTaskQueue

API = "/api/v1"


@dataclass
class _Tenant:
    id: str
    headers: dict[str, str]


@dataclass
class _World:
    client: httpx.AsyncClient
    admin: dict[str, str]
    mongo: MongoConnection
    acme: _Tenant
    globex: _Tenant

    @property
    def jobs(self) -> JobRepository:
        return JobRepository(self.mongo.db)

    async def kb(self, tenant: _Tenant, name: str = "docs") -> str:
        response = await self.client.post(
            f"{API}/knowledge-bases", json={"name": name}, headers=tenant.headers
        )
        assert response.status_code == 201, response.text
        return response.json()["id"]

    async def crawl(
        self, tenant: _Tenant, kb_id: str, path: str = "/docs/", **body: Any
    ) -> CrawlJob:
        response = await self.client.post(
            f"{API}/knowledge-bases/{kb_id}/crawls",
            json={"url": f"http://127.0.0.1:8080{path}", **body},
            headers=tenant.headers,
        )
        assert response.status_code == 202, response.text
        return await self.jobs.get_or_raise(response.json()["id"])

    async def get(self, path: str, tenant: _Tenant | None = None, **params: Any) -> httpx.Response:
        tenant = tenant or self.acme
        return await self.client.get(f"{API}{path}", headers=tenant.headers, params=params)


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
    app.dependency_overrides[get_task_queue] = RecordingTaskQueue
    admin = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    async with (
        LifespanManager(app, startup_timeout=APP_STARTUP_TIMEOUT),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
    ):
        yield _World(client, admin, mongo, await _bootstrap(client, admin, "Acme"),
                     await _bootstrap(client, admin, "Globex"))  # fmt: skip
    await mongo.close()


def _report() -> CrawlReport:
    return CrawlReport(
        pages={PageStatus.SUCCEEDED: 8, PageStatus.FAILED: 2},
        failures=[
            ReasonCount(status=PageStatus.FAILED, reason=PageReason.TIMEOUT, count=2,
                        samples=[PageSample(url="http://127.0.0.1:8080/docs/slow")])
        ],
    )  # fmt: skip


# ---- #
# Tests
# ---- #


class TestGetJob_S85:
    """S85 — `GET /jobs/{id}`: status, live counters, usage, report, webhook delivery."""

    async def test_a_queued_job(self, world: _World) -> None:
        job = await world.crawl(world.acme, await world.kb(world.acme))

        response = await world.get(f"/jobs/{job.id}")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == job.id and body["status"] == "queued"
        assert body["kb_id"] == job.kb_id and body["source_id"] == job.source_id
        assert (body["duration_seconds"], body["report"], body["webhook"]) == (None, None, None)
        assert "tenant_id" not in body and "run_id" not in body and "slot" not in body

    async def test_a_running_job_shows_live_counters_and_usage(self, world: _World) -> None:
        job = await world.crawl(world.acme, await world.kb(world.acme))
        await world.jobs.start(job.id, "run-1")
        await world.jobs.increment(job.id, "run-1", discovered=12, fetched=5, failed=1)
        await world.jobs.add_usage(job.id, "run-1", requests=2, tokens=900, cached=3)

        body = (await world.get(f"/jobs/{job.id}")).json()

        assert body["status"] == "running" and body["attempts"] == 1
        assert body["counters"]["discovered"] == 12 and body["counters"]["fetched"] == 5
        assert body["usage"] == {"requests": 2, "tokens": 900, "cached": 3}
        assert body["duration_seconds"] is not None and body["duration_seconds"] >= 0
        assert body["finished_at"] is None

    async def test_a_failed_job_carries_its_error_and_report(self, world: _World) -> None:
        job = await world.crawl(world.acme, await world.kb(world.acme))
        await world.jobs.start(job.id, "run-1")
        await world.jobs.set_report(job.id, "run-1", _report())
        await world.jobs.fail(job.id, JobError(code="embedding_auth_failed", message="401"),
                              run_id="run-1")  # fmt: skip

        body = (await world.get(f"/jobs/{job.id}")).json()

        assert body["status"] == "failed"
        assert body["error"] == {"code": "embedding_auth_failed", "message": "401"}
        assert body["report"]["pages"] == {"succeeded": 8, "failed": 2}
        [failure] = body["report"]["failures"]
        assert (failure["reason"], failure["count"]) == ("timeout", 2)
        assert failure["samples"][0]["url"].endswith("/docs/slow")
        finished = body["finished_at"] is not None
        assert finished and body["duration_seconds"] >= 0

    async def test_the_webhook_delivery_is_shown(
        self, world: _World, override_settings: OverrideSettings
    ) -> None:
        await world.client.post(f"{API}/admin/tenants/{world.acme.id}/webhook-secret",
                                headers=world.admin)  # fmt: skip
        job = await world.crawl(world.acme, await world.kb(world.acme),
                                webhook_url="http://127.0.0.1:9000/hook")  # fmt: skip
        await world.jobs.request_cancel(world.acme.id, job.id)
        attempt = WebhookAttempt(attempt=1, at=utc_now(), status_code=503, error="http_error",
                                 detail="HTTP 503", duration_ms=12)  # fmt: skip
        await world.jobs.record_webhook_attempt(job.id, attempt, WebhookStatus.RETRYING)

        body = (await world.get(f"/jobs/{job.id}")).json()

        assert body["status"] == "cancelled"
        webhook = body["webhook"]
        assert (webhook["status"], webhook["attempts"], webhook["delivered_at"]) == (
            "retrying", 1, None,
        )  # fmt: skip
        [entry] = webhook["log"]
        assert (entry["status_code"], entry["error"]) == (503, "http_error")
        assert body["config"]["webhook_url"] == "http://127.0.0.1:9000/hook"

    async def test_unknown_foreign_and_malformed_ids(self, world: _World) -> None:
        job = await world.crawl(world.acme, await world.kb(world.acme))

        foreign = await world.get(f"/jobs/{job.id}", tenant=world.globex)
        unknown = await world.get(f"/jobs/job_{'f' * 32}")
        malformed = await world.get("/jobs/kb_" + "1" * 32)

        assert foreign.status_code == unknown.status_code == 404
        assert foreign.json()["error"]["code"] == unknown.json()["error"]["code"] == "job_not_found"
        assert malformed.status_code == 422

    async def test_the_crawl_location_points_here(self, world: _World) -> None:
        kb = await world.kb(world.acme)
        created = await world.client.post(
            f"{API}/knowledge-bases/{kb}/crawls",
            json={"url": "http://127.0.0.1:8080/docs/"}, headers=world.acme.headers,
        )  # fmt: skip

        fetched = await world.client.get(created.headers["location"], headers=world.acme.headers)

        assert fetched.status_code == 200 and fetched.json()["id"] == created.json()["id"]

    async def test_requires_a_tenant_key(self, world: _World) -> None:
        assert (await world.client.get(f"{API}/jobs")).status_code == 401
        assert (await world.client.get(f"{API}/jobs/job_{'f' * 32}")).status_code == 401


class TestListJobs_S85:
    """S85 — `GET /jobs`: newest first, KB / source / status filters, cursor pagination."""

    async def test_newest_first_with_filters(self, world: _World) -> None:
        docs, blog = await world.kb(world.acme, "docs"), await world.kb(world.acme, "blog")
        first = await world.crawl(world.acme, docs, "/docs/")
        second = await world.crawl(world.acme, docs, "/guides/")
        third = await world.crawl(world.acme, blog, "/blog/")
        await world.jobs.request_cancel(world.acme.id, first.id)
        await world.jobs.start(third.id, "run-1")
        await world.crawl(world.globex, await world.kb(world.globex), "/docs/")  # not ours

        everything = (await world.get("/jobs")).json()
        by_kb = (await world.get("/jobs", kb_id=docs)).json()
        by_source = (await world.get("/jobs", source_id=second.source_id)).json()
        running = (await world.get("/jobs", status="running")).json()
        either = (await world.get("/jobs", status=["running", "cancelled"])).json()

        def ids(page: dict[str, Any]) -> list[str]:
            return [item["id"] for item in page["items"]]

        assert ids(everything) == [third.id, second.id, first.id]
        assert ids(by_kb) == [second.id, first.id]
        assert ids(by_source) == [second.id]
        assert ids(running) == [third.id]
        assert ids(either) == [third.id, first.id]
        assert "report" not in everything["items"][0]  # the list is the summary view

    async def test_cursor_pagination(self, world: _World) -> None:
        kb = await world.kb(world.acme)
        created = [await world.crawl(world.acme, kb, f"/section{n}/") for n in range(5)]

        first = (await world.get("/jobs", limit=2)).json()
        second = (await world.get("/jobs", limit=2, cursor=first["next_cursor"])).json()
        third = (await world.get("/jobs", limit=2, cursor=second["next_cursor"])).json()

        seen = [item["id"] for page in (first, second, third) for item in page["items"]]
        assert seen == [job.id for job in reversed(created)]
        assert third["next_cursor"] is None

    @pytest.mark.parametrize(
        "params",
        [{"kb_id": "nope"}, {"source_id": "kb_" + "1" * 32}, {"status": "lost"},
         {"limit": 0}, {"limit": 10_000}, {"cursor": "garbage"}],
    )  # fmt: skip
    async def test_invalid_filters_are_422(self, world: _World, params: dict[str, Any]) -> None:
        response = await world.get("/jobs", **params)

        assert response.status_code == 422, response.text

    async def test_a_tenant_sees_only_its_jobs(self, world: _World) -> None:
        mine = await world.crawl(world.acme, await world.kb(world.acme))

        theirs = (await world.get("/jobs", tenant=world.globex)).json()
        filtered = (await world.get("/jobs", tenant=world.globex, kb_id=mine.kb_id)).json()

        assert theirs["items"] == [] and filtered["items"] == []

    async def test_the_kb_jobs_index_exists(self, world: _World) -> None:
        info = await world.mongo.db[CrawlJob.collection_name].index_information()

        assert info["tenant_kb_jobs"]["key"] == [("tenant_id", 1), ("kb_id", 1), ("_id", -1)]
