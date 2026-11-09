"""`POST /knowledge-bases/{kb_id}/crawls` through the real app and MongoDB: validation,
SSRF prechecks, tenant quotas, the source upsert, duplicate submissions, enqueueing."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from rag_engine.api import create_app
from rag_engine.api.deps import get_task_queue
from rag_engine.config import Settings
from rag_engine.infrastructure import AppInfrastructure, MongoConnection
from rag_engine.repositories import JobRepository
from rag_engine.repositories.sources import SourceRepository
from tests.conftest import APP_STARTUP_TIMEOUT, OverrideSettings
from tests.support.fakes import RecordingTaskQueue

API = "/api/v1"
SEED = "http://127.0.0.1:8080/docs/"  # vetted without connecting; loopback needs the test flag


@dataclass
class _Tenant:
    id: str
    headers: dict[str, str]


@dataclass
class _World:
    client: httpx.AsyncClient
    admin: dict[str, str]
    mongo: MongoConnection
    tasks: RecordingTaskQueue
    acme: _Tenant
    globex: _Tenant

    @property
    def jobs(self) -> JobRepository:
        return JobRepository(self.mongo.db)

    @property
    def sources(self) -> SourceRepository:
        return SourceRepository(self.mongo.db)

    async def kb(self, tenant: _Tenant | None = None, name: str = "docs") -> str:
        tenant = tenant or self.acme
        response = await self.client.post(
            f"{API}/knowledge-bases", json={"name": name}, headers=tenant.headers
        )
        assert response.status_code == 201, response.text
        return response.json()["id"]

    async def crawl(self, kb_id: str, tenant: _Tenant | None = None, **body: Any) -> httpx.Response:
        tenant = tenant or self.acme
        return await self.client.post(
            f"{API}/knowledge-bases/{kb_id}/crawls",
            json={"url": SEED, **body},
            headers=tenant.headers,
        )

    async def quotas(self, tenant: _Tenant, **quotas: int) -> None:
        response = await self.client.patch(
            f"{API}/admin/tenants/{tenant.id}", json={"quotas": quotas}, headers=self.admin
        )
        assert response.status_code == 200, response.text

    async def webhook_secret(self, tenant: _Tenant) -> None:
        response = await self.client.post(
            f"{API}/admin/tenants/{tenant.id}/webhook-secret", headers=self.admin
        )
        assert response.status_code == 201

    async def count(self, collection: str) -> int:
        return await self.mongo.db[collection].count_documents({})


async def _bootstrap(client: httpx.AsyncClient, admin: dict[str, str], name: str) -> _Tenant:
    tenant = (await client.post(f"{API}/admin/tenants", json={"name": name}, headers=admin)).json()
    issued = await client.post(f"{API}/admin/tenants/{tenant['id']}/keys", json={}, headers=admin)
    return _Tenant(tenant["id"], {"Authorization": f"Bearer {issued.json()['api_key']}"})


@pytest.fixture
async def world(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[_World]:
    override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True, CRAWL_DEFAULT_MAX_PAGES=500)
    app = create_app(infrastructure_factory=AppInfrastructure)
    tasks = RecordingTaskQueue()
    app.dependency_overrides[get_task_queue] = lambda: tasks
    admin = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    async with (
        LifespanManager(app, startup_timeout=APP_STARTUP_TIMEOUT),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
    ):
        yield _World(client, admin, mongo, tasks, await _bootstrap(client, admin, "Acme"),
                     await _bootstrap(client, admin, "Globex"))  # fmt: skip
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestStartCrawl_S84:
    """S84 — start a crawl: 202 + job, source upserted on the normalized seed, enqueued."""

    async def test_a_crawl_is_queued(self, world: _World) -> None:
        kb = await world.kb()

        response = await world.crawl(kb, max_depth=2, path_prefix="/docs/",
                                     exclude_patterns=["/docs/archive/**"],
                                     js_rendering="never")  # fmt: skip

        assert response.status_code == 202, response.text
        body = response.json()
        job = await world.jobs.get_or_raise(body["id"])
        source = await world.sources.get_or_raise(job.source_id)
        assert response.headers["location"] == f"{API}/jobs/{job.id}"
        assert body["status"] == "queued" and body["kb_id"] == kb
        assert body["seed_url"] == SEED == source.seed_url
        assert body["config"]["max_depth"] == 2 and body["config"]["path_prefix"] == "/docs/"
        assert body["config"]["exclude_patterns"] == ["/docs/archive/**"]
        assert body["config"]["js_rendering"] == "never"
        assert body["config"]["max_pages"] == 500  # the service default, within the quota
        assert set(body) >= {"counters", "usage", "error", "attempts", "created_at"}
        assert "tenant_id" not in body and "run_id" not in body and "slot" not in body
        assert source.last_job_id == job.id and source.kb_id == kb
        assert world.tasks.sent == [("run_crawl_job", job.id)]

    async def test_the_seed_is_normalized_into_one_source(self, world: _World) -> None:
        kb = await world.kb()
        first = await world.crawl(kb, url="HTTP://127.0.0.1:8080/docs/#intro")

        again = await world.crawl(kb, url="http://127.0.0.1:8080/./docs/")

        assert first.status_code == 202 and first.json()["seed_url"] == SEED
        assert again.status_code == 200  # the source is already being crawled
        assert again.json()["id"] == first.json()["id"]
        assert world.tasks.sent == [("run_crawl_job", first.json()["id"])]  # enqueued once
        assert await world.count("sources") == 1 and await world.count("crawl_jobs") == 1

    async def test_a_duplicate_keeps_its_own_settings(self, world: _World) -> None:
        kb = await world.kb()
        first = await world.crawl(kb, max_depth=1)

        again = await world.crawl(kb, max_depth=4)

        assert again.status_code == 200 and again.json()["config"]["max_depth"] == 1
        assert again.json()["id"] == first.json()["id"]

    async def test_a_finished_crawl_can_be_resubmitted(self, world: _World) -> None:
        kb = await world.kb()
        first = (await world.crawl(kb)).json()
        await world.jobs.request_cancel(world.acme.id, first["id"])

        second = await world.crawl(kb)

        assert second.status_code == 202 and second.json()["id"] != first["id"]
        assert second.json()["source_id"] == first["source_id"]  # same source, new version

    async def test_kbs_are_tenant_scoped_and_live(self, world: _World) -> None:
        kb = await world.kb()
        deleted = await world.kb(name="gone")
        await world.client.delete(f"{API}/knowledge-bases/{deleted}", headers=world.acme.headers)

        foreign = await world.crawl(kb, tenant=world.globex)
        gone = await world.crawl(deleted)

        assert foreign.status_code == gone.status_code == 404
        assert foreign.json()["error"]["code"] == "knowledge_base_not_found"
        assert await world.count("crawl_jobs") == 0 and await world.count("sources") == 0

    async def test_requires_a_tenant_key(self, world: _World) -> None:
        kb = await world.kb()

        response = await world.client.post(f"{API}/knowledge-bases/{kb}/crawls", json={"url": SEED})

        assert response.status_code == 401


class TestCrawlValidation_S84:
    """S84 — every refusal happens before anything is written."""

    @pytest.mark.parametrize("url", ["ftp://127.0.0.1/", "not a url", "http://", "mailto:a@b.c"])
    async def test_an_invalid_url_is_422(self, world: _World, url: str) -> None:
        kb = await world.kb()

        response = await world.crawl(kb, url=url)

        assert response.status_code == 422
        assert response.json()["error"]["code"] in {"invalid_url", "url_not_allowed"}
        assert await world.count("sources") == 0

    async def test_a_private_target_is_refused_up_front(
        self, world: _World, override_settings: OverrideSettings
    ) -> None:
        kb = await world.kb()
        metadata = await world.crawl(kb, url="http://169.254.169.254/latest/")  # always blocked
        override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=False)

        loopback = await world.crawl(kb)

        for response in (metadata, loopback):
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "url_not_allowed"
        assert "169.254" not in str(loopback.json()["error"]["details"])  # no IP oracle
        assert await world.count("sources") == 0 and world.tasks.sent == []

    async def test_an_unresolvable_host_is_422(self, world: _World) -> None:
        kb = await world.kb()

        response = await world.crawl(kb, url="https://no-such-host.invalid/")

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "host_unresolvable"
        assert response.json()["error"]["details"]["field"] == "url"

    async def test_the_seed_must_be_inside_path_prefix(self, world: _World) -> None:
        kb = await world.kb()

        outside = await world.crawl(kb, path_prefix="/blog/")
        inside = await world.crawl(kb, path_prefix="/docs")

        assert outside.status_code == 422
        assert outside.json()["error"]["code"] == "seed_outside_path_prefix"
        assert inside.status_code == 202

    @pytest.mark.parametrize(
        "body",
        [
            {"max_pages": 0},
            {"max_pages": 10**9},  # above the service's hard cap
            {"max_depth": -1},
            {"path_prefix": "docs"},
            {"include_patterns": ["no-slash"]},
            {"webhook_url": "ftp://hooks.example/"},
            {"js_rendering": "sometimes"},
            {"unknown": 1},
        ],
    )
    async def test_schema_errors_are_422(self, world: _World, body: dict[str, Any]) -> None:
        kb = await world.kb()

        response = await world.crawl(kb, **body)

        assert response.status_code == 422, response.text
        assert await world.count("sources") == 0


class TestCrawlQuotas_S84:
    """S84 — the tenant's page and concurrent-job quotas."""

    async def test_max_pages_defaults_within_the_quota(self, world: _World) -> None:
        await world.quotas(world.acme, max_pages_per_job=50)
        kb = await world.kb()

        response = await world.crawl(kb)

        assert response.status_code == 202 and response.json()["config"]["max_pages"] == 50

    async def test_max_pages_above_the_quota_is_refused_not_lowered(self, world: _World) -> None:
        await world.quotas(world.acme, max_pages_per_job=50)
        kb = await world.kb()

        refused = await world.crawl(kb, max_pages=51)
        allowed = await world.crawl(kb, max_pages=50)

        assert refused.status_code == 403
        assert refused.json()["error"]["details"] == {
            "quota": "max_pages_per_job", "limit": 50, "requested": 51,
        }  # fmt: skip
        assert allowed.status_code == 202

    async def test_concurrent_jobs_quota_creates_nothing(self, world: _World) -> None:
        await world.quotas(world.acme, max_concurrent_jobs=1)
        kb = await world.kb()
        first = await world.crawl(kb)

        refused = await world.crawl(kb, url="http://127.0.0.1:8080/blog/")
        duplicate = await world.crawl(kb)  # the running source is still a 200

        assert first.status_code == 202 and duplicate.status_code == 200
        assert refused.status_code == 403
        assert refused.json()["error"]["details"]["quota"] == "max_concurrent_jobs"
        assert await world.count("sources") == 1  # the refused seed got no source
        assert len(world.tasks.sent) == 1


class TestCrawlWebhook_S84:
    """S84 — `webhook_url`: vetted, and only with a tenant webhook secret."""

    async def test_a_webhook_needs_a_tenant_secret(self, world: _World) -> None:
        kb = await world.kb()
        hook = "http://127.0.0.1:9000/hook"

        missing = await world.crawl(kb, webhook_url=hook)
        await world.webhook_secret(world.acme)
        allowed = await world.crawl(kb, webhook_url=hook)

        assert missing.status_code == 422
        assert missing.json()["error"]["code"] == "webhook_secret_missing"
        assert allowed.status_code == 202 and allowed.json()["config"]["webhook_url"] == hook

    async def test_a_private_webhook_is_refused(self, world: _World) -> None:
        await world.webhook_secret(world.acme)
        kb = await world.kb()

        metadata = await world.crawl(kb, webhook_url="http://169.254.169.254/hook")
        unresolvable = await world.crawl(kb, webhook_url="https://no-such-host.invalid/hook")

        assert metadata.status_code == 422
        assert metadata.json()["error"]["code"] == "webhook_url_not_allowed"
        assert metadata.json()["error"]["details"]["field"] == "webhook_url"
        assert unresolvable.json()["error"]["code"] == "host_unresolvable"
        assert unresolvable.json()["error"]["details"]["field"] == "webhook_url"
        assert await world.count("sources") == 0

    async def test_openapi_documents_both_success_codes(self, world: _World) -> None:
        spec = (await world.client.get(f"{API}/openapi.json")).json()

        operation = spec["paths"]["/api/v1/knowledge-bases/{kb_id}/crawls"]["post"]
        assert {"200", "202", "403", "404", "422"} <= set(operation["responses"])
