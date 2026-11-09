"""`GET /jobs/{id}/pages` and `GET /jobs/{id}/pages/{page_id}` through the real app and
MongoDB: the per-page report, its filters and pagination, the light list view, one page in
full, and tenant / job isolation."""

import hashlib
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
    Chunk,
    ContentKind,
    CrawlJob,
    DiscoveredVia,
    Page,
    PageReason,
    PageStatus,
    chunk_id,
)
from rag_engine.infrastructure import AppInfrastructure, MongoConnection
from rag_engine.repositories import ChunkRepository, JobRepository, PageRepository
from tests.conftest import APP_STARTUP_TIMEOUT, OverrideSettings
from tests.support.fakes import RecordingTaskQueue

API = "/api/v1"
_SITE = "http://127.0.0.1:8080"


@dataclass
class _Tenant:
    id: str
    headers: dict[str, str]


@dataclass
class _World:
    client: httpx.AsyncClient
    mongo: MongoConnection
    acme: _Tenant
    globex: _Tenant

    async def job(self, tenant: _Tenant, path: str = "/docs/") -> CrawlJob:
        kb = await self.client.post(
            f"{API}/knowledge-bases", json={"name": path.strip("/") or "root"},
            headers=tenant.headers,
        )  # fmt: skip
        created = await self.client.post(
            f"{API}/knowledge-bases/{kb.json()['id']}/crawls",
            json={"url": f"{_SITE}{path}"}, headers=tenant.headers,
        )  # fmt: skip
        assert created.status_code == 202, created.text
        return await JobRepository(self.mongo.db).get_or_raise(created.json()["id"])

    async def page(self, job: CrawlJob, n: int, status: PageStatus, **fields: Any) -> Page:
        values: dict[str, Any] = {
            "tenant_id": job.tenant_id, "kb_id": job.kb_id, "source_id": job.source_id,
            "job_id": job.id, "url": f"{_SITE}/docs/p{n}", "url_hash": f"{n:064x}",
            "status": status, "depth": n % 3, "discovered_via": DiscoveredVia.LINK,
        }  # fmt: skip
        if status is PageStatus.SUCCEEDED:
            values |= {"content_kind": ContentKind.HTML, "markdown": f"# Page {n}\n\nBody {n}.",
                       "http_status": 200, "title": f"Page {n}"}  # fmt: skip
        return await PageRepository(self.mongo.db).insert(Page(**{**values, **fields}))

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
        yield _World(client, mongo, await _bootstrap(client, admin, "Acme"),
                     await _bootstrap(client, admin, "Globex"))  # fmt: skip
    await mongo.close()


async def _report(world: _World, job: CrawlJob) -> list[Page]:
    """A small crawl's outcome: 2 succeeded, 1 robots-skipped, 1 out of scope, 2 failed."""
    return [
        await world.page(job, 0, PageStatus.SUCCEEDED),
        await world.page(job, 1, PageStatus.SKIPPED, reason=PageReason.ROBOTS_DISALLOWED,
                         reason_detail="Disallow: /docs/p1"),
        await world.page(job, 2, PageStatus.FAILED, reason=PageReason.TIMEOUT,
                         reason_detail="read timed out after 10 s"),
        await world.page(job, 3, PageStatus.SUCCEEDED),
        await world.page(job, 4, PageStatus.SKIPPED, reason=PageReason.OUT_OF_SCOPE),
        await world.page(job, 5, PageStatus.FAILED, reason=PageReason.HTTP_CLIENT_ERROR,
                         http_status=404),
    ]  # fmt: skip


def _ids(page: dict[str, Any]) -> list[str]:
    return [item["id"] for item in page["items"]]


# ---- #
# Tests
# ---- #


class TestJobPages_S86:
    """S86 — the per-page report: every URL with its outcome, filters, pagination."""

    async def test_every_page_in_discovery_order_without_markdown(self, world: _World) -> None:
        job = await world.job(world.acme)
        pages = await _report(world, job)

        body = (await world.get(f"/jobs/{job.id}/pages")).json()

        assert _ids(body) == [p.id for p in pages] and body["next_cursor"] is None
        first, skipped = body["items"][0], body["items"][1]
        assert first["status"] == "succeeded" and first["http_status"] == 200
        assert first["markdown_chars"] == len("# Page 0\n\nBody 0.")
        assert "markdown" not in first and "deferred_links" not in first
        assert "tenant_id" not in first and "url_hash" not in first
        assert (skipped["status"], skipped["reason"]) == ("skipped", "robots_disallowed")
        assert skipped["reason_detail"] == "Disallow: /docs/p1"
        assert skipped["markdown_chars"] == 0

    async def test_status_and_reason_filters(self, world: _World) -> None:
        job = await world.job(world.acme)
        _, p1, p2, _, p4, p5 = await _report(world, job)
        url = f"/jobs/{job.id}/pages"

        failed = (await world.get(url, status="failed")).json()
        not_ok = (await world.get(url, status=["failed", "skipped"])).json()
        timeouts = (await world.get(url, reason="timeout")).json()
        either = (await world.get(url, reason=["timeout", "out_of_scope"])).json()
        both = (await world.get(url, status="skipped", reason="timeout")).json()

        assert _ids(failed) == [p2.id, p5.id]
        assert _ids(not_ok) == [p1.id, p2.id, p4.id, p5.id]
        assert _ids(timeouts) == [p2.id]
        assert _ids(either) == [p2.id, p4.id]
        assert _ids(both) == []

    async def test_cursor_pagination(self, world: _World) -> None:
        job = await world.job(world.acme)
        pages = [await world.page(job, n, PageStatus.SUCCEEDED) for n in range(7)]
        url = f"/jobs/{job.id}/pages"

        first = (await world.get(url, limit=3)).json()
        second = (await world.get(url, limit=3, cursor=first["next_cursor"])).json()
        third = (await world.get(url, limit=3, cursor=second["next_cursor"])).json()

        assert _ids(first) + _ids(second) + _ids(third) == [p.id for p in pages]
        assert third["next_cursor"] is None
        filtered = (await world.get(url, limit=2, status="succeeded",
                                    cursor=first["next_cursor"])).json()  # fmt: skip
        assert _ids(filtered) == [pages[3].id, pages[4].id]

    async def test_the_list_never_carries_the_markdown(self, world: _World) -> None:
        job = await world.job(world.acme)
        huge = "x" * 1_000_000
        await world.page(job, 0, PageStatus.SUCCEEDED, markdown=huge)

        response = await world.get(f"/jobs/{job.id}/pages")

        assert response.json()["items"][0]["markdown_chars"] == 1_000_000
        assert len(response.content) < 10_000  # projected out in MongoDB, not after

    async def test_only_this_jobs_pages(self, world: _World) -> None:
        job, other = await world.job(world.acme, "/docs/"), await world.job(world.acme, "/blog/")
        mine = await world.page(job, 0, PageStatus.SUCCEEDED)
        await world.page(other, 1, PageStatus.SUCCEEDED)

        body = (await world.get(f"/jobs/{job.id}/pages")).json()

        assert _ids(body) == [mine.id]

    async def test_another_tenants_job_is_404(self, world: _World) -> None:
        job = await world.job(world.acme)
        await _report(world, job)

        foreign = await world.get(f"/jobs/{job.id}/pages", tenant=world.globex)
        unknown = await world.get(f"/jobs/job_{'f' * 32}/pages")

        assert foreign.status_code == unknown.status_code == 404
        assert foreign.json()["error"]["code"] == "job_not_found"

    @pytest.mark.parametrize("params", [{"status": "lost"}, {"reason": "bored"},
                                        {"limit": 0}, {"cursor": "garbage"}])  # fmt: skip
    async def test_invalid_parameters_are_422(self, world: _World, params: dict[str, Any]) -> None:
        job = await world.job(world.acme)

        response = await world.get(f"/jobs/{job.id}/pages", **params)

        assert response.status_code == 422

    async def test_requires_a_tenant_key(self, world: _World) -> None:
        response = await world.client.get(f"{API}/jobs/job_{'f' * 32}/pages")

        assert response.status_code == 401

    async def test_the_reason_index_exists(self, world: _World) -> None:
        info = await world.mongo.db[Page.collection_name].index_information()

        assert info["job_reason"]["key"] == [("job_id", 1), ("reason", 1), ("_id", 1)]


class TestJobPage_S86:
    """S86 — one page in full: Markdown, metadata, chunk count; scoped to its job."""

    async def test_a_page_in_full(self, world: _World) -> None:
        job = await world.job(world.acme)
        page = await world.page(job, 0, PageStatus.SUCCEEDED, description="About widgets",
                                doc_metadata={"author": "Acme"}, extractor="html")  # fmt: skip
        chunks = [
            Chunk(id=chunk_id(page.id, n), tenant_id=job.tenant_id, kb_id=job.kb_id,
                  source_id=job.source_id, version=job.version, page_id=page.id, url=page.url,
                  ordinal=n, text=f"part {n}", embed_text=f"part {n}", token_count=2,
                  content_hash=hashlib.sha256(f"part {n}".encode()).hexdigest())
            for n in range(3)
        ]  # fmt: skip
        await ChunkRepository(world.mongo.db).save_page_chunks(page.id, chunks)

        response = await world.get(f"/jobs/{job.id}/pages/{page.id}")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == page.id and body["markdown"] == "# Page 0\n\nBody 0."
        assert body["markdown_chars"] == len(body["markdown"]) and body["chunks"] == 3
        assert body["description"] == "About widgets"
        assert body["doc_metadata"] == {"author": "Acme"} and body["extractor"] == "html"
        assert "tenant_id" not in body and "deferred_links" not in body

    async def test_a_failed_page_has_no_content(self, world: _World) -> None:
        job = await world.job(world.acme)
        page = await world.page(job, 0, PageStatus.FAILED, reason=PageReason.TIMEOUT)

        body = (await world.get(f"/jobs/{job.id}/pages/{page.id}")).json()

        assert (body["status"], body["reason"], body["markdown"], body["chunks"]) == (
            "failed", "timeout", None, 0,
        )  # fmt: skip

    async def test_pages_are_scoped_to_their_job_and_tenant(self, world: _World) -> None:
        job, other = await world.job(world.acme, "/docs/"), await world.job(world.acme, "/blog/")
        page = await world.page(job, 0, PageStatus.SUCCEEDED)

        via_other_job = await world.get(f"/jobs/{other.id}/pages/{page.id}")
        foreign = await world.get(f"/jobs/{job.id}/pages/{page.id}", tenant=world.globex)
        malformed = await world.get(f"/jobs/{job.id}/pages/chk_{'1' * 32}")

        assert via_other_job.status_code == 404
        assert via_other_job.json()["error"]["code"] == "page_not_found"
        assert foreign.status_code == 404 and foreign.json()["error"]["code"] == "job_not_found"
        assert malformed.status_code == 422

    async def test_openapi_exposes_id_not_mongos_underscore_id(self, world: _World) -> None:
        spec = (await world.client.get(f"{API}/openapi.json")).json()

        for name in ("PageResponse", "PageDetailResponse"):
            properties = spec["components"]["schemas"][name]["properties"]
            assert "id" in properties and "_id" not in properties
