"""S88 checkpoint: the whole service over HTTP with a real Celery worker process.

An operator creates a tenant, an API key and a webhook secret; the tenant creates a
knowledge base and submits crawls of the local fixture site; a real prefork worker
(consuming `crawl`, `maintenance` and `webhooks`) crawls, embeds through the real OpenAI
client against `FakeOpenAI`, writes Qdrant, finalizes and delivers signed webhooks. The
test only observes through the API (plus counting points in Qdrant):

- a crawl succeeds, and the job, its pages, the source and Qdrant agree;
- a re-crawl swaps versions **atomically**: sampled all along, search's active version
  always has its points, then the old version is cleaned up;
- a re-crawl cancelled mid-flight leaves the old version serving;
- deleting the source cascades through the worker.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from celery import Celery

from rag_engine.api import create_app
from rag_engine.api.deps import get_task_queue
from rag_engine.config import Settings
from rag_engine.embeddings import resolve_space
from rag_engine.infrastructure import AppInfrastructure, QdrantConnection
from rag_engine.vectorstores import (
    CollectionNotFoundError,
    CollectionSpec,
    QdrantVectorStore,
    VectorFilter,
)
from rag_engine.webhooks import verify_webhook
from rag_engine.worker.celery_app import configure
from rag_engine.worker.producer import CeleryTaskQueue
from tests.conftest import APP_STARTUP_TIMEOUT, OverrideSettings
from tests.support.celery_worker import CeleryWorker, running_worker
from tests.support.fake_openai import FakeOpenAI
from tests.support.fake_receiver import FakeReceiver
from tests.support.fixture_site import FixtureSite

API = "/api/v1"
_DIMS = 256
_SPEC = CollectionSpec.for_space(resolve_space("text-embedding-3-small", _DIMS))
_JOB_TIMEOUT = 120.0


@pytest.fixture(scope="module")
def fake_openai() -> Iterator[FakeOpenAI]:
    server = FakeOpenAI().start()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def receiver() -> Iterator[FakeReceiver]:
    server = FakeReceiver().start()
    yield server
    server.stop()


@pytest.fixture
def producer(integration_settings: Settings) -> Iterator[Celery]:
    app = configure(Celery("api-producer"), integration_settings)
    yield app
    app.close()


@pytest.fixture
def worker(
    integration_settings: Settings,
    producer: Celery,
    fake_openai: FakeOpenAI,
    tmp_path: Path,
) -> Iterator[CeleryWorker]:
    fake_openai.fail_with(None)
    env = {
        "SSRF_ALLOW_PRIVATE_NETWORKS": "true",  # the fixture site and receiver are loopback
        "CRAWL_JS_RENDERING": "never",
        "CRAWL_CONCURRENCY": "2",
        "CRAWL_PER_DOMAIN_RPS": "5",  # slow enough to cancel a crawl mid-flight
        "CRAWL_READ_TIMEOUT_SECONDS": "1",
        "CRAWL_FETCH_TIMEOUT_SECONDS": "2",
        "CRAWL_MAX_RETRIES": "1",
        "CRAWL_RETRY_BASE_DELAY_SECONDS": "0.05",
        "CRAWL_RETRY_MAX_DELAY_SECONDS": "0.1",
        "CRAWL_RETRY_AFTER_MAX_SECONDS": "1",
        "CRAWL_CANCEL_POLL_SECONDS": "0.2",
        "CRAWL_HEARTBEAT_SECONDS": "0.5",
        "CRAWL_VERSION_CLEANUP_DELAY_SECONDS": "1",
        "OPENAI_BASE_URL": fake_openai.base_url,
        "OPENAI_MAX_RETRIES": "0",
        "WEBHOOK_TIMEOUT_SECONDS": "2",
        "CASCADE_RETRY_DELAY_SECONDS": "1",
    }
    with running_worker(
        integration_settings, producer, tmp_path / "worker.log",
        queues="crawl,maintenance,webhooks", concurrency=3, extra_env=env,
    ) as running:  # fmt: skip
        yield running


@dataclass
class _Api:
    client: httpx.AsyncClient
    store: QdrantVectorStore
    worker: CeleryWorker
    tenant_id: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    webhook_secret: str = ""

    async def call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        return await self.client.request(method, f"{API}{path}", headers=self.headers, **kwargs)

    async def ok(self, method: str, path: str, status: int = 200, **kwargs: Any) -> Any:
        response = await self.call(method, path, **kwargs)
        assert response.status_code == status, (response.status_code, response.text)
        return response.json()

    async def job(self, job_id: str) -> dict[str, Any]:
        return await self.ok("GET", f"/jobs/{job_id}")

    async def finished(self, job_id: str) -> dict[str, Any]:
        """Poll `GET /jobs/{id}` until the job ends."""
        job = await _until(lambda: self.job(job_id), lambda j: j["status"] in (
            "succeeded", "failed", "cancelled"), what=f"job {job_id} to finish",
            worker=self.worker)  # fmt: skip
        return job

    async def source(self, kb_id: str, source_id: str) -> dict[str, Any]:
        return await self.ok("GET", f"/knowledge-bases/{kb_id}/sources/{source_id}")

    async def points(self, version: str) -> int:
        try:
            return await self.store.count(_SPEC, VectorFilter(self.tenant_id, versions=(version,)))
        except CollectionNotFoundError:
            return 0

    async def pages(self, job_id: str, **params: Any) -> list[dict[str, Any]]:
        items, cursor = [], None
        while True:
            query = {**params, "limit": 200, **({"cursor": cursor} if cursor else {})}
            page = await self.ok("GET", f"/jobs/{job_id}/pages", params=query)
            items += page["items"]
            if not (cursor := page["next_cursor"]):
                return items


async def _until[T](
    probe: Callable[[], Awaitable[T]],
    done: Callable[[T], bool],
    *,
    what: str,
    worker: CeleryWorker,
    interval: float = 0.2,
) -> T:
    deadline = time.monotonic() + _JOB_TIMEOUT
    while True:
        value = await probe()
        if done(value):
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}: {value}\n{worker.output()}")
        await asyncio.sleep(interval)


@pytest.fixture
async def api(
    integration_settings: Settings,
    override_settings: OverrideSettings,
    producer: Celery,
    worker: CeleryWorker,
) -> AsyncIterator[_Api]:
    override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)  # the API's submission precheck
    app = create_app(infrastructure_factory=AppInfrastructure)
    app.dependency_overrides[get_task_queue] = lambda: CeleryTaskQueue(producer)
    admin = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    qdrant = QdrantConnection(integration_settings)
    await qdrant.connect()
    async with (
        LifespanManager(app, startup_timeout=APP_STARTUP_TIMEOUT),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                          timeout=30) as client,
    ):  # fmt: skip
        api = _Api(client, QdrantVectorStore(qdrant.client), worker, headers=admin)
        # the operator: tenant, API key, webhook secret
        tenant = await api.ok("POST", "/admin/tenants", 201, json={"name": "Acme"})
        key = await api.ok("POST", f"/admin/tenants/{tenant['id']}/keys", 201, json={})
        secret = await api.ok("POST", f"/admin/tenants/{tenant['id']}/webhook-secret", 201)
        api.tenant_id, api.webhook_secret = tenant["id"], secret["secret"]
        api.headers = {"Authorization": f"Bearer {key['api_key']}"}
        yield api
    await qdrant.close()


async def _kb(api: _Api) -> str:
    kb = await api.ok("POST", "/knowledge-bases", 201, json={
        "name": "docs", "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": _DIMS,
    })  # fmt: skip
    return kb["id"]


def _docs_crawl(site: FixtureSite, receiver: FakeReceiver, **body: Any) -> dict[str, Any]:
    return {"url": site.url("/docs/"), "path_prefix": "/docs/", "max_depth": 3,
            "max_pages": 10, "webhook_url": receiver.url(), **body}  # fmt: skip


# ---- #
# Tests
# ---- #


class TestApiE2E_S88:
    """S88 — API E2E with a real Celery worker: crawl, consistency, atomic re-crawl swap,
    cancel mid-crawl, cascade delete."""

    async def test_crawl_then_recrawl_swaps_atomically(
        self, api: _Api, fixture_site: FixtureSite, receiver: FakeReceiver
    ) -> None:
        receiver.reset()
        kb = await _kb(api)

        # ---- first crawl ---- #
        submitted = await api.ok("POST", f"/knowledge-bases/{kb}/crawls", 202,
                                 json=_docs_crawl(fixture_site, receiver))  # fmt: skip
        first = await api.finished(submitted["id"])

        assert first["status"] == "succeeded", first
        counters = first["counters"]
        assert counters["fetched"] > 0 and counters["chunked"] == counters["embedded"] > 0
        assert first["usage"]["requests"] > 0 and first["usage"]["tokens"] > 0
        assert first["report"]["pages"]["succeeded"] == counters["fetched"]
        source = await api.source(kb, first["source_id"])
        assert source["active_version"] == first["id"] and source["last_job_id"] == first["id"]
        assert source["stats"]["pages"] == first["report"]["pages"]["succeeded"]
        assert source["stats"]["chunks"] == counters["chunked"]
        # the per-page report agrees with the job, and so does Qdrant
        pages = await api.pages(first["id"])
        succeeded = [p for p in pages if p["status"] == "succeeded"]
        assert len(succeeded) == source["stats"]["pages"] and len(pages) >= len(succeeded)
        assert all(p["url"].startswith(fixture_site.url("/docs/")) for p in succeeded)
        chunk_total = 0
        for page in succeeded:
            detail = await api.ok("GET", f"/jobs/{first['id']}/pages/{page['id']}")
            assert detail["markdown"] and detail["markdown_chars"] == len(detail["markdown"])
            chunk_total += detail["chunks"]
        assert chunk_total == counters["chunked"] == await api.points(first["id"])
        # the completion webhook arrived signed, and the job shows it delivered
        delivered = await _until(lambda: api.job(first["id"]),
                                 lambda j: (j["webhook"] or {}).get("status") == "delivered",
                                 what="the webhook", worker=api.worker)  # fmt: skip
        assert delivered["webhook"]["attempts"] == 1
        [hook] = [r for r in receiver.received
                  if r.headers["x-ragengine-delivery"].endswith(first["id"][4:])]  # fmt: skip
        verify_webhook(hook.body, hook.headers["x-ragengine-signature"], api.webhook_secret)
        assert hook.headers["x-ragengine-event"] == "job.succeeded"

        # ---- re-crawl: the swap is atomic ---- #
        resubmitted = await api.ok("POST", f"/knowledge-bases/{kb}/crawls", 202,
                                   json=_docs_crawl(fixture_site, receiver))  # fmt: skip
        second_id = resubmitted["id"]
        assert resubmitted["source_id"] == first["source_id"] and second_id != first["id"]
        samples: list[tuple[str, int]] = []
        deadline = time.monotonic() + _JOB_TIMEOUT
        while True:  # sample what search would serve, all through the re-crawl
            active = (await api.source(kb, first["source_id"]))["active_version"]
            samples.append((active, await api.points(active)))
            job = await api.job(second_id)
            if job["status"] in ("succeeded", "failed", "cancelled"):
                break
            assert time.monotonic() < deadline, f"re-crawl timed out\n{api.worker.output()}"
            await asyncio.sleep(0.1)
        second = await api.job(second_id)

        assert second["status"] == "succeeded", second
        assert all(points > 0 for _, points in samples), samples  # never an empty source
        versions = [version for version, _ in samples]
        assert len(samples) >= 3 and versions[0] == first["id"]  # sampled from before the flip
        assert set(versions) <= {first["id"], second_id}
        assert versions == sorted(versions, key=lambda v: v == second_id)  # v1 … then v2, once
        source = await api.source(kb, first["source_id"])
        assert source["active_version"] == second_id
        assert await api.points(second_id) == second["counters"]["chunked"] > 0
        # the previous version is deleted shortly after the swap (cleanup task, 1 s delay)
        await _until(lambda: api.points(first["id"]), lambda n: n == 0,
                     what="the old version's cleanup", worker=api.worker)  # fmt: skip
        old_pages = await api.pages(first["id"])
        assert len(old_pages) == len(pages)  # the old job's report stays readable
        listed = await api.ok("GET", "/jobs", params={"source_id": first["source_id"]})
        assert [j["id"] for j in listed["items"]] == [second_id, first["id"]]

    async def test_a_cancelled_recrawl_keeps_the_old_version(
        self, api: _Api, fixture_site: FixtureSite, receiver: FakeReceiver
    ) -> None:
        receiver.reset()
        kb = await _kb(api)
        whole_site = {"url": fixture_site.url("/"), "max_depth": 3, "max_pages": 40,
                      "webhook_url": receiver.url()}  # fmt: skip
        live = await api.finished(
            (await api.ok("POST", f"/knowledge-bases/{kb}/crawls", 202, json=whole_site))["id"]
        )
        assert live["status"] == "succeeded", live
        live_points = await api.points(live["id"])
        assert live_points > 0

        recrawl = await api.ok("POST", f"/knowledge-bases/{kb}/crawls", 202, json=whole_site)
        await _until(lambda: api.job(recrawl["id"]),
                     lambda j: j["status"] == "running" and j["counters"]["fetched"] >= 1,
                     what="the re-crawl to start fetching", worker=api.worker)  # fmt: skip
        duplicate = await api.call("POST", f"/knowledge-bases/{kb}/crawls", json=whole_site)
        cancel = await api.call("DELETE", f"/jobs/{recrawl['id']}")
        cancelled = await api.finished(recrawl["id"])

        assert duplicate.status_code == 200 and duplicate.json()["id"] == recrawl["id"]
        assert cancel.status_code == 202 and cancel.json()["cancel_requested_at"]
        assert cancelled["status"] == "cancelled", cancelled
        assert cancelled["counters"]["fetched"] < 40  # it really stopped early
        source = await api.source(kb, live["source_id"])
        assert source["active_version"] == live["id"]  # the old version keeps serving
        assert await api.points(live["id"]) == live_points
        assert await api.points(recrawl["id"]) == 0  # the cancelled crawl's data is gone
        pages = await api.pages(recrawl["id"], status="skipped", reason="job_cancelled")
        assert pages  # the frontier left behind is recorded as cancelled
        hooked = await _until(lambda: api.job(recrawl["id"]),
                              lambda j: (j["webhook"] or {}).get("status") == "delivered",
                              what="the cancel webhook", worker=api.worker)  # fmt: skip
        assert hooked["webhook"]["attempts"] == 1
        events = {r.headers["x-ragengine-event"] for r in receiver.received}
        assert {"job.succeeded", "job.cancelled"} <= events

    async def test_deleting_the_source_cascades_through_the_worker(
        self, api: _Api, fixture_site: FixtureSite, receiver: FakeReceiver
    ) -> None:
        kb = await _kb(api)
        job = await api.finished((await api.ok(
            "POST", f"/knowledge-bases/{kb}/crawls", 202,
            json=_docs_crawl(fixture_site, receiver, webhook_url=None),
        ))["id"])  # fmt: skip
        assert job["status"] == "succeeded" and await api.points(job["id"]) > 0

        deleted = await api.call("DELETE", f"/knowledge-bases/{kb}/sources/{job['source_id']}")
        await _until(lambda: api.points(job["id"]), lambda n: n == 0,
                     what="the cascade to delete the vectors", worker=api.worker)  # fmt: skip
        gone = await _until(lambda: api.call("GET", f"/jobs/{job['id']}"),
                            lambda r: r.status_code == 404, what="the job records to go",
                            worker=api.worker)  # fmt: skip

        assert deleted.status_code == 202 and deleted.json()["status"] == "deleting"
        assert gone.json()["error"]["code"] == "job_not_found"
        source = await api.call("GET", f"/knowledge-bases/{kb}/sources/{job['source_id']}")
        assert source.status_code == 404
        listed = await api.ok("GET", f"/knowledge-bases/{kb}/sources")
        assert listed["items"] == []
