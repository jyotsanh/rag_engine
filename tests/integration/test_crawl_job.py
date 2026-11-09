"""`run_crawl_job`: claim → crawl the fixture site → ingest → finalize, on real services."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

import pytest

from rag_engine.config import JsRenderMode, Settings
from rag_engine.domain import (
    CrawlConfig,
    CrawlJob,
    JobStatus,
    KnowledgeBase,
    PageStatus,
    Source,
)
from rag_engine.embeddings import (
    Embedder,
    EmbeddingSpace,
    InputType,
    ProviderResult,
    resolve_space,
)
from rag_engine.repositories import JobRepository, KnowledgeBaseRepository, SourceRepository
from rag_engine.runtime import AppRuntime
from rag_engine.services import search_scope
from rag_engine.vectorstores import CollectionSpec, VectorFilter
from rag_engine.worker.context import worker_context
from rag_engine.worker.crawl_job import CrawlJobDeps, CrawlJobRunner, JobOutcome
from tests.conftest import OverrideSettings
from tests.support.fake_openai import FakeOpenAI
from tests.support.fakes import FakeEmbedder
from tests.support.fixture_site import FixtureSite

_TENANT = "ten_" + "1" * 32
_DIMS = 256


@pytest.fixture(scope="module")
def fake_openai() -> Iterator[FakeOpenAI]:
    server = FakeOpenAI().start()
    yield server
    server.stop()


@pytest.fixture
def config(
    integration_settings: Settings, override_settings: OverrideSettings, fake_openai: FakeOpenAI
) -> Settings:
    fake_openai.fail_with(None)
    return override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True,
        CRAWL_JS_RENDERING=JsRenderMode.NEVER,
        CRAWL_CONCURRENCY=4,
        CRAWL_PER_DOMAIN_RPS=50.0,
        CRAWL_READ_TIMEOUT_SECONDS=1,
        CRAWL_FETCH_TIMEOUT_SECONDS=2,
        CRAWL_MAX_RETRIES=1,
        CRAWL_RETRY_BASE_DELAY_SECONDS=0.05,
        CRAWL_RETRY_MAX_DELAY_SECONDS=0.1,
        CRAWL_HEARTBEAT_SECONDS=0.05,
        CRAWL_JOB_RETRY_DELAY_SECONDS=5,
        OPENAI_BASE_URL=fake_openai.base_url,
        OPENAI_MAX_RETRIES=0,
        EMBEDDING_CACHE_ENABLED=True,
    )


@dataclass
class _World:
    runtime: AppRuntime
    site: FixtureSite
    config: Settings
    enqueued: list[tuple[str, float]] = field(default_factory=list)

    @property
    def jobs(self) -> JobRepository:
        return JobRepository(self.runtime.mongo.db)

    @property
    def sources(self) -> SourceRepository:
        return SourceRepository(self.runtime.mongo.db)

    async def job(self, *, kb: KnowledgeBase | None = None, max_pages: int = 10) -> CrawlJob:
        return await _seed_job(self.runtime, self.site, kb=kb, max_pages=max_pages)

    def runner(self, **overrides: object) -> CrawlJobRunner:
        deps = CrawlJobDeps.from_runtime(
            self.runtime, enqueue=lambda job_id, delay: self.enqueued.append((job_id, delay)),
            **overrides,
        )  # fmt: skip
        return CrawlJobRunner(deps)


async def _seed_job(
    runtime: AppRuntime, site: FixtureSite, *, kb: KnowledgeBase | None = None, max_pages: int = 10
) -> CrawlJob:
    db = runtime.mongo.db
    if kb is None:
        kb = await KnowledgeBaseRepository(db).insert(
            KnowledgeBase(tenant_id=_TENANT, name=f"kb-{id(site)}-{max_pages}",
                          embedding_model="text-embedding-3-small", embedding_dimensions=_DIMS)
        )  # fmt: skip
    seed = site.url("/docs/")
    source, _ = await SourceRepository(db).get_or_create(_TENANT, kb.id, seed)
    return await JobRepository(db).insert(
        CrawlJob(tenant_id=_TENANT, kb_id=kb.id, source_id=source.id, seed_url=seed,
                 config=CrawlConfig(max_pages=max_pages, max_depth=3, path_prefix="/docs/"))
    )  # fmt: skip


@pytest.fixture
async def world(config: Settings, fixture_site: FixtureSite) -> AsyncIterator[_World]:
    runtime = AppRuntime()
    await runtime.start()
    yield _World(runtime, fixture_site, config)
    await runtime.stop()


def _spec() -> CollectionSpec:
    return CollectionSpec.for_space(resolve_space("text-embedding-3-small", _DIMS))


async def _points(world: _World, job: CrawlJob) -> int:
    store = world.runtime.vector_store
    assert store is not None
    return await store.count(_spec(), VectorFilter(_TENANT, versions=(job.id,)))


async def _until_running(world: _World, job: CrawlJob) -> CrawlJob:
    for _ in range(500):  # ≤ 10 s
        current = await world.jobs.get_or_raise(job.id)
        if current.status is JobStatus.RUNNING:
            return current
        await asyncio.sleep(0.02)
    raise AssertionError("the job never started")


class _SlowEmbedder(FakeEmbedder):
    """Calls `on_first` on its first batch, then takes its time."""

    def __init__(self, *args: object, on_first=None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.on_first = on_first

    async def _embed(self, texts: list[str], input_type: InputType) -> ProviderResult:
        if self.on_first is not None:
            hook, self.on_first = self.on_first, None
            await hook()
        await asyncio.sleep(0.5)
        return await super()._embed(texts, input_type)


# ---- #
# Tests
# ---- #


class TestRunCrawlJob_S77:
    """S77 — run_crawl_job: transitions, heartbeat, cancellation, error capture, finalize."""

    async def test_a_job_runs_to_success_and_goes_live(
        self, world: _World, fake_openai: FakeOpenAI
    ) -> None:
        job = await world.job()

        outcome = await world.runner().run(job.id)

        done = await world.jobs.get_or_raise(job.id)
        source = await world.sources.get_or_raise(job.source_id)
        assert outcome is JobOutcome.SUCCEEDED
        assert (done.status, done.error, done.heartbeat_at) == (JobStatus.SUCCEEDED, None, None)
        assert source.active_version == job.id  # the atomic swap happened
        assert source.stats.pages >= 3 and source.stats.chunks == done.counters.chunked > 0
        assert done.counters.embedded == done.counters.chunked == await _points(world, job)
        assert done.usage.requests > 0 and fake_openai.requests  # the real client, over HTTP
        assert done.report is not None and done.report.unfinished == 0
        scope = await search_scope(world.sources, _TENANT, job.kb_id)
        assert scope is not None and scope.versions == (job.id,)

    async def test_a_redelivered_task_is_a_no_op(self, world: _World) -> None:
        job = await world.job()
        await world.runner().run(job.id)
        before = await world.jobs.get_or_raise(job.id)

        again = await world.runner().run(job.id)

        assert again is JobOutcome.SKIPPED
        assert (await world.jobs.get_or_raise(job.id)).updated_at == before.updated_at

    async def test_a_recrawl_flips_to_the_new_version(self, world: _World) -> None:
        first = await world.job()
        await world.runner().run(first.id)
        kb = await KnowledgeBaseRepository(world.runtime.mongo.db).get_or_raise(first.kb_id)
        second = await world.job(kb=kb)

        assert await world.runner().run(second.id) is JobOutcome.SUCCEEDED

        source = await world.sources.get_or_raise(first.source_id)
        assert source.active_version == second.id
        assert await _points(world, first) > 0  # the old version waits for cleanup (Step 78)

    async def test_a_cancel_during_the_crawl_stops_it(
        self, world: _World, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_PER_DOMAIN_RPS=3.0, CRAWL_CONCURRENCY=1)  # a slow crawl
        job = await world.job(max_pages=20)
        running = asyncio.create_task(world.runner().run(job.id))
        await _until_running(world, job)
        await asyncio.sleep(0.3)

        await world.jobs.request_cancel(_TENANT, job.id)  # Mongo only: the heartbeat finds it
        outcome = await running

        done = await world.jobs.get_or_raise(job.id)
        assert outcome is JobOutcome.CANCELLED and done.status is JobStatus.CANCELLED
        assert (await world.sources.get_or_raise(job.source_id)).active_version is None

    async def test_a_cancel_during_ingestion_stops_it(self, world: _World) -> None:
        job = await world.job()

        async def cancel() -> None:
            await world.jobs.request_cancel(_TENANT, job.id)

        runner = world.runner(embedder_factory=lambda space: _SlowEmbedder(space, on_first=cancel))

        outcome = await runner.run(job.id)

        assert outcome is JobOutcome.CANCELLED
        assert (await world.jobs.get_or_raise(job.id)).status is JobStatus.CANCELLED
        assert (await world.sources.get_or_raise(job.source_id)).active_version is None

    async def test_a_permanent_embedding_error_fails_the_job(
        self, world: _World, fake_openai: FakeOpenAI
    ) -> None:
        fake_openai.fail_with(401)
        job = await world.job()

        outcome = await world.runner().run(job.id)

        done = await world.jobs.get_or_raise(job.id)
        assert outcome is JobOutcome.FAILED
        assert done.error is not None and done.error.code == "embedding_auth_failed"
        assert done.report is not None  # the crawl's report survives the failure
        assert (await world.sources.get_or_raise(job.source_id)).active_version is None
        assert world.enqueued == []

    async def test_a_transient_outage_requeues_until_attempts_run_out(
        self, world: _World, fake_openai: FakeOpenAI
    ) -> None:
        fake_openai.fail_with(500)
        job = await world.job()

        outcomes = [await world.runner().run(job.id) for _ in range(3)]

        done = await world.jobs.get_or_raise(job.id)
        assert outcomes == [JobOutcome.REQUEUED, JobOutcome.REQUEUED, JobOutcome.FAILED]
        assert world.enqueued == [(job.id, 5.0), (job.id, 10.0)]  # backoff x attempt
        assert (done.attempts, done.status) == (3, JobStatus.FAILED)
        assert done.error is not None and done.error.code == "embedding_unavailable"

    async def test_a_deleted_source_fails_the_job_before_crawling(self, world: _World) -> None:
        job = await world.job()
        await world.sources.mark_deleting(job.source_id, _TENANT, job.kb_id)

        outcome = await world.runner().run(job.id)

        done = await world.jobs.get_or_raise(job.id)
        assert outcome is JobOutcome.FAILED
        assert done.error is not None and done.error.code == "source_deleted"
        assert world.site.hits("/docs/") == 0

    async def test_a_superseded_run_stops_without_touching_the_job(
        self, world: _World, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_PER_DOMAIN_RPS=3.0, CRAWL_CONCURRENCY=1)
        job = await world.job(max_pages=20)
        running = asyncio.create_task(world.runner().run(job.id))
        held = await _until_running(world, job)
        assert held.run_id is not None

        await world.jobs.requeue(job.id, stale_run_id=held.run_id)  # Step 80 took it over
        outcome = await running

        assert outcome is JobOutcome.SUPERSEDED
        assert (await world.jobs.get_or_raise(job.id)).status is JobStatus.QUEUED

    async def test_a_bug_fails_the_job_as_an_internal_error(self, world: _World) -> None:
        job = await world.job()

        def broken(space: EmbeddingSpace) -> Embedder:
            raise RuntimeError("embedder misconfigured")

        outcome = await world.runner(embedder_factory=broken).run(job.id)

        done = await world.jobs.get_or_raise(job.id)
        assert outcome is JobOutcome.FAILED
        assert done.error is not None and done.error.code == "internal_error"
        assert "embedder misconfigured" in done.error.message
        queued = await world.runtime.mongo.db["pages"].count_documents(
            {"job_id": job.id, "status": {"$in": [PageStatus.QUEUED.value, "in_progress"]}}
        )
        assert queued == 0


class TestRunCrawlJobTask_S77:
    """S77 — the Celery task itself, end to end in-process (fake OpenAI over HTTP)."""

    def test_the_task_runs_a_job_to_success(
        self, config: Settings, fixture_site: FixtureSite, fake_openai: FakeOpenAI
    ) -> None:
        from rag_engine.worker.tasks.crawl import run_crawl_job

        try:
            job = worker_context.run(_seed_job(worker_context.runtime, fixture_site))

            outcome = run_crawl_job.apply(args=[job.id]).get(timeout=120)

            done = worker_context.run(JobRepository(worker_context.runtime.mongo.db).get(job.id))
            source = worker_context.run(
                SourceRepository(worker_context.runtime.mongo.db).get(job.source_id)
            )
            assert outcome == "succeeded"
            assert done is not None and done.status is JobStatus.SUCCEEDED
            assert isinstance(source, Source) and source.active_version == job.id
            assert fake_openai.requests
        finally:
            worker_context.shutdown()
