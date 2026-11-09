"""Cascade deletes in real MongoDB (+ the reference vector store, and one run on real
Qdrant): a deleted source's / knowledge base's vectors, chunks, pages, jobs and records."""

import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from rag_engine.config import Settings
from rag_engine.domain import (
    Chunk,
    ContentHashClaim,
    ContentKind,
    CrawlJob,
    JobStatus,
    KnowledgeBase,
    Page,
    PageStatus,
    ResourceStatus,
    Source,
    SourceStats,
    Tenant,
    chunk_id,
    new_id,
    utc_now,
)
from rag_engine.embeddings import SparseVector
from rag_engine.infrastructure import MongoConnection, QdrantConnection
from rag_engine.repositories import (
    ChunkRepository,
    ContentClaimRepository,
    JobRepository,
    KnowledgeBaseRepository,
    PageRepository,
)
from rag_engine.repositories.sources import SourceRepository
from rag_engine.services import CascadeDeleter, CascadeStatus, DeletionSweeper
from rag_engine.vectorstores import (
    BaseVectorStore,
    CollectionSpec,
    QdrantVectorStore,
    VectorFilter,
    VectorPoint,
)
from tests.conftest import OverrideSettings
from tests.support.fakes import InMemoryVectorStore, fake_vector

_TENANT = "ten_" + "1" * 32
_OTHER_TENANT = "ten_" + "2" * 32
_DIMS = 256


@dataclass
class _Env:
    mongo: MongoConnection
    store: BaseVectorStore
    config: Settings
    sent: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def kbs(self) -> KnowledgeBaseRepository:
        return KnowledgeBaseRepository(self.mongo.db)

    @property
    def sources(self) -> SourceRepository:
        return SourceRepository(self.mongo.db)

    @property
    def jobs(self) -> JobRepository:
        return JobRepository(self.mongo.db)

    @property
    def deleter(self) -> CascadeDeleter:
        return CascadeDeleter(self.mongo.db, self.store)

    async def kb(self, tenant_id: str = _TENANT) -> KnowledgeBase:
        kb = await self.kbs.insert(
            KnowledgeBase(tenant_id=tenant_id, name=new_id("kb"), embedding_dimensions=_DIMS,
                          embedding_model="text-embedding-3-small")
        )  # fmt: skip
        await self.store.ensure_collection(_spec(kb))
        return kb

    async def source(self, kb: KnowledgeBase, n: int = 0) -> Source:
        source, _ = await self.sources.get_or_create(
            kb.tenant_id, kb.id, f"https://site{n}.example/"
        )
        return source

    async def crawled(self, source: Source, *, pages: int = 3, live: bool = True) -> CrawlJob:
        """A finished job of `source` with pages, chunks, vectors and a content claim."""
        jobs = self.jobs
        job = await jobs.insert(
            CrawlJob(tenant_id=source.tenant_id, kb_id=source.kb_id, source_id=source.id,
                     seed_url=source.seed_url)
        )  # fmt: skip
        await self._data(job, pages)
        run_id = new_id("run")
        await jobs.start(job.id, run_id)
        await jobs.begin_finalizing(job.id, run_id)
        if live:
            await self.sources.activate_version(source.id, job.version, stats=SourceStats())
        await jobs.succeed(job.id, run_id)
        return job

    async def _data(self, job: CrawlJob, pages: int) -> None:
        kb = await self.kbs.get_or_raise(job.kb_id)
        points = []
        for n in range(pages):
            page = await PageRepository(self.mongo.db).insert(
                Page(tenant_id=job.tenant_id, kb_id=job.kb_id, source_id=job.source_id,
                     job_id=job.id, url=f"{job.seed_url}p{n}", url_hash=f"{n:064x}",
                     status=PageStatus.SUCCEEDED, content_kind=ContentKind.HTML,
                     markdown=f"# Page {n}")
            )  # fmt: skip
            text = f"page {n} of {job.id}"
            chunk = Chunk(
                id=chunk_id(page.id, 0), tenant_id=job.tenant_id, kb_id=job.kb_id,
                source_id=job.source_id, version=job.version, page_id=page.id, url=page.url,
                ordinal=0, text=text, embed_text=text, token_count=5,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
            )  # fmt: skip
            await ChunkRepository(self.mongo.db).save_page_chunks(page.id, [chunk])
            points.append(
                VectorPoint.from_chunk(chunk, fake_vector(text, _DIMS), SparseVector((n,), (1.0,)))
            )
        await self.store.upsert(_spec(kb), points)
        await ContentClaimRepository(self.mongo.db).claim(
            job.id, "c" * 64, new_id("page"), job.seed_url
        )

    async def points(
        self, kb: KnowledgeBase, *, kb_ids: tuple[str, ...] = (), source_ids: tuple[str, ...] = ()
    ) -> int:
        where = VectorFilter(kb.tenant_id, kb_ids=kb_ids, source_ids=source_ids)
        return await self.store.count(_spec(kb), where)

    async def mongo_count(self, collection: str, **filter_: str) -> int:
        return await self.mongo.db[collection].count_documents(filter_)


def _spec(kb: KnowledgeBase) -> CollectionSpec:
    return CollectionSpec.for_space(kb.embedding_space)


@pytest.fixture
async def env(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[_Env]:
    config = override_settings(CASCADE_STALE_SECONDS=900, CASCADE_RETRY_DELAY_SECONDS=30)
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Tenant, KnowledgeBase, Source, CrawlJob, Page, Chunk,
                                ContentHashClaim])  # fmt: skip
    yield _Env(mongo, InMemoryVectorStore(), config)
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestCascadeDeletes_S81:
    """S81 — `CascadeDeleter`: source and KB cascades (vectors, chunks, pages, claims, jobs,
    records), waiting for crawl jobs to stop, idempotency, isolation."""

    async def test_a_source_cascade_removes_everything_of_that_source_only(self, env: _Env) -> None:
        # Arrange: two crawls of the doomed source (old + live), and a neighbour source
        kb = await env.kb()
        doomed, neighbour = await env.source(kb, 1), await env.source(kb, 2)
        old = await env.crawled(doomed, pages=2)
        live = await env.crawled(doomed, pages=3)
        kept = await env.crawled(neighbour, pages=4)
        await env.sources.mark_deleting(doomed.id, _TENANT, kb.id)

        # Act
        result = await env.deleter.delete_source(_TENANT, doomed.id)

        # Assert
        assert result.status is CascadeStatus.DELETED
        assert (result.points, result.chunks, result.pages, result.jobs) == (5, 5, 5, 2)
        assert await env.points(kb, source_ids=(doomed.id,)) == 0
        for collection in ("chunks", "pages"):
            assert await env.mongo_count(collection, source_id=doomed.id) == 0
        assert await env.mongo_count("crawl_jobs", source_id=doomed.id) == 0
        for job in (old, live):
            assert await env.mongo_count("content_hash_claims", job_id=job.id) == 0
        assert await env.sources.get(doomed.id) is None
        # the neighbour is untouched
        assert await env.points(kb, source_ids=(neighbour.id,)) == 4
        assert await env.mongo_count("chunks", source_id=neighbour.id) == 4
        assert await env.mongo_count("content_hash_claims", job_id=kept.id) == 1
        assert (await env.sources.get_or_raise(neighbour.id)).status is ResourceStatus.ACTIVE

    async def test_a_live_source_is_never_cascaded(self, env: _Env) -> None:
        kb = await env.kb()
        source = await env.source(kb)
        await env.crawled(source)

        result = await env.deleter.delete_source(_TENANT, source.id)

        assert result.status is CascadeStatus.NOT_DELETING
        assert await env.points(kb, source_ids=(source.id,)) == 3
        assert await env.sources.get(source.id) is not None

    async def test_a_repeated_or_foreign_cascade_is_a_no_op(self, env: _Env) -> None:
        kb = await env.kb()
        source = await env.source(kb)
        await env.crawled(source)
        await env.sources.mark_deleting(source.id, _TENANT, kb.id)

        foreign = await env.deleter.delete_source(_OTHER_TENANT, source.id)
        first = await env.deleter.delete_source(_TENANT, source.id)
        again = await env.deleter.delete_source(_TENANT, source.id)

        assert foreign.status is CascadeStatus.ALREADY_GONE  # another tenant can't reach it
        assert first.status is CascadeStatus.DELETED
        assert again.status is CascadeStatus.ALREADY_GONE

    async def test_a_queued_job_is_cancelled_and_the_cascade_proceeds(self, env: _Env) -> None:
        kb = await env.kb()
        source = await env.source(kb)
        await env.crawled(source)
        queued = await env.jobs.insert(
            CrawlJob(tenant_id=_TENANT, kb_id=kb.id, source_id=source.id,
                     seed_url=source.seed_url)
        )  # fmt: skip
        await env.sources.mark_deleting(source.id, _TENANT, kb.id)

        result = await env.deleter.delete_source(_TENANT, source.id)

        assert result.status is CascadeStatus.DELETED and result.jobs == 2
        assert await env.jobs.get(queued.id) is None

    async def test_a_running_job_is_flagged_and_the_cascade_waits(self, env: _Env) -> None:
        # Arrange: a crawl still writing vectors under its own version
        kb = await env.kb()
        source = await env.source(kb)
        await env.crawled(source)
        running = await env.jobs.insert(
            CrawlJob(tenant_id=_TENANT, kb_id=kb.id, source_id=source.id,
                     seed_url=source.seed_url)
        )  # fmt: skip
        await env.jobs.start(running.id, "run-1")
        await env.sources.mark_deleting(source.id, _TENANT, kb.id)

        # Act: the cascade asks the job to stop and leaves every byte in place
        waiting = await env.deleter.delete_source(_TENANT, source.id)

        # Assert
        flagged = await env.jobs.get_or_raise(running.id)
        assert waiting.status is CascadeStatus.WAITING_FOR_JOBS
        assert waiting.waiting_for == [running.id]
        assert flagged.cancel_requested_at is not None and flagged.status is JobStatus.RUNNING
        assert await env.points(kb, source_ids=(source.id,)) == 3
        assert await env.sources.get(source.id) is not None

        # the worker sees the flag and cancels; the retried cascade then completes
        await env.jobs.cancel(running.id, "run-1")
        done = await env.deleter.delete_source(_TENANT, source.id)
        assert done.status is CascadeStatus.DELETED
        assert await env.points(kb, source_ids=(source.id,)) == 0

    async def test_a_finalizing_job_is_waited_for(self, env: _Env) -> None:
        kb = await env.kb()
        source = await env.source(kb)
        job = await env.jobs.insert(
            CrawlJob(tenant_id=_TENANT, kb_id=kb.id, source_id=source.id,
                     seed_url=source.seed_url)
        )  # fmt: skip
        await env.jobs.start(job.id, "run-1")
        await env.jobs.begin_finalizing(job.id, "run-1")
        await env.sources.mark_deleting(source.id, _TENANT, kb.id)

        result = await env.deleter.delete_source(_TENANT, source.id)

        current = await env.jobs.get_or_raise(job.id)
        assert result.status is CascadeStatus.WAITING_FOR_JOBS
        assert current.status is JobStatus.FINALIZING  # not cancellable: left to finish

    async def test_a_cascade_touches_the_source_it_works_on(self, env: _Env) -> None:
        kb = await env.kb()
        source = await env.source(kb)
        job = await env.jobs.insert(
            CrawlJob(tenant_id=_TENANT, kb_id=kb.id, source_id=source.id,
                     seed_url=source.seed_url)
        )  # fmt: skip
        await env.jobs.start(job.id, "run-1")
        marked = await env.sources.mark_deleting(source.id, _TENANT, kb.id)

        await env.deleter.delete_source(_TENANT, source.id)  # waits

        touched = await env.sources.get_or_raise(source.id)
        assert touched.updated_at >= marked.updated_at

    async def test_a_kb_cascade_removes_all_its_sources_and_the_kb(self, env: _Env) -> None:
        # Arrange: two sources (one already deleting on its own), and another KB
        kb, other_kb = await env.kb(), await env.kb()
        first, second = await env.source(kb, 1), await env.source(kb, 2)
        await env.crawled(first, pages=2)
        await env.crawled(second, pages=3)
        await env.crawled(await env.source(other_kb, 3), pages=4)
        await env.sources.mark_deleting(second.id, _TENANT, kb.id)
        await env.kbs.mark_deleting(kb.id, _TENANT)

        # Act
        result = await env.deleter.delete_knowledge_base(_TENANT, kb.id)

        # Assert
        assert result.status is CascadeStatus.DELETED
        assert (result.sources, result.points, result.chunks, result.jobs) == (2, 5, 5, 2)
        assert await env.points(kb, kb_ids=(kb.id,)) == 0
        assert await env.kbs.get(kb.id) is None
        for collection in ("sources", "crawl_jobs", "pages", "chunks"):
            assert await env.mongo_count(collection, kb_id=kb.id) == 0
        assert await env.points(other_kb, kb_ids=(other_kb.id,)) == 4
        assert await env.mongo_count("sources", kb_id=other_kb.id) == 1

    async def test_a_kb_cascade_waits_for_every_job_of_the_kb(self, env: _Env) -> None:
        kb = await env.kb()
        idle, busy = await env.source(kb, 1), await env.source(kb, 2)
        await env.crawled(idle)
        job = await env.jobs.insert(
            CrawlJob(tenant_id=_TENANT, kb_id=kb.id, source_id=busy.id, seed_url=busy.seed_url)
        )
        await env.jobs.start(job.id, "run-1")
        await env.kbs.mark_deleting(kb.id, _TENANT)

        waiting = await env.deleter.delete_knowledge_base(_TENANT, kb.id)

        assert waiting.status is CascadeStatus.WAITING_FOR_JOBS
        assert waiting.waiting_for == [job.id]
        assert await env.points(kb, kb_ids=(kb.id,)) == 3  # nothing deleted yet, not even idle
        # its sources are hidden as deleting straight away
        statuses = {s.id: s.status for s in await env.sources.find({"kb_id": kb.id})}
        assert set(statuses.values()) == {ResourceStatus.DELETING}

        await env.jobs.cancel(job.id, "run-1")
        done = await env.deleter.delete_knowledge_base(_TENANT, kb.id)
        assert done.status is CascadeStatus.DELETED and await env.kbs.get(kb.id) is None

    async def test_a_live_kb_is_never_cascaded(self, env: _Env) -> None:
        kb = await env.kb()
        await env.crawled(await env.source(kb))

        result = await env.deleter.delete_knowledge_base(_TENANT, kb.id)

        assert result.status is CascadeStatus.NOT_DELETING
        assert await env.points(kb, kb_ids=(kb.id,)) == 3

    async def test_a_kb_cascade_survives_a_crash_halfway(self, env: _Env) -> None:
        # one source already fully cascaded (record gone) but its stray vectors remain
        kb = await env.kb()
        source = await env.source(kb)
        await env.crawled(source)
        await env.kbs.mark_deleting(kb.id, _TENANT)
        await env.sources.mark_kb_sources_deleting(kb.id)
        await env.sources.delete(source.id)  # the record went, the vectors didn't

        result = await env.deleter.delete_knowledge_base(_TENANT, kb.id)

        assert result.status is CascadeStatus.DELETED and result.points == 3  # KB-wide net
        assert await env.points(kb, kb_ids=(kb.id,)) == 0

    async def test_end_to_end_on_real_qdrant(
        self, env: _Env, integration_settings: Settings
    ) -> None:
        qdrant = QdrantConnection(integration_settings)
        await qdrant.connect()
        env.store = QdrantVectorStore(qdrant.client)
        try:
            kb = await env.kb()
            doomed, kept = await env.source(kb, 1), await env.source(kb, 2)
            await env.crawled(doomed, pages=3)
            await env.crawled(kept, pages=2)
            await env.sources.mark_deleting(doomed.id, _TENANT, kb.id)

            result = await env.deleter.delete_source(_TENANT, doomed.id)

            assert result.status is CascadeStatus.DELETED and result.points == 3
            assert await env.points(kb, source_ids=(doomed.id,)) == 0
            assert await env.points(kb, source_ids=(kept.id,)) == 2
        finally:
            await qdrant.close()


class TestDeletionSweep_S81:
    """S81 — `DeletionSweeper`: restarts cascades nobody touched for `CASCADE_STALE_SECONDS`."""

    def _sweeper(self, env: _Env, *, after: float) -> DeletionSweeper:
        later = utc_now() + timedelta(seconds=after)
        return DeletionSweeper(
            env.mongo.db,
            enqueue_source=lambda t, s: env.sent.append(("delete_source", t, s)),
            enqueue_kb=lambda t, k: env.sent.append(("delete_knowledge_base", t, k)),
            config=env.config,
            clock=lambda: later,
        )

    async def test_stalled_deletions_are_restarted_once_per_window(self, env: _Env) -> None:
        kb, gone_kb = await env.kb(), await env.kb()
        source, live = await env.source(kb, 1), await env.source(kb, 2)
        await env.sources.mark_deleting(source.id, _TENANT, kb.id)
        await env.kbs.mark_deleting(gone_kb.id, _TENANT)

        fresh = await self._sweeper(env, after=60).run()  # marked a minute ago: in progress
        stalled = await self._sweeper(env, after=901).run()
        again = await self._sweeper(env, after=901).run()  # touched by the first restart

        assert (fresh.sources, fresh.knowledge_bases) == ([], [])
        assert (stalled.sources, stalled.knowledge_bases) == ([source.id], [gone_kb.id])
        assert (again.sources, again.knowledge_bases) == ([], [])
        assert env.sent == [
            ("delete_knowledge_base", _TENANT, gone_kb.id),
            ("delete_source", _TENANT, source.id),
        ]
        assert live.id not in {s for _, _, s in env.sent}

    async def test_touch_deleting_is_conditional(self, env: _Env) -> None:
        kb = await env.kb()
        source = await env.source(kb)
        marked = await env.sources.mark_deleting(source.id, _TENANT, kb.id)
        live = await env.source(kb, 9)

        first = await env.sources.touch_deleting(source.id, marked.updated_at)
        stale_read = await env.sources.touch_deleting(source.id, marked.updated_at)
        not_deleting = await env.sources.touch_deleting(live.id)

        assert (first, stale_read, not_deleting) == (True, False, False)

    async def test_deleting_since_indexes_exist(self, env: _Env) -> None:
        for collection in ("sources", "knowledge_bases"):
            info = await env.mongo.db[collection].index_information()
            assert info["deleting_since"]["partialFilterExpression"] == {"status": "deleting"}


class TestCascadeTasks_S81:
    """S81 — the `delete_source` / `delete_knowledge_base` Celery tasks, in-process."""

    def test_the_tasks_cascade_a_source_and_a_kb(self, integration_settings: Settings) -> None:
        from rag_engine.worker.context import worker_context
        from rag_engine.worker.tasks.maintenance import delete_knowledge_base, delete_source

        try:
            runtime = worker_context.runtime
            assert runtime.vector_store is not None
            db = runtime.mongo.db
            env = _Env(runtime.mongo, runtime.vector_store, integration_settings)

            async def arrange() -> tuple[KnowledgeBase, Source]:
                kb = await env.kb()
                one, two = await env.source(kb, 1), await env.source(kb, 2)
                await env.crawled(one)
                await env.crawled(two)
                await env.sources.mark_deleting(one.id, _TENANT, kb.id)
                return kb, one

            kb, one = worker_context.run(arrange())

            by_source = delete_source.apply(args=[_TENANT, one.id]).get(timeout=60)
            worker_context.run(env.kbs.mark_deleting(kb.id, _TENANT))
            by_kb = delete_knowledge_base.apply(args=[_TENANT, kb.id]).get(timeout=60)

            assert by_source["status"] == "deleted" and by_source["points"] == 3
            assert by_kb["status"] == "deleted" and by_kb["sources"] == 1
            assert worker_context.run(db["knowledge_bases"].count_documents({"_id": kb.id})) == 0
            assert worker_context.run(db["sources"].count_documents({"kb_id": kb.id})) == 0
        finally:
            worker_context.shutdown()
