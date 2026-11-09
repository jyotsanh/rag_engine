"""Ingestion pipeline: stored pages → chunks (Mongo) → embeddings → points under the version."""

import asyncio
import zlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import pytest

from rag_engine.chunking import ChunkingConfig, StructureChunker, tokenizer_for
from rag_engine.config import Settings
from rag_engine.domain import (
    Chunk,
    ContentKind,
    CrawlJob,
    JobError,
    Page,
    PageReason,
    PageStatus,
    TenantUsage,
    UsagePurpose,
)
from rag_engine.embeddings import (
    CachedEmbedder,
    EmbeddingAuthError,
    InputType,
    ProviderResult,
    SparseVector,
    resolve_space,
)
from rag_engine.infrastructure import MongoConnection, QdrantConnection
from rag_engine.ingestion import IngestionPipeline
from rag_engine.repositories import (
    ChunkRepository,
    JobRepository,
    JobRunSupersededError,
    PageRepository,
    UsageRepository,
)
from rag_engine.services import UsageMeter
from rag_engine.vectorstores import (
    CollectionSpec,
    QdrantVectorStore,
    VectorCollections,
    VectorFilter,
    VectorPoint,
)
from tests.conftest import OverrideSettings
from tests.support.fakes import FakeEmbedder, InMemoryEmbeddingCache, InMemoryVectorStore

_TENANT, _KB, _SOURCE = "ten_" + "1" * 32, "kb_" + "1" * 32, "src_" + "1" * 32
_SPACE = resolve_space("text-embedding-3-small", 256)
_BANNER = "We use cookies to improve your experience. By continuing you accept our policy."
_CHUNKER = StructureChunker(
    tokenizer_for(_SPACE.model), ChunkingConfig(target_tokens=60, max_tokens=160, overlap_tokens=10)
)


class _FakeSparse:
    """Deterministic sparse vectors: one index per distinct word (crc32)."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        self.calls += 1
        out = []
        for text in texts:
            words = {zlib.crc32(w.lower().encode()) for w in text.split()}
            out.append(SparseVector.from_pairs(sorted(words), [1.0] * len(words)))
        return out


def _markdown(n: int, sections: int = 3) -> str:
    body = "\n\n".join(
        f"## Topic {n}.{s}\n\nThe WX-{n}{s} widget sorts {n * 100 + s} items per minute. "
        f"It is calibrated weekly and reports its status every {s + 5} seconds."
        for s in range(sections)
    )
    return f"# Page {n}\n\n{_BANNER}\n\n{body}\n"


@dataclass
class _Env:
    mongo: MongoConnection
    pages: PageRepository
    chunks: ChunkRepository
    jobs: JobRepository
    usage: UsageRepository
    config: Settings

    async def job(self) -> CrawlJob:
        job = await self.jobs.insert(
            CrawlJob(tenant_id=_TENANT, kb_id=_KB, source_id=_SOURCE,
                     seed_url="https://acme.example/")
        )  # fmt: skip
        return await self.jobs.start(job.id, "run-1")

    async def page(self, job: CrawlJob, n: int, markdown: str | None, **fields: object) -> Page:
        values: dict[str, object] = {
            "tenant_id": _TENANT, "kb_id": _KB, "source_id": _SOURCE, "job_id": job.id,
            "url": f"https://acme.example/p{n}", "url_hash": f"{n:064x}",
            "title": f"Page {n} | Acme", "status": PageStatus.SUCCEEDED,
            "content_kind": ContentKind.HTML, "markdown": markdown,
        }  # fmt: skip
        return await self.pages.insert(Page.model_validate({**values, **fields}))

    def pipeline(self, store, embedder=None, sparse=None, **kwargs) -> IngestionPipeline:
        return IngestionPipeline(
            pages=self.pages, chunks=self.chunks, jobs=self.jobs, chunker=_CHUNKER,
            embedder=embedder or FakeEmbedder(_SPACE), sparse=sparse or _FakeSparse(),
            store=store, collections=VectorCollections(store),
            meter=UsageMeter(self.jobs, self.usage), config=self.config, **kwargs,
        )  # fmt: skip


@pytest.fixture
async def env(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[_Env]:
    config = override_settings(INGEST_EMBED_BATCH_SIZE=16, INGEST_QUEUE_SIZE=2)
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Page, Chunk, CrawlJob, TenantUsage])
    yield _Env(mongo, PageRepository(mongo.db), ChunkRepository(mongo.db),
               JobRepository(mongo.db), UsageRepository(mongo.db), config)  # fmt: skip
    await mongo.close()


def _spec() -> CollectionSpec:
    return CollectionSpec.for_space(_SPACE)


def _version_filter(job: CrawlJob, **extra: object) -> VectorFilter:
    return VectorFilter(_TENANT, versions=(job.version,), **extra)  # ty: ignore[invalid-argument-type]


# ---- #
# Tests
# ---- #


class TestIngestionPipeline_S76:
    """S76 — Ingestion: strip → chunk → embed (dense + sparse) → upsert, staged and bounded."""

    async def test_pages_become_stored_chunks_and_points(self, env: _Env) -> None:
        job = await env.job()
        for n in range(8):
            await env.page(job, n, _markdown(n))
        store = InMemoryVectorStore()

        report = await env.pipeline(store).run(job, "run-1")

        stored = [c async for c in env.chunks.iter_version(_SOURCE, job.version)]
        assert report.pages == 8 and report.empty_pages == 0
        assert report.chunks == len(stored) == report.embedded > 8  # several chunks per page
        assert await store.count(_spec(), _version_filter(job)) == len(stored)
        assert not any(_BANNER in c.text for c in stored)  # the job-level strip ran
        points = store.collections[_spec().name][1]
        assert {p.payload.chunk_id for p in points.values()} == {c.id for c in stored}
        assert all(p.sparse and len(p.sparse) for p in points.values())
        counters = (await env.jobs.get_or_raise(job.id)).counters
        assert (counters.chunked, counters.embedded) == (len(stored), len(stored))
        usage = (await env.jobs.get_or_raise(job.id)).usage
        assert usage.requests == report.batches and usage.tokens == report.tokens > 0
        (row,) = await env.usage.for_tenant(_TENANT)
        assert (row.purpose, row.tokens) == (UsagePurpose.INGEST, report.tokens)

    async def test_only_the_jobs_succeeded_pages_are_ingested(self, env: _Env) -> None:
        # an earlier job of the same source (finished: a source has one active job, S79)
        other = await env.job()
        await env.jobs.fail(other.id, JobError(code="engine_crashed", message="boom"))
        job = await env.job()
        await env.page(job, 1, _markdown(1))
        await env.page(job, 2, None, status=PageStatus.FAILED, reason=PageReason.TIMEOUT,
                       content_kind=None)  # fmt: skip
        await env.page(other, 3, _markdown(3))

        report = await env.pipeline(InMemoryVectorStore()).run(job, "run-1")

        assert report.pages == 1
        assert await env.chunks.count_version(_SOURCE, other.version) == 0

    async def test_a_page_left_empty_by_the_strip_has_no_chunks(self, env: _Env) -> None:
        job = await env.job()
        for n in range(6):
            await env.page(job, n, _markdown(n))
        empty = await env.page(job, 99, f"# Cookies\n\n{_BANNER}\n")

        report = await env.pipeline(InMemoryVectorStore()).run(job, "run-1")

        assert report.empty_pages == 1
        assert await env.chunks.for_page(empty.id) == []

    async def test_a_rerun_is_idempotent_and_free_through_the_cache(self, env: _Env) -> None:
        job = await env.job()
        for n in range(5):
            await env.page(job, n, _markdown(n))
        store, cache = InMemoryVectorStore(), InMemoryEmbeddingCache()
        first_inner = FakeEmbedder(_SPACE)
        first = await env.pipeline(store, CachedEmbedder(first_inner, cache)).run(job, "run-1")

        second_inner = FakeEmbedder(_SPACE)
        second = await env.pipeline(store, CachedEmbedder(second_inner, cache)).run(job, "run-1")

        assert second.chunks == first.chunks == second.embedded
        assert await store.count(_spec(), _version_filter(job)) == first.chunks  # no duplicates
        assert second_inner.calls == [] and second.cached == first.chunks  # nothing re-paid
        counters = (await env.jobs.get_or_raise(job.id)).counters
        assert counters.chunked == counters.embedded == first.chunks  # reconciled, not doubled

    async def test_a_page_that_rechunks_smaller_loses_its_stale_points(self, env: _Env) -> None:
        job = await env.job()
        page = await env.page(job, 1, _markdown(1, sections=6))
        store = InMemoryVectorStore()
        before = await env.pipeline(store).run(job, "run-1")
        await env.pages.collection.update_one(  # the page's content shrinks (e.g. a re-run)
            {"_id": page.id}, {"$set": {"markdown": _markdown(1, sections=1)}}
        )

        after = await env.pipeline(store).run(job, "run-1")

        assert after.chunks < before.chunks
        assert await store.count(_spec(), _version_filter(job, page_ids=(page.id,))) == after.chunks
        assert len(await env.chunks.for_page(page.id)) == after.chunks

    async def test_embedding_batches_span_pages_within_the_size(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(INGEST_EMBED_BATCH_SIZE=5)
        job = await env.job()
        for n in range(6):
            await env.page(job, n, _markdown(n, sections=2))
        embedder, batches = FakeEmbedder(_SPACE), 0

        async def count_batches() -> None:
            nonlocal batches
            batches += 1

        report = await env.pipeline(InMemoryVectorStore(), embedder, on_batch=count_batches).run(
            job, "run-1"
        )

        sizes = [len(texts) for texts, kind in embedder.calls if kind is InputType.DOCUMENT]
        assert all(size <= 5 for size in sizes) and sum(sizes) == report.chunks
        assert max(sizes) == 5 and batches == report.batches == len(sizes)

    async def test_stages_apply_backpressure(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(INGEST_EMBED_BATCH_SIZE=2, INGEST_QUEUE_SIZE=1)
        job = await env.job()
        for n in range(10):
            await env.page(job, n, _markdown(n, sections=2))
        upserted, ahead = 0, 0

        class _SlowStore(InMemoryVectorStore):
            async def _upsert(self, spec: CollectionSpec, points: list[VectorPoint]) -> None:
                nonlocal upserted, ahead
                saved = await env.chunks.count_version(_SOURCE, job.version)
                ahead = max(ahead, saved - upserted)
                await asyncio.sleep(0.02)
                await super()._upsert(spec, points)
                upserted += len(points)

        await env.pipeline(_SlowStore()).run(job, "run-1")

        # queue 1 + one batch in each stage + one page being chunked: a small, fixed window
        assert ahead <= 2 * (1 + 3) + 6, ahead

    async def test_a_permanent_embedding_error_stops_the_pipeline(self, env: _Env) -> None:
        job = await env.job()
        for n in range(4):
            await env.page(job, n, _markdown(n))
        embedder = FakeEmbedder(_SPACE)

        def reject(texts: list[str], kind: InputType) -> ProviderResult:
            raise EmbeddingAuthError()

        embedder.respond = reject

        with pytest.raises(EmbeddingAuthError):  # itself, not an ExceptionGroup
            async with asyncio.timeout(20):
                await env.pipeline(InMemoryVectorStore(), embedder).run(job, "run-1")

    async def test_a_superseded_run_stops(self, env: _Env) -> None:
        job = await env.job()
        await env.page(job, 1, _markdown(1))
        await env.jobs.requeue(job.id, stale_run_id="run-1")

        with pytest.raises(JobRunSupersededError):
            async with asyncio.timeout(20):
                await env.pipeline(InMemoryVectorStore()).run(job, "run-1")

    async def test_end_to_end_into_qdrant(self, env: _Env, integration_settings: Settings) -> None:
        job = await env.job()
        for n in range(4):
            await env.page(job, n, _markdown(n))
        qdrant = QdrantConnection(integration_settings)
        await qdrant.connect()
        store = QdrantVectorStore(qdrant.client)
        try:
            report = await env.pipeline(store).run(job, "run-1")

            assert await store.count(_spec(), _version_filter(job)) == report.chunks > 0
            query = (await FakeEmbedder(_SPACE).embed_query("anything")).vector
            hits = await store.search_dense(_spec(), query, _version_filter(job), limit=3)
            assert len(hits) == 3 and all(h.payload.version == job.version for h in hits)
        finally:
            await qdrant.close()
