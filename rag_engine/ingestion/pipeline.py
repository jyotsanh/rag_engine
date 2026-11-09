"""Ingestion pipeline (Step 76): a crawled job's pages → chunks → vectors under its version.

**When it runs.** After the crawl, over the job's stored succeeded pages: the boilerplate
strip (Step 40) is a job-level statistic (a block is boilerplate only if most pages carry
it), so no page can be stripped, hence chunked, before the crawl has seen them all.
Ingestion is then a pure function of stored state: a crashed run is simply re-run.

**Stages**, connected by bounded queues (`INGEST_QUEUE_SIZE` batches), so memory stays
flat and embedding overlaps upserting:

1. **chunk** (pages in discovery order): strip the page's boilerplate, clean it again
   (Step 45 prunes emptied sections), chunk it (Steps 55-58, in a worker thread: CPU), and
   **save the page's chunks** in Mongo (idempotent, Step 59). Chunks are grouped into
   embedding batches of `INGEST_EMBED_BATCH_SIZE` that span pages;
2. **embed**: dense (`Embedder`, normally the cached OpenAI embedder: unchanged chunks
   cost nothing, Step 64) and sparse BM25 (Step 66) concurrently, both on `embed_text`.
   Each batch's usage is metered on the job and the tenant (Step 65);
3. **upsert**: points under the job's version (deterministic ids, Steps 67/69). A page
   whose new chunking has a different number of chunks than a previous run stored first
   loses its old points in this version, so no stale ordinal lingers.

Counters: `chunked` / `embedded` are reconciled at the end from what is stored (chunks of
the version in Mongo, points of the version in the vector store), so a re-run never
double counts. Any stage failure (e.g. a permanent `EmbeddingError`, a superseded run)
cancels the others and is raised as itself; the job task (Step 77) decides the outcome.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

from rag_engine.chunking import BaseChunker, ChunkInput
from rag_engine.config import Settings, settings
from rag_engine.domain import Chunk, CrawlJob, Page
from rag_engine.embeddings import Embedder, SparseVector
from rag_engine.extraction import BoilerplateIndex, ExtractionError, TextCleaner
from rag_engine.repositories import ChunkRepository, JobRepository, PageRepository
from rag_engine.services.usage import UsageMeter
from rag_engine.vectorstores import BaseVectorStore, CollectionSpec, VectorFilter, VectorPoint

_DONE = object()  # end-of-stream marker between stages


class SparseEmbedder(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...


class CollectionEnsurer(Protocol):
    """`VectorCollections` in production (Step 74)."""

    async def ensure(self, space) -> CollectionSpec: ...


@dataclass(frozen=True, slots=True)
class IngestReport:
    pages: int  # succeeded pages read
    empty_pages: int  # nothing left after the boilerplate strip / cleaning
    chunks: int  # chunks of the version stored in Mongo
    embedded: int  # points of the version in the vector store
    batches: int
    cached: int  # chunks whose dense vector came from the embedding cache
    tokens: int  # provider-billed embedding tokens


@dataclass
class _Batch:
    chunks: list[Chunk] = field(default_factory=list)
    # pages in this batch whose stored chunk count changed (their old points must go first)
    reset_pages: set[str] = field(default_factory=set)
    dense: list[list[float]] = field(default_factory=list)
    sparse: list[SparseVector] = field(default_factory=list)


@dataclass
class _Totals:
    pages: int = 0
    empty_pages: int = 0
    batches: int = 0
    cached: int = 0
    tokens: int = 0


class IngestionPipeline:
    def __init__(
        self,
        *,
        pages: PageRepository,
        chunks: ChunkRepository,
        jobs: JobRepository,
        chunker: BaseChunker,
        embedder: Embedder,
        sparse: SparseEmbedder,
        store: BaseVectorStore,
        collections: CollectionEnsurer,
        meter: UsageMeter,
        config: Settings = settings,
        cleaner: TextCleaner | None = None,
        on_batch: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """`on_batch` runs after every upserted batch (the job task's heartbeat, Step 77)."""
        self._pages = pages
        self._chunks = chunks
        self._jobs = jobs
        self._chunker = chunker
        self._embedder = embedder
        self._sparse = sparse
        self._store = store
        self._collections = collections
        self._meter = meter
        self._config = config
        self._cleaner = cleaner or TextCleaner(config=config)
        self._on_batch = on_batch
        self.batch_size = config.INGEST_EMBED_BATCH_SIZE
        self.queue_size = config.INGEST_QUEUE_SIZE

    async def run(self, job: CrawlJob, run_id: str) -> IngestReport:
        if job.run_id != run_id:
            raise ValueError("the job must be held by this run_id")
        spec = await self._collections.ensure(self._embedder.space)
        index = await self._boilerplate_index(job)
        totals = _Totals()
        to_embed: asyncio.Queue[_Batch | object] = asyncio.Queue(self.queue_size)
        to_upsert: asyncio.Queue[_Batch | object] = asyncio.Queue(self.queue_size)
        log = logger.bind(job_id=job.id, run_id=run_id)
        log.info("ingesting job {} into {}", job.id, spec.name)

        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._chunk_stage(job, index, totals, to_embed))
                group.create_task(self._embed_stage(job, run_id, totals, to_embed, to_upsert))
                group.create_task(self._upsert_stage(spec, job, to_upsert))
        except BaseExceptionGroup as failures:
            raise _first(failures) from None  # the failing stage's own error, not a group

        stored_chunks = await self._chunks.count_version(job.source_id, job.version)
        embedded = await self._store.count(
            spec, VectorFilter(job.tenant_id, source_ids=(job.source_id,), versions=(job.version,))
        )
        await self._jobs.reconcile_counters(
            job.id, run_id, chunked=stored_chunks, embedded=embedded
        )
        report = IngestReport(
            pages=totals.pages,
            empty_pages=totals.empty_pages,
            chunks=stored_chunks,
            embedded=embedded,
            batches=totals.batches,
            cached=totals.cached,
            tokens=totals.tokens,
        )
        log.info("ingested job {}: {}", job.id, report)
        return report

    # ---- stage 0: the job-level boilerplate statistics ---- #

    async def _boilerplate_index(self, job: CrawlJob) -> BoilerplateIndex:
        index = BoilerplateIndex(config=self._config)
        async for page in self._pages.iter_succeeded(job.id):
            if page.markdown:
                index.add(page.id, page.markdown)  # discovery order: the hub page keeps copies
        return index

    # ---- stage 1: chunk ---- #

    async def _chunk_stage(
        self,
        job: CrawlJob,
        index: BoilerplateIndex,
        totals: _Totals,
        out: asyncio.Queue[_Batch | object],
    ) -> None:
        batch = _Batch()
        try:
            async for page in self._pages.iter_succeeded(job.id):
                totals.pages += 1
                chunks = await asyncio.to_thread(self._chunk_page, page, index)
                if not chunks:
                    totals.empty_pages += 1
                previous = len(await self._chunks.for_page(page.id))
                await self._chunks.save_page_chunks(page.id, chunks)
                if previous and previous != len(chunks):
                    batch.reset_pages.add(page.id)
                for chunk in chunks:
                    batch.chunks.append(chunk)
                    if len(batch.chunks) >= self.batch_size:
                        await out.put(batch)
                        batch = _Batch()
            if batch.chunks or batch.reset_pages:
                await out.put(batch)
        finally:
            await out.put(_DONE)

    def _chunk_page(self, page: Page, index: BoilerplateIndex) -> list[Chunk]:
        stripped = index.strip(page.id, page.markdown or "").markdown
        try:
            markdown = self._cleaner.clean(stripped).markdown
        except ExtractionError:  # nothing worth indexing is left
            return []
        return self._chunker.chunk_documents(ChunkInput.from_page(page, markdown=markdown))

    # ---- stage 2: embed ---- #

    async def _embed_stage(
        self,
        job: CrawlJob,
        run_id: str,
        totals: _Totals,
        inbox: asyncio.Queue[_Batch | object],
        out: asyncio.Queue[_Batch | object],
    ) -> None:
        try:
            while (item := await inbox.get()) is not _DONE:
                assert isinstance(item, _Batch)
                if item.chunks:
                    texts = [chunk.embed_text for chunk in item.chunks]
                    dense, sparse = await asyncio.gather(
                        self._embedder.embed_documents(texts), self._sparse.embed_documents(texts)
                    )
                    await self._meter.record_ingest(job, run_id, dense)
                    item.dense, item.sparse = dense.vectors, sparse
                    totals.batches += 1
                    totals.cached += dense.cached
                    totals.tokens += dense.usage.tokens
                await out.put(item)
        finally:
            await out.put(_DONE)

    # ---- stage 3: upsert ---- #

    async def _upsert_stage(
        self, spec: CollectionSpec, job: CrawlJob, inbox: asyncio.Queue[_Batch | object]
    ) -> None:
        while (item := await inbox.get()) is not _DONE:
            assert isinstance(item, _Batch)
            for page_id in sorted(item.reset_pages):
                await self._store.delete_by_filter(
                    spec,
                    VectorFilter(job.tenant_id, versions=(job.version,), page_ids=(page_id,)),
                )
            points = [
                VectorPoint.from_chunk(chunk, dense, sparse)
                for chunk, dense, sparse in zip(item.chunks, item.dense, item.sparse, strict=True)
            ]
            await self._store.upsert(spec, points)
            if self._on_batch is not None:
                await self._on_batch()


def _first(group: BaseExceptionGroup) -> BaseException:
    """The first real failure of a stage (its own traceback); cancelled siblings aren't in it."""
    error: BaseException = group.exceptions[0]
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error
