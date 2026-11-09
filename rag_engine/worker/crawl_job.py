"""Running one crawl job end to end (Step 77): claim → crawl → ingest → finalize.

`CrawlJobRunner.run(job_id)` is what the `run_crawl_job` Celery task executes (on the
worker process's loop, Step 75). It is **idempotent under redelivery** (late acks): a job
that is no longer `queued` is skipped, and `JobRepository.start` lets exactly one run
claim it.

1. **Claim**: `jobs.start(job, new run_id)`; load the KB (embedding space) and the source.
   A KB or source being deleted fails the job (`knowledge_base_deleted` /
   `source_deleted`). The source's current `active_version` is remembered for the
   compare-and-swap at the end.
2. **Crawl** (Steps 50-54): the engine wired as in production: the SSRF-safe HTTP client,
   robots.txt cached in Redis, the per-domain rate limiter, sitemaps, the Redis cancel
   flag, and the process's browser pool when JS rendering is on.
3. **Ingest** (Step 76): chunks + dense (cached OpenAI) + sparse BM25 vectors under the
   job's version.
4. **Finalize** (Steps 73, 78): `running → finalizing`, the **atomic compare-and-swap
   flip** of the source's `active_version` (with the new stats), the **previous version's
   cleanup enqueued** (`cleanup_version`, after `CRAWL_VERSION_CLEANUP_DELAY_SECONDS`, so
   searches that started just before the flip finish), content claims released,
   `→ succeeded`.

A failed or cancelled job never flipped, so the old version keeps serving; its **own
version's vectors and chunks are deleted** right away (`VersionCleaner`), falling back to
the cleanup task if the vector store is unavailable. A requeued job keeps its data (the
next run resumes and reuses it).

**Heartbeat**: every `CRAWL_HEARTBEAT_SECONDS` beside the work. It proves the run is
alive (Step 80 recovers stale jobs) and carries cancel requests: during the crawl it stops
the engine (in-flight pages drain, Step 52), during ingestion it cancels the pipeline; the
job then ends `cancelled`. Finalizing is not cancellable. A superseded run (requeued
elsewhere) stops quietly without touching the job.

**Failures**:
- permanent (bad credentials, a refused request, a deleted source, a version conflict,
  Celery's soft time limit `time_limit`, a bug `internal_error`) → `failed` with that code;
- transient (MongoDB, Redis, the vector store or the embedding API unavailable, or rate
  limited past its retries) → **requeued** (a pending cancel makes it `cancelled`) and
  re-enqueued with backoff (`CRAWL_JOB_RETRY_DELAY_SECONDS` x attempts), up to
  `CRAWL_JOB_MAX_ATTEMPTS`; then `failed`.
On failure the crawl report is stored first (`fail` releases the job), and pages still
queued or in progress are closed out (`job_failed`, Step 53).
"""

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger
from redis.asyncio import Redis

from rag_engine.chunking import get_chunker
from rag_engine.config import JsRenderMode, Settings, log_context, settings
from rag_engine.crawling import (
    CrawlEngine,
    CrawlSummary,
    DomainRateLimiter,
    RedisCancelFlags,
    RedisRobotsStore,
    RobotsCache,
    SitemapDiscoverer,
    build_report,
    finish_cancelled,
    render_politeness,
)
from rag_engine.crawling.fetchers import AutoFetcher, BrowserPool, HttpFetcher
from rag_engine.domain import (
    CrawlJob,
    CrawlStopReason,
    JobError,
    JobStatus,
    PageStatus,
    ResourceStatus,
    SourceStats,
    new_id,
    utc_now,
)
from rag_engine.embeddings import (
    CachedEmbedder,
    Embedder,
    EmbeddingError,
    EmbeddingSpace,
    get_embedder,
    get_sparse_embedder,
    tokenizer_for,
)
from rag_engine.exceptions import AppBaseException
from rag_engine.infrastructure import (
    DatabaseUnavailableError,
    MongoDatabase,
    RedisKeyspace,
    RedisUnavailableError,
    VectorStoreUnavailableError,
)
from rag_engine.ingestion import IngestionPipeline, IngestReport, SparseEmbedder
from rag_engine.repositories import (
    ChunkRepository,
    ContentClaimRepository,
    EmbeddingCacheRepository,
    InvalidJobTransitionError,
    JobRepository,
    JobRunSupersededError,
    KnowledgeBaseRepository,
    PageRepository,
    SourceRepository,
    UsageRepository,
)
from rag_engine.runtime import AppRuntime
from rag_engine.security import SafeHttpClient
from rag_engine.services.cleanup import VersionCleaner
from rag_engine.services.usage import UsageMeter
from rag_engine.vectorstores import BaseVectorStore, VectorCollections

_TRANSIENT = (DatabaseUnavailableError, RedisUnavailableError, VectorStoreUnavailableError)


class JobOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUEUED = "requeued"  # a transient failure: queued again with backoff
    SKIPPED = "skipped"  # not queued any more (redelivered task, claimed by another run)
    SUPERSEDED = "superseded"  # another run took the job over mid-flight


# outcomes that end the job, so it owes its completion webhook (Step 83)
_FINISHED = frozenset({JobOutcome.SUCCEEDED, JobOutcome.FAILED, JobOutcome.CANCELLED})


class _JobRejectedError(Exception):
    """The job can't run as specified (its KB or source is gone)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


type Enqueue = Callable[[str, float], None]  # (job_id, countdown seconds)
type EnqueueWebhook = Callable[[str], None]  # (job_id): the completion webhook (Step 83)
# (tenant_id, kb_id, source_id, version, countdown seconds): the `cleanup_version` task
type EnqueueCleanup = Callable[[str, str, str, str, float], None]


@dataclass
class CrawlJobDeps:
    db: MongoDatabase
    redis: Redis
    keys: RedisKeyspace
    store: BaseVectorStore
    collections: VectorCollections
    enqueue: Enqueue
    enqueue_cleanup: EnqueueCleanup
    config: Settings = field(default_factory=lambda: settings)
    embedder_factory: Callable[[EmbeddingSpace], Embedder] | None = None
    sparse: SparseEmbedder | None = None
    browser_pool: BrowserPool | None = None
    http_factory: Callable[[], AbstractAsyncContextManager[SafeHttpClient]] | None = None
    enqueue_webhook: EnqueueWebhook | None = None

    @classmethod
    def from_runtime(
        cls,
        runtime: AppRuntime,
        *,
        enqueue: Enqueue,
        enqueue_cleanup: EnqueueCleanup,
        **overrides: Any,
    ) -> CrawlJobDeps:
        assert runtime.vector_store is not None and runtime.collections is not None
        values: dict[str, Any] = {
            "db": runtime.mongo.db,
            "redis": runtime.redis.client,
            "keys": runtime.redis.keys,
            "store": runtime.vector_store,
            "collections": runtime.collections,
            "config": runtime.config,
        }
        return cls(enqueue=enqueue, enqueue_cleanup=enqueue_cleanup, **{**values, **overrides})


@dataclass
class _Run:
    job: CrawlJob
    run_id: str
    # start → crawl → ingest → finalize, or → finishing once the run is ending the job
    # (cancel/fail/requeue): the heartbeat stops, since the job is about to leave our hands
    phase: str = "start"
    previous_version: str | None = None
    engine: CrawlEngine | None = None
    summary: CrawlSummary | None = None
    cancel_requested: bool = False
    superseded: bool = False
    notes: dict[str, Any] = field(default_factory=dict)


class CrawlJobRunner:
    def __init__(self, deps: CrawlJobDeps) -> None:
        self.deps = deps
        self.config = deps.config
        db = deps.db
        self.jobs = JobRepository(db)
        self.pages = PageRepository(db, self.config)
        self.chunks = ChunkRepository(db)
        self.claims = ContentClaimRepository(db)
        self.kbs = KnowledgeBaseRepository(db)
        self.sources = SourceRepository(db)
        self.usage = UsageRepository(db)
        self.flags = RedisCancelFlags(deps.redis, deps.keys, self.config)
        self.cleaner = VersionCleaner(db, deps.store)

    # ---- entry point ---- #

    async def run(self, job_id: str) -> JobOutcome:
        job = await self.jobs.get(job_id)
        if job is None or job.status is not JobStatus.QUEUED:
            logger.bind(job_id=job_id).info(
                "job {} not queued ({}), skipping", job_id, job.status if job else "missing"
            )
            return JobOutcome.SKIPPED
        run_id = new_id("run")
        try:
            job = await self.jobs.start(job_id, run_id)
        except InvalidJobTransitionError, JobRunSupersededError:
            return JobOutcome.SKIPPED  # another delivery claimed it first
        with log_context(job_id=job.id, tenant_id=job.tenant_id):
            logger.info("job {} started as {} (attempt {})", job.id, run_id, job.attempts)
            outcome = await self._run_claimed(_Run(job, run_id))
            logger.info("job {} finished: {}", job.id, outcome)
            if outcome in _FINISHED and job.config.webhook_url is not None:
                self._enqueue_webhook(job.id)
            return outcome

    async def _run_claimed(self, run: _Run) -> JobOutcome:
        work = asyncio.create_task(self._work(run))
        beat = asyncio.create_task(self._heartbeat(run, work))
        try:
            return await work
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise  # we are being cancelled from outside: not ours to handle
            if run.superseded:
                return JobOutcome.SUPERSEDED
            return await self._cancel(run)  # the heartbeat cancelled the ingestion
        except JobRunSupersededError:
            return JobOutcome.SUPERSEDED
        except Exception as exc:
            return await self._failed(run, exc)
        finally:
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)

    # ---- the work ---- #

    async def _work(self, run: _Run) -> JobOutcome:
        job = run.job
        kb = await self.kbs.get(job.kb_id)
        if kb is None or kb.status is not ResourceStatus.ACTIVE:
            raise _JobRejectedError("knowledge_base_deleted", "the knowledge base was deleted")
        source = await self.sources.get(job.source_id)
        if source is None or source.status is not ResourceStatus.ACTIVE:
            raise _JobRejectedError("source_deleted", "the source was deleted")
        run.previous_version = source.active_version

        run.phase = "crawl"
        run.summary = await self._crawl(run)
        if run.cancel_requested or (
            run.summary.stop is not None and run.summary.stop.reason is CrawlStopReason.CANCELLED
        ):
            return await self._cancel(run)

        run.phase = "ingest"
        report = await self._ingest(run, kb.embedding_space)
        if run.cancel_requested:
            return await self._cancel(run)

        run.phase = "finalize"
        return await self._finalize(run, report)

    async def _crawl(self, run: _Run) -> CrawlSummary:
        deps, config, job = self.deps, self.config, run.job
        http_factory = deps.http_factory or (lambda: SafeHttpClient(config))
        async with http_factory() as client:
            http = HttpFetcher(client, config)
            robots = RobotsCache(http, RedisRobotsStore(deps.redis, deps.keys), config=config)
            limiter = DomainRateLimiter(deps.redis, deps.keys, config)
            render = config.CRAWL_JS_RENDERING is not JsRenderMode.NEVER
            browser_scope = (
                deps.browser_pool.job_context(job_id=job.id)
                if render and deps.browser_pool is not None
                else nullcontext(None)
            )
            async with browser_scope as browser:
                engine = CrawlEngine(
                    job,
                    run_id=run.run_id,
                    pages=self.pages,
                    jobs=self.jobs,
                    claims=self.claims,
                    fetcher=AutoFetcher(
                        http,
                        browser,
                        config=config,
                        before_render=render_politeness(robots, limiter),
                    ),
                    robots=robots,
                    limiter=limiter,
                    sitemaps=SitemapDiscoverer(http, robots=robots, limiter=limiter, config=config),
                    cancel=self.flags,
                    config=config,
                )
                run.engine = engine
                try:
                    return await engine.run()
                finally:
                    run.engine = None

    async def _ingest(self, run: _Run, space: EmbeddingSpace) -> IngestReport:
        deps, config = self.deps, self.config
        embedder = (deps.embedder_factory or self._default_embedder)(space)
        try:
            pipeline = IngestionPipeline(
                pages=self.pages,
                chunks=self.chunks,
                jobs=self.jobs,
                chunker=get_chunker(tokenizer_for(space.model), config=config),
                embedder=embedder,
                sparse=deps.sparse or get_sparse_embedder(),
                store=deps.store,
                collections=deps.collections,
                meter=UsageMeter(self.jobs, self.usage),
                config=config,
            )
            return await pipeline.run(run.job, run.run_id)
        finally:
            await embedder.aclose()

    def _default_embedder(self, space: EmbeddingSpace) -> Embedder:
        embedder = get_embedder(space, config=self.config)
        if self.config.EMBEDDING_CACHE_ENABLED:
            return CachedEmbedder(embedder, EmbeddingCacheRepository(self.deps.db))
        return embedder

    async def _finalize(self, run: _Run, report: IngestReport) -> JobOutcome:
        job = run.job
        await self.jobs.begin_finalizing(job.id, run.run_id)
        pages = await self.pages.count_by_status(job.id)
        stats = SourceStats(
            pages=pages.get(PageStatus.SUCCEEDED, 0),
            chunks=report.chunks,
            last_succeeded_at=utc_now(),
        )
        previous = await self.sources.activate_version(
            job.source_id,
            job.version,
            stats=stats,
            check_previous=True,
            expected_previous=run.previous_version,
        )
        if previous is not None and previous != job.version:
            self._enqueue_cleanup(job, previous, self.config.CRAWL_VERSION_CLEANUP_DELAY_SECONDS)
        await self.claims.release_job(job.id)
        await self.jobs.succeed(job.id, run.run_id)
        return JobOutcome.SUCCEEDED

    def _enqueue_webhook(self, job_id: str) -> None:
        """The job's terminal transition marked its webhook pending; send it now (the
        recovery sweep re-sends it if this message is lost)."""
        if self.deps.enqueue_webhook is None:
            return
        try:
            self.deps.enqueue_webhook(job_id)
        except Exception:
            logger.opt(exception=True).error("could not enqueue the webhook of job {}", job_id)

    def _enqueue_cleanup(self, job: CrawlJob, version: str, countdown: float) -> None:
        try:
            self.deps.enqueue_cleanup(job.tenant_id, job.kb_id, job.source_id, version, countdown)
        except Exception:  # never fail a job that already went live over its cleanup
            logger.opt(exception=True).error("could not enqueue cleanup of version {}", version)

    async def _discard_own_version(self, run: _Run) -> None:
        """A failed/cancelled job's own vectors and chunks: never served, delete them now."""
        job = run.job
        try:
            await self.cleaner.delete_version(
                tenant_id=job.tenant_id,
                kb_id=job.kb_id,
                source_id=job.source_id,
                version=job.version,
            )
        except _TRANSIENT as exc:
            logger.warning("could not delete version {} now ({}); queued", job.version, exc.code)
            self._enqueue_cleanup(job, job.version, self.config.CRAWL_JOB_RETRY_DELAY_SECONDS)
        except Exception:
            logger.opt(exception=True).error("could not delete version {}", job.version)

    # ---- heartbeat and cancellation ---- #

    async def _heartbeat(self, run: _Run, work: asyncio.Task[JobOutcome]) -> None:
        while True:
            await asyncio.sleep(self.config.CRAWL_HEARTBEAT_SECONDS)
            if run.phase == "finishing":
                return  # the job is (being) released: "not held" is expected, not a takeover
            try:
                job = await self.jobs.heartbeat(run.job.id, run.run_id)
            except JobRunSupersededError:
                logger.warning("job {} was taken over by another run", run.job.id)
                run.superseded = True
                work.cancel()
                return
            except DatabaseUnavailableError:
                logger.warning("heartbeat of job {} failed; retrying", run.job.id)
                continue
            if job.cancel_requested_at is None or run.cancel_requested:
                continue
            if run.phase == "crawl":
                run.cancel_requested = True
                if run.engine is not None:
                    run.engine.cancel("the job was cancelled")
            elif run.phase == "ingest":
                run.cancel_requested = True
                work.cancel()
            # finalizing (the version swap) is not cancellable

    async def _cancel(self, run: _Run) -> JobOutcome:
        run.phase = "finishing"
        await finish_cancelled(self.jobs, self.flags, run.job.id, run.run_id)
        await self.claims.release_job(run.job.id)
        await self._discard_own_version(run)
        return JobOutcome.CANCELLED

    # ---- failures ---- #

    async def _failed(self, run: _Run, exc: Exception) -> JobOutcome:
        run.phase = "finishing"
        code, message, transient = _classify(exc)
        job = run.job
        if transient and job.attempts < self.config.CRAWL_JOB_MAX_ATTEMPTS:
            requeued = await self.jobs.requeue(job.id, stale_run_id=run.run_id)
            if requeued.status is JobStatus.CANCELLED:
                return JobOutcome.CANCELLED  # a cancel was pending: requeue cancelled it
            delay = self.config.CRAWL_JOB_RETRY_DELAY_SECONDS * job.attempts
            logger.warning("job {} hit {} ({}); requeued in {}s", job.id, code, message, delay)
            self.deps.enqueue(job.id, delay)
            return JobOutcome.REQUEUED
        logger.error("job {} failed: {} ({})", job.id, code, message)
        if run.summary is None:  # the crawl didn't finish: store what we know first
            try:
                report = await build_report(self.pages, job.id)
                await self.jobs.set_report(job.id, run.run_id, report)
            except Exception:
                logger.opt(exception=True).warning("could not store the failed job's report")
        await self.jobs.fail(job.id, JobError(code=code, message=message[:2000]), run_id=run.run_id)
        await self.pages.close_out_job(job.id, f"job failed: {code}")
        await self.claims.release_job(job.id)
        await self._discard_own_version(run)
        return JobOutcome.FAILED


def _classify(exc: Exception) -> tuple[str, str, bool]:
    """(error code, message, transient?) for a failed run."""
    if isinstance(exc, _JobRejectedError):
        return exc.code, str(exc), False
    if isinstance(exc, SoftTimeLimitExceeded):
        return "time_limit", "the job exceeded the worker's time limit", False
    if isinstance(exc, EmbeddingError):
        return exc.code, exc.message, exc.retryable
    if isinstance(exc, _TRANSIENT):
        return exc.code, exc.message, True
    if isinstance(exc, AppBaseException):
        return exc.code, exc.message, False
    logger.opt(exception=exc).error("crawl job crashed")
    return "internal_error", f"{type(exc).__name__}: {exc}", False
