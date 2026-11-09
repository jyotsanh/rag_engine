"""Cascade deletes (Step 81): a deleted source's or knowledge base's data, then its record.

The API only marks a source or KB `deleting` (hidden at once, its seed URL or name freed)
and enqueues the cascade, which runs on the `maintenance` queue:

**Source** (`delete_source`):
1. **Its crawl jobs stop first.** An active job is cancelled (`request_cancel`: a queued
   one at once, a running one is flagged for its worker; a finalizing one can't be and is
   simply waited for). While any is active the cascade returns `waiting_for_jobs` and the
   task re-checks later: deleting vectors under a job that still upserts would leave
   orphans behind. A job can't start for a deleting source (the runner refuses it at
   claim), and a crawl submission never targets one (Step 84 creates a new source).
2. Its **vectors** (every version, from its KB's collection), **chunks**, then for each of
   its jobs the **pages** and content claims, then its **job records**, and finally the
   **source record**. Records go last, so a cascade that crashed mid-way is found again
   by the sweep and simply re-run: every step is idempotent.

**Knowledge base** (`delete_knowledge_base`): its live sources are marked deleting too,
each one is cascaded as above (all its jobs must stop first), the KB's remaining vectors
are deleted with one KB-wide filter (a safety net), then the KB record. The collection is
shared by every KB with the same embedding space, so it stays.

Kept on purpose: tenant usage (billing history, Step 65) and the embedding cache (shared
by content hash, pruned by Step 97).

Every pass `touch`es the record it works on, so the deletion sweep (`DeletionSweeper`)
only restarts cascades nobody touched for `CASCADE_STALE_SECONDS`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from loguru import logger

from rag_engine.config import Settings, settings
from rag_engine.domain import CrawlJob, JobStatus, KnowledgeBase, ResourceStatus, utc_now
from rag_engine.infrastructure import MongoDatabase
from rag_engine.repositories import (
    ChunkRepository,
    ContentClaimRepository,
    InvalidJobTransitionError,
    JobRepository,
    KnowledgeBaseRepository,
    PageRepository,
    SourceRepository,
)
from rag_engine.vectorstores import (
    BaseVectorStore,
    CollectionNotFoundError,
    CollectionSpec,
    VectorFilter,
)


class CascadeStatus(StrEnum):
    DELETED = "deleted"  # data and record gone
    ALREADY_GONE = "already_gone"  # nothing left (a repeated or racing cascade)
    WAITING_FOR_JOBS = "waiting_for_jobs"  # active crawl jobs are being cancelled: retry
    NOT_DELETING = "not_deleting"  # the record is live: never cascaded


@dataclass
class CascadeResult:
    status: CascadeStatus
    points: int = 0
    chunks: int = 0
    pages: int = 0
    jobs: int = 0
    sources: int = 0
    waiting_for: list[str] = field(default_factory=list)  # active job ids

    def add(self, other: CascadeResult) -> None:
        self.points += other.points
        self.chunks += other.chunks
        self.pages += other.pages
        self.jobs += other.jobs
        self.sources += other.sources
        self.waiting_for += other.waiting_for


class CascadeDeleter:
    def __init__(self, db: MongoDatabase, store: BaseVectorStore) -> None:
        self.store = store
        self.kbs = KnowledgeBaseRepository(db)
        self.sources = SourceRepository(db)
        self.jobs = JobRepository(db)
        self.pages = PageRepository(db)
        self.chunks = ChunkRepository(db)
        self.claims = ContentClaimRepository(db)

    # ---- source ---- #

    async def delete_source(self, tenant_id: str, source_id: str) -> CascadeResult:
        source = await self.sources.get(source_id, scope={"tenant_id": tenant_id})
        if source is None:
            return CascadeResult(CascadeStatus.ALREADY_GONE)
        if source.status is not ResourceStatus.DELETING:
            logger.error("refusing to cascade live source {}", source_id)
            return CascadeResult(CascadeStatus.NOT_DELETING)
        await self.sources.touch_deleting(source_id)
        log = logger.bind(source_id=source_id, kb_id=source.kb_id, tenant_id=tenant_id)

        job = await self.jobs.find_active_for_source(tenant_id, source_id)
        active = await self._stop(tenant_id, [job] if job is not None else [])
        if active:
            log.info("source {} waits for {} crawl job(s) to stop", source_id, len(active))
            return CascadeResult(CascadeStatus.WAITING_FOR_JOBS, waiting_for=active)

        kb = await self.kbs.get(source.kb_id)
        result = CascadeResult(CascadeStatus.DELETED, sources=1)
        result.points = await self._delete_points(
            kb, VectorFilter(tenant_id, source_ids=(source_id,))
        )
        result.chunks = await self.chunks.delete_source(source_id)
        job_ids = await self.jobs.ids_for_source(source_id)
        for job_id in job_ids:
            result.pages += await self.pages.delete_job(job_id)
            await self.claims.release_job(job_id)
        result.jobs = await self.jobs.delete_for_source(source_id)
        await self.sources.delete(source_id, scope={"tenant_id": tenant_id})
        log.info("deleted source {}: {}", source_id, result)
        return result

    # ---- knowledge base ---- #

    async def delete_knowledge_base(self, tenant_id: str, kb_id: str) -> CascadeResult:
        kb = await self.kbs.get(kb_id, scope={"tenant_id": tenant_id})
        if kb is None:
            return CascadeResult(CascadeStatus.ALREADY_GONE)
        if kb.status is not ResourceStatus.DELETING:
            logger.error("refusing to cascade live knowledge base {}", kb_id)
            return CascadeResult(CascadeStatus.NOT_DELETING)
        await self.kbs.touch_deleting(kb_id)
        log = logger.bind(kb_id=kb_id, tenant_id=tenant_id)
        await self.sources.mark_kb_sources_deleting(kb_id)

        # every job of the KB stops before any of its data goes
        active = await self._stop(tenant_id, await self.jobs.active_for_kb(tenant_id, kb_id))
        if active:
            log.info("knowledge base {} waits for {} crawl job(s)", kb_id, len(active))
            return CascadeResult(CascadeStatus.WAITING_FOR_JOBS, waiting_for=active)

        result = CascadeResult(CascadeStatus.DELETED)
        for source_id in await self.sources.ids_for_kb(kb_id):
            part = await self.delete_source(tenant_id, source_id)
            if part.status is CascadeStatus.WAITING_FOR_JOBS:  # a job slipped in meanwhile
                return CascadeResult(CascadeStatus.WAITING_FOR_JOBS, waiting_for=part.waiting_for)
            result.add(part)
        # a safety net for vectors whose source record is already gone
        result.points += await self._delete_points(kb, VectorFilter(tenant_id, kb_ids=(kb_id,)))
        await self.kbs.remove(kb_id, scope={"tenant_id": tenant_id})
        log.info("deleted knowledge base {}: {}", kb_id, result)
        return result

    # ---- helpers ---- #

    async def _stop(self, tenant_id: str, jobs: list[CrawlJob]) -> list[str]:
        """Cancel the given active jobs; returns those still active afterwards."""
        still_active: list[str] = []
        for job in jobs:
            try:
                current = await self.jobs.request_cancel(tenant_id, job.id)
            except InvalidJobTransitionError:
                current = await self.jobs.get_or_raise(job.id)  # finalizing, or just finished
            if not current.status.is_terminal:
                still_active.append(job.id)
            elif current.status is JobStatus.CANCELLED:
                logger.bind(job_id=job.id).info("cancelled crawl job {} (deletion)", job.id)
        return still_active

    async def _delete_points(self, kb: KnowledgeBase | None, where: VectorFilter) -> int:
        if kb is None:
            return 0  # its record is gone: the KB cascade already deleted its vectors
        spec = CollectionSpec.for_space(kb.embedding_space)
        try:
            points = await self.store.count(spec, where)
            if points:
                await self.store.delete_by_filter(spec, where)
        except CollectionNotFoundError:
            return 0  # nothing was ever ingested into this space
        return points


# ---- the sweep ---- #

type EnqueueSourceDeletion = Callable[[str, str], None]  # (tenant_id, source_id)
type EnqueueKbDeletion = Callable[[str, str], None]  # (tenant_id, kb_id)


@dataclass
class SweepReport:
    sources: list[str] = field(default_factory=list)
    knowledge_bases: list[str] = field(default_factory=list)


class DeletionSweeper:
    """Restarts cascades nobody touched for `CASCADE_STALE_SECONDS` (Celery beat): the
    enqueue after marking failed, the task was lost, or its worker died."""

    def __init__(
        self,
        db: MongoDatabase,
        *,
        enqueue_source: EnqueueSourceDeletion,
        enqueue_kb: EnqueueKbDeletion,
        config: Settings = settings,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.kbs = KnowledgeBaseRepository(db)
        self.sources = SourceRepository(db)
        self._enqueue_source = enqueue_source
        self._enqueue_kb = enqueue_kb
        self._config = config
        self._clock = clock

    async def run(self) -> SweepReport:
        report = SweepReport()
        now = self._clock()
        before = now - timedelta(seconds=self._config.CASCADE_STALE_SECONDS)
        for kb in await self.kbs.find_deleting_since(before):
            if await self.kbs.touch_deleting(kb.id, kb.updated_at, now=now):
                self._enqueue_kb(kb.tenant_id, kb.id)
                report.knowledge_bases.append(kb.id)
        for source in await self.sources.find_deleting_since(before):
            if await self.sources.touch_deleting(source.id, source.updated_at, now=now):
                self._enqueue_source(source.tenant_id, source.id)
                report.sources.append(source.id)
        if report.sources or report.knowledge_bases:
            logger.warning("restarted stalled cascades: {}", report)
        return report
