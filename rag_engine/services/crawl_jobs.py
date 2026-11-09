"""Crawl job submission under the concurrency guards (Step 79).

1. **One active job per source.** A submission for a source that already has a queued,
   running or finalizing job returns *that* job (`created=False`) instead of starting a
   second crawl. The database enforces it (`one_active_job_per_source`, a partial unique
   index), so racing submissions or API replicas can't both insert.
2. **Tenant max concurrent jobs.** A tenant's active jobs (queued + running + finalizing)
   can't exceed `TenantQuotas.max_concurrent_jobs`; a new job past it is refused with
   `QuotaExceededError` (403 `quota_exceeded`). Queued jobs count, or a tenant could pile up
   an unbounded backlog that drains past its limit. Each active job holds one of the
   tenant's slots (`tenant_job_slot`, a partial unique index), so a burst of submissions
   fills exactly the free slots: never more, and none refused while one is free.
3. The duplicate check comes **before** the quota, so re-submitting a source that is
   already crawling never fails on the quota it is itself using.

Enqueueing the task is the caller's job (Step 84), and only when `created` is true.
"""

from dataclasses import dataclass

from rag_engine.config import logger
from rag_engine.domain import CrawlConfig, CrawlJob, Source, Tenant
from rag_engine.exceptions import QuotaExceededError
from rag_engine.repositories import JobRepository, JobSlotsExhaustedError
from rag_engine.repositories.sources import SourceRepository


@dataclass(frozen=True, slots=True)
class JobSubmission:
    job: CrawlJob
    created: bool  # False: the source's already-active job was returned


class CrawlJobService:
    def __init__(self, jobs: JobRepository, sources: SourceRepository) -> None:
        self._jobs = jobs
        self._sources = sources

    async def submit(self, tenant: Tenant, source: Source, config: CrawlConfig) -> JobSubmission:
        """Queue a crawl of `source`, or return the job already crawling it.

        `config` is already validated and clamped to the tenant's quotas (Step 84). The
        returned job of a duplicate submission keeps its own config.
        """
        if source.tenant_id != tenant.id:
            raise ValueError("the source belongs to another tenant")
        existing = await self._jobs.find_active_for_source(tenant.id, source.id)
        if existing is not None:
            return self._duplicate(existing)

        limit = tenant.quotas.max_concurrent_jobs
        job = CrawlJob(
            tenant_id=tenant.id,
            kb_id=source.kb_id,
            source_id=source.id,
            seed_url=source.seed_url,
            config=config,
        )
        try:
            job, created = await self._jobs.create_active(job, slots=limit)
        except JobSlotsExhaustedError as exc:
            raise QuotaExceededError(
                "Concurrent crawl job limit reached; wait for a job to finish or cancel one",
                details={
                    "quota": "max_concurrent_jobs",
                    "limit": limit,
                    "active": await self._jobs.count_active_for_tenant(tenant.id),
                },
            ) from exc
        if not created:
            return self._duplicate(job)

        await self._sources.set_last_job(source.id, job.id)
        logger.bind(job_id=job.id, source_id=source.id, kb_id=source.kb_id).info(
            "queued crawl job for {} (slot {})", source.seed_url, job.slot
        )
        return JobSubmission(job=job, created=True)

    @staticmethod
    def _duplicate(job: CrawlJob) -> JobSubmission:
        logger.bind(job_id=job.id, source_id=job.source_id).info(
            "source already has an active crawl job ({}); returning it", job.status
        )
        return JobSubmission(job=job, created=False)
