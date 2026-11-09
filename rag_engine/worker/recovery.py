"""Crash recovery (Step 80): jobs whose worker died, and queued jobs whose task was lost.

`JobRecovery.run()` is what the `recover_jobs` maintenance task runs on every Celery beat
tick (`CRAWL_RECOVERY_INTERVAL_SECONDS`). Every step is a guarded transition, so two
sweeps racing each other, or a sweep racing a worker that was only slow, can't both act.

**Stale held jobs**: a `running` or `finalizing` job whose heartbeat is older than
`CRAWL_STALE_HEARTBEAT_SECONDS` lost its worker (killed, OOM, a lost node; or a run whose
own failure handling couldn't reach Mongo, Step 77).
- A **finalizing** job whose source already serves its version flipped before the worker
  died: it is completed (`succeeded`), not re-run.
- Fewer than `CRAWL_JOB_MAX_ATTEMPTS` attempts → `requeue(job, stale_run_id)` (guarded on
  the dead run and on its heartbeat still being stale, so a worker that was only slow and
  just beat is left alone; the dead run is cut off: anything it still tries is superseded)
  and the task is
  sent again. The next run's frontier `recover()` hands the dead run's in-flight pages
  back and keeps finished ones (Step 48); ingestion is idempotent (Step 76), so the job
  resumes rather than restarts. A pending cancel request makes `requeue` cancel the job
  instead; its unfinished pages are then closed out `job_cancelled`.
- Out of attempts → `failed` with `worker_lost`: the crawl report is stored, unfinished
  pages are closed out, content claims released.
- A cancelled or failed job never went live, so its own version is queued for cleanup
  (`cleanup_version` never deletes an active version).

**Lost queued jobs**: a `queued` job untouched for `CRAWL_QUEUED_STALE_SECONDS` (an
enqueue that failed after the insert, a broker that lost its data) is sent to the crawl
queue again. `touch_queued` bumps `updated_at` first, so a job waiting in a long backlog
gets at most one extra message per window; a duplicate delivery is a no-op (the runner
only claims a `queued` job once).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from loguru import logger

from rag_engine.config import Settings, settings
from rag_engine.crawling import build_report
from rag_engine.domain import CrawlJob, JobError, JobStatus, PageReason, utc_now
from rag_engine.infrastructure import MongoDatabase
from rag_engine.repositories import (
    ContentClaimRepository,
    InvalidJobTransitionError,
    JobRepository,
    JobRunSupersededError,
    PageRepository,
    SourceRepository,
)

WORKER_LOST = "worker_lost"
_BATCH = 100

type Enqueue = Callable[[str, float], None]  # (job_id, countdown seconds)
type EnqueueCleanup = Callable[[str, str, str, str, float], None]


@dataclass
class RecoveryReport:
    requeued: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)  # finalizing jobs that had flipped
    resent: list[str] = field(default_factory=list)  # lost queued jobs sent again
    skipped: list[str] = field(default_factory=list)  # changed under us (the worker woke up)

    @property
    def acted(self) -> bool:
        return any((self.requeued, self.failed, self.cancelled, self.completed, self.resent))


class JobRecovery:
    def __init__(
        self,
        db: MongoDatabase,
        *,
        enqueue: Enqueue,
        enqueue_cleanup: EnqueueCleanup,
        config: Settings = settings,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.jobs = JobRepository(db, clock=clock)
        self.pages = PageRepository(db, config)
        self.claims = ContentClaimRepository(db)
        self.sources = SourceRepository(db)
        self._enqueue = enqueue
        self._enqueue_cleanup = enqueue_cleanup
        self._config = config
        self._clock = clock

    async def run(self) -> RecoveryReport:
        report = RecoveryReport()
        now = self._clock()
        stale_before = now - timedelta(seconds=self._config.CRAWL_STALE_HEARTBEAT_SECONDS)
        for job in await self.jobs.find_stale(stale_before, limit=_BATCH):
            try:
                await self._recover_held(job, report, stale_before)
            except InvalidJobTransitionError, JobRunSupersededError:
                report.skipped.append(job.id)  # its worker woke up, or another sweep won
        queued_before = now - timedelta(seconds=self._config.CRAWL_QUEUED_STALE_SECONDS)
        for job in await self.jobs.find_stale_queued(queued_before, limit=_BATCH):
            if await self.jobs.touch_queued(job.id, job.updated_at):
                logger.bind(job_id=job.id).warning(
                    "job {} queued since {} without a worker; sending it again",
                    job.id,
                    job.updated_at,
                )
                self._enqueue(job.id, 0)
                report.resent.append(job.id)
        if report.acted:
            logger.info("crash recovery: {}", report)
        return report

    async def _recover_held(
        self, job: CrawlJob, report: RecoveryReport, stale_before: datetime
    ) -> None:
        run_id = job.run_id
        assert run_id is not None  # a held job always has its run (model invariant)
        log = logger.bind(job_id=job.id, tenant_id=job.tenant_id)
        if job.status is JobStatus.FINALIZING and await self._already_live(job):
            await self.jobs.succeed(job.id, run_id)
            await self.claims.release_job(job.id)
            log.warning("job {} had gone live before its worker died; completed", job.id)
            report.completed.append(job.id)
            return

        if job.attempts < self._config.CRAWL_JOB_MAX_ATTEMPTS:
            requeued = await self.jobs.requeue(
                job.id, stale_run_id=run_id, heartbeat_before=stale_before
            )
            if requeued.status is JobStatus.CANCELLED:
                await self.pages.close_out_job(
                    job.id, "job cancelled", reason=PageReason.JOB_CANCELLED
                )
                await self._release(job)
                log.warning("job {} lost its worker with a cancel pending; cancelled", job.id)
                report.cancelled.append(job.id)
                return
            self._enqueue(job.id, 0)
            log.warning(
                "job {} lost its worker (heartbeat {}); requeued, attempt {} of {} next",
                job.id,
                job.heartbeat_at,
                job.attempts + 1,
                self._config.CRAWL_JOB_MAX_ATTEMPTS,
            )
            report.requeued.append(job.id)
            return

        message = (
            f"the worker stopped responding (last heartbeat {job.heartbeat_at:%Y-%m-%dT%H:%M:%SZ})"
            f" on attempt {job.attempts} of {self._config.CRAWL_JOB_MAX_ATTEMPTS}"
        )
        current = await self.jobs.get_or_raise(job.id)
        if not _still_stale(current, run_id, stale_before):
            report.skipped.append(job.id)  # don't overwrite a live run's report
            return
        try:  # what the dead run crawled, for the per-page report (Step 86)
            await self.jobs.set_report(job.id, run_id, await build_report(self.pages, job.id))
        except JobRunSupersededError:
            raise
        except Exception:
            log.opt(exception=True).warning("could not store the lost job's report")
        await self.jobs.fail(
            job.id,
            JobError(code=WORKER_LOST, message=message),
            run_id=run_id,
            heartbeat_before=stale_before,
        )
        await self.pages.close_out_job(job.id, f"job failed: {WORKER_LOST}")
        await self._release(job)
        log.error("job {} failed: {}", job.id, message)
        report.failed.append(job.id)

    async def _already_live(self, job: CrawlJob) -> bool:
        source = await self.sources.get(job.source_id)
        return source is not None and source.active_version == job.version

    async def _release(self, job: CrawlJob) -> None:
        """A cancelled/failed job's claims and (never served) version."""
        await self.claims.release_job(job.id)
        try:
            self._enqueue_cleanup(job.tenant_id, job.kb_id, job.source_id, job.version, 0)
        except Exception:
            logger.opt(exception=True).error("could not enqueue cleanup of {}", job.version)


def _still_stale(job: CrawlJob, run_id: str, stale_before: datetime) -> bool:
    return (
        job.run_id == run_id
        and job.heartbeat_at is not None
        and job.heartbeat_at < stale_before
        and job.status in (JobStatus.RUNNING, JobStatus.FINALIZING)
    )
