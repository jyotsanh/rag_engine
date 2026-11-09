"""Crawl jobs: the job state machine, live counters and the worker heartbeat.

**State machine** (`JOB_TRANSITIONS`): `queued → running → finalizing → succeeded`, any
active status → `failed`, `queued | running → cancelled`, and `running | finalizing →
queued` when the worker died (Step 80). Every transition is one conditional
`find_one_and_update` on the current status, so two racing transitions can't both win:
of ten workers starting the same job exactly one gets it. A refused transition raises
`InvalidJobTransitionError` naming the current status.

**Run ownership**: `start()` hands the job to a worker run (`run_id`, also the frontier's,
Step 48). Everything that run does afterwards (heartbeat, counters, finalize, finish) is
guarded on that `run_id`; once the job was requeued to another run, finished or
cancelled, the old run gets `JobRunSupersededError` and must stop. That keeps a worker
that was presumed dead from finalizing over the run that replaced it.

**Counters** use atomic `$inc` (no read-modify-write), so concurrent crawler tasks and the
ingest pipeline can bump them freely. `set_counters()` overwrites them, for reconciling
with the page table after a crash (an increment can be lost between writing a page and
counting it).

**One active job per source** (Step 79): a partial unique index on `source_id` over the
active statuses (`one_active_job_per_source`). `create_active()` inserts a job or, when the
source already has an active one, returns that job instead, whoever raced to insert first.
**Tenant concurrency** works the same way: each active job holds one of the tenant's
`max_concurrent_jobs` slots (`tenant_job_slot`, unique over active jobs), so the database,
not a count-then-insert, caps it. A finished job drops out of both indexes, freeing them.
"""

import random
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, ClassVar

from pymongo import ASCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from rag_engine.domain import (
    ACTIVE_JOB_FILTER,
    JOB_TRANSITIONS,
    CrawlJob,
    CrawlReport,
    JobCounters,
    JobError,
    JobStatus,
    MongoDocument,
    utc_now,
)
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.infrastructure.mongo import MongoDatabase
from rag_engine.repositories.base import BaseRepository
from rag_engine.repositories.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidJobTransitionError,
    JobNotFoundError,
    JobRunSupersededError,
    JobSlotsExhaustedError,
)
from rag_engine.repositories.pagination import DEFAULT_PAGE_SIZE, Page

_HELD = (JobStatus.RUNNING, JobStatus.FINALIZING)  # owned by a worker run
_ACTIVE_PER_SOURCE_KEY = {"source_id": 1}  # key pattern of `one_active_job_per_source`
_SLOT_KEY = {"tenant_id": 1, "slot": 1}  # key pattern of `tenant_job_slot`
# a DuplicateKeyError on the source names an active job; it can finish before we read it, so
# the insert is retried a few times before giving up
_CREATE_ATTEMPTS = 5


def _sources_of(target: JobStatus) -> list[str]:
    """Statuses allowed to move to `target`."""
    return sorted(s.value for s, targets in JOB_TRANSITIONS.items() if target in targets)


class JobRepository(BaseRepository[CrawlJob]):
    model: ClassVar[type[MongoDocument]] = CrawlJob
    not_found_error: ClassVar[type[DocumentNotFoundError]] = JobNotFoundError

    def __init__(self, db: MongoDatabase, *, clock: Callable[[], datetime] = utc_now) -> None:
        super().__init__(db)
        self._clock = clock

    # ---- reads (tenant-scoped for the API) ---- #

    async def get_for_tenant(self, tenant_id: str, job_id: str) -> CrawlJob:
        return await self.get_or_raise(job_id, scope={"tenant_id": tenant_id})

    async def list_for_tenant(
        self,
        tenant_id: str,
        *,
        status: JobStatus | None = None,
        source_id: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> Page[CrawlJob]:
        """Newest first, optionally one source's or one status's jobs."""
        filter_: dict[str, Any] = {}
        if status is not None:
            filter_["status"] = status.value
        if source_id is not None:
            filter_["source_id"] = source_id
        return await self.paginate(
            filter_, scope={"tenant_id": tenant_id}, limit=limit, cursor=cursor
        )

    async def find_stale(self, heartbeat_before: datetime, *, limit: int = 100) -> list[CrawlJob]:
        """Held jobs whose worker hasn't beaten since `heartbeat_before` (Step 80)."""
        return await self.find(
            {
                "status": {"$in": [s.value for s in _HELD]},
                "heartbeat_at": {"$lt": heartbeat_before},
            },
            sort=[("heartbeat_at", ASCENDING)],
            limit=limit,
        )

    # ---- active jobs (Step 79) ---- #

    async def find_active_for_source(self, tenant_id: str, source_id: str) -> CrawlJob | None:
        """The source's queued, running or finalizing job, if any (at most one exists)."""
        return await self.find_one(
            {"source_id": source_id, **ACTIVE_JOB_FILTER}, scope={"tenant_id": tenant_id}
        )

    async def count_active_for_tenant(self, tenant_id: str) -> int:
        """The tenant's concurrent jobs: queued, running or finalizing."""
        try:
            return await self.count(ACTIVE_JOB_FILTER, scope={"tenant_id": tenant_id})
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def create_active(
        self, job: CrawlJob, *, slots: int | None = None
    ) -> tuple[CrawlJob, bool]:
        """Insert a new queued job unless its source already has an active one.

        Returns `(job, True)` when `job` was inserted, `(existing, False)` when another
        active job of the same source (same tenant) won: the partial unique index
        `one_active_job_per_source` decides, so of N racing submissions exactly one inserts.

        With `slots` (the tenant's `max_concurrent_jobs`), the job also claims a free
        concurrency slot `0 ≤ slot < slots`; the `tenant_job_slot` partial unique index lets
        only one active job hold each, so racing submissions can never exceed the limit.
        `JobSlotsExhaustedError` when none is free (and the source has no active job).
        """
        if job.status is not JobStatus.QUEUED:
            raise ValueError("a new job must be queued")
        if slots is not None and slots < 1:
            raise ValueError("slots must be >= 1")
        # every lost slot race means another job took that slot, so this always ends
        attempts = _CREATE_ATTEMPTS + (slots or 0)
        for _ in range(attempts):
            if slots is not None:
                slot = await self._free_slot(job.tenant_id, slots)
                if slot is None:
                    existing = await self.find_active_for_source(job.tenant_id, job.source_id)
                    if existing is not None:
                        return existing, False
                    raise JobSlotsExhaustedError(details={"tenant_id": job.tenant_id})
                job.slot = slot
            try:
                await self.collection.insert_one(job.to_mongo())
                return job, True
            except DuplicateKeyError as exc:
                key = (exc.details or {}).get("keyPattern")
                if key == _SLOT_KEY:
                    continue  # another submission took that slot first: pick again
                if key != _ACTIVE_PER_SOURCE_KEY:
                    raise self._duplicate(exc) from exc
            except PyMongoError as exc:
                raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
            existing = await self.find_active_for_source(job.tenant_id, job.source_id)
            if existing is not None:
                return existing, False
            # the active job finished between our insert and the read: the slot is free now
        raise DuplicateDocumentError(
            details={"collection": self.model.collection_name, "fields": ["source_id"]}
        )

    async def _free_slot(self, tenant_id: str, slots: int) -> int | None:
        """A random slot below `slots` that none of the tenant's active jobs holds.

        Also `None` once the tenant has `slots` active jobs in any slot, so lowering a
        tenant's quota takes effect even while old jobs hold slots above the new limit.
        """
        try:
            held = await self.collection.distinct(
                "slot", {"tenant_id": tenant_id, **ACTIVE_JOB_FILTER}
            )
            if await self.count_active_for_tenant(tenant_id) >= slots:
                return None
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        free = sorted(set(range(slots)) - set(held))
        # random, not lowest-first, so racing submissions rarely contend for the same slot
        return random.choice(free) if free else None

    # ---- transitions ---- #

    async def start(self, job_id: str, run_id: str) -> CrawlJob:
        """`queued → running` for this worker run; counts the attempt.

        `started_at` keeps the first start, so a requeued job reports its real duration.
        """
        if not run_id or len(run_id) > 100:
            raise ValueError("run_id must be 1-100 characters")
        now = self._clock()
        update = [
            {
                "$set": {
                    "status": JobStatus.RUNNING.value,
                    "run_id": {"$literal": run_id},  # never read as a field path
                    "attempts": {"$add": ["$attempts", 1]},
                    "started_at": {"$ifNull": ["$started_at", now]},
                    "heartbeat_at": now,
                    "updated_at": now,
                }
            }
        ]
        return await self._transition(job_id, JobStatus.RUNNING, update)

    async def begin_finalizing(self, job_id: str, run_id: str) -> CrawlJob:
        """`running → finalizing`: the crawl is done, the version swap starts (Step 78)."""
        return await self._transition(
            job_id, JobStatus.FINALIZING, self._set(JobStatus.FINALIZING), run_id=run_id
        )

    async def succeed(self, job_id: str, run_id: str) -> CrawlJob:
        """`finalizing → succeeded`."""
        return await self._finish(job_id, JobStatus.SUCCEEDED, run_id=run_id)

    async def fail(self, job_id: str, error: JobError, *, run_id: str | None = None) -> CrawlJob:
        """Any active status → `failed`.

        The worker passes its `run_id`. System code (a job that never started, crash
        recovery giving up, Step 80) passes none and fails the job whoever holds it.
        """
        return await self._finish(
            job_id, JobStatus.FAILED, run_id=run_id, extra={"error": error.model_dump()}
        )

    async def cancel(self, job_id: str, run_id: str) -> CrawlJob:
        """`running → cancelled`, by the worker once in-flight pages drained (Step 52)."""
        return await self._finish(job_id, JobStatus.CANCELLED, run_id=run_id)

    async def requeue(self, job_id: str, *, stale_run_id: str) -> CrawlJob:
        """`running | finalizing → queued` after `stale_run_id`'s worker died (Step 80).

        Guarded on the run, so a job a new run already picked up isn't requeued twice.
        The old run is cut off: everything it tries afterwards is superseded. A running
        job with a pending cancel request is **cancelled** instead (its worker is gone, so
        there is nothing left to drain); check the returned status.
        """
        now = self._clock()
        try:
            raw = await self.collection.find_one_and_update(
                {
                    "_id": job_id,
                    "status": JobStatus.RUNNING.value,
                    "run_id": stale_run_id,
                    "cancel_requested_at": {"$ne": None},
                },
                self._set(JobStatus.CANCELLED, now=now, finished_at=now, heartbeat_at=None),
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        if raw is not None:
            return self._load(raw)
        update = self._set(JobStatus.QUEUED, now=now, run_id=None, heartbeat_at=None)
        return await self._transition(job_id, JobStatus.QUEUED, update, run_id=stale_run_id)

    async def request_cancel(self, tenant_id: str, job_id: str) -> CrawlJob:
        """The API's cancel: a queued job is cancelled at once, a running one is flagged.

        A running job keeps running until its worker sees `cancel_requested_at` (via the
        heartbeat, or Step 52's Redis flag), drains and calls `cancel()`. Cancelling an
        already-cancelled job is a no-op. Finalizing and finished jobs can't be cancelled.
        """
        scope = {"tenant_id": tenant_id}
        now = self._clock()
        try:
            queued = await self.collection.find_one_and_update(
                self._where({"_id": job_id, "status": JobStatus.QUEUED.value}, scope),
                {
                    "$set": {
                        "status": JobStatus.CANCELLED.value,
                        "cancel_requested_at": now,
                        "finished_at": now,
                        "updated_at": now,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if queued is not None:
                return self._load(queued)
            running = await self.collection.find_one_and_update(
                self._where({"_id": job_id, "status": JobStatus.RUNNING.value}, scope),
                [
                    {
                        "$set": {
                            "cancel_requested_at": {"$ifNull": ["$cancel_requested_at", now]},
                            "updated_at": now,
                        }
                    }
                ],
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        if running is not None:
            return self._load(running)
        current = await self.get_or_raise(job_id, scope=scope)
        if current.status is JobStatus.CANCELLED:
            return current
        raise InvalidJobTransitionError(
            details={"id": job_id, "from": current.status.value, "to": "cancelled"}
        )

    async def _finish(
        self,
        job_id: str,
        target: JobStatus,
        *,
        run_id: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> CrawlJob:
        now = self._clock()
        update = self._set(target, now=now, finished_at=now, heartbeat_at=None, **(extra or {}))
        return await self._transition(job_id, target, update, run_id=run_id)

    def _set(
        self, target: JobStatus, *, now: datetime | None = None, **fields: Any
    ) -> dict[str, Any]:
        stamp = now if now is not None else self._clock()
        return {"$set": {"status": target.value, "updated_at": stamp, **fields}}

    async def _transition(
        self,
        job_id: str,
        target: JobStatus,
        update: dict[str, Any] | list[dict[str, Any]],
        *,
        run_id: str | None = None,
    ) -> CrawlJob:
        filter_: dict[str, Any] = {"_id": job_id, "status": {"$in": _sources_of(target)}}
        if run_id is not None:
            filter_["run_id"] = run_id
        try:
            raw = await self.collection.find_one_and_update(
                filter_, update, return_document=ReturnDocument.AFTER
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        if raw is not None:
            return self._load(raw)
        current = await self.get_or_raise(job_id)
        if run_id is not None and (current.status not in _HELD or current.run_id != run_id):
            # requeued to another run, or finished/cancelled under this run's feet
            raise JobRunSupersededError(
                details={"id": job_id, "run_id": run_id, "status": current.status.value}
            )
        raise InvalidJobTransitionError(
            details={"id": job_id, "from": current.status.value, "to": target.value}
        )

    # ---- while running ---- #

    async def heartbeat(self, job_id: str, run_id: str) -> CrawlJob:
        """Prove the run is alive; returns the job so the worker sees `cancel_requested_at`."""
        now = self._clock()
        raw = await self._held_update(
            job_id, run_id, {"$set": {"heartbeat_at": now, "updated_at": now}}
        )
        return self._load(raw)

    async def increment(self, job_id: str, run_id: str, **deltas: int) -> JobCounters:
        """Atomically add to counters, e.g. `increment(job, run, fetched=1, failed=0)`."""
        unknown = set(deltas) - JobCounters.names()
        if unknown:
            raise ValueError(f"unknown job counters: {sorted(unknown)}")
        if any(delta < 0 for delta in deltas.values()):
            raise ValueError("counter increments must be >= 0")
        inc = {f"counters.{name}": delta for name, delta in deltas.items() if delta}
        if not inc:
            return (await self.get_or_raise(job_id)).counters
        raw = await self._held_update(
            job_id, run_id, {"$inc": inc, "$set": {"updated_at": self._clock()}}
        )
        return self._load(raw).counters

    async def add_usage(
        self, job_id: str, run_id: str, *, requests: int, tokens: int, cached: int = 0
    ) -> None:
        """Embedding usage of this run (Step 65): `$inc` on the job's `usage`."""
        if requests < 0 or tokens < 0 or cached < 0:
            raise ValueError("usage increments must be >= 0")
        await self._held_update(
            job_id,
            run_id,
            {
                "$inc": {
                    "usage.requests": requests,
                    "usage.tokens": tokens,
                    "usage.cached": cached,
                },
                "$set": {"updated_at": self._clock()},
            },
        )

    async def set_counters(self, job_id: str, run_id: str, counters: JobCounters) -> CrawlJob:
        """Overwrite every counter (reconciling with the page table after a resume)."""
        raw = await self._held_update(
            job_id,
            run_id,
            {"$set": {"counters": counters.model_dump(), "updated_at": self._clock()}},
        )
        return self._load(raw)

    async def reconcile_counters(self, job_id: str, run_id: str, **values: int) -> JobCounters:
        """Overwrite only the named counters with values recounted from the page table.

        Unlike `set_counters`, counters another component is still `$inc`-ing (chunked,
        embedded) are left alone. Returns the counters as stored afterwards.
        """
        unknown = set(values) - JobCounters.names()
        if unknown:
            raise ValueError(f"unknown job counters: {sorted(unknown)}")
        if any(value < 0 for value in values.values()):
            raise ValueError("counter values must be >= 0")
        if not values:
            return (await self.get_or_raise(job_id)).counters
        fields = {f"counters.{name}": value for name, value in values.items()}
        raw = await self._held_update(
            job_id, run_id, {"$set": {**fields, "updated_at": self._clock()}}
        )
        return self._load(raw).counters

    async def set_report(self, job_id: str, run_id: str, report: CrawlReport) -> CrawlJob:
        """Store the run's crawl report (replaces an earlier run's)."""
        raw = await self._held_update(
            job_id,
            run_id,
            {"$set": {"report": report.model_dump(), "updated_at": self._clock()}},
        )
        return self._load(raw)

    async def _held_update(
        self, job_id: str, run_id: str, update: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply `update` only while this run holds the job; superseded otherwise."""
        try:
            raw = await self.collection.find_one_and_update(
                {"_id": job_id, "run_id": run_id, "status": {"$in": [s.value for s in _HELD]}},
                update,
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        if raw is not None:
            return raw
        if await self.get(job_id) is None:
            raise JobNotFoundError(details={"id": job_id})
        raise JobRunSupersededError(details={"id": job_id, "run_id": run_id})
