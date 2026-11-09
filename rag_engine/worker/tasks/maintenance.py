"""`maintenance` queue tasks: health, version cleanup (Step 78), crash recovery (Step 80,
on Celery beat), cascade deletes and their sweep (Step 81)."""

import os
from collections.abc import Coroutine
from dataclasses import asdict
from typing import Any

from celery import Task

from rag_engine.infrastructure import (
    DatabaseUnavailableError,
    RedisUnavailableError,
    VectorStoreUnavailableError,
    check_readiness,
)
from rag_engine.services.cascade import (
    CascadeDeleter,
    CascadeResult,
    CascadeStatus,
    DeletionSweeper,
)
from rag_engine.services.cleanup import VersionCleaner
from rag_engine.worker.celery_app import (
    DELETE_KNOWLEDGE_BASE_TASK,
    DELETE_SOURCE_TASK,
    MAINTENANCE_QUEUE,
    RECOVER_JOBS_TASK,
    SWEEP_DELETIONS_TASK,
    celery_app,
)
from rag_engine.worker.context import run_async, worker_context
from rag_engine.worker.producer import send_kb_deletion, send_source_deletion
from rag_engine.worker.recovery import JobRecovery

CLEANUP_VERSION = "rag_engine.worker.tasks.maintenance.cleanup_version"
_CLEANUP_MAX_RETRIES = 6
_TRANSIENT = (DatabaseUnavailableError, RedisUnavailableError, VectorStoreUnavailableError)

_HEALTH_BUDGET_SECONDS = 5.0


@celery_app.task(name="rag_engine.worker.tasks.maintenance.health")
def health() -> dict[str, Any]:
    """The worker's own view of its dependencies: proves a worker process is consuming the
    queue *and* its runtime (Mongo, Redis, Qdrant, collections) works from inside it."""
    report = run_async(
        check_readiness(worker_context.runtime.health_checks(), _HEALTH_BUDGET_SECONDS)
    )
    return {"pid": os.getpid(), **report.model_dump(mode="json")}


def enqueue_version_cleanup(
    tenant_id: str, kb_id: str, source_id: str, version: str, countdown: float = 0
) -> None:
    """Send `cleanup_version` by name to the `maintenance` queue."""
    celery_app.send_task(
        CLEANUP_VERSION,
        args=[tenant_id, kb_id, source_id, version],
        queue=MAINTENANCE_QUEUE,
        countdown=countdown or None,
    )


@celery_app.task(name=CLEANUP_VERSION, bind=True, max_retries=_CLEANUP_MAX_RETRIES)
def cleanup_version(
    self: Task, tenant_id: str, kb_id: str, source_id: str, version: str
) -> dict[str, Any]:
    """Delete a version's vectors and chunks (never the active one). Idempotent; transient
    failures are retried with exponential backoff (30 s, 60 s, ... capped at 10 min)."""
    store = worker_context.runtime.vector_store
    assert store is not None
    cleaner = VersionCleaner(worker_context.runtime.mongo.db, store)
    try:
        result = run_async(
            cleaner.delete_version(
                tenant_id=tenant_id, kb_id=kb_id, source_id=source_id, version=version
            )
        )
    except _TRANSIENT as exc:
        raise self.retry(exc=exc, countdown=min(600, 30 * 2**self.request.retries)) from exc
    return asdict(result)


@celery_app.task(name=RECOVER_JOBS_TASK)
def recover_jobs() -> dict[str, Any]:
    """Requeue/fail jobs whose worker died and re-send lost queued jobs (Celery beat).

    Transient failures aren't retried: the next beat tick runs the sweep again.
    """
    from rag_engine.worker.tasks.crawl import enqueue_crawl_job  # crawl imports this module

    recovery = JobRecovery(
        worker_context.runtime.mongo.db,
        enqueue=enqueue_crawl_job,
        enqueue_cleanup=enqueue_version_cleanup,
        config=worker_context.runtime.config,
    )
    report = run_async(recovery.run())
    return asdict(report)


def _cascade(self: Task, run: Coroutine[Any, Any, CascadeResult]) -> dict[str, Any]:
    """Run a cascade; back off on outages, then settle its result."""
    try:
        result = run_async(run)
    except _TRANSIENT as exc:
        raise self.retry(exc=exc, countdown=min(600, 30 * 2**self.request.retries)) from exc
    return settle_cascade(self, result, worker_context.runtime.config.CASCADE_RETRY_DELAY_SECONDS)


def settle_cascade(self: Task, result: CascadeResult, retry_delay: float) -> dict[str, Any]:
    """A cascade waiting for crawl jobs to stop is retried after `retry_delay`.

    Unbounded on purpose: Step 80 settles any job left active (cancels it, or fails it
    `worker_lost`), so the wait ends; the deletion sweep covers a lost retry.
    (`max_retries=None` would mean the task's default limit, so it's lifted explicitly.)
    """
    if result.status is CascadeStatus.WAITING_FOR_JOBS:
        raise self.retry(countdown=retry_delay, max_retries=self.request.retries + 1)
    return asdict(result)


@celery_app.task(name=DELETE_SOURCE_TASK, bind=True, max_retries=_CLEANUP_MAX_RETRIES)
def delete_source(self: Task, tenant_id: str, source_id: str) -> dict[str, Any]:
    """Cascade a deleted source: vectors, chunks, pages, jobs, then the record (Step 81)."""
    return _cascade(self, _deleter().delete_source(tenant_id, source_id))


@celery_app.task(name=DELETE_KNOWLEDGE_BASE_TASK, bind=True, max_retries=_CLEANUP_MAX_RETRIES)
def delete_knowledge_base(self: Task, tenant_id: str, kb_id: str) -> dict[str, Any]:
    """Cascade a deleted knowledge base: every source as above, then the KB (Step 81)."""
    return _cascade(self, _deleter().delete_knowledge_base(tenant_id, kb_id))


@celery_app.task(name=SWEEP_DELETIONS_TASK)
def sweep_deletions() -> dict[str, Any]:
    """Restart cascades nobody touched for `CASCADE_STALE_SECONDS` (Celery beat)."""
    sweeper = DeletionSweeper(
        worker_context.runtime.mongo.db,
        enqueue_source=send_source_deletion,
        enqueue_kb=send_kb_deletion,
        config=worker_context.runtime.config,
    )
    return asdict(run_async(sweeper.run()))


def _deleter() -> CascadeDeleter:
    store = worker_context.runtime.vector_store
    assert store is not None
    return CascadeDeleter(worker_context.runtime.mongo.db, store)
