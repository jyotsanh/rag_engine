"""`maintenance` queue tasks: health, version cleanup (Step 78), crash recovery (Step 80,
on Celery beat); cascades (Step 81)."""

import os
from dataclasses import asdict
from typing import Any

from celery import Task

from rag_engine.infrastructure import (
    DatabaseUnavailableError,
    RedisUnavailableError,
    VectorStoreUnavailableError,
    check_readiness,
)
from rag_engine.services.cleanup import VersionCleaner
from rag_engine.worker.celery_app import MAINTENANCE_QUEUE, RECOVER_JOBS_TASK, celery_app
from rag_engine.worker.context import run_async, worker_context
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
