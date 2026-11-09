"""`maintenance` queue tasks (cleanup, cascades and recovery arrive in Steps 78-81)."""

import os
from typing import Any

from rag_engine.infrastructure import check_readiness
from rag_engine.worker.celery_app import celery_app
from rag_engine.worker.context import run_async, worker_context

_HEALTH_BUDGET_SECONDS = 5.0


@celery_app.task(name="rag_engine.worker.tasks.maintenance.health")
def health() -> dict[str, Any]:
    """The worker's own view of its dependencies: proves a worker process is consuming the
    queue *and* its runtime (Mongo, Redis, Qdrant, collections) works from inside it."""
    report = run_async(
        check_readiness(worker_context.runtime.health_checks(), _HEALTH_BUDGET_SECONDS)
    )
    return {"pid": os.getpid(), **report.model_dump(mode="json")}
