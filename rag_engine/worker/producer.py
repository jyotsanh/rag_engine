"""Sending tasks by name, for producers that must not import the task modules (the API).

`TaskQueue` is what the API's services depend on (`get_task_queue`, overridable in tests);
`CeleryTaskQueue` publishes to the broker. Publishing is blocking I/O, so it runs in a
thread. **A failed publish never fails the request**: the record is already marked, and
the beat sweeps restart work whose message was lost (Steps 80-81).
"""

import asyncio
from typing import Protocol

from celery import Celery
from loguru import logger

from rag_engine.worker.celery_app import (
    CRAWL_QUEUE,
    DELETE_KNOWLEDGE_BASE_TASK,
    DELETE_SOURCE_TASK,
    DELIVER_WEBHOOK_TASK,
    MAINTENANCE_QUEUE,
    RUN_CRAWL_JOB_TASK,
    WEBHOOKS_QUEUE,
    celery_app,
)


class TaskQueue(Protocol):
    async def enqueue_crawl_job(self, job_id: str) -> None: ...

    async def enqueue_job_webhook(self, job_id: str) -> None: ...

    async def delete_source(self, tenant_id: str, source_id: str) -> None: ...

    async def delete_knowledge_base(self, tenant_id: str, kb_id: str) -> None: ...


class CeleryTaskQueue:
    def __init__(self, app: Celery = celery_app) -> None:
        """`app`: the producer app (the module's, configured from settings, by default)."""
        self._app = app

    async def enqueue_crawl_job(self, job_id: str) -> None:
        """A submitted crawl (Step 84). If this publish fails, the job stays queued and
        crash recovery re-sends it after `CRAWL_QUEUED_STALE_SECONDS` (Step 80)."""
        await self._send(RUN_CRAWL_JOB_TASK, [job_id], queue=CRAWL_QUEUE)

    async def enqueue_job_webhook(self, job_id: str) -> None:
        """A job the API itself ended (a queued job cancelled, Step 87); the recovery
        sweep sends it anyway if this publish fails (Step 83)."""
        await self._send(DELIVER_WEBHOOK_TASK, [job_id], queue=WEBHOOKS_QUEUE)

    async def delete_source(self, tenant_id: str, source_id: str) -> None:
        await self._send(DELETE_SOURCE_TASK, [tenant_id, source_id])

    async def delete_knowledge_base(self, tenant_id: str, kb_id: str) -> None:
        await self._send(DELETE_KNOWLEDGE_BASE_TASK, [tenant_id, kb_id])

    async def _send(self, name: str, args: list[str], *, queue: str = MAINTENANCE_QUEUE) -> None:
        try:
            await asyncio.to_thread(self._app.send_task, name, args=args, queue=queue)
        except Exception:
            logger.opt(exception=True).error(
                "could not enqueue {}{}; a recovery sweep will resend it", name, args
            )


def send_source_deletion(tenant_id: str, source_id: str) -> None:
    """Synchronous variant for worker code (the deletion sweep)."""
    celery_app.send_task(DELETE_SOURCE_TASK, args=[tenant_id, source_id], queue=MAINTENANCE_QUEUE)


def send_kb_deletion(tenant_id: str, kb_id: str) -> None:
    celery_app.send_task(
        DELETE_KNOWLEDGE_BASE_TASK, args=[tenant_id, kb_id], queue=MAINTENANCE_QUEUE
    )


def send_job_webhook(job_id: str, countdown: float = 0) -> None:
    """Deliver a finished job's completion webhook (Step 83), on the `webhooks` queue."""
    celery_app.send_task(
        DELIVER_WEBHOOK_TASK, args=[job_id], queue=WEBHOOKS_QUEUE, countdown=countdown or None
    )
