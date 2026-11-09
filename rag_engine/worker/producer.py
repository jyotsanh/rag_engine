"""Sending tasks by name, for producers that must not import the task modules (the API).

`TaskQueue` is what the API's services depend on (`get_task_queue`, overridable in tests);
`CeleryTaskQueue` publishes to the broker. Publishing is blocking I/O, so it runs in a
thread. **A failed publish never fails the request**: the record is already marked, and
the beat sweeps restart work whose message was lost (Steps 80-81).
"""

import asyncio
from typing import Protocol

from loguru import logger

from rag_engine.worker.celery_app import (
    DELETE_KNOWLEDGE_BASE_TASK,
    DELETE_SOURCE_TASK,
    DELIVER_WEBHOOK_TASK,
    MAINTENANCE_QUEUE,
    WEBHOOKS_QUEUE,
    celery_app,
)


class TaskQueue(Protocol):
    async def delete_source(self, tenant_id: str, source_id: str) -> None: ...

    async def delete_knowledge_base(self, tenant_id: str, kb_id: str) -> None: ...


class CeleryTaskQueue:
    async def delete_source(self, tenant_id: str, source_id: str) -> None:
        await self._send(DELETE_SOURCE_TASK, [tenant_id, source_id])

    async def delete_knowledge_base(self, tenant_id: str, kb_id: str) -> None:
        await self._send(DELETE_KNOWLEDGE_BASE_TASK, [tenant_id, kb_id])

    @staticmethod
    async def _send(name: str, args: list[str]) -> None:
        try:
            await asyncio.to_thread(celery_app.send_task, name, args=args, queue=MAINTENANCE_QUEUE)
        except Exception:
            logger.opt(exception=True).error("could not enqueue {}{}; the sweep will", name, args)


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
