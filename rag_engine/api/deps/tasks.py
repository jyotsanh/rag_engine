"""The task queue routes hand work to (Celery by name). Tests override `get_task_queue`."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from rag_engine.worker.producer import CeleryTaskQueue, TaskQueue


@lru_cache(maxsize=1)
def get_task_queue() -> TaskQueue:
    return CeleryTaskQueue()


Tasks = Annotated[TaskQueue, Depends(get_task_queue)]
