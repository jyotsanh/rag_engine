"""Celery signals: per-process logging, async runtime and teardown (Step 75)."""

from typing import Any

from celery import signals
from loguru import logger

from rag_engine.config import setup_logging
from rag_engine.worker.context import worker_context

_UNSUPPORTED_POOLS = (".thread", ".gevent", ".eventlet")  # celery.concurrency.<name>


@signals.worker_init.connect
def _on_worker_init(sender: Any = None, **_: Any) -> None:
    """Main worker process: logging, and refuse pools that would share one event loop."""
    setup_logging()
    pool = getattr(getattr(sender, "pool_cls", None), "__module__", "") or ""
    if any(name in pool for name in _UNSUPPORTED_POOLS):
        raise RuntimeError(f"worker pool {pool!r} is not supported: use prefork or solo")


@signals.worker_process_init.connect
def _on_worker_process_init(**_: Any) -> None:
    """A prefork child: logging, then its own loop and runtime (connections after fork)."""
    setup_logging()
    worker_context.start()


@signals.worker_process_shutdown.connect
def _on_worker_process_shutdown(**_: Any) -> None:
    _shutdown()


@signals.worker_shutdown.connect
def _on_worker_shutdown(**_: Any) -> None:
    _shutdown()  # the solo pool runs tasks in the main process


def _shutdown() -> None:
    try:
        worker_context.shutdown()
    except Exception:  # never block the worker's exit on a close error
        logger.opt(exception=True).error("worker runtime shutdown failed")
