"""Async runtime of a Celery worker process (Step 75).

Celery tasks are synchronous; our code is async, and its clients (MongoDB, Redis, Qdrant,
the browser pool) are bound to the event loop they were created on. So each worker
process owns **one persistent event loop** (`asyncio.Runner`) and one started
`AppRuntime` on it, and every task runs its coroutine on that same loop
(`WorkerContext.run`).

- **Fork safety**: the context is created *after* fork (`worker_process_init`, or lazily
  on the first task for the solo pool) and records its pid. A context inherited through
  `fork()` is discarded, never reused: the child builds its own clients and loop.
- **Shutdown** (`worker_process_shutdown` / `worker_shutdown`): stop the runtime (closing
  every connection), then the process's browser pool, then the loop.
- Supported pools: **prefork** (production: one context per child) and **solo** (tests,
  debugging). The threads/gevent pools would share one loop across threads, and are
  refused.
"""

import asyncio
import os
from collections.abc import Callable, Coroutine
from typing import Any

from loguru import logger

from rag_engine.crawling.fetchers.browser import shutdown_process_browser_pool
from rag_engine.runtime import AppRuntime

type RuntimeFactory = Callable[[], AppRuntime]


class WorkerContext:
    def __init__(self, runtime_factory: RuntimeFactory = AppRuntime) -> None:
        self._factory = runtime_factory
        self._runner: asyncio.Runner | None = None
        self._runtime: AppRuntime | None = None
        self._pid: int | None = None

    @property
    def started(self) -> bool:
        return self._runtime is not None and self._pid == os.getpid()

    @property
    def runtime(self) -> AppRuntime:
        """This process's started runtime (started on first access)."""
        self.start()
        assert self._runtime is not None
        return self._runtime

    def start(self) -> None:
        """Create the loop and start the runtime in this process (idempotent)."""
        if self.started:
            return
        if self._pid is not None and self._pid != os.getpid():
            # inherited through fork: the parent's loop and clients are not ours to touch
            self._runner, self._runtime = None, None
        runner = asyncio.Runner()
        runtime = self._factory()
        try:
            runner.run(runtime.start())
        except BaseException:
            runner.close()
            raise
        self._runner, self._runtime, self._pid = runner, runtime, os.getpid()
        logger.info("worker process {} ready", self._pid)

    def run[T](self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run `coroutine` to completion on this process's loop (starting it if needed)."""
        if not self.started:
            try:
                self.start()
            except BaseException:
                coroutine.close()
                raise
        assert self._runner is not None
        return self._runner.run(coroutine)

    def shutdown(self) -> None:
        """Stop the runtime, the browser pool and the loop. Safe to call more than once."""
        if not self.started:
            self._runner, self._runtime, self._pid = None, None, None
            return
        assert self._runner is not None and self._runtime is not None
        try:
            self._runner.run(self._runtime.stop())
            self._runner.run(shutdown_process_browser_pool())
        finally:
            self._runner.close()
            logger.info("worker process {} stopped", self._pid)
            self._runner, self._runtime, self._pid = None, None, None


# the process-wide context every task uses
worker_context = WorkerContext()


def run_async[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return worker_context.run(coroutine)
