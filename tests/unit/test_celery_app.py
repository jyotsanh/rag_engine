"""Celery app configuration and the per-process async worker context (no broker needed)."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from rag_engine.config import Settings
from rag_engine.worker import context as context_module
from rag_engine.worker.celery_app import (
    QUEUES,
    TASK_ROUTES,
    VISIBILITY_MARGIN_SECONDS,
    celery_app,
    configure,
)
from rag_engine.worker.context import WorkerContext
from rag_engine.worker.signals import _on_worker_init


class _FakeRuntime:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.loops: list[int] = []
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")
        if self.fail:
            raise ConnectionError("mongo down")

    async def stop(self) -> None:
        self.events.append("stop")


async def _loop_id() -> int:
    return id(asyncio.get_running_loop())


# ---- #
# Tests
# ---- #


class TestCeleryApp_S75:
    """S75 — Celery app: at-least-once config, queues/routes, per-process async runtime."""

    # ---- configuration ---- #

    def test_delivery_guarantees(self) -> None:
        conf = celery_app.conf

        assert conf.task_acks_late is True
        assert conf.task_reject_on_worker_lost is True
        assert conf.worker_prefetch_multiplier == 1
        assert (conf.task_serializer, conf.result_serializer) == ("json", "json")
        assert conf.accept_content == ["json"]
        assert conf.worker_hijack_root_logger is False
        assert conf.enable_utc is True

    def test_visibility_timeout_outlasts_the_longest_task(self) -> None:
        config = Settings(_env_file=None, CELERY_TASK_SOFT_TIME_LIMIT_SECONDS=100,
                          CELERY_TASK_TIME_LIMIT_SECONDS=200)  # fmt: skip
        app = configure(type(celery_app)("probe"), config)

        timeout = app.conf.broker_transport_options["visibility_timeout"]
        assert timeout == 200 + VISIBILITY_MARGIN_SECONDS > app.conf.task_time_limit
        assert (app.conf.task_soft_time_limit, app.conf.task_time_limit) == (100, 200)

    def test_broker_and_backend_use_their_own_redis_dbs(self) -> None:
        config = Settings(
            _env_file=None, REDIS_HOST="redis.internal", REDIS_PORT=6380, REDIS_DB=0,
            REDIS_CELERY_BROKER_DB=1, REDIS_CELERY_RESULT_DB=2,
        )  # fmt: skip
        app = configure(type(celery_app)("probe"), config)

        assert app.conf.broker_url.endswith("redis.internal:6380/1")
        assert app.conf.result_backend.endswith("redis.internal:6380/2")

    def test_queues_and_routes(self) -> None:
        router = celery_app.amqp.router

        assert [q.name for q in celery_app.conf.task_queues] == list(QUEUES)
        assert celery_app.conf.task_default_queue == "maintenance"
        for module, queue in [("crawl", "crawl"), ("maintenance", "maintenance"),
                              ("webhooks", "webhooks")]:  # fmt: skip
            route = router.route({}, f"rag_engine.worker.tasks.{module}.anything")
            assert route["queue"].name == queue
        assert set(TASK_ROUTES) == {f"rag_engine.worker.tasks.{q}.*" for q in QUEUES}
        celery_app.loader.import_default_modules()  # what a worker does at startup (`include`)
        assert "rag_engine.worker.tasks.maintenance.health" in celery_app.tasks

    def test_pools_sharing_one_loop_are_refused(self) -> None:
        for module in ("celery.concurrency.thread", "celery.concurrency.gevent"):
            sender = SimpleNamespace(pool_cls=type("Pool", (), {"__module__": module}))
            with pytest.raises(RuntimeError, match="not supported"):
                _on_worker_init(sender=sender)
        prefork = SimpleNamespace(
            pool_cls=type("Pool", (), {"__module__": "celery.concurrency.prefork"})
        )
        _on_worker_init(sender=prefork)  # fine

    # ---- the per-process async context ---- #

    def test_every_task_runs_on_the_same_process_loop(self) -> None:
        runtime = _FakeRuntime()
        context = WorkerContext(lambda: runtime)  # ty: ignore[invalid-argument-type]

        loops = [context.run(_loop_id()) for _ in range(3)]

        assert len(set(loops)) == 1  # clients bound to this loop stay usable
        assert runtime.events == ["start"]  # started once, lazily
        context.shutdown()

    def test_shutdown_stops_the_runtime_and_the_browser_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime = _FakeRuntime()
        browser_stops: list[str] = []

        async def stop_browsers() -> None:
            browser_stops.append("stopped")

        monkeypatch.setattr(context_module, "shutdown_process_browser_pool", stop_browsers)
        context = WorkerContext(lambda: runtime)  # ty: ignore[invalid-argument-type]
        context.start()

        context.shutdown()
        context.shutdown()  # idempotent

        assert runtime.events == ["start", "stop"]
        assert browser_stops == ["stopped"]
        assert not context.started

    def test_a_forked_child_builds_its_own_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtimes: list[_FakeRuntime] = []

        def factory() -> _FakeRuntime:
            runtimes.append(_FakeRuntime())
            return runtimes[-1]

        context = WorkerContext(factory)  # ty: ignore[invalid-argument-type]
        context.start()
        parent_pid = os.getpid()
        monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)  # "after fork"

        context.start()

        assert len(runtimes) == 2  # never reuses the parent's clients
        assert context.started
        context.shutdown()

    def test_a_failed_start_propagates_and_leaves_nothing_open(self) -> None:
        context = WorkerContext(lambda: _FakeRuntime(fail=True))  # ty: ignore[invalid-argument-type]
        coroutine = _loop_id()

        with pytest.raises(ConnectionError):
            context.run(coroutine)

        assert not context.started
        assert coroutine.cr_frame is None  # closed, so no "never awaited" warning
