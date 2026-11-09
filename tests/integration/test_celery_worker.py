"""A real Celery worker process (prefork) against the test containers."""

import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import redis
from celery import Celery

from rag_engine.config import Settings
from rag_engine.worker.celery_app import configure
from tests.support.celery_worker import HEALTH_TASK, CeleryWorker, running_worker


@pytest.fixture
def producer(integration_settings: Settings) -> Iterator[Celery]:
    app = configure(Celery("producer"), integration_settings)
    yield app
    app.close()


@pytest.fixture
def worker(
    integration_settings: Settings, producer: Celery, tmp_path: Path
) -> Iterator[CeleryWorker]:
    with running_worker(integration_settings, producer, tmp_path / "worker.log") as running:
        yield running


# ---- #
# Tests
# ---- #


class TestCeleryWorker_S75:
    """S75 — A real prefork worker: per-process runtime, routing, graceful shutdown."""

    def test_a_task_runs_with_the_process_runtime(self, worker: CeleryWorker) -> None:
        result = worker.health()

        assert result["ready"] is True
        assert {name: dep["status"] for name, dep in result["dependencies"].items()} == {
            "mongo": "ok", "redis": "ok", "qdrant": "ok", "vector_collections": "ok",
        }  # fmt: skip
        assert result["pid"] not in (os.getpid(), worker.process.pid)  # a prefork child

    def test_tasks_keep_working_on_the_same_process_loop(self, worker: CeleryWorker) -> None:
        results = [worker.health() for _ in range(6)]

        assert all(r["ready"] for r in results)  # clients stay usable across tasks

    def test_a_queue_nobody_consumes_keeps_its_message(
        self, worker: CeleryWorker, integration_settings: Settings
    ) -> None:
        pending = worker.producer.send_task(HEALTH_TASK, queue="crawl")  # this worker: maintenance

        time.sleep(3)

        assert pending.state == "PENDING"
        password = integration_settings.REDIS_PASSWORD
        broker = redis.Redis(host=integration_settings.REDIS_HOST,
                             port=integration_settings.REDIS_PORT,
                             password=password.get_secret_value() if password else None,
                             db=integration_settings.REDIS_CELERY_BROKER_DB)  # fmt: skip
        assert broker.llen("crawl") == 1  # still queued for a crawl worker
        broker.close()

    def test_sigterm_shuts_the_runtime_down_cleanly(self, worker: CeleryWorker) -> None:
        worker.process.send_signal(signal.SIGTERM)

        code = worker.process.wait(timeout=30)

        output = worker.log.read_text()
        assert code == 0, output[-3000:]
        assert output.count("ready") >= 2 and "stopped" in output  # each child started/stopped
