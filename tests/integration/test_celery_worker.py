"""A real Celery worker process (prefork) against the test containers."""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import redis
from celery import Celery

from rag_engine.config import Settings
from rag_engine.worker.celery_app import configure
from tests.conftest import TEST_ENV

_HEALTH = "rag_engine.worker.tasks.maintenance.health"


def _worker_env(config: Settings) -> dict[str, str]:
    """The worker subprocess sees the same containers and database as the test."""
    qdrant_key = config.QDRANT_API_KEY
    return {
        **os.environ,
        **TEST_ENV,
        "MONGO_URI": config.MONGO_URI.get_secret_value() if config.MONGO_URI else "",
        "MONGO_DB": config.MONGO_DB,
        "REDIS_HOST": config.REDIS_HOST,
        "REDIS_PORT": str(config.REDIS_PORT),
        "REDIS_PASSWORD": config.REDIS_PASSWORD.get_secret_value() if config.REDIS_PASSWORD else "",
        "QDRANT_HOST": config.QDRANT_HOST,
        "QDRANT_HTTP_PORT": str(config.QDRANT_HTTP_PORT),
        "QDRANT_GRPC_PORT": str(config.QDRANT_GRPC_PORT),
        "QDRANT_API_KEY": qdrant_key.get_secret_value() if qdrant_key else "",
        "QDRANT_HTTPS": "false",
        "LOG_JSON": "false",
    }


class _Worker:
    def __init__(self, process: subprocess.Popen[str], log: Path, producer: Celery) -> None:
        self.process = process
        self.log = log
        self.producer = producer

    def health(self, *, queue: str = "maintenance", timeout: float = 60) -> dict:
        return self.producer.send_task(_HEALTH, queue=queue).get(timeout=timeout)


@pytest.fixture
def producer(integration_settings: Settings) -> Iterator[Celery]:
    app = configure(Celery("producer"), integration_settings)
    yield app
    app.close()


@pytest.fixture
def worker(integration_settings: Settings, producer: Celery, tmp_path: Path) -> Iterator[_Worker]:
    log = tmp_path / "worker.log"
    with log.open("w") as sink:
        process = subprocess.Popen(
            [sys.executable, "-m", "celery", "--app", "rag_engine.worker.celery_app", "worker",
             "--pool", "prefork", "--concurrency", "2", "--queues", "maintenance",
             "--loglevel", "INFO", "--without-mingle", "--without-gossip"],
            env=_worker_env(integration_settings), stdout=sink, stderr=subprocess.STDOUT,
            text=True,
        )  # fmt: skip
    running = _Worker(process, log, producer)
    try:
        deadline = time.monotonic() + 90
        while True:  # ready = a task actually round-trips
            try:
                running.health(timeout=5)
                break
            except Exception:
                if process.poll() is not None or time.monotonic() > deadline:
                    output = log.read_text()[-3000:]
                    raise AssertionError(f"worker never became ready:\n{output}") from None
        yield running
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


# ---- #
# Tests
# ---- #


class TestCeleryWorker_S75:
    """S75 — A real prefork worker: per-process runtime, routing, graceful shutdown."""

    def test_a_task_runs_with_the_process_runtime(self, worker: _Worker) -> None:
        result = worker.health()

        assert result["ready"] is True
        assert {name: dep["status"] for name, dep in result["dependencies"].items()} == {
            "mongo": "ok", "redis": "ok", "qdrant": "ok", "vector_collections": "ok",
        }  # fmt: skip
        assert result["pid"] not in (os.getpid(), worker.process.pid)  # a prefork child

    def test_tasks_keep_working_on_the_same_process_loop(self, worker: _Worker) -> None:
        results = [worker.health() for _ in range(6)]

        assert all(r["ready"] for r in results)  # clients stay usable across tasks

    def test_a_queue_nobody_consumes_keeps_its_message(
        self, worker: _Worker, integration_settings: Settings
    ) -> None:
        pending = worker.producer.send_task(_HEALTH, queue="crawl")  # this worker: maintenance

        time.sleep(3)

        assert pending.state == "PENDING"
        password = integration_settings.REDIS_PASSWORD
        broker = redis.Redis(host=integration_settings.REDIS_HOST,
                             port=integration_settings.REDIS_PORT,
                             password=password.get_secret_value() if password else None,
                             db=integration_settings.REDIS_CELERY_BROKER_DB)  # fmt: skip
        assert broker.llen("crawl") == 1  # still queued for a crawl worker
        broker.close()

    def test_sigterm_shuts_the_runtime_down_cleanly(self, worker: _Worker) -> None:
        worker.process.send_signal(signal.SIGTERM)

        code = worker.process.wait(timeout=30)

        output = worker.log.read_text()
        assert code == 0, output[-3000:]
        assert output.count("ready") >= 2 and "stopped" in output  # each child started/stopped
