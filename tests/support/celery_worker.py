"""A real Celery worker subprocess (prefork) against the test containers (Steps 75, 88)."""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

from celery import Celery

from rag_engine.config import Settings
from tests.conftest import TEST_ENV

HEALTH_TASK = "rag_engine.worker.tasks.maintenance.health"


def worker_env(config: Settings, extra: Mapping[str, str] | None = None) -> dict[str, str]:
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
        **(extra or {}),
    }


class CeleryWorker:
    def __init__(self, process: subprocess.Popen[str], log: Path, producer: Celery) -> None:
        self.process = process
        self.log = log
        self.producer = producer

    def health(self, *, queue: str = "maintenance", timeout: float = 60) -> dict:
        return self.producer.send_task(HEALTH_TASK, queue=queue).get(timeout=timeout)

    def output(self, tail: int = 4000) -> str:
        return self.log.read_text()[-tail:]


@contextmanager
def running_worker(
    config: Settings,
    producer: Celery,
    log: Path,
    *,
    queues: str = "maintenance",
    concurrency: int = 2,
    extra_env: Mapping[str, str] | None = None,
    ready_timeout: float = 90,
) -> Generator[CeleryWorker]:
    """Start a worker, wait until a task round-trips on `maintenance`, stop it after."""
    with log.open("w") as sink:
        process = subprocess.Popen(
            [sys.executable, "-m", "celery", "--app", "rag_engine.worker.celery_app", "worker",
             "--pool", "prefork", "--concurrency", str(concurrency), "--queues", queues,
             "--loglevel", "INFO", "--without-mingle", "--without-gossip"],
            env=worker_env(config, extra_env), stdout=sink, stderr=subprocess.STDOUT, text=True,
        )  # fmt: skip
    worker = CeleryWorker(process, log, producer)
    try:
        deadline = time.monotonic() + ready_timeout
        while True:  # ready = a task actually round-trips
            try:
                worker.health(timeout=5)
                break
            except Exception:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError(
                        f"worker never became ready:\n{worker.output(3000)}"
                    ) from None
        yield worker
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
