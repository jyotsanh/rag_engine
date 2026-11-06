"""Logging: JSON record shape, context-ID propagation, stdlib interception."""

import asyncio
import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from rag_engine.config import get_log_context, log_context, logger, setup_logging


@pytest.fixture
def json_stream() -> Iterator[io.StringIO]:
    """Configure JSON logging into an in-memory stream; restore defaults afterwards."""
    stream = io.StringIO()
    setup_logging(level="DEBUG", json_logs=True, stream=stream)
    yield stream
    logger.remove()
    logging.root.handlers = []


def _records(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


# ---- #
# Tests
# ---- #


class TestLogging_S4:
    """S4 — Logging: JSON sink shape, request/job/tenant context IDs, stdlib intercept."""

    def test_json_record_has_expected_shape(self, json_stream: io.StringIO) -> None:
        logger.info("hello {}", "world")

        (record,) = _records(json_stream)
        assert record["message"] == "hello world"
        assert record["level"] == "INFO"
        assert record["logger"] == __name__
        assert record["function"] == "test_json_record_has_expected_shape"
        assert isinstance(record["line"], int)
        assert record["timestamp"]
        assert record["request_id"] is None
        assert record["job_id"] is None
        assert record["tenant_id"] is None
        assert record["extra"] == {}

    def test_context_ids_attached_inside_block_and_cleared_after(
        self, json_stream: io.StringIO
    ) -> None:
        # Act
        with log_context(request_id="req-1", tenant_id="ten-1"):
            logger.info("inside")
        logger.info("outside")

        # Assert
        inside, outside = _records(json_stream)
        assert (inside["request_id"], inside["tenant_id"], inside["job_id"]) == (
            "req-1",
            "ten-1",
            None,
        )
        assert (outside["request_id"], outside["tenant_id"]) == (None, None)

    def test_nested_context_restores_outer_values(self) -> None:
        with log_context(job_id="outer"):
            with log_context(job_id="inner", request_id="r"):
                assert get_log_context() == {
                    "request_id": "r",
                    "job_id": "inner",
                    "tenant_id": None,
                }
            assert get_log_context()["job_id"] == "outer"
            assert get_log_context()["request_id"] is None

    def test_concurrent_tasks_keep_isolated_context(self, json_stream: io.StringIO) -> None:
        async def work(job_id: str) -> None:
            with log_context(job_id=job_id):
                await asyncio.sleep(0)
                logger.info("step for {}", job_id)

        async def main() -> None:
            await asyncio.gather(work("job-a"), work("job-b"))

        asyncio.run(main())

        # the asyncio stdlib logger may also emit (intercepted) DEBUG lines
        records = [r for r in _records(json_stream) if r["message"].startswith("step for")]
        assert {(r["message"], r["job_id"]) for r in records} == {
            ("step for job-a", "job-a"),
            ("step for job-b", "job-b"),
        }

    def test_explicit_bind_wins_over_context(self, json_stream: io.StringIO) -> None:
        with log_context(job_id="ctx"):
            logger.bind(job_id="bound", url="https://x.test").info("bound")

        (record,) = _records(json_stream)
        assert record["job_id"] == "bound"
        assert record["extra"] == {"url": "https://x.test"}

    def test_exception_is_serialized(self, json_stream: io.StringIO) -> None:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logger.exception("failed")

        (record,) = _records(json_stream)
        assert record["level"] == "ERROR"
        assert record["exception"]["type"] == "RuntimeError"
        assert record["exception"]["value"] == "boom"
        assert "RuntimeError: boom" in record["exception"]["traceback"]

    def test_stdlib_loggers_are_intercepted(self, json_stream: io.StringIO) -> None:
        with log_context(request_id="req-std"):
            logging.getLogger("uvicorn.error").warning("from %s", "uvicorn")
            logging.getLogger("some.library").info("from library")

        records = _records(json_stream)
        assert [(r["message"], r["level"], r["logger"], r["request_id"]) for r in records] == [
            ("from uvicorn", "WARNING", "uvicorn.error", "req-std"),
            ("from library", "INFO", "some.library", "req-std"),
        ]
        assert all(r["extra"] == {} for r in records)

    def test_uvicorn_access_log_is_silenced(self, json_stream: io.StringIO) -> None:
        logging.getLogger("uvicorn.access").info("GET / 200")

        assert _records(json_stream) == []

    def test_level_filters_lower_records(self) -> None:
        stream = io.StringIO()
        setup_logging(level="WARNING", json_logs=True, stream=stream)
        try:
            logger.info("dropped")
            logger.warning("kept")
        finally:
            logger.remove()
            logging.root.handlers = []

        assert [r["message"] for r in _records(stream)] == ["kept"]

    def test_human_format_includes_context_ids(self) -> None:
        stream = io.StringIO()
        setup_logging(level="INFO", json_logs=False, stream=stream)
        try:
            with log_context(request_id="req-h", job_id="job-h"):
                logger.info("human line")
        finally:
            logger.remove()
            logging.root.handlers = []

        line = stream.getvalue()
        assert "human line" in line
        assert "req=req-h" in line
        assert "job=job-h" in line

    def test_unknown_context_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown log context fields"), log_context(user="x"):
            pass
