"""Cancellation without infrastructure: stop-reason ranking, settings, Redis failures."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError

from rag_engine.config import Settings
from rag_engine.crawling import RedisCancelFlags, finish_cancelled, request_cancel
from rag_engine.domain import CrawlJob, CrawlStopReason, JobStatus, PageReason
from rag_engine.infrastructure import RedisKeyspace, RedisUnavailableError

_JOB = CrawlJob(
    tenant_id="ten_" + "1" * 32,
    kb_id="kb_" + "1" * 32,
    source_id="src_" + "1" * 32,
    seed_url="https://a.example/",
)


class _DownRedis:
    """Every command fails like an unreachable Redis."""

    def __getattr__(self, name: str) -> Any:
        async def fail(*args: object, **kwargs: object) -> None:
            raise RedisConnectionError("connection refused")

        return fail


class _Jobs:
    """The two `JobRepository` calls the helpers make."""

    def __init__(self, status: JobStatus) -> None:
        self.status = status
        self.calls: list[str] = []

    async def request_cancel(self, tenant_id: str, job_id: str) -> CrawlJob:
        self.calls.append("request_cancel")
        now = datetime.now(UTC)
        return _JOB.model_copy(
            update={"status": self.status, "run_id": "run-1", "started_at": now,
                    "cancel_requested_at": now}
        )  # fmt: skip

    async def cancel(self, job_id: str, run_id: str) -> CrawlJob:
        self.calls.append("cancel")
        now = datetime.now(UTC)
        return _JOB.model_copy(
            update={"status": JobStatus.CANCELLED, "started_at": now, "finished_at": now}
        )


def _down_flags() -> RedisCancelFlags:
    return RedisCancelFlags(_DownRedis(), RedisKeyspace("test"))  # ty: ignore[invalid-argument-type]


# ---- #
# Tests
# ---- #


class TestCancellation_S52:
    """S52 — Cancellation: stop ranking, settings, and Redis outages that must not fail it."""

    def test_cancel_is_a_hard_stop_that_outranks_the_caps(self) -> None:
        cancelled = CrawlStopReason.CANCELLED

        assert cancelled.is_hard and cancelled.page_reason is PageReason.JOB_CANCELLED
        assert [r.rank for r in CrawlStopReason] == [0, 1, 1, 2]
        assert {r.page_reason for r in CrawlStopReason} == {
            PageReason.MAX_PAGES,
            PageReason.MAX_TOTAL_BYTES,
            PageReason.MAX_DURATION,
            PageReason.JOB_CANCELLED,
        }

    def test_flag_ttl_must_outlive_a_job(self) -> None:
        with pytest.raises(ValidationError, match="CRAWL_CANCEL_FLAG_TTL_SECONDS"):
            Settings(_env_file=None, CRAWL_MAX_JOB_SECONDS=7200, CRAWL_CANCEL_FLAG_TTL_SECONDS=60)
        with pytest.raises(ValidationError):
            Settings(_env_file=None, CRAWL_CANCEL_POLL_SECONDS=0)

    async def test_flag_errors_are_typed(self) -> None:
        flags = _down_flags()

        for call in (flags.request, flags.is_cancelled, flags.clear):
            with pytest.raises(RedisUnavailableError):
                await call(_JOB.id)

    async def test_request_cancel_survives_a_redis_outage(self) -> None:
        jobs = _Jobs(JobStatus.RUNNING)

        job = await request_cancel(jobs, _down_flags(), _JOB.tenant_id, _JOB.id)  # ty: ignore[invalid-argument-type]

        assert job.status is JobStatus.RUNNING and job.cancel_requested_at is not None
        assert jobs.calls == ["request_cancel"]

    async def test_finish_cancelled_survives_a_redis_outage(self) -> None:
        jobs = _Jobs(JobStatus.RUNNING)

        job = await finish_cancelled(jobs, _down_flags(), _JOB.id, "run-1")  # ty: ignore[invalid-argument-type]

        assert job.status is JobStatus.CANCELLED
        assert jobs.calls == ["cancel"]
