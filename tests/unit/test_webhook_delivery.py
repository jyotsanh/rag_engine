"""Webhook delivery pieces that need no network: backoff, Retry-After, payload, settling
the Celery task, settings and routing."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from rag_engine.config import Settings
from rag_engine.domain import CrawlConfig, CrawlJob, CrawlReport, JobStatus, PageStatus
from rag_engine.webhooks.delivery import (
    AttemptResult,
    DeliveryOutcome,
    _parse_retry_after,
    build_payload,
    event_id,
    render_body,
    retry_delay,
)
from rag_engine.worker.celery_app import DELIVER_WEBHOOK_TASK, celery_app

_NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


def _config(**fields: float) -> Settings:
    values = {"WEBHOOK_RETRY_BASE_SECONDS": 30, "WEBHOOK_RETRY_MAX_SECONDS": 3600, **fields}
    return Settings(_env_file=None, **values)  # type: ignore


def _finished_job() -> CrawlJob:
    return CrawlJob(
        tenant_id="ten_" + "1" * 32, kb_id="kb_" + "1" * 32, source_id="src_" + "1" * 32,
        seed_url="https://acme.example/", status=JobStatus.SUCCEEDED, run_id="run-1",
        started_at=_NOW - timedelta(minutes=5), finished_at=_NOW, attempts=1,
        config=CrawlConfig(webhook_url="https://hooks.acme.example/rag"),
        report=CrawlReport(pages={PageStatus.SUCCEEDED: 7, PageStatus.SKIPPED: 2}),
    )  # fmt: skip


class _RetryCalled(Exception):
    pass


class _FakeTask:
    def __init__(self, retries: int) -> None:
        self.request = SimpleNamespace(retries=retries)
        self.retry_kwargs: dict[str, object] = {}

    def retry(self, **kwargs: object) -> Exception:
        self.retry_kwargs = kwargs
        return _RetryCalled()


# ---- #
# Tests
# ---- #


class TestWebhookDeliveryUnits_S83:
    """S83 — backoff with jitter and Retry-After, the payload, task settling, wiring."""

    @pytest.mark.parametrize(
        ("attempt", "low", "high"),
        [(1, 15, 30), (2, 30, 60), (3, 60, 120), (8, 1800, 3600), (20, 1800, 3600)],
    )
    def test_backoff_is_exponential_capped_with_equal_jitter(
        self, attempt: int, low: float, high: float
    ) -> None:
        config = _config()

        lowest = retry_delay(attempt, retry_after=None, config=config, rng=lambda: 0.0)
        highest = retry_delay(attempt, retry_after=None, config=config, rng=lambda: 1.0)

        assert (lowest, highest) == (low, high)

    def test_retry_after_is_a_floor_capped_at_the_maximum(self) -> None:
        config = _config()

        assert retry_delay(1, retry_after=120, config=config, rng=lambda: 0.0) == 120
        assert retry_delay(1, retry_after=5, config=config, rng=lambda: 0.0) == 15
        assert retry_delay(1, retry_after=10**6, config=config, rng=lambda: 0.0) == 3600

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("120", 120.0), (" 7 ", 7.0), ("Mon, 09 Nov 2026 12:02:00 GMT", 120.0),
         ("Mon, 09 Nov 2026 11:00:00 GMT", 0.0), ("soon", None), ("", None), (None, None)],
    )  # fmt: skip
    def test_retry_after_parsing(self, value: str | None, expected: float | None) -> None:
        assert _parse_retry_after(value, _NOW) == expected

    def test_payload(self) -> None:
        job = _finished_job()

        payload = build_payload(job)
        body = render_body(payload)

        assert payload["id"] == event_id(job) == "evt_" + job.id.removeprefix("job_")
        assert payload["type"] == "job.succeeded"
        assert payload["created_at"] == _NOW.isoformat()
        data = payload["data"]["job"]
        assert data["id"] == job.id and data["status"] == "succeeded" and data["error"] is None
        assert data["report"] == {"pages": {"succeeded": 7, "skipped": 2},
                                  "stop_reason": None, "stop_detail": None}  # fmt: skip
        assert json.loads(body) == payload
        assert b" " not in body.replace(b"https://acme", b"")  # compact
        assert body == render_body(build_payload(job))  # stable bytes across attempts

    def test_a_retry_outcome_reschedules_the_task_without_celerys_limit(self) -> None:
        from rag_engine.worker.tasks.webhooks import settle_attempt

        task = _FakeTask(retries=9)
        retry = AttemptResult(DeliveryOutcome.RETRY, attempt=2, retry_in=60.0)

        with pytest.raises(_RetryCalled):
            settle_attempt(task, retry)  # ty: ignore[invalid-argument-type]

        assert task.retry_kwargs == {"countdown": 60.0, "max_retries": 10}

    @pytest.mark.parametrize("outcome", [DeliveryOutcome.DELIVERED, DeliveryOutcome.FAILED,
                                         DeliveryOutcome.SKIPPED])  # fmt: skip
    def test_other_outcomes_end_the_task(self, outcome: DeliveryOutcome) -> None:
        from rag_engine.worker.tasks.webhooks import settle_attempt

        task = _FakeTask(retries=0)

        result = settle_attempt(task, AttemptResult(outcome, attempt=1))  # ty: ignore[invalid-argument-type]

        assert result["outcome"] == outcome.value and task.retry_kwargs == {}

    def test_the_task_is_registered_on_the_webhooks_queue(self) -> None:
        celery_app.loader.import_default_modules()

        assert DELIVER_WEBHOOK_TASK in celery_app.tasks
        assert celery_app.amqp.router.route({}, DELIVER_WEBHOOK_TASK)["queue"].name == "webhooks"

    def test_settings(self) -> None:
        config = Settings(_env_file=None)

        assert (config.WEBHOOK_TIMEOUT_SECONDS, config.WEBHOOK_MAX_ATTEMPTS) == (10, 8)
        with pytest.raises(ValueError, match="WEBHOOK_STALE_SECONDS"):
            _config(WEBHOOK_STALE_SECONDS=3600)
        with pytest.raises(ValueError, match="WEBHOOK_RETRY_BASE_SECONDS"):
            Settings(_env_file=None, WEBHOOK_RETRY_BASE_SECONDS=10, WEBHOOK_RETRY_MAX_SECONDS=5)
