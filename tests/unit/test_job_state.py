"""Crawl job state machine: the transition table and the model's held-job invariant."""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from rag_engine.domain import (
    ACTIVE_JOB_STATUSES,
    JOB_TRANSITIONS,
    TERMINAL_JOB_STATUSES,
    CrawlJob,
    JobCounters,
    JobStatus,
)

Q, R, F = JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FINALIZING
OK, FAIL, CANCEL = JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED

_EXPECTED_EDGES = {
    (Q, R),
    (Q, FAIL),
    (Q, CANCEL),
    (R, F),
    (R, FAIL),
    (R, CANCEL),
    (R, Q),
    (F, OK),
    (F, FAIL),
    (F, Q),
}


def _job(**overrides: Any) -> CrawlJob:
    fields: dict[str, Any] = {
        "tenant_id": "ten_" + "1" * 32,
        "kb_id": "kb_" + "1" * 32,
        "source_id": "src_" + "1" * 32,
        "seed_url": "https://acme.example/",
    }
    return CrawlJob(**(fields | overrides))


# ---- #
# Tests
# ---- #


class TestJobStateMachine_S49:
    """S49 — CrawlJob state machine: allowed transitions and held-job invariants."""

    def test_transition_table_is_exactly_the_documented_one(self) -> None:
        edges = {(a, b) for a, targets in JOB_TRANSITIONS.items() for b in targets}

        assert set(JOB_TRANSITIONS) == set(JobStatus)
        assert edges == _EXPECTED_EDGES

    @pytest.mark.parametrize("source", list(JobStatus))
    @pytest.mark.parametrize("target", list(JobStatus))
    def test_can_become_follows_the_table(self, source: JobStatus, target: JobStatus) -> None:
        assert source.can_become(target) is ((source, target) in _EXPECTED_EDGES)

    def test_terminal_statuses_are_final_and_every_active_one_can_fail(self) -> None:
        assert all(not JOB_TRANSITIONS[s] for s in TERMINAL_JOB_STATUSES)
        assert all(s.can_become(FAIL) for s in ACTIVE_JOB_STATUSES)

    def test_finalizing_cannot_be_cancelled(self) -> None:
        # the version swap either completes or fails; a half-cancelled swap would lose data
        assert not F.can_become(CANCEL)

    def test_success_is_only_reached_through_finalizing(self) -> None:
        assert [s for s in JobStatus if s.can_become(OK)] == [F]
        assert [s for s in JobStatus if s.can_become(F)] == [R]

    @pytest.mark.parametrize("status", [R, F])
    def test_held_job_needs_run_and_start(self, status: JobStatus) -> None:
        now = datetime(2026, 11, 7, tzinfo=UTC)

        with pytest.raises(ValidationError, match="run_id and started_at"):
            _job(status=status, started_at=now)
        with pytest.raises(ValidationError, match="run_id and started_at"):
            _job(status=status, run_id="run-1")
        assert _job(status=status, run_id="run-1", started_at=now).status is status

    def test_counter_names(self) -> None:
        assert JobCounters.names() == {
            "discovered",
            "fetched",
            "failed",
            "skipped",
            "duplicates",
            "chunked",
            "embedded",
        }
