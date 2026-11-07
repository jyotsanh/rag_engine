"""Crawl job repository in real MongoDB: guarded transitions, run ownership, counters."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import pytest

from rag_engine.config import Settings
from rag_engine.domain import (
    CrawlJob,
    CrawlReport,
    CrawlStopReason,
    JobCounters,
    JobError,
    JobStatus,
    PageReason,
    PageSample,
    PageStatus,
    ReasonCount,
    utc_now,
)
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import (
    InvalidJobTransitionError,
    JobNotFoundError,
    JobRepository,
    JobRunSupersededError,
)

_TENANT = "ten_" + "1" * 32
_OTHER_TENANT = "ten_" + "2" * 32
_KB = "kb_" + "1" * 32
_SOURCE = "src_" + "1" * 32
_OTHER_SOURCE = "src_" + "2" * 32
_MISSING = "job_" + "f" * 32
_ERROR = JobError(code="engine_crashed", message="boom")


class _Clock:
    def __init__(self) -> None:
        self.now = utc_now()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
async def repo(integration_settings: Settings, clock: _Clock) -> AsyncIterator[JobRepository]:
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([CrawlJob])
    yield JobRepository(mongo.db, clock=clock)
    await mongo.close()


async def _job(
    repo: JobRepository, *, tenant_id: str = _TENANT, source_id: str = _SOURCE
) -> CrawlJob:
    return await repo.insert(
        CrawlJob(
            tenant_id=tenant_id, kb_id=_KB, source_id=source_id, seed_url="https://acme.example/"
        )
    )


async def _running(repo: JobRepository, run_id: str = "run-1") -> CrawlJob:
    job = await _job(repo)
    return await repo.start(job.id, run_id)


# ---- #
# Tests
# ---- #


class TestJobRepository_S49:
    """S49 — CrawlJob repository: state machine, run guard, `$inc` counters, heartbeat."""

    # ---- happy path ---- #

    async def test_job_runs_to_success(self, repo: JobRepository, clock: _Clock) -> None:
        job = await _job(repo)
        started_at = clock.now

        running = await repo.start(job.id, "run-1")
        clock.advance(60)
        finalizing = await repo.begin_finalizing(job.id, "run-1")
        clock.advance(5)
        done = await repo.succeed(job.id, "run-1")

        assert running.status is JobStatus.RUNNING
        assert (running.run_id, running.attempts) == ("run-1", 1)
        assert running.started_at == running.heartbeat_at == started_at
        assert finalizing.status is JobStatus.FINALIZING
        assert done.status is JobStatus.SUCCEEDED
        assert done.finished_at == clock.now and done.heartbeat_at is None
        assert done.run_id == "run-1"  # kept as an audit
        assert await repo.get_or_raise(job.id) == done

    # ---- refused transitions ---- #

    async def test_start_twice_is_refused(self, repo: JobRepository) -> None:
        job = await _running(repo)

        with pytest.raises(InvalidJobTransitionError) as exc:
            await repo.start(job.id, "run-2")

        assert exc.value.details == {"id": job.id, "from": "running", "to": "running"}

    async def test_steps_cannot_be_skipped(self, repo: JobRepository) -> None:
        queued = await _job(repo)
        running = await _running(repo)

        with pytest.raises(JobRunSupersededError):  # a queued job isn't held by any run
            await repo.begin_finalizing(queued.id, "run-1")
        with pytest.raises(InvalidJobTransitionError) as exc:
            await repo.succeed(running.id, "run-1")  # running → succeeded skips finalizing
        assert exc.value.details["from"] == "running"

    async def test_finished_jobs_are_final(self, repo: JobRepository) -> None:
        job = await _running(repo)
        await repo.fail(job.id, _ERROR, run_id="run-1")

        with pytest.raises(InvalidJobTransitionError):
            await repo.start(job.id, "run-2")
        with pytest.raises(InvalidJobTransitionError):
            await repo.fail(job.id, _ERROR)
        with pytest.raises(JobRunSupersededError):
            await repo.succeed(job.id, "run-1")

    async def test_unknown_job(self, repo: JobRepository) -> None:
        with pytest.raises(JobNotFoundError):
            await repo.start(_MISSING, "run-1")
        with pytest.raises(JobNotFoundError):
            await repo.heartbeat(_MISSING, "run-1")

    async def test_run_id_is_validated_and_stored_literally(self, repo: JobRepository) -> None:
        job = await _job(repo)

        with pytest.raises(ValueError, match="run_id"):
            await repo.start(job.id, "")
        running = await repo.start(job.id, "$status")  # not read as a field path

        assert running.run_id == "$status"

    # ---- failure ---- #

    @pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FINALIZING])
    async def test_system_can_fail_any_active_job(
        self, repo: JobRepository, status: JobStatus
    ) -> None:
        job = await _job(repo)
        if status is not JobStatus.QUEUED:
            await repo.start(job.id, "run-1")
        if status is JobStatus.FINALIZING:
            await repo.begin_finalizing(job.id, "run-1")

        failed = await repo.fail(job.id, _ERROR)

        assert failed.status is JobStatus.FAILED
        assert failed.error == _ERROR and failed.finished_at is not None

    # ---- run ownership ---- #

    async def test_another_run_is_superseded(self, repo: JobRepository) -> None:
        job = await _running(repo, "run-1")

        for attempt in (
            repo.heartbeat(job.id, "run-2"),
            repo.increment(job.id, "run-2", fetched=1),
            repo.begin_finalizing(job.id, "run-2"),
            repo.cancel(job.id, "run-2"),
            repo.fail(job.id, _ERROR, run_id="run-2"),
        ):
            with pytest.raises(JobRunSupersededError):
                await attempt
        assert (await repo.get_or_raise(job.id)).status is JobStatus.RUNNING

    async def test_concurrent_starts_have_one_winner(self, repo: JobRepository) -> None:
        job = await _job(repo)

        results = await asyncio.gather(
            *(repo.start(job.id, f"run-{n}") for n in range(10)), return_exceptions=True
        )

        winners = [r for r in results if isinstance(r, CrawlJob)]
        assert len(winners) == 1
        assert all(isinstance(r, InvalidJobTransitionError) for r in results if r not in winners)
        stored = await repo.get_or_raise(job.id)
        assert (stored.run_id, stored.attempts) == (winners[0].run_id, 1)

    # ---- crash recovery (Step 80 uses these) ---- #

    async def test_requeue_cuts_off_the_dead_run(self, repo: JobRepository, clock: _Clock) -> None:
        job = await _running(repo, "run-dead")
        first_start = clock.now
        clock.advance(600)

        requeued = await repo.requeue(job.id, stale_run_id="run-dead")
        again = await repo.start(job.id, "run-new")

        assert requeued.status is JobStatus.QUEUED
        assert requeued.run_id is None and requeued.heartbeat_at is None
        assert (again.attempts, again.started_at, again.heartbeat_at) == (2, first_start, clock.now)
        with pytest.raises(JobRunSupersededError):
            await repo.heartbeat(job.id, "run-dead")
        with pytest.raises(JobRunSupersededError):  # already picked up: not requeued twice
            await repo.requeue(job.id, stale_run_id="run-dead")

    async def test_requeue_while_finalizing(self, repo: JobRepository) -> None:
        job = await _running(repo)
        await repo.begin_finalizing(job.id, "run-1")

        requeued = await repo.requeue(job.id, stale_run_id="run-1")

        assert requeued.status is JobStatus.QUEUED

    async def test_requeue_cancels_a_job_whose_cancel_was_pending(
        self, repo: JobRepository
    ) -> None:
        job = await _running(repo, "run-dead")
        await repo.request_cancel(_TENANT, job.id)

        result = await repo.requeue(job.id, stale_run_id="run-dead")

        assert result.status is JobStatus.CANCELLED and result.finished_at is not None

    async def test_find_stale(self, repo: JobRepository, clock: _Clock) -> None:
        old = await _running(repo, "run-old")
        clock.advance(300)
        fresh = await _running(repo, "run-fresh")
        await _job(repo)  # queued: never stale

        stale = await repo.find_stale(clock.now - timedelta(seconds=120))
        await repo.heartbeat(old.id, "run-old")
        after_beat = await repo.find_stale(clock.now - timedelta(seconds=120))

        assert [j.id for j in stale] == [old.id]
        assert after_beat == [] and fresh.id not in {j.id for j in stale}

    # ---- cancellation ---- #

    async def test_cancel_a_queued_job_immediately(self, repo: JobRepository) -> None:
        job = await _job(repo)

        cancelled = await repo.request_cancel(_TENANT, job.id)
        repeat = await repo.request_cancel(_TENANT, job.id)

        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.cancel_requested_at == cancelled.finished_at is not None
        assert repeat == cancelled  # idempotent

    async def test_cancel_a_running_job_through_its_worker(
        self, repo: JobRepository, clock: _Clock
    ) -> None:
        job = await _running(repo)
        requested_at = clock.now

        flagged = await repo.request_cancel(_TENANT, job.id)
        clock.advance(10)
        await repo.request_cancel(_TENANT, job.id)  # a second request keeps the first time
        seen = await repo.heartbeat(job.id, "run-1")
        cancelled = await repo.cancel(job.id, "run-1")

        assert flagged.status is JobStatus.RUNNING
        assert seen.cancel_requested_at == requested_at
        assert cancelled.status is JobStatus.CANCELLED and cancelled.finished_at == clock.now

    async def test_finalizing_or_finished_jobs_cannot_be_cancelled(
        self, repo: JobRepository
    ) -> None:
        finalizing = await _running(repo)
        await repo.begin_finalizing(finalizing.id, "run-1")
        done = await _running(repo, "run-2")
        await repo.begin_finalizing(done.id, "run-2")
        await repo.succeed(done.id, "run-2")

        for job_id, status in ((finalizing.id, "finalizing"), (done.id, "succeeded")):
            with pytest.raises(InvalidJobTransitionError) as exc:
                await repo.request_cancel(_TENANT, job_id)
            assert exc.value.details["from"] == status

    async def test_cancel_is_tenant_scoped(self, repo: JobRepository) -> None:
        job = await _job(repo)

        with pytest.raises(JobNotFoundError):
            await repo.request_cancel(_OTHER_TENANT, job.id)
        assert (await repo.get_or_raise(job.id)).status is JobStatus.QUEUED

    # ---- counters and heartbeat ---- #

    async def test_concurrent_increments_are_exact(self, repo: JobRepository) -> None:
        job = await _running(repo)

        await asyncio.gather(
            *(repo.increment(job.id, "run-1", discovered=3, fetched=1) for _ in range(50)),
            *(repo.increment(job.id, "run-1", failed=1, skipped=2) for _ in range(20)),
            *(repo.increment(job.id, "run-1", chunked=5, embedded=5) for _ in range(30)),
        )

        counters = (await repo.get_or_raise(job.id)).counters
        assert counters == JobCounters(
            discovered=150, fetched=50, failed=20, skipped=40, chunked=150, embedded=150
        )

    async def test_increment_validation(self, repo: JobRepository) -> None:
        job = await _running(repo)

        with pytest.raises(ValueError, match="unknown"):
            await repo.increment(job.id, "run-1", bogus=1)
        with pytest.raises(ValueError, match=">= 0"):
            await repo.increment(job.id, "run-1", fetched=-1)
        unchanged = await repo.increment(job.id, "run-1", fetched=0)

        assert unchanged == JobCounters()

    async def test_counters_frozen_once_finished(self, repo: JobRepository) -> None:
        job = await _running(repo)
        await repo.increment(job.id, "run-1", fetched=2)
        await repo.cancel(job.id, "run-1")

        with pytest.raises(JobRunSupersededError):
            await repo.increment(job.id, "run-1", fetched=1)
        assert (await repo.get_or_raise(job.id)).counters.fetched == 2

    async def test_counters_keep_counting_while_finalizing(self, repo: JobRepository) -> None:
        job = await _running(repo)
        await repo.begin_finalizing(job.id, "run-1")

        counters = await repo.increment(job.id, "run-1", embedded=4)

        assert counters.embedded == 4

    async def test_set_counters_and_usage(self, repo: JobRepository) -> None:
        job = await _running(repo)
        await repo.increment(job.id, "run-1", fetched=7)

        reconciled = await repo.set_counters(job.id, "run-1", JobCounters(fetched=5, failed=1))
        await repo.add_usage(job.id, "run-1", requests=2, tokens=900)
        await repo.add_usage(job.id, "run-1", requests=1, tokens=100)

        stored = await repo.get_or_raise(job.id)
        assert reconciled.counters == JobCounters(fetched=5, failed=1)
        assert (stored.usage.requests, stored.usage.tokens) == (3, 1000)
        with pytest.raises(ValueError):
            await repo.add_usage(job.id, "run-1", requests=-1, tokens=0)

    async def test_heartbeat_moves_forward(self, repo: JobRepository, clock: _Clock) -> None:
        job = await _running(repo)
        clock.advance(30)

        beaten = await repo.heartbeat(job.id, "run-1")

        assert beaten.heartbeat_at == clock.now
        assert beaten.started_at == clock.now - timedelta(seconds=30)
        assert beaten.cancel_requested_at is None

    # ---- reads ---- #

    async def test_tenant_scoped_reads(self, repo: JobRepository) -> None:
        mine = [await _job(repo) for _ in range(3)]
        other_source = await _job(repo, source_id=_OTHER_SOURCE)
        theirs = await _job(repo, tenant_id=_OTHER_TENANT)
        await repo.start(mine[0].id, "run-1")

        everything = await repo.list_for_tenant(_TENANT, limit=10)
        running = await repo.list_for_tenant(_TENANT, status=JobStatus.RUNNING)
        by_source = await repo.list_for_tenant(_TENANT, source_id=_OTHER_SOURCE)
        first_page = await repo.list_for_tenant(_TENANT, limit=2)
        second_page = await repo.list_for_tenant(_TENANT, limit=2, cursor=first_page.next_cursor)

        assert [j.id for j in everything.items] == [other_source.id, *[j.id for j in mine[::-1]]]
        assert [j.id for j in running.items] == [mine[0].id]
        assert [j.id for j in by_source.items] == [other_source.id]
        assert len(first_page.items) == 2 and second_page.next_cursor is None
        assert (await repo.get_for_tenant(_TENANT, mine[1].id)).id == mine[1].id
        with pytest.raises(JobNotFoundError):
            await repo.get_for_tenant(_TENANT, theirs.id)


class TestJobReport_S53:
    """S53 — Job accounting writes: the stored crawl report and counter reconciliation."""

    async def test_report_round_trips_and_is_replaced(self, repo: JobRepository) -> None:
        job = await _running(repo)
        first = CrawlReport(
            seed_url="https://acme.example/",
            stop_reason=CrawlStopReason.MAX_PAGES,
            stop_detail="budget",
            pages={PageStatus.SUCCEEDED: 3, PageStatus.FAILED: 2},
            failures=[
                ReasonCount(status=PageStatus.FAILED, reason=PageReason.TIMEOUT, count=2,
                            samples=[PageSample(url="https://acme.example/slow", detail="t")])
            ],
        )  # fmt: skip

        stored = await repo.set_report(job.id, "run-1", first)
        await repo.set_report(job.id, "run-1", CrawlReport(seed_url="https://acme.example/b"))
        reloaded = await repo.get_or_raise(job.id)

        assert stored.report == first
        assert reloaded.report is not None
        assert reloaded.report.seed_url == "https://acme.example/b"
        assert reloaded.report.failures == []

    async def test_report_is_guarded_by_the_run(self, repo: JobRepository) -> None:
        job = await _running(repo)

        with pytest.raises(JobRunSupersededError):
            await repo.set_report(job.id, "run-2", CrawlReport())

    async def test_reconcile_overwrites_only_the_named_counters(self, repo: JobRepository) -> None:
        job = await _running(repo)
        await repo.increment(job.id, "run-1", failed=9, skipped=1, embedded=4, fetched=3)

        counters = await repo.reconcile_counters(job.id, "run-1", failed=2, skipped=5)
        unchanged = await repo.reconcile_counters(job.id, "run-1")

        assert counters == JobCounters(failed=2, skipped=5, embedded=4, fetched=3)
        assert unchanged == counters
        with pytest.raises(ValueError, match="unknown"):
            await repo.reconcile_counters(job.id, "run-1", nope=1)
        with pytest.raises(ValueError, match=">= 0"):
            await repo.reconcile_counters(job.id, "run-1", failed=-1)
        with pytest.raises(JobRunSupersededError):
            await repo.reconcile_counters(job.id, "run-2", failed=0)
