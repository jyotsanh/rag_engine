"""Crash recovery in real MongoDB: stale held jobs are requeued, failed, cancelled or
completed; lost queued jobs are sent again. The clock is controlled."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import Frontier, FrontierJob
from rag_engine.domain import (
    ContentHashClaim,
    ContentKind,
    CrawlJob,
    DiscoveredVia,
    JobError,
    JobStatus,
    Page,
    PageReason,
    PageStatus,
    Source,
    SourceStats,
    new_id,
    utc_now,
)
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import (
    ContentClaimRepository,
    InvalidJobTransitionError,
    JobRepository,
    JobRunSupersededError,
    PageRepository,
)
from rag_engine.repositories.sources import SourceRepository
from rag_engine.worker.recovery import WORKER_LOST, JobRecovery, RecoveryReport
from tests.conftest import OverrideSettings

_TENANT = "ten_" + "1" * 32
_KB = "kb_" + "1" * 32
_STALE = 300.0  # CRAWL_STALE_HEARTBEAT_SECONDS
_QUEUED_STALE = 1800.0  # CRAWL_QUEUED_STALE_SECONDS


class _Clock:
    def __init__(self) -> None:
        self.now = utc_now()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class _Env:
    mongo: MongoConnection
    clock: _Clock
    config: Settings
    enqueued: list[tuple[str, float]] = field(default_factory=list)
    cleanups: list[tuple[str, str, str, str, float]] = field(default_factory=list)

    @property
    def jobs(self) -> JobRepository:
        return JobRepository(self.mongo.db, clock=self.clock)

    @property
    def pages(self) -> PageRepository:
        return PageRepository(self.mongo.db, self.config)

    @property
    def sources(self) -> SourceRepository:
        return SourceRepository(self.mongo.db)

    def recovery(self) -> JobRecovery:
        return JobRecovery(
            self.mongo.db,
            enqueue=lambda job_id, delay: self.enqueued.append((job_id, delay)),
            enqueue_cleanup=lambda *request: self.cleanups.append(request),
            config=self.config,
            clock=self.clock,
        )

    async def recover(self) -> RecoveryReport:
        return await self.recovery().run()

    async def source(self) -> Source:
        source, _ = await self.sources.get_or_create(
            _TENANT, _KB, f"https://{new_id('s')[-12:]}.example/"
        )
        return source

    async def queued(self) -> CrawlJob:
        source = await self.source()
        return await self.jobs.insert(
            CrawlJob(tenant_id=_TENANT, kb_id=_KB, source_id=source.id, seed_url=source.seed_url)
        )

    async def running(self, run_id: str = "run-dead", *, attempts: int = 1) -> CrawlJob:
        """A job started `attempts` times, held by `run_id` (earlier runs requeued)."""
        job = await self.queued()
        for n in range(1, attempts):
            await self.jobs.start(job.id, f"run-old-{n}")
            await self.jobs.requeue(job.id, stale_run_id=f"run-old-{n}")
        return await self.jobs.start(job.id, run_id)


@pytest.fixture
async def env(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[_Env]:
    config = override_settings(
        CRAWL_HEARTBEAT_SECONDS=30,
        CRAWL_STALE_HEARTBEAT_SECONDS=_STALE,
        CRAWL_QUEUED_STALE_SECONDS=_QUEUED_STALE,
        CRAWL_JOB_MAX_ATTEMPTS=3,
    )
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([CrawlJob, Page, Source, ContentHashClaim])
    yield _Env(mongo, _Clock(), config)
    await mongo.close()


async def _crawled_pages(env: _Env, job: CrawlJob) -> list[Page]:
    """The dead run's frontier: one page finished, one in flight, one never claimed."""
    frontier = Frontier(
        env.pages,
        FrontierJob(job.tenant_id, job.kb_id, job.source_id, job.id),
        run_id=job.run_id or "run-dead",
        config=env.config,
    )
    for n in range(3):
        await frontier.add(f"{job.seed_url}p{n}", depth=0, via=DiscoveredVia.SEED)
    done, in_flight = await frontier.claim(), await frontier.claim()
    assert done is not None and in_flight is not None
    await env.pages.collection.update_one(
        {"_id": done.id},
        {
            "$set": {
                "status": PageStatus.SUCCEEDED.value,
                "content_kind": ContentKind.HTML.value,
                "markdown": "# Done",
                "lease_expires_at": None,
            }
        },
    )
    return [done, in_flight]


# ---- #
# Tests
# ---- #


class TestJobRecovery_S80:
    """S80 — `JobRecovery`: stale running/finalizing jobs → requeue up to N attempts, then
    `failed/worker_lost`; pending cancels; finalized-before-death; lost queued jobs."""

    async def test_a_live_job_is_left_alone(self, env: _Env) -> None:
        job = await env.running()
        env.clock.advance(_STALE - 1)

        report = await env.recover()

        assert report == RecoveryReport()
        assert (await env.jobs.get_or_raise(job.id)).status is JobStatus.RUNNING
        assert env.enqueued == []

    async def test_a_stale_job_is_requeued_and_sent_again(self, env: _Env) -> None:
        job = await env.running()
        env.clock.advance(_STALE + 1)

        report = await env.recover()

        current = await env.jobs.get_or_raise(job.id)
        assert report.requeued == [job.id]
        assert current.status is JobStatus.QUEUED
        assert (current.run_id, current.heartbeat_at, current.attempts) == (None, None, 1)
        assert env.enqueued == [(job.id, 0)]
        assert env.cleanups == []  # a requeued job keeps its data for the next run
        # the dead run is cut off, and the next run resumes as attempt 2
        with pytest.raises(JobRunSupersededError):
            await env.jobs.heartbeat(job.id, "run-dead")
        restarted = await env.jobs.start(job.id, "run-new")
        assert restarted.attempts == 2

    async def test_a_stale_finalizing_job_that_did_not_flip_is_requeued(self, env: _Env) -> None:
        job = await env.running()
        await env.jobs.begin_finalizing(job.id, "run-dead")
        env.clock.advance(_STALE + 1)

        report = await env.recover()

        assert report.requeued == [job.id]
        assert (await env.jobs.get_or_raise(job.id)).status is JobStatus.QUEUED

    async def test_a_finalizing_job_that_already_went_live_is_completed(self, env: _Env) -> None:
        # Arrange: the worker flipped the source to its version, then died before `succeed`
        job = await env.running()
        await env.jobs.begin_finalizing(job.id, "run-dead")
        await env.sources.activate_version(job.source_id, job.version, stats=SourceStats())
        await ContentClaimRepository(env.mongo.db).claim(
            job.id, "a" * 64, new_id("page"), "https://x/"
        )
        env.clock.advance(_STALE + 1)

        # Act
        report = await env.recover()

        # Assert
        current = await env.jobs.get_or_raise(job.id)
        assert report.completed == [job.id]
        assert current.status is JobStatus.SUCCEEDED and current.error is None
        assert env.enqueued == [] and env.cleanups == []  # its version is the live one
        assert await ContentClaimRepository(env.mongo.db).count({"job_id": job.id}) == 0

    async def test_out_of_attempts_the_job_fails_worker_lost(self, env: _Env) -> None:
        job = await env.running(attempts=3)
        done, in_flight = await _crawled_pages(env, job)
        await ContentClaimRepository(env.mongo.db).claim(job.id, "a" * 64, done.id, done.url)
        env.clock.advance(_STALE + 1)

        report = await env.recover()

        current = await env.jobs.get_or_raise(job.id)
        assert report.failed == [job.id]
        assert current.status is JobStatus.FAILED and current.attempts == 3
        assert current.error is not None and current.error.code == WORKER_LOST
        assert "attempt 3 of 3" in current.error.message
        assert current.report is not None  # what the dead run crawled is kept
        assert current.report.pages[PageStatus.SUCCEEDED] == 1
        outcomes = {p.url: (p.status, p.reason) for p in await env.pages.find({"job_id": job.id})}
        assert outcomes.pop(done.url) == (PageStatus.SUCCEEDED, None)
        assert outcomes.pop(in_flight.url) == (PageStatus.FAILED, PageReason.JOB_FAILED)
        assert list(outcomes.values()) == [(PageStatus.SKIPPED, PageReason.JOB_FAILED)]
        assert await ContentClaimRepository(env.mongo.db).count({"job_id": job.id}) == 0
        assert env.enqueued == []
        assert env.cleanups == [(_TENANT, _KB, job.source_id, job.version, 0)]

    async def test_a_pending_cancel_cancels_the_job(self, env: _Env) -> None:
        job = await env.running()
        _, in_flight = await _crawled_pages(env, job)
        await env.jobs.request_cancel(_TENANT, job.id)
        env.clock.advance(_STALE + 1)

        report = await env.recover()

        current = await env.jobs.get_or_raise(job.id)
        assert report.cancelled == [job.id]
        assert current.status is JobStatus.CANCELLED and current.finished_at is not None
        page = await env.pages.get_or_raise(in_flight.id)
        assert (page.status, page.reason) == (PageStatus.FAILED, PageReason.JOB_CANCELLED)
        assert env.enqueued == []
        assert env.cleanups == [(_TENANT, _KB, job.source_id, job.version, 0)]

    async def test_a_worker_that_beats_just_in_time_is_not_requeued(
        self, env: _Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the slow worker's heartbeat lands between the sweep's read and its write
        job = await env.running()
        env.clock.advance(_STALE + 1)
        recovery = env.recovery()
        find_stale = recovery.jobs.find_stale

        async def then_beat(before: datetime, *, limit: int = 100) -> list[CrawlJob]:
            stale = await find_stale(before, limit=limit)
            await env.jobs.heartbeat(job.id, "run-dead")
            return stale

        monkeypatch.setattr(recovery.jobs, "find_stale", then_beat)

        # Act
        report = await recovery.run()

        # Assert
        assert report.skipped == [job.id] and report.requeued == []
        assert (await env.jobs.get_or_raise(job.id)).status is JobStatus.RUNNING
        assert env.enqueued == []

    async def test_a_worker_that_beats_just_in_time_is_not_failed(
        self, env: _Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = await env.running(attempts=3)
        env.clock.advance(_STALE + 1)
        recovery = env.recovery()
        find_stale = recovery.jobs.find_stale

        async def then_beat(before: datetime, *, limit: int = 100) -> list[CrawlJob]:
            stale = await find_stale(before, limit=limit)
            await env.jobs.heartbeat(job.id, "run-dead")
            return stale

        monkeypatch.setattr(recovery.jobs, "find_stale", then_beat)

        report = await recovery.run()

        current = await env.jobs.get_or_raise(job.id)
        assert report.skipped == [job.id] and report.failed == []
        assert current.status is JobStatus.RUNNING and current.report is None

    async def test_guarded_transitions_refuse_a_fresh_heartbeat(self, env: _Env) -> None:
        job = await env.running()
        cutoff = env.clock.now  # the heartbeat isn't older than this

        with pytest.raises(InvalidJobTransitionError):
            await env.jobs.requeue(job.id, stale_run_id="run-dead", heartbeat_before=cutoff)
        with pytest.raises(InvalidJobTransitionError):
            await env.jobs.fail(
                job.id, JobError(code="x", message="y"), run_id="run-dead", heartbeat_before=cutoff
            )

        assert (await env.jobs.get_or_raise(job.id)).status is JobStatus.RUNNING

    async def test_concurrent_sweeps_recover_a_job_once(self, env: _Env) -> None:
        jobs = [await env.running(f"run-dead-{n}") for n in range(5)]
        env.clock.advance(_STALE + 1)

        reports = await asyncio.gather(*(env.recover() for _ in range(4)))

        requeued = [job_id for r in reports for job_id in r.requeued]
        assert sorted(requeued) == sorted(j.id for j in jobs)  # each exactly once
        assert sorted(job_id for job_id, _ in env.enqueued) == sorted(j.id for j in jobs)

    async def test_a_repeatedly_dying_job_ends_failed(self, env: _Env) -> None:
        job = await env.queued()
        outcomes = []
        for attempt in range(1, 4):
            await env.jobs.start(job.id, f"run-{attempt}")
            env.clock.advance(_STALE + 1)
            report = await env.recover()
            outcomes.append("requeued" if report.requeued else "failed" if report.failed else "?")

        assert outcomes == ["requeued", "requeued", "failed"]
        current = await env.jobs.get_or_raise(job.id)
        assert current.status is JobStatus.FAILED and current.attempts == 3

    # ---- lost queued jobs ---- #

    async def test_a_lost_queued_job_is_sent_again_once_per_window(self, env: _Env) -> None:
        lost = await env.queued()
        env.clock.advance(_QUEUED_STALE - 60)
        fresh = await env.queued()  # created recently: not lost
        await env.jobs.touch_queued(fresh.id, fresh.updated_at)
        env.clock.advance(61)

        first = await env.recover()
        again = await env.recover()  # right away: already re-sent in this window
        env.clock.advance(_QUEUED_STALE + 1)
        later = await env.recover()

        assert first.resent == [lost.id]
        assert again.resent == []
        assert sorted(later.resent) == sorted([lost.id, fresh.id])
        assert [job_id for job_id, _ in env.enqueued] == [lost.id, *later.resent]
        assert (await env.jobs.get_or_raise(lost.id)).status is JobStatus.QUEUED

    async def test_touch_queued_is_conditional(self, env: _Env) -> None:
        job = await env.queued()
        env.clock.advance(10)

        first = await env.jobs.touch_queued(job.id, job.updated_at)
        second = await env.jobs.touch_queued(job.id, job.updated_at)  # stale read: refused
        await env.jobs.start(job.id, "run-1")
        current = await env.jobs.get_or_raise(job.id)
        not_queued = await env.jobs.touch_queued(job.id, current.updated_at)

        assert (first, second, not_queued) == (True, False, False)

    async def test_requeued_jobs_waiting_on_their_backoff_are_not_resent(self, env: _Env) -> None:
        # a transient failure requeues with a countdown ≤ retry delay x attempts (< window)
        job = await env.running()
        await env.jobs.requeue(job.id, stale_run_id="run-dead")
        env.clock.advance(env.config.CRAWL_JOB_RETRY_DELAY_SECONDS * 3)

        report = await env.recover()

        assert report.resent == []
