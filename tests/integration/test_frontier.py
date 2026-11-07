"""Frontier in real MongoDB: dedupe, priority order, atomic claims, retries, crash recovery."""

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import (
    Frontier,
    FrontierEntry,
    FrontierJob,
    SkippedEntry,
    frontier_priority,
    normalize_and_hash,
)
from rag_engine.domain import ContentKind, DiscoveredVia, Page, PageReason, PageStatus, utc_now
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import PageOutcome, PageRepository
from tests.conftest import OverrideSettings

_JOB = FrontierJob(
    tenant_id="ten_" + "1" * 32,
    kb_id="kb_" + "1" * 32,
    source_id="src_" + "1" * 32,
    job_id="job_" + "1" * 32,
)
_OTHER_JOB = FrontierJob(_JOB.tenant_id, _JOB.kb_id, _JOB.source_id, "job_" + "2" * 32)
_SITE = "https://acme.example"
_LEASE = 600
_MAX_ATTEMPTS = 3  # CRAWL_MAX_RETRIES=2


class _Clock:
    """Controllable `utc_now` replacement shared by every frontier in a test."""

    def __init__(self) -> None:
        self.now = utc_now()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
async def pages(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[PageRepository]:
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Page])
    yield PageRepository(mongo.db)
    await mongo.close()


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def make_frontier(
    pages: PageRepository, clock: _Clock, override_settings: OverrideSettings
) -> Callable[..., Frontier]:
    config = override_settings(CRAWL_FRONTIER_LEASE_SECONDS=_LEASE, CRAWL_MAX_RETRIES=2)

    def _make(run_id: str = "run-1", job: FrontierJob = _JOB) -> Frontier:
        return Frontier(pages, job, run_id=run_id, config=config, clock=clock)

    return _make


def _link(path: str, depth: int = 1, via: DiscoveredVia = DiscoveredVia.LINK) -> FrontierEntry:
    return FrontierEntry(url=f"{_SITE}{path}", depth=depth, via=via, discovered_from=f"{_SITE}/")


async def _drain(frontier: Frontier) -> list[Page]:
    claimed = []
    while (page := await frontier.claim()) is not None:
        claimed.append(page)
    return claimed


def _succeeded() -> PageOutcome:
    return PageOutcome(
        status=PageStatus.SUCCEEDED, content_kind=ContentKind.HTML, markdown="# Done\n"
    )


# ---- #
# Tests
# ---- #


class TestFrontier_S48:
    """S48 — Frontier: unique-index dedupe, sitemap/shallow-first claims, leases, recovery."""

    # ---- enqueue and dedupe ---- #

    async def test_add_queues_a_normalized_page_once(
        self, make_frontier, pages: PageRepository
    ) -> None:
        frontier = make_frontier()

        first = await frontier.add(
            "HTTPS://ACME.example:443/docs#intro", depth=2, via=DiscoveredVia.LINK
        )
        again = await frontier.add(f"{_SITE}/docs", depth=2, via=DiscoveredVia.LINK)

        url, url_hash = normalize_and_hash(f"{_SITE}/docs")
        stored = await pages.find({"job_id": _JOB.job_id})
        assert (first, again) == (True, False)
        assert len(stored) == 1
        page = stored[0]
        assert (page.url, page.url_hash, page.status) == (url, url_hash, PageStatus.QUEUED)
        assert page.priority == frontier_priority(DiscoveredVia.LINK, 2)
        assert (page.tenant_id, page.kb_id, page.source_id) == (
            _JOB.tenant_id,
            _JOB.kb_id,
            _JOB.source_id,
        )

    async def test_add_many_counts_new_urls_and_keeps_the_best_duplicate(
        self, make_frontier, pages: PageRepository
    ) -> None:
        frontier = make_frontier()
        await frontier.add(f"{_SITE}/known", depth=1, via=DiscoveredVia.LINK)

        inserted = await frontier.add_many(
            [
                _link("/known"),
                _link("/a", depth=4),
                _link("/a", depth=2),  # same URL again in the batch, shallower
                _link("/b"),
                FrontierEntry(url="not a url", depth=1, via=DiscoveredVia.LINK),
            ]
        )

        assert inserted == 2
        a = await pages.find_one({"url": f"{_SITE}/a"})
        assert a is not None and a.depth == 2
        assert await pages.count({"job_id": _JOB.job_id}) == 3

    async def test_concurrent_adds_of_one_url_create_one_page(
        self, make_frontier, pages: PageRepository
    ) -> None:
        frontiers = [make_frontier(f"run-{n}") for n in range(20)]

        created = await asyncio.gather(
            *(f.add(f"{_SITE}/same", depth=1, via=DiscoveredVia.LINK) for f in frontiers)
        )

        assert created.count(True) == 1
        assert await pages.count({"job_id": _JOB.job_id}) == 1

    async def test_rediscovery_promotes_a_queued_page(
        self, make_frontier, pages: PageRepository
    ) -> None:
        frontier = make_frontier()
        await frontier.add(f"{_SITE}/deep", depth=5, via=DiscoveredVia.LINK)

        await frontier.add(f"{_SITE}/deep", depth=7, via=DiscoveredVia.LINK)  # worse: ignored
        mid = await pages.find_one({"url": f"{_SITE}/deep"})
        await frontier.add_many([_link("/deep", depth=2)])  # better: promoted
        promoted = await pages.find_one({"url": f"{_SITE}/deep"})

        assert mid is not None and mid.depth == 5
        assert promoted is not None
        assert (promoted.depth, promoted.priority) == (2, frontier_priority(DiscoveredVia.LINK, 2))

    async def test_claimed_or_finished_pages_are_never_promoted(
        self, make_frontier, pages: PageRepository
    ) -> None:
        frontier = make_frontier()
        await frontier.add(f"{_SITE}/held", depth=4, via=DiscoveredVia.LINK)
        await frontier.add(f"{_SITE}/done", depth=4, via=DiscoveredVia.LINK)
        held = await frontier.claim()
        done = await frontier.claim()
        assert held is not None and done is not None
        await pages.record_outcome(done.id, _succeeded())

        await frontier.add_many([_link("/held", depth=1), _link("/done", depth=1)])

        for page_id, status in ((held.id, PageStatus.IN_PROGRESS), (done.id, PageStatus.SUCCEEDED)):
            page = await pages.get_or_raise(page_id)
            assert (page.status, page.depth) == (status, 4)

    async def test_jobs_are_isolated(self, make_frontier, pages: PageRepository) -> None:
        mine, theirs = make_frontier(), make_frontier(job=_OTHER_JOB)

        await mine.add(f"{_SITE}/x", depth=1, via=DiscoveredVia.LINK)
        created_in_other_job = await theirs.add(f"{_SITE}/x", depth=1, via=DiscoveredVia.LINK)
        claimed = await _drain(theirs)

        assert created_in_other_job is True
        assert [p.job_id for p in claimed] == [_OTHER_JOB.job_id]
        assert (await mine.stats()).ready == 1

    # ---- claim order ---- #

    async def test_claims_are_sitemap_and_shallow_first(self, make_frontier) -> None:
        # Arrange: discovered in an order unrelated to priority
        frontier = make_frontier()
        await frontier.add_many(
            [
                _link("/l2-first", depth=2),
                _link("/l1-first", depth=1),
                _link("/c1", depth=1, via=DiscoveredVia.CANONICAL),
                _link("/l1-second", depth=1),
            ]
        )
        await frontier.add_many(
            [_link(f"/sitemap-{n}", depth=0, via=DiscoveredVia.SITEMAP) for n in range(3)]
        )
        await frontier.add(f"{_SITE}/", depth=0, via=DiscoveredVia.SEED)

        # Act
        order = [p.url.removeprefix(_SITE) for p in await _drain(frontier)]

        # Assert: depth, then seed < sitemap < canonical < link, then discovery order
        assert order == [
            "/",
            "/sitemap-0",
            "/sitemap-1",
            "/sitemap-2",
            "/c1",
            "/l1-first",
            "/l1-second",
            "/l2-first",
        ]

    async def test_claim_leases_the_page_to_the_run(self, make_frontier, clock: _Clock) -> None:
        frontier = make_frontier("run-a")
        await frontier.add(f"{_SITE}/p", depth=1, via=DiscoveredVia.LINK)

        page = await frontier.claim()

        assert page is not None
        assert page.status is PageStatus.IN_PROGRESS
        assert (page.claimed_by, page.attempts) == ("run-a", 1)
        assert page.lease_expires_at == clock.now + timedelta(seconds=_LEASE)
        assert await frontier.claim() is None

    async def test_concurrent_claims_never_share_a_page(self, make_frontier) -> None:
        await make_frontier().add_many([_link(f"/p{n}") for n in range(40)])
        frontiers = [make_frontier(f"run-{n}") for n in range(8)]

        claimed = await asyncio.gather(*(_drain(f) for f in frontiers))

        urls = [page.url for batch in claimed for page in batch]
        assert len(urls) == 40 and len(set(urls)) == 40
        assert sum(1 for batch in claimed if batch) > 1  # really concurrent, not one winner

    # ---- retries and release ---- #

    async def test_retry_waits_for_its_time(self, make_frontier, clock: _Clock) -> None:
        frontier = make_frontier()
        await frontier.add(f"{_SITE}/flaky", depth=1, via=DiscoveredVia.LINK)
        page = await frontier.claim()
        assert page is not None

        assert await frontier.retry_later(page.id, 30) is True
        stats = await frontier.stats()
        too_early = await frontier.claim()
        clock.advance(30)
        retried = await frontier.claim()

        assert (stats.ready, stats.delayed, stats.in_flight) == (0, 1, 0)
        assert stats.next_retry_at == clock.now
        assert not stats.drained
        assert too_early is None
        assert retried is not None and retried.id == page.id
        assert (retried.attempts, retried.retry_at) == (2, None)

    async def test_retry_keeps_the_queue_moving(self, make_frontier) -> None:
        frontier = make_frontier()
        await frontier.add_many([_link("/slow"), _link("/next")])
        slow = await frontier.claim()
        assert slow is not None

        await frontier.retry_later(slow.id, 60)
        nxt = await frontier.claim()

        assert nxt is not None and nxt.url == f"{_SITE}/next"

    async def test_only_the_holding_run_can_give_a_page_back(self, make_frontier) -> None:
        holder, other = make_frontier("run-a"), make_frontier("run-b")
        await holder.add(f"{_SITE}/p", depth=1, via=DiscoveredVia.LINK)
        page = await holder.claim()
        assert page is not None

        assert await other.retry_later(page.id, 0) is False
        assert await other.release(page.id) is False
        assert (await holder.stats()).in_flight == 1

    async def test_release_refunds_the_attempt(self, make_frontier) -> None:
        frontier = make_frontier()
        await frontier.add(f"{_SITE}/p", depth=1, via=DiscoveredVia.LINK)
        page = await frontier.claim()
        assert page is not None

        assert await frontier.release(page.id) is True
        again = await frontier.claim()

        assert again is not None and again.attempts == 1

    async def test_negative_retry_delay_rejected(self, make_frontier) -> None:
        with pytest.raises(ValueError, match="delay_seconds"):
            await make_frontier().retry_later("page_" + "1" * 32, -1)

    # ---- leases and crash recovery ---- #

    async def test_expired_lease_is_reclaimed_once_nothing_else_is_ready(
        self, make_frontier, clock: _Clock
    ) -> None:
        dead, alive = make_frontier("run-dead"), make_frontier("run-alive")
        await dead.add_many([_link("/orphan"), _link("/fresh")])
        orphan = await dead.claim()
        assert orphan is not None

        clock.advance(_LEASE - 1)
        before_expiry = [p.url for p in await _drain(alive)]
        clock.advance(1)
        reclaimed = await alive.claim()

        assert before_expiry == [f"{_SITE}/fresh"]
        assert reclaimed is not None and reclaimed.id == orphan.id
        assert (reclaimed.claimed_by, reclaimed.attempts) == ("run-alive", 2)
        assert await dead.retry_later(orphan.id, 0) is False  # the dead run lost it

    async def test_page_out_of_attempts_fails_as_worker_lost(
        self, make_frontier, clock: _Clock, pages: PageRepository
    ) -> None:
        await make_frontier().add(f"{_SITE}/poison", depth=1, via=DiscoveredVia.LINK)

        claims = []
        for n in range(_MAX_ATTEMPTS + 1):  # each crawler "crashes" holding the page
            claims.append(await make_frontier(f"run-{n}").claim())
            clock.advance(_LEASE)

        page = await pages.find_one({"url": f"{_SITE}/poison"})
        assert [c.attempts for c in claims if c] == [1, 2, 3]
        assert claims[-1] is None
        assert page is not None
        assert (page.status, page.reason) == (PageStatus.FAILED, PageReason.WORKER_LOST)
        assert page.reason_detail is not None and "3 attempt" in page.reason_detail
        assert Page.model_validate(page.to_mongo())  # consistent with the Page rules

    async def test_recover_requeues_pages_held_by_a_dead_run(
        self, make_frontier, pages: PageRepository
    ) -> None:
        crashed = make_frontier("run-1")
        await crashed.add_many([_link(f"/p{n}") for n in range(3)])
        assert len(await _drain(crashed)) == 3

        resumed = make_frontier("run-2")
        report = await resumed.recover()
        own = await resumed.claim()
        again = await resumed.recover()  # pages it holds itself are left alone

        assert (report.requeued, report.abandoned) == (3, 0)
        assert own is not None and own.attempts == 2
        assert (again.requeued, again.abandoned) == (0, 0)
        stored = await pages.find({"job_id": _JOB.job_id})
        assert Counter(p.status for p in stored) == {
            PageStatus.QUEUED: 2,
            PageStatus.IN_PROGRESS: 1,
        }

    async def test_recover_abandons_pages_out_of_attempts(
        self, make_frontier, pages: PageRepository
    ) -> None:
        await make_frontier().add_many([_link("/poison"), _link("/ok")])
        for n in range(_MAX_ATTEMPTS):  # the poison page crashes every run
            frontier = make_frontier(f"run-{n}")
            await frontier.recover()
            page = await frontier.claim()
            assert page is not None and page.url == f"{_SITE}/poison"

        report = await make_frontier("run-last").recover()

        poison = await pages.find_one({"url": f"{_SITE}/poison"})
        assert (report.requeued, report.abandoned) == (0, 1)
        assert poison is not None and poison.reason is PageReason.WORKER_LOST

    async def test_crawl_resumes_after_a_crash_without_redoing_finished_pages(
        self, make_frontier, pages: PageRepository
    ) -> None:
        # Arrange: run-1 finishes 2 pages, then crashes holding 2 more
        first = make_frontier("run-1")
        await first.add(f"{_SITE}/", depth=0, via=DiscoveredVia.SEED)
        await first.add_many([_link(f"/p{n}") for n in range(9)])
        finished = [await first.claim(), await first.claim()]
        for page in finished:
            assert page is not None
            await pages.record_outcome(page.id, _succeeded())
        in_flight = [await first.claim(), await first.claim()]

        # Act: run-2 resumes the job and crawls to the end
        second = make_frontier("run-2")
        report = await second.recover()
        resumed = await _drain(second)
        for page in resumed:
            await pages.record_outcome(page.id, _succeeded())

        # Assert
        assert report.requeued == 2
        assert {p.id for p in resumed}.isdisjoint(p.id for p in finished if p)
        assert {p.id for p in in_flight if p} <= {p.id for p in resumed}
        assert len(resumed) == 8
        stats = await second.stats()
        assert stats.drained and stats.next_retry_at is None
        stored = await pages.find({"job_id": _JOB.job_id})
        assert {p.status for p in stored} == {PageStatus.SUCCEEDED}
        assert all(p.lease_expires_at is None for p in stored)  # outcomes cleared the lease

    # ---- schema ---- #

    async def test_claim_index_exists(self, pages: PageRepository) -> None:
        indexes = {i["name"]: i async for i in await pages.collection.list_indexes()}

        assert list(indexes["job_frontier"]["key"]) == ["job_id", "status", "priority", "_id"]
        assert indexes["job_url_unique"]["unique"] is True


class TestFrontierEngineSupport_S50:
    """S50 — Frontier additions for the engine: claim a given URL, delayed release, skips."""

    async def test_claim_url_takes_that_queued_page(self, make_frontier) -> None:
        frontier = make_frontier()
        await frontier.add_many([_link("/first"), _link("/target")])

        target = await frontier.claim_url(f"{_SITE}/target#section")
        again = await frontier.claim_url(f"{_SITE}/target")
        unknown = await frontier.claim_url(f"{_SITE}/nope")
        invalid = await frontier.claim_url("not a url")

        assert target is not None and target.url == f"{_SITE}/target"
        assert (target.status, target.claimed_by, target.attempts) == (
            PageStatus.IN_PROGRESS,
            "run-1",
            1,
        )
        assert again is unknown is invalid is None

    async def test_release_with_a_delay_refunds_and_waits(
        self, make_frontier, clock: _Clock
    ) -> None:
        frontier = make_frontier()
        await frontier.add(f"{_SITE}/busy-host", depth=1, via=DiscoveredVia.LINK)
        page = await frontier.claim()
        assert page is not None

        assert await frontier.release(page.id, delay_seconds=2) is True
        too_early = await frontier.claim()
        clock.advance(2)
        later = await frontier.claim()

        assert too_early is None
        assert later is not None and later.attempts == 1  # the first claim was refunded
        with pytest.raises(ValueError, match="delay_seconds"):
            await frontier.release(page.id, delay_seconds=-1)

    async def test_add_skipped_never_overrides_a_known_url(
        self, make_frontier, pages: PageRepository
    ) -> None:
        frontier = make_frontier()
        await frontier.add(f"{_SITE}/reached", depth=1, via=DiscoveredVia.LINK)

        recorded = await frontier.add_skipped(
            [
                SkippedEntry(url=f"{_SITE}/reached", depth=6, reason=PageReason.MAX_DEPTH),
                SkippedEntry(url=f"{_SITE}/held", depth=2, reason=PageReason.NOFOLLOW,
                             detail="rel=nofollow", discovered_from=f"{_SITE}/"),
                SkippedEntry(url=f"{_SITE}/held", depth=3, reason=PageReason.NOFOLLOW),
                SkippedEntry(url="mailto:x@acme.example", depth=1, reason=PageReason.OUT_OF_SCOPE),
            ]
        )  # fmt: skip

        reached = await pages.find_one({"url": f"{_SITE}/reached"})
        held = await pages.find_one({"url": f"{_SITE}/held"})
        assert recorded == 1
        assert reached is not None and reached.status is PageStatus.QUEUED
        assert held is not None
        assert (held.status, held.reason, held.reason_detail, held.depth) == (
            PageStatus.SKIPPED,
            PageReason.NOFOLLOW,
            "rel=nofollow",
            2,
        )
        assert (await frontier.stats()).ready == 1  # skipped rows are never claimable


class TestFrontierCaps_S51:
    """S51 — Frontier support for caps: charge the page budget, count it, skip the queue."""

    async def test_charge_marks_only_a_page_this_run_holds(self, make_frontier) -> None:
        frontier = make_frontier()
        other_run = make_frontier("run-2")
        await frontier.add_many([_link("/a"), _link("/b")])
        held = await frontier.claim()
        elsewhere = await other_run.claim()
        assert held is not None and elsewhere is not None

        charged = await frontier.charge(held.id)
        not_ours = await frontier.charge(elsewhere.id)
        await frontier.release(held.id)
        reclaimed = await frontier.claim()

        assert (charged, not_ours) == (True, False)
        assert await frontier.count_charged() == 1
        assert reclaimed is not None and reclaimed.id == held.id and reclaimed.charged is True

    async def test_count_charged_is_per_job(self, make_frontier) -> None:
        frontier = make_frontier()
        other_job = make_frontier(job=_OTHER_JOB)
        await frontier.add(f"{_SITE}/a", depth=1, via=DiscoveredVia.LINK)
        await other_job.add(f"{_SITE}/a", depth=1, via=DiscoveredVia.LINK)
        page = await other_job.claim()
        assert page is not None
        await other_job.charge(page.id)

        assert await frontier.count_charged() == 0
        assert await other_job.count_charged() == 1

    async def test_skip_queued_records_what_is_left(
        self, make_frontier, pages: PageRepository
    ) -> None:
        frontier = make_frontier()
        await frontier.add_many([_link(f"/p{n}") for n in range(5)])
        charged = await frontier.claim()  # /p0: charged, then waiting for a retry
        held = await frontier.claim()  # /p1: in flight
        assert charged is not None and held is not None
        await frontier.charge(charged.id)
        await frontier.retry_later(charged.id, 60)

        soft = await frontier.skip_queued(PageReason.MAX_PAGES, "budget", uncharged_only=True)
        retry_left = await pages.find_one({"_id": charged.id})
        hard = await frontier.skip_queued(PageReason.MAX_DURATION, "x" * 600)

        by_url = {p.url: p for p in await pages.find({"job_id": _JOB.job_id})}
        assert (soft, hard) == (3, 1)
        assert retry_left is not None and retry_left.status is PageStatus.QUEUED
        assert by_url[f"{_SITE}/p1"].status is PageStatus.IN_PROGRESS  # in flight: untouched
        assert {by_url[f"{_SITE}/p{n}"].reason for n in (2, 3, 4)} == {PageReason.MAX_PAGES}
        retried = by_url[f"{_SITE}/p0"]
        assert (retried.status, retried.reason, retried.retry_at) == (
            PageStatus.SKIPPED,
            PageReason.MAX_DURATION,
            None,
        )
        assert retried.reason_detail is not None and len(retried.reason_detail) == 500
        assert (await frontier.stats()).ready == 0
