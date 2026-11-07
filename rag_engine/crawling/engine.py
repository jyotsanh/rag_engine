"""The crawl engine: one job's crawl loop, with bounded concurrency over the frontier.

`CrawlEngine(job, run_id=..., ...).run()` crawls a job that a worker run has started
(`JobRepository.start`, Step 49) and returns a `CrawlSummary`. Job status transitions,
the heartbeat and the final `cancelled` transition belong to the caller; caps and the
cancel poll are enforced here.

**Start**
1. `frontier.recover()` re-queues pages an earlier (crashed) run held: a resumed job
   continues where it stopped.
2. The seed is queued and crawled **first, on its own**. If it redirects to another host
   (`acme.com → acme.io`), the scope is re-anchored on the final host before anything else
   is judged against it.
3. Sitemap discovery (Step 31) runs in the background; its URLs join the frontier at depth 0
   while `CRAWL_CONCURRENCY` workers drain it.

**Per page** (a worker's claimed `Page`):
scope → robots.txt → rate limit → fetch (auto: HTTP, browser render when needed) → follow
a redirect → extract → clean → links → dedupe → enqueue outlinks → persist the outcome.
- **Rate limit**: a crawler waits at most `CRAWL_RATE_LIMIT_MAX_WAIT_SECONDS` for the host's
  slot; a longer wait releases the page until then, so one slow host can't pin every slot.
- **Fetch failures** go through the retry policy (Step 28). A retry re-queues the page with a
  not-before time (no sleeping in a slot); a `Retry-After` also pauses the whole host.
- **Redirects**: the URL that serves the content owns it. The redirecting URL is recorded as
  `skipped/redirected`, and the content is processed as the target URL's page (claimed
  directly, so it isn't fetched again). Otherwise a linked target and its redirect would
  be content duplicates of each other, depending on crawl order.
- **noindex** pages are recorded `skipped/noindex`; their links are still followed.
- **Canonical cycles** (A says B, B says A) would skip both pages: a canonical whose target
  was itself skipped as `canonical_elsewhere` is ignored, so one of them keeps the content.
- Outlinks and canonical targets are enqueued **before** the outcome is written, so the
  frontier never looks drained while a page's links are still being added.
- An unexpected exception fails only that page (`internal_error`, logged).

**End**: when nothing is ready, delayed or in flight and sitemap discovery has finished.
Links that were never crawled (out of scope, too deep, nofollow) are then recorded as
`skipped`, but only if no crawlable link reached them meanwhile: the frontier's unique index
makes "followed later" win. Each page keeps its held-back links on its own row
(`deferred_links`), so the report includes links seen by an earlier, crashed run.

**Caps** (Step 51). Each one stops the crawl gracefully, with a reason
(`CrawlSummary.stop`); the first cap hit is the reason, unless a hard stop overrides a soft one.
- `max_pages` (the job's config) is a **page budget**: a page is charged once, right before
  its first fetch (`Page.charged`), so retries and rate-limit releases don't spend it again
  and a resumed run reads it back. A redirect target reuses the alias's fetch, uncharged.
  Spending the last page is a *soft* stop: queued pages not yet charged are recorded
  `skipped/max_pages` (pages found later too, without a robots.txt lookup), while charged
  pages waiting for a retry still finish. Sitemap discovery stops.
- `CRAWL_MAX_TOTAL_BYTES` over the job (`bytes_downloaded` counter, persisted) and
  `CRAWL_MAX_JOB_SECONDS` since the job's first start are *hard* stops: nothing more is
  fetched. Pages in flight finish (bounded by the page deadline); everything still queued,
  retries included, is recorded `skipped/max_total_bytes` or `skipped/max_duration`. The
  byte cap is checked after each fetch, so it can be overshot by the pages in flight.
- Each cap is re-checked when a run starts, so a resumed job stops at once if it was spent.

**Cancellation** (Step 52). While workers run, a poller checks the job's cancel signal (the
Redis flag, `RedisCancelFlags`) every `CRAWL_CANCEL_POLL_SECONDS`; `cancel()` does the same
for a caller that learned it otherwise (the heartbeat's `cancel_requested_at`). It is a hard
stop that outranks the caps: in-flight pages drain, the rest becomes `skipped/job_cancelled`,
and `summary.stop.reason is CANCELLED` tells the caller to `finish_cancelled(...)`. A job
whose cancel was requested before this run started stops without fetching. A Redis outage
during a poll is logged and polling continues.
- **Per page**, `CRAWL_PAGE_TIMEOUT_SECONDS` bounds the rate-limit wait + fetch + render (on
  top of the fetchers' own timeouts); over it the fetch fails as a retryable `timeout`.
  Extraction has its own deadline (`EXTRACTION_TIMEOUT_SECONDS`).

Counters (Step 49): `discovered` = queued pages created; `fetched` once per successful
fetch (`bytes_downloaded` with it); `failed`/`skipped`/`duplicates` once per recorded
outcome. A run that lost the job (`JobRunSupersededError`) or the database stops the whole
engine; pages it held are recovered by the next run.
"""

import asyncio
import contextlib
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Protocol
from urllib.parse import urlsplit

from loguru import logger

from rag_engine.config import Settings, settings
from rag_engine.crawling.cancellation import CancelSignal
from rag_engine.crawling.dedupe import ContentClaimStore, ContentDeduper, DedupeOutcome
from rag_engine.crawling.fetchers import (
    FetchError,
    FetchResult,
    FetchTimeoutError,
    HttpStatusError,
)
from rag_engine.crawling.frontier import (
    Frontier,
    FrontierEntry,
    FrontierJob,
    RecoveryReport,
    SkippedEntry,
)
from rag_engine.crawling.links import LinkExtractor, PageLinks
from rag_engine.crawling.rate_limit import RateLimitWaitExceededError, Reservation
from rag_engine.crawling.retry import RetryPolicy
from rag_engine.crawling.robots import RobotsRules, RobotsStatus
from rag_engine.crawling.scope import CrawlScope
from rag_engine.crawling.sitemap import SitemapReport
from rag_engine.crawling.url import UrlNormalizationError, normalize_and_hash, normalize_url
from rag_engine.domain import (
    MAX_DEFERRED_LINKS,
    ContentKind,
    CrawlJob,
    CrawlStopReason,
    DeferredLink,
    DiscoveredVia,
    JobStatus,
    Page,
    PageReason,
    PageStatus,
    utc_now,
)
from rag_engine.extraction import (
    ExtractedDocument,
    ExtractionError,
    ExtractionInput,
    ExtractorRegistry,
    TextCleaner,
    default_registry,
)
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError, RedisUnavailableError
from rag_engine.repositories import (
    JobRepository,
    JobRunSupersededError,
    PageNotFoundError,
    PageOutcome,
    PageRepository,
)

_IDLE_SECONDS = 1.0  # longest a worker sleeps before re-checking the frontier
_FLUSH_BATCH = 1000  # never-crawled links written per bulk insert at the end
_HTTP_SCHEMES = frozenset({"http", "https"})
_STATUS_COUNTER = {
    PageStatus.FAILED: "failed",
    PageStatus.SKIPPED: "skipped",
    PageStatus.DUPLICATE: "duplicates",
}


class PageFetcher(Protocol):
    """`AutoFetcher` in production (`HttpFetcher` fits too)."""

    async def fetch(self, url: str) -> FetchResult: ...


class RobotsSource(Protocol):
    async def rules_for(self, url: str) -> RobotsRules: ...


class RateLimiter(Protocol):
    async def acquire(
        self, url: str, *, crawl_delay: float | None = None, max_wait: float | None = None
    ) -> Reservation: ...

    async def pause(self, url: str, seconds: float) -> float: ...


class SitemapSource(Protocol):
    async def discover(
        self, seed_url: str, scope: CrawlScope, *, max_urls: int | None = None
    ) -> SitemapReport: ...


def render_politeness(
    robots: RobotsSource, limiter: RateLimiter
) -> Callable[[str], Awaitable[None]]:
    """`AutoFetcher(before_render=...)`: the browser's re-request books a rate-limit slot too."""

    async def before_render(url: str) -> None:
        rules = await robots.rules_for(url)
        await limiter.acquire(url, crawl_delay=rules.crawl_delay)

    return before_render


@dataclass(frozen=True, slots=True)
class CrawlStop:
    """Why the crawl stopped before its frontier drained."""

    reason: CrawlStopReason
    detail: str


@dataclass(frozen=True, slots=True)
class CrawlSummary:
    seed_url: str  # where the crawl was anchored (the seed's final URL after redirects)
    pages: dict[PageStatus, int]
    reasons: Counter[PageReason]
    recovered: RecoveryReport
    robots_status: RobotsStatus | None  # the seed origin's robots.txt verdict
    robots_detail: str | None
    sitemap: SitemapReport | None
    sitemap_error: str | None
    skipped_links: int  # never-crawled links recorded as skipped pages
    skipped_links_dropped: int  # this run: non-http(s) links, or over the per-page cap
    stop: CrawlStop | None = None  # a cap stopped the crawl (None: the frontier drained)


class CrawlEngine:
    def __init__(
        self,
        job: CrawlJob,
        *,
        run_id: str,
        pages: PageRepository,
        jobs: JobRepository,
        claims: ContentClaimStore,
        fetcher: PageFetcher,
        robots: RobotsSource,
        limiter: RateLimiter,
        sitemaps: SitemapSource | None = None,
        cancel: CancelSignal | None = None,
        registry: ExtractorRegistry | None = None,
        cleaner: TextCleaner | None = None,
        links: LinkExtractor | None = None,
        policy: RetryPolicy | None = None,
        config: Settings = settings,
    ) -> None:
        if job.status is not JobStatus.RUNNING or job.run_id != run_id:
            raise ValueError("the job must be running under this run_id (JobRepository.start)")
        self._job = job
        self._run_id = run_id
        self._pages = pages
        self._jobs = jobs
        self._claims = claims
        self._fetcher = fetcher
        self._robots = robots
        self._limiter = limiter
        self._sitemaps = sitemaps
        self._cancel = cancel
        self._cancel_poll = config.CRAWL_CANCEL_POLL_SECONDS
        self._cancel_poll_failing = False
        self._registry = registry or default_registry(config)
        self._cleaner = cleaner or TextCleaner(config=config)
        self._links = links or LinkExtractor(config)
        self._policy = policy or RetryPolicy.from_settings(config)
        self._config = config
        self._concurrency = config.CRAWL_CONCURRENCY
        self._max_wait = config.CRAWL_RATE_LIMIT_MAX_WAIT_SECONDS
        self._page_timeout = config.CRAWL_PAGE_TIMEOUT_SECONDS
        self._max_pages = job.config.max_pages
        self._max_bytes = config.CRAWL_MAX_TOTAL_BYTES
        self._max_seconds = config.CRAWL_MAX_JOB_SECONDS
        self._deadline = (job.started_at or utc_now()) + timedelta(seconds=self._max_seconds)
        self._charged = 0  # pages counted against max_pages (read back in run())
        self._bytes = job.counters.bytes_downloaded
        self._stop: CrawlStop | None = None
        self._sitemap_task: asyncio.Task[None] | None = None
        self._frontier = Frontier(
            pages,
            FrontierJob(job.tenant_id, job.kb_id, job.source_id, job.id),
            run_id=run_id,
            config=config,
        )
        self._scope = CrawlScope.for_job(job.seed_url, job.config)
        self._deduper = self._make_deduper()
        self._wake = asyncio.Event()
        self._done = False
        self._sitemap_done = False
        self._deferred_dropped = 0

    @property
    def frontier(self) -> Frontier:
        return self._frontier

    @property
    def scope(self) -> CrawlScope:
        return self._scope

    @property
    def stop(self) -> CrawlStop | None:
        return self._stop

    def _make_deduper(self) -> ContentDeduper:
        return ContentDeduper(self._job.id, self._scope, self._claims, config=self._config)

    # ---- the loop ---- #

    async def run(self) -> CrawlSummary:
        log = logger.bind(job_id=self._job.id, run_id=self._run_id)
        recovered = await self._frontier.recover()
        self._charged = await self._frontier.count_charged()
        seed_url = normalize_url(self._job.seed_url)
        created = await self._frontier.add(seed_url, depth=0, via=DiscoveredVia.SEED)
        if created:
            await self._count(discovered=1)
        await self._check_stops_at_start()
        anchored = await self._anchor(seed_url)
        rules = await self._robots.rules_for(anchored)
        log.info("crawling {} (scope host {})", anchored, self._scope.seed_host)

        sitemap: list[SitemapReport | None] = [None]
        sitemap_error: list[str | None] = [None]
        if not self._hard_stopped:
            async with asyncio.TaskGroup() as group:
                if self._stop is None:
                    self._sitemap_task = group.create_task(
                        self._discover_sitemaps(anchored, sitemap, sitemap_error)
                    )
                else:
                    self._sitemap_done = True
                workers = [group.create_task(self._worker()) for _ in range(self._concurrency)]
                poller = group.create_task(self._poll_cancel())
                await asyncio.wait(workers)
                poller.cancel()  # the crawl is over; a worker's error is raised by the group
            self._sitemap_task = None
        if self._stop is not None and self._sitemaps is not None and sitemap[0] is None:
            sitemap_error[0] = sitemap_error[0] or f"stopped early: {self._stop.reason}"

        await self._skip_rest()
        skipped = await self._flush_deferred()
        summary = CrawlSummary(
            seed_url=anchored,
            pages=await self._pages.count_by_status(self._job.id),
            reasons=await self._pages.count_by_reason(self._job.id),
            recovered=recovered,
            robots_status=rules.status,
            robots_detail=rules.detail,
            sitemap=sitemap[0],
            sitemap_error=sitemap_error[0],
            skipped_links=skipped,
            skipped_links_dropped=self._deferred_dropped,
            stop=self._stop,
        )
        log.info(
            "crawl finished ({}): {}",
            f"stopped: {self._stop.reason}" if self._stop else "drained",
            {str(k): v for k, v in summary.pages.items()},
        )
        return summary

    async def _anchor(self, seed_url: str) -> str:
        """Crawl the seed alone first; returns the URL the crawl is anchored on."""
        page = await self._frontier.claim_url(seed_url)
        if page is not None:
            await self._process(page)
        _, seed_hash = normalize_and_hash(seed_url)
        seed = await self._pages.get_by_url_hash(self._job.id, seed_hash)
        if (
            seed is not None
            and seed.status is PageStatus.SKIPPED
            and seed.reason is PageReason.REDIRECTED
            and seed.final_url is not None
        ):
            self._reanchor(seed.final_url)  # also on a resume, where the seed was done before
            return normalize_url(seed.final_url)
        return seed_url

    def _reanchor(self, final_url: str) -> None:
        host = urlsplit(final_url).hostname
        if not host or host == self._scope.seed_host:
            return
        logger.bind(job_id=self._job.id).info(
            "seed redirected: scope re-anchored {} -> {}", self._scope.seed_host, host
        )
        self._scope = self._scope.with_seed_host(host)
        self._deduper = self._make_deduper()

    async def _discover_sitemaps(
        self, seed_url: str, report: list[SitemapReport | None], error: list[str | None]
    ) -> None:
        try:
            if self._sitemaps is None:
                return
            try:
                found = await self._sitemaps.discover(
                    seed_url, self._scope, max_urls=self._job.config.max_pages
                )
            except JobRunSupersededError, DatabaseUnavailableError:
                raise
            except Exception as exc:  # discovery problems never stop the crawl
                logger.bind(job_id=self._job.id).exception("sitemap discovery failed")
                error[0] = f"{type(exc).__name__}: {exc}"[:500]
                return
            report[0] = found
            created = await self._frontier.add_many(
                FrontierEntry(url=entry.url, depth=0, via=DiscoveredVia.SITEMAP)
                for entry in found.entries
            )
            if created:
                await self._count(discovered=created)
        finally:
            self._sitemap_done = True
            self._notify()

    async def _worker(self) -> None:
        while not self._done:
            if self._fetching_stopped():
                return  # a hard stop: the pages left are recorded by _skip_rest
            page = await self._frontier.claim()
            if page is not None:
                await self._process(page)
                continue
            timeout = _IDLE_SECONDS
            if self._sitemap_done:
                stats = await self._frontier.stats()
                if stats.drained:
                    self._done = True
                    self._notify()
                    return
                if stats.ready:
                    continue  # lost a claim race: try again right away
                if not stats.in_flight and stats.next_retry_at is not None:
                    until = (stats.next_retry_at - utc_now()).total_seconds()
                    timeout = min(max(until, 0.0), 30.0)
            await self._idle(min(timeout, max(self._seconds_left(), 0.0)))

    def _notify(self) -> None:
        """Wake idle workers (work was added, a page finished, discovery ended)."""
        self._wake.set()
        self._wake = asyncio.Event()

    async def _idle(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self._wake.wait()

    # ---- caps ---- #

    @property
    def _hard_stopped(self) -> bool:
        return self._stop is not None and self._stop.reason.is_hard

    def _seconds_left(self) -> float:
        return (self._deadline - utc_now()).total_seconds()

    def _fetching_stopped(self) -> bool:
        """True once nothing more may be fetched (a hard stop, or the job's time is up)."""
        if not self._hard_stopped and self._seconds_left() <= 0:
            self._stop_crawl(
                CrawlStopReason.MAX_DURATION,
                f"the job reached its maximum duration of {self._max_seconds} s",
            )
        return self._hard_stopped

    def cancel(self, detail: str = "the job was cancelled") -> bool:
        """Stop the crawl as cancelled (in-flight pages drain). True if this call did it."""
        return self._stop_crawl(CrawlStopReason.CANCELLED, detail)

    def _stop_crawl(self, reason: CrawlStopReason, detail: str) -> bool:
        """Record why the crawl stops; True if this call stopped it (or outranked the stop)."""
        if self._stop is not None and reason.rank <= self._stop.reason.rank:
            return False
        self._stop = CrawlStop(reason, detail)
        logger.bind(job_id=self._job.id).info("crawl stopping: {} ({})", reason, detail)
        if self._sitemap_task is not None:
            self._sitemap_task.cancel()  # its URLs would only be skipped
        self._notify()
        return True

    async def _check_stops_at_start(self) -> None:
        """A job cancelled meanwhile, or resumed with a spent budget, stops right away."""
        if self._job.cancel_requested_at is not None:
            self.cancel("the job's cancel was requested before this run started")
        await self._check_cancel_flag()
        if self._charged >= self._max_pages:
            await self._page_budget_spent()
        if self._bytes >= self._max_bytes:
            self._stop_crawl(CrawlStopReason.MAX_TOTAL_BYTES, self._bytes_detail())
        self._fetching_stopped()

    async def _poll_cancel(self) -> None:
        """Runs beside the workers until the crawl ends or the cancel flag shows up."""
        if self._cancel is None:
            return
        while not self._hard_stopped:
            await asyncio.sleep(self._cancel_poll)
            await self._check_cancel_flag()

    async def _check_cancel_flag(self) -> None:
        if self._cancel is None:
            return
        try:
            cancelled = await self._cancel.is_cancelled(self._job.id)
        except RedisUnavailableError as exc:
            if not self._cancel_poll_failing:  # once per outage, not once per poll
                logger.bind(job_id=self._job.id).warning("cancel flag unreadable: {}", exc)
            self._cancel_poll_failing = True
            return
        self._cancel_poll_failing = False
        if cancelled:
            self.cancel()

    def _pages_detail(self) -> str:
        return f"the job's page budget ({self._max_pages} pages) was spent"

    def _bytes_detail(self) -> str:
        return f"downloaded {self._bytes} bytes; the job's budget is {self._max_bytes} bytes"

    async def _page_budget_spent(self) -> None:
        if self._stop_crawl(CrawlStopReason.MAX_PAGES, self._pages_detail()):
            skipped = await self._frontier.skip_queued(
                PageReason.MAX_PAGES, self._pages_detail(), uncharged_only=True
            )
            if skipped:
                await self._count(skipped=skipped)

    async def _charge(self, page: Page) -> bool:
        """Spend one page of the budget on `page` before its first fetch.

        False when the budget is gone (the page is recorded `skipped/max_pages`) or this run
        lost the page's lease.
        """
        if self._charged >= self._max_pages:
            await self._finish(page, _skip(PageReason.MAX_PAGES, self._pages_detail()))
            return False
        self._charged += 1  # before the await: concurrent workers see it at once
        if not await self._frontier.charge(page.id):
            self._charged -= 1
            logger.bind(job_id=self._job.id, url=page.url).warning("lost the page's lease")
            return False
        if self._charged >= self._max_pages:
            await self._page_budget_spent()
        return True

    async def _add_bytes(self, downloaded: int) -> None:
        self._bytes += downloaded
        await self._count(bytes_downloaded=downloaded)
        if self._bytes >= self._max_bytes:
            self._stop_crawl(CrawlStopReason.MAX_TOTAL_BYTES, self._bytes_detail())

    async def _skip_rest(self) -> None:
        """After a hard stop: everything still queued is recorded with the stop's reason."""
        if self._stop is None or not self._stop.reason.is_hard:
            return  # drained, or a soft stop (whose leftovers were skipped as they came)
        skipped = await self._frontier.skip_queued(self._stop.reason.page_reason, self._stop.detail)
        if skipped:
            await self._count(skipped=skipped)
            self._notify()

    # ---- one page ---- #

    async def _process(self, page: Page) -> None:
        try:
            await self._crawl(page)
        except JobRunSupersededError, DatabaseUnavailableError:
            raise
        except Exception as exc:
            logger.bind(job_id=self._job.id, url=page.url).exception("page processing failed")
            await self._finish(
                page,
                PageOutcome(
                    status=PageStatus.FAILED,
                    reason=PageReason.INTERNAL_ERROR,
                    reason_detail=f"{type(exc).__name__}: {exc}",
                ),
            )

    async def _gate(self, page: Page) -> tuple[PageOutcome | None, RobotsRules]:
        """Scope and robots.txt: a skip outcome, or None when the page may be crawled."""
        rules = await self._robots.rules_for(page.url)
        decision = self._scope.check(page.url, page.depth)
        if not decision.allowed:
            detail = str(decision.detail) if decision.detail else None
            return _skip(decision.reason or PageReason.OUT_OF_SCOPE, detail), rules
        if not rules.can_fetch(page.url):
            if rules.status in (RobotsStatus.OK, RobotsStatus.STALE):
                detail = "disallowed by robots.txt"
            else:
                detail = f"robots.txt {rules.status}: {rules.detail or 'no detail'}"
            return _skip(PageReason.ROBOTS_DISALLOWED, detail), rules
        return None, rules

    async def _crawl(self, page: Page) -> None:
        if self._fetching_stopped():
            await self._frontier.release(page.id)  # recorded by _skip_rest
            return
        if not page.charged and self._charged >= self._max_pages:
            # the budget is spent: no robots.txt lookup or anything else for this page
            await self._finish(page, _skip(PageReason.MAX_PAGES, self._pages_detail()))
            return
        skip, rules = await self._gate(page)
        if skip is not None:
            await self._finish(page, skip)
            return
        if not page.charged and not await self._charge(page):
            return

        deadline = asyncio.get_running_loop().time() + self._page_timeout
        try:
            async with asyncio.timeout_at(deadline):
                await self._limiter.acquire(
                    page.url, crawl_delay=rules.crawl_delay, max_wait=self._max_wait
                )
        except RateLimitWaitExceededError as exc:
            # the host is booked further ahead than a crawler should wait: come back then
            await self._frontier.release(page.id, delay_seconds=exc.reservation.wait_seconds)
            self._notify()
            return
        except TimeoutError:
            await self._fetch_failed(page, self._page_timeout_error(page, "the rate-limit wait"))
            return
        if self._fetching_stopped():  # the time ran out while waiting for the host's slot
            await self._frontier.release(page.id)
            return
        try:
            async with asyncio.timeout_at(deadline):
                result = await self._fetcher.fetch(page.url)
        except FetchError as error:
            await self._fetch_failed(page, error)
            return
        except TimeoutError:
            await self._fetch_failed(page, self._page_timeout_error(page, "the fetch"))
            return
        await self._add_bytes(result.bytes_downloaded)

        try:
            final = normalize_url(result.final_url)
        except UrlNormalizationError:
            final = page.url
        if final != page.url:
            await self._redirected(page, result, final)
        else:
            await self._content(page, result)

    def _page_timeout_error(self, page: Page, stage: str) -> FetchTimeoutError:
        return FetchTimeoutError(
            f"page exceeded CRAWL_PAGE_TIMEOUT_SECONDS ({self._page_timeout:g} s) in {stage}",
            url=page.url,
            detail=f"page deadline of {self._page_timeout:g} s exceeded in {stage}",
        )

    async def _fetch_failed(self, page: Page, error: FetchError) -> None:
        if isinstance(error, HttpStatusError) and error.retry_after:
            await self._limiter.pause(page.url, error.retry_after)  # the whole host waits
        decision = self._policy.decide(error, max(page.attempts, 1))
        if decision.retry:
            if await self._frontier.retry_later(page.id, decision.delay_seconds):
                self._notify()
            return
        await self._finish(
            page,
            PageOutcome(
                status=decision.final_status or PageStatus.FAILED,
                reason=decision.reason,
                reason_detail=decision.detail,
                http_status=error.status_code if isinstance(error, HttpStatusError) else None,
            ),
        )

    async def _redirected(self, page: Page, result: FetchResult, final: str) -> None:
        """The content belongs to the redirect target's page; this URL is only an alias."""
        if page.discovered_via is DiscoveredVia.SEED:
            self._reanchor(final)
        created = await self._frontier.add(
            final, depth=page.depth, via=page.discovered_via, discovered_from=page.url
        )
        target = await self._frontier.claim_url(final)
        fetched = target is None  # count the fetch once: on the target when it uses it
        if target is not None:
            skip, _ = await self._gate(target)
            if skip is not None:
                await self._finish(target, skip, fetched=True)
            else:
                await self._content(target, result)
        await self._finish(
            page,
            PageOutcome(
                status=PageStatus.SKIPPED,
                reason=PageReason.REDIRECTED,
                reason_detail=f"redirected to {final}",
                final_url=result.final_url,
                bytes_downloaded=result.bytes_downloaded,
                fetch_ms=result.elapsed_ms,
                fetched_at=utc_now(),
            ),
            fetched=fetched,
            discovered=int(created),
        )

    async def _content(self, page: Page, result: FetchResult) -> None:
        started = time.perf_counter()
        fetched = _fetched(result)
        try:
            document = await self._registry.extract(ExtractionInput.from_fetch(result))
            cleaned = self._cleaner.clean(document.markdown)
        except ExtractionError as exc:
            outcome = replace(
                fetched, status=exc.page_status, reason=exc.reason, reason_detail=exc.detail
            )
            await self._finish(page, outcome, fetched=True)
            return
        described = _described(fetched, document, int((time.perf_counter() - started) * 1000))

        links = self._outlinks(page, result, document)
        discovered = 0
        if links is not None:
            discovered += await self._frontier.add_many(
                FrontierEntry(
                    url=link.url, depth=page.depth + 1, via=DiscoveredVia.LINK,
                    discovered_from=page.url,
                )
                for link in links.follow
            )  # fmt: skip
            described = replace(described, deferred_links=self._deferred(links))

        if links is not None and links.noindex:
            outcome = replace(
                described,
                status=PageStatus.SKIPPED,
                reason=PageReason.NOINDEX,
                reason_detail="noindex (robots meta or X-Robots-Tag)",
            )
            await self._finish(page, outcome, fetched=True, discovered=discovered)
            return

        canonical = links.canonical_url if links is not None else None
        if canonical is not None and await self._canonical_target_deferred(canonical):
            canonical = None  # a canonical cycle: keep the content here
        decision = await self._deduper.check(
            page_id=page.id,
            url=page.url,
            markdown=cleaned.markdown,
            depth=page.depth,
            canonical_url=canonical,
            aliases=(result.final_url, result.requested_url, *result.redirect_chain),
        )
        reason_detail = None
        if decision.outcome is DedupeOutcome.CANONICAL_ELSEWHERE and decision.canonical_url:
            reason_detail = f"canonical: {decision.canonical_url}"
            created = await self._frontier.add(
                decision.canonical_url,
                depth=page.depth,
                via=DiscoveredVia.CANONICAL,
                discovered_from=page.url,
            )
            discovered += int(created)
        outcome = replace(
            described,
            status=decision.page_status,
            reason=decision.reason,
            reason_detail=reason_detail,
            markdown=cleaned.markdown if decision.outcome is DedupeOutcome.UNIQUE else None,
            content_hash=decision.content_hash,
            canonical_url=decision.canonical_url,
            duplicate_of=decision.duplicate_of,
        )
        await self._finish(page, outcome, fetched=True, discovered=discovered)

    def _outlinks(
        self, page: Page, result: FetchResult, document: ExtractedDocument
    ) -> PageLinks | None:
        if document.content_kind is ContentKind.HTML and result.text is not None:
            return self._links.extract(
                result.text,
                result.final_url,
                scope=self._scope,
                depth=page.depth,
                headers=result.headers,
            )
        if document.outlinks:  # links inside PDFs / Office documents
            return self._links.filter_urls(
                document.outlinks, result.final_url, scope=self._scope, depth=page.depth
            )
        return None

    async def _canonical_target_deferred(self, canonical: str) -> bool:
        """True when the canonical target itself was skipped for *its* canonical."""
        try:
            _, target_hash = normalize_and_hash(canonical)
        except UrlNormalizationError:
            return False
        target = await self._pages.get_by_url_hash(self._job.id, target_hash)
        return (
            target is not None
            and target.status is PageStatus.SKIPPED
            and target.reason is PageReason.CANONICAL_ELSEWHERE
        )

    async def _finish(
        self, page: Page, outcome: PageOutcome, *, fetched: bool = False, discovered: int = 0
    ) -> None:
        try:
            await self._pages.record_outcome(page.id, outcome, held_by=self._run_id)
        except PageNotFoundError:
            # another run reclaimed it after our lease expired: its outcome wins
            logger.bind(job_id=self._job.id, url=page.url).warning("lost the page's lease")
            return
        deltas = {"fetched": int(fetched), "discovered": discovered}
        if (counter := _STATUS_COUNTER.get(outcome.status)) is not None:
            deltas[counter] = 1
        await self._count(**deltas)
        self._notify()

    async def _count(self, **deltas: int) -> None:
        await self._jobs.increment(self._job.id, self._run_id, **deltas)

    # ---- never-crawled links ---- #

    def _deferred(self, links: PageLinks) -> tuple[DeferredLink, ...]:
        """Links held back on this page (out of scope, too deep, nofollow), kept on its row."""
        candidates = [
            *(
                (r.url, r.decision.reason or PageReason.OUT_OF_SCOPE,
                 str(r.decision.detail) if r.decision.detail else None)
                for r in links.rejected
            ),
            *((link.url, PageReason.NOFOLLOW, "rel=nofollow") for link in links.nofollow),
        ]  # fmt: skip
        kept: dict[str, DeferredLink] = {}
        for url, reason, detail in candidates:
            if url in kept:
                continue
            if urlsplit(url).scheme.lower() not in _HTTP_SCHEMES or len(kept) >= MAX_DEFERRED_LINKS:
                self._deferred_dropped += 1  # mailto:, tel:, ... can't be a page row
                continue
            kept[url] = DeferredLink(url=url, reason=reason, detail=detail)
        return tuple(kept.values())

    async def _flush_deferred(self) -> int:
        """Record never-crawled links as skipped pages, unless some crawl reached them.

        Read back from the page rows, so links seen by an earlier (crashed) run count too.
        """
        recorded = 0
        batch: list[SkippedEntry] = []
        async for page_url, depth, links in self._pages.iter_deferred_links(self._job.id):
            batch.extend(
                SkippedEntry(
                    url=link.url, depth=depth + 1, reason=link.reason, detail=link.detail,
                    discovered_from=page_url,
                )
                for link in links
            )  # fmt: skip
            if len(batch) >= _FLUSH_BATCH:
                recorded += await self._frontier.add_skipped(batch)
                batch = []
        if batch:
            recorded += await self._frontier.add_skipped(batch)
        if recorded:
            await self._count(skipped=recorded)
        return recorded


def _skip(reason: PageReason, detail: str | None) -> PageOutcome:
    return PageOutcome(status=PageStatus.SKIPPED, reason=reason, reason_detail=detail)


def _fetched(result: FetchResult) -> PageOutcome:
    """Fetch facts of a page; callers `replace()` the status and the rest."""
    return PageOutcome(
        status=PageStatus.SUCCEEDED,
        http_status=result.status_code,
        final_url=result.final_url,
        content_type=result.mime_type[:200] or None,
        rendered_with=result.rendered_with,
        js_signals=result.js_signals,
        render_note=(result.render_note or "")[:500] or None,
        bytes_downloaded=result.bytes_downloaded,
        fetch_ms=result.elapsed_ms,
        fetched_at=utc_now(),
    )


def _described(base: PageOutcome, document: ExtractedDocument, extract_ms: int) -> PageOutcome:
    def cut(value: str | None, limit: int) -> str | None:
        return value[:limit] if value else None

    return replace(
        base,
        content_kind=document.content_kind,
        extractor=cut(document.extractor, 50),
        title=cut(document.title, 1000),
        description=cut(document.description, 2000),
        lang=cut(document.lang, 10),
        doc_metadata=dict(document.metadata),
        extract_ms=extract_ms,
    )
