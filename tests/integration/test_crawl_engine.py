"""Crawl engine in real MongoDB (+ Redis for the fixture-site crawl).

Scripted tests drive the engine with an in-memory fetcher/robots/limiter for exact edge
cases; the fixture-site tests crawl the local edge-case website over real HTTP.
"""

import asyncio
import contextlib
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from rag_engine.config import JsRenderMode, Settings
from rag_engine.crawling import (
    CachedRobots,
    CrawlEngine,
    DomainRateLimiter,
    RateLimitWaitExceededError,
    RedisCancelFlags,
    Reservation,
    RetryPolicy,
    RobotsCache,
    RobotsRules,
    RobotsStatus,
    SitemapDiscoverer,
    finish_cancelled,
    normalize_and_hash,
    origin_of,
    rate_limit_host,
    request_cancel,
)
from rag_engine.crawling.fetchers import (
    AutoFetcher,
    FetchResult,
    FetchTimeoutError,
    HttpFetcher,
    HttpStatusError,
    NetworkFetchError,
    RedirectLoopFetchError,
    SsrfFetchError,
    TooLargeError,
    TooManyRedirectsFetchError,
    UnsupportedContentTypeError,
)
from rag_engine.domain import (
    ContentHashClaim,
    ContentKind,
    CrawlConfig,
    CrawlJob,
    CrawlStopReason,
    DiscoveredVia,
    JobCounters,
    JobStatus,
    Page,
    PageReason,
    PageStatus,
    utc_now,
)
from rag_engine.extraction import CleanedText, TextCleaner
from rag_engine.infrastructure import MongoConnection, RedisConnection, RedisUnavailableError
from rag_engine.repositories import (
    ContentClaimRepository,
    JobRepository,
    JobRunSupersededError,
    PageRepository,
)
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fakes import InMemoryRobotsStore
from tests.support.fixture_site import EXPECTED_PAGE_OUTCOME, SITE_PAGES, Expect, FixtureSite

_TENANT = "ten_" + "1" * 32
_KB = "kb_" + "1" * 32
_SOURCE = "src_" + "1" * 32
_SITE = "https://acme.example"
_TEXT = "Widgets are sorted at 1200 items per minute by the WX-2048."


# ---- in-memory collaborators for scripted crawls ---- #


def _html(body: str, *, title: str = "Page", head: str = "") -> str:
    return f"<html><head><title>{title}</title>{head}</head><body><main>{body}</main></body></html>"


def _page(
    url: str, html: str, *, final_url: str | None = None, chain: tuple[str, ...] = ()
) -> FetchResult:
    body = html.encode()
    return FetchResult(
        requested_url=url,
        final_url=final_url or url,
        redirect_chain=chain,
        status_code=200,
        mime_type="text/html",
        content_kind=ContentKind.HTML,
        body=body,
        text=html,
        encoding="utf-8",
        bytes_downloaded=len(body),
        elapsed_ms=3,
        headers={"content-type": "text/html; charset=utf-8"},
    )


@dataclass
class _ScriptedFetcher:
    """`PageFetcher` answering from {url: FetchResult | exception}; unknown URLs 404."""

    responses: Mapping[str, FetchResult | Exception]
    calls: list[str] = field(default_factory=list)
    on_fetch: Callable[[str], object] | None = None

    async def fetch(self, url: str) -> FetchResult:
        self.calls.append(url)
        if self.on_fetch is not None:
            result = self.on_fetch(url)
            if asyncio.iscoroutine(result):
                await result
        outcome = self.responses.get(url, HttpStatusError(url=url, status_code=404))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _AllowAllRobots:
    async def rules_for(self, url: str) -> RobotsRules:
        entry = CachedRobots(
            status=RobotsStatus.UNAVAILABLE, body="", fetched_at=0, expires_at=1e12
        )
        return RobotsRules.from_cached(origin_of(url), entry, "ragenginebot")


@dataclass
class _Limiter:
    """Never waits; `refuse` lists URLs whose first slot is too far away."""

    refuse: set[str] = field(default_factory=set)
    acquired: list[str] = field(default_factory=list)
    paused: list[tuple[str, float]] = field(default_factory=list)

    async def acquire(
        self, url: str, *, crawl_delay: float | None = None, max_wait: float | None = None
    ) -> Reservation:
        reservation = Reservation(rate_limit_host(url), True, 0.0, 0.0, 0.0, 1)
        if url in self.refuse:
            self.refuse.discard(url)
            raise RateLimitWaitExceededError(Reservation(reservation.host, False, 0.2, 0.0, 0.0, 1))
        self.acquired.append(url)
        return reservation

    async def pause(self, url: str, seconds: float) -> float:
        self.paused.append((url, seconds))
        return seconds


class _ExplodingCleaner(TextCleaner):
    """A bug in the pipeline for one page (anything mentioning BOOM)."""

    def clean(self, markdown: str) -> CleanedText:
        if "BOOM" in markdown:
            raise RuntimeError("cleaner bug")
        return super().clean(markdown)


@dataclass
class _Env:
    pages: PageRepository
    jobs: JobRepository
    claims: ContentClaimRepository
    config: Settings

    async def job(self, seed: str, **config: object) -> CrawlJob:
        job = await self.jobs.insert(
            CrawlJob(
                tenant_id=_TENANT,
                kb_id=_KB,
                source_id=_SOURCE,
                seed_url=seed,
                config=CrawlConfig.model_validate({"max_depth": 5, **config}),
            )
        )
        return await self.jobs.start(job.id, "run-1")

    def engine(
        self, job: CrawlJob, fetcher: object, *, run_id: str = "run-1", **kwargs: object
    ) -> CrawlEngine:
        return CrawlEngine(
            job,
            run_id=run_id,
            pages=self.pages,
            jobs=self.jobs,
            claims=self.claims,
            fetcher=fetcher,  # ty: ignore[invalid-argument-type]
            robots=kwargs.pop("robots", _AllowAllRobots()),  # ty: ignore[invalid-argument-type]
            limiter=kwargs.pop("limiter", _Limiter()),  # ty: ignore[invalid-argument-type]
            config=self.config,
            **kwargs,  # ty: ignore[invalid-argument-type]
        )

    async def outcomes(self, job: CrawlJob) -> dict[str, Page]:
        return {p.url: p for p in await self.pages.find({"job_id": job.id})}


@pytest.fixture
async def env(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[_Env]:
    config = override_settings(CRAWL_CONCURRENCY=3)
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Page, CrawlJob, ContentHashClaim])
    yield _Env(
        PageRepository(mongo.db), JobRepository(mongo.db), ContentClaimRepository(mongo.db), config
    )
    await mongo.close()


def _status(page: Page) -> tuple[PageStatus, PageReason | None]:
    return page.status, page.reason


# ---- fixture-site crawl ---- #


@pytest.fixture
def site_config(override_settings: OverrideSettings, integration_settings: Settings) -> Settings:
    return override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True,
        CRAWL_JS_RENDERING=JsRenderMode.NEVER,  # the browser E2E is Step 54
        CRAWL_CONCURRENCY=4,
        CRAWL_PER_DOMAIN_RPS=50.0,  # robots.txt's Crawl-delay (0.1 s) sets the pace
        CRAWL_READ_TIMEOUT_SECONDS=1,
        CRAWL_FETCH_TIMEOUT_SECONDS=2,
        CRAWL_MAX_RETRIES=2,
        CRAWL_RETRY_BASE_DELAY_SECONDS=0.05,
        CRAWL_RETRY_MAX_DELAY_SECONDS=0.1,
        CRAWL_RETRY_AFTER_MAX_SECONDS=1,
    )


@dataclass
class _Site:
    env: _Env
    site: FixtureSite
    config: Settings
    redis: RedisConnection
    http: HttpFetcher

    def engine(
        self, job: CrawlJob, *, run_id: str = "run-1", fetcher: object = None
    ) -> CrawlEngine:
        robots = RobotsCache(self.http, InMemoryRobotsStore(), config=self.config)
        limiter = DomainRateLimiter(self.redis.client, self.redis.keys, self.config)
        return CrawlEngine(
            job,
            run_id=run_id,
            pages=self.env.pages,
            jobs=self.env.jobs,
            claims=self.env.claims,
            fetcher=fetcher or AutoFetcher(self.http, None, config=self.config),  # ty: ignore[invalid-argument-type]
            robots=robots,
            limiter=limiter,
            sitemaps=SitemapDiscoverer(
                self.http, robots=robots, limiter=limiter, config=self.config
            ),
            config=self.config,
        )


@pytest.fixture
async def site(
    env: _Env, fixture_site: FixtureSite, site_config: Settings, integration_settings: Settings
) -> AsyncIterator[_Site]:
    redis = RedisConnection(integration_settings)
    await redis.connect()
    async with SafeHttpClient(site_config) as client:
        yield _Site(env, fixture_site, site_config, redis, HttpFetcher(client, site_config))
    await redis.close()


# without a browser the SPA shell has no content, and its JS-only link is never seen
_NEEDS_BROWSER = {Expect.JS_RENDERED}
_JS_ONLY = {"/app/advanced"}


def _assert_fixture_outcomes(site: FixtureSite, outcomes: dict[str, Page]) -> None:
    duplicates = []
    for path, expect in SITE_PAGES.items():
        url = site.url(path)
        if not url.startswith("http") or path in _JS_ONLY:
            continue
        normalized, _ = normalize_and_hash(url)
        page = outcomes.get(normalized)
        if expect in (Expect.DUPLICATE,) or path == "/dup-a.html":
            duplicates.append(page)
            continue
        if expect in _NEEDS_BROWSER:
            assert page is not None and page.status is not PageStatus.QUEUED, path
            continue
        if expect is Expect.OUT_OF_SCOPE and page is None:
            continue  # only recorded when some page links to it
        assert page is not None, f"{path} was never discovered"
        assert _status(page) == EXPECTED_PAGE_OUTCOME[expect], (path, page.reason_detail)
    # dup-a / dup-b: whichever was fetched second duplicates the first
    assert Counter(p.status for p in duplicates if p) == {
        PageStatus.SUCCEEDED: 1,
        PageStatus.DUPLICATE: 1,
    }
    original = next(p for p in duplicates if p and p.status is PageStatus.SUCCEEDED)
    copy = next(p for p in duplicates if p and p.status is PageStatus.DUPLICATE)
    assert copy.duplicate_of == original.id


# ---- #
# Tests
# ---- #


class TestCrawlEngine_S50:
    """S50 — Crawl engine core: per-URL pipeline, bounded concurrency, fixture-site crawl."""

    # ---- the fixture site over real HTTP ---- #

    async def test_crawls_the_fixture_site(self, site: _Site) -> None:
        # Arrange
        job = await site.env.job(site.site.url("/"), max_depth=10)

        # Act
        started = time.monotonic()
        summary = await site.engine(job).run()
        elapsed = time.monotonic() - started

        # Assert: every page's outcome
        outcomes = await site.env.outcomes(job)
        _assert_fixture_outcomes(site.site, outcomes)
        assert all(
            p.status not in (PageStatus.QUEUED, PageStatus.IN_PROGRESS) for p in outcomes.values()
        )
        succeeded = [p for p in outcomes.values() if p.status is PageStatus.SUCCEEDED]
        assert all(p.markdown and p.content_hash and p.fetched_at for p in succeeded)
        kinds = Counter(p.content_kind for p in succeeded)
        assert kinds[ContentKind.HTML] >= 15
        assert {ContentKind.PDF, ContentKind.DOCX, ContentKind.PPTX, ContentKind.XLSX} <= set(kinds)

        # Assert: requests (robots respected, nothing fetched twice, retries happened)
        requested = Counter(r.path for r in site.site.requests)
        assert requested["/private/secret.html"] == 0
        assert requested["/data.json"] == requested["/images/logo.png"] == 0  # asset: never fetched
        # every redirect hop lands on getting-started; beyond those, at most one direct fetch
        direct = requested["/docs/getting-started.html"] - requested["/redirect-me"]
        assert direct <= 1
        assert requested["/flaky"] == 3  # 503 twice, then 200
        assert requested["/error"] == 3  # CRAWL_MAX_RETRIES=2 → 3 attempts
        assert requested["/rate-limited"] == 1  # Retry-After over the cap: gives up at once
        pages_fetched = [
            p for p, n in requested.items() if not p.startswith(("/robots", "/sitemap"))
        ]
        repeat = {p: requested[p] for p in pages_fetched if requested[p] > 1}
        redirect_hops = {"/redirect-me", "/docs/getting-started.html"}  # see above
        assert set(repeat) <= {"/flaky", "/error", "/slow", *redirect_hops}, repeat

        # Assert: politeness. robots.txt asks for a 0.1 s Crawl-delay; slots are booked exactly
        # 0.1 s apart (Step 30), arrival jitters by event-loop scheduling, so check the rate:
        # at most 10 requests (+1 at the window edge) in any 1 s window.
        # One fetch follows its redirects at once (one slot); robots.txt is fetched unthrottled.
        hops = {"/robots.txt", "/redirect-me", "/redirect-chain", "/loop-a", "/loop-b"}
        hops.add("/docs/getting-started.html")
        starts = sorted(r.at for r in site.site.requests if r.path not in hops)
        busiest = max(sum(1 for t in starts if s0 <= t < s0 + 1.0) for s0 in starts)
        assert busiest <= 11, busiest

        # Assert: summary and counters agree with the page table
        assert summary.robots_status is RobotsStatus.OK
        assert summary.sitemap is not None and site.site.url("/orphan.html") in summary.sitemap.urls
        assert summary.pages == {
            s: n for s, n in Counter(p.status for p in outcomes.values()).items()
        }
        counters = (await site.env.jobs.get_or_raise(job.id)).counters
        by_status = summary.pages
        assert counters.failed == by_status[PageStatus.FAILED]
        assert counters.skipped == by_status[PageStatus.SKIPPED]
        assert counters.duplicates == by_status[PageStatus.DUPLICATE]
        assert counters.discovered == len(outcomes) - summary.skipped_links
        assert counters.fetched >= by_status[PageStatus.SUCCEEDED] + by_status[PageStatus.DUPLICATE]
        assert elapsed < 60

    async def test_resumes_after_a_crash_without_refetching_finished_pages(
        self, site: _Site
    ) -> None:
        # Arrange: run-1 crashes (its task is cancelled) after a handful of pages
        job = await site.env.job(site.site.url("/"), max_depth=10)
        fetched_enough = asyncio.Event()
        http = AutoFetcher(site.http, None, config=site.config)

        class _CountingFetcher:
            calls = 0

            async def fetch(self, url: str) -> FetchResult:
                type(self).calls += 1
                if self.calls >= 12:
                    fetched_enough.set()
                return await http.fetch(url)

        crashed = site.engine(job, run_id="run-1", fetcher=_CountingFetcher())
        task = asyncio.create_task(crashed.run())
        await fetched_enough.wait()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        done_before = {
            p.url
            for p in await site.env.pages.find({"job_id": job.id})
            if p.status is PageStatus.SUCCEEDED
        }

        # Act: the job is requeued (Step 80) and a new run resumes it
        await site.env.jobs.requeue(job.id, stale_run_id="run-1")
        resumed_job = await site.env.jobs.start(job.id, "run-2")
        site.site.reset()
        summary = await site.engine(resumed_job, run_id="run-2").run()

        # Assert
        outcomes = await site.env.outcomes(job)
        _assert_fixture_outcomes(site.site, outcomes)
        # redirects crawled in run-2 still land on their (finished) target: not a refetch
        hops = {site.site.url("/docs/getting-started.html")}
        refetched = {site.site.base_url + r.path for r in site.site.requests} & done_before - hops
        assert refetched == set()
        assert summary.recovered.abandoned == 0

    # ---- scripted edge cases ---- #

    async def test_redirect_content_is_recorded_under_the_target(self, env: _Env) -> None:
        target = f"{_SITE}/docs/start"
        fetcher = _ScriptedFetcher(
            {
                f"{_SITE}/": _page(
                    f"{_SITE}/",
                    _html(f"<p>{_TEXT}</p><a href='/go'>go</a><a href='/docs/start'>start</a>"),
                ),
                f"{_SITE}/go": _page(
                    f"{_SITE}/go",
                    _html(f"<h1>Start</h1><p>{_TEXT} Start here.</p>"),
                    final_url=target,
                    chain=(f"{_SITE}/go",),
                ),
                target: _page(target, _html(f"<h1>Start</h1><p>{_TEXT} Start here.</p>")),
            }
        )
        job = await env.job(f"{_SITE}/")

        await env.engine(job, fetcher).run()

        outcomes = await env.outcomes(job)
        assert _status(outcomes[f"{_SITE}/go"]) == (PageStatus.SKIPPED, PageReason.REDIRECTED)
        assert outcomes[f"{_SITE}/go"].final_url == target
        assert _status(outcomes[target]) == (PageStatus.SUCCEEDED, None)
        assert fetcher.calls.count(target) <= 1  # never fetched again for the redirect

    async def test_seed_redirect_to_another_host_reanchors_the_scope(self, env: _Env) -> None:
        moved = "https://acme.io/"
        fetcher = _ScriptedFetcher(
            {
                f"{_SITE}/": _page(
                    f"{_SITE}/",
                    _html(f"<p>{_TEXT}</p><a href='/about'>about</a><a href='{_SITE}/old'>old</a>"),
                    final_url=moved,
                    chain=(f"{_SITE}/",),
                ),
                "https://acme.io/about": _page(
                    "https://acme.io/about", _html(f"<p>About us. {_TEXT}</p>")
                ),
            }
        )
        job = await env.job(f"{_SITE}/")

        engine = env.engine(job, fetcher)
        summary = await engine.run()

        outcomes = await env.outcomes(job)
        assert summary.seed_url == moved and engine.scope.seed_host == "acme.io"
        assert _status(outcomes[f"{_SITE}/"]) == (PageStatus.SKIPPED, PageReason.REDIRECTED)
        assert _status(outcomes[moved]) == (PageStatus.SUCCEEDED, None)
        assert _status(outcomes["https://acme.io/about"]) == (PageStatus.SUCCEEDED, None)
        assert _status(outcomes[f"{_SITE}/old"]) == (PageStatus.SKIPPED, PageReason.OUT_OF_SCOPE)

    async def test_mutual_canonicals_keep_one_copy(self, env: _Env) -> None:
        a, b = f"{_SITE}/a", f"{_SITE}/b"
        content = f"<p>{_TEXT} Shared article.</p>"
        fetcher = _ScriptedFetcher(
            {
                f"{_SITE}/": _page(f"{_SITE}/", _html(f"<p>{_TEXT} Home.</p><a href='/a'>a</a>")),
                a: _page(a, _html(content, head=f"<link rel='canonical' href='{b}'>")),
                b: _page(b, _html(content, head=f"<link rel='canonical' href='{a}'>")),
            }
        )
        job = await env.job(f"{_SITE}/")

        await env.engine(job, fetcher).run()

        outcomes = await env.outcomes(job)
        assert _status(outcomes[a]) == (PageStatus.SKIPPED, PageReason.CANONICAL_ELSEWHERE)
        assert outcomes[b].discovered_via is DiscoveredVia.CANONICAL
        assert _status(outcomes[b]) == (PageStatus.SUCCEEDED, None)  # the cycle is broken here

    async def test_noindex_page_is_skipped_but_its_links_are_followed(self, env: _Env) -> None:
        fetcher = _ScriptedFetcher(
            {
                f"{_SITE}/": _page(f"{_SITE}/", _html(f"<p>{_TEXT}</p><a href='/hub'>hub</a>")),
                f"{_SITE}/hub": _page(
                    f"{_SITE}/hub",
                    _html(
                        f"<p>Hub. {_TEXT}</p><a href='/leaf'>leaf</a>",
                        head="<meta name='robots' content='noindex'>",
                    ),
                ),
                f"{_SITE}/leaf": _page(f"{_SITE}/leaf", _html(f"<p>Leaf. {_TEXT}</p>")),
            }
        )
        job = await env.job(f"{_SITE}/")

        await env.engine(job, fetcher).run()

        outcomes = await env.outcomes(job)
        assert _status(outcomes[f"{_SITE}/hub"]) == (PageStatus.SKIPPED, PageReason.NOINDEX)
        assert outcomes[f"{_SITE}/hub"].markdown is None
        assert _status(outcomes[f"{_SITE}/leaf"]) == (PageStatus.SUCCEEDED, None)

    async def test_never_crawled_links_are_reported_unless_reached_otherwise(
        self, env: _Env
    ) -> None:
        fetcher = _ScriptedFetcher(
            {
                f"{_SITE}/": _page(
                    f"{_SITE}/",
                    _html(
                        f"<p>{_TEXT}</p><a href='/a'>a</a><a href='/c'>c</a>"
                        "<a href='/held' rel='nofollow'>held</a><a href='https://offsite.example/x'>x</a>"
                        "<a href='mailto:hi@acme.example'>mail</a>"
                    ),
                ),
                f"{_SITE}/a": _page(
                    f"{_SITE}/a",
                    _html(f"<p>A. {_TEXT}</p><a href='/c'>c</a><a href='/deep'>deep</a>"),
                ),
                f"{_SITE}/c": _page(f"{_SITE}/c", _html(f"<p>C. {_TEXT}</p>")),
            }
        )
        job = await env.job(f"{_SITE}/", max_depth=1)

        summary = await env.engine(job, fetcher).run()

        outcomes = await env.outcomes(job)
        assert _status(outcomes[f"{_SITE}/c"]) == (PageStatus.SUCCEEDED, None)
        assert _status(outcomes[f"{_SITE}/deep"]) == (PageStatus.SKIPPED, PageReason.MAX_DEPTH)
        assert _status(outcomes[f"{_SITE}/held"]) == (PageStatus.SKIPPED, PageReason.NOFOLLOW)
        assert _status(outcomes["https://offsite.example/x"]) == (
            PageStatus.SKIPPED,
            PageReason.OUT_OF_SCOPE,
        )
        assert f"{_SITE}/held" not in fetcher.calls and f"{_SITE}/deep" not in fetcher.calls
        assert (summary.skipped_links, summary.skipped_links_dropped) == (3, 1)  # mailto: dropped
        assert (await env.jobs.get_or_raise(job.id)).counters.skipped == 3

    async def test_retries_rate_limit_waits_and_failures(self, env: _Env) -> None:
        flaky = f"{_SITE}/flaky"
        attempts = {"n": 0}
        responses: dict[str, FetchResult | Exception] = {
            f"{_SITE}/": _page(
                f"{_SITE}/",
                _html(
                    f"<p>{_TEXT}</p><a href='/flaky'>f</a><a href='/busy'>b</a>"
                    "<a href='/gone'>g</a><a href='/throttled'>t</a>"
                ),
            ),
            f"{_SITE}/busy": _page(f"{_SITE}/busy", _html(f"<p>Busy host page. {_TEXT}</p>")),
            f"{_SITE}/throttled": HttpStatusError(
                url=f"{_SITE}/throttled", status_code=429, retry_after=0.5
            ),
        }

        class _Flaky(_ScriptedFetcher):
            async def fetch(self, url: str) -> FetchResult:
                if url == flaky and attempts["n"] < 2:
                    attempts["n"] += 1
                    self.calls.append(url)
                    raise HttpStatusError(url=url, status_code=503)
                if url == flaky:
                    self.calls.append(url)
                    return _page(url, _html(f"<p>Recovered. {_TEXT}</p>"))
                return await super().fetch(url)

        fetcher = _Flaky(responses)
        limiter = _Limiter(refuse={f"{_SITE}/busy"})
        job = await env.job(f"{_SITE}/")
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, max_retry_after=5)

        await env.engine(job, fetcher, limiter=limiter, policy=policy).run()

        outcomes = await env.outcomes(job)
        assert _status(outcomes[flaky]) == (PageStatus.SUCCEEDED, None)
        assert (fetcher.calls.count(flaky), outcomes[flaky].attempts) == (3, 3)
        busy = outcomes[f"{_SITE}/busy"]
        assert _status(busy) == (PageStatus.SUCCEEDED, None)
        assert busy.attempts == 1  # the rate-limit release refunded its first claim
        gone = outcomes[f"{_SITE}/gone"]
        assert _status(gone) == (PageStatus.FAILED, PageReason.HTTP_CLIENT_ERROR)
        assert gone.http_status == 404 and fetcher.calls.count(f"{_SITE}/gone") == 1
        throttled = outcomes[f"{_SITE}/throttled"]
        assert _status(throttled) == (PageStatus.FAILED, PageReason.RATE_LIMITED)
        assert (f"{_SITE}/throttled", 0.5) in limiter.paused  # the whole host was paused
        assert fetcher.calls.count(f"{_SITE}/throttled") == 3

    async def test_a_bug_fails_only_that_page(self, env: _Env) -> None:
        fetcher = _ScriptedFetcher(
            {
                f"{_SITE}/": _page(
                    f"{_SITE}/", _html(f"<p>{_TEXT}</p><a href='/bad'>b</a><a href='/good'>g</a>")
                ),
                f"{_SITE}/bad": _page(f"{_SITE}/bad", _html(f"<p>BOOM {_TEXT}</p>")),
                f"{_SITE}/good": _page(f"{_SITE}/good", _html(f"<p>Good. {_TEXT}</p>")),
            }
        )
        job = await env.job(f"{_SITE}/")

        await env.engine(job, fetcher, cleaner=_ExplodingCleaner(config=env.config)).run()

        outcomes = await env.outcomes(job)
        bad = outcomes[f"{_SITE}/bad"]
        assert _status(bad) == (PageStatus.FAILED, PageReason.INTERNAL_ERROR)
        assert bad.reason_detail == "RuntimeError: cleaner bug"
        assert _status(outcomes[f"{_SITE}/good"]) == (PageStatus.SUCCEEDED, None)

    async def test_counters_match_the_page_table(self, env: _Env) -> None:
        fetcher = _ScriptedFetcher(
            {
                f"{_SITE}/": _page(
                    f"{_SITE}/",
                    _html(f"<p>{_TEXT}</p>" + "".join(f"<a href='/p{n}'>p</a>" for n in range(12))),
                ),
                **{
                    f"{_SITE}/p{n}": _page(f"{_SITE}/p{n}", _html(f"<p>Page {n % 4}. {_TEXT}</p>"))
                    for n in range(10)
                },
            }
        )
        job = await env.job(f"{_SITE}/")

        summary = await env.engine(job, fetcher).run()

        counters = (await env.jobs.get_or_raise(job.id)).counters
        assert summary.pages == {
            PageStatus.SUCCEEDED: 5,
            PageStatus.DUPLICATE: 6,
            PageStatus.FAILED: 2,
        }
        downloaded = sum(p.bytes_downloaded for p in (await env.outcomes(job)).values())
        assert downloaded > 0
        assert counters == JobCounters(
            discovered=13, fetched=11, failed=2, duplicates=6, bytes_downloaded=downloaded
        )

    async def test_a_superseded_run_stops(self, env: _Env) -> None:
        job = await env.job(f"{_SITE}/")

        async def requeue_mid_crawl(url: str) -> None:
            if url.endswith("/p3"):
                await env.jobs.requeue(job.id, stale_run_id="run-1")

        fetcher = _ScriptedFetcher(
            {
                f"{_SITE}/": _page(
                    f"{_SITE}/",
                    _html(f"<p>{_TEXT}</p>" + "".join(f"<a href='/p{n}'>p</a>" for n in range(10))),
                ),
                **{
                    f"{_SITE}/p{n}": _page(f"{_SITE}/p{n}", _html(f"<p>Page {n}. {_TEXT}</p>"))
                    for n in range(10)
                },
            },
            on_fetch=requeue_mid_crawl,
        )

        with pytest.raises(ExceptionGroup) as exc:
            await env.engine(job, fetcher).run()

        assert exc.group_contains(JobRunSupersededError)
        assert len(fetcher.calls) < 11  # it stopped instead of crawling on

    def test_engine_needs_a_started_job(self, env: _Env) -> None:
        job = CrawlJob(tenant_id=_TENANT, kb_id=_KB, source_id=_SOURCE, seed_url=f"{_SITE}/")

        with pytest.raises(ValueError, match="running"):
            env.engine(job, _ScriptedFetcher({}))


# ---- caps (Step 51) ---- #


def _site_of(count: int, *, extra: str = "") -> dict[str, FetchResult | Exception]:
    """A seed linking /p0 … /p{count-1}, each page with distinct content."""
    links = "".join(f"<a href='/p{n}'>p</a>" for n in range(count))
    return {
        f"{_SITE}/": _page(f"{_SITE}/", _html(f"<p>{_TEXT}</p>{extra}{links}")),
        **{
            f"{_SITE}/p{n}": _page(f"{_SITE}/p{n}", _html(f"<p>Page number {n}. {_TEXT}</p>"))
            for n in range(count)
        },
    }


@dataclass
class _RecordingRobots(_AllowAllRobots):
    looked_up: list[str] = field(default_factory=list)

    async def rules_for(self, url: str) -> RobotsRules:
        self.looked_up.append(url)
        return await super().rules_for(url)


class TestCrawlCaps_S51:
    """S51 — Caps & timeouts: page budget, byte budget, wall time, per-page deadline."""

    async def test_page_budget_stops_taking_new_pages(self, env: _Env) -> None:
        fetcher = _ScriptedFetcher(_site_of(10))
        robots = _RecordingRobots()
        job = await env.job(f"{_SITE}/", max_pages=4)

        summary = await env.engine(job, fetcher, robots=robots).run()

        outcomes = await env.outcomes(job)
        fetched = {url for url, page in outcomes.items() if page.charged}
        budget_skips = {
            url for url, page in outcomes.items() if page.reason is PageReason.MAX_PAGES
        }
        assert len(fetcher.calls) == len(set(fetcher.calls)) == 4
        assert fetched == set(fetcher.calls)
        assert len(budget_skips) == 7
        assert all(outcomes[url].status is PageStatus.SKIPPED for url in budget_skips)
        assert not budget_skips & set(robots.looked_up)  # skipped without a robots lookup
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.MAX_PAGES
        assert "4 pages" in summary.stop.detail
        assert PageStatus.QUEUED not in summary.pages
        assert summary.reasons[PageReason.MAX_PAGES] == 7
        counters = (await env.jobs.get_or_raise(job.id)).counters
        assert (counters.fetched, counters.skipped, counters.discovered) == (4, 7, 11)

    async def test_a_charged_page_still_finishes_its_retries(self, env: _Env) -> None:
        flaky = f"{_SITE}/flaky"
        responses = {
            f"{_SITE}/": _page(f"{_SITE}/", _html(f"<p>{_TEXT}</p><a href='/flaky'>f</a>")),
        }

        class _Flaky(_ScriptedFetcher):
            async def fetch(self, url: str) -> FetchResult:
                if url == flaky:
                    self.calls.append(url)
                    if self.calls.count(flaky) == 1:
                        raise HttpStatusError(url=url, status_code=503)
                    return _page(url, _html(f"<p>Recovered. {_TEXT}</p><a href='/later'>l</a>"))
                return await super().fetch(url)

        fetcher = _Flaky(responses)
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, max_retry_after=5)
        job = await env.job(f"{_SITE}/", max_pages=2)

        summary = await env.engine(job, fetcher, policy=policy).run()

        outcomes = await env.outcomes(job)
        assert _status(outcomes[flaky]) == (PageStatus.SUCCEEDED, None)
        assert (outcomes[flaky].attempts, fetcher.calls.count(flaky)) == (2, 2)
        assert _status(outcomes[f"{_SITE}/later"]) == (PageStatus.SKIPPED, PageReason.MAX_PAGES)
        assert f"{_SITE}/later" not in fetcher.calls
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.MAX_PAGES

    async def test_a_resumed_job_reads_back_its_page_budget(self, env: _Env) -> None:
        job = await env.job(f"{_SITE}/", max_pages=5)

        async def crash_on_second_page(url: str) -> None:
            if url.endswith("/p0"):
                await env.jobs.requeue(job.id, stale_run_id="run-1")

        run1 = _ScriptedFetcher(_site_of(10), on_fetch=crash_on_second_page)
        with pytest.raises(ExceptionGroup):
            await env.engine(job, run1).run()
        resumed = await env.jobs.start(job.id, "run-2")
        run2 = _ScriptedFetcher(_site_of(10))

        summary = await env.engine(resumed, run2, run_id="run-2").run()

        outcomes = await env.outcomes(job)
        assert len(set(run1.calls) | set(run2.calls)) == 5  # one budget across both runs
        assert sum(page.charged for page in outcomes.values()) == 5
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.MAX_PAGES
        assert {p.status for p in outcomes.values()} <= {
            PageStatus.SUCCEEDED,
            PageStatus.SKIPPED,
        }

    async def test_byte_budget_is_a_hard_stop(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        responses = _site_of(8, extra="<a href='/flaky'>f</a>")
        responses[f"{_SITE}/flaky"] = HttpStatusError(url=f"{_SITE}/flaky", status_code=503)
        size = max(r.bytes_downloaded for r in responses.values() if isinstance(r, FetchResult))
        override_settings(
            CRAWL_CONCURRENCY=1, CRAWL_MAX_PAGE_BYTES=size, CRAWL_MAX_TOTAL_BYTES=size * 3
        )
        fetcher = _ScriptedFetcher(responses)
        # the 503 page waits a minute for its retry: the stop must not wait for it
        policy = RetryPolicy(max_attempts=3, base_delay=60, max_delay=60, max_retry_after=5)
        job = await env.job(f"{_SITE}/")
        started = time.monotonic()

        summary = await env.engine(job, fetcher, policy=policy).run()

        outcomes = await env.outcomes(job)
        downloaded = (await env.jobs.get_or_raise(job.id)).counters.bytes_downloaded
        assert time.monotonic() - started < 10
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.MAX_TOTAL_BYTES
        sizes = [
            r.bytes_downloaded for u in fetcher.calls if isinstance(r := responses[u], FetchResult)
        ]
        assert downloaded == sum(sizes)
        assert sum(sizes[:-1]) < size * 3 <= downloaded  # stopped on the fetch that crossed it
        assert PageStatus.QUEUED not in summary.pages
        assert _status(outcomes[f"{_SITE}/flaky"]) == (
            PageStatus.SKIPPED,
            PageReason.MAX_TOTAL_BYTES,
        )  # a pending retry is skipped too
        unfetched = [p for u, p in outcomes.items() if u not in fetcher.calls]
        assert unfetched and all(
            _status(p) == (PageStatus.SKIPPED, PageReason.MAX_TOTAL_BYTES) for p in unfetched
        )
        counters = (await env.jobs.get_or_raise(job.id)).counters
        assert counters.skipped == summary.reasons[PageReason.MAX_TOTAL_BYTES]

    async def test_wall_time_is_a_hard_stop(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_MAX_JOB_SECONDS=1)
        responses = _site_of(20, extra="<a href='/flaky'>f</a>")
        responses[f"{_SITE}/flaky"] = HttpStatusError(url=f"{_SITE}/flaky", status_code=503)

        async def slow(url: str) -> None:
            await asyncio.sleep(0.25)

        fetcher = _ScriptedFetcher(responses, on_fetch=slow)
        policy = RetryPolicy(max_attempts=3, base_delay=30, max_delay=30, max_retry_after=5)
        job = await env.job(f"{_SITE}/")
        started = time.monotonic()

        summary = await env.engine(job, fetcher, policy=policy).run()

        elapsed = time.monotonic() - started
        outcomes = await env.outcomes(job)
        assert elapsed < 3  # in-flight pages drained; nobody waited for the 30 s retry
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.MAX_DURATION
        assert "1 s" in summary.stop.detail
        assert 1 < len(fetcher.calls) < 21
        assert not {PageStatus.QUEUED, PageStatus.IN_PROGRESS} & set(summary.pages)
        assert summary.reasons[PageReason.MAX_DURATION] >= 1
        assert _status(outcomes[f"{_SITE}/flaky"]) == (PageStatus.SKIPPED, PageReason.MAX_DURATION)
        assert summary.sitemap_error is None  # no sitemap source configured

    async def test_a_resumed_job_past_a_cap_stops_at_once(self, env: _Env) -> None:
        over_time = await env.job(f"{_SITE}/")
        await env.jobs.collection.update_one(
            {"_id": over_time.id}, {"$set": {"started_at": utc_now() - timedelta(hours=3)}}
        )
        over_time = await env.jobs.get_or_raise(over_time.id)
        fetcher = _ScriptedFetcher(_site_of(3))

        summary = await env.engine(over_time, fetcher).run()

        seed = (await env.outcomes(over_time))[f"{_SITE}/"]
        assert fetcher.calls == []
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.MAX_DURATION
        assert (_status(seed), seed.attempts) == ((PageStatus.SKIPPED, PageReason.MAX_DURATION), 0)

    async def test_a_resumed_job_over_its_byte_budget_stops_at_once(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_MAX_PAGE_BYTES=100, CRAWL_MAX_TOTAL_BYTES=100)
        job = await env.job(f"{_SITE}/")
        await env.jobs.increment(job.id, "run-1", bytes_downloaded=100)
        job = await env.jobs.get_or_raise(job.id)
        fetcher = _ScriptedFetcher(_site_of(3))

        summary = await env.engine(job, fetcher).run()

        assert fetcher.calls == []
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.MAX_TOTAL_BYTES
        assert summary.reasons == Counter({PageReason.MAX_TOTAL_BYTES: 1})

    async def test_page_deadline_fails_hanging_work_as_a_retryable_timeout(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(
            CRAWL_READ_TIMEOUT_SECONDS=0.05,
            CRAWL_FETCH_TIMEOUT_SECONDS=0.1,
            CRAWL_RENDER_TIMEOUT_SECONDS=0.1,
            CRAWL_PAGE_TIMEOUT_SECONDS=0.3,
        )
        hang, stuck_host = f"{_SITE}/hang", f"{_SITE}/stuck"

        async def hang_forever(url: str) -> None:
            if url == hang:
                await asyncio.Event().wait()

        class _StuckLimiter(_Limiter):
            async def acquire(
                self, url: str, *, crawl_delay: float | None = None, max_wait: float | None = None
            ) -> Reservation:
                if url == stuck_host:
                    await asyncio.Event().wait()
                return await super().acquire(url, crawl_delay=crawl_delay, max_wait=max_wait)

        responses = {
            f"{_SITE}/": _page(
                f"{_SITE}/",
                _html(f"<p>{_TEXT}</p><a href='/hang'>h</a><a href='/stuck'>s</a>"
                      "<a href='/fine'>f</a>"),
            ),
            f"{_SITE}/fine": _page(f"{_SITE}/fine", _html(f"<p>Fine. {_TEXT}</p>")),
        }  # fmt: skip
        fetcher = _ScriptedFetcher(responses, on_fetch=hang_forever)
        policy = RetryPolicy(max_attempts=2, base_delay=0.01, max_delay=0.02, max_retry_after=5)
        job = await env.job(f"{_SITE}/")

        summary = await env.engine(job, fetcher, limiter=_StuckLimiter(), policy=policy).run()

        outcomes = await env.outcomes(job)
        for url, stage in ((hang, "the fetch"), (stuck_host, "the rate-limit wait")):
            page = outcomes[url]
            assert _status(page) == (PageStatus.FAILED, PageReason.TIMEOUT), url
            assert page.attempts == 2  # retried once, like any timeout
            assert page.reason_detail is not None and stage in page.reason_detail
        assert fetcher.calls.count(hang) == 2 and stuck_host not in fetcher.calls
        assert _status(outcomes[f"{_SITE}/fine"]) == (PageStatus.SUCCEEDED, None)
        assert summary.stop is None


# ---- cancellation (Step 52) ---- #


@pytest.fixture
async def flags(env: _Env, integration_settings: Settings) -> AsyncIterator[RedisCancelFlags]:
    redis = RedisConnection(integration_settings)
    await redis.connect()
    yield RedisCancelFlags(redis.client, redis.keys, env.config)
    await redis.close()


@dataclass
class _Signal:
    """`CancelSignal` that turns on when `on` is set; `errors` polls fail first."""

    on: bool = False
    errors: int = 0
    polls: int = 0

    async def is_cancelled(self, job_id: str) -> bool:
        self.polls += 1
        if self.errors:
            self.errors -= 1
            raise RedisUnavailableError(details={"reason": "ConnectionError"})
        return self.on


class TestCrawlCancellation_S52:
    """S52 — Cancellation: the Redis flag is polled, in-flight pages drain, job cancelled."""

    async def test_cancel_flag_drains_in_flight_pages_and_skips_the_rest(
        self, env: _Env, flags: RedisCancelFlags, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_CANCEL_POLL_SECONDS=0.05)
        job = await env.job(f"{_SITE}/")
        slow = f"{_SITE}/p3"
        responses = _site_of(20)
        responses[slow] = _page(
            slow, _html(f"<p>Slow page. {_TEXT}</p><a href='/found-late'>l</a>")
        )

        async def cancel_via_api(url: str) -> None:
            if url == slow:
                await request_cancel(env.jobs, flags, _TENANT, job.id)
                await asyncio.sleep(0.3)  # still fetching when the poller sees the flag
            else:
                await asyncio.sleep(0.05)

        fetcher = _ScriptedFetcher(responses, on_fetch=cancel_via_api)

        summary = await env.engine(job, fetcher, cancel=flags).run()
        cancelled = await finish_cancelled(env.jobs, flags, job.id, "run-1")

        outcomes = await env.outcomes(job)
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.CANCELLED
        assert 1 < len(fetcher.calls) < 21
        # drained: every page whose fetch started was finished normally
        assert all(_status(outcomes[url]) == (PageStatus.SUCCEEDED, None) for url in fetcher.calls)
        assert outcomes[slow].markdown is not None
        late = outcomes[f"{_SITE}/found-late"]  # a link of the in-flight page
        assert _status(late) == (PageStatus.SKIPPED, PageReason.JOB_CANCELLED)
        rest = [p for url, p in outcomes.items() if url not in fetcher.calls]
        assert rest and {_status(p) for p in rest} == {
            (PageStatus.SKIPPED, PageReason.JOB_CANCELLED)
        }
        assert not {PageStatus.QUEUED, PageStatus.IN_PROGRESS} & set(summary.pages)
        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.counters.skipped == len(rest)
        assert await flags.is_cancelled(job.id) is False  # cleared

    async def test_a_cancel_requested_before_the_run_stops_without_fetching(
        self, env: _Env
    ) -> None:
        job = await env.job(f"{_SITE}/")
        job = await env.jobs.request_cancel(_TENANT, job.id)  # Mongo only: no Redis flag
        fetcher = _ScriptedFetcher(_site_of(3))
        signal = _Signal()

        summary = await env.engine(job, fetcher, cancel=signal).run()

        seed = (await env.outcomes(job))[f"{_SITE}/"]
        assert fetcher.calls == []
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.CANCELLED
        assert "before this run started" in summary.stop.detail
        assert (_status(seed), seed.attempts) == ((PageStatus.SKIPPED, PageReason.JOB_CANCELLED), 0)

    async def test_cancel_outranks_a_cap_and_wakes_idle_workers(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_CANCEL_POLL_SECONDS=0.05)
        flaky = f"{_SITE}/flaky"
        responses: dict[str, FetchResult | Exception] = {
            f"{_SITE}/": _page(f"{_SITE}/", _html(f"<p>{_TEXT}</p><a href='/flaky'>f</a>")),
            flaky: HttpStatusError(url=flaky, status_code=503),
        }
        signal = _Signal()

        async def cancel_soon(url: str) -> None:
            if url == flaky:
                asyncio.get_running_loop().call_later(0.2, setattr, signal, "on", True)

        fetcher = _ScriptedFetcher(responses, on_fetch=cancel_soon)
        # the budget is spent (soft stop) and the flaky page waits 30 s for its retry
        policy = RetryPolicy(max_attempts=3, base_delay=30, max_delay=30, max_retry_after=5)
        job = await env.job(f"{_SITE}/", max_pages=2)
        started = time.monotonic()

        summary = await env.engine(job, fetcher, cancel=signal, policy=policy).run()

        assert time.monotonic() - started < 3
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.CANCELLED
        outcomes = await env.outcomes(job)
        assert _status(outcomes[flaky]) == (PageStatus.SKIPPED, PageReason.JOB_CANCELLED)

    async def test_the_engine_can_be_cancelled_directly(self, env: _Env) -> None:
        job = await env.job(f"{_SITE}/")
        engines: list[CrawlEngine] = []
        results: list[bool] = []

        async def heartbeat_saw_cancel(url: str) -> None:
            if url.endswith("/p1"):
                results.extend([engines[0].cancel(), engines[0].cancel()])

        fetcher = _ScriptedFetcher(_site_of(10), on_fetch=heartbeat_saw_cancel)
        engines.append(env.engine(job, fetcher))

        summary = await engines[0].run()

        assert results == [True, False]  # the second call changes nothing
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.CANCELLED
        assert summary.reasons[PageReason.JOB_CANCELLED] >= 1
        assert len(fetcher.calls) < 11

    async def test_a_redis_outage_during_polls_is_tolerated(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_CANCEL_POLL_SECONDS=0.02)
        signal = _Signal(errors=3)

        async def slow(url: str) -> None:
            if url.endswith("/p0"):
                signal.on = True  # Redis back, flag set
            await asyncio.sleep(0.1)

        fetcher = _ScriptedFetcher(_site_of(10), on_fetch=slow)
        job = await env.job(f"{_SITE}/")

        summary = await env.engine(job, fetcher, cancel=signal).run()

        assert signal.errors == 0 and signal.polls > 3
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.CANCELLED

    async def test_the_poller_never_delays_the_end_of_a_crawl(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_CANCEL_POLL_SECONDS=30)
        signal = _Signal()
        job = await env.job(f"{_SITE}/")
        started = time.monotonic()

        summary = await env.engine(job, _ScriptedFetcher(_site_of(2)), cancel=signal).run()

        assert time.monotonic() - started < 5
        assert summary.stop is None
        assert signal.polls == 1  # the start check only


# ---- error accounting (Step 53) ---- #


def _pdf(url: str, body: bytes) -> FetchResult:
    return replace(
        _page(url, ""),
        mime_type="application/pdf",
        content_kind=ContentKind.PDF,
        body=body,
        text=None,
        encoding=None,
        bytes_downloaded=len(body),
        headers={"content-type": "application/pdf"},
    )


@dataclass
class _FlakyRedisLimiter(_Limiter):
    """Raises a Redis connection error for `broken` URLs (`times` per URL, -1 = always)."""

    broken: dict[str, int] = field(default_factory=dict)

    async def acquire(
        self, url: str, *, crawl_delay: float | None = None, max_wait: float | None = None
    ) -> Reservation:
        left = self.broken.get(url, 0)
        if left:
            self.broken[url] = left - 1 if left > 0 else left
            raise RedisConnectionError("Error 111 connecting to redis:6379")
        return await super().acquire(url, crawl_delay=crawl_delay, max_wait=max_wait)


class _DownRobots:
    async def rules_for(self, url: str) -> RobotsRules:
        raise RedisConnectionError("Error 111 connecting to redis:6379")


# (path, response, expected status, expected reason)
_FAILURES: list[tuple[str, Callable[[str], FetchResult | Exception], PageStatus, PageReason]] = [
    ("/gone", lambda u: HttpStatusError(url=u, status_code=404),
     PageStatus.FAILED, PageReason.HTTP_CLIENT_ERROR),
    ("/broken", lambda u: HttpStatusError(url=u, status_code=500),
     PageStatus.FAILED, PageReason.HTTP_SERVER_ERROR),
    ("/throttled", lambda u: HttpStatusError(url=u, status_code=429),
     PageStatus.FAILED, PageReason.RATE_LIMITED),
    ("/slow", lambda u: FetchTimeoutError("read timeout", url=u),
     PageStatus.FAILED, PageReason.TIMEOUT),
    ("/down", lambda u: NetworkFetchError("connection reset", url=u),
     PageStatus.FAILED, PageReason.NETWORK_ERROR),
    ("/internal", lambda u: SsrfFetchError("private address", url=u),
     PageStatus.FAILED, PageReason.SSRF_BLOCKED),
    ("/huge", lambda u: TooLargeError("over 20 MB", url=u),
     PageStatus.FAILED, PageReason.TOO_LARGE),
    ("/loop", lambda u: RedirectLoopFetchError("a -> b -> a", url=u),
     PageStatus.FAILED, PageReason.REDIRECT_LOOP),
    ("/hops", lambda u: TooManyRedirectsFetchError("6 hops", url=u),
     PageStatus.FAILED, PageReason.TOO_MANY_REDIRECTS),
    ("/video", lambda u: UnsupportedContentTypeError("video", url=u, content_type="video/mp4"),
     PageStatus.SKIPPED, PageReason.UNSUPPORTED_CONTENT_TYPE),
    ("/blank", lambda u: _page(u, _html("")),
     PageStatus.SKIPPED, PageReason.EMPTY_CONTENT),
    ("/shell", lambda u: replace(_page(u, _html("")), render_note="render_failed: crashed"),
     PageStatus.FAILED, PageReason.RENDER_FAILED),
    ("/corrupt.pdf", lambda u: _pdf(u, b"%PDF-1.7 this is not really a pdf"),
     PageStatus.FAILED, PageReason.EXTRACTION_FAILED),
    ("/bug", lambda u: _page(u, _html(f"<p>BOOM {_TEXT}</p>")),
     PageStatus.FAILED, PageReason.INTERNAL_ERROR),
    ("/redis", lambda u: _page(u, _html(f"<p>Never reached. {_TEXT}</p>")),
     PageStatus.FAILED, PageReason.SERVICE_UNAVAILABLE),
]  # fmt: skip


class TestEngineErrorAccounting_S53:
    """S53 — Error accounting: every failure is a classified page row; the job's report."""

    async def test_every_failure_kind_ends_as_a_classified_page_row(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_MAX_RETRIES=1)  # 2 attempts for retryable failures
        links = "".join(f"<a href='{path}'>x</a>" for path, *_ in _FAILURES)
        responses: dict[str, FetchResult | Exception] = {
            f"{_SITE}/": _page(f"{_SITE}/", _html(f"<p>{_TEXT}</p>{links}")),
            **{f"{_SITE}{path}": respond(f"{_SITE}{path}") for path, respond, *_ in _FAILURES},
        }
        limiter = _FlakyRedisLimiter(broken={f"{_SITE}/redis": -1})
        policy = RetryPolicy(max_attempts=2, base_delay=0.01, max_delay=0.02, max_retry_after=5)
        job = await env.job(f"{_SITE}/")

        summary = await env.engine(
            job,
            _ScriptedFetcher(responses),
            limiter=limiter,
            policy=policy,
            cleaner=_ExplodingCleaner(config=env.config),
        ).run()

        outcomes = await env.outcomes(job)
        for path, _, status, reason in _FAILURES:
            page = outcomes[f"{_SITE}{path}"]
            assert _status(page) == (status, reason), (path, page.reason_detail)
            assert page.reason_detail, path
        assert outcomes[f"{_SITE}/redis"].attempts == 2
        assert "gave up after 2 attempts" in (outcomes[f"{_SITE}/redis"].reason_detail or "")
        assert outcomes[f"{_SITE}/gone"].http_status == 404
        # the report: failures and skips by reason, consistent with the page table
        report = summary.report
        assert report is not None and report.unfinished == 0
        expected_failures = Counter(r for _, _, s, r in _FAILURES if s is PageStatus.FAILED)
        assert {row.reason: row.count for row in report.failures} == expected_failures
        assert {row.reason: row.count for row in report.skips} == {
            PageReason.UNSUPPORTED_CONTENT_TYPE: 1,
            PageReason.EMPTY_CONTENT: 1,
        }
        gone = next(r for r in report.failures if r.reason is PageReason.HTTP_CLIENT_ERROR)
        assert [(s.url, s.http_status) for s in gone.samples] == [(f"{_SITE}/gone", 404)]
        assert report.failed == len(expected_failures) == summary.pages[PageStatus.FAILED]
        stored = (await env.jobs.get_or_raise(job.id)).report
        assert stored == report
        counters = (await env.jobs.get_or_raise(job.id)).counters
        assert (counters.failed, counters.skipped) == (len(expected_failures), 2)

    async def test_a_redis_blip_is_retried_and_the_page_recovers(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_MAX_RETRIES=2)
        limiter = _FlakyRedisLimiter(broken={f"{_SITE}/p0": 1})
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02, max_retry_after=5)
        job = await env.job(f"{_SITE}/")

        await env.engine(job, _ScriptedFetcher(_site_of(2)), limiter=limiter, policy=policy).run()

        page = (await env.outcomes(job))[f"{_SITE}/p0"]
        assert (_status(page), page.attempts) == ((PageStatus.SUCCEEDED, None), 2)

    async def test_a_failing_redirect_target_never_stays_in_progress(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        # one worker: nobody else can claim /new before the redirect handler does
        override_settings(CRAWL_CONCURRENCY=1)
        old, new = f"{_SITE}/old", f"{_SITE}/new"
        responses = {
            f"{_SITE}/": _page(f"{_SITE}/", _html(f"<p>{_TEXT}</p><a href='/old'>o</a>")),
            old: _page(old, _html(f"<p>BOOM {_TEXT}</p>"), final_url=new, chain=(old,)),
        }
        job = await env.job(f"{_SITE}/")
        started = time.monotonic()

        summary = await env.engine(
            job, _ScriptedFetcher(responses), cleaner=_ExplodingCleaner(config=env.config)
        ).run()

        outcomes = await env.outcomes(job)
        assert time.monotonic() - started < 5  # not waiting for a 10-minute lease
        assert _status(outcomes[new]) == (PageStatus.FAILED, PageReason.INTERNAL_ERROR)
        assert _status(outcomes[old]) == (PageStatus.SKIPPED, PageReason.REDIRECTED)
        assert summary.report is not None and summary.report.unfinished == 0

    async def test_counters_are_reconciled_with_the_page_table(self, env: _Env) -> None:
        job = await env.job(f"{_SITE}/")
        # a page a crashed run claimed on every attempt: recover() fails it as worker_lost,
        # which never passes through the engine's counter updates
        await env.pages.insert(
            Page(tenant_id=_TENANT, kb_id=_KB, source_id=_SOURCE, job_id=job.id,
                 url=f"{_SITE}/cursed", url_hash=normalize_and_hash(f"{_SITE}/cursed")[1],
                 status=PageStatus.IN_PROGRESS, claimed_by="run-0", attempts=99)
        )  # fmt: skip
        await env.jobs.increment(job.id, "run-1", skipped=5)  # a counter that drifted up

        summary = await env.engine(job, _ScriptedFetcher(_site_of(2))).run()

        cursed = (await env.outcomes(job))[f"{_SITE}/cursed"]
        counters = (await env.jobs.get_or_raise(job.id)).counters
        assert _status(cursed) == (PageStatus.FAILED, PageReason.WORKER_LOST)
        assert (counters.failed, counters.skipped) == (1, 0)
        assert summary.report is not None and summary.report.abandoned_pages == 1

    async def test_a_robots_store_outage_fails_pages_without_stopping_the_crawl(
        self, env: _Env, override_settings: OverrideSettings
    ) -> None:
        override_settings(CRAWL_MAX_RETRIES=1)
        policy = RetryPolicy(max_attempts=2, base_delay=0.01, max_delay=0.02, max_retry_after=5)
        job = await env.job(f"{_SITE}/")
        fetcher = _ScriptedFetcher(_site_of(2))

        summary = await env.engine(job, fetcher, robots=_DownRobots(), policy=policy).run()

        seed = (await env.outcomes(job))[f"{_SITE}/"]
        assert _status(seed) == (PageStatus.FAILED, PageReason.SERVICE_UNAVAILABLE)
        assert fetcher.calls == []  # never fetched without a robots verdict
        report = summary.report
        assert report is not None and report.robots_status is None
        assert (report.robots_detail or "").startswith("robots.txt verdict unavailable")
