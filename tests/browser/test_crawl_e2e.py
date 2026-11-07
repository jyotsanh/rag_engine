"""Crawl engine E2E: the whole fixture site, wired like production, with a real browser.

Real headless Chromium (JS shells re-rendered), real MongoDB and Redis (robots cache, rate
limiter, cancel flag), the fixture website over loopback HTTP. Needs Chromium *and* Docker;
skipped when either is missing. Marked `browser` (directory) and `integration`.
"""

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import pytest

from rag_engine.config import JsRenderMode, Settings
from rag_engine.crawling import (
    CrawlEngine,
    CrawlSummary,
    DomainRateLimiter,
    RedisCancelFlags,
    RedisRobotsStore,
    RobotsCache,
    RobotsStatus,
    SitemapDiscoverer,
    finish_cancelled,
    normalize_and_hash,
    render_politeness,
    request_cancel,
)
from rag_engine.crawling.fetchers import AutoFetcher, BrowserPool, HttpFetcher
from rag_engine.domain import (
    ContentHashClaim,
    ContentKind,
    CrawlConfig,
    CrawlJob,
    CrawlStopReason,
    JobStatus,
    Page,
    PageReason,
    PageStatus,
    RenderedWith,
)
from rag_engine.infrastructure import MongoConnection, RedisConnection
from rag_engine.repositories import ContentClaimRepository, JobRepository, PageRepository
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import (
    EXPECTED_PAGE_OUTCOME,
    SITE_PAGES,
    FixtureSite,
    assert_crawl_outcomes,
)

pytestmark = pytest.mark.integration

_TENANT = "ten_" + "1" * 32
_KB = "kb_" + "1" * 32
_SOURCE = "src_" + "1" * 32


@dataclass
class _Crawl:
    """The production wiring Step 77 will do, around one job."""

    site: FixtureSite
    config: Settings
    mongo: MongoConnection
    redis: RedisConnection
    pool: BrowserPool
    http: HttpFetcher

    def __post_init__(self) -> None:
        self.pages = PageRepository(self.mongo.db)
        self.jobs = JobRepository(self.mongo.db)
        self.robots = RobotsCache(
            self.http, RedisRobotsStore(self.redis.client, self.redis.keys), config=self.config
        )
        self.limiter = DomainRateLimiter(self.redis.client, self.redis.keys, self.config)

    async def run(
        self,
        *,
        during: Callable[[CrawlJob], Coroutine[Any, Any, None]] | None = None,
        **config: object,
    ) -> tuple[CrawlJob, CrawlSummary, dict[str, Page]]:
        """Crawl the site as one job; `during(job)` runs alongside the engine (e.g. a cancel)."""
        job = await self.jobs.insert(
            CrawlJob(
                tenant_id=_TENANT,
                kb_id=_KB,
                source_id=_SOURCE,
                seed_url=self.site.url("/"),
                config=CrawlConfig.model_validate({"max_depth": 10, **config}),
            )
        )
        job = await self.jobs.start(job.id, "run-1")
        async with self.pool.job_context(job_id=job.id) as browser:
            fetcher = AutoFetcher(
                self.http,
                browser,
                config=self.config,
                before_render=render_politeness(self.robots, self.limiter),
            )
            engine = CrawlEngine(
                job,
                run_id="run-1",
                pages=self.pages,
                jobs=self.jobs,
                claims=ContentClaimRepository(self.mongo.db),
                fetcher=fetcher,
                robots=self.robots,
                limiter=self.limiter,
                sitemaps=SitemapDiscoverer(
                    self.http, robots=self.robots, limiter=self.limiter, config=self.config
                ),
                cancel=RedisCancelFlags(self.redis.client, self.redis.keys, self.config),
                config=self.config,
            )
            if during is None:
                summary = await engine.run()
            else:
                async with asyncio.TaskGroup() as group:
                    group.create_task(during(job))
                    crawling = group.create_task(engine.run())
                summary = crawling.result()
        pages = {p.url: p for p in await self.pages.find({"job_id": job.id})}
        return await self.jobs.get_or_raise(job.id), summary, pages

    def page(self, pages: dict[str, Page], path: str) -> Page:
        return pages[normalize_and_hash(self.site.url(path))[0]]


@pytest.fixture
async def crawl(
    override_settings: OverrideSettings,
    integration_settings: Settings,
    fixture_site: FixtureSite,
) -> AsyncIterator[_Crawl]:
    config = override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True,
        CRAWL_JS_RENDERING=JsRenderMode.AUTO,
        CRAWL_RENDER_NETWORKIDLE_SECONDS=2,
        CRAWL_RENDER_SCROLL_SECONDS=0.5,
        CRAWL_CONCURRENCY=4,
        CRAWL_PER_DOMAIN_RPS=50.0,  # robots.txt's Crawl-delay (0.1 s) sets the pace
        CRAWL_READ_TIMEOUT_SECONDS=1,
        CRAWL_FETCH_TIMEOUT_SECONDS=2,
        CRAWL_MAX_RETRIES=2,
        CRAWL_RETRY_BASE_DELAY_SECONDS=0.05,
        CRAWL_RETRY_MAX_DELAY_SECONDS=0.1,
        CRAWL_RETRY_AFTER_MAX_SECONDS=1,
        CRAWL_CANCEL_POLL_SECONDS=0.2,
    )
    mongo = MongoConnection(config)
    redis = RedisConnection(config)
    await mongo.connect()
    await mongo.ensure_indexes([Page, CrawlJob, ContentHashClaim])
    await redis.connect()
    pool = BrowserPool(config)
    try:
        async with SafeHttpClient(config) as client:
            yield _Crawl(fixture_site, config, mongo, redis, pool, HttpFetcher(client, config))
    finally:
        await pool.stop()
        await redis.close()
        await mongo.close()


# ---- #
# Tests
# ---- #


class TestCrawlEngineE2E_S54:
    """S54 — Crawl engine E2E: the full fixture site, real browser, every page's outcome."""

    async def test_every_fixture_page_ends_as_expected(self, crawl: _Crawl) -> None:
        # Act
        job, summary, pages = await crawl.run()

        # Assert: every page of the site, the SPA included
        assert_crawl_outcomes(crawl.site, pages, rendered=True)
        assert not {PageStatus.QUEUED, PageStatus.IN_PROGRESS} & {p.status for p in pages.values()}

        # the JS shell was rendered, and only JS shells were
        app = crawl.page(pages, "/app/")
        assert app.rendered_with is RenderedWith.BROWSER
        assert app.markdown is not None and "Widget configurator" in app.markdown
        assert "empty_mount" in app.js_signals
        rendered = {url for url, p in pages.items() if p.rendered_with is RenderedWith.BROWSER}
        assert rendered == {app.url}
        advanced = crawl.page(pages, "/app/advanced")  # linked only from the rendered DOM
        assert advanced.status is PageStatus.SUCCEEDED
        assert advanced.discovered_from == app.url

        # the plan's named cases
        loop = crawl.page(pages, "/loop-a")
        assert (loop.status, loop.reason) == (PageStatus.FAILED, PageReason.REDIRECT_LOOP)
        missing = crawl.page(pages, "/missing.html")
        assert (missing.status, missing.reason, missing.http_status) == (
            PageStatus.FAILED,
            PageReason.HTTP_CLIENT_ERROR,
            404,
        )
        pdf = crawl.page(pages, "/files/widget-spec.pdf")
        assert pdf.content_kind is ContentKind.PDF
        assert pdf.markdown is not None and "1200 items per minute" in pdf.markdown
        offsite = pages["https://offsite.example/page"]
        assert (offsite.status, offsite.reason) == (PageStatus.SKIPPED, PageReason.OUT_OF_SCOPE)
        secret = crawl.page(pages, "/private/secret.html")
        assert (secret.status, secret.reason) == (PageStatus.SKIPPED, PageReason.ROBOTS_DISALLOWED)

        # requests: robots respected, the shell fetched by HTTP then once more by the browser
        requested = Counter(r.path for r in crawl.site.requests)
        assert requested["/private/secret.html"] == 0
        assert requested["/app/"] == 2
        assert requested["/app/advanced"] == 1

        # the job: counters and the stored report agree with the page table
        assert summary.stop is None
        report = job.report
        assert report is not None and report == summary.report
        by_status = Counter(p.status for p in pages.values())
        assert report.pages == dict(by_status) and report.unfinished == 0
        assert report.robots_status == RobotsStatus.OK
        assert report.sitemap_urls and report.sitemap_urls > 0
        failures = {row.reason: row.count for row in report.failures}
        expected = Counter(
            EXPECTED_PAGE_OUTCOME[e][1]
            for e in SITE_PAGES.values()
            if EXPECTED_PAGE_OUTCOME[e][0] is PageStatus.FAILED
        )
        assert failures == expected
        assert sum(failures.values()) == by_status[PageStatus.FAILED]
        assert (job.counters.failed, job.counters.skipped, job.counters.duplicates) == (
            by_status[PageStatus.FAILED],
            by_status[PageStatus.SKIPPED],
            by_status[PageStatus.DUPLICATE],
        )
        assert job.counters.bytes_downloaded > 0

    async def test_a_cancel_through_the_api_drains_and_cancels_the_job(self, crawl: _Crawl) -> None:
        flags = RedisCancelFlags(crawl.redis.client, crawl.redis.keys, crawl.config)

        async def cancel_after_some_pages(job: CrawlJob) -> None:
            for _ in range(1500):  # the server runs in a thread: poll its request log (≤ 30 s)
                if len(crawl.site.requests) >= 12:
                    break
                await asyncio.sleep(0.02)
            await request_cancel(crawl.jobs, flags, _TENANT, job.id)

        # Act
        job, summary, pages = await crawl.run(during=cancel_after_some_pages)
        job = await finish_cancelled(crawl.jobs, flags, job.id, "run-1")

        # Assert
        assert summary.stop is not None and summary.stop.reason is CrawlStopReason.CANCELLED
        assert job.status is JobStatus.CANCELLED
        assert await flags.is_cancelled(job.id) is False
        statuses = Counter(p.status for p in pages.values())
        assert not {PageStatus.QUEUED, PageStatus.IN_PROGRESS} & set(statuses)
        cancelled = [p for p in pages.values() if p.reason is PageReason.JOB_CANCELLED]
        assert cancelled and all(p.fetched_at is None for p in cancelled)  # never fetched
        assert statuses[PageStatus.SUCCEEDED] >= 1  # pages in flight drained normally
        assert job.report is not None and job.report.stop_reason is CrawlStopReason.CANCELLED
