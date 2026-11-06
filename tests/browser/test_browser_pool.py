"""Playwright browser pool: lazy per-process Chromium, context per job, blocked resources."""

import asyncio
import os
import socket
import time
from collections.abc import AsyncIterator, Callable

import pytest
from playwright.async_api import Request

import rag_engine.crawling.fetchers.browser as browser_module
from rag_engine.config import Settings
from rag_engine.crawling.fetchers import (
    BrowserJobContext,
    BrowserPool,
    BrowserPoolError,
    RenderError,
    process_browser_pool,
    shutdown_process_browser_pool,
)
from rag_engine.domain import PageReason
from tests.conftest import OverrideSettings
from tests.support.fixture_site import FixtureSite

type MakePool = Callable[..., BrowserPool]


@pytest.fixture
async def make_pool(override_settings: OverrideSettings) -> AsyncIterator[MakePool]:
    pools: list[BrowserPool] = []

    def make(**overrides: object) -> BrowserPool:
        # the fixture site lives on loopback, which the browser's SSRF guard (S35) blocks
        config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True, **overrides)
        pool = BrowserPool(config)
        pools.append(pool)
        return pool

    yield make
    for pool in pools:
        await pool.stop()


@pytest.fixture
async def pool(make_pool: MakePool) -> BrowserPool:
    return make_pool()


# ---- #
# Tests
# ---- #


class TestBrowserPool_S34:
    """S34 — Playwright browser pool: per-process Chromium, job contexts, resource blocking."""

    # ---- rendering ---- #

    async def test_renders_the_fixture_spa(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context(job_id="job_1") as job:
            page = await job.render(fixture_site.url("/app/"))

        assert page.status_code == 200
        assert page.final_url == fixture_site.url("/app/")
        assert "<h1>Widget configurator</h1>" in page.html  # produced by JavaScript
        assert 'href="/app/advanced"' in page.html
        assert page.elapsed_ms > 0

    async def test_browser_sends_our_user_agent(
        self, pool: BrowserPool, fixture_site: FixtureSite, config_user_agent: str
    ) -> None:
        async with pool.job_context() as job:
            await job.render(fixture_site.url("/about.html"))

        agents = {r.user_agent for r in fixture_site.requests if r.path == "/about.html"}
        assert agents == {config_user_agent}

    # ---- resource blocking ---- #

    async def test_images_fonts_and_media_are_blocked(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job:
            page = await job.render(fixture_site.url("/browser/resources.html"))
            stats = job.stats

        assert 'data-js="ran"' in page.html  # scripts still run
        assert fixture_site.hits("/browser/style.css") == 1  # stylesheets still load
        for blocked in ("/browser/logo.png", "/browser/font.woff2", "/browser/clip.mp4"):
            assert fixture_site.hits(blocked) == 0, blocked
        assert stats.blocked["resource:image"] == 1
        assert stats.blocked["resource:font"] == 1
        assert stats.blocked["resource:media"] >= 1
        assert stats.requests >= 6

    async def test_blocked_types_are_configurable(
        self, make_pool: MakePool, fixture_site: FixtureSite
    ) -> None:
        pool = make_pool(CRAWL_BROWSER_BLOCKED_RESOURCES=["stylesheet", "script"])

        async with pool.job_context() as job:
            page = await job.render(fixture_site.url("/browser/resources.html"))

        assert fixture_site.hits("/browser/style.css") == 0
        assert fixture_site.hits("/browser/app.js") == 0
        assert 'data-js="ran"' not in page.html
        assert fixture_site.hits("/browser/logo.png") == 1

    async def test_request_guard_can_abort_requests(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        seen: list[str] = []

        async def guard(request: Request) -> str | None:
            seen.append(request.url)
            return "denied" if request.url.endswith(".css") else None

        async with pool.job_context(request_guard=guard) as job:
            await job.render(fixture_site.url("/browser/resources.html"))
            stats = job.stats

        assert fixture_site.hits("/browser/style.css") == 0
        assert stats.blocked["guard:denied"] == 1
        assert fixture_site.url("/browser/resources.html") in seen  # documents are guarded too
        assert not any(url.endswith(".png") for url in seen)  # blocked types never reach it

    # ---- isolation & lifecycle ---- #

    async def test_each_job_gets_an_isolated_context(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        url = fixture_site.url("/about.html")

        async with pool.job_context(job_id="a") as job_a:
            async with job_a.page() as page:
                await page.goto(url)
                await page.evaluate("document.cookie = 'session=tenant-a; path=/'")
                assert "tenant-a" in await page.evaluate("document.cookie")
            async with job_a.page() as page:  # same job: shared cookie jar
                await page.goto(url)
                assert "tenant-a" in await page.evaluate("document.cookie")
            async with pool.job_context(job_id="b") as job_b, job_b.page() as page:
                await page.goto(url)
                assert await page.evaluate("document.cookie") == ""

    async def test_contexts_and_pages_are_closed(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job:
            assert pool.active_contexts == 1
            async with job.page() as page:
                await page.goto(fixture_site.url("/about.html"))
            assert job.context.pages == []

        browser = await pool.browser()
        assert pool.active_contexts == 0
        assert browser.contexts == []

    async def test_context_is_closed_when_the_job_fails(self, pool: BrowserPool) -> None:
        with pytest.raises(ValueError, match="boom"):
            async with pool.job_context():
                raise ValueError("boom")

        assert pool.active_contexts == 0
        assert (await pool.browser()).contexts == []

    async def test_launch_is_lazy_and_lifecycle_idempotent(self, pool: BrowserPool) -> None:
        assert not pool.is_running and pool.generation == 0

        await pool.start()
        await pool.start()
        assert pool.is_running and pool.generation == 1

        await pool.stop()
        await pool.stop()
        assert not pool.is_running

    async def test_crashed_browser_is_relaunched(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        await (await pool.browser()).close()  # simulate a Chromium crash
        assert not pool.is_running

        async with pool.job_context() as job:
            page = await job.render(fixture_site.url("/about.html"))

        assert page.status_code == 200
        assert pool.generation == 2

    async def test_browser_is_recycled_when_idle(self, make_pool: MakePool) -> None:
        pool = make_pool(CRAWL_BROWSER_RECYCLE_AFTER_CONTEXTS=2)

        async with pool.job_context():
            pass
        async with pool.job_context(), pool.job_context():  # due, but busy: no relaunch yet
            assert pool.generation == 1
        async with pool.job_context():  # idle and due: relaunched first
            assert pool.generation == 2

    async def test_page_concurrency_is_limited(
        self, make_pool: MakePool, fixture_site: FixtureSite
    ) -> None:
        pool = make_pool(CRAWL_BROWSER_MAX_PAGES=2)
        open_now = peak = 0

        async def visit(job: BrowserJobContext) -> None:
            nonlocal open_now, peak
            async with job.page() as page:
                open_now += 1
                peak = max(peak, open_now)
                await page.goto(fixture_site.url("/about.html"))
                await asyncio.sleep(0.1)
                open_now -= 1

        async with pool.job_context() as job:
            await asyncio.gather(*(visit(job) for _ in range(6)))

        assert peak == 2

    # ---- failures ---- #

    async def test_page_timeout(self, make_pool: MakePool, fixture_site: FixtureSite) -> None:
        pool = make_pool(CRAWL_RENDER_TIMEOUT_SECONDS=1)

        async with pool.job_context() as job:  # launch first: only the render is timed
            start = time.perf_counter()
            with pytest.raises(RenderError) as exc:
                await job.render(fixture_site.url("/slow?delay=10"))

        assert exc.value.detail == "timeout"
        assert exc.value.reason is PageReason.RENDER_FAILED
        assert time.perf_counter() - start < 4

    async def test_unreachable_page_is_a_render_error(self, pool: BrowserPool) -> None:
        # a just-freed port refuses connections (Chromium rejects "unsafe" ports like 1 itself)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        async with pool.job_context() as job:
            with pytest.raises(RenderError) as exc:
                await job.render(f"http://127.0.0.1:{port}/")

        # S35: the request is made by our client; its failure reaches the page as an abort
        assert "net::ERR_" in str(exc.value)
        assert exc.value.reason is PageReason.RENDER_FAILED
        assert job.stats.blocked["network"] == 1

    # ---- process binding ---- #

    async def test_pool_refuses_a_second_event_loop(self, pool: BrowserPool) -> None:
        other_loop = asyncio.Runner()
        await asyncio.to_thread(other_loop.run, pool.start())  # bound to the other loop
        try:
            with pytest.raises(BrowserPoolError, match="another event loop"):
                await pool.browser()
        finally:
            await asyncio.to_thread(other_loop.run, pool.stop())
            await asyncio.to_thread(other_loop.close)
        pool._reset_state()  # stopped on its own loop; forget it so teardown is a no-op

    async def test_pool_starts_fresh_after_fork(
        self, pool: BrowserPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent_browser = await pool.browser()
        parent_driver = pool._playwright
        real_pid = os.getpid()

        monkeypatch.setattr(browser_module.os, "getpid", lambda: real_pid + 1)
        child_browser = await pool.browser()

        assert child_browser is not parent_browser
        assert pool.generation == 1  # state was reset, then one launch in the "child"
        # a real child never touches the parent's objects; here we tidy them up ourselves
        await parent_browser.close()
        assert parent_driver is not None
        await parent_driver.stop()

    async def test_process_pool_singleton(self) -> None:
        first, second = process_browser_pool(), process_browser_pool()
        await first.start()

        await shutdown_process_browser_pool()

        assert first is second
        assert not first.is_running
        assert process_browser_pool() is not first
        await shutdown_process_browser_pool()


@pytest.fixture
def config_user_agent(override_settings: OverrideSettings) -> str:
    return override_settings().CRAWL_USER_AGENT
