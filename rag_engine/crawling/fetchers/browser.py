"""Headless Chromium for JS-rendered pages: one browser per worker process, a context per job.

`BrowserPool` launches Chromium lazily on first use and keeps it for the life of the
worker process:
- **Fork-safe**: a pool inherited through `fork()` (Celery prefork) notices the new pid
  and launches its own browser; the parent's Playwright objects are never touched.
- **One event loop**: Playwright objects belong to the loop that created them. Using
  the pool from another loop raises `BrowserPoolError` instead of hanging. Step 75 runs
  each worker process on one persistent `asyncio.Runner`.
- **Self-healing**: a crashed/disconnected browser is relaunched on the next use, and the
  browser is recycled (relaunched while idle) after `CRAWL_BROWSER_RECYCLE_AFTER_CONTEXTS`
  job contexts to bound Chromium's memory creep.
- At most `CRAWL_BROWSER_MAX_PAGES` pages are open at once per process.

`BrowserPool.job_context()` gives each crawl job an isolated browser context (cookies,
storage, cache) with our User-Agent. Every request is routed through one handler that
aborts `CRAWL_BROWSER_BLOCKED_RESOURCES` types (image/font/media by default) and then
asks an optional `request_guard`. Step 35 plugs the SSRF guard in there, and **until it
does the browser must only be pointed at trusted URLs**.
"""

import asyncio
import os
import time
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Request,
    Route,
    ViewportSize,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from rag_engine.config import Settings, settings
from rag_engine.crawling.fetchers.errors import FetchError
from rag_engine.domain import PageReason

# returns a reason to abort the request, or None to let it through
type RequestGuard = Callable[[Request], Awaitable[str | None]]
type WaitUntil = Literal["commit", "domcontentloaded", "load", "networkidle"]

VIEWPORT: ViewportSize = {"width": 1280, "height": 900}
_LAUNCH_ARGS = (
    "--disable-dev-shm-usage",  # containers ship a tiny /dev/shm
    "--disable-background-networking",
    "--disable-component-update",
    "--no-first-run",
)


class BrowserPoolError(RuntimeError):
    """The pool was used in a way it cannot support (e.g. from a second event loop)."""


class RenderError(FetchError):
    """The browser could not load or render the page (timeout, crash, net error)."""

    reason = PageReason.RENDER_FAILED


@dataclass(frozen=True, slots=True)
class RenderedPage:
    requested_url: str
    final_url: str
    status_code: int | None  # main document response; None if no response (e.g. about:blank)
    html: str
    elapsed_ms: int


@dataclass(slots=True)
class ContextStats:
    requests: int = 0  # requests seen by the router (incl. aborted ones)
    blocked: Counter[str] = field(default_factory=Counter)  # "resource:image", "guard:<reason>"


class BrowserJobContext:
    """One crawl job's isolated browser context."""

    def __init__(
        self, pool: BrowserPool, context: BrowserContext, stats: ContextStats, timeout: float
    ) -> None:
        self._pool = pool
        self._context = context
        self._timeout = timeout
        self.stats = stats

    @property
    def context(self) -> BrowserContext:
        return self._context

    @asynccontextmanager
    async def page(self) -> AsyncGenerator[Page]:
        """A fresh page with the render timeout applied; waits for a free page slot."""
        async with self._pool._page_slot():
            page = await self._context.new_page()
            page.set_default_timeout(self._timeout * 1000)
            page.set_default_navigation_timeout(self._timeout * 1000)
            try:
                yield page
            finally:
                with suppress(PlaywrightError):  # browser already gone
                    await page.close()

    async def render(self, url: str, *, wait_until: WaitUntil = "load") -> RenderedPage:
        """Navigate and return the DOM as HTML. Step 36 adds networkidle caps and scrolling."""
        start = time.perf_counter()
        async with self.page() as page:
            try:
                # backstop over Playwright's own timeouts (navigation + content)
                async with asyncio.timeout(self._timeout * 2 + 5):
                    response = await page.goto(url, wait_until=wait_until)
                    html = await page.content()
            except (PlaywrightTimeoutError, TimeoutError) as exc:
                raise RenderError(
                    f"render exceeded {self._timeout}s", url=url, detail="timeout"
                ) from exc
            except PlaywrightError as exc:
                message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                raise RenderError(message, url=url, detail=message[:200]) from exc
            return RenderedPage(
                requested_url=url,
                final_url=page.url,
                status_code=response.status if response is not None else None,
                html=html,
                elapsed_ms=int((time.perf_counter() - start) * 1000),
            )


class BrowserPool:
    def __init__(self, config: Settings = settings) -> None:
        self._user_agent = config.CRAWL_USER_AGENT
        self._blocked = frozenset(config.CRAWL_BROWSER_BLOCKED_RESOURCES)
        self._max_pages = config.CRAWL_BROWSER_MAX_PAGES
        self._recycle_after = config.CRAWL_BROWSER_RECYCLE_AFTER_CONTEXTS
        self._headless = config.CRAWL_BROWSER_HEADLESS
        self._timeout = config.CRAWL_RENDER_TIMEOUT_SECONDS
        self._reset_state()

    def _reset_state(self) -> None:
        self._pid: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._lock: asyncio.Lock | None = None
        self._pages: asyncio.Semaphore | None = None
        self._contexts_since_launch = 0
        self._active_contexts = 0
        self.generation = 0  # browsers launched by this pool in this process

    # ---- lifecycle ---- #

    @property
    def is_running(self) -> bool:
        return (
            self._browser is not None and self._pid == os.getpid() and self._browser.is_connected()
        )

    @property
    def active_contexts(self) -> int:
        return self._active_contexts

    def _bind(self) -> asyncio.Lock:
        """Tie the pool to this process and running loop (first use, or after a fork)."""
        loop = asyncio.get_running_loop()
        if self._pid != os.getpid():
            if self._pid is not None:
                logger.info("browser pool inherited through fork: starting fresh")
            self._reset_state()
            self._pid, self._loop = os.getpid(), loop
            self._lock, self._pages = asyncio.Lock(), asyncio.Semaphore(self._max_pages)
        elif self._loop is not loop:
            raise BrowserPoolError(
                "browser pool is bound to another event loop; run each worker process on "
                "one persistent loop"
            )
        assert self._lock is not None
        return self._lock

    async def start(self) -> None:
        """Launch now instead of on first use (idempotent)."""
        await self.browser()

    async def browser(self) -> Browser:
        async with self._bind():
            recycle = (
                self._contexts_since_launch >= self._recycle_after and self._active_contexts == 0
            )
            if self._browser is not None and self._browser.is_connected() and not recycle:
                return self._browser
            if recycle:
                logger.info("recycling browser after {} contexts", self._contexts_since_launch)
            elif self._browser is not None:
                logger.warning("browser disconnected: relaunching")
            await self._close_browser()
            await self._launch()
            assert self._browser is not None
            return self._browser

    async def _launch(self) -> None:
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless, args=list(_LAUNCH_ARGS)
        )
        self._contexts_since_launch = 0
        self.generation += 1
        logger.info("launched Chromium {} (generation {})", self._browser.version, self.generation)

    async def _close_browser(self) -> None:
        if self._browser is not None:
            with suppress(PlaywrightError):  # already dead
                await self._browser.close()
            self._browser = None

    async def stop(self) -> None:
        """Close the browser and the Playwright driver (idempotent; worker shutdown)."""
        if self._pid != os.getpid():  # never launched here, or inherited through fork
            self._reset_state()
            return
        async with self._bind():
            await self._close_browser()
            if self._playwright is not None:
                await self._playwright.stop()
                self._playwright = None

    # ---- contexts & pages ---- #

    @asynccontextmanager
    async def _page_slot(self) -> AsyncGenerator[None]:
        assert self._pages is not None
        async with self._pages:
            yield

    @asynccontextmanager
    async def job_context(
        self, *, job_id: str | None = None, request_guard: RequestGuard | None = None
    ) -> AsyncGenerator[BrowserJobContext]:
        browser = await self.browser()
        context = await browser.new_context(
            user_agent=self._user_agent,
            viewport=VIEWPORT,
            locale="en-US",
            java_script_enabled=True,
            ignore_https_errors=False,
        )
        stats = ContextStats()
        await context.route("**/*", self._router(stats, request_guard))
        self._contexts_since_launch += 1
        self._active_contexts += 1
        logger.debug("browser context opened for job {}", job_id or "-")
        try:
            yield BrowserJobContext(self, context, stats, self._timeout)
        finally:
            self._active_contexts -= 1
            with suppress(PlaywrightError):  # the browser crashed underneath us
                await context.close()

    def _router(
        self, stats: ContextStats, guard: RequestGuard | None
    ) -> Callable[[Route, Request], Awaitable[None]]:
        blocked = self._blocked

        async def handle(route: Route, request: Request) -> None:
            stats.requests += 1
            reason: str | None = None
            if request.resource_type in blocked:
                reason = f"resource:{request.resource_type}"
            elif guard is not None and (verdict := await guard(request)) is not None:
                reason = f"guard:{verdict}"
            try:
                if reason is None:
                    await route.continue_()
                else:
                    stats.blocked[reason] += 1
                    await route.abort("blockedbyclient")
            except PlaywrightError:
                pass  # page/context closed while the request was in flight

        return handle


_process_pool: BrowserPool | None = None


def process_browser_pool() -> BrowserPool:
    """This worker process's pool (created on first call; Chromium launches on first use)."""
    global _process_pool
    if _process_pool is None:
        _process_pool = BrowserPool()
    return _process_pool


async def shutdown_process_browser_pool() -> None:
    """Worker-process shutdown hook (Step 75 wires it to Celery's signal)."""
    global _process_pool
    if _process_pool is not None:
        await _process_pool.stop()
        _process_pool = None
