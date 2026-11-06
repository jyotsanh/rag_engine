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
storage, cache) with our User-Agent.

SSRF enforcement: **Chromium never opens a connection itself.**
- Every request is intercepted (`context.route`). After the policy checks (blocked
  resource types, methods other than GET/HEAD/POST/OPTIONS, the optional
  `request_guard`), the request is **performed by `SafeHttpClient.send_once`**. That
  vets the host through the SSRF guard and connects only to the vetted IP (no DNS
  rebinding between check and connect). The response is handed back with
  `route.fulfill`. Redirects are returned to the browser, so each hop comes back through
  the router and is vetted again.
- Backstop for traffic that never reaches the router (preconnect, prefetch, an
  accidental `continue_`): Chromium runs behind a proxy on a reserved local port that
  never listens, with loopback not bypassed, so any such connection fails.
- WebSockets are refused (`route_web_socket`), downloads disabled, service workers
  blocked (they could fetch outside page routing), and WebRTC is kept off UDP.
"""

import asyncio
import os
import socket
import time
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Literal

import httpx
from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Request,
    Response,
    Route,
    ViewportSize,
    WebSocketRoute,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from rag_engine.config import Settings, settings
from rag_engine.crawling.fetchers.errors import FetchError
from rag_engine.domain import PageReason
from rag_engine.security import (
    REDIRECT_STATUSES,
    HostResolutionError,
    SafeHttpClient,
    SsrfBlockedError,
    SsrfGuard,
)

# returns a reason to abort the request, or None to let it through
type RequestGuard = Callable[[Request], Awaitable[str | None]]
type WaitUntil = Literal["commit", "domcontentloaded", "load", "networkidle"]

VIEWPORT: ViewportSize = {"width": 1280, "height": 900}
_LAUNCH_ARGS = (
    "--disable-dev-shm-usage",  # containers ship a tiny /dev/shm
    "--disable-background-networking",
    "--disable-component-update",
    "--no-first-run",
    "--dns-prefetch-disable",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--proxy-bypass-list=<-loopback>",  # loopback must hit the dead proxy too
)
# state-changing methods are never sent on a crawler's behalf (GraphQL reads use POST)
ALLOWED_METHODS = frozenset({"GET", "HEAD", "POST", "OPTIONS"})
# not forwarded from the browser: hop-by-hop, set by our client, or HTTP/2 pseudo-headers
_DROP_REQUEST_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-connection",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "accept-encoding",
    }
)
# never carried to another origin when we follow a subresource redirect ourselves
_CREDENTIAL_HEADERS = frozenset({"cookie", "authorization"})
# not passed back to the browser: the body is already decoded and fully buffered
_DROP_RESPONSE_HEADERS = frozenset(
    {"content-encoding", "content-length", "transfer-encoding", "connection", "keep-alive"}
)


class _Aborted(Exception):
    """A forwarded request that must be aborted: `stat` for ContextStats, Chromium error code."""

    def __init__(self, stat: str, error_code: str) -> None:
        super().__init__(stat)
        self.stat = stat
        self.error_code = error_code


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
        self,
        pool: BrowserPool,
        context: BrowserContext,
        stats: ContextStats,
        redirects: dict[Page, str],
        *,
        timeout: float,
        max_redirects: int,
    ) -> None:
        self._pool = pool
        self._context = context
        self._timeout = timeout
        self._max_redirects = max_redirects
        self._redirects = redirects  # main-frame redirect targets parked by the router
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
                self._redirects.pop(page, None)
                with suppress(PlaywrightError):  # browser already gone
                    await page.close()

    async def goto(
        self, page: Page, url: str, *, wait_until: WaitUntil = "load"
    ) -> Response | None:
        """`page.goto` that follows main-frame redirects **as new navigations**.

        The router never lets Chromium follow a document redirect itself (that hop would
        bypass interception): it parks the vetted `Location` and aborts. Navigating there
        again keeps `page.url`, relative links and `location` correct, and the new hop is
        vetted like any other request.
        """
        chain = [url]
        while True:
            try:
                return await page.goto(chain[-1], wait_until=wait_until)
            except PlaywrightError:
                target = self._redirects.pop(page, None)
                if target is None:
                    raise
            if target in chain:
                raise RenderError(
                    f"redirect loop at {target}", url=url, detail=" -> ".join([*chain, target])
                )
            if len(chain) > self._max_redirects:
                raise RenderError(
                    f"more than {self._max_redirects} redirects",
                    url=url,
                    detail="too_many_redirects",
                )
            chain.append(target)

    async def render(self, url: str, *, wait_until: WaitUntil = "load") -> RenderedPage:
        """Navigate and return the DOM as HTML. Step 36 adds networkidle caps and scrolling."""
        start = time.perf_counter()
        async with self.page() as page:
            try:
                # backstop over Playwright's own timeouts (navigation + content)
                async with asyncio.timeout(self._timeout * 2 + 5):
                    response = await self.goto(page, url, wait_until=wait_until)
                    html = await page.content()
            except (PlaywrightTimeoutError, TimeoutError) as exc:
                raise RenderError(
                    f"render exceeded {self._timeout}s", url=url, detail="timeout"
                ) from exc
            except RenderError:
                raise
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
    def __init__(self, config: Settings = settings, *, ssrf_guard: SsrfGuard | None = None) -> None:
        self._config = config
        self._ssrf_guard = ssrf_guard
        self._max_bytes = config.CRAWL_MAX_PAGE_BYTES
        self._fetch_timeout = config.CRAWL_FETCH_TIMEOUT_SECONDS
        self._user_agent = config.CRAWL_USER_AGENT
        self._blocked = frozenset(config.CRAWL_BROWSER_BLOCKED_RESOURCES)
        self._max_pages = config.CRAWL_BROWSER_MAX_PAGES
        self._recycle_after = config.CRAWL_BROWSER_RECYCLE_AFTER_CONTEXTS
        self._headless = config.CRAWL_BROWSER_HEADLESS
        self._timeout = config.CRAWL_RENDER_TIMEOUT_SECONDS
        self._max_redirects = config.CRAWL_MAX_REDIRECTS
        self._reset_state()

    def _reset_state(self) -> None:
        self._pid: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._lock: asyncio.Lock | None = None
        self._pages: asyncio.Semaphore | None = None
        self._client: SafeHttpClient | None = None
        previous: socket.socket | None = getattr(self, "_dead_proxy", None)
        if previous is not None:
            previous.close()  # our own copy of the fd (also after a fork)
        self._dead_proxy: socket.socket | None = None
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
        if self._client is None:
            self._client = SafeHttpClient(self._config, guard=self._ssrf_guard)
        if self._dead_proxy is None:
            # bound but never listening: every connection to it is refused at once
            self._dead_proxy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._dead_proxy.bind(("127.0.0.1", 0))
        port = self._dead_proxy.getsockname()[1]
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=[*_LAUNCH_ARGS, f"--proxy-server=http://127.0.0.1:{port}"],
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
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            if self._dead_proxy is not None:
                self._dead_proxy.close()
                self._dead_proxy = None

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
            accept_downloads=False,
            service_workers="block",
        )
        stats = ContextStats()
        redirects: dict[Page, str] = {}
        await context.route("**/*", self._router(stats, redirects, request_guard))
        await context.route_web_socket("**/*", self._refuse_websocket(stats))
        self._contexts_since_launch += 1
        self._active_contexts += 1
        logger.debug("browser context opened for job {}", job_id or "-")
        try:
            yield BrowserJobContext(
                self,
                context,
                stats,
                redirects,
                timeout=self._timeout,
                max_redirects=self._max_redirects,
            )
        finally:
            self._active_contexts -= 1
            with suppress(PlaywrightError):  # the browser crashed underneath us
                await context.close()

    def _router(
        self, stats: ContextStats, redirects: dict[Page, str], guard: RequestGuard | None
    ) -> Callable[[Route, Request], Awaitable[None]]:
        blocked = self._blocked

        async def handle(route: Route, request: Request) -> None:
            stats.requests += 1
            reason: str | None = None
            if request.resource_type in blocked:
                reason = f"resource:{request.resource_type}"
            elif request.method not in ALLOWED_METHODS:
                reason = f"method:{request.method}"
            elif guard is not None and (verdict := await guard(request)) is not None:
                reason = f"guard:{verdict}"
            try:
                if reason is not None:
                    stats.blocked[reason] += 1
                    await route.abort("blockedbyclient")
                    return
                await self._forward(route, request, stats, redirects)
            except PlaywrightError:
                pass  # page/context closed while the request was in flight

        return handle

    async def _forward(
        self, route: Route, request: Request, stats: ContextStats, redirects: dict[Page, str]
    ) -> None:
        """Perform the request through the SSRF-safe client and fulfill the route with it.

        Redirects never reach Chromium (it would follow them outside interception):
        main-frame navigations are parked for `BrowserJobContext.goto` to re-navigate,
        everything else is followed here, every hop vetted again.
        """
        headers = {
            name: value
            for name, value in (await request.all_headers()).items()
            if name.lower() not in _DROP_REQUEST_HEADERS and not name.startswith(":")
        }
        method, url, body = request.method, request.url, request.post_data_buffer
        main_frame_navigation = (
            request.is_navigation_request() and request.frame.parent_frame is None
        )
        try:
            for _hop in range(self._max_redirects + 1):
                status, response_headers, content = await self._fetch(method, url, headers, body)
                location = response_headers.get("location")
                if status not in REDIRECT_STATUSES or not location:
                    break
                target = str(httpx.URL(url).join(location))
                if main_frame_navigation:
                    redirects[request.frame.page] = target
                    await route.abort("aborted")
                    return
                if httpx.URL(target).netloc != httpx.URL(url).netloc:
                    headers = {
                        k: v for k, v in headers.items() if k.lower() not in _CREDENTIAL_HEADERS
                    }
                if status == 303 or (status in (301, 302) and method == "POST"):
                    method, body = "GET", None
                url = target
            else:
                stats.blocked["too_many_redirects"] += 1
                await route.abort("failed")
                return
        except _Aborted as aborted:
            stats.blocked[aborted.stat] += 1
            await route.abort(aborted.error_code)
            return
        await route.fulfill(status=status, headers=_fulfill_headers(response_headers), body=content)

    async def _fetch(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, httpx.Headers, bytes]:
        """One vetted, pinned hop with a size and time cap → (status, headers, decoded body)."""
        assert self._client is not None
        try:
            async with (
                asyncio.timeout(self._fetch_timeout),
                self._client.send_once(method, url, headers=headers, content=body) as response,
            ):
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content += chunk
                    if len(content) > self._max_bytes:
                        raise _Aborted("too_large", "failed")
                return response.status_code, response.headers, bytes(content)
        except SsrfBlockedError as exc:
            logger.bind(url=url).info("browser request blocked: {}", exc.rejection)
            raise _Aborted(f"ssrf:{exc.rejection.value}", "blockedbyclient") from exc
        except HostResolutionError as exc:
            raise _Aborted("dns", "namenotresolved") from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise _Aborted("timeout", "timedout") from exc
        except (httpx.TransportError, httpx.DecodingError) as exc:
            raise _Aborted("network", "connectionfailed") from exc

    @staticmethod
    def _refuse_websocket(stats: ContextStats) -> Callable[[WebSocketRoute], Awaitable[None]]:
        async def refuse(ws: WebSocketRoute) -> None:
            # never `connect_to_server()`: the socket is closed before reaching anything
            stats.blocked["websocket"] += 1
            await ws.close(code=1008, reason="websockets are disabled for crawling")

        return refuse


def _fulfill_headers(headers: httpx.Headers) -> dict[str, str]:
    """httpx headers → `route.fulfill` headers. Repeated headers are joined, with
    `Set-Cookie` values newline-separated (Playwright splits them back)."""
    merged: dict[str, list[str]] = {}
    for name, value in headers.multi_items():
        if name.lower() not in _DROP_RESPONSE_HEADERS:
            merged.setdefault(name.lower(), []).append(value)
    return {
        name: ("\n" if name == "set-cookie" else ", ").join(values)
        for name, values in merged.items()
    }


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
