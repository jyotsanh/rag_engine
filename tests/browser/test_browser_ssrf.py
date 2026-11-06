"""Browser SSRF enforcement: every browser request is vetted, pinned and made by our client."""

import json
from collections import defaultdict
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from playwright.async_api import Error as PlaywrightError

from rag_engine.config import Settings
from rag_engine.crawling.fetchers import BrowserJobContext, BrowserPool, RenderError
from rag_engine.crawling.fetchers.browser import _fulfill_headers
from rag_engine.security import SsrfGuard
from tests.conftest import OverrideSettings
from tests.support.fixture_site import FixtureSite


class _StubResolver:
    """Names only this resolver knows; a list of answers per name plays in order (last repeats)."""

    def __init__(self, answers: dict[str, list[list[str]]]) -> None:
        self._answers = answers
        self.calls: defaultdict[str, int] = defaultdict(int)

    async def __call__(self, host: str, port: int) -> list[str]:
        self.calls[host] += 1
        if host == "127.0.0.1":
            return ["127.0.0.1"]
        plays = self._answers.get(host)
        if plays is None:
            raise OSError(f"no such host {host}")
        return plays[min(self.calls[host], len(plays)) - 1]


type MakePool = Callable[..., BrowserPool]

_RESOLVER_ANSWERS = {
    "site.test": [["127.0.0.1"]],  # unknown to the OS: only our client can reach it
    "evil.test": [["169.254.169.254"]],
    "rebind.test": [["127.0.0.1"], ["169.254.169.254"]],  # DNS rebinding after 1st lookup
}


@pytest.fixture
def resolver() -> _StubResolver:
    return _StubResolver(_RESOLVER_ANSWERS)


@pytest.fixture
async def make_pool(
    override_settings: OverrideSettings, resolver: _StubResolver
) -> AsyncIterator[MakePool]:
    pools: list[BrowserPool] = []

    def make(*, allow_private: bool = True, **overrides: object) -> BrowserPool:
        config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=allow_private, **overrides)
        pool = BrowserPool(config, ssrf_guard=SsrfGuard(config, resolver=resolver))
        pools.append(pool)
        return pool

    yield make
    for pool in pools:
        await pool.stop()


@pytest.fixture
def pool(make_pool: MakePool) -> BrowserPool:
    return make_pool()


async def _run_probes(job: BrowserJobContext, url: str) -> dict[str, str]:
    async with job.page() as page:
        await page.goto(url)
        await page.wait_for_function("document.body.dataset.done === '1'", timeout=15_000)
        return await page.evaluate("Object.assign({}, document.body.dataset)")


# ---- #
# Tests
# ---- #


class TestBrowserSsrf_S35:
    """S35 — Browser SSRF enforcement: vetted+pinned requests, no WS/SW/downloads, backstop."""

    async def test_the_browser_never_resolves_or_connects_itself(
        self, pool: BrowserPool, fixture_site: FixtureSite, resolver: _StubResolver
    ) -> None:
        # "site.test" exists only in our stub resolver: the page can only load if every
        # request (document and subresources) is made by our pinned client
        url = f"http://site.test:{fixture_site.port}/browser/resources.html"

        async with pool.job_context() as job:
            page = await job.render(url)

        assert page.status_code == 200
        assert 'data-js="ran"' in page.html  # the script came through our client too
        assert resolver.calls["site.test"] >= 3  # document, stylesheet, script: each vetted
        hosts = {r.host for r in fixture_site.requests}
        assert hosts == {f"site.test:{fixture_site.port}"}  # Host header preserved

    async def test_strict_guard_blocks_a_loopback_document(
        self, make_pool: MakePool, fixture_site: FixtureSite
    ) -> None:
        pool = make_pool(allow_private=False)

        async with pool.job_context() as job:
            with pytest.raises(RenderError) as exc:
                await job.render(fixture_site.url("/about.html"))
            stats = job.stats

        assert "ERR_BLOCKED_BY_CLIENT" in str(exc.value)
        assert fixture_site.hits("/about.html") == 0  # nothing ever reached the server
        assert sum(n for k, n in stats.blocked.items() if k.startswith("ssrf:")) == 1

    async def test_in_page_requests_are_policed(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job:
            results = await _run_probes(job, fixture_site.url("/browser/ssrf.html"))
            stats = job.stats

        assert results["metadata"] == "blocked"  # 169.254.169.254 even with private allowed
        assert results["metaname"] == "blocked"  # metadata.google.internal by name
        assert results["evil"] == "blocked"  # a public-looking name resolving to metadata
        assert results["same"] == "ok"
        assert results["post"] == "ok"  # reads over POST (GraphQL) are allowed
        assert (results["put"], results["del"]) == ("blocked", "blocked")
        assert results["ws"] == "closed"
        assert results["sw"] == "blocked"
        assert stats.blocked["method:PUT"] == 1 and stats.blocked["method:DELETE"] == 1
        assert stats.blocked["websocket"] == 1
        assert sum(n for k, n in stats.blocked.items() if k.startswith("ssrf:")) == 3
        methods = [r.method for r in fixture_site.requests if r.path == "/browser/echo"]
        assert sorted(methods) == ["GET", "POST"]  # PUT/DELETE never reached the server
        assert fixture_site.hits("/browser/ws") == 0
        assert fixture_site.hits("/browser/sw.js") == 0

    async def test_every_redirect_hop_is_vetted(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job:
            followed = await job.render(fixture_site.url("/redirect-chain"))
            with pytest.raises(RenderError):
                await job.render(fixture_site.url("/redirect-metadata"))
            stats = job.stats

        assert followed.final_url == fixture_site.url("/docs/getting-started.html")
        assert followed.status_code == 200
        assert fixture_site.hits("/redirect-metadata") == 1  # hop 1 allowed, hop 2 blocked
        assert sum(n for k, n in stats.blocked.items() if k.startswith("ssrf:")) == 1

    async def test_navigation_redirect_loop_and_cap(
        self, make_pool: MakePool, fixture_site: FixtureSite
    ) -> None:
        pool = make_pool(CRAWL_MAX_REDIRECTS=1)

        async with pool.job_context() as job:
            with pytest.raises(RenderError, match="redirect loop"):
                await job.render(fixture_site.url("/loop-a"))
            with pytest.raises(RenderError, match="more than 1 redirects"):
                await job.render(fixture_site.url("/redirect-chain"))  # 2 hops

    async def test_subresource_redirects_are_followed_and_vetted(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job, job.page() as page:
            await job.goto(page, fixture_site.url("/about.html"))
            followed = await page.evaluate(
                "fetch('/redirect-to?url=/browser/echo').then((r) => r.json())"
            )
            to_metadata = await page.evaluate(
                "fetch('/redirect-to?url=http://169.254.169.254/latest/meta-data/')"
                ".then(() => 'ok', () => 'blocked')"
            )
            stats = job.stats

        assert followed["method"] == "GET"
        assert to_metadata == "blocked"
        assert sum(n for k, n in stats.blocked.items() if k.startswith("ssrf:")) == 1

    async def test_credentials_are_not_carried_across_origins(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        other = f"http://site.test:{fixture_site.port}/browser/echo"

        async with pool.job_context() as job, job.page() as page:
            await job.goto(page, fixture_site.url("/browser/set-cookie"))
            same = await page.evaluate(
                "fetch('/redirect-to?url=/browser/echo').then((r) => r.json())"
            )
            cross = await page.evaluate(f"fetch('/redirect-to?url={other}').then((r) => r.json())")

        assert "first=1" in same["cookie"]  # same-origin hop keeps the cookie
        assert cross["cookie"] is None  # 127.0.0.1's cookies never reach site.test

    async def test_dns_rebinding_cannot_slip_through(
        self, pool: BrowserPool, fixture_site: FixtureSite, resolver: _StubResolver
    ) -> None:
        # first lookup → loopback (allowed in tests); every later one → the metadata IP
        url = f"http://rebind.test:{fixture_site.port}/browser/resources.html"

        async with pool.job_context() as job:
            page = await job.render(url)
            stats = job.stats

        assert page.status_code == 200
        assert fixture_site.hits("/browser/resources.html") == 1
        assert fixture_site.hits("/browser/style.css") == 0  # rebound: blocked, not connected
        assert fixture_site.hits("/browser/app.js") == 0
        assert sum(n for k, n in stats.blocked.items() if k.startswith("ssrf:")) >= 2
        assert resolver.calls["rebind.test"] >= 3

    async def test_downloads_are_refused(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job:
            with pytest.raises(RenderError, match=r"[Dd]ownload"):
                await job.render(fixture_site.url("/browser/download.bin"))

    async def test_traffic_outside_routing_hits_the_dead_proxy(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job:
            await job.context.unroute("**/*")  # simulate a request escaping interception
            async with job.page() as page:
                with pytest.raises(PlaywrightError, match="ERR_PROXY_CONNECTION_FAILED"):
                    await page.goto(fixture_site.url("/about.html"))

        assert fixture_site.hits("/about.html") == 0

    async def test_cookies_round_trip_through_the_forwarder(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job, job.page() as page:
            await page.goto(fixture_site.url("/browser/set-cookie"))
            echoed = await page.evaluate("fetch('/browser/echo').then((r) => r.text())")

        cookie = json.loads(echoed)["cookie"]
        assert "first=1" in cookie and "second=2" in cookie  # both Set-Cookie headers kept

    async def test_compressed_responses_are_decoded(
        self, pool: BrowserPool, fixture_site: FixtureSite
    ) -> None:
        async with pool.job_context() as job:
            page = await job.render(fixture_site.url("/gzip-page"))

        assert page.status_code == 200
        assert "<body" in page.html  # served gzip-encoded; the browser got plain HTML

    async def test_oversized_responses_are_aborted(
        self, make_pool: MakePool, fixture_site: FixtureSite
    ) -> None:
        pool = make_pool(CRAWL_MAX_PAGE_BYTES=1024 * 1024)

        async with pool.job_context() as job:
            with pytest.raises(RenderError):
                await job.render(fixture_site.url("/large.html"))
            stats = job.stats

        assert stats.blocked["too_large"] == 1

    def test_fulfill_headers(self) -> None:
        headers = httpx.Headers(
            [
                ("Content-Type", "text/html"),
                ("Content-Encoding", "gzip"),
                ("Content-Length", "123"),
                ("Transfer-Encoding", "chunked"),
                ("Set-Cookie", "a=1; Path=/"),
                ("Set-Cookie", "b=2; Path=/"),
                ("Vary", "Accept"),
                ("Vary", "Cookie"),
            ]
        )

        assert _fulfill_headers(headers) == {
            "content-type": "text/html",
            "set-cookie": "a=1; Path=/\nb=2; Path=/",
            "vary": "Accept, Cookie",
        }
