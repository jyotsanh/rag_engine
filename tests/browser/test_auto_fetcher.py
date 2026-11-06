"""Auto fetcher: HTTP first, JS-shell detection, browser re-render with idle/scroll caps."""

import time
from collections.abc import AsyncIterator, Callable

import pytest
from playwright.async_api import Request

from rag_engine.config import JsRenderMode, Settings
from rag_engine.crawling import CrawlScope, JsShellDetector, LinkExtractor
from rag_engine.crawling.fetchers import (
    AutoFetcher,
    BrowserJobContext,
    BrowserPool,
    FetchResult,
    HttpFetcher,
    HttpStatusError,
    RenderedPage,
)
from rag_engine.domain import ContentKind, RenderedWith
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import FixtureSite


class _World:
    def __init__(self, config: Settings, http: HttpFetcher, job: BrowserJobContext) -> None:
        self.config = config
        self.http = http
        self.job = job

    def fetcher(
        self, mode: JsRenderMode = JsRenderMode.AUTO, *, browser: bool = True
    ) -> AutoFetcher:
        return AutoFetcher(self.http, self.job if browser else None, config=self.config, mode=mode)


type MakeWorld = Callable[..., object]


@pytest.fixture
async def world(override_settings: OverrideSettings) -> AsyncIterator[_World]:
    config: Settings = override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True, CRAWL_RENDER_NETWORKIDLE_SECONDS=3
    )
    pool = BrowserPool(config)
    try:
        async with SafeHttpClient(config) as client, pool.job_context(job_id="job_auto") as job:
            yield _World(config, HttpFetcher(client, config), job)
    finally:
        await pool.stop()


# ---- #
# Tests
# ---- #


class TestAutoFetcher_S36:
    """S36 — Auto fetcher: httpx → detector → Playwright re-render, `rendered_with` recorded."""

    # ---- mode: auto ---- #

    async def test_js_shell_is_rendered(self, world: _World, fixture_site: FixtureSite) -> None:
        result = await world.fetcher().fetch(fixture_site.url("/app/"))

        assert result.rendered_with is RenderedWith.BROWSER
        assert result.text is not None and "Widget configurator" in result.text
        assert {"empty_mount", "noscript_hint"} <= set(result.js_signals)
        assert result.render_note is None
        assert (result.mime_type, result.encoding) == ("text/html", "utf-8")
        assert result.body == result.text.encode("utf-8")
        assert result.final_url == fixture_site.url("/app/")

    async def test_links_only_in_the_rendered_dom_are_found(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        result = await world.fetcher().fetch(fixture_site.url("/app/"))
        assert result.text is not None
        scope = CrawlScope.from_config(fixture_site.url("/"), max_depth=5)

        links = LinkExtractor().extract(result.text, result.final_url, scope=scope, depth=1)

        assert fixture_site.url("/app/advanced") in [link.url for link in links.follow]

    async def test_server_rendered_page_skips_the_browser(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        result = await world.fetcher().fetch(fixture_site.url("/about.html"))

        assert result.rendered_with is RenderedWith.HTTP
        assert result.js_signals == ()
        assert world.job.stats.requests == 0  # the browser was never used
        assert fixture_site.hits("/about.html") == 1

    async def test_documents_are_never_rendered(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        result = await world.fetcher(JsRenderMode.ALWAYS).fetch(
            fixture_site.url("/files/widget-spec.pdf")
        )

        assert result.content_kind is ContentKind.PDF
        assert result.rendered_with is RenderedWith.HTTP
        assert world.job.stats.requests == 0

    async def test_http_errors_propagate_without_rendering(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        with pytest.raises(HttpStatusError):
            await world.fetcher(JsRenderMode.ALWAYS).fetch(fixture_site.url("/missing.html"))

        assert world.job.stats.requests == 0

    # ---- modes: never / always / no browser ---- #

    async def test_never_mode_uses_http_only(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        result = await world.fetcher(JsRenderMode.NEVER).fetch(fixture_site.url("/app/"))

        assert result.rendered_with is RenderedWith.HTTP
        assert result.js_signals == ()  # the detector is not even run
        assert world.job.stats.requests == 0

    async def test_always_mode_renders_every_html_page(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        result = await world.fetcher(JsRenderMode.ALWAYS).fetch(fixture_site.url("/about.html"))

        assert result.rendered_with is RenderedWith.BROWSER
        assert fixture_site.hits("/about.html") == 2  # HTTP fetch + browser render

    async def test_without_a_browser_the_http_result_is_kept(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        result = await world.fetcher(browser=False).fetch(fixture_site.url("/app/"))

        assert result.rendered_with is RenderedWith.HTTP
        assert "empty_mount" in result.js_signals  # still recorded for the page report

    # ---- keep the better result ---- #

    async def test_render_failure_falls_back_to_http(
        self, override_settings: OverrideSettings, fixture_site: FixtureSite
    ) -> None:
        config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)

        async def deny_documents(request: Request) -> str | None:
            return "no_documents" if request.resource_type == "document" else None

        pool = BrowserPool(config)
        try:
            async with (
                SafeHttpClient(config) as client,
                pool.job_context(request_guard=deny_documents) as job,
            ):
                fetcher = AutoFetcher(HttpFetcher(client, config), job, config=config)
                result = await fetcher.fetch(fixture_site.url("/app/"))
        finally:
            await pool.stop()

        assert result.rendered_with is RenderedWith.HTTP
        assert result.render_note is not None
        assert result.render_note.startswith("render_failed: ")
        assert result.text is not None and '<div id="root"></div>' in result.text

    async def test_render_that_loses_content_is_discarded(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        # the raw HTML has text; its script wipes the body in the browser
        result = await world.fetcher().fetch(fixture_site.url("/browser/wipe.html"))

        assert result.rendered_with is RenderedWith.HTTP
        assert result.render_note == "render_not_better"
        assert result.text is not None and "warranty terms" in result.text

    def test_non_2xx_render_is_discarded(self, world: _World) -> None:
        http = FetchResult(
            requested_url="https://a.example/",
            final_url="https://a.example/",
            redirect_chain=("https://a.example/",),
            status_code=200,
            mime_type="text/html",
            content_kind=ContentKind.HTML,
            body=b"<div id='root'></div><script></script>",
            text="<div id='root'></div><script></script>",
            encoding="utf-8",
            bytes_downloaded=38,
            elapsed_ms=5,
            headers={},
        )
        rendered = RenderedPage("https://a.example/", "https://a.example/", 404, "<p>x</p>", 9)
        fetcher = world.fetcher()

        result = fetcher._better(http, JsShellDetector().detect(http.text or ""), rendered)

        assert result is not http and result.render_note == "render_status_404"
        assert result.rendered_with is RenderedWith.HTTP

    async def test_js_redirect_during_render_is_recorded(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        result = await world.fetcher().fetch(fixture_site.url("/browser/js-redirect.html"))

        assert result.rendered_with is RenderedWith.BROWSER
        assert result.final_url == fixture_site.url("/docs/")
        assert result.redirect_chain[-1] == fixture_site.url("/docs/")
        assert result.text is not None and "Documentation" in result.text

    # ---- settling: network idle and auto-scroll ---- #

    async def test_network_idle_waits_for_late_content(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        url = fixture_site.url("/browser/late.html")

        hasty = await world.job.render(url)  # "load" only
        patient = await world.job.render(url, idle_timeout=5)

        rendered = '<div id="root"><h1>Late content</h1>'  # the script source has the text too
        assert rendered not in hasty.html and hasty.network_idle is None
        assert rendered in patient.html and patient.network_idle is True

    async def test_network_idle_wait_is_capped(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        start = time.perf_counter()

        page = await world.job.render(fixture_site.url("/browser/polling.html"), idle_timeout=1)

        assert page.network_idle is False  # never idle: the cap ended the wait
        assert "Live dashboard" in page.html
        assert time.perf_counter() - start < 5

    async def test_auto_scroll_loads_lazy_content(
        self, world: _World, fixture_site: FixtureSite
    ) -> None:
        url = fixture_site.url("/browser/lazy.html")

        still = await world.job.render(url, idle_timeout=2)
        scrolled = await world.job.render(
            url, idle_timeout=2, scroll_seconds=5, scroll_max_steps=20
        )

        assert "Lazy section" not in still.html
        assert "Lazy section" in scrolled.html
        assert scrolled.scroll_steps >= 2
        assert fixture_site.hits("/browser/lazy-chunk") == 1

    async def test_auto_scroll_is_capped(self, world: _World, fixture_site: FixtureSite) -> None:
        page = await world.job.render(
            fixture_site.url("/browser/lazy.html"), scroll_seconds=5, scroll_max_steps=2
        )

        assert page.scroll_steps == 2  # 4000px page, 900px viewport: the step cap stopped it
        assert "Lazy section" not in page.html
