"""JS-shell detector: which raw HTML pages need a browser render."""

from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import JsShellDetector, JsShellThresholds, JsSignal
from rag_engine.crawling.fetchers import HttpFetcher
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_PAGES, SITE_ROOT, Expect, FixtureSite

_PROSE = (
    "Acme Widgets builds modular widget assemblies for industrial automation. The Widget "
    "Controller exposes a REST API, streams telemetry over TLS and supports staged firmware "
    "rollouts across fleets of devices. "
)


def _page(body: str, head: str = "") -> str:
    return f"<!doctype html><html><head><title>t</title>{head}</head><body>{body}</body></html>"


def _detect(html: str) -> tuple[bool, set[JsSignal]]:
    verdict = JsShellDetector().detect(html)
    return verdict.needs_rendering, set(verdict.signals)


_BUNDLE = '<script src="/static/js/main.4f2a.js"></script>'
_ANALYTICS = (
    '<script async src="https://www.googletagmanager.com/gtag/js?id=G-X"></script>'
    "<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments)}"
    "gtag('js',new Date());gtag('config','G-X');</script>"
)


@pytest.fixture
async def fetcher(override_settings: OverrideSettings) -> AsyncIterator[HttpFetcher]:
    config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


# ---- #
# Tests
# ---- #


class TestJsShellDetector_S33:
    """S33 — JS-shell detector: text length, empty mounts, noscript hints, script ratio."""

    # ---- shells ---- #

    def test_create_react_app_shell(self) -> None:
        html = _page(
            "<noscript>You need to enable JavaScript to run this app.</noscript>"
            '<div id="root"></div>' + _BUNDLE
        )

        verdict = JsShellDetector().detect(html)

        assert verdict.needs_rendering
        assert verdict.mount == "#root"
        assert {JsSignal.VERY_LOW_TEXT, JsSignal.EMPTY_MOUNT, JsSignal.NOSCRIPT_HINT} <= set(
            verdict.signals
        )
        assert verdict.text_chars == 0  # the <noscript> text is not visible text

    def test_next_client_only_shell(self) -> None:
        data = '{"props":{"pageProps":{}},"page":"/","buildId":"' + "x" * 5000 + '"}'
        html = _page(
            '<div id="__next"></div>'
            f'<script id="__NEXT_DATA__" type="application/json">{data}</script>'
            '<script src="/_next/static/chunks/main.js"></script>'
        )

        verdict = JsShellDetector().detect(html)

        assert verdict.needs_rendering
        assert verdict.mount == "#__next"
        assert verdict.script_bytes == 0  # __NEXT_DATA__ is data, not code
        assert verdict.script_count == 1

    def test_angular_root_component(self) -> None:
        verdict = JsShellDetector().detect(_page("<app-root></app-root>" + _BUNDLE))

        assert verdict.needs_rendering
        assert verdict.mount == "<app-root>"

    @pytest.mark.parametrize(
        "mount",
        [
            '<div id="app"><div class="spinner"><span></span></div></div>',
            '<div id="root"><style>.spinner{color:red}</style><svg><text>logo</text></svg></div>',
            '<div id="___gatsby"><div id="gatsby-focus-wrapper"></div></div>',
            '<main data-reactroot=""></main>',
        ],
        ids=["spinner", "style-and-svg", "gatsby", "data-reactroot"],
    )
    def test_mount_without_visible_text_counts_as_empty(self, mount: str) -> None:
        needs, signals = _detect(_page(mount + _BUNDLE))

        assert needs
        assert JsSignal.EMPTY_MOUNT in signals

    def test_unrendered_angularjs_templates(self) -> None:
        html = _page(
            "<h1>{{ product.name }}</h1><p>{{ product.description }}</p>"
            "<span>{{ product.price | currency }}</span>"
            '<script src="/angular.min.js"></script>',
            '<base href="/">',
        )

        needs, signals = _detect(html)

        assert needs
        assert JsSignal.TEMPLATE_MARKERS in signals

    def test_ajax_crawling_meta(self) -> None:
        html = _page("<p>Loading…</p>" + _BUNDLE, '<meta name="fragment" content="!">')

        needs, signals = _detect(html)

        assert needs
        assert JsSignal.AJAX_FRAGMENT_META in signals

    def test_loading_placeholder_with_bundle(self) -> None:
        needs, signals = _detect(_page("<div class='boot'>Loading, please wait…</div>" + _BUNDLE))

        assert needs
        assert {JsSignal.VERY_LOW_TEXT, JsSignal.SCRIPT_HEAVY} <= signals

    # ---- not shells ---- #

    def test_server_rendered_next_page(self) -> None:
        html = _page(
            f'<div id="__next"><main><h1>Docs</h1><p>{_PROSE * 3}</p></main></div>'
            '<script id="__NEXT_DATA__" type="application/json">{"props":{}}</script>'
            '<script src="/_next/static/chunks/main.js"></script>'
        )

        verdict = JsShellDetector().detect(html)

        assert not verdict.needs_rendering
        assert verdict.mount is None

    def test_content_page_with_analytics(self) -> None:
        needs, signals = _detect(_page(f"<article><p>{_PROSE * 4}</p></article>" + _ANALYTICS))

        assert not needs
        assert JsSignal.EMPTY_MOUNT not in signals

    def test_short_page_without_scripts_is_never_rendered(self) -> None:
        verdict = JsShellDetector().detect(_page("<p>Call us: 555-0100</p>"))

        assert verdict.score >= 3  # very low text alone would pass the threshold...
        assert not verdict.needs_rendering  # ...but no script could add anything

    def test_short_page_with_a_small_script(self) -> None:
        html = _page(
            "<h1>Contact</h1><p>Write to support@acme.example or call 555-0100 on weekdays.</p>"
            "<script>document.body.classList.add('js')</script>"
        )

        verdict = JsShellDetector().detect(html)

        assert verdict.signals == (JsSignal.LOW_TEXT,)
        assert not verdict.needs_rendering

    def test_rich_article_with_a_stray_widget_mount(self) -> None:
        html = _page(f"<article>{_PROSE * 12}</article><div id='app'></div>" + _BUNDLE)

        verdict = JsShellDetector().detect(html)

        assert JsSignal.EMPTY_MOUNT in verdict.signals
        assert JsSignal.RICH_TEXT in verdict.signals
        assert not verdict.needs_rendering

    def test_json_ld_only_page_has_no_executable_scripts(self) -> None:
        html = _page(
            '<script type="application/ld+json">{"@type":"Organization"}</script>'
            '<div id="root"></div>'
        )

        verdict = JsShellDetector().detect(html)

        assert verdict.script_count == 0
        assert not verdict.needs_rendering

    def test_non_visible_text_is_ignored(self) -> None:
        html = _page(
            "<p>Visible words.</p><script>var hidden = 'not visible text';</script>"
            "<style>.x{content:'nope'}</style><template><p>later</p></template>"
            "<noscript>Enable JavaScript</noscript><svg><text>chart label</text></svg>",
            "<title>Title is in head</title>",
        )

        verdict = JsShellDetector().detect(html)

        assert verdict.text_chars == len("Visible words.")

    @pytest.mark.parametrize(
        ("noscript", "hint"),
        [
            ("You need to enable JavaScript to run this app.", True),
            ("This site requires JavaScript.", True),
            ("JavaScript is disabled in your browser.", True),
            ("Please turn on JavaScript and reload.", True),
            ("JavaScript must be enabled to use this site.", True),
            ('<img src="/pixel.gif" alt="">', False),
            ("Sorry, your browser does not support video.", False),
        ],
    )
    def test_noscript_hints(self, noscript: str, hint: bool) -> None:
        _, signals = _detect(_page(f"<noscript>{noscript}</noscript><p>{_PROSE}</p>" + _BUNDLE))

        assert (JsSignal.NOSCRIPT_HINT in signals) is hint

    @pytest.mark.parametrize("html", ["", "   ", "<<>>", "\x00\x01", "<html>"])
    def test_empty_or_garbage_html(self, html: str) -> None:
        verdict = JsShellDetector().detect(html)

        assert not verdict.needs_rendering

    def test_thresholds_are_configurable(self) -> None:
        shell = _page('<div id="root"></div>' + _BUNDLE)

        strict = JsShellDetector(JsShellThresholds(threshold=20)).detect(shell)
        default = JsShellDetector().detect(shell)

        assert default.needs_rendering and not strict.needs_rendering
        assert strict.score == default.score

    # ---- fixture site ---- #

    def test_every_static_fixture_page_but_the_spa_is_not_a_shell(self) -> None:
        detector = JsShellDetector()
        js_pages = {p for p, e in SITE_PAGES.items() if e is Expect.JS_RENDERED}
        assert js_pages == {"/app/"}

        verdicts = {
            "/" + str(path.relative_to(SITE_ROOT)).replace("index.html", ""): detector.detect(
                path.read_text(errors="replace")
            )
            for path in SITE_ROOT.rglob("*.html")
        }

        assert {p for p, v in verdicts.items() if v.needs_rendering} == js_pages

    async def test_fixture_spa_fetched_over_http(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        page = await fetcher.fetch(fixture_site.url("/app/"))
        assert page.text is not None

        verdict = JsShellDetector().detect(page.text)

        assert verdict.needs_rendering
        assert verdict.mount == "#root"
        assert JsSignal.NOSCRIPT_HINT in verdict.signals
