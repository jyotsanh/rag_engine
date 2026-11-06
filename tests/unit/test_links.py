"""Link extraction: resolution, <base>, canonical, nofollow policy, documents, scope."""

from collections.abc import AsyncIterator, Callable

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import (
    CrawlScope,
    LinkExtractor,
    LinkSource,
    PageLinks,
    ScopeRejection,
    UrlNormalizationError,
    canonical_from_link_header,
    normalize_url,
    parse_x_robots_tag,
)
from rag_engine.crawling.fetchers import HttpFetcher
from rag_engine.domain import PageReason
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_PAGES, Expect, FixtureSite

_PAGE = "https://acme.example/docs/guide/intro.html"

type Extract = Callable[..., PageLinks]


def _html(body: str, head: str = "") -> str:
    return f"<!doctype html><html><head>{head}</head><body>{body}</body></html>"


def _a(href: str, text: str = "x", rel: str | None = None) -> str:
    rel_attr = f' rel="{rel}"' if rel is not None else ""
    return f'<a href="{href}"{rel_attr}>{text}</a>'


@pytest.fixture
def extract(override_settings: OverrideSettings) -> Extract:
    def run(
        html: str,
        page_url: str = _PAGE,
        *,
        depth: int = 0,
        max_depth: int = 5,
        headers: dict[str, str] | None = None,
        **overrides: object,
    ) -> PageLinks:
        config = override_settings(**overrides) if overrides else override_settings()
        scope = CrawlScope.from_config("https://acme.example/", max_depth=max_depth)
        return LinkExtractor(config).extract(
            html, page_url, scope=scope, depth=depth, headers=headers
        )

    return run


def _follow(result: PageLinks) -> list[str]:
    return [link.url for link in result.follow]


def _rejections(result: PageLinks) -> dict[str, ScopeRejection | None]:
    return {r.url: r.decision.detail for r in result.rejected}


@pytest.fixture
async def fetcher(override_settings: OverrideSettings) -> AsyncIterator[HttpFetcher]:
    config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


# ---- #
# Tests
# ---- #


class TestLinkExtraction_S32:
    """S32 — Link extraction: <a>/<area>/<base>/canonical/meta refresh, nofollow, scope."""

    # ---- resolution & normalization ---- #

    def test_relative_links_resolve_against_the_page(self, extract: Extract) -> None:
        html = _html(
            _a("next.html")
            + _a("../api/")
            + _a("/about")
            + _a("//acme.example/x")
            + _a("?q=1")
            + _a("#section")  # same page: dropped
            + _a("intro.html#top")  # same page: dropped
        )

        result = extract(html)

        assert _follow(result) == [
            "https://acme.example/docs/guide/next.html",
            "https://acme.example/docs/api/",
            "https://acme.example/about",
            "https://acme.example/x",
            "https://acme.example/docs/guide/intro.html?q=1",
        ]

    def test_links_are_normalized_and_deduplicated(self, extract: Extract) -> None:
        html = _html(
            _a("/Page?b=2&a=1", "first")
            + _a("HTTPS://ACME.example:443/Page?a=1&b=2&utm_source=mail", "second")
            + _a("/page", "lowercase path is a different URL")
        )

        result = extract(html)

        assert _follow(result) == ["https://acme.example/Page?a=1&b=2", "https://acme.example/page"]
        assert result.follow[0].text == "first"  # first occurrence kept

    def test_page_url_is_normalized(self, extract: Extract) -> None:
        result = extract(
            _html(_a("/docs/guide/intro.html")), "HTTPS://Acme.Example/docs/guide/intro.html#x"
        )

        assert result.page_url == _PAGE
        assert result.follow == []  # the link points at the page itself

    # ---- <base href> ---- #

    @pytest.mark.parametrize(
        ("head", "expected_base", "expected_link"),
        [
            ('<base href="/v2/">', "https://acme.example/v2/", "https://acme.example/v2/page.html"),
            (
                '<base href="../">',
                "https://acme.example/docs/",
                "https://acme.example/docs/page.html",
            ),
            (
                '<base target="_blank"><base href="/v2/"><base href="/v3/">',
                "https://acme.example/v2/",  # the first <base> with an href wins
                "https://acme.example/v2/page.html",
            ),
            ('<base href="">', _PAGE, "https://acme.example/docs/guide/page.html"),
        ],
        ids=["absolute-path", "relative", "first-href-wins", "empty"],
    )
    def test_base_href(
        self, extract: Extract, head: str, expected_base: str, expected_link: str
    ) -> None:
        result = extract(_html(_a("page.html"), head))

        assert result.base_url == expected_base
        assert _follow(result) == [expected_link]

    def test_base_on_another_host_moves_links_out_of_scope(self, extract: Extract) -> None:
        result = extract(_html(_a("page.html"), '<base href="https://cdn.other.example/">'))

        assert result.follow == []
        assert _rejections(result) == {
            "https://cdn.other.example/page.html": ScopeRejection.OTHER_HOST
        }

    # ---- scope, schemes, invalid ---- #

    def test_scope_is_applied_with_depth_plus_one(self, extract: Extract) -> None:
        html = _html(_a("https://offsite.example/a") + _a("/logo.png") + _a("/deeper.html"))

        result = extract(html, depth=5, max_depth=5)

        rejections = {r.url: (r.decision.reason, r.decision.detail) for r in result.rejected}
        assert rejections == {
            "https://offsite.example/a": (PageReason.OUT_OF_SCOPE, ScopeRejection.OTHER_HOST),
            "https://acme.example/logo.png": (
                PageReason.OUT_OF_SCOPE,
                ScopeRejection.ASSET_EXTENSION,
            ),
            "https://acme.example/deeper.html": (PageReason.MAX_DEPTH, ScopeRejection.TOO_DEEP),
        }

    def test_non_http_schemes_are_rejected_not_invalid(self, extract: Extract) -> None:
        hrefs = [
            "mailto:a@acme.example",
            "tel:+15550100",
            "javascript:void(0)",
            "data:text/html,hi",
            "ftp://acme.example/f",
        ]

        result = extract(_html("".join(_a(h) for h in hrefs)))

        assert set(_rejections(result)) == set(hrefs)
        assert set(_rejections(result).values()) == {ScopeRejection.UNSUPPORTED_SCHEME}
        assert result.invalid == 0

    def test_invalid_and_empty_hrefs(self, extract: Extract) -> None:
        html = _html(
            _a("http://") + _a("https://[bad") + _a("   ") + "<a name='anchor'>no href</a>"
        )

        result = extract(html)

        assert result.invalid == 2
        assert result.follow == [] and result.rejected == []

    # ---- nofollow policy ---- #

    def test_rel_nofollow_links_are_held_back(self, extract: Extract) -> None:
        html = _html(
            _a("/a", rel="nofollow") + _a("/b", rel="external NoFollow") + _a("/c", rel="noopener")
        )

        result = extract(html)

        assert _follow(result) == ["https://acme.example/c"]
        assert [link.url for link in result.nofollow] == [
            "https://acme.example/a",
            "https://acme.example/b",
        ]

    @pytest.mark.parametrize("nofollow_first", [True, False])
    def test_a_followable_link_wins_over_nofollow(
        self, extract: Extract, nofollow_first: bool
    ) -> None:
        links = [_a("/a", rel="nofollow"), _a("/a")]
        html = _html("".join(links if nofollow_first else reversed(links)))

        result = extract(html)

        assert _follow(result) == ["https://acme.example/a"]
        assert result.nofollow == []

    def test_nofollow_can_be_ignored_by_config(self, extract: Extract) -> None:
        html = _html(_a("/a", rel="nofollow"), '<meta name="robots" content="nofollow">')

        result = extract(html, CRAWL_RESPECT_NOFOLLOW=False)

        assert _follow(result) == ["https://acme.example/a"]
        assert result.nofollow_page  # still reported

    @pytest.mark.parametrize(
        ("meta", "noindex", "nofollow"),
        [
            ('<meta name="robots" content="nofollow">', False, True),
            ('<meta name="ROBOTS" content="NoIndex, NoFollow">', True, True),
            ('<meta name="robots" content="none">', True, True),
            ('<meta name="robots" content="noindex">', True, False),
            ('<meta name="RagEngineBot" content="nofollow">', False, True),
            ('<meta name="googlebot" content="noindex, nofollow">', False, False),
            ('<meta name="robots" content="index, follow">', False, False),
        ],
    )
    def test_meta_robots(self, extract: Extract, meta: str, noindex: bool, nofollow: bool) -> None:
        result = extract(_html(_a("/a"), meta))

        assert (result.noindex, result.nofollow_page) == (noindex, nofollow)
        assert bool(result.nofollow) is nofollow
        assert bool(result.follow) is not nofollow

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("noindex", {"noindex"}),
            ("noindex, nofollow", {"noindex", "nofollow"}),
            ("googlebot: nofollow", set()),
            ("RagEngineBot: nofollow", {"nofollow"}),
            ("googlebot: noindex, ragenginebot: nofollow", {"nofollow"}),
            ("ragenginebot: noindex, nofollow, otherbot: none", {"noindex", "nofollow"}),
            (
                "unavailable_after: 25 Jun 2010 15:00:00 PST",
                {"unavailable_after: 25 jun 2010 15:00:00 pst"},
            ),
            ("", set()),
            (None, set()),
        ],
    )
    def test_parse_x_robots_tag(self, value: str | None, expected: set[str]) -> None:
        assert parse_x_robots_tag(value, "RagEngineBot") == expected

    def test_x_robots_tag_header_applies(self, extract: Extract) -> None:
        result = extract(_html(_a("/a")), headers={"X-Robots-Tag": "noindex, nofollow"})

        assert result.noindex and result.nofollow_page
        assert result.follow == []

    # ---- canonical ---- #

    def test_html_canonical_is_resolved(self, extract: Extract) -> None:
        result = extract(_html("", '<link rel="canonical" href="../intro/?utm_source=x">'))

        assert result.canonical_url == "https://acme.example/docs/intro/"
        assert not result.canonical_is_self

    def test_self_canonical(self, extract: Extract) -> None:
        result = extract(_html("", f'<link rel="canonical" href="{_PAGE}#top">'))

        assert result.canonical_is_self

    def test_link_header_canonical_and_html_precedence(self, extract: Extract) -> None:
        header = {
            "Link": '<https://acme.example/style.css>; rel="preload", </canon>; rel="canonical"'
        }

        from_header = extract(_html(""), headers=header)
        from_html = extract(_html("", '<link rel="canonical" href="/html">'), headers=header)

        assert from_header.canonical_url == "https://acme.example/canon"
        assert from_html.canonical_url == "https://acme.example/html"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ('<https://a.example/x>; rel="canonical"', "https://a.example/x"),
            ("<https://a.example/x>; rel=canonical", "https://a.example/x"),
            ('<https://a.example/p.css>; rel="preload"', None),
            (
                '<https://a.example/a>; rel="alternate", <https://a.example/c>; rel="canonical"',
                "https://a.example/c",
            ),
            (None, None),
        ],
    )
    def test_canonical_from_link_header(self, value: str | None, expected: str | None) -> None:
        assert canonical_from_link_header(value) == expected

    # ---- other sources, documents, anchor text ---- #

    @pytest.mark.parametrize(
        "content",
        ["0; url=/moved.html", "5;URL='https://acme.example/moved.html'", "0,url=moved.html"],
    )
    def test_meta_refresh_target_is_a_link(self, extract: Extract, content: str) -> None:
        page = "https://acme.example/old.html"

        result = extract(_html("", f'<meta http-equiv="Refresh" content="{content}">'), page)

        assert _follow(result) == ["https://acme.example/moved.html"]
        assert result.follow[0].source is LinkSource.META_REFRESH

    def test_meta_refresh_without_url_is_ignored(self, extract: Extract) -> None:
        result = extract(_html("", '<meta http-equiv="refresh" content="30">'))

        assert result.follow == []

    def test_area_links(self, extract: Extract) -> None:
        html = _html('<map name="m"><area href="/region.html" alt="Region"></map>')

        result = extract(html)

        assert _follow(result) == ["https://acme.example/region.html"]
        assert result.follow[0].source is LinkSource.AREA

    def test_documents_are_flagged(self, extract: Extract) -> None:
        html = _html(_a("/f/spec.PDF") + _a("/f/guide.docx") + _a("/page.html") + _a("/docs/"))

        result = extract(html)

        assert [link.is_document for link in result.follow] == [True, True, False, False]

    def test_anchor_text_is_collapsed_and_truncated(self, extract: Extract) -> None:
        html = _html(f'<a href="/a">  Read\n   <b>the</b>\tguide {"x" * 300}</a>')

        text = extract(html).follow[0].text

        assert text.startswith("Read the guide x")
        assert len(text) == 200

    def test_links_per_page_cap(self, extract: Extract) -> None:
        html = _html("".join(_a(f"/p{i}") for i in range(10)))

        result = extract(html, CRAWL_MAX_LINKS_PER_PAGE=3)

        assert result.capped
        assert _follow(result) == [f"https://acme.example/p{i}" for i in range(3)]

    # ---- robustness ---- #

    @pytest.mark.parametrize("html", ["", "   ", "<<<>>>", "<a href='/x'", "\x00\x01 garbage"])
    def test_malformed_html_does_not_raise(self, extract: Extract, html: str) -> None:
        result = extract(html)

        assert result.invalid == 0

    @pytest.mark.parametrize(
        "head",
        ['<base href="https://[broken/">', '<link rel="canonical" href="https://[broken/">'],
        ids=["base", "canonical"],
    )
    def test_malformed_base_or_canonical_is_ignored(self, extract: Extract, head: str) -> None:
        # regression: normalize_url(href, base=...) leaked urljoin's ValueError (Step 23)
        result = extract(_html(_a("/a"), head))

        assert result.base_url == _PAGE
        assert result.canonical_url is None
        assert _follow(result) == ["https://acme.example/a"]

    @pytest.mark.parametrize(
        ("url", "base"),
        [("https://[bad", "https://a.example/"), ("/x", "https://[bad"), ("https://[bad", None)],
    )
    def test_normalize_url_raises_its_own_error_for_malformed_ipv6(
        self, url: str, base: str | None
    ) -> None:
        with pytest.raises(UrlNormalizationError):
            normalize_url(url, base=base)

    def test_xhtml_with_encoding_declaration(self, extract: Extract) -> None:
        # lxml refuses `str` input carrying an XML encoding declaration; we feed it bytes
        html = '<?xml version="1.0" encoding="ISO-8859-1"?>' + _html(_a("/café.html", "Café"))

        result = extract(html)

        assert _follow(result) == ["https://acme.example/caf%C3%A9.html"]
        assert result.follow[0].text == "Café"

    def test_filter_urls_for_non_html_sources(self, override_settings: OverrideSettings) -> None:
        scope = CrawlScope.from_config("https://acme.example/", max_depth=5)
        extractor = LinkExtractor(override_settings())

        result = extractor.filter_urls(
            [
                "https://acme.example/a",
                "/b",
                "https://acme.example/a",
                "mailto:x@y",
                "https://o.example/",
            ],
            "https://acme.example/doc.pdf",
            scope=scope,
            depth=1,
        )

        assert _follow(result) == ["https://acme.example/a", "https://acme.example/b"]
        assert len(result.rejected) == 2

    # ---- fixture site ---- #

    async def test_fixture_home_page(self, fetcher: HttpFetcher, fixture_site: FixtureSite) -> None:
        page = await fetcher.fetch(fixture_site.url("/"))
        scope = CrawlScope.from_config(page.final_url, max_depth=10)
        assert page.text is not None

        result = LinkExtractor().extract(
            page.text, page.final_url, scope=scope, depth=0, headers=page.headers
        )

        follow = set(_follow(result))
        for path in (
            "/docs/",
            "/docs/getting-started.html",
            "/files/widget-spec.pdf",
            "/redirect-me",
        ):
            assert fixture_site.url(path) in follow
        # the tracked link keeps its non-tracking parameter, so it is a distinct URL
        assert fixture_site.url("/docs/getting-started.html?ref=home") in follow
        assert {link.url for link in result.follow if link.is_document} == {
            fixture_site.url(p)
            for p in (
                "/files/widget-spec.pdf",
                "/files/huge-catalog.pdf",
                "/files/install-guide.docx",
                "/files/roadmap.pptx",
                "/files/pricing.xlsx",
                "/files/scanned.pdf",
                "/files/encrypted.pdf",
            )
        }
        assert [link.url for link in result.nofollow] == [fixture_site.url("/nofollow-target.html")]
        # every rejected link that the fixture documents is expected to be out of scope
        documented = {fixture_site.url(p): e for p, e in SITE_PAGES.items()}
        assert {documented[r.url] for r in result.rejected} == {Expect.OUT_OF_SCOPE}
        assert len(result.rejected) == 7  # other host, offsite, json, png, mailto, tel, js
        assert fixture_site.url("/") not in follow  # "#downloads" is the page itself

    async def test_fixture_relative_link_and_canonical(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        scope = CrawlScope.from_config(fixture_site.url("/"), max_depth=10)
        extractor = LinkExtractor()
        reference = await fetcher.fetch(fixture_site.url("/docs/api/reference.html"))
        variant = await fetcher.fetch(fixture_site.url("/canonical-variant.html"))
        assert reference.text is not None and variant.text is not None

        ref_links = extractor.extract(reference.text, reference.final_url, scope=scope, depth=3)
        var_links = extractor.extract(variant.text, variant.final_url, scope=scope, depth=1)

        assert fixture_site.url("/docs/api/deep/level4.html") in _follow(ref_links)
        assert var_links.canonical_url == fixture_site.url("/docs/getting-started.html")
        assert not var_links.canonical_is_self
