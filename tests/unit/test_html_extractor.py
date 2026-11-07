"""HTML extractor: trafilatura main content → Markdown, markdownify fallback, metadata."""

from collections.abc import AsyncIterator

import pytest

import rag_engine.extraction.html as html_module
from rag_engine.config import Settings
from rag_engine.crawling.fetchers import HttpFetcher
from rag_engine.domain import ContentKind
from rag_engine.extraction import (
    EmptyContentError,
    ExtractedDocument,
    ExtractionInput,
    ExtractorRegistry,
    HtmlExtractor,
)
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import FixtureSite

_URL = "https://acme.example/docs/page.html"


def _page(body: str, head: str = "", html_attrs: str = ' lang="en"') -> str:
    return f"<!doctype html><html{html_attrs}><head>{head}</head><body>{body}</body></html>"


async def _extract(html: str, url: str = _URL, **headers: str) -> ExtractedDocument:
    source = ExtractionInput(
        url=url,
        mime_type="text/html",
        body=html.encode("utf-8"),
        text=html,
        encoding="utf-8",
        headers={k.replace("_", "-"): v for k, v in headers.items()},
    )
    return await HtmlExtractor().extract(source)


@pytest.fixture
async def fetcher(override_settings: OverrideSettings) -> AsyncIterator[HttpFetcher]:
    config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


async def _extract_fixture(fetcher: HttpFetcher, site: FixtureSite, path: str) -> ExtractedDocument:
    result = await fetcher.fetch(site.url(path))
    return await HtmlExtractor().extract(ExtractionInput.from_fetch(result))


_ARTICLE = """
<header class="site-header"><nav><a href="/">Home</a> | <a href="/docs/">Docs</a></nav></header>
<main>
<h1>Calibrating the torque sensor</h1>
<p>The torque sensor drifts by up to two percent per year, so calibrate it every quarter.
Calibration takes about ten minutes and requires the reference weight kit.</p>
<h2>Steps</h2>
<ol><li>Unload the conveyor belt completely.</li><li>Attach the reference weight.</li>
<li>Run the calibration command shown below.</li></ol>
<pre><code class="language-python">from widgets import Sensor
Sensor("torque").calibrate(reference_kg=5)
</code></pre>
<h2>Tolerances</h2>
<table><tr><th>Model</th><th>max_drift_pct</th></tr>
<tr><td>WX-2048</td><td>2.0</td></tr><tr><td>WX-4096</td><td>1.5</td></tr></table>
</main>
<footer class="site-footer"><p>© 2026 Acme Widgets Inc. All rights reserved.</p></footer>
"""


# ---- #
# Tests
# ---- #


class TestHtmlExtractor_S39:
    """S39 — HTML extractor: trafilatura → Markdown (tables/code/headings), fallback, metadata."""

    # ---- main content ---- #

    async def test_structure_is_preserved(self) -> None:
        document = await _extract(_page(_ARTICLE))
        markdown = document.markdown

        assert markdown.startswith("# Calibrating the torque sensor\n")
        assert "\n## Steps\n" in markdown and "\n## Tolerances\n" in markdown
        assert "1. Unload the conveyor belt completely." in markdown
        assert (
            '```python\nfrom widgets import Sensor\nSensor("torque").calibrate(reference_kg=5)\n```'
            in markdown
        )
        assert "| Model | max_drift_pct |" in markdown  # no "max\\_drift\\_pct" escapes
        assert "| WX-4096 | 1.5 |" in markdown
        assert document.metadata["extraction"] == "trafilatura"

    async def test_page_chrome_is_dropped(self) -> None:
        markdown = (await _extract(_page(_ARTICLE))).markdown

        assert "Home" not in markdown
        assert "All rights reserved" not in markdown

    async def test_fixture_docs_page(self, fetcher: HttpFetcher, fixture_site: FixtureSite) -> None:
        document = await _extract_fixture(fetcher, fixture_site, "/docs/configuration.html")

        assert document.title == "Configuration reference"
        assert (
            document.description
            == "Configuration reference - Acme Widgets documentation fixture page."
        )
        assert document.lang == "en"
        assert (
            "| listen_port | integer | 7070 | TCP port for the control API |" in document.markdown
        )
        assert "```toml\nlisten_port = 7070\n" in document.markdown
        assert document.outlinks == ()  # HTML links come from LinkExtractor (Step 32)

    async def test_fixture_guide_keeps_inline_formatting(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        document = await _extract_fixture(fetcher, fixture_site, "/docs/getting-started.html")

        assert "`/var/lib/widgetd`" in document.markdown
        assert "**ONLINE**" in document.markdown
        assert "### Verify the registration" in document.markdown
        assert document.markdown.count("```bash\n") == 2

    async def test_latin1_page_is_decoded(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        document = await _extract_fixture(fetcher, fixture_site, "/latin1.html")

        assert document.title == "Café menu"
        assert document.lang == "fr"
        assert "Crème brûlée" in document.markdown

    async def test_short_pages_are_kept(self) -> None:
        document = await _extract(
            _page("<main><h1>Status</h1><p>All widget lines operational.</p></main>")
        )

        assert "All widget lines operational." in document.markdown

    async def test_rendered_spa_dom(self) -> None:
        # what the browser returns for the fixture SPA (Step 36)
        rendered = _page(
            '<div id="root"><h1>Widget configurator</h1><p>Choose a chassis, a sensor package, '
            "and a firmware channel to configure your widget.</p></div><script>/* app */</script>"
        )

        document = await _extract(rendered)

        assert "# Widget configurator" in document.markdown
        assert "firmware channel" in document.markdown

    async def test_unrendered_js_shell_is_empty(self) -> None:
        shell = _page(
            "<noscript>You need to enable JavaScript to run this app.</noscript>"
            '<div id="root"></div><script src="/main.js"></script>'
        )

        with pytest.raises(EmptyContentError):
            await _extract(shell)

    @pytest.mark.parametrize("html", ["", "   ", "<html><body><script>x()</script></body></html>"])
    async def test_no_text_is_empty_content(self, html: str) -> None:
        with pytest.raises(EmptyContentError):
            await _extract(html)

    # ---- markdownify fallback ---- #

    async def test_fallback_when_trafilatura_finds_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(html_module, "trafilatura_extract", lambda *_args, **_kwargs: None)

        document = await _extract(_page(_ARTICLE + "<script>track()</script><form><input></form>"))
        markdown = document.markdown

        assert document.metadata["extraction"] == "markdownify"
        assert markdown.startswith("# Calibrating the torque sensor")
        assert "## Steps" in markdown
        assert "```python\nfrom widgets import Sensor" in markdown
        assert "| WX-2048 | 2.0 |" in markdown
        assert "Home" not in markdown and "All rights reserved" not in markdown  # nav/footer
        assert "track()" not in markdown

    async def test_fallback_when_trafilatura_crashes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("parser bug")

        monkeypatch.setattr(html_module, "trafilatura_extract", boom)

        document = await _extract(_page(_ARTICLE))

        assert document.metadata["extraction"] == "markdownify"
        assert "torque sensor drifts" in document.markdown

    # ---- metadata ---- #

    @pytest.mark.parametrize(
        ("head", "body", "title"),
        [
            (
                "<title>Pricing | Acme Widgets</title>",
                "<p>Three plans for teams of every size.</p>",
                "Pricing",  # site-name suffix removed
            ),
            ("<title>Pricing | Acme Widgets</title>", "<h1>Plans</h1><p>Three plans.</p>", "Plans"),
            ("", "<main><h1>Only a heading</h1><p>Some words here.</p></main>", "Only a heading"),
        ],
        ids=["title-tag-cleaned", "h1-preferred", "h1-without-title"],
    )
    async def test_title(self, head: str, body: str, title: str) -> None:
        assert (await _extract(_page(body, head))).title == title

    async def test_description_falls_back_to_open_graph(self) -> None:
        head = '<meta property="og:description" content="Open Graph summary.">'

        document = await _extract(_page("<p>Body text of the page.</p>", head))

        assert document.description == "Open Graph summary."

    @pytest.mark.parametrize(
        ("html_attrs", "head", "header", "lang"),
        [
            (' lang="en-US"', "", None, "en"),
            (' lang="pt_BR"', "", None, "pt"),
            (' xml:lang="de"', "", None, "de"),
            ("", "", "nl-BE, en;q=0.5", "nl"),
            ("", '<meta http-equiv="content-language" content="es">', None, "es"),
            (' lang="ja"', "", "en", "ja"),  # markup wins over the header
            (' lang="not a language"', "", None, None),
            ("", "", None, None),
        ],
    )
    async def test_language(
        self, html_attrs: str, head: str, header: str | None, lang: str | None
    ) -> None:
        html = _page("<p>Some body text for the page.</p>", head, html_attrs=html_attrs)
        headers = {"content_language": header} if header else {}

        assert (await _extract(html, **headers)).lang == lang

    async def test_author_and_date_from_markup_only(self) -> None:
        head = (
            '<meta name="author" content="Dana Engineer">'
            '<meta property="article:published_time" content="2025-03-14T09:00:00Z">'
            '<meta property="og:site_name" content="Acme Engineering Blog">'
        )
        document = await _extract(_page("<article><p>Release notes for March.</p></article>", head))
        undated = await _extract(_page("<p>Written in 2019, updated © 2026 Acme.</p>"))

        assert document.metadata["author"] == "Dana Engineer"
        assert document.metadata["published"] == "2025-03-14"
        assert document.metadata["site_name"] == "Acme Engineering Blog"
        assert undated.metadata["published"] is None  # no date guessed from body text
        assert undated.metadata["site_name"] is None  # not the URL's host name

    # ---- inputs ---- #

    async def test_body_is_decoded_when_text_is_missing(self) -> None:
        # e.g. HTML sniffed from a response labelled application/pdf (Step 38)
        html = _page("<p>Café crème.</p>", '<meta charset="iso-8859-1">')
        source = ExtractionInput(url=_URL, mime_type="application/pdf", body=html.encode("latin-1"))

        document = await HtmlExtractor().extract(source)

        assert "Café crème." in document.markdown

    async def test_xhtml_with_encoding_declaration(self) -> None:
        xhtml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en"><head><title>X</title></head>'
            "<body><h1>XHTML page</h1><p>Strict markup still extracts fine.</p></body></html>"
        )

        document = await _extract(xhtml)

        assert "Strict markup still extracts fine." in document.markdown
        assert document.lang == "en"

    async def test_registered_in_the_registry(self) -> None:
        registry = ExtractorRegistry([HtmlExtractor()])
        html = _page("<main><h1>Registry</h1><p>Dispatch by sniffed kind.</p></main>")
        source = ExtractionInput(url=_URL, mime_type="text/html", body=html.encode(), text=html)

        document = await registry.extract(source)

        assert (document.content_kind, document.extractor) == (ContentKind.HTML, "html")
