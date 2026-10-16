"""Fixture website: every edge-case route behaves as documented in `SITE_PAGES`."""

import gzip
import io
import re
from collections.abc import Iterator
from urllib.parse import urljoin

import httpx
import pymupdf
import pytest
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

from tests.support.fixture_site import (
    FLAKY_FAILURES,
    LARGE_BODY_BYTES,
    METADATA_URL,
    SITE_PAGES,
    SITE_ROOT,
    Expect,
    FixtureSite,
)

_HREF = re.compile(r'href="([^"]+)"')


@pytest.fixture
def http(fixture_site: FixtureSite) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=fixture_site.base_url, follow_redirects=False, timeout=10) as client:
        yield client


def _paths(expect: Expect) -> list[str]:
    return [p for p, e in SITE_PAGES.items() if e is expect]


# ---- #
# Tests
# ---- #


class TestFixtureSite_S9:
    """S9 — Fixture website: static pages, templating, dynamic edge cases, documents."""

    def test_home_page_templates_absolute_urls(
        self, http: httpx.Client, fixture_site: FixtureSite
    ) -> None:
        response = http.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert f'href="{fixture_site.other_host_url}/docs/"' in response.text
        assert "{{" not in response.text

    def test_no_unrendered_placeholders_anywhere(self, http: httpx.Client) -> None:
        for path in SITE_ROOT.rglob("*"):
            if path.suffix in {".html", ".xml", ".txt"}:
                rel = "/" + path.relative_to(SITE_ROOT).as_posix()
                assert "{{" not in http.get(rel).text, rel

    def test_manifest_covers_every_link_on_home_page(
        self, http: httpx.Client, fixture_site: FixtureSite
    ) -> None:
        hrefs = {href.replace("&amp;", "&") for href in _HREF.findall(http.get("/").text)} - {
            "#downloads"
        }
        linked = {urljoin(fixture_site.base_url + "/", href) for href in hrefs}
        # the tracked link is the same page as /docs/getting-started.html after normalization
        linked = {u for u in linked if "utm_source" not in u}
        manifest = {fixture_site.url(p) for p in SITE_PAGES}

        assert linked - manifest == set()

    @pytest.mark.parametrize("path", _paths(Expect.HTML) + _paths(Expect.SITEMAP_ONLY))
    def test_html_pages_are_served(self, http: httpx.Client, path: str) -> None:
        response = http.get(path)

        assert response.status_code == 200, path
        assert response.headers["content-type"].startswith("text/html")
        assert "<h1>" in response.text

    def test_pages_share_boilerplate(self, http: httpx.Client) -> None:
        pages = [http.get(p).text for p in ("/about.html", "/docs/", "/blog/post-1.html")]

        for marker in ("cookie-banner", "© 2026 Acme Widgets Inc.", 'aria-label="Main"'):
            assert all(marker in page for page in pages)

    def test_directory_without_slash_redirects(self, http: httpx.Client) -> None:
        response = http.get("/docs")

        assert response.status_code == 301
        assert response.headers["location"] == "/docs/"

    def test_robots_disallows_private_and_points_to_sitemap(
        self, http: httpx.Client, fixture_site: FixtureSite
    ) -> None:
        robots = http.get("/robots.txt").text

        assert "Disallow: /private/" in robots
        assert "Crawl-delay: 0.1" in robots
        assert f"Sitemap: {fixture_site.base_url}/sitemap_index.xml" in robots

    def test_sitemap_index_and_gzipped_child(
        self, http: httpx.Client, fixture_site: FixtureSite
    ) -> None:
        index = http.get("/sitemap_index.xml").text
        gz = http.get("/sitemap-docs.xml.gz")
        pages = http.get("/sitemap-pages.xml").text

        assert f"<loc>{fixture_site.base_url}/sitemap-docs.xml.gz</loc>" in index
        assert gz.headers["content-type"] == "application/gzip"
        assert (
            f"{fixture_site.base_url}/docs/configuration.html"
            in gzip.decompress(gz.content).decode()
        )
        assert f"{fixture_site.base_url}/orphan.html" in pages
        assert "https://offsite.example/from-sitemap" in pages

    def test_orphan_page_is_not_linked_from_html(self, http: httpx.Client) -> None:
        linked = [
            p
            for p in _paths(Expect.HTML)
            if p.endswith((".html", "/")) and "orphan.html" in http.get(p).text
        ]

        assert linked == []

    @pytest.mark.parametrize(
        ("path", "status", "location"),
        [
            ("/redirect-me", 301, "/docs/getting-started.html"),
            ("/redirect-chain", 302, "/redirect-me"),
            ("/loop-a", 302, "/loop-b"),
            ("/loop-b", 302, "/loop-a"),
            ("/redirect-metadata", 302, METADATA_URL),
        ],
    )
    def test_redirects(self, http: httpx.Client, path: str, status: int, location: str) -> None:
        response = http.get(path)

        assert response.status_code == status
        assert response.headers["location"] == location

    def test_error_and_missing(self, http: httpx.Client) -> None:
        assert http.get("/error").status_code == 500
        assert http.get("/missing.html").status_code == 404

    def test_flaky_fails_then_recovers_and_reset_rearms(
        self, http: httpx.Client, fixture_site: FixtureSite
    ) -> None:
        statuses = [http.get("/flaky").status_code for _ in range(FLAKY_FAILURES + 1)]
        fixture_site.reset()
        after_reset = http.get("/flaky")

        assert statuses == [503] * FLAKY_FAILURES + [200]
        assert after_reset.status_code == 503
        assert after_reset.headers["retry-after"] == "1"

    def test_rate_limited_always_429(self, http: httpx.Client) -> None:
        responses = [http.get("/rate-limited") for _ in range(3)]

        assert {r.status_code for r in responses} == {429}
        assert responses[0].headers["retry-after"] == "2"

    def test_slow_page_times_out_and_delay_is_tunable(self, http: httpx.Client) -> None:
        with pytest.raises(httpx.ReadTimeout):
            http.get("/slow", timeout=httpx.Timeout(5, read=0.3))

        assert http.get("/slow", params={"delay": "0.05"}).status_code == 200

    def test_large_html_streams_without_content_length(self, http: httpx.Client) -> None:
        received = 0
        with http.stream("GET", "/large.html") as response:
            assert "content-length" not in response.headers
            for chunk in response.iter_bytes():
                received += len(chunk)
                if received > 21 * 1024 * 1024:
                    break

        assert received > 20 * 1024 * 1024

    def test_large_bin_declares_content_length(self, http: httpx.Client) -> None:
        with http.stream("GET", "/files/large.bin") as response:
            assert int(response.headers["content-length"]) == LARGE_BODY_BYTES
            assert response.headers["content-type"] == "application/octet-stream"

    @pytest.mark.parametrize(
        ("path", "content_type"),
        [("/data.json", "application/json"), ("/images/logo.png", "image/png")],
    )
    def test_unsupported_types(self, http: httpx.Client, path: str, content_type: str) -> None:
        assert http.get(path).headers["content-type"] == content_type

    def test_spa_has_no_content_without_javascript(self, http: httpx.Client) -> None:
        html = http.get("/app/").text
        without_scripts = re.sub(r"<script.*?</script>", "", html, flags=re.S)

        assert '<div id="root"></div>' in without_scripts
        assert "Choose a chassis" not in without_scripts
        assert "Choose a chassis" in html  # only inside the script
        assert "<noscript>" in html

    def test_latin1_page_declares_charset_only_in_meta(self, http: httpx.Client) -> None:
        response = http.get("/latin1.html")

        assert response.headers["content-type"] == "text/html"
        assert "Crème brûlée" in response.content.decode("latin-1")
        assert b"charset=iso-8859-1" in response.content

    def test_duplicate_pages_share_main_content(self, http: httpx.Client) -> None:
        def main(path: str) -> str:
            match = re.search(r"<main>(.*)</main>", http.get(path).text, flags=re.S)
            assert match is not None
            return match.group(1)

        assert main("/dup-a.html") == main("/dup-b.html")

    def test_canonical_variant_points_elsewhere(
        self, http: httpx.Client, fixture_site: FixtureSite
    ) -> None:
        html = http.get("/canonical-variant.html").text

        target = f"{fixture_site.base_url}/docs/getting-started.html"
        assert f'<link rel="canonical" href="{target}">' in html

    def test_nofollow_link_is_marked(self, http: httpx.Client) -> None:
        assert 'href="/nofollow-target.html" rel="nofollow"' in http.get("/").text

    def test_other_hostname_reaches_same_server(self, fixture_site: FixtureSite) -> None:
        response = httpx.get(f"{fixture_site.other_host_url}/docs/")

        assert response.status_code == 200
        assert fixture_site.requests[-1].host == f"localhost:{fixture_site.port}"

    def test_path_traversal_is_rejected(self, fixture_site: FixtureSite) -> None:
        response = httpx.get(f"{fixture_site.base_url}/%2e%2e/conftest.py")

        assert response.status_code == 404

    def test_requests_are_recorded_and_reset(
        self, http: httpx.Client, fixture_site: FixtureSite
    ) -> None:
        http.get("/about.html", headers={"User-Agent": "RagEngineBot/test"})
        http.get("/about.html?x=1")

        assert fixture_site.hits("/about.html") == 1
        assert fixture_site.hits("/about.html?x=1") == 1
        assert fixture_site.requests[0].user_agent == "RagEngineBot/test"
        fixture_site.reset()
        assert fixture_site.requests == []

    def test_pdf_has_text(self, http: httpx.Client) -> None:
        doc = pymupdf.open(stream=http.get("/files/widget-spec.pdf").content, filetype="pdf")

        text = "".join(page.get_text() for page in doc)
        assert doc.page_count == 2
        assert "WX-2048" in text
        assert "M12 A-coded" in text

    def test_scanned_pdf_has_no_text_layer(self, http: httpx.Client) -> None:
        doc = pymupdf.open(stream=http.get("/files/scanned.pdf").content, filetype="pdf")

        assert "".join(page.get_text() for page in doc).strip() == ""

    def test_encrypted_pdf_needs_password(self, http: httpx.Client) -> None:
        doc = pymupdf.open(stream=http.get("/files/encrypted.pdf").content, filetype="pdf")

        assert doc.needs_pass

    def test_docx_has_headings_list_and_table(self, http: httpx.Client) -> None:
        doc = Document(io.BytesIO(http.get("/files/install-guide.docx").content))

        headings = [
            p.text for p in doc.paragraphs if p.style and (p.style.name or "").startswith("Heading")
        ]
        assert headings == ["Installation Guide", "Mounting", "Wiring"]
        assert doc.tables[0].rows[1].cells[1].text == "+24 V"

    def test_pptx_has_slides_notes_and_table(self, http: httpx.Client) -> None:
        prs = Presentation(io.BytesIO(http.get("/files/roadmap.pptx").content))

        slides = list(prs.slides)
        assert len(slides) == 3
        assert slides[0].shapes.title.text == "Product Roadmap 2026"
        assert "Speaker notes" in slides[0].notes_slide.notes_text_frame.text
        assert any(shape.has_table for shape in slides[2].shapes)

    def test_xlsx_has_two_sheets(self, http: httpx.Client) -> None:
        wb = load_workbook(io.BytesIO(http.get("/files/pricing.xlsx").content), read_only=True)

        assert wb.sheetnames == ["Prices", "Discounts"]
        rows = list(wb["Prices"].iter_rows(values_only=True))
        assert rows[1] == ("WX-2048", "Modular sorter", 1899)
