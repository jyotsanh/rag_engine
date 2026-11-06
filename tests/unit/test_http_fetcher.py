"""HTTP fetcher against the fixture site: kinds, sizes, timeouts, charsets, typed errors."""

import time
from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.crawling.fetchers import (
    FetchResult,
    FetchTimeoutError,
    HttpFetcher,
    HttpStatusError,
    NetworkFetchError,
    RedirectLoopFetchError,
    SsrfFetchError,
    TooLargeError,
    TooManyRedirectsFetchError,
    UnsupportedContentTypeError,
    decode_html,
    kind_for,
    parse_content_type,
    parse_retry_after,
)
from rag_engine.domain import ContentKind, PageReason, RenderedWith
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import FixtureSite


@pytest.fixture
def config(override_settings: OverrideSettings) -> Settings:
    return override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True,
        CRAWL_MAX_PAGE_BYTES=2 * 1024 * 1024,  # small cap keeps the size tests fast
        CRAWL_READ_TIMEOUT_SECONDS=2,
        CRAWL_FETCH_TIMEOUT_SECONDS=3,
    )


@pytest.fixture
async def fetcher(config: Settings) -> AsyncIterator[HttpFetcher]:
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


async def _fetch(fetcher: HttpFetcher, site: FixtureSite, path: str) -> FetchResult:
    return await fetcher.fetch(site.base_url + path)


# ---- #
# Tests
# ---- #


class TestHttpFetcher_S27:
    """S27 — HTTP fetcher: streaming caps, timeouts, content-type allowlist, charset, errors."""

    async def test_html_page(self, fetcher: HttpFetcher, fixture_site: FixtureSite) -> None:
        result = await _fetch(fetcher, fixture_site, "/about.html")

        assert result.status_code == 200
        assert result.content_kind is ContentKind.HTML
        assert result.mime_type == "text/html"
        assert result.encoding == "utf-8"
        assert result.text is not None and "About Acme Widgets" in result.text
        assert result.bytes_downloaded == len(result.body)
        assert result.rendered_with is RenderedWith.HTTP
        assert result.elapsed_ms >= 0

    async def test_redirects_are_followed(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        result = await _fetch(fetcher, fixture_site, "/redirect-chain")

        assert result.final_url == f"{fixture_site.base_url}/docs/getting-started.html"
        assert len(result.redirect_chain) == 3
        assert result.requested_url.endswith("/redirect-chain")

    @pytest.mark.parametrize(
        ("path", "kind"),
        [
            ("/files/widget-spec.pdf", ContentKind.PDF),
            ("/files/install-guide.docx", ContentKind.DOCX),
            ("/files/roadmap.pptx", ContentKind.PPTX),
            ("/files/pricing.xlsx", ContentKind.XLSX),
            ("/files/octet/widget-spec.pdf", ContentKind.PDF),  # octet-stream + .pdf
            ("/files/untyped/widget-spec.pdf", ContentKind.PDF),  # no content-type + .pdf
        ],
    )
    async def test_documents_keep_raw_bytes(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite, path: str, kind: ContentKind
    ) -> None:
        result = await _fetch(fetcher, fixture_site, path)

        assert result.content_kind is kind
        assert result.text is None and result.encoding is None
        assert len(result.body) > 1000

    async def test_unsupported_type_is_rejected_without_downloading(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        with pytest.raises(UnsupportedContentTypeError) as exc:
            await _fetch(fetcher, fixture_site, "/api/widgets")

        assert exc.value.reason is PageReason.UNSUPPORTED_CONTENT_TYPE
        assert exc.value.content_type == "application/json"

    async def test_declared_oversize_fails_fast(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        start = time.perf_counter()

        with pytest.raises(TooLargeError) as exc:
            await _fetch(fetcher, fixture_site, "/files/huge-catalog.pdf")

        assert exc.value.detail == "content_length"
        assert time.perf_counter() - start < 1  # 25 MiB never downloaded

    async def test_streamed_oversize_is_cut_off(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        with pytest.raises(TooLargeError) as exc:
            await _fetch(fetcher, fixture_site, "/large.html")

        assert exc.value.detail == "streamed"
        assert exc.value.reason is PageReason.TOO_LARGE

    async def test_gzip_is_decoded(self, fetcher: HttpFetcher, fixture_site: FixtureSite) -> None:
        result = await _fetch(fetcher, fixture_site, "/gzip-page")

        assert result.text is not None and "Compressed page" in result.text

    async def test_gzip_bomb_is_capped_on_decoded_size(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        with pytest.raises(TooLargeError):
            await _fetch(fetcher, fixture_site, "/gzip-bomb")

    async def test_charset_from_meta(self, fetcher: HttpFetcher, fixture_site: FixtureSite) -> None:
        result = await _fetch(fetcher, fixture_site, "/latin1.html")

        assert result.encoding == "iso8859-1"
        assert result.text is not None and "Crème brûlée" in result.text

    async def test_charset_from_bom(self, fetcher: HttpFetcher, fixture_site: FixtureSite) -> None:
        result = await _fetch(fetcher, fixture_site, "/bom-page")

        assert result.encoding == "utf-8"
        assert result.text is not None and "naïve façade" in result.text
        assert not result.text.startswith("﻿")

    @pytest.mark.parametrize(
        ("path", "status", "reason"),
        [
            ("/missing.html", 404, PageReason.HTTP_CLIENT_ERROR),
            ("/error", 500, PageReason.HTTP_SERVER_ERROR),
            ("/rate-limited", 429, PageReason.RATE_LIMITED),
        ],
    )
    async def test_http_errors(
        self,
        fetcher: HttpFetcher,
        fixture_site: FixtureSite,
        path: str,
        status: int,
        reason: PageReason,
    ) -> None:
        with pytest.raises(HttpStatusError) as exc:
            await _fetch(fetcher, fixture_site, path)

        assert (exc.value.status_code, exc.value.reason) == (status, reason)
        assert exc.value.retry_after == (2.0 if status == 429 else None)

    async def test_read_timeout(self, fetcher: HttpFetcher, fixture_site: FixtureSite) -> None:
        with pytest.raises(FetchTimeoutError) as exc:
            await _fetch(fetcher, fixture_site, "/slow?delay=5")

        assert exc.value.reason is PageReason.TIMEOUT

    async def test_total_timeout_stops_a_dripping_server(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        start = time.perf_counter()

        with pytest.raises(FetchTimeoutError) as exc:
            await _fetch(fetcher, fixture_site, "/drip")

        assert exc.value.detail == "total_timeout"
        assert 2.5 < time.perf_counter() - start < 5

    @pytest.mark.parametrize(
        ("path", "error", "reason"),
        [
            ("/loop-a", RedirectLoopFetchError, PageReason.REDIRECT_LOOP),
            ("/redirect-metadata", SsrfFetchError, PageReason.SSRF_BLOCKED),
        ],
    )
    async def test_redirect_failures(
        self,
        fetcher: HttpFetcher,
        fixture_site: FixtureSite,
        path: str,
        error: type[Exception],
        reason: PageReason,
    ) -> None:
        with pytest.raises(error) as exc:
            await _fetch(fetcher, fixture_site, path)

        assert getattr(exc.value, "reason", None) is reason

    async def test_too_many_redirects(
        self, override_settings: OverrideSettings, fixture_site: FixtureSite
    ) -> None:
        config = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True, CRAWL_MAX_REDIRECTS=1)
        async with SafeHttpClient(config) as client:
            with pytest.raises(TooManyRedirectsFetchError) as exc:
                await HttpFetcher(client, config).fetch(fixture_site.base_url + "/redirect-chain")

        assert exc.value.reason is PageReason.TOO_MANY_REDIRECTS

    async def test_connection_refused_is_a_network_error(self, fetcher: HttpFetcher) -> None:
        with pytest.raises(NetworkFetchError) as exc:
            await fetcher.fetch("http://127.0.0.1:1/")

        assert exc.value.reason is PageReason.NETWORK_ERROR

    async def test_dns_failure_is_a_network_error(self, fetcher: HttpFetcher) -> None:
        with pytest.raises(NetworkFetchError) as exc:
            await fetcher.fetch("http://does-not-exist.invalid/")

        assert exc.value.detail == "dns"

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("text/html; charset=UTF-8", ("text/html", "utf-8")),
            ('Text/HTML; Charset="iso-8859-1"', ("text/html", "iso-8859-1")),
            ("application/pdf", ("application/pdf", None)),
            (None, ("", None)),
        ],
    )
    def test_parse_content_type(self, header: str | None, expected: tuple) -> None:
        assert parse_content_type(header) == expected

    @pytest.mark.parametrize(
        ("mime", "url", "kind"),
        [
            ("application/xhtml+xml", "https://a.example/x", ContentKind.HTML),
            ("application/octet-stream", "https://a.example/a.DOCX", ContentKind.DOCX),
            ("application/octet-stream", "https://a.example/a.bin", None),
            ("", "https://a.example/page.html", ContentKind.HTML),
            ("text/plain", "https://a.example/a.pdf", None),  # an explicit wrong type wins
            ("image/png", "https://a.example/a.html", None),
        ],
    )
    def test_kind_for(self, mime: str, url: str, kind: ContentKind | None) -> None:
        assert kind_for(mime, url) is kind

    @pytest.mark.parametrize(
        ("body", "header", "encoding"),
        [
            (b"<p>caf\xc3\xa9</p>", "utf-8", "utf-8"),
            (b'<meta charset="windows-1252"><p>caf\xe9</p>', None, "cp1252"),
            (b"<p>caf\xe9</p>", "bogus-charset", None),  # unknown header → detection
            ("<p>привет мир, это тест кодировки</p>".encode("cp1251"), None, None),
        ],
    )
    def test_decode_html(self, body: bytes, header: str | None, encoding: str | None) -> None:
        text, used = decode_html(body, header)

        assert "�" not in text or encoding is None
        if encoding is not None:
            assert used == encoding
        assert text.startswith("<")

    def test_retry_after_parsing(self) -> None:
        assert parse_retry_after("120") == 120.0
        assert parse_retry_after(None) is None
        assert parse_retry_after("soon") is None
        assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past


class TestFetchRaw_S29:
    """S29 — `HttpFetcher.fetch_raw`: any 2xx body (robots.txt, sitemaps), capped or truncated."""

    async def test_plain_text_is_returned(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        raw = await fetcher.fetch_raw(
            fixture_site.url("/robots.txt"), accept="text/plain", max_bytes=10_000
        )

        assert raw.status_code == 200
        assert raw.mime_type == "text/plain"
        assert b"Disallow: /private/" in raw.body
        assert raw.truncated is False
        assert raw.bytes_downloaded == len(raw.body)

    async def test_no_content_type_allowlist(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        # `fetch()` skips JSON without downloading it; `fetch_raw` returns it
        raw = await fetcher.fetch_raw(
            fixture_site.url("/api/widgets"), accept="*/*", max_bytes=10_000
        )

        assert raw.mime_type == "application/json"
        assert raw.body

    async def test_truncate_cuts_at_max_bytes(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        raw = await fetcher.fetch_raw(
            fixture_site.url("/large.html"), accept="*/*", max_bytes=1000, truncate=True
        )

        assert raw.truncated is True
        assert len(raw.body) == raw.bytes_downloaded == 1000

    async def test_over_max_bytes_fails_without_truncate(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        with pytest.raises(TooLargeError):
            await fetcher.fetch_raw(fixture_site.url("/large.html"), accept="*/*", max_bytes=1000)

    async def test_gzip_bomb_is_truncated_on_decoded_bytes(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        raw = await fetcher.fetch_raw(
            fixture_site.url("/gzip-bomb"), accept="*/*", max_bytes=4096, truncate=True
        )

        assert raw.truncated is True
        assert len(raw.body) == 4096

    @pytest.mark.parametrize(
        ("path", "error"),
        [
            ("/missing.html", HttpStatusError),
            ("/loop-a", RedirectLoopFetchError),
            ("/redirect-metadata", SsrfFetchError),
        ],
    )
    async def test_errors_match_fetch(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite, path: str, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            await fetcher.fetch_raw(fixture_site.url(path), accept="*/*", max_bytes=10_000)

    async def test_total_timeout_applies(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        start = time.perf_counter()

        with pytest.raises(FetchTimeoutError) as exc:
            await fetcher.fetch_raw(fixture_site.url("/drip"), accept="*/*", max_bytes=10_000_000)

        assert exc.value.detail == "total_timeout"
        assert time.perf_counter() - start < 5
