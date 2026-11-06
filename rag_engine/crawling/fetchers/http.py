"""Plain HTTP fetcher: one URL in, a `FetchResult` or a typed `FetchError` out."""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import httpx

from rag_engine.config import Settings, settings
from rag_engine.crawling.fetchers.content import (
    ACCEPT_HEADER,
    decode_html,
    kind_for,
    parse_content_type,
)
from rag_engine.crawling.fetchers.errors import (
    FetchError,
    FetchTimeoutError,
    HttpStatusError,
    NetworkFetchError,
    RedirectLoopFetchError,
    SsrfFetchError,
    TooLargeError,
    TooManyRedirectsFetchError,
    UnsupportedContentTypeError,
)
from rag_engine.domain import ContentKind, RenderedWith, utc_now
from rag_engine.security import (
    HostResolutionError,
    RedirectLoopError,
    SafeHttpClient,
    SsrfBlockedError,
    TooManyRedirectsError,
)

# response headers later steps care about (canonical links, robots directives, language)
_KEPT_HEADERS = ("content-language", "link", "x-robots-tag", "last-modified", "etag")


@dataclass(frozen=True, slots=True)
class FetchResult:
    requested_url: str
    final_url: str
    redirect_chain: tuple[str, ...]
    status_code: int
    mime_type: str
    content_kind: ContentKind
    body: bytes = field(repr=False)
    text: str | None = field(repr=False)  # decoded HTML; None for binary documents
    encoding: str | None
    bytes_downloaded: int
    elapsed_ms: int
    headers: dict[str, str]
    rendered_with: RenderedWith = RenderedWith.HTTP


def parse_retry_after(value: str | None) -> float | None:
    """`Retry-After` as seconds: either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    return max((when - utc_now()).total_seconds(), 0.0)


@dataclass(frozen=True, slots=True)
class RawResponse:
    """A 2xx body fetched without the page content-type allowlist (`HttpFetcher.fetch_raw`)."""

    requested_url: str
    final_url: str
    redirect_chain: tuple[str, ...]
    status_code: int
    mime_type: str
    charset: str | None
    body: bytes = field(repr=False)
    truncated: bool  # cut at max_bytes (only with truncate=True)
    bytes_downloaded: int
    elapsed_ms: int


def _raise_for_status(response: httpx.Response, url: str) -> None:
    if not 200 <= response.status_code < 300:
        raise HttpStatusError(
            url=url,
            status_code=response.status_code,
            retry_after=parse_retry_after(response.headers.get("retry-after")),
        )


class HttpFetcher:
    def __init__(self, client: SafeHttpClient, config: Settings = settings) -> None:
        self._client = client
        self._max_bytes = config.CRAWL_MAX_PAGE_BYTES
        self._total_timeout = config.CRAWL_FETCH_TIMEOUT_SECONDS

    async def fetch(self, url: str) -> FetchResult:
        """A crawlable page: 2xx, allowlisted content type, at most `CRAWL_MAX_PAGE_BYTES`."""
        start = time.perf_counter()
        async with self._fetch_errors(url):
            return await self._fetch(url, start)

    async def fetch_raw(
        self, url: str, *, accept: str, max_bytes: int, truncate: bool = False
    ) -> RawResponse:
        """Any 2xx body with no content-type allowlist (robots.txt, sitemaps).

        Same SSRF, redirect, timeout and error semantics as `fetch()`. Bodies over
        `max_bytes` raise `TooLargeError`, or are cut at `max_bytes` with `truncate=True`.
        """
        start = time.perf_counter()
        async with (
            self._fetch_errors(url),
            self._client.stream_get(url, headers={"Accept": accept}) as result,
        ):
            response = result.response
            _raise_for_status(response, result.url)
            body, received, truncated = await self._read_capped(
                response, result.url, max_bytes, truncate=truncate
            )
            mime, charset = parse_content_type(response.headers.get("content-type"))
        return RawResponse(
            requested_url=url,
            final_url=result.url,
            redirect_chain=tuple(result.chain),
            status_code=response.status_code,
            mime_type=mime,
            charset=charset,
            body=body,
            truncated=truncated,
            bytes_downloaded=received,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
        )

    @asynccontextmanager
    async def _fetch_errors(self, url: str) -> AsyncGenerator[None]:
        """Total wall-clock cap, and lower-level failures translated to typed `FetchError`s."""
        try:
            async with asyncio.timeout(self._total_timeout):
                yield
        except FetchError:
            raise
        except TimeoutError as exc:
            raise FetchTimeoutError(
                f"fetch exceeded {self._total_timeout}s", url=url, detail="total_timeout"
            ) from exc
        except httpx.TimeoutException as exc:
            raise FetchTimeoutError(
                str(exc) or "timed out", url=url, detail=type(exc).__name__
            ) from exc
        except SsrfBlockedError as exc:
            raise SsrfFetchError(exc.message, url=url, detail=exc.rejection.value) from exc
        except RedirectLoopError as exc:
            raise RedirectLoopFetchError(str(exc), url=url, detail=" -> ".join(exc.chain)) from exc
        except TooManyRedirectsError as exc:
            raise TooManyRedirectsFetchError(str(exc), url=url) from exc
        except HostResolutionError as exc:
            raise NetworkFetchError(str(exc), url=url, detail="dns") from exc
        except (httpx.TransportError, httpx.DecodingError) as exc:
            raise NetworkFetchError(
                str(exc) or type(exc).__name__, url=url, detail=type(exc).__name__
            ) from exc

    @staticmethod
    async def _read_capped(
        response: httpx.Response, url: str, max_bytes: int, *, truncate: bool
    ) -> tuple[bytes, int, bool]:
        """(body, decoded bytes received, truncated). Capped on decoded bytes: stops gzip bombs."""
        chunks: list[bytes] = []
        received = 0
        async for chunk in response.aiter_bytes():
            if received + len(chunk) > max_bytes:
                if not truncate:
                    raise TooLargeError(
                        f"body exceeds {max_bytes} bytes", url=url, detail="streamed"
                    )
                chunks.append(chunk[: max_bytes - received])
                return b"".join(chunks), max_bytes, True
            received += len(chunk)
            chunks.append(chunk)
        return b"".join(chunks), received, False

    async def _fetch(self, url: str, start: float) -> FetchResult:
        async with self._client.stream_get(url, headers={"Accept": ACCEPT_HEADER}) as result:
            response = result.response
            final_url = result.url
            _raise_for_status(response, final_url)

            mime, charset = parse_content_type(response.headers.get("content-type"))
            kind = kind_for(mime, final_url)
            if kind is None:  # decided from headers: the body is never downloaded
                raise UnsupportedContentTypeError(
                    f"unsupported content type {mime or '(none)'}", url=final_url, content_type=mime
                )

            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self._max_bytes:
                raise TooLargeError(
                    f"declared size {declared} exceeds {self._max_bytes}",
                    url=final_url,
                    detail="content_length",
                )

            body, received, _ = await self._read_capped(
                response, final_url, self._max_bytes, truncate=False
            )

        text, encoding = (None, None)
        if kind is ContentKind.HTML:
            text, encoding = decode_html(body, charset)
        return FetchResult(
            requested_url=url,
            final_url=final_url,
            redirect_chain=tuple(result.chain),
            status_code=response.status_code,
            mime_type=mime,
            content_kind=kind,
            body=body,
            text=text,
            encoding=encoding,
            bytes_downloaded=received,
            elapsed_ms=int((time.perf_counter() - start) * 1000),
            headers={h: response.headers[h] for h in _KEPT_HEADERS if h in response.headers},
        )
