"""Plain HTTP fetcher: one URL in, a `FetchResult` or a typed `FetchError` out."""

import asyncio
import time
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


class HttpFetcher:
    def __init__(self, client: SafeHttpClient, config: Settings = settings) -> None:
        self._client = client
        self._max_bytes = config.CRAWL_MAX_PAGE_BYTES
        self._total_timeout = config.CRAWL_FETCH_TIMEOUT_SECONDS

    async def fetch(self, url: str) -> FetchResult:
        start = time.perf_counter()
        try:
            async with asyncio.timeout(self._total_timeout):
                return await self._fetch(url, start)
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

    async def _fetch(self, url: str, start: float) -> FetchResult:
        async with self._client.stream_get(url, headers={"Accept": ACCEPT_HEADER}) as result:
            response = result.response
            final_url = result.url
            if not 200 <= response.status_code < 300:
                raise HttpStatusError(
                    url=final_url,
                    status_code=response.status_code,
                    retry_after=parse_retry_after(response.headers.get("retry-after")),
                )

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

            chunks: list[bytes] = []
            received = 0
            async for chunk in response.aiter_bytes():  # decoded: also caps gzip bombs
                received += len(chunk)
                if received > self._max_bytes:
                    raise TooLargeError(
                        f"body exceeds {self._max_bytes} bytes", url=final_url, detail="streamed"
                    )
                chunks.append(chunk)
            body = b"".join(chunks)

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
