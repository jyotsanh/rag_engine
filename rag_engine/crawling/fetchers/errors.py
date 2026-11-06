"""Typed fetch failures. Each carries the `PageReason` it is recorded as.

Step 28 decides which are retryable; these classes only describe what happened.
"""

from rag_engine.domain import PageReason


class FetchError(Exception):
    reason: PageReason = PageReason.NETWORK_ERROR

    def __init__(self, message: str, *, url: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.detail = detail or message


class NetworkFetchError(FetchError):
    reason = PageReason.NETWORK_ERROR


class FetchTimeoutError(FetchError):
    reason = PageReason.TIMEOUT


class SsrfFetchError(FetchError):
    reason = PageReason.SSRF_BLOCKED


class RedirectLoopFetchError(FetchError):
    reason = PageReason.REDIRECT_LOOP


class TooManyRedirectsFetchError(FetchError):
    reason = PageReason.TOO_MANY_REDIRECTS


class TooLargeError(FetchError):
    reason = PageReason.TOO_LARGE


class UnsupportedContentTypeError(FetchError):
    """Not an error of the site: the page is *skipped*, not failed."""

    reason = PageReason.UNSUPPORTED_CONTENT_TYPE

    def __init__(self, message: str, *, url: str, content_type: str | None) -> None:
        super().__init__(message, url=url, detail=content_type or "missing content-type")
        self.content_type = content_type


class HttpStatusError(FetchError):
    """Non-2xx final response."""

    def __init__(self, *, url: str, status_code: int, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status_code}", url=url, detail=f"http_{status_code}")
        self.status_code = status_code
        self.retry_after = retry_after

    @property
    def reason(self) -> PageReason:  # type: ignore
        if self.status_code == 429:
            return PageReason.RATE_LIMITED
        if self.status_code >= 500:
            return PageReason.HTTP_SERVER_ERROR
        return PageReason.HTTP_CLIENT_ERROR
