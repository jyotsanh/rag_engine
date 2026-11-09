"""HTTP client for requests made on a caller's behalf, hardened against SSRF.

- Every hop (the first request and each redirect) is vetted by `SsrfGuard`.
- Connections go only to the vetted IPs: a custom httpcore network backend maps
  the hostname to its pinned addresses at connect time, so a DNS answer that
  changes after vetting (DNS rebinding) is never used. The hostname is still
  used for the `Host` header, TLS SNI and certificate verification.
- The backend refuses any host that was not vetted, so pinning can't be bypassed.
- Redirects are followed manually: re-vetted, loop-detected, capped.
- No cookie jar, no auth, no environment proxies (`httpx.AsyncClient` would add
  all three; cookies would leak between tenants' crawls).
"""

import contextlib
import ssl
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, cast

import httpcore
import httpx

from rag_engine.config import Settings, logger, settings
from rag_engine.security.ssrf import SsrfGuard, VettedTarget

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class RedirectError(Exception):
    def __init__(self, message: str, chain: list[str]) -> None:
        super().__init__(message)
        self.chain = chain


class RedirectLoopError(RedirectError):
    pass


class TooManyRedirectsError(RedirectError):
    pass


class UnvettedHostError(httpcore.ConnectError):
    """A connection was attempted to a host the guard never approved."""


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connects `(host, port)` only to the IPs registered by `pin()`.

    Pins live in a bounded LRU: every request re-vets and re-pins before
    connecting, so an evicted pin is simply recreated on next use.
    """

    MAX_PINS = 10_000

    def __init__(self, inner: httpcore.AsyncNetworkBackend | None = None) -> None:
        # cast: httpcore exports a placeholder name when anyio is missing; it is installed here
        self._inner = cast(httpcore.AsyncNetworkBackend, inner or httpcore.AnyIOBackend())
        self._pins: OrderedDict[tuple[str, int], tuple[str, ...]] = OrderedDict()

    def pin(self, target: VettedTarget) -> None:
        key = (target.host, target.port)
        self._pins[key] = tuple(str(a) for a in target.addresses)
        self._pins.move_to_end(key)
        while len(self._pins) > self.MAX_PINS:
            self._pins.popitem(last=False)

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = self._pins.get((host.lower().rstrip("."), port))
        if not addresses:
            raise UnvettedHostError(f"refusing to connect to unvetted host {host}:{port}")
        last_error: Exception | None = None
        for address in addresses:  # vetted order; first that answers wins
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def connect_unix_socket(self, *args: Any, **kwargs: Any) -> httpcore.AsyncNetworkStream:
        raise UnvettedHostError("unix sockets are never allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class _PinnedTransport(httpx.AsyncBaseTransport):
    """Minimal httpx transport over an httpcore pool with the pinned backend."""

    def __init__(self, pool: httpcore.AsyncConnectionPool) -> None:
        self._pool = pool

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.AsyncByteStream)
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        try:
            core_response = await self._pool.handle_async_request(core_request)
        except httpcore.TimeoutException as exc:
            raise httpx.TimeoutException(str(exc), request=request) from exc
        except httpcore.NetworkError as exc:
            raise httpx.NetworkError(str(exc), request=request) from exc
        except httpcore.ProtocolError as exc:
            raise httpx.ProtocolError(str(exc), request=request) from exc

        return httpx.Response(
            status_code=core_response.status,
            headers=core_response.headers,
            stream=_ResponseStream(core_response.stream),
            extensions=core_response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        if hasattr(self._stream, "aclose"):
            await self._stream.aclose()


@dataclass
class SafeResponse:
    """The final (non-redirect) response, unread; `chain` lists every URL visited."""

    response: httpx.Response
    chain: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return self.chain[-1]


class SafeHttpClient:
    def __init__(
        self,
        config: Settings = settings,
        guard: SsrfGuard | None = None,
        ssl_context: ssl.SSLContext | None = None,
        max_connections: int | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._config = config
        self._guard = guard or SsrfGuard(config)
        self._backend = PinnedNetworkBackend()
        # the crawler's timeouts by default; webhook delivery passes its own (Step 83)
        self._timeout = timeout or httpx.Timeout(
            connect=config.CRAWL_CONNECT_TIMEOUT_SECONDS,
            read=config.CRAWL_READ_TIMEOUT_SECONDS,
            write=config.CRAWL_READ_TIMEOUT_SECONDS,
            pool=config.CRAWL_CONNECT_TIMEOUT_SECONDS,
        )
        pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context or httpx.create_ssl_context(),
            max_connections=max_connections or max(config.CRAWL_CONCURRENCY * 2, 10),
            keepalive_expiry=30.0,
            http1=True,
            http2=False,
            network_backend=self._backend,
        )
        self._transport = _PinnedTransport(pool)
        self._user_agent = config.CRAWL_USER_AGENT

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> SafeHttpClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _request(
        self,
        url: str,
        headers: dict[str, str] | None,
        method: str = "GET",
        content: bytes | None = None,
    ) -> httpx.Request:
        merged = {"User-Agent": self._user_agent}
        merged |= headers or {}
        # only encodings httpx can decode here; a browser would advertise br/zstd too
        merged["Accept-Encoding"] = "gzip, deflate"
        return httpx.Request(
            method,
            url,
            headers=merged,
            content=content,
            extensions={"timeout": self._timeout.as_dict()},
        )

    @contextlib.asynccontextmanager
    async def send_once(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> AsyncGenerator[httpx.Response]:
        """One vetted, pinned request; redirects are **returned, not followed**.

        For callers that handle redirects themselves (the headless browser re-requests
        every hop, and each hop comes back through here to be vetted). Body unread.
        Raises `SsrfBlockedError` / `HostResolutionError` and httpx transport errors.
        """
        target = await self._guard.vet(url)
        self._backend.pin(target)
        response = await self._transport.handle_async_request(
            self._request(url, headers, method=method, content=content)
        )
        try:
            yield response
        finally:
            await response.aclose()

    @contextlib.asynccontextmanager
    async def stream_get(
        self, url: str, headers: dict[str, str] | None = None
    ) -> AsyncGenerator[SafeResponse]:
        """GET `url`, following redirects safely. The final response body is unread.

        Raises `SsrfBlockedError` / `HostResolutionError` for a bad hop,
        `RedirectLoopError` / `TooManyRedirectsError`, and httpx transport errors.
        """
        chain: list[str] = []
        current = url
        response: httpx.Response | None = None
        try:
            for _hop in range(self._config.CRAWL_MAX_REDIRECTS + 1):
                if current in chain:
                    raise RedirectLoopError(f"redirect loop at {current}", [*chain, current])
                chain.append(current)
                target = await self._guard.vet(current)
                self._backend.pin(target)
                response = await self._transport.handle_async_request(
                    self._request(current, headers)
                )
                location = response.headers.get("location")
                if response.status_code not in REDIRECT_STATUSES or not location:
                    yield SafeResponse(response=response, chain=chain)
                    return
                await response.aclose()
                response = None
                current = str(httpx.URL(current).join(location))
                logger.bind(url=chain[-1], location=current).debug("following redirect")
            raise TooManyRedirectsError(
                f"more than {self._config.CRAWL_MAX_REDIRECTS} redirects", chain
            )
        finally:
            if response is not None:
                await response.aclose()
