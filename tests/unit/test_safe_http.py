"""SSRF-safe HTTP client: IP pinning, Host/SNI preservation, redirect vetting."""

import json
import socket
import ssl
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import trustme
import uvicorn
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from rag_engine.config import Settings
from rag_engine.security import (
    PinnedNetworkBackend,
    RedirectLoopError,
    SafeHttpClient,
    SsrfBlockedError,
    SsrfGuard,
    SsrfRejection,
    TooManyRedirectsError,
    UnvettedHostError,
)
from tests.conftest import OverrideSettings
from tests.support.fixture_site import FixtureSite


class _StubResolver:
    """Fake DNS: maps hostnames to fixed answers and counts lookups."""

    def __init__(self, records: dict[str, list[str]]) -> None:
        self.records = records
        self.calls: list[str] = []

    async def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append(host)
        if host not in self.records:
            raise OSError("NXDOMAIN")
        return self.records[host]


def _client(
    config: Settings, records: dict[str, list[str]], ssl_context: ssl.SSLContext | None = None
) -> tuple[SafeHttpClient, _StubResolver]:
    resolver = _StubResolver(records)
    guard = SsrfGuard(config, resolver=resolver)
    return SafeHttpClient(config, guard=guard, ssl_context=ssl_context), resolver


@pytest.fixture
def config(override_settings: OverrideSettings) -> Settings:
    # the fixture site lives on loopback, so private networks must be allowed here
    return override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)


# ---- tiny HTTPS server with a trustme certificate ---- #


class _TlsServer:
    def __init__(self, cert_dir: Path, ca: trustme.CA, hostname: str) -> None:
        cert = ca.issue_cert(hostname)
        cert_file, key_file = cert_dir / "cert.pem", cert_dir / "key.pem"
        cert.cert_chain_pems[0].write_to_path(str(cert_file))
        for extra in cert.cert_chain_pems[1:]:
            extra.write_to_path(str(cert_file), append=True)
        cert.private_key_pem.write_to_path(str(key_file))

        async def hello(request):
            return PlainTextResponse(f"host={request.headers['host']}")

        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(
                Starlette(routes=[Route("/", hello)]),
                ssl_certfile=str(cert_file),
                ssl_keyfile=str(key_file),
                log_config=None,
                lifespan="off",
            )
        )
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True
        )

    def __enter__(self) -> _TlsServer:
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline, "TLS server did not start"
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self.sock.close()


@pytest.fixture
def tls_ca() -> trustme.CA:
    return trustme.CA()


@pytest.fixture
def tls_server(tmp_path: Path, tls_ca: trustme.CA) -> Iterator[_TlsServer]:
    with _TlsServer(tmp_path, tls_ca, "secure.test") as server:
        yield server


# ---- #
# Tests
# ---- #


class TestSafeHttpClient_S26:
    """S26 — SSRF-safe HTTP client: pinned IPs, preserved Host/SNI, vetted redirect hops."""

    async def test_connects_to_pinned_ip_and_keeps_host_header(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        # "acme.test" is not resolvable by the OS: success proves we used the pinned IP
        client, resolver = _client(config, {"acme.test": ["127.0.0.1"]})
        url = f"http://acme.test:{fixture_site.port}/about.html"

        async with client, client.stream_get(url) as result:
            body = await result.response.aread()

        assert result.response.status_code == 200
        assert b"About Acme Widgets" in body
        assert fixture_site.requests[-1].host == f"acme.test:{fixture_site.port}"
        assert fixture_site.requests[-1].user_agent == config.CRAWL_USER_AGENT
        assert resolver.calls == ["acme.test"]  # resolved exactly once, by the guard

    async def test_follows_redirects_and_reports_chain(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        client, _ = _client(config, {})
        start = f"{fixture_site.base_url}/redirect-chain"

        async with client, client.stream_get(start) as result:
            await result.response.aread()

        assert result.response.status_code == 200
        assert result.chain == [
            start,
            f"{fixture_site.base_url}/redirect-me",
            f"{fixture_site.base_url}/docs/getting-started.html",
        ]
        assert result.url.endswith("/docs/getting-started.html")

    async def test_redirect_loop_is_detected(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        client, _ = _client(config, {})

        with pytest.raises(RedirectLoopError) as exc:
            async with client, client.stream_get(f"{fixture_site.base_url}/loop-a"):
                pass

        assert [u.rsplit("/", 1)[-1] for u in exc.value.chain] == ["loop-a", "loop-b", "loop-a"]

    async def test_redirect_cap(
        self, override_settings: OverrideSettings, fixture_site: FixtureSite
    ) -> None:
        config = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True, CRAWL_MAX_REDIRECTS=1)
        client, _ = _client(config, {})

        with pytest.raises(TooManyRedirectsError):
            async with client, client.stream_get(f"{fixture_site.base_url}/redirect-chain"):
                pass

    async def test_redirect_to_metadata_ip_is_blocked_before_connecting(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        client, _ = _client(config, {})

        with pytest.raises(SsrfBlockedError) as exc:
            async with client, client.stream_get(f"{fixture_site.base_url}/redirect-metadata"):
                pass

        assert exc.value.rejection is SsrfRejection.METADATA_ADDRESS
        assert [r.path for r in fixture_site.requests] == ["/redirect-metadata"]

    async def test_redirect_to_host_resolving_internal_is_blocked(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        client, resolver = _client(config, {"evil.test": ["169.254.169.254"]})
        url = f"{fixture_site.base_url}/redirect-to?url=http://evil.test:{fixture_site.port}/"

        with pytest.raises(SsrfBlockedError):
            async with client, client.stream_get(url):
                pass

        assert resolver.calls == ["evil.test"]
        assert len(fixture_site.requests) == 1  # nothing connected to the second hop

    async def test_redirect_to_disallowed_scheme_is_blocked(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        client, _ = _client(config, {})
        url = f"{fixture_site.base_url}/redirect-to?url=file:///etc/passwd"

        with pytest.raises(SsrfBlockedError) as exc:
            async with client, client.stream_get(url):
                pass

        assert exc.value.rejection is SsrfRejection.UNSUPPORTED_SCHEME

    async def test_strict_guard_blocks_the_first_hop(
        self, override_settings: OverrideSettings, fixture_site: FixtureSite
    ) -> None:
        strict = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=False)
        client, _ = _client(strict, {})

        with pytest.raises(SsrfBlockedError):
            async with client, client.stream_get(fixture_site.base_url + "/"):
                pass

        assert fixture_site.requests == []

    async def test_backend_refuses_unvetted_hosts(self) -> None:
        backend = PinnedNetworkBackend()

        with pytest.raises(UnvettedHostError):
            await backend.connect_tcp("never-vetted.test", 80)
        with pytest.raises(UnvettedHostError):
            await backend.connect_unix_socket("/var/run/docker.sock")

    def test_pin_table_is_bounded(self) -> None:
        import ipaddress

        from rag_engine.security import VettedTarget

        backend = PinnedNetworkBackend()
        backend.MAX_PINS = 3
        for i in range(5):
            backend.pin(
                VettedTarget("http", f"h{i}.test", 80, (ipaddress.ip_address("127.0.0.1"),))
            )

        assert list(backend._pins) == [("h2.test", 80), ("h3.test", 80), ("h4.test", 80)]

    async def test_falls_back_to_next_pinned_address(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        # 127.0.0.2 is loopback too but nothing listens there for this port
        client, _ = _client(config, {"multi.test": ["127.0.0.2", "127.0.0.1"]})

        async with client, client.stream_get(f"http://multi.test:{fixture_site.port}/") as result:
            await result.response.aread()

        assert result.response.status_code == 200

    async def test_tls_uses_hostname_for_sni_and_verification(
        self, config: Settings, tls_server: _TlsServer, tls_ca: trustme.CA
    ) -> None:
        context = ssl.create_default_context()
        tls_ca.configure_trust(context)
        client, _ = _client(config, {"secure.test": ["127.0.0.1"]}, ssl_context=context)

        async with client, client.stream_get(f"https://secure.test:{tls_server.port}/") as result:
            body = (await result.response.aread()).decode()

        assert body == f"host=secure.test:{tls_server.port}"

    async def test_tls_rejects_certificate_for_another_name(
        self, config: Settings, tls_server: _TlsServer, tls_ca: trustme.CA
    ) -> None:
        context = ssl.create_default_context()
        tls_ca.configure_trust(context)
        # pinned to the same server, but the client expects a certificate for other.test
        client, _ = _client(config, {"other.test": ["127.0.0.1"]}, ssl_context=context)

        with pytest.raises(httpx.NetworkError):
            async with client, client.stream_get(f"https://other.test:{tls_server.port}/"):
                pass

    async def test_tls_rejects_untrusted_certificate(
        self, config: Settings, tls_server: _TlsServer
    ) -> None:
        client, _ = _client(config, {"secure.test": ["127.0.0.1"]})  # system CAs only

        with pytest.raises(httpx.NetworkError):
            async with client, client.stream_get(f"https://secure.test:{tls_server.port}/"):
                pass


class TestSendOnce_S35:
    """S35 — `SafeHttpClient.send_once`: one vetted, pinned hop that returns redirects."""

    async def test_redirect_is_returned_not_followed(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        async with (
            SafeHttpClient(config) as client,
            client.send_once("GET", fixture_site.url("/redirect-me")) as response,
        ):
            assert response.status_code == 301
            assert response.headers["location"].endswith("/docs/getting-started.html")

        assert fixture_site.hits("/docs/getting-started.html") == 0

    async def test_method_and_body_are_sent(
        self, config: Settings, fixture_site: FixtureSite
    ) -> None:
        async with (
            SafeHttpClient(config) as client,
            client.send_once(
                "POST", fixture_site.url("/browser/echo"), content=b"hello world"
            ) as response,
        ):
            echoed = json.loads(await response.aread())

        assert echoed == {"method": "POST", "cookie": None, "body_bytes": 11}

    async def test_hop_is_vetted_before_connecting(self, config: Settings) -> None:
        client, resolver = _client(config, {"evil.test": ["169.254.169.254"]})

        async with client:
            with pytest.raises(SsrfBlockedError):
                async with client.send_once("GET", "http://evil.test/"):
                    pass

        assert resolver.calls == ["evil.test"]
