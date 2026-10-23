"""SSRF guard: address classification matrix, DNS handling, ports, hostnames."""

import asyncio
import ipaddress

import pytest

from rag_engine.config import Settings
from rag_engine.security import (
    HostResolutionError,
    SsrfBlockedError,
    SsrfGuard,
    SsrfRejection,
    classify,
)
from rag_engine.security.ssrf import _system_resolver
from tests.conftest import OverrideSettings
from tests.support.fixture_site import METADATA_URL, FixtureSite

R = SsrfRejection


def _guard(
    override_settings: OverrideSettings,
    records: dict[str, list[str]] | None = None,
    *,
    allow_private: bool = False,
    ports: list[int] | None = None,
) -> SsrfGuard:
    config: Settings = override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=allow_private,
        SSRF_ALLOWED_PORTS=ports or [80, 443],
    )

    async def resolver(host: str, port: int) -> list[str]:
        if records is None or host not in records:
            raise OSError("name or service not known")
        return records[host]

    return SsrfGuard(config, resolver=resolver)


# ---- #
# Tests
# ---- #


class TestSsrfGuard_S25:
    """S25 — SSRF guard: only public unicast targets; metadata always blocked."""

    @pytest.mark.parametrize(
        ("address", "rejection"),
        [
            ("8.8.8.8", None),
            ("1.1.1.1", None),
            ("2606:4700::1111", None),
            ("127.0.0.1", R.LOOPBACK),
            ("127.255.255.254", R.LOOPBACK),
            ("::1", R.LOOPBACK),
            ("10.1.2.3", R.PRIVATE_ADDRESS),
            ("172.16.0.1", R.PRIVATE_ADDRESS),
            ("172.31.255.255", R.PRIVATE_ADDRESS),
            ("192.168.1.1", R.PRIVATE_ADDRESS),
            ("100.64.0.1", R.PRIVATE_ADDRESS),  # CGNAT
            ("fd12:3456::1", R.PRIVATE_ADDRESS),  # ULA
            ("169.254.1.1", R.LINK_LOCAL),
            ("fe80::1", R.LINK_LOCAL),
            ("169.254.169.254", R.METADATA_ADDRESS),
            ("fd00:ec2::254", R.METADATA_ADDRESS),
            ("100.100.100.200", R.METADATA_ADDRESS),
            ("0.0.0.0", R.UNSPECIFIED),
            ("::", R.UNSPECIFIED),
            ("224.0.0.1", R.MULTICAST),
            ("ff02::1", R.MULTICAST),
            ("ff0e::1", R.MULTICAST),  # global-scope multicast reports is_global=True
            ("255.255.255.255", R.RESERVED),
            ("240.0.0.1", R.RESERVED),
            ("192.0.2.10", R.RESERVED),  # TEST-NET-1
            ("198.18.0.1", R.RESERVED),  # benchmarking
            ("2001:db8::1", R.RESERVED),  # documentation
            ("::ffff:127.0.0.1", R.LOOPBACK),  # IPv4-mapped
            ("::ffff:169.254.169.254", R.METADATA_ADDRESS),
            ("::ffff:8.8.8.8", None),
            ("64:ff9b::7f00:1", R.LOOPBACK),  # NAT64 → 127.0.0.1 (is_global=True!)
            ("64:ff9b::a9fe:a9fe", R.METADATA_ADDRESS),  # NAT64 → 169.254.169.254
            ("2002:7f00:1::", R.LOOPBACK),  # 6to4 → 127.0.0.1
            ("2002:0808:0808::", R.RESERVED),  # 6to4 of a public IP: still not plain unicast
            ("2001:0:4136:e378:8000:63bf:3fff:fdd2", R.RESERVED),  # Teredo
        ],
    )
    def test_classification_matrix(self, address: str, rejection: SsrfRejection | None) -> None:
        assert classify(ipaddress.ip_address(address)) == rejection

    @pytest.mark.parametrize(
        ("address", "rejection"),
        [
            ("127.0.0.1", None),
            ("10.0.0.1", None),
            ("100.64.0.1", None),
            ("fe80::1", None),
            ("169.254.169.254", R.METADATA_ADDRESS),
            ("64:ff9b::a9fe:a9fe", R.METADATA_ADDRESS),
            ("0.0.0.0", R.UNSPECIFIED),
            ("224.0.0.1", R.MULTICAST),
            ("240.0.0.1", R.RESERVED),
            ("255.255.255.255", R.RESERVED),
            ("192.0.2.10", R.RESERVED),
            ("2001:db8::1", R.RESERVED),
            ("fd12:3456::1", None),
        ],
    )
    def test_allow_private_still_blocks_dangerous_classes(
        self, address: str, rejection: SsrfRejection | None
    ) -> None:
        assert classify(ipaddress.ip_address(address), allow_private=True) == rejection

    async def test_public_host_passes_with_deduplicated_addresses(
        self, override_settings: OverrideSettings
    ) -> None:
        guard = _guard(
            override_settings,
            {"acme.example": ["93.184.215.14", "93.184.215.14", "2606:2800:21f::1"]},
        )

        target = await guard.vet("https://Acme.Example/docs")

        assert (target.scheme, target.host, target.port) == ("https", "acme.example", 443)
        assert [str(a) for a in target.addresses] == ["93.184.215.14", "2606:2800:21f::1"]

    async def test_one_private_record_blocks_the_host(
        self, override_settings: OverrideSettings
    ) -> None:
        guard = _guard(override_settings, {"rebind.example": ["93.184.215.14", "10.0.0.5"]})

        with pytest.raises(SsrfBlockedError) as exc:
            await guard.vet("https://rebind.example/")

        assert exc.value.rejection is R.PRIVATE_ADDRESS
        assert exc.value.status_code == 422
        assert exc.value.code == "url_not_allowed"
        assert "10.0.0.5" not in str(exc.value.details)  # no DNS oracle

    async def test_ip_literals_skip_dns(self, override_settings: OverrideSettings) -> None:
        guard = _guard(override_settings, records={})  # any DNS call would fail

        assert (await guard.vet("http://8.8.8.8/")).addresses == (ipaddress.ip_address("8.8.8.8"),)
        with pytest.raises(SsrfBlockedError) as exc:
            await guard.vet("http://[::ffff:10.0.0.1]/")
        assert exc.value.rejection is R.PRIVATE_ADDRESS

    async def test_dns_failure_is_not_an_ssrf_verdict(
        self, override_settings: OverrideSettings
    ) -> None:
        guard = _guard(override_settings, records={})

        with pytest.raises(HostResolutionError):
            await guard.vet("https://does-not-exist.example/")

    async def test_dns_timeout(self, override_settings: OverrideSettings) -> None:
        config = override_settings()

        async def slow(host: str, port: int) -> list[str]:
            await asyncio.sleep(5)
            return []

        guard = SsrfGuard(config, resolver=slow, dns_timeout_seconds=0.05)

        with pytest.raises(HostResolutionError, match="timed out"):
            await guard.vet("https://slow.example/")

    async def test_no_records_is_blocked(self, override_settings: OverrideSettings) -> None:
        guard = _guard(override_settings, {"empty.example": []})

        with pytest.raises(SsrfBlockedError) as exc:
            await guard.vet("https://empty.example/")

        assert exc.value.rejection is R.NO_ADDRESSES

    async def test_ipv6_zone_ids_are_stripped(self, override_settings: OverrideSettings) -> None:
        guard = _guard(override_settings, {"zoned.example": ["fe80::1%eth0"]})

        with pytest.raises(SsrfBlockedError) as exc:
            await guard.vet("https://zoned.example/")

        assert exc.value.rejection is R.LINK_LOCAL

    @pytest.mark.parametrize(
        ("url", "rejection"),
        [
            ("ftp://acme.example/", R.UNSUPPORTED_SCHEME),
            ("file:///etc/passwd", R.UNSUPPORTED_SCHEME),
            ("gopher://acme.example/", R.UNSUPPORTED_SCHEME),
            ("https://acme.example:8443/", R.PORT_NOT_ALLOWED),
            ("http://acme.example:22/", R.PORT_NOT_ALLOWED),
            ("http://acme.example:6379/", R.PORT_NOT_ALLOWED),
            ("http://localhost/", R.BLOCKED_HOSTNAME),
            ("http://api.localhost/", R.BLOCKED_HOSTNAME),
            ("http://metadata.google.internal/", R.BLOCKED_HOSTNAME),
            ("http://METADATA.GOOGLE.INTERNAL./", R.BLOCKED_HOSTNAME),
            ("http://instance-data/", R.BLOCKED_HOSTNAME),
        ],
    )
    async def test_static_rejections(
        self, override_settings: OverrideSettings, url: str, rejection: SsrfRejection
    ) -> None:
        guard = _guard(override_settings, {"acme.example": ["93.184.215.14"]})

        with pytest.raises(SsrfBlockedError) as exc:
            await guard.vet(url)

        assert exc.value.rejection is rejection

    async def test_configured_ports_are_allowed(self, override_settings: OverrideSettings) -> None:
        guard = _guard(override_settings, {"acme.example": ["93.184.215.14"]}, ports=[443, 8443])

        assert (await guard.vet("https://acme.example:8443/")).port == 8443

    async def test_allow_private_permits_loopback_and_any_port_but_not_metadata(
        self, override_settings: OverrideSettings
    ) -> None:
        guard = _guard(
            override_settings,
            {"metadata.google.internal": ["169.254.169.254"], "localhost": ["127.0.0.1"]},
            allow_private=True,
        )

        assert (await guard.vet("http://127.0.0.1:54321/")).port == 54321
        assert (await guard.vet("http://localhost:54321/")).host == "localhost"
        with pytest.raises(SsrfBlockedError) as by_ip:
            await guard.vet("http://169.254.169.254/latest/meta-data/")
        with pytest.raises(SsrfBlockedError) as by_name:
            await guard.vet("http://metadata.google.internal/")
        assert by_ip.value.rejection is R.METADATA_ADDRESS
        assert by_name.value.rejection is R.BLOCKED_HOSTNAME

    async def test_system_resolver_resolves_localhost(self) -> None:
        addresses = await _system_resolver("localhost", 80)

        assert {ipaddress.ip_address(a).is_loopback for a in addresses} == {True}

    async def test_fixture_site_reachable_only_with_private_networks_allowed(
        self, override_settings: OverrideSettings, fixture_site: FixtureSite
    ) -> None:
        strict = SsrfGuard(override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=False))
        relaxed = SsrfGuard(override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True))

        with pytest.raises(SsrfBlockedError):
            await strict.vet(fixture_site.base_url + "/")
        assert (await relaxed.vet(fixture_site.base_url + "/")).port == fixture_site.port
        with pytest.raises(SsrfBlockedError) as metadata:
            await relaxed.vet(METADATA_URL)  # the fixture's /redirect-metadata target
        assert metadata.value.rejection is R.METADATA_ADDRESS
