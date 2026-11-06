"""Server-side request forgery guard for every outbound request we make on a
caller's behalf (crawler, robots/sitemaps, webhooks).

A target passes only if:
- the scheme is http/https and the port is allowed (`SSRF_ALLOWED_PORTS`);
- the host is not a known internal/metadata name;
- *every* address it resolves to is globally routable unicast (Python's
  `is_global`, which follows the IANA special-purpose registries), including
  IPv4 addresses embedded in IPv4-mapped, NAT64, 6to4 and Teredo IPv6 forms.
  One bad record rejects the host, which defeats DNS answers mixing public and
  private IPs.

`SSRF_ALLOW_PRIVATE_NETWORKS` (tests/local only, forbidden in prod) additionally
allows private/loopback/link-local/CGNAT addresses and any port. Cloud metadata
endpoints, unspecified, multicast and reserved addresses stay blocked regardless.

The vetted addresses are returned so the HTTP client can connect to exactly
those (Step 26 pins them), closing the DNS-rebinding window between check and use.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

from rag_engine.config import Settings, settings
from rag_engine.exceptions import AppValidationError

type IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
type Resolver = Callable[[str, int], Awaitable[list[str]]]

DEFAULT_PORTS = {"http": 80, "https": 443}

METADATA_ADDRESSES: frozenset[IPAddress] = frozenset(
    ipaddress.ip_address(a)
    for a in (
        "169.254.169.254",  # AWS, GCP, Azure, OpenStack, DigitalOcean, Oracle
        "169.254.170.2",  # AWS ECS task metadata
        "169.254.169.123",  # AWS time sync
        "100.100.100.200",  # Alibaba Cloud
        "192.0.0.192",  # Oracle Cloud (legacy)
        "fd00:ec2::254",  # AWS IMDS over IPv6
        "fd00:ec2::23",  # AWS time sync over IPv6
    )
)
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata",
        "metadata.google.internal",
        "instance-data",
        "instance-data.ec2.internal",
    }
)
# what "private networks" means for SSRF_ALLOW_PRIVATE_NETWORKS. Deliberately explicit:
# `ipaddress.is_private` also covers reserved, broadcast and documentation ranges.
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "fc00::/7")
)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")


class SsrfRejection(StrEnum):
    UNSUPPORTED_SCHEME = "unsupported_scheme"
    PORT_NOT_ALLOWED = "port_not_allowed"
    BLOCKED_HOSTNAME = "blocked_hostname"
    METADATA_ADDRESS = "metadata_address"
    LOOPBACK = "loopback"
    PRIVATE_ADDRESS = "private_address"
    LINK_LOCAL = "link_local"
    MULTICAST = "multicast"
    UNSPECIFIED = "unspecified"
    RESERVED = "reserved"  # documentation, benchmarking, 6to4/Teredo, other special-purpose
    NO_ADDRESSES = "no_addresses"


class SsrfBlockedError(AppValidationError):
    """The target is not a public internet address. 422 when raised for a submitted URL."""

    code = "url_not_allowed"
    default_message = "URL points to a non-public network location"

    def __init__(self, rejection: SsrfRejection, *, host: str | None = None) -> None:
        # never echo resolved IPs to clients: that would turn the guard into a DNS oracle
        super().__init__(details={"reason": rejection.value, "host": host})
        self.rejection = rejection


class HostResolutionError(Exception):
    """DNS lookup failed or timed out (a network error, not an SSRF verdict)."""


@dataclass(frozen=True, slots=True)
class VettedTarget:
    scheme: str
    host: str
    port: int
    addresses: tuple[IPAddress, ...]


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]
    if ip in _NAT64 or ip in _NAT64_LOCAL:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def classify(ip: IPAddress, *, allow_private: bool = False) -> SsrfRejection | None:
    """`None` if the address may be contacted, else why not."""
    if ip in METADATA_ADDRESSES:
        return SsrfRejection.METADATA_ADDRESS
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            inner = classify(embedded, allow_private=allow_private)
            if inner is not None:
                return inner
            # the IPv4 is fine, but 6to4/Teredo/mapped are still not plain global unicast
            return None if (ip.is_global or ip.ipv4_mapped is not None) else SsrfRejection.RESERVED
    if ip.is_unspecified:
        return SsrfRejection.UNSPECIFIED
    if ip.is_multicast:
        return SsrfRejection.MULTICAST
    if ip.is_global:
        return None
    if ip.is_loopback:
        return None if allow_private else SsrfRejection.LOOPBACK
    if ip.is_link_local:
        return None if allow_private else SsrfRejection.LINK_LOCAL
    if any(ip in network for network in PRIVATE_NETWORKS):
        return None if allow_private else SsrfRejection.PRIVATE_ADDRESS
    return SsrfRejection.RESERVED


async def _system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
    )
    return [str(info[4][0]) for info in infos]


class SsrfGuard:
    def __init__(
        self,
        config: Settings = settings,
        resolver: Resolver = _system_resolver,
        dns_timeout_seconds: float | None = None,
    ) -> None:
        self._allow_private = config.SSRF_ALLOW_PRIVATE_NETWORKS
        self._allowed_ports = frozenset(config.SSRF_ALLOWED_PORTS)
        self._resolver = resolver
        self._dns_timeout = dns_timeout_seconds or config.CRAWL_CONNECT_TIMEOUT_SECONDS

    def check_static(self, url: str) -> tuple[str, str, int]:
        """Scheme, hostname and port checks that need no DNS. Returns (scheme, host, port)."""
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        if scheme not in DEFAULT_PORTS:
            raise SsrfBlockedError(SsrfRejection.UNSUPPORTED_SCHEME)
        host = (parts.hostname or "").rstrip(".").lower()
        if not host:
            raise SsrfBlockedError(SsrfRejection.NO_ADDRESSES)
        try:
            port = parts.port or DEFAULT_PORTS[scheme]
        except ValueError as exc:
            raise SsrfBlockedError(SsrfRejection.PORT_NOT_ALLOWED, host=host) from exc
        if not self._allow_private and port not in self._allowed_ports:
            raise SsrfBlockedError(SsrfRejection.PORT_NOT_ALLOWED, host=host)
        is_localhost = host == "localhost" or host.endswith(".localhost")
        # metadata names stay blocked even when private networks are allowed
        if (host in _BLOCKED_HOSTNAMES or is_localhost) and not (
            self._allow_private and is_localhost
        ):
            raise SsrfBlockedError(SsrfRejection.BLOCKED_HOSTNAME, host=host)
        return scheme, host, port

    async def vet(self, url: str) -> VettedTarget:
        """Full check including DNS. Raises `SsrfBlockedError` or `HostResolutionError`."""
        scheme, host, port = self.check_static(url)
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            addresses = await self._resolve(host, port)
        if not addresses:
            raise SsrfBlockedError(SsrfRejection.NO_ADDRESSES, host=host)
        for address in addresses:
            rejection = classify(address, allow_private=self._allow_private)
            if rejection is not None:
                raise SsrfBlockedError(rejection, host=host)
        unique = tuple(dict.fromkeys(addresses))
        return VettedTarget(scheme=scheme, host=host, port=port, addresses=unique)

    async def _resolve(self, host: str, port: int) -> list[IPAddress]:
        try:
            async with asyncio.timeout(self._dns_timeout):
                raw = await self._resolver(host, port)
        except TimeoutError as exc:
            raise HostResolutionError(f"DNS lookup for {host} timed out") from exc
        except OSError as exc:
            raise HostResolutionError(f"DNS lookup for {host} failed: {exc}") from exc
        # strip IPv6 zone ids (fe80::1%eth0) before parsing
        return [ipaddress.ip_address(a.split("%", 1)[0]) for a in raw]
