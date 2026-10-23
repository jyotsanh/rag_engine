"""Which discovered URLs a crawl may fetch.

A `CrawlScope` is built once per job from the (normalized) seed URL and the
job's `CrawlConfig`, then answers `check(url, depth)` for every candidate:

- host: same host as the seed (`www.` and the bare domain count as the same
  host); with `include_subdomains`, any host under the seed's registrable
  domain (Public Suffix List, private section included, bundled snapshot,
  never downloaded). Scheme and port are not part of the comparison.
- `path_prefix`: optional, e.g. `/docs/`.
- include/exclude globs on the path: `*` matches within one segment, `**`
  across segments, `?` one character. Exclude wins; no includes = include all.
- asset files (images, styles, scripts, fonts, media, archives, binaries) are
  skipped before fetching; documents (PDF/Office) are in scope.
- depth: links deeper than `max_depth` hops from the seed are skipped.

Input URLs must already be normalized (`normalize_url`).
"""

import ipaddress
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import tldextract

from rag_engine.domain import CrawlConfig, PageReason
from rag_engine.domain.types import compile_glob

ASSET_EXTENSIONS = frozenset(
    {
        # images
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp", ".ico", ".svg", ".tif", ".tiff",
        # styles, scripts, data feeds
        ".css", ".js", ".mjs", ".map", ".json", ".xml", ".rss", ".atom", ".wasm",
        # fonts
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        # audio / video
        ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv",
        # archives / binaries / installers
        ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar", ".exe", ".msi", ".dmg",
        ".pkg", ".deb", ".rpm", ".apk", ".iso", ".bin", ".jar",
    }
)  # fmt: skip

_PSL = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)


class ScopeRejection(StrEnum):
    """Detail stored in `Page.reason_detail` next to the coarse `PageReason`."""

    OTHER_HOST = "other_host"
    OUTSIDE_PATH_PREFIX = "outside_path_prefix"
    EXCLUDED_PATTERN = "excluded_pattern"
    NOT_INCLUDED = "not_included"
    ASSET_EXTENSION = "asset_extension"
    UNSUPPORTED_SCHEME = "unsupported_scheme"
    TOO_DEEP = "too_deep"


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    allowed: bool
    reason: PageReason | None = None
    detail: ScopeRejection | None = None


_ALLOWED = ScopeDecision(allowed=True)


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _is_ip_or_single_label(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return "." not in host


@lru_cache(maxsize=4096)
def registrable_domain(host: str) -> str:
    """`docs.acme.co.uk` → `acme.co.uk`; unknown suffixes fall back to the last two labels."""
    if _is_ip_or_single_label(host):
        return host
    domain = _PSL(host).top_domain_under_public_suffix
    return domain or ".".join(host.split(".")[-2:])


@dataclass(frozen=True, slots=True)
class CrawlScope:
    seed_host: str
    max_depth: int
    include_subdomains: bool = False
    path_prefix: str | None = None
    include: tuple[re.Pattern[str], ...] = ()
    exclude: tuple[re.Pattern[str], ...] = ()

    @classmethod
    def from_config(
        cls,
        seed_url: str,
        *,
        max_depth: int,
        include_subdomains: bool = False,
        path_prefix: str | None = None,
        include_patterns: list[str] | tuple[str, ...] = (),
        exclude_patterns: list[str] | tuple[str, ...] = (),
    ) -> CrawlScope:
        host = urlsplit(seed_url).hostname
        if not host:
            raise ValueError(f"seed URL has no host: {seed_url!r}")
        return cls(
            seed_host=host,
            max_depth=max_depth,
            include_subdomains=include_subdomains,
            path_prefix=path_prefix,
            include=tuple(compile_glob(p) for p in include_patterns),
            exclude=tuple(compile_glob(p) for p in exclude_patterns),
        )

    @classmethod
    def for_job(cls, seed_url: str, config: CrawlConfig) -> CrawlScope:
        return cls.from_config(
            seed_url,
            max_depth=config.max_depth,
            include_subdomains=config.include_subdomains,
            path_prefix=config.path_prefix,
            include_patterns=config.include_patterns,
            exclude_patterns=config.exclude_patterns,
        )

    def with_seed_host(self, host: str) -> CrawlScope:
        """Re-anchor after the seed redirected to another host (e.g. bare → www)."""
        return CrawlScope(
            seed_host=host,
            max_depth=self.max_depth,
            include_subdomains=self.include_subdomains,
            path_prefix=self.path_prefix,
            include=self.include,
            exclude=self.exclude,
        )

    def allows_host(self, host: str) -> bool:
        host = host.lower()
        if _strip_www(host) == _strip_www(self.seed_host):
            return True
        if not self.include_subdomains or _is_ip_or_single_label(host):
            return False
        site = registrable_domain(self.seed_host)
        return host == site or host.endswith("." + site)

    def check(self, url: str, depth: int) -> ScopeDecision:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            return ScopeDecision(False, PageReason.OUT_OF_SCOPE, ScopeRejection.UNSUPPORTED_SCHEME)
        if not parts.hostname or not self.allows_host(parts.hostname):
            return ScopeDecision(False, PageReason.OUT_OF_SCOPE, ScopeRejection.OTHER_HOST)
        path = parts.path or "/"
        if PurePosixPath(path).suffix.lower() in ASSET_EXTENSIONS:
            return ScopeDecision(False, PageReason.OUT_OF_SCOPE, ScopeRejection.ASSET_EXTENSION)
        if self.path_prefix is not None and not path.startswith(self.path_prefix):
            return ScopeDecision(False, PageReason.OUT_OF_SCOPE, ScopeRejection.OUTSIDE_PATH_PREFIX)
        if any(p.match(path) for p in self.exclude):
            return ScopeDecision(False, PageReason.OUT_OF_SCOPE, ScopeRejection.EXCLUDED_PATTERN)
        if self.include and not any(p.match(path) for p in self.include):
            return ScopeDecision(False, PageReason.OUT_OF_SCOPE, ScopeRejection.NOT_INCLUDED)
        if depth > self.max_depth:
            return ScopeDecision(False, PageReason.MAX_DEPTH, ScopeRejection.TOO_DEEP)
        return _ALLOWED
