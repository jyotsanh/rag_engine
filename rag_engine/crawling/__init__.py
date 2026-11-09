"""Crawling: URLs, scope, robots, rate limits, fetchers, frontier, dedupe and the engine.

The engine's names are exported **lazily** (PEP 562 `__getattr__`): the engine depends on
`rag_engine.extraction`, which itself uses low-level helpers from `crawling.fetchers`.
Importing any `rag_engine.crawling.*` module runs this `__init__`, so an eager engine
import here made `import rag_engine.extraction` circular. `from rag_engine.crawling import
CrawlEngine` still works (resolved on first access), and type checkers see the names
through the `TYPE_CHECKING` import.
"""

from typing import TYPE_CHECKING, Any

from .cancellation import CancelSignal, RedisCancelFlags, finish_cancelled, request_cancel
from .dedupe import (
    ContentClaimStore,
    ContentDeduper,
    DedupeDecision,
    DedupeOutcome,
    content_hash,
)
from .fetchers import RawFetcher
from .frontier import (
    Frontier,
    FrontierEntry,
    FrontierJob,
    FrontierStats,
    RecoveryReport,
    SkippedEntry,
    frontier_priority,
)
from .js_shell import (
    JsShellDetector,
    JsShellThresholds,
    JsShellVerdict,
    JsSignal,
)
from .links import (
    Link,
    LinkExtractor,
    LinkSource,
    PageLinks,
    RejectedLink,
    canonical_from_link_header,
    parse_x_robots_tag,
)
from .rate_limit import (
    DomainRateLimiter,
    RateLimitWaitExceededError,
    Reservation,
    rate_limit_host,
)
from .report import build_report
from .retry import (
    Classification,
    FailureKind,
    RetriesExhaustedError,
    RetryDecision,
    RetryPolicy,
    call_with_retries,
    classify,
)
from .robots import (
    ROBOTS_MAX_BYTES,
    CachedRobots,
    RedisRobotsStore,
    RobotsCache,
    RobotsRules,
    RobotsStatus,
    RobotsStore,
    origin_of,
    robots_token,
)
from .scope import (
    ASSET_EXTENSIONS,
    CrawlScope,
    ScopeDecision,
    ScopeRejection,
    registrable_domain,
)
from .sitemap import (
    MAX_ENTRIES_PER_FILE,
    ParsedSitemap,
    SitemapCap,
    SitemapDiscoverer,
    SitemapEntry,
    SitemapFailure,
    SitemapParseError,
    SitemapProblem,
    SitemapReport,
    inflate,
    parse_lastmod,
    parse_sitemap,
)
from .url import (
    TRACKING_PARAMS,
    TRACKING_PREFIXES,
    UrlNormalizationError,
    normalize_and_hash,
    normalize_url,
    url_hash,
)

__all__ = [
    "ASSET_EXTENSIONS",
    "MAX_ENTRIES_PER_FILE",
    "ROBOTS_MAX_BYTES",
    "TRACKING_PARAMS",
    "TRACKING_PREFIXES",
    "CachedRobots",
    "CancelSignal",
    "Classification",
    "ContentClaimStore",
    "ContentDeduper",
    "CrawlEngine",
    "CrawlScope",
    "CrawlStop",
    "CrawlSummary",
    "DedupeDecision",
    "DedupeOutcome",
    "DomainRateLimiter",
    "FailureKind",
    "Frontier",
    "FrontierEntry",
    "FrontierJob",
    "FrontierStats",
    "JsShellDetector",
    "JsShellThresholds",
    "JsShellVerdict",
    "JsSignal",
    "Link",
    "LinkExtractor",
    "LinkSource",
    "PageFetcher",
    "PageLinks",
    "ParsedSitemap",
    "RateLimitWaitExceededError",
    "RateLimiter",
    "RawFetcher",
    "RecoveryReport",
    "RedisCancelFlags",
    "RedisRobotsStore",
    "RejectedLink",
    "Reservation",
    "RetriesExhaustedError",
    "RetryDecision",
    "RetryPolicy",
    "RobotsCache",
    "RobotsRules",
    "RobotsSource",
    "RobotsStatus",
    "RobotsStore",
    "ScopeDecision",
    "ScopeRejection",
    "SitemapCap",
    "SitemapDiscoverer",
    "SitemapEntry",
    "SitemapFailure",
    "SitemapParseError",
    "SitemapProblem",
    "SitemapReport",
    "SitemapSource",
    "SkippedEntry",
    "UrlNormalizationError",
    "build_report",
    "call_with_retries",
    "canonical_from_link_header",
    "classify",
    "content_hash",
    "finish_cancelled",
    "frontier_priority",
    "inflate",
    "normalize_and_hash",
    "normalize_url",
    "origin_of",
    "parse_lastmod",
    "parse_sitemap",
    "parse_x_robots_tag",
    "rate_limit_host",
    "registrable_domain",
    "render_politeness",
    "request_cancel",
    "robots_token",
    "url_hash",
]

_ENGINE_EXPORTS = frozenset(
    {
        "CrawlEngine",
        "CrawlStop",
        "CrawlSummary",
        "PageFetcher",
        "RateLimiter",
        "RobotsSource",
        "SitemapSource",
        "render_politeness",
    }
)

if TYPE_CHECKING:
    from .engine import (
        CrawlEngine,
        CrawlStop,
        CrawlSummary,
        PageFetcher,
        RateLimiter,
        RobotsSource,
        SitemapSource,
        render_politeness,
    )


def __getattr__(name: str) -> Any:
    if name in _ENGINE_EXPORTS:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
