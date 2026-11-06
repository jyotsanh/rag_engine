"""Sitemap discovery: robots.txt `Sitemap:` lines + `/sitemap.xml`, index recursion, gzip.

`SitemapDiscoverer.discover(seed_url, scope)` returns the in-scope page URLs that the
site's sitemaps list, plus a report of what was fetched, skipped and why.

Safety and caps:
- Every sitemap goes through the SSRF-safe fetcher with Step 28 retries and, when a
  limiter is given, the per-domain rate limit (Step 30).
- Sitemaps declared in robots.txt may live on any host (cross-submission is allowed by
  the protocol). Index children and the `/sitemap.xml` fallback must be on an in-scope
  host and allowed by that host's robots.txt.
- XML is parsed incrementally with defusedxml (DTDs/entities refused: no billion laughs,
  no XXE). Gzip is detected by magic bytes (`.gz` files are often served as
  `application/gzip`, not `Content-Encoding`) and inflated with a hard size cap.
- Caps: `SITEMAP_MAX_BYTES` per file (compressed and inflated), `SITEMAP_MAX_FILES`
  fetched, `SITEMAP_MAX_DEPTH` index nesting, 50 000 entries per file (protocol limit),
  and the caller's `max_urls`.
- Page URLs are normalized, de-duplicated and filtered through the job's `CrawlScope`.
  robots.txt rules for those pages are *not* applied here; the engine checks them per URL.
"""

import io
import zlib
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element, ParseError

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import iterparse
from loguru import logger

from rag_engine.config import Settings, settings
from rag_engine.crawling.fetchers import HttpStatusError, RawFetcher
from rag_engine.crawling.rate_limit import DomainRateLimiter
from rag_engine.crawling.retry import RetriesExhaustedError, RetryPolicy, call_with_retries
from rag_engine.crawling.robots import RobotsCache, RobotsStatus, origin_of
from rag_engine.crawling.scope import CrawlScope
from rag_engine.crawling.url import UrlNormalizationError, normalize_url
from rag_engine.domain import PageReason

SITEMAP_FALLBACK_PATH = "/sitemap.xml"
MAX_ENTRIES_PER_FILE = 50_000  # sitemaps.org protocol limit
_ACCEPT = "application/xml, text/xml;q=0.9, application/gzip;q=0.8, text/plain;q=0.5, */*;q=0.1"
_GZIP_MAGIC = b"\x1f\x8b"


class SitemapProblem(StrEnum):
    FETCH_FAILED = "fetch_failed"  # detail: "<PageReason>: <detail>"
    TOO_LARGE = "too_large"  # download or inflated body over SITEMAP_MAX_BYTES
    INVALID_XML = "invalid_xml"
    FORBIDDEN_XML = "forbidden_xml"  # DTD / entity declarations
    UNSUPPORTED_FORMAT = "unsupported_format"  # RSS, Atom, HTML soft-404s, ...
    TOO_DEEP = "too_deep"  # index nesting over SITEMAP_MAX_DEPTH
    OUT_OF_SCOPE_HOST = "out_of_scope_host"
    ROBOTS_DISALLOWED = "robots_disallowed"


class SitemapCap(StrEnum):
    MAX_URLS = "max_urls"
    MAX_FILES = "max_files"
    ENTRIES_PER_FILE = "entries_per_file"


@dataclass(frozen=True, slots=True)
class SitemapEntry:
    url: str  # normalized, in scope
    lastmod: datetime | None
    sitemap: str  # the sitemap that listed it


@dataclass(frozen=True, slots=True)
class SitemapFailure:
    sitemap: str
    problem: SitemapProblem
    detail: str | None = None


@dataclass(slots=True)
class SitemapReport:
    entries: list[SitemapEntry] = field(default_factory=list)
    sitemaps_read: list[str] = field(default_factory=list)
    failures: list[SitemapFailure] = field(default_factory=list)
    out_of_scope: int = 0  # page URLs rejected by the crawl scope
    invalid: int = 0  # <loc> values that are not crawlable URLs
    duplicates: int = 0
    caps_hit: set[SitemapCap] = field(default_factory=set)

    @property
    def urls(self) -> list[str]:
        return [e.url for e in self.entries]


@dataclass(frozen=True, slots=True)
class _Pending:
    url: str
    depth: int  # 0 = top level (robots-declared or fallback)
    declared: bool  # listed in robots.txt: any host allowed
    fallback: bool = False  # the guessed /sitemap.xml: a 404 is not a failure


@dataclass(frozen=True, slots=True)
class ParsedSitemap:
    kind: str  # "urlset" | "sitemapindex"; a plain-text URL list is a "urlset"
    items: list[tuple[str, datetime | None]]
    capped: bool


class SitemapParseError(Exception):
    """A sitemap that cannot be used; `problem` says why."""

    def __init__(self, problem: SitemapProblem, detail: str | None = None) -> None:
        super().__init__(detail or problem.value)
        self.problem = problem
        self.detail = detail


def parse_lastmod(value: str | None) -> datetime | None:
    """W3C datetime (`2024-05-01`, `2024-05-01T10:00:00Z`, with offsets) → aware UTC datetime."""
    if not value or not (text := value.strip()):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def inflate(body: bytes, max_bytes: int) -> bytes:
    """Gunzip when the body starts with the gzip magic, never producing more than `max_bytes`."""
    if not body.startswith(_GZIP_MAGIC):
        return body
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data = decompressor.decompress(body, max_bytes + 1)
    except zlib.error as exc:
        raise SitemapParseError(SitemapProblem.INVALID_XML, f"bad gzip: {exc}") from exc
    if len(data) > max_bytes:
        raise SitemapParseError(SitemapProblem.TOO_LARGE, f"inflates past {max_bytes} bytes")
    if not decompressor.eof:  # cut off mid-stream: the XML would be silently partial
        raise SitemapParseError(SitemapProblem.INVALID_XML, "truncated gzip stream")
    return data


def _local(tag: object) -> str:
    """Tag name without namespace: plenty of sitemaps use a wrong or missing xmlns."""
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _child_text(element: Element, name: str) -> str | None:
    for child in element:
        if _local(child.tag) == name:
            return (child.text or "").strip() or None
    return None


def _parse_xml(data: bytes) -> ParsedSitemap:
    events: Iterator[tuple[str, Element]] = iterparse(
        io.BytesIO(data), events=("start", "end"), forbid_dtd=True
    )
    kind = ""
    root: Element | None = None
    items: list[tuple[str, datetime | None]] = []
    try:
        for event, element in events:
            if root is None:  # first event: the document element
                root, kind = element, _local(element.tag)
                if kind not in ("urlset", "sitemapindex"):
                    raise SitemapParseError(SitemapProblem.UNSUPPORTED_FORMAT, f"<{kind}> document")
                continue
            if event != "end" or _local(element.tag) not in ("url", "sitemap"):
                continue
            if loc := _child_text(element, "loc"):
                items.append((loc, parse_lastmod(_child_text(element, "lastmod"))))
            root.clear()  # keep memory flat on 50k-entry files
            if len(items) >= MAX_ENTRIES_PER_FILE:
                return ParsedSitemap(kind, items, capped=True)
    except DefusedXmlException as exc:
        raise SitemapParseError(SitemapProblem.FORBIDDEN_XML, type(exc).__name__) from exc
    except ParseError as exc:
        raise SitemapParseError(SitemapProblem.INVALID_XML, str(exc)) from exc
    if not kind:
        raise SitemapParseError(SitemapProblem.INVALID_XML, "empty document")
    return ParsedSitemap(kind, items, capped=False)


def _parse_text(data: bytes) -> ParsedSitemap:
    """Plain-text sitemap: one absolute URL per line (sitemaps.org allows it)."""
    lines = data.decode("utf-8", errors="replace").splitlines()
    urls = [line.strip() for line in lines if line.strip()]
    if not urls or not all(u.lower().startswith(("http://", "https://")) for u in urls[:5]):
        raise SitemapParseError(SitemapProblem.UNSUPPORTED_FORMAT, "neither XML nor a URL list")
    capped = len(urls) > MAX_ENTRIES_PER_FILE
    return ParsedSitemap("urlset", [(u, None) for u in urls[:MAX_ENTRIES_PER_FILE]], capped)


def parse_sitemap(data: bytes) -> ParsedSitemap:
    head = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(b"<"):
        if head[:200].lower().lstrip().startswith((b"<!doctype html", b"<html")):
            raise SitemapParseError(SitemapProblem.UNSUPPORTED_FORMAT, "HTML page")
        return _parse_xml(data)
    return _parse_text(head)


class SitemapDiscoverer:
    def __init__(
        self,
        fetcher: RawFetcher,
        *,
        robots: RobotsCache | None = None,
        limiter: DomainRateLimiter | None = None,
        config: Settings = settings,
        policy: RetryPolicy | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._robots = robots
        self._limiter = limiter
        self._policy = policy or RetryPolicy.from_settings(config)
        self._max_bytes = config.SITEMAP_MAX_BYTES
        self._max_files = config.SITEMAP_MAX_FILES
        self._max_depth = config.SITEMAP_MAX_DEPTH
        self._default_max_urls = config.CRAWL_HARD_MAX_PAGES

    async def discover(
        self, seed_url: str, scope: CrawlScope, *, max_urls: int | None = None
    ) -> SitemapReport:
        """Sitemaps from the seed origin's robots.txt, then `/sitemap.xml`."""
        origin = origin_of(seed_url)
        declared: list[str] = []
        if self._robots is not None:
            rules = await self._robots.rules_for(seed_url)
            if rules.status in (RobotsStatus.OK, RobotsStatus.STALE):
                declared = list(rules.sitemaps)
        start = [_Pending(url, 0, declared=True) for url in declared]
        start.append(_Pending(origin + SITEMAP_FALLBACK_PATH, 0, declared=False, fallback=True))
        return await self._run(start, scope, max_urls)

    async def read(
        self, sitemap_urls: list[str], scope: CrawlScope, *, max_urls: int | None = None
    ) -> SitemapReport:
        """Read explicitly given sitemaps (treated like robots.txt-declared ones)."""
        return await self._run(
            [_Pending(u, 0, declared=True) for u in sitemap_urls], scope, max_urls
        )

    async def _run(
        self, start: list[_Pending], scope: CrawlScope, max_urls: int | None
    ) -> SitemapReport:
        limit = max_urls if max_urls is not None else self._default_max_urls
        report = SitemapReport()
        seen_sitemaps: set[str] = set()
        seen_urls: set[str] = set()
        queue: deque[_Pending] = deque()
        for pending in start:
            self._enqueue(queue, seen_sitemaps, pending)

        while queue:
            if len(report.entries) >= limit:
                report.caps_hit.add(SitemapCap.MAX_URLS)
                break
            if len(report.sitemaps_read) >= self._max_files:
                report.caps_hit.add(SitemapCap.MAX_FILES)
                break
            pending = queue.popleft()
            try:
                parsed = await self._load(pending, scope)
            except SitemapParseError as exc:
                report.failures.append(SitemapFailure(pending.url, exc.problem, exc.detail))
                continue
            if parsed is None:  # fallback /sitemap.xml absent: nothing to report
                continue
            report.sitemaps_read.append(pending.url)
            if parsed.capped:
                report.caps_hit.add(SitemapCap.ENTRIES_PER_FILE)

            if parsed.kind == "sitemapindex":
                for loc, _ in parsed.items:
                    child = _Pending(loc.strip(), pending.depth + 1, declared=False)
                    if child.depth > self._max_depth:
                        report.failures.append(
                            SitemapFailure(
                                child.url, SitemapProblem.TOO_DEEP, f"depth {child.depth}"
                            )
                        )
                        continue
                    self._enqueue(queue, seen_sitemaps, child)
                continue

            for loc, lastmod in parsed.items:
                if len(report.entries) >= limit:
                    report.caps_hit.add(SitemapCap.MAX_URLS)
                    break
                try:
                    url = normalize_url(loc)
                except UrlNormalizationError:
                    report.invalid += 1
                    continue
                if url in seen_urls:
                    report.duplicates += 1
                    continue
                seen_urls.add(url)
                if not scope.check(url, depth=0).allowed:
                    report.out_of_scope += 1
                    continue
                report.entries.append(SitemapEntry(url, lastmod, pending.url))

        logger.info(
            "sitemaps: {} read, {} urls, {} out of scope, {} failures, caps {}",
            len(report.sitemaps_read),
            len(report.entries),
            report.out_of_scope,
            len(report.failures),
            sorted(report.caps_hit) or "-",
        )
        return report

    @staticmethod
    def _enqueue(queue: deque[_Pending], seen: set[str], pending: _Pending) -> None:
        try:
            key = normalize_url(pending.url)
        except UrlNormalizationError:
            return
        if key not in seen:
            seen.add(key)
            queue.append(_Pending(key, pending.depth, pending.declared, pending.fallback))

    async def _load(self, pending: _Pending, scope: CrawlScope) -> ParsedSitemap | None:
        url = pending.url
        if not pending.declared:
            host = urlsplit(url).hostname or ""
            if not scope.allows_host(host):
                raise SitemapParseError(SitemapProblem.OUT_OF_SCOPE_HOST, host)
            if self._robots is not None and not await self._robots.can_fetch(url):
                raise SitemapParseError(SitemapProblem.ROBOTS_DISALLOWED)

        if self._limiter is not None:
            crawl_delay = None
            if self._robots is not None:
                crawl_delay = (await self._robots.rules_for(url)).crawl_delay
            await self._limiter.acquire(url, crawl_delay=crawl_delay)

        try:
            response = await call_with_retries(
                lambda: self._fetcher.fetch_raw(url, accept=_ACCEPT, max_bytes=self._max_bytes),
                self._policy,
            )
        except RetriesExhaustedError as exc:
            error = exc.error
            if pending.fallback and isinstance(error, HttpStatusError) and error.status_code < 500:
                return None
            if exc.decision.reason is PageReason.TOO_LARGE:
                raise SitemapParseError(SitemapProblem.TOO_LARGE, exc.decision.detail) from exc
            detail = f"{exc.decision.reason}: {exc.decision.detail}"
            raise SitemapParseError(SitemapProblem.FETCH_FAILED, detail) from exc

        try:
            return parse_sitemap(inflate(response.body, self._max_bytes))
        except SitemapParseError as exc:
            if pending.fallback and exc.problem is SitemapProblem.UNSUPPORTED_FORMAT:
                return None  # soft-404 page at the guessed location
            raise
