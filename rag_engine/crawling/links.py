"""Outlinks of an HTML page: resolved, normalized, de-duplicated and run through the scope.

Sources: `<a href>`, `<area href>` and `<meta http-equiv="refresh" content="0; url=…">`.
Relative URLs resolve against the first `<base href>` (itself resolved against the page
URL), else the page's final URL.

Robots directives:
- `rel="nofollow"` on a link, or page-wide `nofollow` (`<meta name="robots">`, a meta for
  our own UA token, or an `X-Robots-Tag` header without a UA prefix or with ours) moves
  links to `nofollow` when `CRAWL_RESPECT_NOFOLLOW` is on. If a URL is linked both ways
  on one page, the followable link wins.
- `noindex` is reported (`PageLinks.noindex`) for the extraction step to act on.

Canonical: `<link rel="canonical">` wins over a `Link: <…>; rel="canonical"` header. It is
normalized but not scope-checked: deciding "canonical elsewhere" is the dedupe step's job.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit

import lxml.html
from lxml import etree

from rag_engine.config import Settings, settings
from rag_engine.crawling.fetchers.content import EXTENSION_KINDS
from rag_engine.crawling.robots import robots_token
from rag_engine.crawling.scope import CrawlScope, ScopeDecision, ScopeRejection
from rag_engine.crawling.url import UrlNormalizationError, normalize_url
from rag_engine.domain import ContentKind, PageReason

_HTTP_SCHEMES = frozenset({"http", "https"})
_MAX_ANCHOR_TEXT = 200
_WHITESPACE = re.compile(r"\s+")
_REFRESH_URL = re.compile(r"""^\s*[\d.]*\s*[;,]?\s*url\s*=\s*['"]?([^'"]+)['"]?\s*$""", re.I)
_LINK_HEADER = re.compile(r"<([^>]*)>([^<]*)")
_REL_CANONICAL = re.compile(r"""\brel\s*=\s*"?[^";,]*\bcanonical\b""", re.I)
# X-Robots-Tag directives that carry a value after ':' (not a user-agent prefix)
_VALUED_DIRECTIVES = frozenset(
    {"unavailable_after", "max-snippet", "max-image-preview", "max-video-preview"}
)


class LinkSource(StrEnum):
    ANCHOR = "a"
    AREA = "area"
    META_REFRESH = "meta_refresh"


@dataclass(frozen=True, slots=True)
class Link:
    url: str  # normalized
    text: str  # anchor text, whitespace-collapsed, truncated
    source: LinkSource
    is_document: bool  # URL extension says PDF/DOCX/PPTX/XLSX (content type decides later)


@dataclass(frozen=True, slots=True)
class RejectedLink:
    url: str  # normalized, or the raw absolute URL for unsupported schemes
    decision: ScopeDecision


@dataclass(slots=True)
class PageLinks:
    page_url: str
    base_url: str
    canonical_url: str | None = None
    noindex: bool = False
    nofollow_page: bool = False  # page-wide nofollow directive
    follow: list[Link] = field(default_factory=list)  # in scope: enqueue these
    nofollow: list[Link] = field(default_factory=list)  # in scope but not to be followed
    rejected: list[RejectedLink] = field(default_factory=list)  # scope said no
    invalid: int = 0  # hrefs that are not URLs at all
    capped: bool = False  # more than CRAWL_MAX_LINKS_PER_PAGE candidates

    @property
    def canonical_is_self(self) -> bool:
        return self.canonical_url is None or self.canonical_url == self.page_url


def _text(element: etree._Element) -> str:
    raw = element.text_content() if isinstance(element, lxml.html.HtmlElement) else ""
    return _WHITESPACE.sub(" ", raw).strip()[:_MAX_ANCHOR_TEXT]


def _directives(content: str) -> set[str]:
    return {d.strip().lower() for d in content.split(",") if d.strip()}


def parse_x_robots_tag(value: str | None, token: str) -> set[str]:
    """Directives from an X-Robots-Tag value that apply to us (no UA prefix, or our token).

    `googlebot: noindex, nofollow` applies only to googlebot; a UA prefix scopes the
    directives after it until the next prefix.
    """
    if not value:
        return set()
    applies, current_ua = set(), None
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        head, sep, rest = part.partition(":")
        if sep and head.strip().lower() not in _VALUED_DIRECTIVES and " " not in head.strip():
            current_ua, part = head.strip().lower(), rest.strip()
            if not part:
                continue
        if current_ua is None or current_ua == token.lower():
            applies.add(part.lower())
    return applies


def canonical_from_link_header(value: str | None) -> str | None:
    if not value:
        return None
    for target, params in _LINK_HEADER.findall(value):
        if _REL_CANONICAL.search(params):
            return target.strip() or None
    return None


class LinkExtractor:
    def __init__(self, config: Settings = settings) -> None:
        self._respect_nofollow = config.CRAWL_RESPECT_NOFOLLOW
        self._max_links = config.CRAWL_MAX_LINKS_PER_PAGE
        self._token = robots_token(config.CRAWL_USER_AGENT).lower()

    def extract(
        self,
        html: str,
        page_url: str,
        *,
        scope: CrawlScope,
        depth: int,
        headers: Mapping[str, str] | None = None,
    ) -> PageLinks:
        """Links of an HTML page fetched from `page_url` (its final URL) at crawl `depth`."""
        page_url = normalize_url(page_url)
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        result = PageLinks(page_url=page_url, base_url=page_url)
        page_directives = parse_x_robots_tag(headers.get("x-robots-tag"), self._token)

        root = _parse(html)
        candidates: list[tuple[str, str, LinkSource, bool]] = []  # (href, text, source, nofollow)
        html_canonical: str | None = None
        if root is not None:
            result.base_url = _base_url(root, page_url)
            for meta in root.iter("meta"):
                name = (meta.get("name") or "").strip().lower()
                if name in ("robots", self._token):
                    page_directives |= _directives(meta.get("content") or "")
                elif (meta.get("http-equiv") or "").strip().lower() == "refresh":
                    match = _REFRESH_URL.match(meta.get("content") or "")
                    if match:
                        candidates.append((match.group(1), "", LinkSource.META_REFRESH, False))
            html_canonical = _html_canonical(root)
            for element in root.iter("a", "area"):
                href = element.get("href")
                if href is None:
                    continue
                rel = {r.lower() for r in (element.get("rel") or "").split()}
                source = LinkSource.ANCHOR if element.tag == "a" else LinkSource.AREA
                candidates.append((href, _text(element), source, "nofollow" in rel))

        result.noindex = bool(page_directives & {"noindex", "none"})
        result.nofollow_page = bool(page_directives & {"nofollow", "none"})
        canonical = html_canonical or canonical_from_link_header(headers.get("link"))
        result.canonical_url = self._resolve(canonical, result.base_url) if canonical else None

        if len(candidates) > self._max_links:
            result.capped = True
            candidates = candidates[: self._max_links]
        self._classify(result, candidates, scope, depth)
        return result

    def filter_urls(
        self, urls: Iterable[str], page_url: str, *, scope: CrawlScope, depth: int
    ) -> PageLinks:
        """Same normalization/scope/dedupe for links found outside HTML (e.g. PDF annotations)."""
        page_url = normalize_url(page_url)
        result = PageLinks(page_url=page_url, base_url=page_url)
        candidates = [(u, "", LinkSource.ANCHOR, False) for u in urls]
        if len(candidates) > self._max_links:
            result.capped = True
            candidates = candidates[: self._max_links]
        self._classify(result, candidates, scope, depth)
        return result

    @staticmethod
    def _resolve(href: str, base: str) -> str | None:
        try:
            return normalize_url(href, base=base)
        except UrlNormalizationError:
            return None

    def _classify(
        self,
        result: PageLinks,
        candidates: list[tuple[str, str, LinkSource, bool]],
        scope: CrawlScope,
        depth: int,
    ) -> None:
        followed: dict[str, Link] = {}
        held_back: dict[str, Link] = {}
        rejected: dict[str, RejectedLink] = {}
        for href, text, source, rel_nofollow in candidates:
            href = href.strip()
            if not href:
                continue
            try:
                absolute = urljoin(result.base_url, href)
                scheme = urlsplit(absolute).scheme.lower()
            except ValueError:  # e.g. "https://[bad": malformed IPv6 netloc
                result.invalid += 1
                continue
            if scheme not in _HTTP_SCHEMES:  # mailto:, tel:, javascript:, data:, ftp: ...
                if scheme and absolute not in rejected:
                    decision = ScopeDecision(
                        False, PageReason.OUT_OF_SCOPE, ScopeRejection.UNSUPPORTED_SCHEME
                    )
                    rejected[absolute] = RejectedLink(absolute, decision)
                elif not scheme:
                    result.invalid += 1
                continue
            url = self._resolve(href, result.base_url)
            if url is None:
                result.invalid += 1
                continue
            if url == result.page_url or url in followed or url in rejected:
                continue  # self-links (incl. #fragment) and repeats
            decision = scope.check(url, depth + 1)
            if not decision.allowed:
                rejected[url] = RejectedLink(url, decision)
                continue
            suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
            kind = EXTENSION_KINDS.get(suffix)
            link = Link(url, text, source, kind is not None and kind is not ContentKind.HTML)
            nofollow = self._respect_nofollow and (rel_nofollow or result.nofollow_page)
            if nofollow:
                held_back.setdefault(url, link)
            else:
                followed[url] = link
                held_back.pop(url, None)  # a followable link wins
        result.follow = list(followed.values())
        result.nofollow = list(held_back.values())
        result.rejected = list(rejected.values())


def _parse(html: str) -> lxml.html.HtmlElement | None:
    if not html or not html.strip():
        return None
    # bytes + explicit encoding: lxml refuses str input that carries an XML encoding declaration
    parser = lxml.html.HTMLParser(encoding="utf-8")
    try:
        return lxml.html.document_fromstring(html.encode("utf-8", errors="replace"), parser=parser)
    except etree.ParserError, ValueError:
        return None


def _base_url(root: lxml.html.HtmlElement, page_url: str) -> str:
    for base in root.iter("base"):
        href = (base.get("href") or "").strip()
        if not href:
            continue
        try:
            resolved = normalize_url(href, base=page_url)
        except UrlNormalizationError:
            return page_url
        return resolved  # only the first <base href> counts (HTML spec)
    return page_url


def _html_canonical(root: lxml.html.HtmlElement) -> str | None:
    for link in root.iter("link"):
        rel = {r.lower() for r in (link.get("rel") or "").split()}
        if "canonical" in rel and (href := (link.get("href") or "").strip()):
            return href
    return None
