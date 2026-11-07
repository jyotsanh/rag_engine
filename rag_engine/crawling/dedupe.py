"""Content dedupe within one crawl job: one indexed copy per distinct content.

`ContentDeduper.check(...)` decides, for a page whose Markdown has been extracted and
cleaned (Step 45), one of:

- **canonical_elsewhere** (page *skipped*, reason `canonical_elsewhere`): the page declares
  a canonical URL (Step 32) that is **in scope** and is not the page itself. Its content
  hash is **not** claimed, so the canonical page can claim it when the engine fetches it
  (the engine enqueues the canonical URL). A canonical on another host/outside the scope
  is ignored, so the content isn't lost. "The page itself" includes every URL that
  reached it (requested URL, redirect chain, final URL): "A redirects to B, B's canonical
  is A" must not skip the only copy.
- **duplicate** (status `duplicate`, `duplicate_of` = the original page): another page of
  this job already claimed the same content hash.
- **unique**: this page claimed the hash; it is the original that gets chunked/embedded.

Claims are atomic through the store (Mongo unique index in production), so concurrent
fetches of `/a` and `/b` with equal content produce exactly one original. Re-checking a
page that already owns the hash (a resumed job) is unique again: idempotent.

The first page to claim wins. Crawl order is hub/shallow first (Step 48), so originals
tend to be the better-linked URLs. Only exact duplicates of the cleaned Markdown are
caught; near-duplicates (a different timestamp line) are not.
"""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rag_engine.config import Settings, settings
from rag_engine.crawling.scope import CrawlScope
from rag_engine.crawling.url import UrlNormalizationError, normalize_url
from rag_engine.domain import ContentHashClaim, PageReason, PageStatus


def content_hash(markdown: str) -> str:
    """sha256 hex of cleaned Markdown (`Page.content_hash`)."""
    return hashlib.sha256(markdown.strip().encode("utf-8")).hexdigest()


class ContentClaimStore(Protocol):
    """`ContentClaimRepository` in production; an in-memory fake in unit tests."""

    async def claim(
        self, job_id: str, content_hash: str, page_id: str, url: str
    ) -> ContentHashClaim: ...


class DedupeOutcome(StrEnum):
    UNIQUE = "unique"
    DUPLICATE = "duplicate"
    CANONICAL_ELSEWHERE = "canonical_elsewhere"


@dataclass(frozen=True, slots=True)
class DedupeDecision:
    outcome: DedupeOutcome
    content_hash: str
    canonical_url: str | None = None  # normalized; set when the page declared one
    duplicate_of: str | None = None  # original page id (DUPLICATE)
    duplicate_of_url: str | None = None

    @property
    def page_status(self) -> PageStatus:
        return {
            DedupeOutcome.UNIQUE: PageStatus.SUCCEEDED,
            DedupeOutcome.DUPLICATE: PageStatus.DUPLICATE,
            DedupeOutcome.CANONICAL_ELSEWHERE: PageStatus.SKIPPED,
        }[self.outcome]

    @property
    def reason(self) -> PageReason | None:
        if self.outcome is DedupeOutcome.CANONICAL_ELSEWHERE:
            return PageReason.CANONICAL_ELSEWHERE
        return None


class ContentDeduper:
    def __init__(
        self,
        job_id: str,
        scope: CrawlScope,
        store: ContentClaimStore,
        *,
        config: Settings = settings,
    ) -> None:
        self._job_id = job_id
        self._scope = scope
        self._store = store
        self._respect_canonical = config.CRAWL_RESPECT_CANONICAL

    async def check(
        self,
        *,
        page_id: str,
        url: str,
        markdown: str,
        depth: int,
        canonical_url: str | None = None,
        aliases: tuple[str, ...] = (),
    ) -> DedupeDecision:
        """`url` is the page's final URL; `aliases` the requested URL and redirect chain."""
        digest = content_hash(markdown)
        canonical = _normalized(canonical_url)
        if self._is_canonical_elsewhere(canonical, url, aliases, depth):
            return DedupeDecision(DedupeOutcome.CANONICAL_ELSEWHERE, digest, canonical)

        owner = await self._store.claim(self._job_id, digest, page_id, url)
        if owner.page_id == page_id:
            return DedupeDecision(DedupeOutcome.UNIQUE, digest, canonical)
        return DedupeDecision(
            DedupeOutcome.DUPLICATE,
            digest,
            canonical,
            duplicate_of=owner.page_id,
            duplicate_of_url=owner.url,
        )

    def _is_canonical_elsewhere(
        self, canonical: str | None, url: str, aliases: tuple[str, ...], depth: int
    ) -> bool:
        if not self._respect_canonical or canonical is None:
            return False
        selves = {u for u in (_normalized(a) for a in (url, *aliases)) if u is not None}
        if canonical in selves:
            return False
        return self._scope.check(canonical, depth).allowed


def _normalized(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return normalize_url(url)
    except UrlNormalizationError:
        return None
