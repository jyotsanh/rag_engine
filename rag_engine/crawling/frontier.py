"""The crawl frontier: a Mongo-backed queue of one job's URLs, stored as `Page` documents.

- **Enqueue** inserts a `queued` page per URL. The `(job_id, url_hash)` unique index is the
  dedupe: adding a known URL is a no-op, race-safe across concurrent crawlers. If a URL is
  rediscovered at a better priority while still queued, it is **promoted** (shallower depth,
  better source), so `max_depth` sees its shortest known path.
- **Order**: lowest `priority` first, then discovery order (`_id` is time-ordered).
  `priority = depth * 10 + rank(via)` with seed < sitemap < canonical < link: sitemap
  entries (depth 0) and shallow pages go first; a canonical target jumps the links found
  at its depth, so the page it replaces gets claimed early.
- **Claim** is one atomic `find_one_and_update`: `queued → in_progress`, `attempts += 1`,
  `claimed_by = run_id`, a lease until now + `CRAWL_FRONTIER_LEASE_SECONDS`. Two crawlers
  never get the same page. Pages waiting for a retry (`retry_at` in the future) are not
  claimable yet.
- **Retries** (`retry_later`) put a page back as `queued` with a not-before time instead
  of sleeping inside a concurrency slot (Step 28's policy decides the delay).
- **Crash recovery**: a resumed job calls `recover()`, which re-queues every page still held
  by an earlier run. Pages whose lease expired (a crawler died mid-job) are reclaimed by
  `claim()` once nothing else is ready. A page that has used up all its attempts
  (`CRAWL_MAX_RETRIES + 1`) without reporting back fails as `worker_lost`, so a page that
  crashes the worker every time cannot loop forever.

The engine records outcomes through `PageRepository.record_outcome`, which also clears the
lease. Each job's frontier is bound to one `run_id` (a new one per attempt of the job).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from loguru import logger
from pymongo import ASCENDING, ReturnDocument, UpdateOne
from pymongo.errors import PyMongoError

from rag_engine.config import Settings, settings
from rag_engine.crawling.url import UrlNormalizationError, normalize_and_hash
from rag_engine.domain import DiscoveredVia, Page, PageReason, PageStatus, utc_now
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.repositories import PageRepository

_VIA_RANK = {
    DiscoveredVia.SEED: 0,
    DiscoveredVia.SITEMAP: 1,
    DiscoveredVia.CANONICAL: 2,
    DiscoveredVia.LINK: 3,
}
_DEPTH_STEP = 10  # leaves room for more discovery sources without renumbering stored pages
_CLAIM_ORDER = [("priority", ASCENDING), ("_id", ASCENDING)]


def frontier_priority(via: DiscoveredVia, depth: int) -> int:
    """Lower is claimed first: by depth, then seed < sitemap < canonical < link."""
    if depth < 0:
        raise ValueError("depth must be >= 0")
    return depth * _DEPTH_STEP + _VIA_RANK[via]


@dataclass(frozen=True, slots=True)
class FrontierJob:
    """The job a frontier belongs to; every page it creates carries these ids."""

    tenant_id: str
    kb_id: str
    source_id: str
    job_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class FrontierEntry:
    url: str
    depth: int
    via: DiscoveredVia
    discovered_from: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SkippedEntry:
    url: str
    depth: int
    reason: PageReason
    detail: str | None = None
    discovered_from: str | None = None


@dataclass(frozen=True, slots=True)
class FrontierStats:
    ready: int  # claimable now
    delayed: int  # queued for a retry later
    in_flight: int  # claimed, outcome not recorded yet
    next_retry_at: datetime | None  # earliest delayed retry

    @property
    def drained(self) -> bool:
        """Nothing left to claim or wait for: the crawl part of the job is done."""
        return self.ready == self.delayed == self.in_flight == 0


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    requeued: int  # pages an earlier run held, back in the queue
    abandoned: int  # pages out of attempts, failed as worker_lost


class Frontier:
    def __init__(
        self,
        pages: PageRepository,
        job: FrontierJob,
        *,
        run_id: str,
        config: Settings = settings,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if not run_id or len(run_id) > 100:
            raise ValueError("run_id must be 1-100 characters")
        self._pages = pages
        self._collection = pages.collection
        self._job = job
        self.run_id = run_id
        self._lease = timedelta(seconds=config.CRAWL_FRONTIER_LEASE_SECONDS)
        self.max_attempts = config.CRAWL_MAX_RETRIES + 1
        self._clock = clock

    # ---- enqueue ---- #

    async def add(
        self,
        url: str,
        *,
        depth: int,
        via: DiscoveredVia,
        discovered_from: str | None = None,
    ) -> bool:
        """Queue one URL. True if it was new; a known URL is only promoted (if better)."""
        entry = FrontierEntry(url=url, depth=depth, via=via, discovered_from=discovered_from)
        page = self._page(entry)
        _, created = await self._pages.add_discovered(page)
        if not created:
            await self._promote([page])
        return created

    async def add_many(self, entries: Iterable[FrontierEntry]) -> int:
        """Queue many URLs (sitemap seeding, a page's outlinks). Returns how many were new.

        Invalid URLs are skipped (callers pass normalized URLs already); a URL listed twice
        keeps its best priority.
        """
        best: dict[str, Page] = {}
        for entry in entries:
            try:
                page = self._page(entry)
            except UrlNormalizationError:
                logger.debug("frontier: skipping invalid URL {!r}", entry.url)
                continue
            known = best.get(page.url_hash)
            if known is None or page.priority < known.priority:
                best[page.url_hash] = page
        pages = list(best.values())
        inserted = await self._pages.add_many_discovered(pages)
        if inserted < len(pages):
            await self._promote(pages)
        return inserted

    async def add_skipped(self, entries: Iterable[SkippedEntry]) -> int:
        """Record URLs that won't be crawled (out of scope, too deep, nofollow) as skipped.

        Only URLs the job doesn't know yet are recorded: a URL that was also queued (some
        other link reached it in scope) keeps its real outcome. Returns how many were new.
        """
        best: dict[str, Page] = {}
        for entry in entries:
            try:
                url, url_hash = normalize_and_hash(entry.url)
            except UrlNormalizationError:
                continue
            if url_hash in best:
                continue
            best[url_hash] = Page(
                tenant_id=self._job.tenant_id,
                kb_id=self._job.kb_id,
                source_id=self._job.source_id,
                job_id=self._job.job_id,
                url=url,
                url_hash=url_hash,
                depth=entry.depth,
                discovered_from=entry.discovered_from,
                discovered_via=DiscoveredVia.LINK,
                priority=frontier_priority(DiscoveredVia.LINK, entry.depth),
                status=PageStatus.SKIPPED,
                reason=entry.reason,
                reason_detail=(entry.detail or "")[:500] or None,
            )
        return await self._pages.add_many_discovered(best.values())

    def _page(self, entry: FrontierEntry) -> Page:
        url, url_hash = normalize_and_hash(entry.url)
        return Page(
            tenant_id=self._job.tenant_id,
            kb_id=self._job.kb_id,
            source_id=self._job.source_id,
            job_id=self._job.job_id,
            url=url,
            url_hash=url_hash,
            depth=entry.depth,
            discovered_from=entry.discovered_from,
            discovered_via=entry.via,
            priority=frontier_priority(entry.via, entry.depth),
        )

    async def _promote(self, pages: list[Page]) -> None:
        """Give still-queued known URLs the better priority/depth of a new discovery."""
        now = self._clock()
        updates = [
            UpdateOne(
                {
                    "job_id": self._job.job_id,
                    "url_hash": page.url_hash,
                    "status": PageStatus.QUEUED.value,
                    "priority": {"$gt": page.priority},
                },
                {
                    "$set": {
                        "priority": page.priority,
                        "depth": page.depth,
                        "discovered_via": page.discovered_via.value,
                        "discovered_from": page.discovered_from,
                        "updated_at": now,
                    }
                },
            )
            for page in pages
        ]
        try:
            await self._collection.bulk_write(updates, ordered=False)
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc

    # ---- claim ---- #

    async def claim(self) -> Page | None:
        """The next page to crawl, leased to this run; None when nothing is claimable now."""
        now = self._clock()
        page = await self._claim_one(self._ready(now), now)
        if page is not None:
            return page
        # nothing queued: take over pages whose crawler died (lease expired)
        await self._abandon_exhausted({"lease_expires_at": {"$lte": now}}, now)
        expired = {
            "job_id": self._job.job_id,
            "status": PageStatus.IN_PROGRESS.value,
            "lease_expires_at": {"$lte": now},
            "attempts": {"$lt": self.max_attempts},
        }
        page = await self._claim_one(expired, now)
        if page is not None:
            logger.warning(
                "frontier: reclaimed {} from {} (lease expired)", page.url, page.claimed_by
            )
        return page

    async def claim_url(self, url: str) -> Page | None:
        """Claim one specific queued page (e.g. a redirect target whose content we hold).

        None if the URL is unknown, invalid, or not queued (done, or held by a crawler).
        """
        try:
            _, url_hash = normalize_and_hash(url)
        except UrlNormalizationError:
            return None
        now = self._clock()
        filter_ = {
            "job_id": self._job.job_id,
            "url_hash": url_hash,
            "status": PageStatus.QUEUED.value,
        }
        return await self._claim_one(filter_, now)

    def _ready(self, now: datetime) -> dict[str, Any]:
        """Queued pages of this job that are claimable now (no retry pending)."""
        return {
            "job_id": self._job.job_id,
            "status": PageStatus.QUEUED.value,
            "$or": [{"retry_at": None}, {"retry_at": {"$lte": now}}],
        }

    async def _claim_one(self, filter_: dict[str, Any], now: datetime) -> Page | None:
        update = {
            "$set": {
                "status": PageStatus.IN_PROGRESS.value,
                "claimed_by": self.run_id,
                "lease_expires_at": now + self._lease,
                "retry_at": None,
                "updated_at": now,
            },
            "$inc": {"attempts": 1},
        }
        try:
            raw = await self._collection.find_one_and_update(
                filter_, update, sort=_CLAIM_ORDER, return_document=ReturnDocument.AFTER
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return Page.from_mongo(raw) if raw is not None else None

    # ---- give back ---- #

    async def retry_later(self, page_id: str, delay_seconds: float) -> bool:
        """Re-queue a page this run holds, claimable again after `delay_seconds`.

        False if this run no longer holds it (its lease was reclaimed by another run).
        """
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")
        now = self._clock()
        return await self._give_back(page_id, now + timedelta(seconds=delay_seconds), now)

    async def release(self, page_id: str, *, delay_seconds: float = 0) -> bool:
        """Put a held page back unprocessed; the attempt is refunded.

        For pages that were never tried: stopping early, or a host whose rate-limit slot is
        further away than a crawler should wait (`delay_seconds` until it is claimable).
        """
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")
        now = self._clock()
        retry_at = now + timedelta(seconds=delay_seconds) if delay_seconds else None
        return await self._give_back(page_id, retry_at, now, refund=True)

    async def _give_back(
        self, page_id: str, retry_at: datetime | None, now: datetime, *, refund: bool = False
    ) -> bool:
        update: dict[str, Any] = {
            "$set": {
                "status": PageStatus.QUEUED.value,
                "retry_at": retry_at,
                "claimed_by": None,
                "lease_expires_at": None,
                "updated_at": now,
            }
        }
        if refund:
            update["$inc"] = {"attempts": -1}
        try:
            result = await self._collection.update_one(
                {
                    "_id": page_id,
                    "job_id": self._job.job_id,
                    "status": PageStatus.IN_PROGRESS.value,
                    "claimed_by": self.run_id,
                },
                update,
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return result.modified_count == 1

    # ---- crash recovery ---- #

    async def recover(self) -> RecoveryReport:
        """On (re)start: re-queue pages earlier runs left in progress (they died)."""
        now = self._clock()
        held_elsewhere = {"claimed_by": {"$ne": self.run_id}}
        abandoned = await self._abandon_exhausted(held_elsewhere, now)
        try:
            result = await self._collection.update_many(
                {
                    "job_id": self._job.job_id,
                    "status": PageStatus.IN_PROGRESS.value,
                    **held_elsewhere,
                },
                {
                    "$set": {
                        "status": PageStatus.QUEUED.value,
                        "claimed_by": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    }
                },
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        report = RecoveryReport(requeued=result.modified_count, abandoned=abandoned)
        if report.requeued or report.abandoned:
            logger.info(
                "frontier: recovered job {}: {} re-queued, {} abandoned",
                self._job.job_id,
                report.requeued,
                report.abandoned,
            )
        return report

    async def _abandon_exhausted(self, condition: dict[str, Any], now: datetime) -> int:
        """Fail in-progress pages that already used every attempt (`worker_lost`)."""
        detail = f"crawler never reported back after {self.max_attempts} attempt(s)"
        try:
            result = await self._collection.update_many(
                {
                    "job_id": self._job.job_id,
                    "status": PageStatus.IN_PROGRESS.value,
                    "attempts": {"$gte": self.max_attempts},
                    **condition,
                },
                {
                    "$set": {
                        "status": PageStatus.FAILED.value,
                        "reason": PageReason.WORKER_LOST.value,
                        "reason_detail": detail,
                        "claimed_by": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    }
                },
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return result.modified_count

    # ---- progress ---- #

    async def stats(self) -> FrontierStats:
        now = self._clock()
        job = self._job.job_id
        queued = PageStatus.QUEUED.value
        try:
            ready = await self._collection.count_documents(self._ready(now))
            delayed = await self._collection.count_documents(
                {"job_id": job, "status": queued, "retry_at": {"$gt": now}}
            )
            in_flight = await self._collection.count_documents(
                {"job_id": job, "status": PageStatus.IN_PROGRESS.value}
            )
            nxt = await self._collection.find_one(
                {"job_id": job, "status": queued, "retry_at": {"$gt": now}},
                projection={"retry_at": 1},
                sort=[("retry_at", ASCENDING)],
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return FrontierStats(ready, delayed, in_flight, nxt["retry_at"] if nxt else None)
