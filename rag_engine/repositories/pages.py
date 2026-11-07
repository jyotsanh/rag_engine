"""Pages of crawl jobs: one document per `(job_id, url_hash)`.

The crawl engine inserts discovered URLs (the unique index de-duplicates them, so
inserting a URL twice is harmless and race-safe), and records each page's outcome as a
`PageOutcome`. An outcome always sets **every** outcome field, so re-recording a retried
page cannot leave a stale `reason` or Markdown behind, and it is validated against the
`Page` model's consistency rules before anything is written.

Markdown above `PAGE_MAX_MARKDOWN_BYTES` is cut at the last block boundary that fits
(`markdown_truncated=True`): MongoDB rejects documents over 16 MB.

Reads used by the pipeline: counts by status / reason (job summary, Step 53), succeeded
pages in discovery order (ingest, Step 40's keeper order) and tenant-scoped pagination
for the API.
"""

from collections import Counter
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from pymongo import ASCENDING
from pymongo.errors import BulkWriteError, PyMongoError

from rag_engine.config import Settings, settings
from rag_engine.domain import (
    ContentKind,
    MongoDocument,
    Page,
    PageReason,
    PageStatus,
    RenderedWith,
)
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.infrastructure.mongo import MongoDatabase
from rag_engine.repositories.base import BaseRepository
from rag_engine.repositories.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    PageNotFoundError,
)
from rag_engine.repositories.pagination import DEFAULT_PAGE_SIZE
from rag_engine.repositories.pagination import Page as ResultPage

_DUPLICATE_KEY = 11000


@dataclass(frozen=True, slots=True, kw_only=True)
class PageOutcome:
    """Everything a crawl decides about a page. Unset fields are written as cleared."""

    status: PageStatus
    reason: PageReason | None = None
    reason_detail: str | None = None
    http_status: int | None = None
    final_url: str | None = None
    content_type: str | None = None
    content_kind: ContentKind | None = None
    rendered_with: RenderedWith | None = None
    js_signals: tuple[str, ...] = ()
    render_note: str | None = None
    extractor: str | None = None
    title: str | None = None
    description: str | None = None
    lang: str | None = None
    doc_metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
    markdown: str | None = None
    content_hash: str | None = None
    canonical_url: str | None = None
    duplicate_of: str | None = None
    bytes_downloaded: int = 0
    fetch_ms: int | None = None
    extract_ms: int | None = None
    fetched_at: datetime | None = None

    def changes(self, max_markdown_bytes: int) -> dict[str, Any]:
        values = asdict(self)
        values["js_signals"] = list(self.js_signals)
        values["doc_metadata"] = dict(self.doc_metadata)
        values["reason_detail"] = (self.reason_detail or "")[:500] or None
        markdown, truncated = truncate_markdown(self.markdown, max_markdown_bytes)
        values["markdown"], values["markdown_truncated"] = markdown, truncated
        # the page is done: drop its frontier lease/retry time (`claimed_by` stays as an audit)
        values["lease_expires_at"] = values["retry_at"] = None
        return values


def truncate_markdown(markdown: str | None, max_bytes: int) -> tuple[str | None, bool]:
    """Cut at the last blank-line block boundary that fits in `max_bytes` (UTF-8)."""
    if markdown is None:
        return None, False
    encoded = markdown.encode("utf-8")
    if len(encoded) <= max_bytes:
        return markdown, False
    head = encoded[:max_bytes].decode("utf-8", errors="ignore")
    boundary = head.rfind("\n\n")
    cut = head[:boundary] if boundary > 0 else head
    return cut.rstrip() + "\n", True


class PageRepository(BaseRepository[Page]):
    model: ClassVar[type[MongoDocument]] = Page
    not_found_error: ClassVar[type[DocumentNotFoundError]] = PageNotFoundError

    def __init__(self, db: MongoDatabase, config: Settings = settings) -> None:
        super().__init__(db)
        self._max_markdown_bytes = config.PAGE_MAX_MARKDOWN_BYTES

    # ---- discovery ---- #

    async def add_discovered(self, page: Page) -> tuple[Page, bool]:
        """Insert a newly discovered URL. `(page, created)`: the existing page on a repeat."""
        try:
            return await self.insert(page), True
        except DuplicateDocumentError:
            existing = await self.get_by_url_hash(page.job_id, page.url_hash)
            if existing is None:  # deleted in between (job cleaned up): give up quietly
                raise
            return existing, False

    async def add_many_discovered(self, pages: Iterable[Page]) -> int:
        """Bulk insert, skipping URLs the job already has. Returns how many were new."""
        documents = [page.to_mongo() for page in pages]
        if not documents:
            return 0
        try:
            result = await self.collection.insert_many(documents, ordered=False)
        except BulkWriteError as exc:
            errors = exc.details.get("writeErrors", [])
            if any(error.get("code") != _DUPLICATE_KEY for error in errors):
                raise DatabaseUnavailableError(details={"reason": "BulkWriteError"}) from exc
            return int(exc.details.get("nInserted", 0))
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return len(result.inserted_ids)

    async def get_by_url_hash(self, job_id: str, url_hash: str) -> Page | None:
        return await self.find_one({"job_id": job_id, "url_hash": url_hash})

    # ---- outcomes ---- #

    async def record_outcome(self, page_id: str, outcome: PageOutcome) -> Page:
        """Write the whole outcome (validated as a `Page` first); returns the stored page."""
        return await self.update(page_id, outcome.changes(self._max_markdown_bytes))

    # ---- job-level reads ---- #

    async def count_by_status(self, job_id: str) -> dict[PageStatus, int]:
        rows = self.collection.aggregate(
            [{"$match": {"job_id": job_id}}, {"$group": {"_id": "$status", "n": {"$sum": 1}}}]
        )
        return {PageStatus(row["_id"]): row["n"] async for row in await rows}

    async def count_by_reason(self, job_id: str) -> Counter[PageReason]:
        rows = self.collection.aggregate(
            [
                {"$match": {"job_id": job_id, "reason": {"$ne": None}}},
                {"$group": {"_id": "$reason", "n": {"$sum": 1}}},
            ]
        )
        return Counter({PageReason(row["_id"]): row["n"] async for row in await rows})

    async def iter_succeeded(self, job_id: str, *, batch_size: int = 100) -> AsyncIterator[Page]:
        """Succeeded pages in discovery order (`_id` is time-ordered), streamed in batches."""
        cursor = self.collection.find(
            {"job_id": job_id, "status": PageStatus.SUCCEEDED.value},
            sort=[("_id", ASCENDING)],
            batch_size=batch_size,
        )
        async for raw in cursor:
            yield self._load(raw)

    async def list_for_job(
        self,
        tenant_id: str,
        job_id: str,
        *,
        status: PageStatus | None = None,
        reason: PageReason | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> ResultPage[Page]:
        """A job's pages for the API, oldest first, scoped to the tenant."""
        filter_: dict[str, Any] = {"job_id": job_id}
        if status is not None:
            filter_["status"] = status.value
        if reason is not None:
            filter_["reason"] = reason.value
        return await self.paginate(
            filter_, scope={"tenant_id": tenant_id}, limit=limit, cursor=cursor, order="asc"
        )

    async def delete_job(self, job_id: str) -> int:
        return await self.delete_many({"job_id": job_id})
