from datetime import datetime
from typing import Self

from pydantic import Field, model_validator
from pymongo import IndexModel

from rag_engine.domain.base import MongoDocument
from rag_engine.domain.enums import ContentKind, PageReason, PageStatus, RenderedWith
from rag_engine.domain.types import (
    HttpUrlStr,
    JobId,
    KnowledgeBaseId,
    NonNegativeInt,
    PageId,
    Sha256Hex,
    SourceId,
    TenantId,
)

_NEEDS_REASON = {PageStatus.SKIPPED, PageStatus.FAILED}


class Page(MongoDocument):
    """One URL seen during one crawl job, with its outcome and extracted Markdown."""

    collection_name = "pages"
    id_prefix = "page"
    indexes = (
        IndexModel([("job_id", 1), ("url_hash", 1)], unique=True, name="job_url_unique"),
        IndexModel([("job_id", 1), ("status", 1), ("_id", 1)], name="job_status"),
    )

    tenant_id: TenantId
    kb_id: KnowledgeBaseId
    source_id: SourceId
    job_id: JobId
    url: HttpUrlStr
    url_hash: Sha256Hex
    depth: NonNegativeInt = 0
    discovered_from: HttpUrlStr | None = None
    status: PageStatus = PageStatus.QUEUED
    reason: PageReason | None = None
    reason_detail: str | None = Field(default=None, max_length=500)
    http_status: int | None = Field(default=None, ge=100, le=599)
    final_url: HttpUrlStr | None = None  # after redirects
    content_type: str | None = Field(default=None, max_length=200)
    content_kind: ContentKind | None = None
    rendered_with: RenderedWith | None = None
    js_signals: list[str] = Field(default_factory=list)  # JS-shell detector (Step 36)
    render_note: str | None = Field(default=None, max_length=500)  # why a render wasn't used
    extractor: str | None = Field(default=None, max_length=50)
    title: str | None = Field(default=None, max_length=1000)
    description: str | None = Field(default=None, max_length=2000)
    lang: str | None = Field(default=None, max_length=10)
    doc_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    markdown: str | None = None
    markdown_truncated: bool = False  # cut at PAGE_MAX_MARKDOWN_BYTES (Mongo's 16 MB limit)
    content_hash: Sha256Hex | None = None
    canonical_url: HttpUrlStr | None = None
    duplicate_of: PageId | None = None
    bytes_downloaded: NonNegativeInt = 0
    fetch_ms: NonNegativeInt | None = None  # HTTP fetch + browser render
    extract_ms: NonNegativeInt | None = None
    attempts: NonNegativeInt = 0
    fetched_at: datetime | None = None

    @model_validator(mode="after")
    def _consistent_outcome(self) -> Self:
        if self.status in _NEEDS_REASON and self.reason is None:
            raise ValueError(f"a {self.status} page needs a reason")
        if self.status not in _NEEDS_REASON and self.reason is not None:
            raise ValueError("reason is only set on skipped or failed pages")
        if self.status is PageStatus.DUPLICATE and self.duplicate_of is None:
            raise ValueError("a duplicate page must reference duplicate_of")
        if self.status is PageStatus.SUCCEEDED and (
            self.content_kind is None or self.markdown is None
        ):
            raise ValueError("a succeeded page needs content_kind and markdown")
        return self
