from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator
from pymongo import IndexModel

from rag_engine.config import settings
from rag_engine.domain.base import MongoDocument
from rag_engine.domain.enums import JobStatus
from rag_engine.domain.types import (
    HttpUrlStr,
    KnowledgeBaseId,
    NonNegativeInt,
    PathPattern,
    SourceId,
    TenantId,
)

MAX_PATTERNS = 50


class CrawlConfig(BaseModel):
    """Per-job crawl limits, already clamped to the tenant's quotas by the API (Step 84)."""

    max_pages: int = Field(
        default_factory=lambda: settings.CRAWL_DEFAULT_MAX_PAGES,
        gt=0,
        le=settings.CRAWL_HARD_MAX_PAGES,
    )
    max_depth: int = Field(
        default_factory=lambda: settings.CRAWL_DEFAULT_MAX_DEPTH,
        ge=0,
        le=settings.CRAWL_HARD_MAX_DEPTH,
    )
    include_patterns: list[PathPattern] = Field(default_factory=list, max_length=MAX_PATTERNS)
    exclude_patterns: list[PathPattern] = Field(default_factory=list, max_length=MAX_PATTERNS)
    webhook_url: HttpUrlStr | None = None


class JobCounters(BaseModel):
    """Updated with atomic `$inc` while the job runs (Step 49)."""

    discovered: NonNegativeInt = 0
    fetched: NonNegativeInt = 0
    failed: NonNegativeInt = 0
    skipped: NonNegativeInt = 0
    duplicates: NonNegativeInt = 0
    chunked: NonNegativeInt = 0
    embedded: NonNegativeInt = 0


class EmbeddingUsage(BaseModel):
    requests: NonNegativeInt = 0
    tokens: NonNegativeInt = 0


class JobError(BaseModel):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(max_length=2000)


class CrawlJob(MongoDocument):
    """One crawl of one source. The job id doubles as the vector `version` it writes."""

    collection_name = "crawl_jobs"
    id_prefix = "job"
    indexes = (
        IndexModel([("tenant_id", 1), ("_id", -1)], name="tenant_jobs"),
        IndexModel([("source_id", 1), ("_id", -1)], name="source_jobs"),
        IndexModel([("status", 1), ("heartbeat_at", 1)], name="status_heartbeat"),
    )

    tenant_id: TenantId
    kb_id: KnowledgeBaseId
    source_id: SourceId
    seed_url: HttpUrlStr
    status: JobStatus = JobStatus.QUEUED
    config: CrawlConfig = Field(default_factory=CrawlConfig)
    counters: JobCounters = Field(default_factory=JobCounters)
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)
    error: JobError | None = None
    attempts: NonNegativeInt = 0
    cancel_requested_at: datetime | None = None
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def version(self) -> str:
        return self.id

    @model_validator(mode="after")
    def _consistent_status(self) -> Self:
        if self.status is JobStatus.FAILED and self.error is None:
            raise ValueError("a failed job must carry an error")
        if self.status.is_terminal and self.finished_at is None:
            raise ValueError("a finished job must have finished_at")
        if not self.status.is_terminal and self.finished_at is not None:
            raise ValueError("finished_at is only set on terminal jobs")
        return self
