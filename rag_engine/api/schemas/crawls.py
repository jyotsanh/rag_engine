"""Bodies for starting a crawl (Step 84) and the crawl job representation it returns."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.config import JsRenderMode, settings
from rag_engine.domain import (
    MAX_PATTERNS,
    CrawlConfig,
    CrawlJob,
    EmbeddingUsage,
    JobCounters,
    JobError,
    JobStatus,
)
from rag_engine.domain.types import HttpUrlStr, PathPattern, PathPrefix


class CrawlCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        min_length=1,
        max_length=2048,
        description="Seed URL. Normalized; one source per normalized seed in a knowledge base.",
        examples=["https://docs.acme.example/"],
    )
    max_pages: int | None = Field(
        default=None,
        gt=0,
        le=settings.CRAWL_HARD_MAX_PAGES,
        description="Page cap. Defaults to the service default, within the tenant's quota; "
        "above the quota is refused (403).",
    )
    max_depth: int = Field(
        default_factory=lambda: settings.CRAWL_DEFAULT_MAX_DEPTH,
        ge=0,
        le=settings.CRAWL_HARD_MAX_DEPTH,
        description="Link hops from the seed (sitemap URLs count as depth 0).",
    )
    include_subdomains: bool = False
    path_prefix: PathPrefix | None = Field(default=None, examples=["/docs/"])
    include_patterns: list[PathPattern] = Field(
        default_factory=list, max_length=MAX_PATTERNS, examples=[["/docs/**"]]
    )
    exclude_patterns: list[PathPattern] = Field(
        default_factory=list, max_length=MAX_PATTERNS, examples=[["/docs/archive/**"]]
    )
    webhook_url: HttpUrlStr | None = Field(
        default=None,
        description="POSTed a signed `job.<status>` event when the crawl ends (see the "
        "webhook docs). Requires a webhook secret on the tenant.",
    )
    js_rendering: JsRenderMode | None = Field(
        default=None, description="Override the service's JS rendering mode for this crawl."
    )

    def to_config(self, max_pages: int) -> CrawlConfig:
        return CrawlConfig(
            max_pages=max_pages,
            **self.model_dump(exclude={"url", "max_pages"}),
        )


class CrawlJobResponse(BaseModel):
    id: str = Field(description="Job id; also the `version` its content is stored under.")
    kb_id: str
    source_id: str
    seed_url: str
    status: JobStatus
    config: CrawlConfig
    counters: JobCounters
    usage: EmbeddingUsage
    error: JobError | None
    attempts: int
    cancel_requested_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, job: CrawlJob) -> CrawlJobResponse:
        return cls.model_validate(job.model_dump(include=set(cls.model_fields)))
