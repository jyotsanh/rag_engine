"""Bodies for starting a crawl (Step 84) and the crawl job views (Steps 84-85)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.config import JsRenderMode, settings
from rag_engine.domain import (
    MAX_PATTERNS,
    CrawlConfig,
    CrawlJob,
    CrawlReport,
    EmbeddingUsage,
    JobCounters,
    JobError,
    JobStatus,
    WebhookAttempt,
    WebhookStatus,
    utc_now,
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


class WebhookView(BaseModel):
    """The completion webhook's delivery (Step 83)."""

    status: WebhookStatus
    attempts: int
    delivered_at: datetime | None
    log: list[WebhookAttempt] = Field(description="The latest attempts, oldest first.")


class CrawlJobDetailResponse(CrawlJobResponse):
    """`GET /jobs/{id}`: the summary plus the crawl report and the webhook's delivery."""

    duration_seconds: float | None = Field(
        description="Since the first start: until it finished, or until now while it runs."
    )
    report: CrawlReport | None = Field(
        description="Written at the end of each crawl run: pages by status, failures and "
        "skips grouped by reason with sample URLs, and why the crawl stopped. Null until a "
        "run's crawl phase ends; meanwhile `counters` show live progress."
    )
    webhook: WebhookView | None = Field(
        description="Only for jobs with a `webhook_url`, once they have finished."
    )

    @classmethod
    def of(cls, job: CrawlJob) -> CrawlJobDetailResponse:
        summary = CrawlJobResponse.of(job).model_dump()
        duration = None
        if job.started_at is not None:
            duration = ((job.finished_at or utc_now()) - job.started_at).total_seconds()
        webhook = None
        if job.webhook is not None:
            webhook = WebhookView(
                status=job.webhook.status,
                attempts=job.webhook.attempts,
                delivered_at=job.webhook.delivered_at,
                log=job.webhook.log,
            )
        return cls(**summary, duration_seconds=duration, report=job.report, webhook=webhook)
