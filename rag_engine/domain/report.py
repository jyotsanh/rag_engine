"""The job's crawl report: what happened to its pages, failures grouped by reason (Step 53).

Written by the crawl engine at the end of each run (and by the worker for a job that
failed, Step 77), stored on the job document, and served as the job's failure summary
(`GET /jobs/{id}`, Step 85). The full per-page list is the pages API (Step 86); the report
keeps a few sample URLs per reason so the summary is useful on its own.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from rag_engine.domain.base import utc_now
from rag_engine.domain.enums import CrawlStopReason, PageReason, PageStatus
from rag_engine.domain.types import NonNegativeInt

REPORT_SAMPLES_PER_REASON = 5


class PageSample(BaseModel):
    url: str
    detail: str | None = Field(default=None, max_length=500)
    http_status: int | None = None


class ReasonCount(BaseModel):
    """Pages that ended with one (status, reason); `samples` are the first ones found."""

    status: PageStatus
    reason: PageReason
    count: NonNegativeInt
    samples: list[PageSample] = Field(default_factory=list, max_length=REPORT_SAMPLES_PER_REASON)


class CrawlReport(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    seed_url: str | None = None  # where the crawl was anchored (after a seed redirect)
    stop_reason: CrawlStopReason | None = None  # None: the frontier drained
    stop_detail: str | None = None
    pages: dict[PageStatus, NonNegativeInt] = Field(default_factory=dict)
    failures: list[ReasonCount] = Field(default_factory=list)  # failed pages, largest first
    skips: list[ReasonCount] = Field(default_factory=list)  # skipped pages, largest first
    robots_status: str | None = None  # the seed origin's robots.txt verdict
    robots_detail: str | None = None
    sitemap_urls: NonNegativeInt | None = None  # URLs the sitemaps listed (None: not read)
    sitemap_error: str | None = None
    skipped_links: NonNegativeInt = 0  # never-crawled links recorded as skipped pages
    skipped_links_dropped: NonNegativeInt = 0  # non-http(s) links / over the per-page cap
    recovered_pages: NonNegativeInt = 0  # this run re-queued from a crashed run
    abandoned_pages: NonNegativeInt = 0  # this run failed as worker_lost

    @property
    def failed(self) -> int:
        return self.pages.get(PageStatus.FAILED, 0)

    @property
    def unfinished(self) -> int:
        """Pages still queued or in progress: 0 for every run that ended normally."""
        return self.pages.get(PageStatus.QUEUED, 0) + self.pages.get(PageStatus.IN_PROGRESS, 0)
