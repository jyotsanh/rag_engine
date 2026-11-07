"""Builds a job's `CrawlReport` from its page table (Step 53).

The page rows are the truth: every page that failed or was skipped carries a classified
`PageReason`, so the report is a breakdown of them plus what only the run knows (the
anchored seed, why it stopped, robots/sitemap verdicts). The engine calls this at the end
of each run; the worker can call it for a job that failed before the engine finished.
"""

from rag_engine.crawling.frontier import RecoveryReport
from rag_engine.crawling.sitemap import SitemapReport
from rag_engine.domain import CrawlReport, CrawlStopReason, PageStatus
from rag_engine.repositories import PageRepository


async def build_report(
    pages: PageRepository,
    job_id: str,
    *,
    page_counts: dict[PageStatus, int] | None = None,
    seed_url: str | None = None,
    stop_reason: CrawlStopReason | None = None,
    stop_detail: str | None = None,
    robots_status: str | None = None,
    robots_detail: str | None = None,
    sitemap: SitemapReport | None = None,
    sitemap_error: str | None = None,
    skipped_links: int = 0,
    skipped_links_dropped: int = 0,
    recovered: RecoveryReport | None = None,
) -> CrawlReport:
    """`page_counts` may be passed when the caller already counted pages by status."""
    counts = page_counts if page_counts is not None else await pages.count_by_status(job_id)
    breakdown = await pages.reason_breakdown(job_id)
    return CrawlReport(
        seed_url=seed_url,
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        pages=dict(counts),
        failures=[row for row in breakdown if row.status is PageStatus.FAILED],
        skips=[row for row in breakdown if row.status is PageStatus.SKIPPED],
        robots_status=robots_status,
        robots_detail=robots_detail,
        sitemap_urls=len(sitemap.entries) if sitemap is not None else None,
        sitemap_error=sitemap_error,
        skipped_links=skipped_links,
        skipped_links_dropped=skipped_links_dropped,
        recovered_pages=recovered.requeued if recovered else 0,
        abandoned_pages=recovered.abandoned if recovered else 0,
    )
