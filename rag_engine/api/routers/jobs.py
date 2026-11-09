"""Reading crawl jobs (Step 85): one job in detail, or the tenant's jobs newest first.

Every read is confined to the caller's tenant: another tenant's job id is a 404, the same
as an unknown one. The per-page report is Step 86, cancelling Step 87.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query

from rag_engine.api.deps import CurrentTenant, Jobs
from rag_engine.api.params import Cursor, Limit
from rag_engine.api.schemas import CrawlJobDetailResponse, CrawlJobResponse
from rag_engine.domain import JobStatus
from rag_engine.domain.types import JobId, KnowledgeBaseId, SourceId
from rag_engine.exceptions import ErrorResponse
from rag_engine.repositories import DEFAULT_PAGE_SIZE, Page

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 404)
}

router = APIRouter(prefix="/jobs", tags=["jobs"], responses=_ERRORS)
JobPath = Annotated[JobId, Path(description="Crawl job id, e.g. `job_…`.")]


@router.get("", summary="List crawl jobs (newest first)")
async def list_jobs(
    ctx: CurrentTenant,
    jobs: Jobs,
    kb_id: Annotated[KnowledgeBaseId | None, Query(description="Only this knowledge base.")] = None,
    source_id: Annotated[SourceId | None, Query(description="Only this source.")] = None,
    status: Annotated[
        list[JobStatus] | None,
        Query(description="Only these statuses (repeat the parameter for several)."),
    ] = None,
    limit: Limit = DEFAULT_PAGE_SIZE,
    cursor: Cursor = None,
) -> Page[CrawlJobResponse]:
    page = await jobs.list_for_tenant(
        ctx.tenant.id, status=status, kb_id=kb_id, source_id=source_id, limit=limit,
        cursor=cursor,
    )  # fmt: skip
    return Page[CrawlJobResponse](
        items=[CrawlJobResponse.of(job) for job in page.items], next_cursor=page.next_cursor
    )


@router.get(
    "/{job_id}",
    summary="Get a crawl job: status, live counters, usage, report, webhook delivery",
)
async def get_job(job_id: JobPath, ctx: CurrentTenant, jobs: Jobs) -> CrawlJobDetailResponse:
    return CrawlJobDetailResponse.of(await jobs.get_for_tenant(ctx.tenant.id, job_id))
