"""Reading crawl jobs (Step 85) and their per-page report (Step 86).

Every read is confined to the caller's tenant: another tenant's job id is a 404, the same
as an unknown one. `DELETE /jobs/{id}` cancels a job (Step 87).
"""

from typing import Annotated, Any

from fastapi import APIRouter, Path, Query, Response
from fastapi import status as http_status

from rag_engine.api.deps import CancelFlags, CurrentTenant, Database, Jobs, Tasks
from rag_engine.api.params import Cursor, Limit
from rag_engine.api.schemas import (
    CrawlJobDetailResponse,
    CrawlJobResponse,
    PageDetailResponse,
    PageResponse,
)
from rag_engine.crawling.cancellation import request_cancel
from rag_engine.domain import JobStatus, PageReason, PageStatus
from rag_engine.domain.types import JobId, KnowledgeBaseId, PageId, SourceId
from rag_engine.exceptions import ErrorResponse
from rag_engine.repositories import DEFAULT_PAGE_SIZE, ChunkRepository, Page, PageRepository

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 404)
}
_CANCEL_ERRORS: dict[int | str, dict[str, Any]] = {
    409: {"model": ErrorResponse, "description": "Finalizing or already finished"},
    202: {"model": CrawlJobDetailResponse, "description": "Running: cancellation requested"},
}

router = APIRouter(prefix="/jobs", tags=["jobs"], responses=_ERRORS)
JobPath = Annotated[JobId, Path(description="Crawl job id, e.g. `job_…`.")]
PagePath = Annotated[PageId, Path(description="Page id, e.g. `page_…`.")]


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


@router.get(
    "/{job_id}/pages",
    summary="List a crawl job's pages (the per-page report, in discovery order)",
    description="Every URL the job saw with its outcome: `succeeded`, or `skipped` / "
    "`failed` / `duplicate` with a stable `reason` code (e.g. `robots_disallowed`, "
    "`out_of_scope`, `timeout`, `http_client_error`) and a human `reason_detail`. Filter "
    "with `status` and/or `reason` (each repeatable). The Markdown is left out; get one "
    "page for it.",
)
async def list_job_pages(
    job_id: JobPath,
    ctx: CurrentTenant,
    jobs: Jobs,
    db: Database,
    status: Annotated[
        list[PageStatus] | None, Query(description="Only these statuses (repeatable).")
    ] = None,
    reason: Annotated[
        list[PageReason] | None, Query(description="Only these reasons (repeatable).")
    ] = None,
    limit: Limit = DEFAULT_PAGE_SIZE,
    cursor: Cursor = None,
) -> Page[PageResponse]:
    job = await jobs.get_for_tenant(ctx.tenant.id, job_id)  # 404 for another tenant's job
    page = await PageRepository(db).list_for_job(
        ctx.tenant.id, job.id, status=status, reason=reason, limit=limit, cursor=cursor
    )
    return Page[PageResponse](
        items=[PageResponse.from_summary(summary) for summary in page.items],
        next_cursor=page.next_cursor,
    )


@router.get(
    "/{job_id}/pages/{page_id}",
    summary="Get one page of a crawl job, with its extracted Markdown",
)
async def get_job_page(
    job_id: JobPath, page_id: PagePath, ctx: CurrentTenant, jobs: Jobs, db: Database
) -> PageDetailResponse:
    job = await jobs.get_for_tenant(ctx.tenant.id, job_id)
    page = await PageRepository(db).get_for_job(ctx.tenant.id, job.id, page_id)
    chunks = await ChunkRepository(db).count({"page_id": page.id})
    return PageDetailResponse.of(page, chunks=chunks)


@router.delete(
    "/{job_id}",
    summary="Cancel a crawl job",
    description=(
        "- **Queued**: cancelled at once (**200**).\n"
        "- **Running**: cancellation is requested (**202**); the worker stops fetching, "
        "lets in-flight pages finish and ends the job `cancelled` within seconds. Poll "
        "`GET /jobs/{id}`.\n"
        "- **Already cancelled**: returned as is (**200**), so retries are safe.\n"
        "- **Finalizing or finished**: **409** `invalid_job_transition` (the version swap "
        "isn't cancellable).\n\n"
        "A cancelled crawl never replaces the source's content: the previous version keeps "
        "serving search, and whatever the cancelled crawl stored is deleted."
    ),
    responses=_CANCEL_ERRORS,
)
async def cancel_job(
    job_id: JobPath,
    ctx: CurrentTenant,
    jobs: Jobs,
    flags: CancelFlags,
    tasks: Tasks,
    response: Response,
) -> CrawlJobDetailResponse:
    before = await jobs.get_for_tenant(ctx.tenant.id, job_id)
    job = await request_cancel(jobs, flags, ctx.tenant.id, job_id)
    if job.status is JobStatus.RUNNING:
        response.status_code = http_status.HTTP_202_ACCEPTED
    elif before.status is JobStatus.QUEUED and job.webhook is not None:
        # ended right here, not by a worker: send its webhook now (the sweep would later)
        await tasks.enqueue_job_webhook(job.id)
    return CrawlJobDetailResponse.of(job)
