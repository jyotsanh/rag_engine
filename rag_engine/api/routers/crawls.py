"""Starting crawls (Step 84). Reading, listing and cancelling jobs: Steps 85-87."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Response, status

from rag_engine.api.deps import CurrentTenant, Database, KnowledgeBases, Sources, Tasks
from rag_engine.api.schemas import CrawlCreate, CrawlJobResponse
from rag_engine.config import settings
from rag_engine.domain.types import KnowledgeBaseId
from rag_engine.exceptions import ErrorResponse
from rag_engine.repositories import JobRepository
from rag_engine.services import CrawlJobService, CrawlSubmissionService

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 404)
}

router = APIRouter(prefix="/knowledge-bases/{kb_id}/crawls", tags=["crawls"], responses=_ERRORS)


def get_service(
    db: Database, kbs: KnowledgeBases, sources: Sources, tasks: Tasks
) -> CrawlSubmissionService:
    return CrawlSubmissionService(kbs, sources, CrawlJobService(JobRepository(db), sources), tasks)


Service = Annotated[CrawlSubmissionService, Depends(get_service)]
KbPath = Annotated[KnowledgeBaseId, Path(description="Knowledge base id, e.g. `kb_…`.")]


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a crawl of a website into a knowledge base",
    description=(
        "Queues a crawl of `url` (and its sitemap) within the scope given. The seed URL is "
        "normalized and identifies the knowledge base's **source**: re-crawling it replaces "
        "that source's content atomically when the new crawl succeeds.\n\n"
        "- **202**: a new crawl job was queued.\n"
        "- **200**: this source is already being crawled; that job is returned unchanged "
        "(its own settings apply).\n\n"
        "Refused with 403 `quota_exceeded` past the tenant's concurrent-job or page quota, "
        "and with 422 for a URL that is invalid, not public (`url_not_allowed`) or doesn't "
        "resolve, or a `webhook_url` without a tenant webhook secret."
    ),
    responses={200: {"model": CrawlJobResponse, "description": "Already being crawled"}},
)
async def start_crawl(
    kb_id: KbPath, body: CrawlCreate, ctx: CurrentTenant, service: Service, response: Response
) -> CrawlJobResponse:
    config = body.to_config(service.resolve_max_pages(ctx.tenant, body.max_pages))
    submission = await service.submit(ctx.tenant, kb_id, body.url, config)
    if not submission.created:
        response.status_code = status.HTTP_200_OK
    response.headers["Location"] = f"{settings.API_V1_STR}/jobs/{submission.job.id}"
    return CrawlJobResponse.of(submission.job)
