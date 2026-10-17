"""Sources inside a knowledge base. Created by submitting a crawl (Step 84)."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, status

from rag_engine.api.deps import CurrentTenant, KnowledgeBases, Sources
from rag_engine.api.params import Cursor, Limit
from rag_engine.api.schemas import SourceResponse
from rag_engine.domain.types import KnowledgeBaseId, SourceId
from rag_engine.exceptions import ErrorResponse
from rag_engine.repositories import DEFAULT_PAGE_SIZE, Page
from rag_engine.services import SourceService

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 404)
}

router = APIRouter(prefix="/knowledge-bases/{kb_id}/sources", tags=["sources"], responses=_ERRORS)


def get_service(kbs: KnowledgeBases, sources: Sources) -> SourceService:
    return SourceService(kbs, sources)


Service = Annotated[SourceService, Depends(get_service)]
KbPath = Annotated[KnowledgeBaseId, Path(description="Knowledge base id, e.g. `kb_…`.")]
SourcePath = Annotated[SourceId, Path(description="Source id, e.g. `src_…`.")]


@router.get("", summary="List a knowledge base's sources (newest first)")
async def list_sources(
    kb_id: KbPath,
    ctx: CurrentTenant,
    service: Service,
    limit: Limit = DEFAULT_PAGE_SIZE,
    cursor: Cursor = None,
) -> Page[SourceResponse]:
    page = await service.list(ctx.tenant, kb_id, limit=limit, cursor=cursor)
    return Page[SourceResponse](
        items=[SourceResponse.of(s) for s in page.items], next_cursor=page.next_cursor
    )


@router.get("/{source_id}", summary="Get a source")
async def get_source(
    kb_id: KbPath, source_id: SourcePath, ctx: CurrentTenant, service: Service
) -> SourceResponse:
    return SourceResponse.of(await service.get(ctx.tenant, kb_id, source_id))


@router.delete(
    "/{source_id}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Delete a source",
    description="Hidden immediately and its seed URL freed; its pages, chunks and vectors "
    "are removed asynchronously.",
)
async def delete_source(
    kb_id: KbPath, source_id: SourcePath, ctx: CurrentTenant, service: Service
) -> SourceResponse:
    return SourceResponse.of(await service.delete(ctx.tenant, kb_id, source_id))
