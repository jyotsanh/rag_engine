"""Tenant endpoints for knowledge bases. Every query is confined to the caller's tenant."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, status

from rag_engine.api.deps import CurrentTenant, KnowledgeBases, Tasks
from rag_engine.api.params import Cursor, Limit
from rag_engine.api.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBasePatch,
    KnowledgeBaseResponse,
)
from rag_engine.domain.types import KnowledgeBaseId
from rag_engine.exceptions import ErrorResponse
from rag_engine.repositories import DEFAULT_PAGE_SIZE, Page
from rag_engine.services import KnowledgeBaseService

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 404, 409)
}

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"], responses=_ERRORS)


def get_service(kbs: KnowledgeBases, tasks: Tasks) -> KnowledgeBaseService:
    return KnowledgeBaseService(kbs, tasks)


Service = Annotated[KnowledgeBaseService, Depends(get_service)]
KbPath = Annotated[KnowledgeBaseId, Path(description="Knowledge base id, e.g. `kb_…`.")]


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a knowledge base (its embedding model is fixed for life)",
)
async def create_knowledge_base(
    body: KnowledgeBaseCreate, ctx: CurrentTenant, service: Service
) -> KnowledgeBaseResponse:
    kb = await service.create(
        ctx.tenant,
        name=body.name,
        description=body.description,
        embedding_model=body.embedding_model,
        embedding_dimensions=body.embedding_dimensions,
    )
    return KnowledgeBaseResponse.of(kb)


@router.get("", summary="List knowledge bases (newest first)")
async def list_knowledge_bases(
    ctx: CurrentTenant,
    service: Service,
    limit: Limit = DEFAULT_PAGE_SIZE,
    cursor: Cursor = None,
) -> Page[KnowledgeBaseResponse]:
    page = await service.list(ctx.tenant, limit=limit, cursor=cursor)
    return Page[KnowledgeBaseResponse](
        items=[KnowledgeBaseResponse.of(kb) for kb in page.items], next_cursor=page.next_cursor
    )


@router.get("/{kb_id}", summary="Get a knowledge base")
async def get_knowledge_base(
    kb_id: KbPath, ctx: CurrentTenant, service: Service
) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse.of(await service.get(ctx.tenant, kb_id))


@router.patch("/{kb_id}", summary="Rename or re-describe a knowledge base")
async def update_knowledge_base(
    kb_id: KbPath, body: KnowledgeBasePatch, ctx: CurrentTenant, service: Service
) -> KnowledgeBaseResponse:
    changes = body.model_dump(exclude_unset=True)
    return KnowledgeBaseResponse.of(await service.update(ctx.tenant, kb_id, changes))


@router.delete(
    "/{kb_id}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Delete a knowledge base",
    description="Hidden immediately and its name freed; its sources, pages and vectors are "
    "removed asynchronously.",
)
async def delete_knowledge_base(
    kb_id: KbPath, ctx: CurrentTenant, service: Service
) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse.of(await service.delete(ctx.tenant, kb_id))
