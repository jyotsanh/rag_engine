"""Tenant-facing knowledge base lifecycle (backs `/knowledge-bases`)."""

from typing import Any

from rag_engine.config import logger
from rag_engine.config.enums import EmbeddingModelName
from rag_engine.domain import KnowledgeBase, Tenant
from rag_engine.embeddings import resolve_space
from rag_engine.exceptions import AppValidationError, QuotaExceededError
from rag_engine.repositories import Page
from rag_engine.repositories.knowledge_bases import KnowledgeBaseRepository
from rag_engine.worker.producer import TaskQueue


class KnowledgeBaseService:
    def __init__(self, kbs: KnowledgeBaseRepository, tasks: TaskQueue | None = None) -> None:
        self._kbs = kbs
        self._tasks = tasks

    def _quota_error(self, tenant: Tenant) -> QuotaExceededError:
        return QuotaExceededError(
            "Knowledge base limit reached; delete one first",
            details={"quota": "max_knowledge_bases", "limit": tenant.quotas.max_knowledge_bases},
        )

    async def create(
        self,
        tenant: Tenant,
        *,
        name: str,
        description: str = "",
        embedding_model: EmbeddingModelName | str | None = None,
        embedding_dimensions: int | None = None,
    ) -> KnowledgeBase:
        """The embedding space is fixed here for the KB's whole life.

        Model: explicit choice, else the tenant's default. Dimensions: explicit
        (validated for the model, typed 422 with min/max), else the model's native size.
        """
        space = resolve_space(
            embedding_model or tenant.default_embedding_model, embedding_dimensions
        )
        limit = tenant.quotas.max_knowledge_bases
        if await self._kbs.count_active(tenant.id) >= limit:
            raise self._quota_error(tenant)

        kb = await self._kbs.create(
            KnowledgeBase(
                tenant_id=tenant.id,
                name=name,
                description=description,
                embedding_model=space.model,
                embedding_dimensions=space.dimensions,
            )
        )
        # count-then-insert can race; re-check and roll back our own insert if we overshot
        if await self._kbs.count_active(tenant.id) > limit:
            await self._kbs.remove(kb.id, scope={"tenant_id": tenant.id})
            raise self._quota_error(tenant)

        logger.bind(kb_id=kb.id, space=space.key).info("created knowledge base {}", kb.name)
        return kb

    async def get(self, tenant: Tenant, kb_id: str) -> KnowledgeBase:
        return await self._kbs.get_active(kb_id, tenant.id)

    async def list(self, tenant: Tenant, *, limit: int, cursor: str | None) -> Page[KnowledgeBase]:
        return await self._kbs.list_active(tenant.id, limit=limit, cursor=cursor)

    async def update(self, tenant: Tenant, kb_id: str, changes: dict[str, Any]) -> KnowledgeBase:
        """Only name and description are editable; the embedding space is locked."""
        locked = {"embedding_model", "embedding_dimensions", "status", "tenant_id"} & set(changes)
        if locked:
            raise AppValidationError(
                "The embedding space and lifecycle fields cannot be changed",
                details={"fields": sorted(locked)},
            )
        return await self._kbs.update_active(kb_id, tenant.id, changes)

    async def delete(self, tenant: Tenant, kb_id: str) -> KnowledgeBase:
        """Marks the KB deleting (hidden immediately, name freed) and enqueues the cascade
        of its sources, pages, chunks and vectors (Step 81)."""
        kb = await self._kbs.mark_deleting(kb_id, tenant.id)
        logger.bind(kb_id=kb.id).info("knowledge base marked for deletion")
        if self._tasks is not None:
            await self._tasks.delete_knowledge_base(tenant.id, kb.id)
        return kb
