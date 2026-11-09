from collections.abc import Mapping
from typing import Any, ClassVar

from pymongo.errors import PyMongoError

from rag_engine.config import EmbeddingModelName
from rag_engine.domain import KnowledgeBase, MongoDocument, ResourceStatus, utc_now
from rag_engine.embeddings.models import EmbeddingSpace
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.repositories.base import BaseRepository, Filter
from rag_engine.repositories.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    KnowledgeBaseNameTakenError,
    KnowledgeBaseNotFoundError,
)
from rag_engine.repositories.pagination import DEFAULT_PAGE_SIZE, Page

_ACTIVE = {"status": ResourceStatus.ACTIVE.value}


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    """Tenant-owned: every query takes the tenant id. Deleting KBs are invisible to reads."""

    model: ClassVar[type[MongoDocument]] = KnowledgeBase
    not_found_error: ClassVar[type[DocumentNotFoundError]] = KnowledgeBaseNotFoundError

    @staticmethod
    def _scope(tenant_id: str) -> dict[str, str]:
        return {"tenant_id": tenant_id}

    async def create(self, kb: KnowledgeBase) -> KnowledgeBase:
        try:
            return await self.insert(kb)
        except DuplicateDocumentError as exc:
            raise KnowledgeBaseNameTakenError(details={"name": kb.name}) from exc

    async def get_active(self, kb_id: str, tenant_id: str) -> KnowledgeBase:
        kb = await self.find_one({"_id": kb_id, **_ACTIVE}, scope=self._scope(tenant_id))
        if kb is None:
            raise KnowledgeBaseNotFoundError(details={"id": kb_id})
        return kb

    async def list_active(
        self, tenant_id: str, *, limit: int = DEFAULT_PAGE_SIZE, cursor: str | None = None
    ) -> Page[KnowledgeBase]:
        return await self.paginate(
            _ACTIVE, scope=self._scope(tenant_id), limit=limit, cursor=cursor
        )

    async def count_active(self, tenant_id: str) -> int:
        return await self.count(_ACTIVE, scope=self._scope(tenant_id))

    async def update_active(
        self, kb_id: str, tenant_id: str, changes: Mapping[str, Any]
    ) -> KnowledgeBase:
        await self.get_active(kb_id, tenant_id)  # deleting KBs cannot be edited
        try:
            return await self.update(kb_id, changes, scope=self._scope(tenant_id))
        except DuplicateDocumentError as exc:
            raise KnowledgeBaseNameTakenError(details={"name": changes.get("name")}) from exc

    async def mark_deleting(self, kb_id: str, tenant_id: str) -> KnowledgeBase:
        """Flip active → deleting atomically; frees the name. Not idempotent by design:
        a second call gets 404 because the KB is no longer visible."""
        now = utc_now()
        result = await self.collection.update_one(
            self._where({"_id": kb_id, **_ACTIVE}, self._scope(tenant_id)),
            {"$set": {"status": ResourceStatus.DELETING.value, "updated_at": now}},
        )
        if result.modified_count != 1:
            raise KnowledgeBaseNotFoundError(details={"id": kb_id})
        return await self.get_or_raise(kb_id, scope=self._scope(tenant_id))

    async def remove(self, kb_id: str, *, scope: Filter | None = None) -> bool:
        """Hard delete (rollback of a create, or the end of the cascade in Step 81)."""
        return await self.delete(kb_id, scope=scope)

    async def embedding_spaces_in_use(self) -> set[EmbeddingSpace]:
        """Every (model, dimensions) any KB uses, deleting ones included (their vectors still
        need the collection until the cascade removes them). For the vector store bootstrap."""
        try:
            cursor = await self.collection.aggregate(
                [
                    {
                        "$group": {
                            "_id": {
                                "model": "$embedding_model",
                                "dimensions": "$embedding_dimensions",
                            }
                        }
                    }
                ]
            )
            rows = [row async for row in cursor]
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return {
            EmbeddingSpace(EmbeddingModelName(row["_id"]["model"]), int(row["_id"]["dimensions"]))
            for row in rows
        }
