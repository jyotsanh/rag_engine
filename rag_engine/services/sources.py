"""Tenant-facing source reads and deletion. Sources are created by crawl submission (Step 84)."""

from rag_engine.config import logger
from rag_engine.domain import Source, Tenant
from rag_engine.repositories import KnowledgeBaseRepository, Page
from rag_engine.repositories.sources import SourceRepository


class SourceService:
    def __init__(self, kbs: KnowledgeBaseRepository, sources: SourceRepository) -> None:
        self._kbs = kbs
        self._sources = sources

    async def _require_kb(self, tenant: Tenant, kb_id: str) -> None:
        # a missing, foreign or deleting KB hides all of its sources (404 knowledge_base_not_found)
        await self._kbs.get_active(kb_id, tenant.id)

    async def list(
        self, tenant: Tenant, kb_id: str, *, limit: int, cursor: str | None
    ) -> Page[Source]:
        await self._require_kb(tenant, kb_id)
        return await self._sources.list_active(tenant.id, kb_id, limit=limit, cursor=cursor)

    async def get(self, tenant: Tenant, kb_id: str, source_id: str) -> Source:
        await self._require_kb(tenant, kb_id)
        return await self._sources.get_active(source_id, tenant.id, kb_id)

    async def delete(self, tenant: Tenant, kb_id: str, source_id: str) -> Source:
        """Hides the source and frees its seed URL; Step 81 removes pages, chunks and vectors.

        Step 87 must cancel an in-flight crawl of this source before deletion completes.
        """
        await self._require_kb(tenant, kb_id)
        source = await self._sources.mark_deleting(source_id, tenant.id, kb_id)
        logger.bind(kb_id=kb_id, source_id=source_id).info("source marked for deletion")
        return source
