"""Deleting a source version's data (Step 78): vectors and chunks, never the live version.

Used by finalization: after a successful swap the **previous** version is deleted (by the
`cleanup_version` maintenance task, delayed a little so in-flight searches finish); a
failed or cancelled job's **own** version is deleted right away (search never served it).

- The vectors go from the KB's collection (its embedding space), filtered to the tenant,
  source and version; the chunks from Mongo (`ChunkRepository.delete_version`). Pages stay:
  they are that job's per-page report (`GET /jobs/{id}/pages`).
- **The active version is never deleted**: the source is re-read at deletion time, so even
  a flip that raced the cleanup can't make it destroy live data.
- Idempotent and tolerant: a missing collection or an already-deleted KB just means there
  is nothing (left) to delete in the vector store.
"""

from dataclasses import dataclass

from loguru import logger

from rag_engine.domain import KnowledgeBase
from rag_engine.infrastructure import MongoDatabase
from rag_engine.repositories import ChunkRepository, KnowledgeBaseRepository, SourceRepository
from rag_engine.vectorstores import (
    BaseVectorStore,
    CollectionNotFoundError,
    CollectionSpec,
    VectorFilter,
)


@dataclass(frozen=True, slots=True)
class CleanupResult:
    version: str
    points: int  # vectors deleted (counted just before the delete)
    chunks: int
    skipped: str | None = None  # why nothing was deleted ("active_version")


class VersionCleaner:
    def __init__(self, db: MongoDatabase, store: BaseVectorStore) -> None:
        self.store = store
        self.sources = SourceRepository(db)
        self.kbs = KnowledgeBaseRepository(db)
        self.chunks = ChunkRepository(db)

    async def delete_version(
        self, *, tenant_id: str, kb_id: str, source_id: str, version: str
    ) -> CleanupResult:
        source = await self.sources.get(source_id)
        if source is not None and source.active_version == version:
            logger.warning("refusing to delete {}: it is {}'s active version", version, source_id)
            return CleanupResult(version, 0, 0, skipped="active_version")
        points = await self._delete_points(tenant_id, await self.kbs.get(kb_id), source_id, version)
        chunks = await self.chunks.delete_version(source_id, version)
        logger.info("deleted version {} of {}: {} points, {} chunks", version, source_id,
                    points, chunks)  # fmt: skip
        return CleanupResult(version, points, chunks)

    async def _delete_points(
        self, tenant_id: str, kb: KnowledgeBase | None, source_id: str, version: str
    ) -> int:
        if kb is None:  # the KB is gone: its cascade (Step 81) removed the vectors
            return 0
        spec = CollectionSpec.for_space(kb.embedding_space)
        where = VectorFilter(tenant_id, source_ids=(source_id,), versions=(version,))
        try:
            points = await self.store.count(spec, where)
            if points:
                await self.store.delete_by_filter(spec, where)
        except CollectionNotFoundError:
            return 0  # never ingested into a collection: nothing to delete
        return points
