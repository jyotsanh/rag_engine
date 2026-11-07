"""Chunks of crawled pages, stored per `(source_id, version)` (Step 59).

Why keep chunks in Mongo when the vectors live in Qdrant:
- **audit**: what exactly was embedded for a page, and under which heading path;
- **re-embedding without re-crawling**: a model change or a lost collection re-embeds a
  version from its stored `embed_text` (`iter_version`);
- **cleanup by version**: a failed/cancelled job's chunks, or the old version after a
  swap, are deleted by `(source_id, version)` (Steps 78, 81).

Writes are **per page and idempotent**: `save_page_chunks` upserts the page's chunks by
their deterministic ids (`chunk_id(page_id, ordinal)`) and deletes ordinals the new
chunking no longer has. Re-chunking a page after a crash, or with a new chunker, never
leaves duplicates or stale tails.
"""

from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

from pymongo import ASCENDING, ReplaceOne
from pymongo.errors import BulkWriteError, PyMongoError

from rag_engine.domain import Chunk, MongoDocument, chunk_id
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.repositories.base import BaseRepository

_VERSION_ORDER = [("page_id", ASCENDING), ("ordinal", ASCENDING)]


class ChunkRepository(BaseRepository[Chunk]):
    model: ClassVar[type[MongoDocument]] = Chunk

    # ---- writes ---- #

    async def save_page_chunks(self, page_id: str, chunks: Sequence[Chunk]) -> int:
        """Make `chunks` the page's chunks (all of one page, ordinals 0..n-1); returns n.

        An empty list removes the page's chunks (e.g. its content is now empty).
        """
        _check_page(page_id, chunks)
        try:
            if chunks:
                await self.collection.bulk_write(
                    [ReplaceOne({"_id": c.id}, c.to_mongo(), upsert=True) for c in chunks],
                    ordered=False,
                )
            await self.collection.delete_many(
                {"page_id": page_id, "ordinal": {"$gte": len(chunks)}}
            )
        except BulkWriteError as exc:
            raise DatabaseUnavailableError(details={"reason": "BulkWriteError"}) from exc
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return len(chunks)

    async def delete_version(self, source_id: str, version: str) -> int:
        """Drop a version's chunks (a failed/cancelled job, or the old version after a swap)."""
        return await self._delete({"source_id": source_id, "version": version})

    async def delete_source(self, source_id: str) -> int:
        """Drop every version of a source (source deletion, Step 81)."""
        return await self._delete({"source_id": source_id})

    async def _delete(self, filter_: dict[str, str]) -> int:
        try:
            return await self.delete_many(filter_)
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc

    # ---- reads ---- #

    async def for_page(self, page_id: str) -> list[Chunk]:
        """A page's chunks in order."""
        return await self.find({"page_id": page_id}, sort=[("ordinal", ASCENDING)])

    async def iter_version(
        self, source_id: str, version: str, *, batch_size: int = 500
    ) -> AsyncIterator[Chunk]:
        """Every chunk of a version, page by page in ordinal order, streamed (re-embedding)."""
        cursor = self.collection.find(
            {"source_id": source_id, "version": version},
            sort=_VERSION_ORDER,
            batch_size=batch_size,
        )
        try:
            async for raw in cursor:
                yield self._load(raw)
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def count_version(self, source_id: str, version: str) -> int:
        return await self.count({"source_id": source_id, "version": version})

    async def versions(self, source_id: str) -> list[str]:
        """Versions of a source that still have chunks (e.g. leftovers to clean up)."""
        try:
            values = await self.collection.distinct("version", {"source_id": source_id})
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return sorted(values)


def _check_page(page_id: str, chunks: Sequence[Chunk]) -> None:
    if any(c.page_id != page_id for c in chunks):
        raise ValueError("every chunk must belong to the page being saved")
    if len({(c.source_id, c.version) for c in chunks}) > 1:
        raise ValueError("a page's chunks belong to one source version")
    if [c.ordinal for c in chunks] != list(range(len(chunks))):
        raise ValueError("chunk ordinals must be 0..n-1 in order")
    if any(c.id != chunk_id(page_id, c.ordinal) for c in chunks):
        raise ValueError("chunk ids must be chunk_id(page_id, ordinal)")
