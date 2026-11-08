"""Embedding cache in MongoDB (Step 64): vectors by `sha256(model|dims|embed_text)`.

`get_many` reads hits by primary key (`$in`, in slices) and refreshes their
`last_used_at`, so the TTL index only evicts entries no job has needed for
`EMBEDDING_CACHE_TTL_DAYS`. `put_many` upserts (a concurrent writer of the same key wrote
the same vector, so last write wins harmlessly).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar

from pymongo import ReplaceOne
from pymongo.errors import BulkWriteError, PyMongoError

from rag_engine.config import EmbeddingModelName
from rag_engine.domain import (
    EmbeddingCacheEntry,
    MongoDocument,
    pack_vector,
    unpack_vector,
    utc_now,
)
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.repositories.base import BaseRepository

_SLICE = 1000  # keys per $in query / documents per bulk write


def _slices[T](items: Sequence[T]) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), _SLICE):
        yield items[start : start + _SLICE]


class EmbeddingCacheRepository(BaseRepository[EmbeddingCacheEntry]):
    model: ClassVar[type[MongoDocument]] = EmbeddingCacheEntry

    async def get_many(self, keys: Sequence[str], *, dimensions: int) -> dict[str, list[float]]:
        """Cached vectors for `keys` (missing keys are absent); hits are marked as used."""
        found: dict[str, list[float]] = {}
        unique = list(dict.fromkeys(keys))
        try:
            for part in _slices(unique):
                cursor = self.collection.find(
                    {"_id": {"$in": list(part)}, "dimensions": dimensions},
                    projection={"vector": 1},
                )
                async for raw in cursor:
                    vector = unpack_vector(bytes(raw["vector"]))
                    if len(vector) == dimensions:  # never serve a corrupt entry
                        found[raw["_id"]] = vector
            hits = list(found)
            now = utc_now()
            for part in _slices(hits):
                await self.collection.update_many(
                    {"_id": {"$in": list(part)}}, {"$set": {"last_used_at": now}}
                )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return found

    async def put_many(
        self,
        vectors: Mapping[str, list[float]],
        *,
        model: EmbeddingModelName,
        dimensions: int,
    ) -> int:
        """Store `key → vector` entries of one space; returns how many were written."""
        if any(len(v) != dimensions for v in vectors.values()):
            raise ValueError(f"every cached vector must have {dimensions} dimensions")
        now = utc_now()
        documents = [
            EmbeddingCacheEntry(
                id=key,
                model=model,
                dimensions=dimensions,
                vector=pack_vector(vector),
                last_used_at=now,
            ).to_mongo()
            for key, vector in vectors.items()
        ]
        try:
            for part in _slices(documents):
                await self.collection.bulk_write(
                    [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in part], ordered=False
                )
        except BulkWriteError as exc:
            raise DatabaseUnavailableError(details={"reason": "BulkWriteError"}) from exc
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return len(documents)
