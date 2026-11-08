"""Vector store contract (Step 67): one backend, many collections (one per embedding space).

Implementations subclass `BaseVectorStore` and implement the `_`-prefixed methods. The
public methods are final and validate before anything reaches the backend:

- every method takes the `CollectionSpec`, so dense vectors must have its dimensions and a
  sparse vector is required exactly when the collection has one (hybrid search needs it);
- upserted points have unique ids and finite vectors; `upsert([])` is a no-op;
- filters always carry a tenant (`VectorFilter`), so nothing crosses tenants;
- `limit` is within 1..`MAX_SEARCH_LIMIT`, `score_threshold` within [-1, 1].

`ensure_collection` is idempotent (True when it created the collection) and must refuse an
existing collection whose schema differs (`CollectionMismatchError`). Searches return hits
best-first. Deletes are by filter (a version, a source, a KB, a tenant: Step 70).
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from rag_engine.config import VectorStoreType
from rag_engine.embeddings import SparseVector
from rag_engine.vectorstores.exceptions import InvalidVectorDataError, InvalidVectorQueryError
from rag_engine.vectorstores.models import CollectionSpec, SearchHit, VectorFilter, VectorPoint

MAX_SEARCH_LIMIT = 200


class BaseVectorStore(ABC):
    kind: ClassVar[VectorStoreType]

    # ---- collections ---- #

    async def ensure_collection(self, spec: CollectionSpec) -> bool:
        return await self._ensure_collection(spec)

    # ---- writes ---- #

    async def upsert(self, spec: CollectionSpec, points: Sequence[VectorPoint]) -> int:
        """Insert or overwrite points by id; returns how many were written."""
        points = list(points)
        if not points:
            return 0
        ids = [p.id for p in points]
        if len(set(ids)) != len(ids):
            raise InvalidVectorDataError("duplicate point ids in one upsert")
        for point in points:
            self._check_dense(spec, point.dense, f"point {point.id}")
            self._check_sparse(spec, point.sparse, f"point {point.id}")
        await self._upsert(spec, points)
        return len(points)

    async def delete_by_filter(self, spec: CollectionSpec, filter_: VectorFilter) -> None:
        await self._delete_by_filter(spec, filter_)

    # ---- reads ---- #

    async def count(self, spec: CollectionSpec, filter_: VectorFilter) -> int:
        return await self._count(spec, filter_)

    async def search_dense(
        self,
        spec: CollectionSpec,
        vector: list[float],
        filter_: VectorFilter,
        *,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        self._check_dense(spec, vector, "query")
        _check_limit(limit)
        if score_threshold is not None and not -1.0 <= score_threshold <= 1.0:
            raise InvalidVectorQueryError("score_threshold must be within [-1, 1]")
        return await self._search_dense(spec, vector, filter_, limit, score_threshold)

    async def search_hybrid(
        self,
        spec: CollectionSpec,
        dense: list[float],
        sparse: SparseVector,
        filter_: VectorFilter,
        *,
        limit: int = 10,
        prefetch_limit: int | None = None,
    ) -> list[SearchHit]:
        """Dense + sparse candidates (`prefetch_limit` each, default 4 x limit) fused by RRF."""
        if not spec.sparse:
            raise InvalidVectorQueryError(f"collection {spec.name} has no sparse vectors")
        self._check_dense(spec, dense, "query")
        _check_limit(limit)
        prefetch = prefetch_limit or min(limit * 4, MAX_SEARCH_LIMIT)
        if not limit <= prefetch <= MAX_SEARCH_LIMIT:
            raise InvalidVectorQueryError(
                f"prefetch_limit must be within {limit}..{MAX_SEARCH_LIMIT}"
            )
        return await self._search_hybrid(spec, dense, sparse, filter_, limit, prefetch)

    async def aclose(self) -> None:  # noqa: B027 (an optional hook, not abstract)
        """Release clients the store owns."""

    # ---- checks ---- #

    @staticmethod
    def _check_dense(spec: CollectionSpec, vector: Sequence[float], what: str) -> None:
        if len(vector) != spec.dimensions:
            raise InvalidVectorDataError(
                f"{what}: {len(vector)} dimensions, collection {spec.name} has {spec.dimensions}",
                details={"collection": spec.name, "dimensions": len(vector)},
            )
        if not all(math.isfinite(v) for v in vector):
            raise InvalidVectorDataError(f"{what}: non-finite vector values")

    @staticmethod
    def _check_sparse(spec: CollectionSpec, sparse: SparseVector | None, what: str) -> None:
        if spec.sparse and sparse is None:
            raise InvalidVectorDataError(f"{what}: collection {spec.name} needs a sparse vector")
        if not spec.sparse and sparse is not None:
            raise InvalidVectorDataError(f"{what}: collection {spec.name} has no sparse vectors")

    # ---- implementations ---- #

    @abstractmethod
    async def _ensure_collection(self, spec: CollectionSpec) -> bool: ...

    @abstractmethod
    async def _upsert(self, spec: CollectionSpec, points: list[VectorPoint]) -> None: ...

    @abstractmethod
    async def _delete_by_filter(self, spec: CollectionSpec, filter_: VectorFilter) -> None: ...

    @abstractmethod
    async def _count(self, spec: CollectionSpec, filter_: VectorFilter) -> int: ...

    @abstractmethod
    async def _search_dense(
        self,
        spec: CollectionSpec,
        vector: list[float],
        filter_: VectorFilter,
        limit: int,
        score_threshold: float | None,
    ) -> list[SearchHit]: ...

    @abstractmethod
    async def _search_hybrid(
        self,
        spec: CollectionSpec,
        dense: list[float],
        sparse: SparseVector,
        filter_: VectorFilter,
        limit: int,
        prefetch_limit: int,
    ) -> list[SearchHit]: ...


def _check_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise InvalidVectorQueryError(f"limit must be within 1..{MAX_SEARCH_LIMIT}")
