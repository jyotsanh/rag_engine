"""Qdrant vector store (Steps 68-72), registered for `VectorStoreType.QDRANT`.

**Collections** (Step 68): one per embedding space (`CollectionSpec.for_space`):
- named dense vector `dense`: size = the space's dimensions, cosine distance;
- named sparse vector `bm25` with **`modifier=IDF`**: Qdrant weighs query terms by their
  inverse document frequency over the collection at query time (Step 66 sends raw BM25
  term weights), so the statistics never go stale;
- keyword payload indexes for every filter field: **`tenant_id` with `is_tenant`** (Qdrant
  co-locates each tenant's points and optimises tenant-filtered search), `kb_id`,
  `source_id`, `version`, `page_id`, and `url` with **prefix matching** enabled
  (`MatchPrefix`, for `VectorFilter.url_prefix`, Step 71).

`ensure_collection` is idempotent and safe under concurrency: a create that loses a race
to another worker (409 / ALREADY_EXISTS) counts as "already exists", and reading the
schema of a collection whose creation is still settling (Qdrant answers 500 for a moment)
is retried briefly. An existing
collection is checked against the spec (`CollectionMismatchError` on any difference), and
missing payload indexes are added, so collections created by an older release catch up.

Errors from either transport (HTTP or gRPC, `QDRANT_PREFER_GRPC`) are translated:
unreachable/5xx/timeouts → `VectorStoreUnavailableError`, unknown collection →
`CollectionNotFoundError`, anything else → `VectorStoreError`.
"""

import asyncio
from collections.abc import Awaitable
from typing import Any

import grpc
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_engine.config import Settings, VectorStoreType, settings
from rag_engine.embeddings import SparseVector
from rag_engine.vectorstores.base import BaseVectorStore
from rag_engine.vectorstores.dispatcher import VectorStoreDispatcher
from rag_engine.vectorstores.exceptions import (
    CollectionMismatchError,
    CollectionNotFoundError,
    VectorStoreError,
    VectorStoreUnavailableError,
)
from rag_engine.vectorstores.models import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    CollectionSpec,
    SearchHit,
    VectorFilter,
    VectorPoint,
)

# filter fields → their keyword index (every VectorFilter field is indexed)
PAYLOAD_INDEXES: dict[str, models.KeywordIndexParams] = {
    "tenant_id": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
    "kb_id": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
    "source_id": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
    "version": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
    "page_id": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
    "url": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, prefix=True),
}

_UNAVAILABLE_GRPC = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
    grpc.StatusCode.INTERNAL,
    grpc.StatusCode.UNKNOWN,
}


# a just-created collection can answer 500 for a moment; ~1.5 s of retries covers it
_SETTLE_ATTEMPTS = 5
_SETTLE_BASE_SECONDS = 0.1


class _AlreadyExistsError(Exception):
    """Internal: a create lost the race to another writer."""


@VectorStoreDispatcher.register(VectorStoreType.QDRANT)
class QdrantVectorStore(BaseVectorStore):
    kind = VectorStoreType.QDRANT

    def __init__(self, client: AsyncQdrantClient, *, config: Settings = settings) -> None:
        """`client`: the process's `QdrantConnection.client` (its owner closes it)."""
        self.client = client
        self.config = config

    # ---- collections (Step 68) ---- #

    async def _ensure_collection(self, spec: CollectionSpec) -> bool:
        created = False
        if not await self._call(self.client.collection_exists(spec.name), spec):
            try:
                await self._call(
                    self.client.create_collection(
                        spec.name,
                        vectors_config={
                            DENSE_VECTOR: models.VectorParams(
                                size=spec.dimensions, distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config=(
                            {SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                            if spec.sparse
                            else None
                        ),
                    ),
                    spec,
                    creating=True,
                )
                created = True
            except _AlreadyExistsError:
                created = False  # another worker created it first: verify like any existing one
        info = await self._collection_info(spec)
        self._check_schema(spec, info)
        await self._ensure_payload_indexes(spec, info)
        return created

    async def _collection_info(self, spec: CollectionSpec) -> models.CollectionInfo:
        """The collection's config. Right after a concurrent create, Qdrant briefly answers
        500 ("0 of 0 read operations failed") while the shards initialise: retry that."""
        for attempt in range(_SETTLE_ATTEMPTS):
            try:
                return await self._call(self.client.get_collection(spec.name), spec)
            except VectorStoreUnavailableError:
                if attempt == _SETTLE_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_SETTLE_BASE_SECONDS * 2**attempt)
        raise AssertionError("unreachable")

    def _check_schema(self, spec: CollectionSpec, info: models.CollectionInfo) -> None:
        params = info.config.params
        problems: list[str] = []
        dense = params.vectors.get(DENSE_VECTOR) if isinstance(params.vectors, dict) else None
        if dense is None:
            problems.append(f"no named dense vector {DENSE_VECTOR!r}")
        else:
            if dense.size != spec.dimensions:
                problems.append(f"dense size {dense.size} != {spec.dimensions}")
            if dense.distance != models.Distance.COSINE:
                problems.append(f"dense distance {dense.distance} != Cosine")
        sparse = (params.sparse_vectors or {}).get(SPARSE_VECTOR)
        if spec.sparse and sparse is None:
            problems.append(f"no sparse vector {SPARSE_VECTOR!r}")
        if spec.sparse and sparse is not None and sparse.modifier != models.Modifier.IDF:
            problems.append(f"sparse modifier {sparse.modifier} != idf")
        if not spec.sparse and sparse is not None:
            problems.append(f"unexpected sparse vector {SPARSE_VECTOR!r}")
        if problems:
            raise CollectionMismatchError(
                f"collection {spec.name} does not match its spec: " + "; ".join(problems),
                details={"collection": spec.name, "problems": problems},
            )

    async def _ensure_payload_indexes(
        self, spec: CollectionSpec, info: models.CollectionInfo
    ) -> None:
        existing = info.payload_schema or {}
        for field, schema in PAYLOAD_INDEXES.items():
            if field not in existing:
                await self._call(
                    self.client.create_payload_index(
                        spec.name, field_name=field, field_schema=schema, wait=True
                    ),
                    spec,
                )

    # ---- points: Steps 69-72 ---- #

    async def _upsert(self, spec: CollectionSpec, points: list[VectorPoint]) -> None:
        raise NotImplementedError("Qdrant upsert is Step 69")

    async def _delete_by_filter(self, spec: CollectionSpec, filter_: VectorFilter) -> None:
        raise NotImplementedError("Qdrant delete is Step 70")

    async def _count(self, spec: CollectionSpec, filter_: VectorFilter) -> int:
        raise NotImplementedError("Qdrant count is Step 70")

    async def _search_dense(
        self,
        spec: CollectionSpec,
        vector: list[float],
        filter_: VectorFilter,
        limit: int,
        score_threshold: float | None,
    ) -> list[SearchHit]:
        raise NotImplementedError("Qdrant dense search is Step 71")

    async def _search_hybrid(
        self,
        spec: CollectionSpec,
        dense: list[float],
        sparse: SparseVector,
        filter_: VectorFilter,
        limit: int,
        prefetch_limit: int,
    ) -> list[SearchHit]:
        raise NotImplementedError("Qdrant hybrid search is Step 72")

    # ---- errors ---- #

    async def _call[T](
        self, operation: Awaitable[T], spec: CollectionSpec, *, creating: bool = False
    ) -> T:
        """Await a client call, translating transport errors to the vector store taxonomy."""
        details: dict[str, Any] = {"collection": spec.name}
        try:
            return await operation
        except UnexpectedResponse as exc:
            status = exc.status_code or 0
            if creating and status == 409:
                raise _AlreadyExistsError from exc
            if status == 404:
                raise CollectionNotFoundError(details=details) from exc
            if status >= 500 or status in (408, 429):
                raise VectorStoreUnavailableError(details={**details, "status": status}) from exc
            raise VectorStoreError(
                f"Qdrant refused the operation ({status})", details={**details, "status": status}
            ) from exc
        except grpc.aio.AioRpcError as exc:
            code = exc.code()
            if creating and code == grpc.StatusCode.ALREADY_EXISTS:
                raise _AlreadyExistsError from exc
            if code == grpc.StatusCode.NOT_FOUND:
                raise CollectionNotFoundError(details=details) from exc
            if code in _UNAVAILABLE_GRPC:
                raise VectorStoreUnavailableError(details={**details, "grpc": code.name}) from exc
            raise VectorStoreError(
                f"Qdrant refused the operation ({code.name})",
                details={**details, "grpc": code.name},
            ) from exc
        except (ResponseHandlingException, ConnectionError, TimeoutError, OSError) as exc:
            raise VectorStoreUnavailableError(
                details={**details, "reason": type(exc).__name__}
            ) from exc
