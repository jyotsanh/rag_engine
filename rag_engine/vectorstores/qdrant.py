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

**Upserts** (Step 69): points are written in batches of `QDRANT_UPSERT_BATCH_SIZE`, in
order, each acknowledged (`wait=True`), so they are searchable when `upsert` returns.
Point ids are deterministic (`point_id(version, chunk_id)`), which makes upserts
idempotent: an unavailable Qdrant is retried (`QDRANT_MAX_RETRIES`, jittered backoff)
without ever duplicating a point, and re-ingesting a page overwrites its points. A point's
sparse vector is omitted when empty (a stopword-only chunk): it simply never matches a
sparse query. Payloads are stored as plain JSON (`ChunkPayload`).

**Deletes and counts** (Step 70) take a `VectorFilter`, translated by `to_qdrant_filter`:
the tenant is always a `must` condition, every other non-empty field adds one
(`MatchAny` over ids, `MatchPrefix` for `url_prefix`, on the indexed keyword fields), so
deleting a version, a source, a KB or a whole tenant is one filtered call. Deletes are
acknowledged (`wait=True`) and retried like upserts (deleting twice is harmless); counts
are exact. Searches (Steps 71-72) use the same translation.

**Dense search** (Step 71): the Query API on the named `dense` vector with the translated
filter, so every search is scoped to a tenant and optionally to KBs, sources, the active
versions (Step 73 passes each source's `active_version`; job ids are globally unique, so
`version ∈ active` is exact), pages and a URL prefix. Scores are cosine similarities,
best first; `score_threshold` drops weaker hits inside Qdrant. Reads are retried like the
other idempotent calls.

**Hybrid search** (Step 72): one Query API call with two filtered prefetches, the dense
vector (`dense`) and the sparse BM25 vector (`bm25`, IDF applied by Qdrant), each
returning `prefetch_limit` candidates, fused by **reciprocal rank fusion** (`RRF_K` = 60,
the same constant as the reference implementation). Exact-term matches (model numbers,
error codes) that dense search ranks low still surface. A query with an empty sparse
vector (only stopwords) skips the sparse prefetch; the dense candidates are still ranked by
RRF, so hybrid scores always mean the same thing.

Errors from either transport (HTTP or gRPC, `QDRANT_PREFER_GRPC`) are translated:
unreachable/5xx/timeouts → `VectorStoreUnavailableError`, unknown collection →
`CollectionNotFoundError`, anything else → `VectorStoreError`.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
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
    RRF_K,
    SPARSE_VECTOR,
    ChunkPayload,
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

    def __init__(
        self,
        client: AsyncQdrantClient,
        *,
        config: Settings = settings,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        """`client`: the process's `QdrantConnection.client` (its owner closes it)."""
        self.client = client
        self.config = config
        self.batch_size = config.QDRANT_UPSERT_BATCH_SIZE
        self._sleep = sleep
        self._rng = rng or random.Random()

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
        missing = [(f, schema) for f, schema in PAYLOAD_INDEXES.items() if f not in existing]
        # independent indexes, each acknowledged (wait=True): create them concurrently
        # (sequentially they dominated startup: ~0.4 s each)
        await asyncio.gather(
            *(
                self._call(
                    self.client.create_payload_index(
                        spec.name, field_name=field, field_schema=schema, wait=True
                    ),
                    spec,
                )
                for field, schema in missing
            )
        )

    # ---- points: Steps 69-72 ---- #

    async def _upsert(self, spec: CollectionSpec, points: list[VectorPoint]) -> None:
        for start in range(0, len(points), self.batch_size):
            batch = [_point_struct(p) for p in points[start : start + self.batch_size]]
            await self._retrying(
                lambda batch=batch: self.client.upsert(spec.name, points=batch, wait=True), spec
            )

    async def _delete_by_filter(self, spec: CollectionSpec, filter_: VectorFilter) -> None:
        selector = models.FilterSelector(filter=to_qdrant_filter(filter_))
        await self._retrying(
            lambda: self.client.delete(spec.name, points_selector=selector, wait=True), spec
        )

    async def _count(self, spec: CollectionSpec, filter_: VectorFilter) -> int:
        result = await self._retrying(
            lambda: self.client.count(
                spec.name, count_filter=to_qdrant_filter(filter_), exact=True
            ),
            spec,
        )
        return result.count

    async def _search_dense(
        self,
        spec: CollectionSpec,
        vector: list[float],
        filter_: VectorFilter,
        limit: int,
        score_threshold: float | None,
    ) -> list[SearchHit]:
        response = await self._retrying(
            lambda: self.client.query_points(
                spec.name,
                query=list(vector),
                using=DENSE_VECTOR,
                query_filter=to_qdrant_filter(filter_),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            ),
            spec,
        )
        return _hits(response.points)

    async def _search_hybrid(
        self,
        spec: CollectionSpec,
        dense: list[float],
        sparse: SparseVector,
        filter_: VectorFilter,
        limit: int,
        prefetch_limit: int,
    ) -> list[SearchHit]:
        where = to_qdrant_filter(filter_)
        prefetch = [
            models.Prefetch(query=list(dense), using=DENSE_VECTOR, filter=where,
                            limit=prefetch_limit)
        ]  # fmt: skip
        if len(sparse):
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse.indices), values=list(sparse.values)
                    ),
                    using=SPARSE_VECTOR,
                    filter=where,
                    limit=prefetch_limit,
                )
            )
        response = await self._retrying(
            lambda: self.client.query_points(
                spec.name,
                prefetch=prefetch,
                query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
                query_filter=where,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ),
            spec,
        )
        return _hits(response.points)

    # ---- retries and errors ---- #

    async def _retrying[T](self, operation: Callable[[], Awaitable[T]], spec: CollectionSpec) -> T:
        """Run an idempotent operation, retrying `VectorStoreUnavailableError` with backoff."""
        retries = self.config.QDRANT_MAX_RETRIES
        for attempt in range(retries + 1):
            try:
                return await self._call(operation(), spec)
            except VectorStoreUnavailableError as exc:
                if attempt == retries:
                    exc.details["attempts"] = attempt + 1
                    raise
                ceiling = min(
                    self.config.QDRANT_RETRY_MAX_DELAY_SECONDS,
                    self.config.QDRANT_RETRY_BASE_DELAY_SECONDS * 2**attempt,
                )
                await self._sleep(ceiling / 2 + self._rng.uniform(0, ceiling / 2))
        raise AssertionError("unreachable")

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


def _point_struct(point: VectorPoint) -> models.PointStruct:
    vectors: dict[str, Any] = {DENSE_VECTOR: list(point.dense)}
    if point.sparse is not None and len(point.sparse):
        vectors[SPARSE_VECTOR] = models.SparseVector(
            indices=list(point.sparse.indices), values=list(point.sparse.values)
        )
    return models.PointStruct(
        id=point.id, vector=vectors, payload=point.payload.model_dump(mode="json")
    )


def to_qdrant_filter(filter_: VectorFilter) -> models.Filter:
    """`VectorFilter` → Qdrant filter (all conditions must hold; the tenant always does)."""
    must: list[models.Condition] = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=filter_.tenant_id))
    ]
    for key, values in (
        ("kb_id", filter_.kb_ids),
        ("source_id", filter_.source_ids),
        ("version", filter_.versions),
        ("page_id", filter_.page_ids),
    ):
        if values:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
    if filter_.url_prefix is not None:
        must.append(
            models.FieldCondition(key="url", match=models.MatchPrefix(prefix=filter_.url_prefix))
        )
    return models.Filter(must=must)


def _hits(points: list[models.ScoredPoint]) -> list[SearchHit]:
    """Qdrant scored points → `SearchHit`s (payloads validated back into `ChunkPayload`)."""
    return [
        SearchHit(str(point.id), float(point.score), ChunkPayload.model_validate(point.payload))
        for point in points
    ]
