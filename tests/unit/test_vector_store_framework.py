"""Vector store framework: models, the base contract (validation), reference semantics."""

import math
import uuid
from collections.abc import Iterator

import pytest

from rag_engine.config import VectorStoreType
from rag_engine.domain import Chunk, chunk_id
from rag_engine.embeddings import SparseVector, resolve_space
from rag_engine.vectorstores import (
    MAX_SEARCH_LIMIT,
    ChunkPayload,
    CollectionMismatchError,
    CollectionNotFoundError,
    CollectionSpec,
    InvalidVectorDataError,
    InvalidVectorQueryError,
    UnsupportedVectorStoreError,
    VectorFilter,
    VectorPoint,
    VectorStoreDispatcher,
    point_id,
)
from tests.support.fakes import InMemoryVectorStore

_T1, _T2 = "ten_" + "1" * 32, "ten_" + "2" * 32
_KB = "kb_" + "1" * 32
_S1, _S2 = "src_" + "1" * 32, "src_" + "2" * 32
_V1, _V2 = "job_" + "1" * 32, "job_" + "2" * 32
_SPEC = CollectionSpec("test_chunks", 3)
_DENSE_ONLY = CollectionSpec("test_dense", 3, sparse=False)


def _payload(n: int, *, tenant: str = _T1, source: str = _S1, version: str = _V1,
             url: str | None = None) -> ChunkPayload:  # fmt: skip
    return ChunkPayload(
        tenant_id=tenant, kb_id=_KB, source_id=source, version=version,
        chunk_id=chunk_id(f"page_{n}", 0), page_id=f"page_{n}",
        url=url or f"https://acme.example/docs/p{n}", ordinal=0, text=f"chunk {n}",
    )  # fmt: skip


def _point(n: int, dense: list[float], sparse: dict[int, float] | None = None,
           **payload: str) -> VectorPoint:  # fmt: skip
    vector = SparseVector.from_pairs(list(sparse), list(sparse.values())) if sparse else None
    body = _payload(n, **payload)
    return VectorPoint(point_id(body.version, body.chunk_id), dense, body,
                       vector if vector is not None else SparseVector((), ()))  # fmt: skip


async def _store(*points: VectorPoint) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.ensure_collection(_SPEC)
    await store.upsert(_SPEC, points)
    return store


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(VectorStoreDispatcher, "_stores", {})
    yield


# ---- #
# Tests
# ---- #


class TestVectorStoreFramework_S67:
    """S67 — Vector store framework: models, `BaseVectorStore` contract, dispatcher."""

    # ---- models ---- #

    def test_one_collection_per_embedding_space(self) -> None:
        small = CollectionSpec.for_space(resolve_space("text-embedding-3-small"))
        reduced = CollectionSpec.for_space(resolve_space("text-embedding-3-small", 512))

        assert (small.name, small.dimensions, small.sparse) == (
            "chunks__text-embedding-3-small__1536", 1536, True)  # fmt: skip
        assert reduced.name == "chunks__text-embedding-3-small__512"
        for bad in ("", "has space", "dots.are.bad"):
            with pytest.raises(ValueError, match="collection name"):
                CollectionSpec(bad, 3)
        with pytest.raises(ValueError, match="dimensions"):
            CollectionSpec("ok", 0)

    def test_point_ids_are_deterministic_uuids_per_version_and_chunk(self) -> None:
        a = point_id(_V1, "chk_a")

        assert a == point_id(_V1, "chk_a") and uuid.UUID(a).version == 5
        assert len({a, point_id(_V2, "chk_a"), point_id(_V1, "chk_b")}) == 3

    def test_points_are_built_from_stored_chunks(self) -> None:
        chunk = Chunk(
            id=chunk_id("page_" + "1" * 32, 2), tenant_id=_T1, kb_id=_KB, source_id=_S1,
            version=_V1, page_id="page_" + "1" * 32, url="https://acme.example/a",
            title="Guide", heading_path=["Setup", "Linux"], ordinal=2, text="Run it.",
            embed_text="Guide\nSetup > Linux\n\nRun it.", token_count=9,
            content_hash="0" * 64, page_number=4,
        )  # fmt: skip

        point = VectorPoint.from_chunk(chunk, [0.1, 0.2, 0.3], SparseVector((5,), (1.0,)))

        assert point.id == point_id(_V1, chunk.id)
        assert point.payload == ChunkPayload(
            tenant_id=_T1, kb_id=_KB, source_id=_S1, version=_V1, chunk_id=chunk.id,
            page_id=chunk.page_id, url=chunk.url, ordinal=2, text="Run it.", title="Guide",
            heading_path=("Setup", "Linux"), page_number=4,
        )  # fmt: skip

    def test_filters_always_have_a_tenant_and_match_like_the_reference(self) -> None:
        payload = _payload(1, url="https://acme.example/docs/install")

        with pytest.raises(ValueError, match="tenant_id"):
            VectorFilter("")
        with pytest.raises(ValueError, match="url_prefix"):
            VectorFilter(_T1, url_prefix="")
        assert VectorFilter(_T1).matches(payload)
        assert not VectorFilter(_T2).matches(payload)
        assert VectorFilter(_T1, kb_ids=(_KB,), source_ids=(_S2, _S1), versions=(_V1,)).matches(
            payload
        )
        assert not VectorFilter(_T1, versions=(_V2,)).matches(payload)
        assert VectorFilter(_T1, url_prefix="https://acme.example/docs/").matches(payload)
        assert not VectorFilter(_T1, url_prefix="https://acme.example/blog/").matches(payload)
        assert not VectorFilter(_T1, page_ids=("page_9",)).matches(payload)

    # ---- the base contract ---- #

    async def test_upsert_validates_against_the_collection(self) -> None:
        store = await _store()

        assert await store.upsert(_SPEC, []) == 0
        with pytest.raises(InvalidVectorDataError, match="2 dimensions"):
            await store.upsert(_SPEC, [_point(1, [1.0, 0.0])])
        with pytest.raises(InvalidVectorDataError, match="non-finite"):
            await store.upsert(_SPEC, [_point(1, [math.nan, 0.0, 0.0])])
        with pytest.raises(InvalidVectorDataError, match="duplicate"):
            await store.upsert(_SPEC, [_point(1, [1.0, 0, 0]), _point(1, [0, 1.0, 0])])
        no_sparse = VectorPoint("p", [1.0, 0, 0], _payload(1))
        with pytest.raises(InvalidVectorDataError, match="needs a sparse vector"):
            await store.upsert(_SPEC, [no_sparse])
        await store.ensure_collection(_DENSE_ONLY)
        with pytest.raises(InvalidVectorDataError, match="has no sparse vectors"):
            await store.upsert(_DENSE_ONLY, [_point(1, [1.0, 0, 0], {1: 1.0})])
        assert await store.count(_SPEC, VectorFilter(_T1)) == 0  # nothing was written

    async def test_queries_are_validated(self) -> None:
        store = await _store()
        await store.ensure_collection(_DENSE_ONLY)
        query, sparse, where = [1.0, 0.0, 0.0], SparseVector((1,), (1.0,)), VectorFilter(_T1)

        for limit in (0, MAX_SEARCH_LIMIT + 1):
            with pytest.raises(InvalidVectorQueryError, match="limit"):
                await store.search_dense(_SPEC, query, where, limit=limit)
        with pytest.raises(InvalidVectorQueryError, match="score_threshold"):
            await store.search_dense(_SPEC, query, where, score_threshold=1.5)
        with pytest.raises(InvalidVectorDataError, match="dimensions"):
            await store.search_dense(_SPEC, [1.0], where)
        with pytest.raises(InvalidVectorQueryError, match="no sparse vectors"):
            await store.search_hybrid(_DENSE_ONLY, query, sparse, where)
        with pytest.raises(InvalidVectorQueryError, match="prefetch_limit"):
            await store.search_hybrid(_SPEC, query, sparse, where, limit=10, prefetch_limit=5)

    # ---- reference semantics (what every backend must do) ---- #

    async def test_collections_are_idempotent_and_checked(self) -> None:
        store = InMemoryVectorStore()

        assert await store.ensure_collection(_SPEC) is True
        assert await store.ensure_collection(_SPEC) is False
        with pytest.raises(CollectionMismatchError):
            await store.ensure_collection(CollectionSpec("test_chunks", 4))
        with pytest.raises(CollectionNotFoundError):
            await store.count(CollectionSpec("absent", 3), VectorFilter(_T1))

    async def test_upsert_overwrites_by_id_and_filters_count(self) -> None:
        store = await _store(
            _point(1, [1, 0, 0]), _point(2, [0, 1, 0], version=_V2),
            _point(3, [0, 0, 1], source=_S2), _point(4, [1, 1, 0], tenant=_T2),
        )  # fmt: skip

        await store.upsert(_SPEC, [_point(1, [0, 1, 0])])  # same id: overwritten, not added

        assert await store.count(_SPEC, VectorFilter(_T1)) == 3
        assert await store.count(_SPEC, VectorFilter(_T1, versions=(_V1,))) == 2
        assert await store.count(_SPEC, VectorFilter(_T1, source_ids=(_S2,))) == 1
        assert await store.count(_SPEC, VectorFilter(_T2)) == 1

    async def test_delete_by_filter_removes_only_the_match(self) -> None:
        store = await _store(
            _point(1, [1, 0, 0]), _point(2, [0, 1, 0], version=_V2),
            _point(3, [0, 1, 0], version=_V2, tenant=_T2),
        )  # fmt: skip

        await store.delete_by_filter(_SPEC, VectorFilter(_T1, versions=(_V2,)))

        assert await store.count(_SPEC, VectorFilter(_T1)) == 1
        assert await store.count(_SPEC, VectorFilter(_T2, versions=(_V2,))) == 1  # other tenant

    async def test_dense_search_ranks_by_cosine_within_the_tenant(self) -> None:
        store = await _store(
            _point(1, [1, 0, 0]), _point(2, [1, 1, 0]), _point(3, [0, 0, 1]),
            _point(4, [1, 0, 0], tenant=_T2),
        )  # fmt: skip

        hits = await store.search_dense(_SPEC, [1.0, 0.0, 0.0], VectorFilter(_T1), limit=2)
        strict = await store.search_dense(
            _SPEC, [1.0, 0, 0], VectorFilter(_T1), score_threshold=0.9
        )

        assert [h.payload.page_id for h in hits] == ["page_1", "page_2"]
        assert hits[0].score == pytest.approx(1.0) and hits[1].score == pytest.approx(0.7071, 1e-3)
        assert [h.payload.page_id for h in strict] == ["page_1"]

    async def test_hybrid_search_surfaces_exact_term_matches(self) -> None:
        store = await _store(
            _point(1, [1, 0, 0], {10: 1.0}),  # semantically closest, no keyword
            _point(2, [0, 1, 0], {42: 2.0}),  # semantically far, has the rare keyword
            _point(3, [0.9, 0.1, 0], {10: 1.0}),
        )  # fmt: skip
        query_dense, query_sparse = [1.0, 0.0, 0.0], SparseVector((42,), (1.0,))

        dense = await store.search_dense(_SPEC, query_dense, VectorFilter(_T1), limit=2)
        hybrid = await store.search_hybrid(
            _SPEC, query_dense, query_sparse, VectorFilter(_T1), limit=2
        )

        assert "page_2" not in [h.payload.page_id for h in dense]
        assert hybrid[0].payload.page_id == "page_2"  # first in sparse + present in dense
        assert len(hybrid) == 2

    async def test_closing(self) -> None:
        store = await _store()

        await store.aclose()

        assert store.closed

    # ---- dispatcher ---- #

    def test_dispatcher_creates_the_active_backend(self, isolated_registry: None) -> None:
        VectorStoreDispatcher.register(VectorStoreType.QDRANT)(InMemoryVectorStore)

        store = VectorStoreDispatcher.create()

        assert isinstance(store, InMemoryVectorStore)
        assert VectorStoreDispatcher.registered() == {VectorStoreType.QDRANT: InMemoryVectorStore}

    def test_dispatcher_errors(self, isolated_registry: None) -> None:
        class _Other(InMemoryVectorStore):
            pass

        with pytest.raises(UnsupportedVectorStoreError, match="no vector store"):
            VectorStoreDispatcher.create()
        VectorStoreDispatcher.register(VectorStoreType.QDRANT)(InMemoryVectorStore)
        with pytest.raises(TypeError, match="already registered"):
            VectorStoreDispatcher.register(VectorStoreType.QDRANT)(_Other)
        with pytest.raises(TypeError, match="not a BaseVectorStore"):
            VectorStoreDispatcher.register(VectorStoreType.QDRANT)(dict)  # ty: ignore[invalid-argument-type]
