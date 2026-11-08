"""Qdrant dense search against a real Qdrant, checked against the in-memory reference."""

import itertools
from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.embeddings import SparseVector
from rag_engine.infrastructure import QdrantConnection
from rag_engine.vectorstores import (
    ChunkPayload,
    CollectionNotFoundError,
    CollectionSpec,
    QdrantVectorStore,
    VectorFilter,
    VectorPoint,
    point_id,
)
from tests.conftest import OverrideSettings
from tests.support.fakes import InMemoryVectorStore, fake_vector

_DIMS = 16
_SPEC = CollectionSpec("dense_search_test", _DIMS)
_T1, _T2 = "ten_" + "1" * 32, "ten_" + "2" * 32
_K1, _K2 = "kb_" + "1" * 32, "kb_" + "2" * 32
_S1, _S2 = "src_" + "1" * 32, "src_" + "2" * 32
_OLD, _ACTIVE = "job_" + "1" * 32, "job_" + "2" * 32


def _corpus() -> list[VectorPoint]:
    """60 points with seeded vectors across tenants, KBs, sources, versions and URL sections."""
    points = []
    combos = itertools.product((_T1, _T2), (_K1, _K2), (_S1, _S2), (_OLD, _ACTIVE))
    for n, ((tenant, kb, source, version), page) in enumerate(
        itertools.product(list(combos)[:15], range(4))
    ):
        section = ("docs", "blog", "docs/api", "pricing")[page]
        payload = ChunkPayload(
            tenant_id=tenant, kb_id=kb, source_id=source, version=version,
            chunk_id=f"chk_{n:032x}", page_id=f"page_{n}",
            url=f"https://acme.example/{section}/{n}", ordinal=0, text=f"chunk {n}",
            title="Acme", heading_path=("Guide",), page_number=page + 1,
        )  # fmt: skip
        points.append(VectorPoint(point_id(version, payload.chunk_id),
                                  fake_vector(f"doc-{n}", _DIMS), payload,
                                  SparseVector((n,), (1.0,))))  # fmt: skip
    return points


_POINTS = _corpus()
_FILTERS = [
    VectorFilter(_T1),
    VectorFilter(_T2),
    VectorFilter(_T1, kb_ids=(_K1,)),
    VectorFilter(_T1, versions=(_ACTIVE,)),
    VectorFilter(_T1, kb_ids=(_K2,), source_ids=(_S1,), versions=(_ACTIVE,)),
    VectorFilter(_T1, url_prefix="https://acme.example/docs/"),
    VectorFilter(_T1, url_prefix="https://acme.example/docs/api/"),
    VectorFilter(_T1, page_ids=("page_1", "page_2", "page_40")),
]
_QUERIES = [fake_vector(f"query-{q}", _DIMS) for q in range(5)]


@pytest.fixture(params=[False, True], ids=["http", "grpc"])
async def stores(
    request: pytest.FixtureRequest,
    integration_settings: Settings,
    override_settings: OverrideSettings,
) -> AsyncIterator[tuple[QdrantVectorStore, InMemoryVectorStore]]:
    connection = QdrantConnection(override_settings(QDRANT_PREFER_GRPC=request.param))
    await connection.connect()
    qdrant, reference = QdrantVectorStore(connection.client), InMemoryVectorStore()
    for store in (qdrant, reference):
        await store.ensure_collection(_SPEC)
        await store.upsert(_SPEC, _POINTS)
    yield qdrant, reference
    await connection.close()


# ---- #
# Tests
# ---- #


class TestQdrantDenseSearch_S71:
    """S71 — Qdrant dense search: filters (tenant/KB/version/source/URL), threshold, ranking."""

    async def test_results_match_the_reference_for_every_filter(
        self, stores: tuple[QdrantVectorStore, InMemoryVectorStore]
    ) -> None:
        qdrant, reference = stores

        for filter_, query in itertools.product(_FILTERS, _QUERIES):
            got = await qdrant.search_dense(_SPEC, query, filter_, limit=7)
            want = await reference.search_dense(_SPEC, query, filter_, limit=7)

            assert [h.id for h in got] == [h.id for h in want], filter_
            assert [h.score for h in got] == pytest.approx([h.score for h in want], abs=1e-5)
            assert all(filter_.matches(h.payload) for h in got)

    async def test_hits_carry_the_full_payload(
        self, stores: tuple[QdrantVectorStore, InMemoryVectorStore]
    ) -> None:
        qdrant, _ = stores
        target = _POINTS[5]

        (hit, *_) = await qdrant.search_dense(_SPEC, target.dense, VectorFilter(_T1), limit=3)

        assert (hit.id, hit.score) == (target.id, pytest.approx(1.0, abs=1e-5))
        assert hit.payload == target.payload

    async def test_another_tenants_identical_vector_is_never_returned(
        self, stores: tuple[QdrantVectorStore, InMemoryVectorStore]
    ) -> None:
        qdrant, _ = stores
        foreign = next(p for p in _POINTS if p.payload.tenant_id == _T2)

        hits = await qdrant.search_dense(_SPEC, foreign.dense, VectorFilter(_T1), limit=50)

        assert foreign.id not in {h.id for h in hits}
        assert all(h.payload.tenant_id == _T1 for h in hits)

    async def test_only_active_versions_are_searched(
        self, stores: tuple[QdrantVectorStore, InMemoryVectorStore]
    ) -> None:
        qdrant, _ = stores
        old = next(p for p in _POINTS if p.payload.tenant_id == _T1 and p.payload.version == _OLD)

        hits = await qdrant.search_dense(
            _SPEC, old.dense, VectorFilter(_T1, versions=(_ACTIVE,)), limit=50
        )

        assert hits and all(h.payload.version == _ACTIVE for h in hits)
        assert old.id not in {h.id for h in hits}

    async def test_url_prefix_scopes_the_search(
        self, stores: tuple[QdrantVectorStore, InMemoryVectorStore]
    ) -> None:
        qdrant, _ = stores
        prefix = "https://acme.example/docs/"

        hits = await qdrant.search_dense(
            _SPEC, _QUERIES[0], VectorFilter(_T1, url_prefix=prefix), limit=50
        )

        expected = sum(
            1 for p in _POINTS if VectorFilter(_T1, url_prefix=prefix).matches(p.payload)
        )
        assert len(hits) == expected > 0
        assert all(h.payload.url.startswith(prefix) for h in hits)  # includes docs/api/

    async def test_score_threshold_drops_weak_hits(
        self, stores: tuple[QdrantVectorStore, InMemoryVectorStore]
    ) -> None:
        qdrant, reference = stores
        query = _QUERIES[1]
        everything = await reference.search_dense(_SPEC, query, VectorFilter(_T1), limit=50)
        threshold = everything[4].score  # the 5th best score

        hits = await qdrant.search_dense(
            _SPEC, query, VectorFilter(_T1), limit=50, score_threshold=threshold - 1e-6
        )

        assert [h.id for h in hits] == [h.id for h in everything[:5]]
        assert all(h.score >= threshold - 1e-5 for h in hits)

    async def test_limits_and_empty_results(
        self, stores: tuple[QdrantVectorStore, InMemoryVectorStore]
    ) -> None:
        qdrant, _ = stores

        top3 = await qdrant.search_dense(_SPEC, _QUERIES[2], VectorFilter(_T1), limit=3)
        none = await qdrant.search_dense(
            _SPEC, _QUERIES[2], VectorFilter(_T1, kb_ids=("kb_" + "9" * 32,))
        )

        assert len(top3) == 3
        assert [h.score for h in top3] == sorted((h.score for h in top3), reverse=True)
        assert none == []

    async def test_an_unknown_collection_is_a_typed_error(
        self, stores: tuple[QdrantVectorStore, InMemoryVectorStore]
    ) -> None:
        qdrant, _ = stores

        with pytest.raises(CollectionNotFoundError):
            await qdrant.search_dense(CollectionSpec("never_created", _DIMS), _QUERIES[0],
                                      VectorFilter(_T1))  # fmt: skip


def test_corpus_is_spread_across_every_dimension() -> None:
    """Guards the fixture: every filter above selects a non-trivial subset."""
    for filter_ in _FILTERS:
        assert 0 < sum(filter_.matches(p.payload) for p in _POINTS) < len(_POINTS), filter_
