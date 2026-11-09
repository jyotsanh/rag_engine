"""Qdrant hybrid search (dense + BM25, RRF) against a real Qdrant and the reference."""

import itertools
from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.embeddings import SparseVector, get_sparse_embedder
from rag_engine.infrastructure import QdrantConnection
from rag_engine.vectorstores import (
    ChunkPayload,
    CollectionSpec,
    QdrantVectorStore,
    VectorFilter,
    VectorPoint,
    point_id,
)
from tests.conftest import OverrideSettings
from tests.support.fakes import InMemoryVectorStore, fake_vector

_DIMS = 16
_SPEC = CollectionSpec("hybrid_search_test", _DIMS)
_T1, _T2 = "ten_" + "1" * 32, "ten_" + "2" * 32
_KB, _SRC, _V = "kb_" + "1" * 32, "src_" + "1" * 32, "job_" + "1" * 32

_TEXTS = [
    "The controller installs on Linux and macOS in under five minutes.",
    "Configure the heartbeat interval and the listen port in config.toml.",
    "Calibration takes about ninety seconds per widget.",
    "The WX-2048 sorter is rated for 1200 items per minute.",
    "Firmware updates are delivered over the air to every widget in the fleet.",
    "Register each widget with its serial number printed on the housing.",
    "Error E-417 means the torque sensor lost its reference position.",
    "Pricing starts at 1899 USD for the modular sorter.",
    "Use four M6 bolts torqued to 9 Nm when mounting the widget.",
    "The REST API lives under /v2 and requires a bearer token.",
    "Logs are written as JSON lines unless log_format is set to text.",
    "Release v3.1 ships fleet-wide over the air updates in August.",
]


async def _corpus() -> list[VectorPoint]:
    """Each text twice: once per tenant (identical content must never cross tenants)."""
    sparse = await get_sparse_embedder().embed_documents(_TEXTS)
    points = []
    for tenant, (n, text) in itertools.product((_T1, _T2), enumerate(_TEXTS)):
        payload = ChunkPayload(
            tenant_id=tenant, kb_id=_KB, source_id=_SRC, version=_V,
            chunk_id=f"chk_{tenant[-1]}{n:031x}", page_id=f"page_{n}",
            url=f"https://acme.example/{'docs' if n % 2 else 'blog'}/{n}", ordinal=0, text=text,
        )  # fmt: skip
        points.append(VectorPoint(point_id(_V, payload.chunk_id),
                                  fake_vector(text, _DIMS), payload, sparse[n]))  # fmt: skip
    return points


@pytest.fixture(params=[False, True], ids=["http", "grpc"])
async def stores(
    request: pytest.FixtureRequest,
    integration_settings: Settings,
    override_settings: OverrideSettings,
) -> AsyncIterator[tuple[QdrantVectorStore, InMemoryVectorStore, list[VectorPoint]]]:
    connection = QdrantConnection(override_settings(QDRANT_PREFER_GRPC=request.param))
    await connection.connect()
    points = await _corpus()
    qdrant, reference = QdrantVectorStore(connection.client), InMemoryVectorStore()
    for store in (qdrant, reference):
        await store.ensure_collection(_SPEC)
        await store.upsert(_SPEC, points)
    yield qdrant, reference, points
    await connection.close()


async def _query(text: str) -> tuple[list[float], SparseVector]:
    return fake_vector(f"query: {text}", _DIMS), await get_sparse_embedder().embed_query(text)


# ---- #
# Tests
# ---- #


class TestQdrantHybridSearch_S72:
    """S72 — Qdrant hybrid search: dense + BM25 prefetch fused by RRF, filtered per tenant."""

    async def test_an_exact_term_surfaces_even_when_dense_misses_it(self, stores) -> None:
        qdrant, _, _ = stores
        target = "Error E-417 means the torque sensor lost its reference position."
        decoy = "Release v3.1 ships fleet-wide over the air updates in August."
        # the dense query *is* the decoy's vector: dense search ranks the decoy first
        dense = fake_vector(decoy, _DIMS)
        sparse = await get_sparse_embedder().embed_query("what does error E-417 mean")

        dense_only = await qdrant.search_dense(_SPEC, dense, VectorFilter(_T1), limit=1)
        hybrid = await qdrant.search_hybrid(_SPEC, dense, sparse, VectorFilter(_T1), limit=3)

        assert dense_only[0].payload.text == decoy
        # RRF: target = 1/60 (sparse rank 0) + its dense share; decoy = 1/60 (dense only)
        assert hybrid[0].payload.text == target

    async def test_results_match_the_reference(self, stores) -> None:
        qdrant, reference, _ = stores
        queries = ["WX-2048 sorter speed", "over the air firmware updates", "mounting bolts Nm",
                   "bearer token for the REST API", "pricing USD"]  # fmt: skip
        filters = [VectorFilter(_T1), VectorFilter(_T1, url_prefix="https://acme.example/docs/")]

        for text, filter_ in itertools.product(queries, filters):
            dense, sparse = await _query(text)
            got = await qdrant.search_hybrid(_SPEC, dense, sparse, filter_, limit=5)
            want = await reference.search_hybrid(_SPEC, dense, sparse, filter_, limit=5)

            assert [h.id for h in got] == [h.id for h in want], (text, filter_)
            assert [h.score for h in got] == pytest.approx([h.score for h in want], rel=1e-4)

    async def test_identical_content_never_crosses_tenants(self, stores) -> None:
        qdrant, _, _ = stores
        dense, sparse = await _query("WX-2048 rated items per minute")

        hits = await qdrant.search_hybrid(_SPEC, dense, sparse, VectorFilter(_T2), limit=24)

        assert hits and all(h.payload.tenant_id == _T2 for h in hits)
        assert len(hits) == len(_TEXTS)  # every T2 point once, no T1 duplicate

    async def test_a_stopword_only_query_falls_back_to_dense_ranking(self, stores) -> None:
        qdrant, _, _ = stores
        dense, sparse = await _query("the and of to")
        assert len(sparse) == 0

        hybrid = await qdrant.search_hybrid(_SPEC, dense, sparse, VectorFilter(_T1), limit=5)
        dense_only = await qdrant.search_dense(_SPEC, dense, VectorFilter(_T1), limit=5)

        assert [h.id for h in hybrid] == [h.id for h in dense_only]

    async def test_limit_and_prefetch(self, stores) -> None:
        qdrant, _, _ = stores
        dense, sparse = await _query("widget fleet updates")

        hits = await qdrant.search_hybrid(_SPEC, dense, sparse, VectorFilter(_T1), limit=2,
                                          prefetch_limit=4)  # fmt: skip

        assert len(hits) == 2
        assert hits[0].score >= hits[1].score
