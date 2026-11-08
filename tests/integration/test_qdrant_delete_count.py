"""Qdrant delete & count by filter against a real Qdrant, checked against the reference."""

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
from tests.support.fakes import fake_vector

_SPEC = CollectionSpec("delete_count_test", 8)
_TENANTS = ("ten_" + "1" * 32, "ten_" + "2" * 32)
_KBS = ("kb_" + "1" * 32, "kb_" + "2" * 32)
_SOURCES = ("src_" + "1" * 32, "src_" + "2" * 32)
_VERSIONS = ("job_" + "1" * 32, "job_" + "2" * 32)
_T1, _T2 = _TENANTS
_K1 = _KBS[0]
_S1 = _SOURCES[0]
_V1, _V2 = _VERSIONS


def _grid() -> list[VectorPoint]:
    """2 tenants x 2 KBs x 2 sources x 2 versions x 4 pages (docs/blog URLs) = 64 points."""
    points = []
    for n, (tenant, kb, source, version, page) in enumerate(
        itertools.product(_TENANTS, _KBS, _SOURCES, _VERSIONS, range(4))
    ):
        section = "docs" if page % 2 == 0 else "blog"
        payload = ChunkPayload(
            tenant_id=tenant, kb_id=kb, source_id=source, version=version,
            chunk_id=f"chk_{n:032x}", page_id=f"page_{page}",
            url=f"https://acme.example/{section}/p{page}", ordinal=0, text=f"point {n}",
        )  # fmt: skip
        points.append(VectorPoint(point_id(version, payload.chunk_id), fake_vector(str(n), 8),
                                  payload, SparseVector((n,), (1.0,))))  # fmt: skip
    return points


_POINTS = _grid()
_FILTERS = [
    VectorFilter(_T1),
    VectorFilter(_T2),
    VectorFilter(_T1, kb_ids=(_K1,)),
    VectorFilter(_T1, kb_ids=_KBS),
    VectorFilter(_T1, source_ids=(_S1,)),
    VectorFilter(_T1, versions=(_V2,)),
    VectorFilter(_T1, kb_ids=(_K1,), source_ids=(_S1,), versions=(_V1,)),
    VectorFilter(_T1, page_ids=("page_1", "page_3")),
    VectorFilter(_T1, url_prefix="https://acme.example/docs/"),
    VectorFilter(_T1, url_prefix="https://acme.example/docs/p2"),
    VectorFilter(_T1, url_prefix="https://other.example/"),
    VectorFilter(_T1, kb_ids=("kb_" + "9" * 32,)),
]


def _expected(points: list[VectorPoint], filter_: VectorFilter) -> int:
    return sum(filter_.matches(p.payload) for p in points)


@pytest.fixture(params=[False, True], ids=["http", "grpc"])
async def store(
    request: pytest.FixtureRequest,
    integration_settings: Settings,
    override_settings: OverrideSettings,
) -> AsyncIterator[QdrantVectorStore]:
    connection = QdrantConnection(override_settings(QDRANT_PREFER_GRPC=request.param))
    await connection.connect()
    store = QdrantVectorStore(connection.client)
    await store.ensure_collection(_SPEC)
    await store.upsert(_SPEC, _POINTS)
    yield store
    await connection.close()


async def _remaining_matches_reference(store: QdrantVectorStore, deleted: VectorFilter) -> None:
    left = [p for p in _POINTS if not deleted.matches(p.payload)]
    for filter_ in _FILTERS:
        assert await store.count(_SPEC, filter_) == _expected(left, filter_), filter_


# ---- #
# Tests
# ---- #


class TestQdrantDeleteCount_S70:
    """S70 — Qdrant delete & count: version/source/KB/tenant filters, exact and isolated."""

    async def test_counts_match_the_reference_semantics(self, store: QdrantVectorStore) -> None:
        for filter_ in _FILTERS:
            assert await store.count(_SPEC, filter_) == _expected(_POINTS, filter_), filter_

        assert await store.count(_SPEC, VectorFilter(_T1)) == 32
        assert (
            await store.count(_SPEC, VectorFilter(_T1, url_prefix="https://acme.example/docs/"))
            == 16
        )

    @pytest.mark.parametrize(
        "deleted",
        [
            VectorFilter(_T1, source_ids=(_S1,), versions=(_V1,)),  # a failed job's version
            VectorFilter(_T1, source_ids=(_S1,)),  # a deleted source
            VectorFilter(_T1, kb_ids=(_K1,)),  # a deleted knowledge base
            VectorFilter(_T1),  # a deleted tenant
            VectorFilter(_T1, versions=(_V1,), page_ids=("page_2",)),  # a re-chunked page
        ],
        ids=["version", "source", "kb", "tenant", "page"],
    )
    async def test_deletes_remove_exactly_the_match(
        self, store: QdrantVectorStore, deleted: VectorFilter
    ) -> None:
        before = _expected(_POINTS, deleted)
        assert before > 0

        await store.delete_by_filter(_SPEC, deleted)

        assert await store.count(_SPEC, deleted) == 0  # wait=True: immediately consistent
        await _remaining_matches_reference(store, deleted)
        assert await store.count(_SPEC, VectorFilter(_T2)) == 32  # the other tenant is untouched

    async def test_deleting_twice_is_harmless(self, store: QdrantVectorStore) -> None:
        deleted = VectorFilter(_T1, versions=(_V2,))

        await store.delete_by_filter(_SPEC, deleted)
        await store.delete_by_filter(_SPEC, deleted)

        await _remaining_matches_reference(store, deleted)

    async def test_an_unknown_collection_is_a_typed_error(self, store: QdrantVectorStore) -> None:
        missing = CollectionSpec("never_created", 8)

        with pytest.raises(CollectionNotFoundError):
            await store.count(missing, VectorFilter(_T1))
        with pytest.raises(CollectionNotFoundError):
            await store.delete_by_filter(missing, VectorFilter(_T1))
