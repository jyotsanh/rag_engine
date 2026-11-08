"""Qdrant upsert against a real Qdrant (HTTP and gRPC): ids, vectors, payloads, batching."""

import math
from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from rag_engine.config import Settings
from rag_engine.domain import Chunk, chunk_id
from rag_engine.embeddings import SparseVector
from rag_engine.infrastructure import QdrantConnection
from rag_engine.vectorstores import (
    ChunkPayload,
    CollectionNotFoundError,
    CollectionSpec,
    QdrantVectorStore,
    VectorPoint,
    point_id,
)
from tests.conftest import OverrideSettings
from tests.support.fakes import fake_vector

_SPEC = CollectionSpec("upsert_test", 16)
_T, _KB, _S = "ten_" + "1" * 32, "kb_" + "1" * 32, "src_" + "1" * 32
_V1, _V2 = "job_" + "1" * 32, "job_" + "2" * 32
_PAGE = "page_" + "1" * 32


def _chunk(ordinal: int, *, version: str = _V1, text: str | None = None) -> Chunk:
    body = text or f"Chunk {ordinal} about the WX-2048."
    return Chunk(
        id=chunk_id(_PAGE, ordinal), tenant_id=_T, kb_id=_KB, source_id=_S, version=version,
        page_id=_PAGE, url="https://acme.example/docs/wx", title="WX docs",
        heading_path=["Specs", "Speed"], ordinal=ordinal, text=body,
        embed_text=f"WX docs\nSpecs > Speed\n\n{body}", token_count=12, content_hash="0" * 64,
        page_number=ordinal + 1,
    )  # fmt: skip


def _point(chunk: Chunk, sparse: SparseVector | None = None) -> VectorPoint:
    if sparse is None:  # not `or`: an empty SparseVector is falsy (it has __len__)
        sparse = SparseVector((3, 17), (1.25, 0.5))
    return VectorPoint.from_chunk(chunk, fake_vector(chunk.text, 16), sparse)


@pytest.fixture(params=[False, True], ids=["http", "grpc"])
async def client(
    request: pytest.FixtureRequest,
    integration_settings: Settings,
    override_settings: OverrideSettings,
) -> AsyncIterator[AsyncQdrantClient]:
    connection = QdrantConnection(override_settings(QDRANT_PREFER_GRPC=request.param))
    await connection.connect()
    yield connection.client
    await connection.close()


@pytest.fixture
async def store(client: AsyncQdrantClient) -> QdrantVectorStore:
    store = QdrantVectorStore(client)
    await store.ensure_collection(_SPEC)
    return store


async def _count(client: AsyncQdrantClient) -> int:
    return (await client.count(_SPEC.name, exact=True)).count


# ---- #
# Tests
# ---- #


class TestQdrantUpsert_S69:
    """S69 — Qdrant upsert: deterministic ids, both vectors, payload, batching, wait=True."""

    async def test_points_are_stored_exactly(
        self, store: QdrantVectorStore, client: AsyncQdrantClient
    ) -> None:
        chunk = _chunk(0)
        point = _point(chunk)

        assert await store.upsert(_SPEC, [point]) == 1

        (record,) = await client.retrieve(_SPEC.name, [point.id], with_vectors=True)
        assert record.id == point_id(_V1, chunk.id)
        assert ChunkPayload.model_validate(record.payload) == ChunkPayload.from_chunk(chunk)
        vectors = record.vector
        assert isinstance(vectors, dict)
        # Qdrant normalises cosine vectors on write
        norm = math.sqrt(sum(v * v for v in point.dense))
        assert vectors["dense"] == pytest.approx([v / norm for v in point.dense], abs=1e-6)
        sparse = vectors["bm25"]
        assert isinstance(sparse, models.SparseVector)
        assert (list(sparse.indices), list(sparse.values)) == ([3, 17], [1.25, 0.5])

    async def test_points_are_visible_as_soon_as_upsert_returns(
        self, store: QdrantVectorStore, client: AsyncQdrantClient
    ) -> None:
        await store.upsert(_SPEC, [_point(_chunk(n)) for n in range(25)])

        assert await _count(client) == 25  # wait=True: no polling needed

    async def test_reupserting_overwrites_instead_of_duplicating(
        self, store: QdrantVectorStore, client: AsyncQdrantClient
    ) -> None:
        await store.upsert(_SPEC, [_point(_chunk(n)) for n in range(3)])

        revised = _chunk(1, text="Chunk 1, revised.")
        await store.upsert(_SPEC, [_point(revised)])

        assert await _count(client) == 3
        (record,) = await client.retrieve(_SPEC.name, [point_id(_V1, revised.id)])
        assert record.payload is not None and record.payload["text"] == "Chunk 1, revised."

    async def test_versions_never_collide(
        self, store: QdrantVectorStore, client: AsyncQdrantClient
    ) -> None:
        await store.upsert(_SPEC, [_point(_chunk(0, version=_V1)), _point(_chunk(0, version=_V2))])

        assert await _count(client) == 2  # same chunk id, different versions: two points

    async def test_batches_follow_the_configured_size(
        self, client: AsyncQdrantClient, override_settings: OverrideSettings
    ) -> None:
        store = QdrantVectorStore(client, config=override_settings(QDRANT_UPSERT_BATCH_SIZE=4))
        await store.ensure_collection(_SPEC)
        sizes: list[int] = []
        original = client.upsert

        async def spy(collection_name: str, points, **kwargs):
            sizes.append(len(points))
            assert kwargs["wait"] is True
            return await original(collection_name, points=points, **kwargs)

        client.upsert = spy  # ty: ignore[invalid-assignment]
        try:
            await store.upsert(_SPEC, [_point(_chunk(n)) for n in range(10)])
        finally:
            client.upsert = original  # ty: ignore[invalid-assignment]

        assert sizes == [4, 4, 2]
        assert await _count(client) == 10

    async def test_an_empty_sparse_vector_is_omitted(
        self, store: QdrantVectorStore, client: AsyncQdrantClient
    ) -> None:
        point = _point(_chunk(0), SparseVector((), ()))  # a stopword-only chunk

        await store.upsert(_SPEC, [point])

        (record,) = await client.retrieve(_SPEC.name, [point.id], with_vectors=True)
        assert isinstance(record.vector, dict) and "bm25" not in record.vector

    async def test_a_dense_only_collection(self, client: AsyncQdrantClient) -> None:
        spec = CollectionSpec("upsert_dense_only", 16, sparse=False)
        store = QdrantVectorStore(client)
        await store.ensure_collection(spec)
        chunk = _chunk(0)

        await store.upsert(spec, [VectorPoint.from_chunk(chunk, fake_vector("x", 16))])

        assert (await client.count(spec.name, exact=True)).count == 1

    async def test_an_unknown_collection_is_a_typed_error(self, client: AsyncQdrantClient) -> None:
        store = QdrantVectorStore(client)

        with pytest.raises(CollectionNotFoundError):
            await store.upsert(CollectionSpec("never_created", 16), [_point(_chunk(0))])
