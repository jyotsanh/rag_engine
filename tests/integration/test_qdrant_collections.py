"""Qdrant collection management against a real Qdrant (HTTP and gRPC transports)."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient, models

from rag_engine.config import Settings
from rag_engine.embeddings import resolve_space
from rag_engine.infrastructure import QdrantConnection
from rag_engine.vectorstores import (
    PAYLOAD_INDEXES,
    CollectionMismatchError,
    CollectionSpec,
    QdrantVectorStore,
    VectorStoreDispatcher,
    VectorStoreUnavailableError,
)
from tests.conftest import OverrideSettings

_SPEC = CollectionSpec.for_space(resolve_space("text-embedding-3-small", 256))


@pytest.fixture(params=[False, True], ids=["http", "grpc"])
async def client(
    request: pytest.FixtureRequest,
    integration_settings: Settings,
    override_settings: OverrideSettings,
) -> AsyncIterator[AsyncQdrantClient]:
    config = override_settings(QDRANT_PREFER_GRPC=request.param)
    connection = QdrantConnection(config)
    await connection.connect()
    yield connection.client
    await connection.close()


@pytest.fixture
def store(client: AsyncQdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(client)


# ---- #
# Tests
# ---- #


class TestQdrantCollections_S68:
    """S68 — Qdrant collections: dense cosine + bm25 IDF, tenant-aware payload indexes."""

    async def test_a_collection_gets_the_full_schema(
        self, store: QdrantVectorStore, client: AsyncQdrantClient
    ) -> None:
        created = await store.ensure_collection(_SPEC)

        info = await client.get_collection(_SPEC.name)
        assert created is True
        assert _SPEC.name == "chunks__text-embedding-3-small__256"
        dense = info.config.params.vectors["dense"]  # ty: ignore[not-subscriptable]
        assert (dense.size, dense.distance) == (256, models.Distance.COSINE)
        sparse = info.config.params.sparse_vectors["bm25"]  # ty: ignore[not-subscriptable]
        assert sparse.modifier == models.Modifier.IDF
        schema = info.payload_schema
        assert set(schema) == set(PAYLOAD_INDEXES)
        assert all(schema[f].data_type == models.PayloadSchemaType.KEYWORD for f in schema)
        tenant, url = schema["tenant_id"].params, schema["url"].params
        assert isinstance(tenant, models.KeywordIndexParams) and tenant.is_tenant is True
        assert isinstance(url, models.KeywordIndexParams) and url.prefix is True

    async def test_ensuring_again_is_a_no_op(self, store: QdrantVectorStore) -> None:
        assert await store.ensure_collection(_SPEC) is True
        assert await store.ensure_collection(_SPEC) is False

    async def test_concurrent_workers_create_it_once(self, store: QdrantVectorStore) -> None:
        results = await asyncio.gather(*(store.ensure_collection(_SPEC) for _ in range(8)))

        assert results.count(True) >= 1  # a lost race reports False, never an error
        assert await store.ensure_collection(_SPEC) is False

    async def test_a_dense_only_collection(
        self, store: QdrantVectorStore, client: AsyncQdrantClient
    ) -> None:
        spec = CollectionSpec("dense_only_test", 8, sparse=False)

        await store.ensure_collection(spec)

        info = await client.get_collection(spec.name)
        assert not info.config.params.sparse_vectors

    @pytest.mark.parametrize(
        ("vectors", "sparse", "problem"),
        [
            ({"dense": models.VectorParams(size=512, distance=models.Distance.COSINE)},
             {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)}, "dense size 512"),
            ({"dense": models.VectorParams(size=256, distance=models.Distance.DOT)},
             {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)}, "distance"),
            ({"dense": models.VectorParams(size=256, distance=models.Distance.COSINE)},
             None, "no sparse vector"),
            ({"dense": models.VectorParams(size=256, distance=models.Distance.COSINE)},
             {"bm25": models.SparseVectorParams()}, "sparse modifier"),
            ({"other": models.VectorParams(size=256, distance=models.Distance.COSINE)},
             None, "no named dense vector"),
        ],
    )  # fmt: skip
    async def test_an_existing_collection_that_differs_is_refused(
        self,
        store: QdrantVectorStore,
        client: AsyncQdrantClient,
        vectors: dict,
        sparse: dict | None,
        problem: str,
    ) -> None:
        await client.create_collection(_SPEC.name, vectors_config=vectors,
                                       sparse_vectors_config=sparse)  # fmt: skip

        with pytest.raises(CollectionMismatchError, match=problem) as exc:
            await store.ensure_collection(_SPEC)

        assert exc.value.details["collection"] == _SPEC.name

    async def test_missing_payload_indexes_are_added(
        self, store: QdrantVectorStore, client: AsyncQdrantClient
    ) -> None:
        await client.create_collection(
            _SPEC.name,
            vectors_config={
                "dense": models.VectorParams(size=256, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )  # fmt: skip  (an "older release" collection: right vectors, no indexes)

        assert await store.ensure_collection(_SPEC) is False

        info = await client.get_collection(_SPEC.name)
        assert set(info.payload_schema) == set(PAYLOAD_INDEXES)

    def test_the_dispatcher_builds_the_qdrant_store(self, client: AsyncQdrantClient) -> None:
        store = VectorStoreDispatcher.create(client=client)

        assert isinstance(store, QdrantVectorStore) and store.client is client


class TestQdrantUnavailable_S68:
    """S68 — an unreachable Qdrant is a typed 503, not a client exception."""

    @pytest.mark.parametrize("grpc", [False, True], ids=["http", "grpc"])
    async def test_unreachable_qdrant(self, grpc: bool) -> None:
        client = AsyncQdrantClient(host="127.0.0.1", port=1, grpc_port=2, prefer_grpc=grpc,
                                   timeout=2, check_compatibility=False)  # fmt: skip
        store = QdrantVectorStore(client)

        with pytest.raises(VectorStoreUnavailableError):
            await store.ensure_collection(_SPEC)

        await client.close()
