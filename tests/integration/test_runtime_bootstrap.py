"""The runtime bootstraps the vector store at startup (real MongoDB + Qdrant)."""

import pytest
from qdrant_client import models

from rag_engine.config import EmbeddingModelName, Settings
from rag_engine.domain import KnowledgeBase, ResourceStatus
from rag_engine.infrastructure import MongoConnection, check_readiness
from rag_engine.repositories import KnowledgeBaseRepository
from rag_engine.runtime import AppRuntime
from rag_engine.vectorstores import CollectionMismatchError, QdrantVectorStore

_TENANT = "ten_" + "1" * 32


async def _knowledge_bases(config: Settings) -> None:
    mongo = MongoConnection(config)
    await mongo.connect()
    await mongo.ensure_indexes([KnowledgeBase])
    repo = KnowledgeBaseRepository(mongo.db)
    for name, model, dims, status in [
        ("docs", EmbeddingModelName.TEXT_EMBEDDING_3_SMALL, 512, ResourceStatus.ACTIVE),
        ("big", EmbeddingModelName.TEXT_EMBEDDING_3_LARGE, 1024, ResourceStatus.ACTIVE),
        ("same", EmbeddingModelName.TEXT_EMBEDDING_3_SMALL, 512, ResourceStatus.ACTIVE),
        ("old", EmbeddingModelName.TEXT_EMBEDDING_3_LARGE, 256, ResourceStatus.DELETING),
    ]:
        kb = KnowledgeBase(
            tenant_id=_TENANT, name=name, status=status,
            embedding_model=model, embedding_dimensions=dims,
        )  # fmt: skip
        await repo.insert(kb)
    await mongo.close()


_EXPECTED = {
    "chunks__text-embedding-3-small__512",
    "chunks__text-embedding-3-large__1024",
    "chunks__text-embedding-3-large__256",  # a deleting KB still needs it for its cascade
    "chunks__text-embedding-3-small__1536",  # the default model's native space
}


# ---- #
# Tests
# ---- #


class TestRuntimeBootstrap_S74:
    """S74 — Startup ensures every collection in use; readiness tracks them."""

    async def test_startup_creates_every_collection_in_use(
        self, integration_settings: Settings
    ) -> None:
        await _knowledge_bases(integration_settings)
        runtime = AppRuntime()

        await runtime.start()
        try:
            names = {c.name for c in (await runtime.qdrant.client.get_collections()).collections}
            assert names == _EXPECTED
            assert isinstance(runtime.vector_store, QdrantVectorStore)
            report = await check_readiness(runtime.health_checks(), budget_seconds=5)
            assert report.ready
            assert set(report.dependencies) == {"mongo", "redis", "qdrant", "vector_collections"}
        finally:
            await runtime.stop()

        second = AppRuntime()
        await second.start()  # a restart: everything exists, nothing is recreated
        try:
            assert second.collections is not None
            assert {c.name for c in second.collections.collections} == _EXPECTED
        finally:
            await second.stop()

    async def test_a_deleted_collection_makes_the_service_not_ready(
        self, integration_settings: Settings
    ) -> None:
        runtime = AppRuntime()
        await runtime.start()
        try:
            await runtime.qdrant.client.delete_collection("chunks__text-embedding-3-small__1536")

            report = await check_readiness(runtime.health_checks(), budget_seconds=5)

            assert not report.ready
            assert report.dependencies["vector_collections"].error == "vector_collection_not_found"
            assert report.dependencies["qdrant"].status == "ok"  # Qdrant itself is fine
        finally:
            await runtime.stop()

    async def test_a_mismatched_collection_fails_startup(
        self, integration_settings: Settings
    ) -> None:
        runtime = AppRuntime()
        await runtime.qdrant.connect()
        await runtime.qdrant.client.create_collection(  # the right name, the wrong size
            "chunks__text-embedding-3-small__1536",
            vectors_config={"dense": models.VectorParams(size=8, distance=models.Distance.COSINE)},
        )
        await runtime.qdrant.close()

        with pytest.raises(CollectionMismatchError):
            await runtime.start()

        assert runtime.mongo._client is None and runtime.qdrant._client is None  # all closed
