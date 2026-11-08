"""Embedding cache in real MongoDB, alone and behind `CachedEmbedder`."""

import math
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from rag_engine.config import EmbeddingModelName, Settings
from rag_engine.domain import (
    EMBEDDING_CACHE_TTL_SECONDS,
    EmbeddingCacheEntry,
    embedding_cache_key,
    utc_now,
)
from rag_engine.embeddings import CachedEmbedder, resolve_space
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import EmbeddingCacheRepository
from tests.support.fakes import FakeEmbedder, fake_vector

_MODEL = EmbeddingModelName.TEXT_EMBEDDING_3_SMALL
_SPACE = resolve_space(_MODEL, 256)


def _key(text: str, dimensions: int = 256) -> str:
    return embedding_cache_key(_MODEL, dimensions, text)


def _close(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-7)
                                    for x, y in zip(a, b, strict=True))  # fmt: skip


@pytest.fixture
async def repo(integration_settings: Settings) -> AsyncIterator[EmbeddingCacheRepository]:
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([EmbeddingCacheEntry])
    yield EmbeddingCacheRepository(mongo.db)
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestEmbeddingCacheRepository_S64:
    """S64 — Embedding cache store: float32 vectors by key, TTL refreshed on use."""

    async def test_vectors_round_trip_by_key(self, repo: EmbeddingCacheRepository) -> None:
        vectors = {_key(t): fake_vector(t, 256) for t in ("a", "b")}

        written = await repo.put_many(vectors, model=_MODEL, dimensions=256)
        found = await repo.get_many([_key("b"), _key("missing"), _key("a"), _key("b")],
                                    dimensions=256)  # fmt: skip

        assert written == 2
        assert set(found) == {_key("a"), _key("b")}
        assert all(_close(found[k], vectors[k]) for k in found)
        stored = await repo.get(_key("a"))
        assert stored is not None and len(stored.vector) == 256 * 4  # packed float32

    async def test_a_dimension_mismatch_is_a_miss(self, repo: EmbeddingCacheRepository) -> None:
        await repo.put_many({_key("a"): fake_vector("a", 256)}, model=_MODEL, dimensions=256)

        assert await repo.get_many([_key("a")], dimensions=512) == {}
        with pytest.raises(ValueError, match="256 dimensions"):
            await repo.put_many({_key("b"): [0.1] * 10}, model=_MODEL, dimensions=256)

    async def test_hits_refresh_last_used(self, repo: EmbeddingCacheRepository) -> None:
        await repo.put_many({_key("a"): fake_vector("a", 256)}, model=_MODEL, dimensions=256)
        old = utc_now() - timedelta(days=60)
        await repo.collection.update_one({"_id": _key("a")}, {"$set": {"last_used_at": old}})

        await repo.get_many([_key("a")], dimensions=256)

        entry = await repo.get(_key("a"))
        assert entry is not None and entry.last_used_at > old + timedelta(days=59)

    async def test_more_keys_than_one_query_slice(self, repo: EmbeddingCacheRepository) -> None:
        vectors = {_key(f"t{i}", 4): [float(i), 0.0, 0.0, 1.0] for i in range(2500)}

        await repo.put_many(vectors, model=_MODEL, dimensions=4)
        found = await repo.get_many(list(vectors), dimensions=4)

        assert len(found) == 2500
        assert found[_key("t1234", 4)] == [1234.0, 0.0, 0.0, 1.0]

    async def test_ttl_index(self, repo: EmbeddingCacheRepository) -> None:
        indexes = await repo.collection.index_information()

        ttl = indexes["embedding_cache_ttl"]
        assert ttl["key"] == [("last_used_at", 1)]
        assert ttl["expireAfterSeconds"] == EMBEDDING_CACHE_TTL_SECONDS == 90 * 24 * 60 * 60

    async def test_a_recrawl_of_unchanged_content_makes_no_provider_calls(
        self, repo: EmbeddingCacheRepository
    ) -> None:
        texts = [f"Guide\\nSection {i}\\n\\nBody {i}." for i in range(30)]
        first_inner = FakeEmbedder(_SPACE)
        await CachedEmbedder(first_inner, repo).embed_documents(texts)

        changed = [*texts[:29], "Guide\\nSection 29\\n\\nBody 29, revised."]
        second_inner = FakeEmbedder(_SPACE)  # a later job, a new embedder
        batch = await CachedEmbedder(second_inner, repo).embed_documents(changed)

        assert [c[0] for c in second_inner.calls] == [[changed[29]]]  # only the edited chunk
        assert batch.cached == 29
        assert all(
            _close(v, fake_vector(t, 256)) for v, t in zip(batch.vectors, changed, strict=True)
        )
