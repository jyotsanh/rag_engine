"""Embedding cache wrapper: only new texts reach the provider; order, dedupe, degradation."""

import math

import pytest

from rag_engine.domain import embedding_cache_key, pack_vector, unpack_vector
from rag_engine.embeddings import CachedEmbedder, EmbeddingUsageCounts, InputType, resolve_space
from tests.support.fakes import FakeEmbedder, InMemoryEmbeddingCache, fake_vector

_SPACE = resolve_space("text-embedding-3-small", 256)


def _cached(store: InMemoryEmbeddingCache | None = None, space=_SPACE):
    inner = FakeEmbedder(space)
    return CachedEmbedder(inner, store or InMemoryEmbeddingCache()), inner


# ---- #
# Tests
# ---- #


class TestEmbeddingCache_S64:
    """S64 — Embedding cache: a re-crawl of unchanged content doesn't re-pay the provider."""

    async def test_a_second_identical_run_costs_nothing(self) -> None:
        store = InMemoryEmbeddingCache()
        embedder, inner = _cached(store)
        texts = ["Docs\\n\\nInstall it.", "Docs\\n\\nConfigure it.", "Docs\\n\\nRun it."]

        first = await embedder.embed_documents(texts)
        second = await embedder.embed_documents(texts)

        assert first.vectors == second.vectors == [fake_vector(t, 256) for t in texts]
        assert (first.cached, first.usage.requests) == (0, 1)
        assert (second.cached, second.usage) == (3, EmbeddingUsageCounts())
        assert len(inner.calls) == 1  # the second run never reached the provider

    async def test_only_misses_are_embedded_and_order_is_kept(self) -> None:
        store = InMemoryEmbeddingCache()
        embedder, inner = _cached(store)
        await embedder.embed_documents(["old 1", "old 2"])
        inner.calls.clear()

        batch = await embedder.embed_documents(["new 1", "old 2", "new 2", "old 1"])

        assert inner.calls == [(["new 1", "new 2"], InputType.DOCUMENT)]
        assert batch.vectors == [fake_vector(t, 256) for t in ["new 1", "old 2", "new 2", "old 1"]]
        assert batch.cached == 2

    async def test_a_repeated_text_is_embedded_once(self) -> None:
        embedder, inner = _cached()

        batch = await embedder.embed_documents(["same", "other", "same", "same"])

        assert inner.calls == [(["same", "other"], InputType.DOCUMENT)]
        assert batch.vectors[0] == batch.vectors[2] == batch.vectors[3]
        assert len(batch) == 4 and batch.cached == 0

    async def test_spaces_never_share_vectors(self) -> None:
        store = InMemoryEmbeddingCache()
        small, _ = _cached(store)
        reduced, inner = _cached(store, resolve_space("text-embedding-3-small", 512))
        await small.embed_documents(["shared text"])

        batch = await reduced.embed_documents(["shared text"])

        assert batch.cached == 0 and len(inner.calls) == 1
        assert len(batch.vector) == 512

    async def test_an_unreadable_cache_embeds_everything(self) -> None:
        store = InMemoryEmbeddingCache()
        embedder, inner = _cached(store)
        await embedder.embed_documents(["a", "b"])
        store.fail_reads = True

        batch = await embedder.embed_documents(["a", "b"])

        assert batch.cached == 0 and len(inner.calls) == 2
        assert batch.vectors == [fake_vector("a", 256), fake_vector("b", 256)]

    async def test_an_unwritable_cache_still_returns_vectors(self) -> None:
        store = InMemoryEmbeddingCache()
        store.fail_writes = True
        embedder, _ = _cached(store)

        batch = await embedder.embed_documents(["a"])

        assert batch.vector == fake_vector("a", 256)
        assert store.vectors == {}

    async def test_queries_and_empty_batches_bypass_the_cache(self) -> None:
        store = InMemoryEmbeddingCache()
        embedder, inner = _cached(store)

        query = await embedder.embed_query("how fast?")
        empty = await embedder.embed_documents([])

        assert query.vector == fake_vector("how fast?", 256)
        assert inner.calls == [(["how fast?"], InputType.QUERY)]
        assert store.reads == [] and store.vectors == {}
        assert empty.vectors == []

    async def test_it_is_a_drop_in_embedder(self) -> None:
        embedder, inner = _cached()

        async with embedder:
            assert (embedder.space, embedder.dimensions) == (_SPACE, 256)

        assert inner.closed

    def test_cache_keys(self) -> None:
        key = embedding_cache_key("text-embedding-3-small", 256, "text")

        assert len(key) == 64 and key == embedding_cache_key("text-embedding-3-small", 256, "text")
        assert key != embedding_cache_key("text-embedding-3-small", 512, "text")
        assert key != embedding_cache_key("text-embedding-3-large", 256, "text")
        assert key != embedding_cache_key("text-embedding-3-small", 256, "text ")
        with pytest.raises(ValueError):
            embedding_cache_key("ada", 256, "text")

    def test_vectors_pack_to_float32(self) -> None:
        vector = fake_vector("x", 1536)

        data = pack_vector(vector)
        restored = unpack_vector(data)

        assert len(data) == 1536 * 4
        assert all(
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
            for a, b in zip(vector, restored, strict=True)
        )
        assert unpack_vector(pack_vector(restored)) == restored  # float32 values are exact
