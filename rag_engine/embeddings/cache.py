"""Embedding cache wrapper: only texts never embedded in this space reach the provider (Step 64).

A re-crawl replaces a source's whole version (full replace), but most pages are
unchanged: their chunks have the same `embed_text` (title + heading path + text, Step 58).
`CachedEmbedder` wraps any embedder and, for `embed_documents`:

1. keys every input `sha256(model|dims|embed_text)` (`embedding_cache_key`);
2. reads the hits from the store (refreshing them so the TTL keeps them);
3. embeds only the **misses**, each distinct text once (a repeated chunk costs one input);
4. stores the new vectors and returns every vector in input order.

`usage` counts only the provider calls actually made; `cached` counts the inputs served
from the cache. Queries are not cached here (Step 92 has a short-lived query cache).

The cache never fails an embedding: if the store errors on read, everything is a miss;
if it errors on write, the vectors are still returned (logged).
"""

from collections.abc import Mapping, Sequence
from typing import Protocol

from loguru import logger

from rag_engine.config import EmbeddingModelName
from rag_engine.domain import embedding_cache_key
from rag_engine.embeddings.base import Embedder, EmbeddingBatch, Vector
from rag_engine.embeddings.models import EmbeddingSpace
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError


class EmbeddingCacheStore(Protocol):
    """`EmbeddingCacheRepository` in production."""

    async def get_many(self, keys: Sequence[str], *, dimensions: int) -> dict[str, list[float]]: ...

    async def put_many(
        self, vectors: Mapping[str, list[float]], *, model: EmbeddingModelName, dimensions: int
    ) -> int: ...


class CachedEmbedder:
    """An `Embedder` that reuses stored vectors for texts it has embedded before."""

    def __init__(self, inner: Embedder, store: EmbeddingCacheStore) -> None:
        self.inner = inner
        self.store = store

    @property
    def space(self) -> EmbeddingSpace:
        return self.inner.space

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def key(self, text: str) -> str:
        return embedding_cache_key(self.space.model, self.dimensions, text)

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        texts = list(texts)
        if not texts:
            return EmbeddingBatch([], self.space)
        keys = [self.key(text) for text in texts]
        try:
            known = await self.store.get_many(keys, dimensions=self.dimensions)
        except DatabaseUnavailableError as exc:
            logger.warning("embedding cache unreadable, embedding everything: {}", exc)
            known = {}

        misses: dict[str, str] = {}  # key → text, each distinct text once, in input order
        for key, text in zip(keys, texts, strict=True):
            if key not in known and key not in misses:
                misses[key] = text
        fresh = await self.inner.embed_documents(list(misses.values()))
        new: dict[str, Vector] = dict(zip(misses, fresh.vectors, strict=True))
        if new:
            try:
                await self.store.put_many(new, model=self.space.model, dimensions=self.dimensions)
            except DatabaseUnavailableError as exc:
                logger.warning(
                    "embedding cache unwritable, {} vectors not cached: {}", len(new), exc
                )

        vectors = [known[key] if key in known else new[key] for key in keys]
        cached = sum(1 for key in keys if key in known)
        return EmbeddingBatch(vectors, self.space, fresh.usage, cached=cached)

    async def embed_query(self, text: str) -> EmbeddingBatch:
        return await self.inner.embed_query(text)

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def __aenter__(self) -> CachedEmbedder:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
