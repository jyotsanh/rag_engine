"""Vector collection bootstrap with the in-memory reference store."""

import asyncio

import pytest

from rag_engine.embeddings import resolve_space
from rag_engine.vectorstores import CollectionNotFoundError, CollectionSpec, VectorCollections
from tests.support.fakes import InMemoryVectorStore

_SMALL = resolve_space("text-embedding-3-small")
_REDUCED = resolve_space("text-embedding-3-small", 512)
_LARGE = resolve_space("text-embedding-3-large")


class _CountingStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.ensures: list[str] = []

    async def _ensure_collection(self, spec: CollectionSpec) -> bool:
        self.ensures.append(spec.name)
        await asyncio.sleep(0.01)  # let concurrent callers interleave
        return await super()._ensure_collection(spec)


# ---- #
# Tests
# ---- #


class TestVectorBootstrap_S74:
    """S74 — Vector collections: bootstrap at startup, ensure on first use, readiness."""

    async def test_bootstrap_ensures_every_space(self) -> None:
        store = InMemoryVectorStore()
        await store.ensure_collection(CollectionSpec.for_space(_LARGE))  # created earlier
        collections = VectorCollections(store)

        report = await collections.bootstrap({_SMALL, _REDUCED, _LARGE})

        assert report.created == (  # deterministic order: by model, then dimensions
            "chunks__text-embedding-3-small__512",
            "chunks__text-embedding-3-small__1536",
        )
        assert report.existing == ("chunks__text-embedding-3-large__3072",)
        assert {c.name for c in collections.collections} == set(store.collections)
        assert all(c.sparse for c in collections.collections)

    async def test_ensure_is_cached_and_single_flight(self) -> None:
        store = _CountingStore()
        collections = VectorCollections(store)

        specs = await asyncio.gather(*(collections.ensure(_REDUCED) for _ in range(10)))
        again = await collections.ensure(_REDUCED)

        assert store.ensures == ["chunks__text-embedding-3-small__512"]  # once, not 11 times
        assert len({s.name for s in specs}) == 1 and again == specs[0]

    async def test_readiness_fails_when_a_collection_disappears(self) -> None:
        store = InMemoryVectorStore()
        collections = VectorCollections(store)
        await collections.bootstrap({_SMALL})
        await collections.check()  # ready

        del store.collections["chunks__text-embedding-3-small__1536"]  # deleted under us

        with pytest.raises(CollectionNotFoundError):
            await collections.check()
        assert store.collections == {}  # read-only: not silently recreated empty
