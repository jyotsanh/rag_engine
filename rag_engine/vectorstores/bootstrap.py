"""Vector store bootstrap (Step 74): every collection in use exists before we serve.

`VectorCollections` is owned by the process's infrastructure:

- **`bootstrap(spaces)`** at startup ensures the collection of every embedding space in
  use (all KBs' spaces, plus the default model's native space so a first KB needs no
  extra round trip). A misconfigured existing collection (`CollectionMismatchError`) fails
  startup loudly; Qdrant being down fails it too (the connection's own ping).
- **`ensure(space)`** for a space first used after startup (a new KB with a new model or
  size): idempotent, cached per process, one creation even when called concurrently. The
  ingest pipeline calls it before upserting (Step 76).
- **`check()`** backs `GET /health/ready` (`vector_collections`): every collection this
  process relies on must still exist. It is read-only: a collection deleted from under
  the service makes it *not ready*; it is not silently recreated empty.
"""

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from rag_engine.embeddings import EmbeddingSpace
from rag_engine.vectorstores.base import BaseVectorStore
from rag_engine.vectorstores.models import CollectionSpec, VectorFilter

# a tenant id no real tenant has: counting it touches the collection without reading data
_PROBE_TENANT = "ten_" + "0" * 32


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    created: tuple[str, ...]
    existing: tuple[str, ...]


@dataclass
class VectorCollections:
    store: BaseVectorStore
    sparse: bool = True  # hybrid search everywhere
    _ready: dict[str, CollectionSpec] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def spec(self, space: EmbeddingSpace) -> CollectionSpec:
        return CollectionSpec.for_space(space, sparse=self.sparse)

    @property
    def collections(self) -> list[CollectionSpec]:
        return list(self._ready.values())

    async def ensure(self, space: EmbeddingSpace) -> CollectionSpec:
        """The space's collection, created if needed (once per process)."""
        spec = self.spec(space)
        if spec.name in self._ready:
            return spec
        lock = self._locks.setdefault(spec.name, asyncio.Lock())
        async with lock:
            if spec.name not in self._ready:
                created = await self.store.ensure_collection(spec)
                if created:
                    logger.info("created vector collection {}", spec.name)
                self._ready[spec.name] = spec
        return spec

    async def bootstrap(self, spaces: set[EmbeddingSpace]) -> BootstrapReport:
        created: list[str] = []
        existing: list[str] = []
        for space in sorted(spaces, key=lambda s: (s.model.value, s.dimensions)):
            spec = self.spec(space)
            if await self.store.ensure_collection(spec):
                created.append(spec.name)
            else:
                existing.append(spec.name)
            self._ready[spec.name] = spec
        logger.info(
            "vector collections ready: {} created, {} existing", len(created), len(existing)
        )
        return BootstrapReport(tuple(created), tuple(existing))

    async def check(self) -> None:
        """Readiness: raises (e.g. `CollectionNotFoundError`) if a collection is gone."""
        for spec in self.collections:
            await self.store.count(spec, VectorFilter(_PROBE_TENANT))
