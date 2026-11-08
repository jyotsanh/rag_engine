"""Vector store dispatcher: `VectorStoreType` (`ACTIVE_VECTOR_STORE`) → implementation (Step 67).

Implementations register with `@VectorStoreDispatcher.register(VectorStoreType.X)` (Qdrant:
Step 68). `VectorStoreDispatcher.create(**deps)` builds the active backend with the
dependencies it needs (the Qdrant store takes the process's `AsyncQdrantClient`).
"""

from collections.abc import Callable
from typing import Any, ClassVar

from rag_engine.config import Settings, VectorStoreType, settings
from rag_engine.vectorstores.base import BaseVectorStore
from rag_engine.vectorstores.exceptions import UnsupportedVectorStoreError


class VectorStoreDispatcher:
    _stores: ClassVar[dict[VectorStoreType, type[BaseVectorStore]]] = {}

    @classmethod
    def register[S: type[BaseVectorStore]](cls, kind: VectorStoreType) -> Callable[[S], S]:
        def register(store: S) -> S:
            if not (isinstance(store, type) and issubclass(store, BaseVectorStore)):
                raise TypeError(f"{store!r} is not a BaseVectorStore subclass")
            if getattr(store, "kind", None) is not kind:
                raise TypeError(f"{store.__name__}.kind must be {kind!r}")
            existing = cls._stores.get(kind)
            if existing is not None and existing is not store:
                raise TypeError(f"vector store {str(kind)!r} is already registered")
            cls._stores[kind] = store
            return store

        return register

    @classmethod
    def registered(cls) -> dict[VectorStoreType, type[BaseVectorStore]]:
        return dict(cls._stores)

    @classmethod
    def create(
        cls, kind: VectorStoreType | None = None, *, config: Settings = settings, **deps: Any
    ) -> BaseVectorStore:
        kind = kind or config.ACTIVE_VECTOR_STORE
        store = cls._stores.get(kind)
        if store is None:
            raise UnsupportedVectorStoreError(
                f"no vector store is registered for {str(kind)!r}",
                details={"kind": str(kind), "registered": sorted(cls._stores)},
            )
        return store(**deps)
