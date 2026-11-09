"""The process runtime: infrastructure connections plus what is built on them (Step 74).

`rag_engine.infrastructure` only owns connections (MongoDB, Redis, Qdrant) and must not
depend on higher layers. `AppRuntime` composes on top of it, and is what the API lifespan
(and the worker, Step 75) starts:

1. connect everything (`AppInfrastructure.start`: Mongo + indexes, Redis, Qdrant);
2. **bootstrap the vector store**: build the active backend (`VectorStoreDispatcher`) and
   ensure the collection of every embedding space in use (all KBs' spaces plus the
   default model's native space), so searches and ingests never hit a missing
   collection. A misconfigured existing collection fails startup loudly.

Any failure closes everything and re-raises: the process never starts half-configured.
Readiness gains `vector_collections` (every relied-on collection still exists).
"""

from collections.abc import Mapping

from rag_engine.config import Settings, settings
from rag_engine.embeddings import resolve_space
from rag_engine.infrastructure import AppInfrastructure, HealthCheck
from rag_engine.repositories import KnowledgeBaseRepository
from rag_engine.vectorstores import BaseVectorStore, VectorCollections, VectorStoreDispatcher


class AppRuntime(AppInfrastructure):
    def __init__(self, *args: object, config: Settings = settings, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # ty: ignore[invalid-argument-type]
        self.config = config
        self.vector_store: BaseVectorStore | None = None
        self.collections: VectorCollections | None = None

    async def start(self) -> None:
        await super().start()  # closes its own connections if it fails
        try:
            self.vector_store = VectorStoreDispatcher.create(
                client=self.qdrant.client, config=self.config
            )
            self.collections = VectorCollections(self.vector_store)
            spaces = await KnowledgeBaseRepository(self.mongo.db).embedding_spaces_in_use()
            spaces.add(resolve_space(self.config.DEFAULT_EMBEDDING_MODEL))
            await self.collections.bootstrap(spaces)
        except BaseException:
            await self.stop()
            raise

    def health_checks(self) -> Mapping[str, HealthCheck]:
        checks = dict(super().health_checks())
        if self.collections is not None:
            checks["vector_collections"] = self.collections.check
        return checks
