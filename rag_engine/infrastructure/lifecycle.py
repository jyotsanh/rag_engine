"""Process-wide infrastructure started in the API lifespan (and later the worker)."""

from typing import Protocol

from rag_engine.config import logger
from rag_engine.infrastructure.mongo import MongoConnection


class Infrastructure(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class AppInfrastructure:
    """Real backing services. Redis (Step 11) and Qdrant (Step 74) attach here too."""

    def __init__(self, mongo: MongoConnection | None = None) -> None:
        self.mongo = mongo or MongoConnection()

    async def start(self) -> None:
        await self.mongo.connect()
        await self.mongo.ensure_indexes()

    async def stop(self) -> None:
        await self.mongo.close()
        logger.info("infrastructure stopped")
