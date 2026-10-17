"""Process-wide infrastructure started in the API lifespan (and later the worker)."""

from collections.abc import Mapping
from typing import Protocol

from rag_engine.config import logger
from rag_engine.infrastructure.health import HealthCheck
from rag_engine.infrastructure.mongo import MongoConnection
from rag_engine.infrastructure.qdrant import QdrantConnection
from rag_engine.infrastructure.redis import RedisConnection


class Infrastructure(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health_checks(self) -> Mapping[str, HealthCheck]: ...


class AppInfrastructure:
    """Real backing services: MongoDB, Redis, Qdrant."""

    def __init__(
        self,
        mongo: MongoConnection | None = None,
        redis: RedisConnection | None = None,
        qdrant: QdrantConnection | None = None,
    ) -> None:
        self.mongo = mongo or MongoConnection()
        self.redis = redis or RedisConnection()
        self.qdrant = qdrant or QdrantConnection()

    async def start(self) -> None:
        """Connect everything; if any service fails, close what was opened and re-raise."""
        try:
            await self.mongo.connect()
            await self.mongo.ensure_indexes()
            await self.redis.connect()
            await self.qdrant.connect()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        # close in reverse order of start; each close is a no-op if never opened
        await self.qdrant.close()
        await self.redis.close()
        await self.mongo.close()
        logger.info("infrastructure stopped")

    def health_checks(self) -> Mapping[str, HealthCheck]:
        return {"mongo": self.mongo.ping, "redis": self.redis.ping, "qdrant": self.qdrant.ping}
