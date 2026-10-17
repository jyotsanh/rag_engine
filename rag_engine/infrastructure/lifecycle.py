"""Process-wide infrastructure started in the API lifespan (and later the worker)."""

from typing import Protocol

from rag_engine.config import logger
from rag_engine.infrastructure.mongo import MongoConnection
from rag_engine.infrastructure.redis import RedisConnection


class Infrastructure(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class AppInfrastructure:
    """Real backing services. Qdrant attaches here in Step 74."""

    def __init__(
        self, mongo: MongoConnection | None = None, redis: RedisConnection | None = None
    ) -> None:
        self.mongo = mongo or MongoConnection()
        self.redis = redis or RedisConnection()

    async def start(self) -> None:
        """Connect everything; if any service fails, close what was opened and re-raise."""
        try:
            await self.mongo.connect()
            await self.mongo.ensure_indexes()
            await self.redis.connect()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        # close in reverse order of start; each close is a no-op if never opened
        await self.redis.close()
        await self.mongo.close()
        logger.info("infrastructure stopped")
