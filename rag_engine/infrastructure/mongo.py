"""Async MongoDB client lifecycle (PyMongo's native asyncio API)."""

from bson.codec_options import CodecOptions
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from rag_engine.config import Settings, logger, settings
from rag_engine.domain.base import MongoDocument, registered_documents
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError

type MongoDatabase = AsyncDatabase[dict]


class MongoConnection:
    """Owns one `AsyncMongoClient` per process and the application database handle."""

    def __init__(self, config: Settings = settings) -> None:
        self._config = config
        self._client: AsyncMongoClient[dict] | None = None

    @property
    def client(self) -> AsyncMongoClient[dict]:
        if self._client is None:
            raise DatabaseUnavailableError("Mongo connection is not open")
        return self._client

    @property
    def db(self) -> MongoDatabase:
        return self.client.get_database(
            self._config.MONGO_DB,
            # tz-aware datetimes in and out; everything we store is UTC
            codec_options=CodecOptions(tz_aware=True, uuid_representation=4),
        )

    async def connect(self) -> None:
        """Create the client and verify the server answers within the selection timeout."""
        if self._client is not None:
            return
        self._client = AsyncMongoClient(
            self._config.mongo_url,
            appname=self._config.PROJECT_NAME,
            serverSelectionTimeoutMS=self._config.MONGO_SERVER_SELECTION_TIMEOUT_MS,
            connectTimeoutMS=self._config.MONGO_CONNECT_TIMEOUT_MS,
            maxPoolSize=self._config.MONGO_MAX_POOL_SIZE,
            tz_aware=True,
            uuidRepresentation="standard",
        )
        try:
            await self.ping()
        except DatabaseUnavailableError:
            await self.close()
            raise
        logger.info("connected to MongoDB database {}", self._config.MONGO_DB)

    async def ping(self) -> None:
        try:
            await self.client.admin.command("ping")
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def ensure_indexes(
        self, documents: list[type[MongoDocument]] | None = None
    ) -> dict[str, list[str]]:
        """Create every registered document's indexes (idempotent); returns names per collection."""
        created: dict[str, list[str]] = {}
        for model in documents if documents is not None else registered_documents():
            if not model.indexes:
                continue
            try:
                names = await self.db[model.collection_name].create_indexes(list(model.indexes))
            except PyMongoError as exc:
                raise DatabaseUnavailableError(
                    "Failed to create indexes",
                    details={"collection": model.collection_name, "reason": str(exc)},
                ) from exc
            created[model.collection_name] = names
            logger.debug("ensured indexes on {}: {}", model.collection_name, names)
        return created

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
