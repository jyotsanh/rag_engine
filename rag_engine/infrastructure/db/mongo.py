"""MongoDB backend — the default raw-document warehouse.

Phase 1 scope: connect + ping + close so the app has a real readiness signal.
The full ``NoSQLBaseDocument`` ODM and document models arrive in Phase 2 and
build on ``get_database()``.
"""

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from rag_engine.config import logger, settings

from .base import BaseDatabase
from .exceptions import DatabaseConnectionError


class MongoDatabase(BaseDatabase):
    def __init__(self) -> None:
        self._client: MongoClient | None = None

    def connect(self) -> None:
        logger.info("connecting to MongoDB at {}", settings.mongo_url)
        # pymongo is lazy — construction performs no I/O; ping() verifies reachability.
        self._client = MongoClient(settings.mongo_url, serverSelectionTimeoutMS=5000)

    def ping(self) -> bool:
        if self._client is None:
            raise DatabaseConnectionError(message="MongoDB client is not connected")
        try:
            self._client.admin.command("ping")
        except PyMongoError as exc:
            raise DatabaseConnectionError(
                message=f"MongoDB is unreachable: {settings.mongo_url}",
                details={"mongo_url": settings.mongo_url, "error": str(exc)},
            ) from exc
        return True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def get_database(self) -> Database:
        if self._client is None:
            raise DatabaseConnectionError(message="MongoDB client is not connected")
        return self._client[settings.MONGO_DB]
