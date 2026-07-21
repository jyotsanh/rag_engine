from rag_engine.schemas import DatabaseType

from .base import BaseDatabase
from .exceptions import DatabaseNotImplementedError
from .mongo import MongoDatabase


class DatabaseFactory:
    @staticmethod
    def from_database(database: DatabaseType) -> BaseDatabase:
        if database == DatabaseType.MONGODB:
            return MongoDatabase()

        elif database == DatabaseType.POSTGRES:
            raise DatabaseNotImplementedError(
                status_code=501,
                details={},
                message="'Postgres' database backend is not implemented.",
            )

        else:
            raise DatabaseNotImplementedError(
                status_code=501,
                details={"requested": str(database)},
                message="Unsupported database backend",
            )


class DatabaseDispatcher:
    factory = DatabaseFactory()

    @classmethod
    def dispatch(cls, database: DatabaseType) -> BaseDatabase:
        """Resolve the concrete ``BaseDatabase`` for the given ``DatabaseType``.

        Args:
            database (DatabaseType): the backend selector (``settings.ACTIVE_DATABASE``).
        Returns:
            BaseDatabase: an un-connected instance — the caller runs ``connect()``.
        """
        return cls.factory.from_database(database)
