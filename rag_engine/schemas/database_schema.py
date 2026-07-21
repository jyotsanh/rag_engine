from enum import StrEnum


class DatabaseType(StrEnum):
    """The raw-document warehouse backends the app can be pointed at.

    The active one is selected via ``settings.ACTIVE_DATABASE`` and resolved by
    :class:`rag_engine.infrastructure.db.DatabaseDispatcher`.
    """

    MONGODB = "mongodb"
    POSTGRES = "postgres"  # not implemented yet — see DatabaseFactory
