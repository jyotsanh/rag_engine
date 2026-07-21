from rag_engine.exceptions import AppBaseException


class DatabaseException(AppBaseException):
    """Base for every database-backend error."""


class DatabaseConnectionError(DatabaseException):
    """The database is unreachable / the connection failed."""


class DatabaseNotImplementedError(DatabaseException):
    """The requested ``DatabaseType`` has no concrete implementation yet."""
