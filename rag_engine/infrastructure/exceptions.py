from rag_engine.exceptions import ServiceUnavailableError


class DatabaseUnavailableError(ServiceUnavailableError):
    """MongoDB could not be reached or rejected the operation at the connection level."""

    code = "database_unavailable"
    default_message = "The database is unavailable"
