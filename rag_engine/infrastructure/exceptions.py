from rag_engine.exceptions import ServiceUnavailableError


class DatabaseUnavailableError(ServiceUnavailableError):
    """MongoDB could not be reached or rejected the operation at the connection level."""

    code = "database_unavailable"
    default_message = "The database is unavailable"


class RedisUnavailableError(ServiceUnavailableError):
    """Redis could not be reached or failed at the connection level."""

    code = "redis_unavailable"
    default_message = "Redis is unavailable"


class VectorStoreUnavailableError(ServiceUnavailableError):
    """The vector store (Qdrant) could not be reached or rejected our credentials."""

    code = "vector_store_unavailable"
    default_message = "The vector store is unavailable"
