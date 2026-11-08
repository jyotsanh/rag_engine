"""Vector store errors (Step 67). Outages reuse `VectorStoreUnavailableError` (503)."""

from rag_engine.exceptions import AppBaseException, AppValidationError
from rag_engine.infrastructure.exceptions import VectorStoreUnavailableError


class VectorStoreError(AppBaseException):
    """The vector store refused an operation (a bug or misconfiguration, not an outage)."""

    status_code = 500
    code = "vector_store_error"
    default_message = "The vector store refused the operation"


class UnsupportedVectorStoreError(VectorStoreError):
    code = "unsupported_vector_store"
    default_message = "Vector store backend is not available"


class CollectionNotFoundError(VectorStoreError):
    """A collection was used before `ensure_collection` created it."""

    code = "vector_collection_not_found"
    default_message = "Vector collection does not exist"


class CollectionMismatchError(VectorStoreError):
    """An existing collection's schema differs from the spec (e.g. other dimensions)."""

    code = "vector_collection_mismatch"
    default_message = "Vector collection does not match its specification"


class InvalidVectorDataError(VectorStoreError):
    """Points/queries that don't fit the collection: wrong dimensions, missing sparse part."""

    code = "invalid_vector_data"
    default_message = "Vector data does not fit the collection"


class InvalidVectorQueryError(AppValidationError):
    """A search request outside the allowed bounds (limit, threshold)."""

    code = "invalid_vector_query"
    default_message = "Invalid vector search parameters"


__all__ = [
    "CollectionMismatchError",
    "CollectionNotFoundError",
    "InvalidVectorDataError",
    "InvalidVectorQueryError",
    "UnsupportedVectorStoreError",
    "VectorStoreError",
    "VectorStoreUnavailableError",
]
