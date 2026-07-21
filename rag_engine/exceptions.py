"""Application exception hierarchy.

Deep code raises typed ``AppBaseException`` subclasses carrying an HTTP
``status_code`` — never a bare ``HTTPException``. The handlers registered in
``rag_engine.api.exception_handlers`` convert these into structured JSON.
"""

from typing import Any


class AppBaseException(Exception):
    """Base for all application errors.

    Args:
        message: human-readable error message (returned to the client).
        status_code: HTTP status code the handler should respond with.
        details: optional structured payload with extra context.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class InfrastructureError(AppBaseException):
    """A backing service (Mongo, Redis, Qdrant, ...) is unavailable or errored."""

    def __init__(
        self,
        message: str = "An infrastructure dependency is unavailable",
        status_code: int = 503,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, details=details)


class NotFoundError(AppBaseException):
    """A requested resource does not exist."""

    def __init__(
        self,
        message: str = "Resource not found",
        status_code: int = 404,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, details=details)
