"""Application exception hierarchy and the error response contract.

Deep code raises typed `AppBaseException` subclasses carrying an HTTP
`status_code` and a stable machine-readable `code`, never a bare
`HTTPException`. The API's exception handlers (Step 6) render them as
`ErrorResponse` JSON.

Feature packages subclass these (e.g. `vectorstores/exceptions.py`) and
override `code` / `default_message`.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code, e.g. `not_found`.")
    message: str = Field(description="Human-readable error message.")
    details: dict[str, Any] = Field(default_factory=dict, description="Structured extra context.")
    request_id: str | None = Field(default=None, description="Correlates with server logs.")


class ErrorResponse(BaseModel):
    """Body of every non-2xx JSON response."""

    error: ErrorBody


class AppBaseException(Exception):
    """Base for all application errors.

    Args:
        message: human-readable error message (returned to the client).
        status_code: HTTP status code the handler should respond with.
        details: optional structured payload with extra context.
        headers: optional response headers (e.g. `Retry-After`).
    """

    status_code: int = 500
    code: ClassVar[str] = "internal_error"
    default_message: ClassVar[str] = "Internal server error"

    def __init__(
        self,
        message: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        if status_code is not None:
            self.status_code = status_code
        self.details: dict[str, Any] = dict(details or {})
        self.headers: dict[str, str] = dict(headers or {})
        super().__init__(self.message)

    def to_response(self, request_id: str | None = None) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorBody(
                code=self.code,
                message=self.message,
                details=self.details,
                request_id=request_id,
            )
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status_code={self.status_code}, code={self.code!r}, "
            f"message={self.message!r})"
        )


class BadRequestError(AppBaseException):
    """The request is syntactically valid but cannot be acted on."""

    status_code = 400
    code = "bad_request"
    default_message = "Bad request"


class UnauthorizedError(AppBaseException):
    """Missing or invalid credentials."""

    status_code = 401
    code = "unauthorized"
    default_message = "Missing or invalid API key"

    def __init__(
        self,
        message: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message, status_code, details, {"WWW-Authenticate": "Bearer", **(headers or {})}
        )


class ForbiddenError(AppBaseException):
    """Authenticated, but not allowed to perform this action."""

    status_code = 403
    code = "forbidden"
    default_message = "Not allowed to perform this action"


class NotFoundError(AppBaseException):
    """A requested resource does not exist (or is not visible to this tenant)."""

    status_code = 404
    code = "not_found"
    default_message = "Resource not found"


class ConflictError(AppBaseException):
    """The request conflicts with the current state of a resource."""

    status_code = 409
    code = "conflict"
    default_message = "Resource state conflict"


class AppValidationError(AppBaseException):
    """Input failed domain validation (beyond request schema validation)."""

    status_code = 422
    code = "validation_error"
    default_message = "Validation failed"


class QuotaExceededError(AppBaseException):
    """A tenant quota (KBs, concurrent jobs, pages per job) would be exceeded."""

    status_code = 403
    code = "quota_exceeded"
    default_message = "Tenant quota exceeded"


class TooManyRequestsError(AppBaseException):
    """Rate limit hit; `retry_after` seconds become the `Retry-After` header."""

    status_code = 429
    code = "rate_limited"
    default_message = "Too many requests"

    def __init__(
        self,
        message: str | None = None,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        headers = {"Retry-After": str(max(retry_after, 0))} if retry_after is not None else None
        merged = {
            **(details or {}),
            **({"retry_after": retry_after} if retry_after is not None else {}),
        }
        super().__init__(message, details=merged, headers=headers)
        self.retry_after = retry_after


class UpstreamError(AppBaseException):
    """An external dependency we call on the client's behalf failed (OpenAI, target site)."""

    status_code = 502
    code = "upstream_error"
    default_message = "Upstream service error"


class ServiceUnavailableError(AppBaseException):
    """Our own backing infrastructure (Mongo, Redis, Qdrant) is unavailable."""

    status_code = 503
    code = "service_unavailable"
    default_message = "A backing service is unavailable"


__all__ = [
    "AppBaseException",
    "AppValidationError",
    "BadRequestError",
    "ConflictError",
    "ErrorBody",
    "ErrorResponse",
    "ForbiddenError",
    "NotFoundError",
    "QuotaExceededError",
    "ServiceUnavailableError",
    "TooManyRequestsError",
    "UnauthorizedError",
    "UpstreamError",
]
