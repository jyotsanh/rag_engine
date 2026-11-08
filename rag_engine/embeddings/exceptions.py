"""Embedding errors (Step 61).

Configuration/validation errors are client-facing 422s. Provider failures are
`EmbeddingError`s carrying `retryable` (and `retry_after`), which the resilience layer
(Step 63) and the ingest pipeline (Step 76) act on; they surface to API callers as 502 (the
provider failed) or 503 (temporarily unavailable / rate limited).
"""

from typing import Any

from rag_engine.exceptions import AppBaseException, AppValidationError, UpstreamError


class UnsupportedEmbeddingModelError(AppValidationError):
    code = "unsupported_embedding_model"
    default_message = "Embedding model is not supported"


class UnsupportedEmbeddingDimensionsError(AppValidationError):
    code = "unsupported_embedding_dimensions"
    default_message = "Requested embedding dimensions are not supported by this model"


class UnsupportedEmbeddingProviderError(AppBaseException):
    """No embedder is registered for the provider, or the model belongs to another one."""

    status_code = 500
    code = "unsupported_embedding_provider"
    default_message = "Embedding provider is not available"


class EmbeddingInputError(AppValidationError):
    """An input can't be embedded as given (empty, over the model's input token limit)."""

    code = "invalid_embedding_input"
    default_message = "Text cannot be embedded"


class EmbeddingError(UpstreamError):
    """The embedding provider failed. `retryable` tells callers whether trying again helps."""

    code = "embedding_failed"
    default_message = "The embedding provider failed"
    retryable: bool = False

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        headers = {}
        if retry_after is not None:
            headers["Retry-After"] = str(max(int(retry_after + 0.999), 0))
        merged = {**(details or {}), **({"retry_after": retry_after} if retry_after else {})}
        super().__init__(message, details=merged, headers=headers)
        self.retry_after = retry_after


class EmbeddingAuthError(EmbeddingError):
    """The provider rejected our credentials (invalid/revoked key, no access to the model)."""

    code = "embedding_auth_failed"
    default_message = "The embedding provider rejected the credentials"


class EmbeddingRequestError(EmbeddingError):
    """The provider refused the request itself (4xx other than auth/rate limit)."""

    code = "embedding_request_rejected"
    default_message = "The embedding provider rejected the request"


class EmbeddingResponseError(EmbeddingError):
    """The provider answered, but not with one finite vector of the right size per input."""

    code = "embedding_response_invalid"
    default_message = "The embedding provider returned an invalid response"


class EmbeddingRateLimitedError(EmbeddingError):
    status_code = 503
    code = "embedding_rate_limited"
    default_message = "The embedding provider is rate limiting requests"
    retryable = True


class EmbeddingUnavailableError(EmbeddingError):
    """Timeouts, connection failures and 5xx: transient."""

    status_code = 503
    code = "embedding_unavailable"
    default_message = "The embedding provider is temporarily unavailable"
    retryable = True
