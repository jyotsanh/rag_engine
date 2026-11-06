"""Exception hierarchy: defaults, overrides, headers, and the error response contract."""

import pytest

from rag_engine.exceptions import (
    AppBaseException,
    AppValidationError,
    BadRequestError,
    ConflictError,
    ErrorResponse,
    ForbiddenError,
    NotFoundError,
    QuotaExceededError,
    ServiceUnavailableError,
    TooManyRequestsError,
    UnauthorizedError,
    UpstreamError,
)


class _KnowledgeBaseNotFound(NotFoundError):
    code = "knowledge_base_not_found"
    default_message = "Knowledge base not found"


# ---- #
# Tests
# ---- #


class TestExceptions_S5:
    """S5 — Exceptions: AppBaseException subclasses and the ErrorResponse schema."""

    @pytest.mark.parametrize(
        ("exc_type", "status_code", "code"),
        [
            (AppBaseException, 500, "internal_error"),
            (BadRequestError, 400, "bad_request"),
            (UnauthorizedError, 401, "unauthorized"),
            (ForbiddenError, 403, "forbidden"),
            (NotFoundError, 404, "not_found"),
            (ConflictError, 409, "conflict"),
            (AppValidationError, 422, "validation_error"),
            (QuotaExceededError, 403, "quota_exceeded"),
            (TooManyRequestsError, 429, "rate_limited"),
            (UpstreamError, 502, "upstream_error"),
            (ServiceUnavailableError, 503, "service_unavailable"),
        ],
    )
    def test_subclass_defaults(
        self, exc_type: type[AppBaseException], status_code: int, code: str
    ) -> None:
        exc = exc_type()

        assert isinstance(exc, AppBaseException)
        assert exc.status_code == status_code
        assert exc.code == code
        assert exc.message == exc_type.default_message
        assert str(exc) == exc.message
        assert exc.details == {}

    def test_message_status_and_details_override(self) -> None:
        exc = ConflictError("job already running", status_code=423, details={"job_id": "j1"})

        assert exc.message == "job already running"
        assert exc.status_code == 423
        assert exc.details == {"job_id": "j1"}
        assert ConflictError.status_code == 409  # class default untouched

    def test_details_are_copied_not_shared(self) -> None:
        details = {"a": 1}

        exc = BadRequestError(details=details)
        details["a"] = 2

        assert exc.details == {"a": 1}
        assert BadRequestError().details is not BadRequestError().details

    def test_feature_subclass_overrides_code_and_message(self) -> None:
        exc = _KnowledgeBaseNotFound(details={"kb_id": "kb1"})

        assert isinstance(exc, NotFoundError)
        assert exc.status_code == 404
        assert exc.code == "knowledge_base_not_found"
        assert exc.message == "Knowledge base not found"

    def test_unauthorized_sets_www_authenticate(self) -> None:
        exc = UnauthorizedError(headers={"X-Extra": "1"})

        assert exc.headers == {"WWW-Authenticate": "Bearer", "X-Extra": "1"}

    def test_too_many_requests_carries_retry_after(self) -> None:
        exc = TooManyRequestsError(retry_after=12, details={"limit": 60})

        assert exc.retry_after == 12
        assert exc.headers == {"Retry-After": "12"}
        assert exc.details == {"limit": 60, "retry_after": 12}

    def test_too_many_requests_without_retry_after_has_no_header(self) -> None:
        exc = TooManyRequestsError()

        assert exc.headers == {}
        assert exc.retry_after is None

    def test_to_response_matches_contract(self) -> None:
        exc = NotFoundError("job not found", details={"job_id": "j9"})

        body = exc.to_response(request_id="req-1").model_dump()

        assert body == {
            "error": {
                "code": "not_found",
                "message": "job not found",
                "details": {"job_id": "j9"},
                "request_id": "req-1",
            }
        }
        assert ErrorResponse.model_validate(body) == exc.to_response(request_id="req-1")

    def test_repr_is_informative(self) -> None:
        assert repr(UpstreamError("openai down")) == (
            "UpstreamError(status_code=502, code='upstream_error', message='openai down')"
        )

    def test_can_be_raised_and_caught_as_base(self) -> None:
        with pytest.raises(AppBaseException) as caught:
            raise QuotaExceededError(details={"quota": "max_kbs"})

        assert caught.value.code == "quota_exceeded"
