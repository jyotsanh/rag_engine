"""Render every error (app, validation, routing, unhandled) as `ErrorResponse` JSON."""

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_engine.api.middlewares import REQUEST_ID_HEADER
from rag_engine.config import logger
from rag_engine.exceptions import AppBaseException, ErrorBody, ErrorResponse

_HTTP_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _render(
    request: Request,
    status_code: int,
    body: ErrorResponse,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    headers = dict(headers or {})
    if (request_id := _request_id(request)) is not None:
        # the outermost ServerErrorMiddleware bypasses RequestIdMiddleware's header injection
        headers.setdefault(REQUEST_ID_HEADER, request_id)
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppBaseException)
    log = logger.bind(code=exc.code, status_code=exc.status_code, details=exc.details)
    if exc.status_code >= 500:
        log.error("app error on {}: {}", request.url.path, exc.message)
    else:
        log.warning("app error on {}: {}", request.url.path, exc.message)
    return _render(request, exc.status_code, exc.to_response(_request_id(request)), exc.headers)


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # drop `input`: it may echo secrets or large payloads back to the client and logs
    errors = [
        {"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")}
        for err in exc.errors()
    ]
    body = ErrorResponse(
        error=ErrorBody(
            code="validation_error",
            message="Request validation failed",
            details={"errors": jsonable_encoder(errors)},
            request_id=_request_id(request),
        )
    )
    return _render(request, 422, body)


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    body = ErrorResponse(
        error=ErrorBody(
            code=_HTTP_STATUS_CODES.get(exc.status_code, "http_error"),
            message=str(exc.detail),
            request_id=_request_id(request),
        )
    )
    return _render(request, exc.status_code, body, dict(exc.headers or {}))


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.opt(exception=exc).error("unhandled error on {}", request.url.path)
    body = ErrorResponse(
        error=ErrorBody(
            code="internal_error",
            message="Internal server error",
            request_id=_request_id(request),
        )
    )
    return _render(request, 500, body)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppBaseException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
