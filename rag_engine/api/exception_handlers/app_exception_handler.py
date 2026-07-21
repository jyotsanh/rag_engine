from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_engine.config import logger
from rag_engine.exceptions import AppBaseException


async def api_exception_handler(request: Request, exc: AppBaseException) -> JSONResponse:
    """Render any :class:`AppBaseException` as a structured JSON error."""
    logger.warning("app exception on {}: {}", request.url.path, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render request-validation failures with a stable 422 shape."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            }
        },
    )
