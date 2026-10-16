from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_engine import __version__
from rag_engine.api.exception_handlers import register_exception_handlers
from rag_engine.api.middlewares import (
    PROCESS_TIME_HEADER,
    REQUEST_ID_HEADER,
    ProcessTimeMiddleware,
    RequestIdMiddleware,
)
from rag_engine.api.routers import api_router
from rag_engine.config import logger, settings, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Owns process-wide resources; infrastructure clients attach to `app.state` here."""
    logger.info("starting {} ({})", settings.PROJECT_NAME, settings.ENVIRONMENT)
    yield
    logger.info("shutting down {}", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    # uvicorn --reload serves from a child process: configure logging here too
    setup_logging()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=__version__,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
    )

    # added last = outermost: RequestId wraps ProcessTime so the access log carries the ID
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=settings.ALLOW_CREDENTIALS,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER, PROCESS_TIME_HEADER, "Retry-After"],
        )
    app.add_middleware(ProcessTimeMiddleware)
    app.add_middleware(RequestIdMiddleware)

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.API_V1_STR)
    return app


app = create_app()
