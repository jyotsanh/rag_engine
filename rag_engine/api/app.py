from collections.abc import AsyncGenerator, Callable
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
from rag_engine.infrastructure import Infrastructure
from rag_engine.runtime import AppRuntime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Starts backing services before serving and stops them on shutdown."""
    logger.info("starting {} ({})", settings.PROJECT_NAME, settings.ENVIRONMENT)
    infrastructure: Infrastructure = app.state.infrastructure_factory()
    await infrastructure.start()
    app.state.infrastructure = infrastructure
    try:
        yield
    finally:
        logger.info("shutting down {}", settings.PROJECT_NAME)
        await infrastructure.stop()


def create_app(
    infrastructure_factory: Callable[[], Infrastructure] = AppRuntime,
) -> FastAPI:
    """Build the API. Tests pass a fake `infrastructure_factory` to run without services."""
    # uvicorn --reload serves from a child process: configure logging here too
    setup_logging()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=__version__,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
    )
    app.state.infrastructure_factory = infrastructure_factory

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
