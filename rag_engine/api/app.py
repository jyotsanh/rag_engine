from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from rag_engine.api.exception_handlers import (
    api_exception_handler,
    validation_exception_handler,
)
from rag_engine.api.middlewares import ProcessTimeMiddleware
from rag_engine.api.routers import api_router
from rag_engine.config import logger, settings
from rag_engine.exceptions import AppBaseException
from rag_engine.infrastructure.lifecycle import StartupApplication


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: resolve + connect every backing resource onto app.state
    await StartupApplication.start(app)
    logger.info("startup complete")

    yield

    # shutdown: release them
    await StartupApplication.stop(app)
    logger.info("shutdown complete")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Middleware layers
## Layer 1: cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Layer 2: middleware for tracking response time
app.add_middleware(ProcessTimeMiddleware)

## Exception handlers
app.add_exception_handler(AppBaseException, api_exception_handler)  # type: ignore
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore

app.include_router(api_router, prefix=settings.API_V1_STR)
