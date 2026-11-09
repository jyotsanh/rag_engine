"""Aggregates every feature router under `settings.API_V1_STR`."""

from typing import Any

from fastapi import APIRouter

from rag_engine.api.routers import admin, crawls, health, knowledge_bases, sources
from rag_engine.exceptions import ErrorResponse

# every endpoint may fail validation or crash; document the shared error body once
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}

api_router = APIRouter(responses=ERROR_RESPONSES)
api_router.include_router(health.router)
api_router.include_router(admin.router)
api_router.include_router(knowledge_bases.router)
api_router.include_router(sources.router)
api_router.include_router(crawls.router)
