from fastapi import APIRouter, Request

from rag_engine.exceptions import InfrastructureError
from rag_engine.infrastructure.db import BaseDatabase

# Aggregates every feature router. Phase 8/9 will `include_router` the
# crawl / jobs / query routers here, each with its own prefix + tags.
api_router = APIRouter()


@api_router.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    """Liveness — the process is up and serving."""
    return {"message": "app is healthy!!!"}


@api_router.get("/ready", tags=["Health"])
def ready(request: Request) -> dict[str, str]:
    """Readiness — the active database backend is reachable."""
    database: BaseDatabase | None = getattr(request.app.state, "database", None)
    if database is None:
        raise InfrastructureError(message="database is not initialised")
    database.ping()
    return {"status": "ready"}
