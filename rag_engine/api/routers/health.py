from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from rag_engine.config import settings
from rag_engine.infrastructure import Infrastructure, ReadinessReport, check_readiness

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    message: str


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Process is up and serving requests. Dependency checks live at `/health/ready`."""
    return HealthResponse(message="app is healthy!!!")


@router.get(
    "/ready",
    response_model=ReadinessReport,
    summary="Readiness probe",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ReadinessReport,
            "description": "At least one dependency is unreachable",
        }
    },
)
async def ready(request: Request, response: Response) -> ReadinessReport:
    """Ping MongoDB, Redis and Qdrant concurrently, each within
    `READINESS_CHECK_TIMEOUT_SECONDS`. Returns 503 (same body) if any check fails,
    so load balancers stop routing traffic while a dependency is down.
    """
    infrastructure: Infrastructure = request.app.state.infrastructure
    report = await check_readiness(
        infrastructure.health_checks(), settings.READINESS_CHECK_TIMEOUT_SECONDS
    )
    if not report.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
