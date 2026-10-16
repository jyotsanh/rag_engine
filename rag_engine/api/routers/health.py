from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    message: str


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Process is up and serving requests. Dependency checks live at `/health/ready`."""
    return HealthResponse(message="app is healthy!!!")
