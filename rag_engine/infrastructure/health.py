"""Concurrent dependency checks behind `GET /health/ready`."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from enum import StrEnum

from pydantic import BaseModel, Field

from rag_engine.config import logger
from rag_engine.exceptions import AppBaseException

type HealthCheck = Callable[[], Awaitable[None]]


class DependencyState(StrEnum):
    OK = "ok"
    ERROR = "error"


class DependencyStatus(BaseModel):
    status: DependencyState
    latency_ms: float = Field(description="Time the check took, including a timeout.")
    error: str | None = Field(
        default=None,
        description="Stable error code (e.g. `timeout`, `database_unavailable`); never internals.",
    )


class ReadinessReport(BaseModel):
    ready: bool
    dependencies: dict[str, DependencyStatus]


async def _run_one(name: str, check: HealthCheck, budget_seconds: float) -> DependencyStatus:
    start = time.perf_counter()
    error: str | None = None
    try:
        async with asyncio.timeout(budget_seconds):
            await check()
    except TimeoutError:
        error = "timeout"
    except AppBaseException as exc:
        error = exc.code
    except Exception:  # an unexpected failure still means "not ready", not a 500
        error = "unexpected_error"
        logger.opt(exception=True).error("readiness check {} crashed", name)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    if error is not None:
        logger.bind(dependency=name, error=error, latency_ms=latency_ms).warning(
            "dependency {} not ready: {}", name, error
        )
        return DependencyStatus(status=DependencyState.ERROR, latency_ms=latency_ms, error=error)
    return DependencyStatus(status=DependencyState.OK, latency_ms=latency_ms)


async def check_readiness(
    checks: Mapping[str, HealthCheck], budget_seconds: float
) -> ReadinessReport:
    """Run every check concurrently; total time is bounded by roughly one `budget_seconds`."""
    names = list(checks)
    results = await asyncio.gather(*(_run_one(n, checks[n], budget_seconds) for n in names))
    dependencies = dict(zip(names, results, strict=True))
    ready = all(d.status is DependencyState.OK for d in results)
    return ReadinessReport(ready=ready, dependencies=dependencies)
