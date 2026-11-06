"""Readiness: concurrent dependency checks and the /health/ready endpoint (no services)."""

import asyncio
import time
from collections.abc import AsyncIterator, Mapping

import httpx
from asgi_lifespan import LifespanManager

from rag_engine.api import create_app
from rag_engine.config import settings
from rag_engine.infrastructure import (
    AppInfrastructure,
    DatabaseUnavailableError,
    DependencyState,
    HealthCheck,
    check_readiness,
)
from tests.conftest import OverrideSettings
from tests.support.fakes import NoopInfrastructure

API = settings.API_V1_STR


async def _ok() -> None:
    return None


async def _db_down() -> None:
    raise DatabaseUnavailableError()


async def _hangs() -> None:
    await asyncio.sleep(10)


async def _crashes() -> None:
    raise RuntimeError("password=hunter2 leaked in message")


async def _client_for(checks: Mapping[str, HealthCheck]) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(lambda: NoopInfrastructure(checks))
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
    ):
        yield client


# ---- #
# Tests
# ---- #


class TestReadiness_S12:
    """S12 — Readiness: per-dependency status, timeouts, 503, no leaked internals."""

    async def test_all_ok_is_ready(self) -> None:
        report = await check_readiness({"mongo": _ok, "redis": _ok}, budget_seconds=1)

        assert report.ready is True
        assert {n: d.status for n, d in report.dependencies.items()} == {
            "mongo": DependencyState.OK,
            "redis": DependencyState.OK,
        }
        assert all(d.latency_ms >= 0 and d.error is None for d in report.dependencies.values())

    async def test_failing_dependency_reports_its_error_code(self) -> None:
        report = await check_readiness({"mongo": _db_down, "redis": _ok}, budget_seconds=1)

        assert report.ready is False
        assert report.dependencies["mongo"].status is DependencyState.ERROR
        assert report.dependencies["mongo"].error == "database_unavailable"
        assert report.dependencies["redis"].status is DependencyState.OK

    async def test_hanging_checks_time_out_concurrently(self) -> None:
        start = time.perf_counter()

        report = await check_readiness({"a": _hangs, "b": _hangs, "c": _hangs}, budget_seconds=0.2)

        elapsed = time.perf_counter() - start
        assert elapsed < 0.6  # concurrent: one budget, not three
        assert {d.error for d in report.dependencies.values()} == {"timeout"}

    async def test_unexpected_exception_is_contained_without_details(self) -> None:
        report = await check_readiness({"x": _crashes}, budget_seconds=1)

        assert report.ready is False
        assert report.dependencies["x"].error == "unexpected_error"
        assert "hunter2" not in report.model_dump_json()

    async def test_no_checks_means_ready(self) -> None:
        assert (await check_readiness({}, budget_seconds=1)).ready is True

    async def test_endpoint_returns_200_when_ready(self) -> None:
        async for client in _client_for({"mongo": _ok, "redis": _ok, "qdrant": _ok}):
            response = await client.get(f"{API}/health/ready")

            assert response.status_code == 200
            body = response.json()
            assert body["ready"] is True
            assert set(body["dependencies"]) == {"mongo", "redis", "qdrant"}

    async def test_endpoint_returns_503_with_report_when_not_ready(self) -> None:
        async for client in _client_for({"mongo": _db_down, "qdrant": _crashes, "redis": _ok}):
            response = await client.get(f"{API}/health/ready")

            assert response.status_code == 503
            body = response.json()
            assert body["ready"] is False
            assert body["dependencies"]["mongo"]["error"] == "database_unavailable"
            assert body["dependencies"]["qdrant"]["error"] == "unexpected_error"
            assert body["dependencies"]["redis"]["status"] == "ok"
            assert "hunter2" not in response.text

    async def test_liveness_stays_up_when_dependencies_are_down(self) -> None:
        async for client in _client_for({"mongo": _db_down}):
            assert (await client.get(f"{API}/health")).status_code == 200
            assert (await client.get(f"{API}/health/ready")).status_code == 503

    async def test_endpoint_uses_configured_budget(
        self, override_settings: OverrideSettings
    ) -> None:
        override_settings(READINESS_CHECK_TIMEOUT_SECONDS=0.1)

        async for client in _client_for({"slow": _hangs}):
            start = time.perf_counter()
            response = await client.get(f"{API}/health/ready")

            assert time.perf_counter() - start < 1
            assert response.json()["dependencies"]["slow"]["error"] == "timeout"

    async def test_openapi_documents_503_readiness_body(self) -> None:
        async for client in _client_for({}):
            spec = (await client.get(f"{API}/openapi.json")).json()

            responses = spec["paths"][f"{API}/health/ready"]["get"]["responses"]
            ref = responses["503"]["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ReadinessReport")

    def test_app_infrastructure_exposes_three_checks(self) -> None:
        assert set(AppInfrastructure().health_checks()) == {"mongo", "redis", "qdrant"}
