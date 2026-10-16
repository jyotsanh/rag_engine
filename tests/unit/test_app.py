"""FastAPI app: health route, middlewares, error rendering, CORS, lifespan."""

import io
import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from pydantic import BaseModel

from rag_engine.api import create_app
from rag_engine.api.middlewares import PROCESS_TIME_HEADER, REQUEST_ID_HEADER
from rag_engine.config import logger, settings, setup_logging
from rag_engine.exceptions import NotFoundError, TooManyRequestsError

API = settings.API_V1_STR


class _Payload(BaseModel):
    count: int
    api_key: str


def _build_app() -> FastAPI:
    """The real app plus throwaway routes that exercise each error path."""
    app = create_app()

    @app.get("/_test/not-found")
    async def not_found() -> None:
        raise NotFoundError("job not found", details={"job_id": "j1"})

    @app.get("/_test/rate-limited")
    async def rate_limited() -> None:
        raise TooManyRequestsError(retry_after=7)

    @app.get("/_test/boom")
    async def boom() -> None:
        raise RuntimeError("secret internals")

    @app.post("/_test/validate")
    async def validate(payload: _Payload) -> _Payload:
        return payload

    return app


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    setup_logging(level="DEBUG", json_logs=True, stream=stream)
    yield stream
    logger.remove()
    logging.root.handlers = []


@pytest.fixture
async def client(log_stream: io.StringIO) -> AsyncIterator[httpx.AsyncClient]:
    app = _build_app()
    # create_app() reconfigures logging; point it back at the captured stream
    setup_logging(level="DEBUG", json_logs=True, stream=log_stream)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http


def _records(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


# ---- #
# Tests
# ---- #


class TestFastApiApp_S6:
    """S6 — FastAPI app: /health, request-id + process-time middleware, error handlers."""

    async def test_health_returns_ok(self, client: httpx.AsyncClient) -> None:
        response = await client.get(f"{API}/health")

        assert response.status_code == 200
        assert response.json() == {"message": "app is healthy!!!"}

    async def test_request_id_generated_and_timing_header_set(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(f"{API}/health")

        assert len(response.headers[REQUEST_ID_HEADER]) == 32
        assert float(response.headers[PROCESS_TIME_HEADER]) >= 0

    async def test_valid_incoming_request_id_is_echoed(self, client: httpx.AsyncClient) -> None:
        response = await client.get(f"{API}/health", headers={REQUEST_ID_HEADER: "abc-123.x_y"})

        assert response.headers[REQUEST_ID_HEADER] == "abc-123.x_y"

    @pytest.mark.parametrize("bad", ["has space", "x" * 129, "evil\nheader"])
    async def test_unsafe_incoming_request_id_is_replaced(
        self, client: httpx.AsyncClient, bad: str
    ) -> None:
        response = await client.get(f"{API}/health", headers={REQUEST_ID_HEADER: bad})

        assert response.headers[REQUEST_ID_HEADER] != bad
        assert len(response.headers[REQUEST_ID_HEADER]) == 32

    async def test_access_log_line_carries_request_id(
        self, client: httpx.AsyncClient, log_stream: io.StringIO
    ) -> None:
        await client.get(f"{API}/health", headers={REQUEST_ID_HEADER: "req-log"})

        access = [r for r in _records(log_stream) if r["extra"].get("path") == f"{API}/health"]
        assert len(access) == 1
        assert access[0]["request_id"] == "req-log"
        assert access[0]["extra"]["status_code"] == 200
        assert access[0]["extra"]["method"] == "GET"

    async def test_app_exception_rendered_as_error_response(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/_test/not-found", headers={REQUEST_ID_HEADER: "req-404"})

        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "code": "not_found",
                "message": "job not found",
                "details": {"job_id": "j1"},
                "request_id": "req-404",
            }
        }

    async def test_app_exception_headers_are_forwarded(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/_test/rate-limited")

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "7"
        assert response.json()["error"]["code"] == "rate_limited"

    async def test_validation_error_hides_input(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/_test/validate", json={"count": "not-a-number", "api_key": "sk-super-secret"}
        )

        body = response.json()
        assert response.status_code == 422
        assert body["error"]["code"] == "validation_error"
        assert body["error"]["details"]["errors"][0]["loc"] == ["body", "count"]
        assert "sk-super-secret" not in response.text

    async def test_unknown_route_uses_error_shape(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/does-not-exist")

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
        assert REQUEST_ID_HEADER in response.headers

    async def test_wrong_method_uses_error_shape(self, client: httpx.AsyncClient) -> None:
        response = await client.post(f"{API}/health")

        assert response.status_code == 405
        assert response.json()["error"]["code"] == "method_not_allowed"

    async def test_unhandled_exception_is_500_without_leaking(
        self, client: httpx.AsyncClient, log_stream: io.StringIO
    ) -> None:
        response = await client.get("/_test/boom", headers={REQUEST_ID_HEADER: "req-500"})

        assert response.status_code == 500
        assert response.json() == {
            "error": {
                "code": "internal_error",
                "message": "Internal server error",
                "details": {},
                "request_id": "req-500",
            }
        }
        assert response.headers[REQUEST_ID_HEADER] == "req-500"
        assert "secret internals" not in response.text
        errors = [r for r in _records(log_stream) if r["level"] == "ERROR"]
        assert errors[0]["exception"]["value"] == "secret internals"

    async def test_openapi_served_under_api_prefix_with_error_schema(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get(f"{API}/openapi.json")

        spec = response.json()
        assert response.status_code == 200
        assert f"{API}/health" in spec["paths"]
        assert "ErrorResponse" in spec["components"]["schemas"]

    async def test_lifespan_logs_startup_and_shutdown(self, log_stream: io.StringIO) -> None:
        app = create_app()
        setup_logging(level="DEBUG", json_logs=True, stream=log_stream)

        async with LifespanManager(app):
            pass

        messages = [r["message"] for r in _records(log_stream)]
        assert any(m.startswith("starting") for m in messages)
        assert any(m.startswith("shutting down") for m in messages)

    async def test_cors_enabled_only_when_origins_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(settings, "BACKEND_CORS_ORIGINS", ["https://dash.example.com"])
        app = create_app()
        transport = httpx.ASGITransport(app=app)

        # Act
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            allowed = await http.options(
                f"{API}/health",
                headers={
                    "Origin": "https://dash.example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            denied = await http.get(f"{API}/health", headers={"Origin": "https://evil.test"})

        # Assert
        assert allowed.headers["access-control-allow-origin"] == "https://dash.example.com"
        assert "access-control-allow-origin" not in denied.headers
