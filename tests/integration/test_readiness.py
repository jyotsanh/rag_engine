"""QdrantConnection and /health/ready against real Mongo/Redis/Qdrant containers."""

import httpx
import pytest
from asgi_lifespan import LifespanManager
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from rag_engine.api import create_app
from rag_engine.config import Settings
from rag_engine.infrastructure import (
    AppInfrastructure,
    QdrantConnection,
    VectorStoreUnavailableError,
)
from tests.conftest import OverrideSettings

# ---- #
# Tests
# ---- #


class TestReadinessIntegration_S12:
    """S12 — Readiness: Qdrant connection lifecycle and real dependency pings."""

    async def test_qdrant_connect_ping_close(self, integration_settings: Settings) -> None:
        connection = QdrantConnection(integration_settings)

        await connection.connect()
        await connection.ping()
        await connection.close()

        with pytest.raises(VectorStoreUnavailableError):
            _ = connection.client

    @pytest.mark.parametrize("prefer_grpc", [True, False])
    async def test_qdrant_wrong_api_key_fails_at_connect(
        self,
        integration_settings: Settings,
        override_settings: OverrideSettings,
        prefer_grpc: bool,
    ) -> None:
        config = override_settings(QDRANT_API_KEY="wrong-key", QDRANT_PREFER_GRPC=prefer_grpc)
        connection = QdrantConnection(config)

        with pytest.raises(VectorStoreUnavailableError):
            await connection.connect()
        with pytest.raises(VectorStoreUnavailableError):
            _ = connection.client

    async def test_qdrant_unreachable_fails(self, override_settings: OverrideSettings) -> None:
        config = override_settings(
            QDRANT_HOST="127.0.0.1",
            QDRANT_HTTP_PORT=1,
            QDRANT_GRPC_PORT=2,
            QDRANT_PREFER_GRPC=False,
            QDRANT_TIMEOUT_SECONDS=1,
        )

        with pytest.raises(VectorStoreUnavailableError):
            await QdrantConnection(config).connect()

    async def test_ready_with_real_services(self, integration_settings: Settings) -> None:
        app = create_app()
        transport = httpx.ASGITransport(app=app)

        async with (
            LifespanManager(app),
            httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ):
            response = await client.get(f"{integration_settings.API_V1_STR}/health/ready")

        body = response.json()
        assert response.status_code == 200
        assert body["ready"] is True
        assert {n: d["status"] for n, d in body["dependencies"].items()} == {
            "mongo": "ok",
            "redis": "ok",
            "qdrant": "ok",
            "vector_collections": "ok",
        }

    async def test_not_ready_when_dependencies_fail_after_startup(
        self, integration_settings: Settings, override_settings: OverrideSettings
    ) -> None:
        app = create_app()
        transport = httpx.ASGITransport(app=app)

        async with (
            LifespanManager(app),
            httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ):
            infra = app.state.infrastructure
            assert isinstance(infra, AppInfrastructure)
            # Arrange: Qdrant starts rejecting our key; Redis becomes unreachable
            bad_key = override_settings(QDRANT_API_KEY="rotated-elsewhere")
            await infra.qdrant.close()
            infra.qdrant = QdrantConnection(bad_key)
            infra.qdrant._client = _build_qdrant_client(bad_key)
            await infra.redis.close()
            # no retries: the real client's backoff would turn this into a `timeout` instead
            infra.redis._client = Redis(
                host="127.0.0.1", port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0)
            )

            # Act
            response = await client.get(f"{integration_settings.API_V1_STR}/health/ready")

            # Assert
            body = response.json()
            assert response.status_code == 503
            assert body["dependencies"]["mongo"]["status"] == "ok"
            assert body["dependencies"]["redis"]["error"] == "redis_unavailable"
            assert body["dependencies"]["qdrant"]["error"] == "vector_store_unavailable"


def _build_qdrant_client(config: Settings):
    from qdrant_client import AsyncQdrantClient

    return AsyncQdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_HTTP_PORT,
        grpc_port=config.QDRANT_GRPC_PORT,
        prefer_grpc=config.QDRANT_PREFER_GRPC,
        https=False,
        api_key=config.QDRANT_API_KEY.get_secret_value() if config.QDRANT_API_KEY else None,
        check_compatibility=False,
    )
