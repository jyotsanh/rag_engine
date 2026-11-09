"""RedisConnection against a real Redis container."""

from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager

from rag_engine.api import create_app
from rag_engine.config import Settings
from rag_engine.infrastructure import (
    AppInfrastructure,
    RedisConnection,
    RedisNamespace,
    RedisUnavailableError,
)
from tests.conftest import APP_STARTUP_TIMEOUT, OverrideSettings


@pytest.fixture
async def redis(integration_settings: Settings) -> AsyncIterator[RedisConnection]:
    connection = RedisConnection(integration_settings)
    await connection.connect()
    yield connection
    await connection.close()


# ---- #
# Tests
# ---- #


class TestRedisInfrastructure_S11:
    """S11 — Redis infrastructure: connection lifecycle, auth, namespace maintenance."""

    async def test_connect_ping_close(self, integration_settings: Settings) -> None:
        connection = RedisConnection(integration_settings)

        await connection.connect()
        await connection.ping()
        await connection.close()

        with pytest.raises(RedisUnavailableError):
            _ = connection.client

    async def test_unreachable_server_fails_fast(self, override_settings: OverrideSettings) -> None:
        config = override_settings(
            REDIS_HOST="127.0.0.1",
            REDIS_PORT=1,
            REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS=0.3,
            REDIS_SOCKET_TIMEOUT_SECONDS=0.3,
        )
        connection = RedisConnection(config)

        with pytest.raises(RedisUnavailableError):
            await connection.connect()
        with pytest.raises(RedisUnavailableError):
            _ = connection.client

    async def test_wrong_password_is_unavailable(
        self, integration_settings: Settings, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(REDIS_PASSWORD="definitely-wrong")

        with pytest.raises(RedisUnavailableError) as exc:
            await RedisConnection(config).connect()

        assert exc.value.status_code == 503

    async def test_responses_are_decoded_strings(self, redis: RedisConnection) -> None:
        key = redis.keys.key(RedisNamespace.LOCK, "probe")

        await redis.client.set(key, "value", ex=60)

        assert await redis.client.get(key) == "value"
        assert 0 < await redis.client.ttl(key) <= 60

    async def test_uses_the_configured_app_db(
        self, redis: RedisConnection, integration_settings: Settings
    ) -> None:
        info = await redis.client.client_info()

        assert int(info["db"]) == integration_settings.REDIS_DB

    async def test_delete_namespace_removes_only_that_namespace(
        self, redis: RedisConnection
    ) -> None:
        # Arrange: more keys than one UNLINK batch, plus keys that must survive
        robots = [redis.keys.key(RedisNamespace.ROBOTS, f"h{i}") for i in range(1_234)]
        async with redis.client.pipeline(transaction=False) as pipe:
            for key in robots:
                pipe.set(key, "1")
            pipe.set(redis.keys.key(RedisNamespace.LOCK, "keep"), "1")
            pipe.set("other_prefix:robots:h1", "1")
            await pipe.execute()

        # Act
        deleted = await redis.delete_namespace(RedisNamespace.ROBOTS)

        # Assert
        assert deleted == len(robots)
        assert await redis.client.exists(*robots) == 0
        assert await redis.client.exists(redis.keys.key(RedisNamespace.LOCK, "keep")) == 1
        assert await redis.client.exists("other_prefix:robots:h1") == 1

    async def test_app_lifespan_connects_redis(self, integration_settings: Settings) -> None:
        app = create_app(infrastructure_factory=AppInfrastructure)  # no vectors needed

        async with LifespanManager(app, startup_timeout=APP_STARTUP_TIMEOUT):
            infra = app.state.infrastructure
            assert isinstance(infra, AppInfrastructure)
            await infra.redis.ping()

        with pytest.raises(RedisUnavailableError):
            _ = infra.redis.client
