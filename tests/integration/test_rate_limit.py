"""Sliding-window rate limiter (real Redis) and per-tenant API limiting (real app)."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from rag_engine.api import create_app
from rag_engine.config import Settings
from rag_engine.infrastructure import RedisConnection, RedisNamespace
from rag_engine.security import SlidingWindowRateLimiter

API = "/api/v1"


@pytest.fixture
async def redis(integration_settings: Settings) -> AsyncIterator[RedisConnection]:
    connection = RedisConnection(integration_settings)
    await connection.connect()
    yield connection
    await connection.close()


@pytest.fixture
def limiter(redis: RedisConnection) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(redis.client)


@dataclass
class _World:
    app: FastAPI
    client: httpx.AsyncClient
    admin: dict[str, str]

    async def tenant(self, name: str, per_minute: int) -> dict[str, str]:
        created = await self.client.post(
            f"{API}/admin/tenants",
            json={"name": name, "quotas": {"api_rate_limit_per_minute": per_minute}},
            headers=self.admin,
        )
        issued = await self.client.post(
            f"{API}/admin/tenants/{created.json()['id']}/keys", json={}, headers=self.admin
        )
        return {"Authorization": f"Bearer {issued.json()['api_key']}"}

    async def call(self, headers: dict[str, str]) -> httpx.Response:
        return await self.client.get(f"{API}/knowledge-bases", headers=headers)


@pytest.fixture
async def world(integration_settings: Settings) -> AsyncIterator[_World]:
    app = create_app()
    admin = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
    ):
        yield _World(app, client, admin)


# ---- #
# Tests
# ---- #


class TestSlidingWindowLimiter_S22:
    """S22 — Per-tenant API rate limiting: atomic sliding window in Redis."""

    async def test_allows_up_to_limit_then_denies(
        self, limiter: SlidingWindowRateLimiter, redis: RedisConnection
    ) -> None:
        key = redis.keys.key(RedisNamespace.API_RATE_LIMIT, "ten_a")

        decisions = [await limiter.hit(key, limit=3, window_seconds=60) for _ in range(4)]

        assert [d.allowed for d in decisions] == [True, True, True, False]
        assert [d.remaining for d in decisions] == [2, 1, 0, 0]
        assert 55 <= decisions[-1].retry_after <= 60
        assert decisions[-1].headers()["X-RateLimit-Reset"] == str(decisions[-1].retry_after)

    async def test_window_slides(
        self, limiter: SlidingWindowRateLimiter, redis: RedisConnection
    ) -> None:
        key = redis.keys.key(RedisNamespace.API_RATE_LIMIT, "ten_b")
        await limiter.hit(key, limit=2, window_seconds=1)
        await limiter.hit(key, limit=2, window_seconds=1)
        denied = await limiter.hit(key, limit=2, window_seconds=1)

        await asyncio.sleep(1.1)
        after = await limiter.hit(key, limit=2, window_seconds=1)

        assert not denied.allowed and denied.retry_after == 1
        assert after.allowed

    async def test_denied_requests_do_not_consume_capacity(
        self, limiter: SlidingWindowRateLimiter, redis: RedisConnection
    ) -> None:
        key = redis.keys.key(RedisNamespace.API_RATE_LIMIT, "ten_c")
        for _ in range(10):
            await limiter.hit(key, limit=2, window_seconds=60)

        assert await redis.client.zcard(key) == 2

    async def test_concurrent_hits_never_exceed_limit(
        self, limiter: SlidingWindowRateLimiter, redis: RedisConnection
    ) -> None:
        key = redis.keys.key(RedisNamespace.API_RATE_LIMIT, "ten_d")

        decisions = await asyncio.gather(
            *(limiter.hit(key, limit=10, window_seconds=60) for _ in range(60))
        )

        assert sum(d.allowed for d in decisions) == 10

    async def test_subjects_are_independent_and_key_expires(
        self, limiter: SlidingWindowRateLimiter, redis: RedisConnection
    ) -> None:
        a = redis.keys.key(RedisNamespace.API_RATE_LIMIT, "ten_e")
        b = redis.keys.key(RedisNamespace.API_RATE_LIMIT, "ten_f")
        await limiter.hit(a, limit=1, window_seconds=60)

        assert not (await limiter.hit(a, limit=1, window_seconds=60)).allowed
        assert (await limiter.hit(b, limit=1, window_seconds=60)).allowed
        assert 0 < await redis.client.pttl(a) <= 60_000

    @pytest.mark.parametrize(("limit", "window"), [(0, 60), (5, 0)])
    async def test_rejects_nonsense_parameters(
        self, limiter: SlidingWindowRateLimiter, limit: int, window: int
    ) -> None:
        with pytest.raises(ValueError):
            await limiter.hit("rag:api_rl:x", limit=limit, window_seconds=window)


class TestTenantApiRateLimit_S22:
    """S22 — Per-tenant API rate limiting: 429 + Retry-After on tenant routes."""

    async def test_success_carries_rate_limit_headers(self, world: _World) -> None:
        headers = await world.tenant("Acme", per_minute=5)

        response = await world.call(headers)

        assert response.status_code == 200
        assert response.headers["x-ratelimit-limit"] == "5"
        assert response.headers["x-ratelimit-remaining"] == "4"

    async def test_over_quota_gets_429_with_retry_after(self, world: _World) -> None:
        headers = await world.tenant("Acme", per_minute=3)

        statuses = [(await world.call(headers)).status_code for _ in range(3)]
        blocked = await world.call(headers)

        assert statuses == [200, 200, 200]
        assert blocked.status_code == 429
        assert 1 <= int(blocked.headers["retry-after"]) <= 60
        assert blocked.headers["x-ratelimit-remaining"] == "0"
        error = blocked.json()["error"]
        assert error["code"] == "rate_limited"
        assert error["details"]["limit"] == 3 and error["details"]["window_seconds"] == 60

    async def test_tenants_are_limited_independently(self, world: _World) -> None:
        noisy = await world.tenant("Noisy", per_minute=1)
        quiet = await world.tenant("Quiet", per_minute=1)

        await world.call(noisy)

        assert (await world.call(noisy)).status_code == 429
        assert (await world.call(quiet)).status_code == 200

    async def test_admin_routes_are_not_limited(self, world: _World) -> None:
        await world.tenant("Acme", per_minute=1)

        statuses = {
            (await world.client.get(f"{API}/admin/tenants", headers=world.admin)).status_code
            for _ in range(5)
        }

        assert statuses == {200}

    async def test_unauthenticated_requests_do_not_count(self, world: _World) -> None:
        headers = await world.tenant("Acme", per_minute=1)
        for _ in range(5):
            await world.client.get(f"{API}/knowledge-bases")

        assert (await world.call(headers)).status_code == 200

    async def test_fails_open_when_redis_is_down(self, world: _World) -> None:
        headers = await world.tenant("Acme", per_minute=1)
        await world.call(headers)  # quota now used up
        infra = world.app.state.infrastructure
        await infra.redis.close()
        infra.redis._client = Redis(
            host="127.0.0.1", port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0)
        )
        world.app.state.rate_limiter = None

        response = await world.call(headers)

        assert response.status_code == 200
        assert "x-ratelimit-limit" not in response.headers
