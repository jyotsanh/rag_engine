"""robots.txt verdicts shared through real Redis."""

import json
from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import (
    CachedRobots,
    RedisRobotsStore,
    RetryPolicy,
    RobotsCache,
    RobotsStatus,
)
from rag_engine.crawling.fetchers import HttpFetcher
from rag_engine.infrastructure import RedisConnection, RedisKeyspace, RedisNamespace
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import FixtureSite

_ORIGIN = "https://acme.example"


@pytest.fixture
async def redis(integration_settings: Settings) -> AsyncIterator[RedisConnection]:
    connection = RedisConnection(integration_settings)
    await connection.connect()
    yield connection
    await connection.close()


@pytest.fixture
def store(redis: RedisConnection) -> RedisRobotsStore:
    return RedisRobotsStore(redis.client, redis.keys)


@pytest.fixture
async def fetcher(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[HttpFetcher]:
    config = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


def _policy() -> RetryPolicy:
    return RetryPolicy(max_attempts=2, base_delay=0.001, max_delay=0.001, max_retry_after=1)


# ---- #
# Tests
# ---- #


class TestRobotsRedisStore_S29:
    """S29 — robots.txt cache in Redis: keys, TTLs, last good copy, sharing across workers."""

    async def test_round_trip_with_ttl(
        self, store: RedisRobotsStore, redis: RedisConnection
    ) -> None:
        entry = CachedRobots(RobotsStatus.OK, "User-agent: *\nDisallow: /x\n", 10.0, 3610.0)

        await store.save(_ORIGIN, entry, ttl=3600)

        key = redis.keys.key(RedisNamespace.ROBOTS, RedisKeyspace.digest(_ORIGIN))
        assert await store.load(_ORIGIN) == entry
        assert 3_590_000 < await redis.client.pttl(key) <= 3_600_000
        assert await store.load_last_good(_ORIGIN) is None  # a separate key

    async def test_last_good_copy_is_a_separate_key(
        self, store: RedisRobotsStore, redis: RedisConnection
    ) -> None:
        good = CachedRobots(RobotsStatus.OK, "User-agent: *\n", 1.0, 2.0)
        stale = CachedRobots(RobotsStatus.STALE, good.body, 5.0, 6.0, "http_503", 1.0)

        await store.save_last_good(_ORIGIN, good, ttl=600)
        await store.save(_ORIGIN, stale, ttl=60)

        assert await store.load_last_good(_ORIGIN) == good
        assert await store.load(_ORIGIN) == stale
        keys = [
            k async for k in redis.client.scan_iter(match=redis.keys.pattern(RedisNamespace.ROBOTS))
        ]
        assert len(keys) == 2

    async def test_unreadable_entry_is_ignored(
        self, store: RedisRobotsStore, redis: RedisConnection
    ) -> None:
        key = redis.keys.key(RedisNamespace.ROBOTS, RedisKeyspace.digest(_ORIGIN))
        await redis.client.set(key, "{not json")
        await redis.client.set(key + ":last_good", json.dumps({"status": "nope"}))

        assert await store.load(_ORIGIN) is None
        assert await store.load_last_good(_ORIGIN) is None

    async def test_workers_share_one_fetch(
        self, store: RedisRobotsStore, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        # two caches = two worker processes, each with its own local cache
        worker_a = RobotsCache(fetcher, store, policy=_policy())
        worker_b = RobotsCache(fetcher, store, policy=_policy())

        assert not await worker_a.can_fetch(fixture_site.url("/private/secret.html"))
        rules_b = await worker_b.rules_for(fixture_site.url("/about.html"))

        assert rules_b.status is RobotsStatus.OK
        assert not rules_b.can_fetch(fixture_site.url("/private/secret.html"))
        assert rules_b.crawl_delay == 0.1
        assert fixture_site.hits("/robots.txt") == 1

    async def test_expired_redis_entry_is_refetched(
        self,
        store: RedisRobotsStore,
        redis: RedisConnection,
        fetcher: HttpFetcher,
        fixture_site: FixtureSite,
    ) -> None:
        await RobotsCache(fetcher, store, policy=_policy()).rules_for(fixture_site.url("/"))
        key = redis.keys.key(RedisNamespace.ROBOTS, RedisKeyspace.digest(fixture_site.base_url))
        await redis.client.delete(key)  # what TTL expiry does

        await RobotsCache(fetcher, store, policy=_policy()).rules_for(fixture_site.url("/"))

        assert fixture_site.hits("/robots.txt") == 2
        assert await store.load_last_good(fixture_site.base_url) is not None
