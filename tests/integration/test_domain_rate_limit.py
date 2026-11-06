"""Per-domain token bucket in real Redis: spacing, bursts, concurrency across workers, pauses."""

import asyncio
import itertools
import time
from collections.abc import AsyncIterator, Callable

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import DomainRateLimiter, RateLimitWaitExceededError
from rag_engine.infrastructure import RedisConnection, RedisKeyspace, RedisNamespace
from tests.conftest import OverrideSettings

_URL = "https://acme.example/page"

type MakeLimiter = Callable[..., DomainRateLimiter]


@pytest.fixture
async def redis(integration_settings: Settings) -> AsyncIterator[RedisConnection]:
    connection = RedisConnection(integration_settings)
    await connection.connect()
    yield connection
    await connection.close()


@pytest.fixture
async def other_worker_redis(integration_settings: Settings) -> AsyncIterator[RedisConnection]:
    """A second, independent connection pool: stands in for another worker process."""
    connection = RedisConnection(integration_settings)
    await connection.connect()
    yield connection
    await connection.close()


@pytest.fixture
def make_limiter(
    integration_settings: Settings, override_settings: OverrideSettings, redis: RedisConnection
) -> MakeLimiter:
    def make(connection: RedisConnection | None = None, **overrides: object) -> DomainRateLimiter:
        config = override_settings(**{"CRAWL_PER_DOMAIN_RPS": 20.0, **overrides})
        conn = connection or redis
        return DomainRateLimiter(conn.client, conn.keys, config)

    return make


def _gaps(starts: list[float]) -> list[float]:
    ordered = sorted(starts)
    return [b - a for a, b in itertools.pairwise(ordered)]


# ---- #
# Tests
# ---- #


class TestDomainRateLimiter_S30:
    """S30 — Per-domain rate limiter: Redis GCRA token bucket shared across workers."""

    async def test_first_request_is_immediate_then_spaced(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter()  # 20 rps → 50 ms

        first = await limiter.reserve(_URL)
        second = await limiter.reserve(_URL)

        assert first.granted and first.wait_seconds == 0
        assert second.start_at - first.start_at == pytest.approx(0.05, abs=1e-6)
        assert 0.03 < second.wait_seconds <= 0.05

    async def test_concurrent_workers_get_evenly_spaced_slots(
        self, make_limiter: MakeLimiter, other_worker_redis: RedisConnection
    ) -> None:
        # 0.5 s slots: all 40 bookings land well inside the first slot, so a backlog exists
        # and every gap is exact (an idle bucket would legitimately give a larger gap)
        worker_a = make_limiter(CRAWL_PER_DOMAIN_RPS=2.0)
        worker_b = make_limiter(other_worker_redis, CRAWL_PER_DOMAIN_RPS=2.0)
        for worker in (worker_a, worker_b):  # load the script on both connection pools
            await worker.reserve("https://warm-up.example/")

        reservations = await asyncio.gather(
            *(w.reserve(_URL) for _ in range(20) for w in (worker_a, worker_b))
        )

        starts = [r.start_at for r in reservations]
        assert len(set(starts)) == 40  # no two requests share a slot
        assert _gaps(starts) == pytest.approx([0.5] * 39, abs=1e-6)
        assert max(r.wait_seconds for r in reservations) == pytest.approx(19.5, abs=0.2)

    async def test_acquire_really_waits(
        self, make_limiter: MakeLimiter, other_worker_redis: RedisConnection
    ) -> None:
        worker_a, worker_b = make_limiter(), make_limiter(other_worker_redis)
        done: list[float] = []

        async def fetch(limiter: DomainRateLimiter) -> None:
            await limiter.acquire(_URL)
            done.append(time.perf_counter())

        start = time.perf_counter()
        await asyncio.gather(*(fetch(w) for _ in range(5) for w in (worker_a, worker_b)))
        elapsed = time.perf_counter() - start

        assert 0.4 <= elapsed < 1.5  # 10 requests, 9 gaps of 50 ms
        assert min(_gaps(done)) > 0.03

    async def test_burst_then_spacing(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter(CRAWL_PER_DOMAIN_RPS=10.0, CRAWL_PER_DOMAIN_BURST=3)

        reservations = [await limiter.reserve(_URL) for _ in range(5)]

        starts = [r.start_at for r in reservations]
        assert starts[1] - starts[0] < 0.01 and starts[2] - starts[0] < 0.01
        assert starts[3] - starts[0] == pytest.approx(0.1, abs=0.01)
        assert starts[4] - starts[3] == pytest.approx(0.1, abs=1e-6)

    async def test_crawl_delay_overrides_rps_and_burst(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter(CRAWL_PER_DOMAIN_BURST=5)

        a = await limiter.reserve(_URL, crawl_delay=0.3)
        b = await limiter.reserve(_URL, crawl_delay=0.3)

        assert b.start_at - a.start_at == pytest.approx(0.3, abs=1e-6)
        assert (b.interval_seconds, b.burst) == (0.3, 1)

    async def test_hosts_are_independent_and_ports_shared(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter(CRAWL_PER_DOMAIN_RPS=1.0)

        await limiter.reserve("https://acme.example/a")
        other_host = await limiter.reserve("https://docs.acme.example/a")
        same_host = await limiter.reserve("http://ACME.example:8080/b")

        assert other_host.wait_seconds == 0
        assert same_host.wait_seconds > 0.9

    async def test_refused_reservation_books_nothing(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter(CRAWL_PER_DOMAIN_RPS=1.0)
        first = await limiter.reserve(_URL)

        refused = await limiter.reserve(_URL, max_wait=0.1)
        booked = await limiter.reserve(_URL)

        assert not refused.granted
        assert refused.wait_seconds > 0.1
        assert booked.start_at == pytest.approx(first.start_at + 1.0, abs=1e-6)

    async def test_acquire_raises_past_max_wait(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter(CRAWL_PER_DOMAIN_RPS=1.0)
        await limiter.acquire(_URL)

        with pytest.raises(RateLimitWaitExceededError) as exc:
            await limiter.acquire(_URL, max_wait=0.2)

        assert exc.value.reservation.host == "acme.example"

    async def test_idle_bucket_refills(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter()
        await limiter.reserve(_URL)

        await asyncio.sleep(0.08)

        assert (await limiter.reserve(_URL)).wait_seconds == 0

    async def test_pause_holds_every_worker(
        self, make_limiter: MakeLimiter, other_worker_redis: RedisConnection
    ) -> None:
        worker_a, worker_b = make_limiter(), make_limiter(other_worker_redis)

        remaining = await worker_a.pause(_URL, 5)
        blocked = await worker_b.reserve(_URL, max_wait=1)
        booked = await worker_b.reserve(_URL)
        after = await worker_a.reserve(_URL)

        assert remaining == pytest.approx(5, abs=0.01)
        assert not blocked.granted and blocked.wait_seconds == pytest.approx(5, abs=0.05)
        assert booked.wait_seconds == pytest.approx(5, abs=0.05)
        assert after.start_at - booked.start_at == pytest.approx(0.05, abs=1e-6)
        assert (await worker_a.reserve("https://other.example/")).wait_seconds == 0

    async def test_pause_is_never_shortened(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter()

        await limiter.pause(_URL, 5)
        remaining = await limiter.pause(_URL, 1)

        assert remaining == pytest.approx(5, abs=0.05)

    async def test_pause_is_capped_and_zero_is_a_noop(self, make_limiter: MakeLimiter) -> None:
        limiter = make_limiter(CRAWL_RETRY_AFTER_MAX_SECONDS=10)

        assert await limiter.pause(_URL, 0) == 0
        assert (await limiter.reserve(_URL)).wait_seconds == 0
        assert await limiter.pause(_URL, 3600) == pytest.approx(10, abs=0.05)

    async def test_bucket_key_expires_when_idle(
        self, make_limiter: MakeLimiter, redis: RedisConnection
    ) -> None:
        limiter = make_limiter(CRAWL_PER_DOMAIN_RPS=1.0)

        await limiter.reserve(_URL)

        key = redis.keys.key(RedisNamespace.DOMAIN_RATE_LIMIT, RedisKeyspace.digest("acme.example"))
        assert 0 < await redis.client.pttl(key) <= 1001
        assert set(await redis.client.hkeys(key)) == {"tat"}

    async def test_microsecond_state_is_stored_exactly(
        self, make_limiter: MakeLimiter, redis: RedisConnection
    ) -> None:
        # Lua's tostring() would print 16-digit timestamps as 1.7e+15 and lose precision
        limiter = make_limiter(CRAWL_PER_DOMAIN_RPS=3.0)  # 333 333 µs: not a round number

        a = await limiter.reserve(_URL)
        b = await limiter.reserve(_URL)

        key = redis.keys.key(RedisNamespace.DOMAIN_RATE_LIMIT, RedisKeyspace.digest("acme.example"))
        assert (await redis.client.hget(key, "tat") or "").isdigit()
        assert b.start_at - a.start_at == pytest.approx(0.333333, abs=1e-6)
