"""Per-domain politeness: a Redis token bucket shared by every crawl worker.

The bucket is GCRA ("generic cell rate algorithm", a token bucket stored as one
timestamp). Each host keeps a theoretical arrival time `tat`. A request may start at
`tat - (burst - 1) · interval` and pushes `tat` forward by one interval. One Lua script
does this atomically on the Redis clock, so workers on different machines (and clocks)
agree.

`reserve()` **books** a start time rather than polling. Concurrent callers get distinct,
evenly spaced slots from a single round-trip, with no thundering herd and FIFO order. The
cost: a caller cancelled while waiting leaves its slot unused, which only slows that host
by one interval.

The interval is the larger of `1 / CRAWL_PER_DOMAIN_RPS` and the robots.txt crawl delay,
which is clamped to `CRAWL_MAX_CRAWL_DELAY_SECONDS`. A host can also be paused, e.g.
when it answers 429 with `Retry-After`, which holds back every worker, not just one URL.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from redis.asyncio import Redis

from rag_engine.config import Settings, settings
from rag_engine.crawling.url import normalize_url
from rag_engine.infrastructure import RedisKeyspace, RedisNamespace

_US = 1_000_000

# KEYS[1] = bucket hash {tat, paused_until} (microseconds, Redis clock)
# ARGV = interval_us, burst, max_wait_us (-1 = unlimited)
# returns {granted (0/1), wait_us, start_us}
# string.format('%d') keeps 16-digit microsecond values exact (tostring would use %.14g)
_RESERVE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000000 + tonumber(t[2])
local interval = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local max_wait = tonumber(ARGV[3])
local state = redis.call('HMGET', KEYS[1], 'tat', 'paused_until')
local tat = tonumber(state[1]) or now
if tat < now then tat = now end
local paused = tonumber(state[2]) or 0
local start = math.max(tat - (burst - 1) * interval, now, paused)
if start > tat then tat = start end
local wait = start - now
if max_wait >= 0 and wait > max_wait then
  return {0, wait, start}
end
tat = tat + interval
redis.call('HSET', KEYS[1], 'tat', string.format('%d', tat))
redis.call('PEXPIRE', KEYS[1], math.floor((math.max(tat, paused) - now) / 1000) + 1)
return {1, wait, start}
"""

# KEYS[1] = bucket hash; ARGV = pause_us. Returns the remaining pause in microseconds.
_PAUSE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000000 + tonumber(t[2])
local target = now + tonumber(ARGV[1])
local state = redis.call('HMGET', KEYS[1], 'tat', 'paused_until')
local paused = tonumber(state[2]) or 0
if target > paused then
  paused = target
  redis.call('HSET', KEYS[1], 'paused_until', string.format('%d', paused))
  local tat = tonumber(state[1]) or 0
  redis.call('PEXPIRE', KEYS[1], math.floor((math.max(tat, paused) - now) / 1000) + 1)
end
return paused - now
"""


def rate_limit_host(url: str) -> str:
    """The politeness key of a URL: its normalized host. Scheme and port are ignored,
    since http/https and other ports of one host usually share one server."""
    host = urlsplit(normalize_url(url)).hostname
    assert host is not None  # normalize_url guarantees a host
    return host


@dataclass(frozen=True, slots=True)
class Reservation:
    host: str
    granted: bool  # False only when the wait would exceed `max_wait` (nothing was booked)
    wait_seconds: float  # until the booked start (or the start that was refused)
    start_at: float  # Redis clock, epoch seconds
    interval_seconds: float
    burst: int


class RateLimitWaitExceededError(Exception):
    """The next free slot for a host is further away than the caller is willing to wait."""

    def __init__(self, reservation: Reservation) -> None:
        super().__init__(
            f"{reservation.host}: next slot in {reservation.wait_seconds:.2f}s exceeds max wait"
        )
        self.reservation = reservation


class DomainRateLimiter:
    def __init__(
        self,
        redis: Redis,
        keys: RedisKeyspace,
        config: Settings = settings,
        *,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self._reserve = redis.register_script(_RESERVE)
        self._pause = redis.register_script(_PAUSE)
        self._keys = keys
        self._sleep = sleep
        self._min_interval = 1.0 / config.CRAWL_PER_DOMAIN_RPS
        self._burst = config.CRAWL_PER_DOMAIN_BURST
        self._max_crawl_delay = config.CRAWL_MAX_CRAWL_DELAY_SECONDS
        self._max_pause = config.CRAWL_RETRY_AFTER_MAX_SECONDS

    def interval_for(self, crawl_delay: float | None) -> float:
        """Seconds between requests to one host: the stricter of our RPS and the site's ask."""
        if crawl_delay is None or crawl_delay <= 0:
            return self._min_interval
        return max(self._min_interval, min(crawl_delay, self._max_crawl_delay))

    def burst_for(self, crawl_delay: float | None) -> int:
        # a site that states a crawl delay gets strictly spaced requests, never a burst
        return 1 if crawl_delay is not None and crawl_delay > 0 else self._burst

    def _key(self, host: str) -> str:
        return self._keys.key(RedisNamespace.DOMAIN_RATE_LIMIT, RedisKeyspace.digest(host))

    async def reserve(
        self, url: str, *, crawl_delay: float | None = None, max_wait: float | None = None
    ) -> Reservation:
        """Book the next slot for the URL's host without waiting for it."""
        if max_wait is not None and max_wait < 0:
            raise ValueError("max_wait must be >= 0")
        host = rate_limit_host(url)
        interval = self.interval_for(crawl_delay)
        burst = self.burst_for(crawl_delay)
        granted, wait_us, start_us = await self._reserve(
            keys=[self._key(host)],
            args=[
                max(round(interval * _US), 1),
                burst,
                -1 if max_wait is None else round(max_wait * _US),
            ],
        )
        return Reservation(
            host=host,
            granted=bool(granted),
            wait_seconds=int(wait_us) / _US,
            start_at=int(start_us) / _US,
            interval_seconds=interval,
            burst=burst,
        )

    async def acquire(
        self, url: str, *, crawl_delay: float | None = None, max_wait: float | None = None
    ) -> Reservation:
        """Wait until this request may go. Raises `RateLimitWaitExceededError` past `max_wait`."""
        reservation = await self.reserve(url, crawl_delay=crawl_delay, max_wait=max_wait)
        if not reservation.granted:
            raise RateLimitWaitExceededError(reservation)
        if reservation.wait_seconds > 0:
            await self._sleep(reservation.wait_seconds)
        return reservation

    async def pause(self, url: str, seconds: float) -> float:
        """Hold every worker off the URL's host for `seconds` (capped at
        `CRAWL_RETRY_AFTER_MAX_SECONDS`). Never shortens an existing pause.
        Returns the remaining pause."""
        if seconds <= 0:
            return 0.0
        pause_us = round(min(seconds, self._max_pause) * _US)
        remaining_us = await self._pause(keys=[self._key(rate_limit_host(url))], args=[pause_us])
        return int(remaining_us) / _US
