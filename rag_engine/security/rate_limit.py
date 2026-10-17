"""Sliding-window rate limiting in Redis (exact, per subject).

Each subject (e.g. a tenant) has a sorted set of request timestamps. One Lua
script trims entries older than the window, counts, and records the request
atomically, so concurrent requests across API replicas can never exceed the
limit. Timestamps come from the Redis server clock (`TIME`), so replicas with
skewed clocks still agree on the window.
"""

import math
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis

# KEYS[1] = zset key; ARGV = limit, window_ms, member
# returns {allowed (0/1), count_after, retry_after_ms}
_SLIDING_WINDOW = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
local count = redis.call('ZCARD', KEYS[1])
if count < limit then
  redis.call('ZADD', KEYS[1], now, now .. '-' .. ARGV[3])
  redis.call('PEXPIRE', KEYS[1], window)
  return {1, count + 1, 0}
end
local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
local retry = window
if oldest[2] then
  retry = tonumber(oldest[2]) + window - now
end
return {0, count, retry}
"""


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    # seconds until a request will be accepted again (0 when allowed)
    retry_after: int
    window_seconds: int

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.retry_after if not self.allowed else self.window_seconds),
        }


class SlidingWindowRateLimiter:
    def __init__(self, redis: Redis) -> None:
        self._script = redis.register_script(_SLIDING_WINDOW)

    async def hit(self, key: str, *, limit: int, window_seconds: int) -> RateLimitDecision:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window_seconds must be positive")
        allowed, count, retry_ms = await self._script(
            keys=[key], args=[limit, window_seconds * 1000, uuid.uuid4().hex]
        )
        return RateLimitDecision(
            allowed=bool(allowed),
            limit=limit,
            remaining=max(limit - int(count), 0),
            retry_after=0 if allowed else max(math.ceil(int(retry_ms) / 1000), 1),
            window_seconds=window_seconds,
        )
