"""robots.txt: fetch through the SSRF-safe fetcher, parse with protego, cache per origin.

Status semantics follow RFC 9309 §2.3.1:
- 2xx: parse it (first `ROBOTS_MAX_BYTES` only; the rest is ignored, not an error).
- 4xx except 429 ("unavailable"): no rules, everything is allowed. Redirect failures and
  SSRF-blocked redirect targets count as unavailable as well.
- 5xx, 429, network errors and timeouts ("unreachable"), after the Step 28 retries: assume
  **complete disallow**. If a copy fetched within `ROBOTS_STALE_MAX_SECONDS` exists, it is
  used instead (status `stale`). Unreachable verdicts are cached only briefly.

Two cache levels: a shared store (Redis, so every worker sees one fetch per origin per
TTL) and an in-process LRU of parsed rules, so checking a URL does not hit Redis or
re-parse. Concurrent lookups of one origin in a process share a single fetch.
"""

import asyncio
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit

from loguru import logger
from protego import Protego
from redis.asyncio import Redis

from rag_engine.config import Settings, settings
from rag_engine.crawling.fetchers import FetchError, HttpStatusError, RawResponse
from rag_engine.crawling.retry import (
    RetriesExhaustedError,
    RetryPolicy,
    call_with_retries,
)
from rag_engine.crawling.url import normalize_url
from rag_engine.domain import PageReason
from rag_engine.infrastructure import RedisKeyspace, RedisNamespace

ROBOTS_PATH = "/robots.txt"
# RFC 9309 §2.5: crawlers must parse at least 500 KiB; anything beyond is ignored
ROBOTS_MAX_BYTES = 500 * 1024
_ACCEPT = "text/plain, */*;q=0.1"
_LOCAL_CACHE_SIZE = 1024


class RobotsStatus(StrEnum):
    OK = "ok"  # fetched and parsed
    UNAVAILABLE = "unavailable"  # 4xx / redirect failure: no restrictions
    UNREACHABLE = "unreachable"  # 5xx / 429 / network: complete disallow
    STALE = "stale"  # unreachable now, serving the last good copy


def origin_of(url: str) -> str:
    """`scheme://host[:port]` of a URL, normalized (default port dropped, host IDNA-encoded)."""
    parts = urlsplit(normalize_url(url))
    return f"{parts.scheme}://{parts.netloc}"


def robots_token(user_agent: str) -> str:
    """Product token of a User-Agent (`RagEngineBot/0.1 (+url)` → `RagEngineBot`).

    Matching on the token rather than the full string stops a group such as
    `User-agent: github` from matching because our UA happens to contain a URL.
    """
    words = user_agent.split("/", 1)[0].split()
    if not words:
        raise ValueError("user agent has no product token")
    return words[0]


@dataclass(frozen=True, slots=True)
class CachedRobots:
    """What the shared store keeps per origin (raw body, so any worker can re-parse it)."""

    status: RobotsStatus
    body: str
    fetched_at: float  # epoch seconds when this verdict was made
    expires_at: float  # epoch seconds
    detail: str | None = None  # e.g. "http_503", "dns_nxdomain"
    source_fetched_at: float | None = None  # STALE: when the reused copy was fetched

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> CachedRobots:
        data = json.loads(raw)
        data["status"] = RobotsStatus(data["status"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RobotsRules:
    """Parsed robots.txt verdict for one origin, as seen by one user-agent token."""

    origin: str
    status: RobotsStatus
    user_agent: str
    fetched_at: float
    expires_at: float
    detail: str | None = None
    _parser: Protego | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_cached(cls, origin: str, entry: CachedRobots, user_agent: str) -> RobotsRules:
        parser = (
            Protego.parse(entry.body)
            if entry.status in (RobotsStatus.OK, RobotsStatus.STALE)
            else None
        )
        return cls(
            origin=origin,
            status=entry.status,
            user_agent=user_agent,
            fetched_at=entry.fetched_at,
            expires_at=entry.expires_at,
            detail=entry.detail,
            _parser=parser,
        )

    def can_fetch(self, url: str) -> bool:
        if urlsplit(url).path == ROBOTS_PATH:  # RFC 9309 §2.2.2: always allowed
            return True
        if self.status is RobotsStatus.UNAVAILABLE:
            return True
        if self.status is RobotsStatus.UNREACHABLE or self._parser is None:
            return False
        return self._parser.can_fetch(url, self.user_agent)

    @property
    def crawl_delay(self) -> float | None:
        """Seconds between requests asked by `Crawl-delay` or `Request-rate` (the larger)."""
        if self._parser is None:
            return None
        delays: list[float] = []
        if (delay := self._parser.crawl_delay(self.user_agent)) is not None:
            delays.append(float(delay))
        rate = self._parser.request_rate(self.user_agent)
        if rate is not None and rate.requests > 0:
            delays.append(rate.seconds / rate.requests)
        return max(delays) if delays else None

    @property
    def sitemaps(self) -> tuple[str, ...]:
        return tuple(self._parser.sitemaps) if self._parser is not None else ()


class RawFetcher(Protocol):
    """What robots needs from a fetcher (`HttpFetcher` in production)."""

    async def fetch_raw(
        self, url: str, *, accept: str, max_bytes: int, truncate: bool = False
    ) -> RawResponse: ...


class RobotsStore(Protocol):
    """Shared cache of robots verdicts per origin (Redis in production)."""

    async def load(self, origin: str) -> CachedRobots | None: ...
    async def save(self, origin: str, entry: CachedRobots, ttl: float) -> None: ...
    async def load_last_good(self, origin: str) -> CachedRobots | None: ...
    async def save_last_good(self, origin: str, entry: CachedRobots, ttl: float) -> None: ...


class RedisRobotsStore:
    """`robots:<digest(origin)>` (current verdict) and `robots:<digest>:last_good`."""

    def __init__(self, redis: Redis, keys: RedisKeyspace) -> None:
        self._redis = redis
        self._keys = keys

    def _key(self, origin: str, *suffix: str) -> str:
        return self._keys.key(RedisNamespace.ROBOTS, RedisKeyspace.digest(origin), *suffix)

    async def _get(self, key: str) -> CachedRobots | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return CachedRobots.from_json(raw)
        except (ValueError, TypeError, KeyError) as exc:  # corrupt / old format: refetch
            logger.warning("ignoring unreadable robots cache entry {}: {}", key, exc)
            return None

    async def _set(self, key: str, entry: CachedRobots, ttl: float) -> None:
        await self._redis.set(key, entry.to_json(), px=max(int(ttl * 1000), 1))

    async def load(self, origin: str) -> CachedRobots | None:
        return await self._get(self._key(origin))

    async def save(self, origin: str, entry: CachedRobots, ttl: float) -> None:
        await self._set(self._key(origin), entry, ttl)

    async def load_last_good(self, origin: str) -> CachedRobots | None:
        return await self._get(self._key(origin, "last_good"))

    async def save_last_good(self, origin: str, entry: CachedRobots, ttl: float) -> None:
        await self._set(self._key(origin, "last_good"), entry, ttl)


class RobotsCache:
    """Answers "may we fetch this URL?" and "how slowly?" for any origin."""

    def __init__(
        self,
        fetcher: RawFetcher,
        store: RobotsStore,
        *,
        config: Settings = settings,
        policy: RetryPolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._fetcher = fetcher
        self._store = store
        self._policy = policy or RetryPolicy.from_settings(config)
        self._clock = clock
        self._user_agent = robots_token(config.CRAWL_USER_AGENT)
        self._ttl = config.ROBOTS_CACHE_TTL_SECONDS
        self._unreachable_ttl = config.ROBOTS_UNREACHABLE_CACHE_TTL_SECONDS
        self._stale_max = config.ROBOTS_STALE_MAX_SECONDS
        self._local: OrderedDict[str, RobotsRules] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[RobotsRules]] = {}

    async def can_fetch(self, url: str) -> bool:
        return (await self.rules_for(url)).can_fetch(url)

    async def rules_for(self, url: str) -> RobotsRules:
        origin = origin_of(url)
        if (rules := self._local.get(origin)) is not None and rules.expires_at > self._clock():
            self._local.move_to_end(origin)
            return rules

        task = self._inflight.get(origin)
        if task is None:  # single flight: one fetch per origin per process
            task = asyncio.ensure_future(self._resolve(origin))
            self._inflight[origin] = task
            task.add_done_callback(lambda _: self._inflight.pop(origin, None))
        # shield: a cancelled caller must not cancel the fetch other callers wait on
        return await asyncio.shield(task)

    async def _resolve(self, origin: str) -> RobotsRules:
        entry = await self._store.load(origin)
        if entry is None or entry.expires_at <= self._clock():
            entry = await self._refresh(origin)
        rules = RobotsRules.from_cached(origin, entry, self._user_agent)
        self._local[origin] = rules
        self._local.move_to_end(origin)
        while len(self._local) > _LOCAL_CACHE_SIZE:
            self._local.popitem(last=False)
        return rules

    async def _refresh(self, origin: str) -> CachedRobots:
        url = origin + ROBOTS_PATH
        now = self._clock()
        try:
            response = await call_with_retries(
                lambda: self._fetcher.fetch_raw(
                    url, accept=_ACCEPT, max_bytes=ROBOTS_MAX_BYTES, truncate=True
                ),
                self._policy,
            )
        except RetriesExhaustedError as exc:
            entry = await self._failure_entry(origin, exc.error, exc.decision.detail, now)
        else:
            body = response.body.decode("utf-8", errors="replace").removeprefix("﻿")
            detail = "truncated" if response.truncated else None
            entry = CachedRobots(RobotsStatus.OK, body, now, now + self._ttl, detail)
            await self._store.save_last_good(origin, entry, self._stale_max)

        ttl = entry.expires_at - now
        await self._store.save(origin, entry, ttl)
        logger.info("robots.txt for {}: {} ({})", origin, entry.status, entry.detail or "-")
        return entry

    async def _failure_entry(
        self, origin: str, error: FetchError, detail: str, now: float
    ) -> CachedRobots:
        if _is_unavailable(error):
            return CachedRobots(RobotsStatus.UNAVAILABLE, "", now, now + self._ttl, detail)

        expires = now + self._unreachable_ttl
        last_good = await self._store.load_last_good(origin)
        if last_good is not None and now - last_good.fetched_at <= self._stale_max:
            return CachedRobots(
                RobotsStatus.STALE,
                last_good.body,
                now,
                expires,
                detail,
                source_fetched_at=last_good.fetched_at,
            )
        return CachedRobots(RobotsStatus.UNREACHABLE, "", now, expires, detail)


_UNAVAILABLE_REASONS = frozenset(
    {PageReason.REDIRECT_LOOP, PageReason.TOO_MANY_REDIRECTS, PageReason.SSRF_BLOCKED}
)


def _is_unavailable(error: FetchError) -> bool:
    """4xx (not 429) and redirect/SSRF failures mean "no robots.txt", not "site down"."""
    if isinstance(error, HttpStatusError):
        return error.status_code < 500 and error.status_code != 429
    return error.reason in _UNAVAILABLE_REASONS
