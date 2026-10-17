"""Async Redis client lifecycle and the application's key namespace.

All keys are built through `RedisKeyspace` so they share one prefix
(`settings.REDIS_KEY_PREFIX`) and a fixed set of namespaces; nothing else
writes raw key strings. Celery's broker/result DBs are separate and not
touched by this client.
"""

import hashlib
from enum import StrEnum

from redis.asyncio import Redis
from redis.exceptions import RedisError

from rag_engine.config import Settings, logger, settings
from rag_engine.infrastructure.exceptions import RedisUnavailableError

_SEPARATOR = ":"
_DELETE_BATCH = 500


class RedisNamespace(StrEnum):
    """Every family of keys the application stores. Add a member before using a new family."""

    API_RATE_LIMIT = "api_rl"  # per-tenant API sliding windows (Step 22)
    ROBOTS = "robots"  # robots.txt cache per origin (Step 29)
    DOMAIN_RATE_LIMIT = "domain_rl"  # per-domain crawl token buckets (Step 30)
    JOB_CANCEL = "job_cancel"  # crawl-job cancel flags (Step 52)
    QUERY_EMBEDDING = "query_emb"  # query vector cache (Step 92)
    LOCK = "lock"  # short distributed locks


class RedisKeyspace:
    """Builds `<prefix>:<namespace>:<part>[:<part>...]` keys.

    Parts must be non-empty and free of whitespace/control characters. Values
    that are long or contain the separator (URLs, hosts with ports) should go
    through `digest()` so keys stay short and unambiguous.
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def key(self, namespace: RedisNamespace, *parts: str) -> str:
        if not parts:
            raise ValueError("a key needs at least one part after the namespace")
        for part in parts:
            if not part or any(ch.isspace() or not ch.isprintable() for ch in part):
                raise ValueError(f"invalid redis key part: {part!r}")
            if _SEPARATOR in part:
                raise ValueError(f"key part contains {_SEPARATOR!r}; use digest(): {part!r}")
        return _SEPARATOR.join((self.prefix, namespace.value, *parts))

    def pattern(self, namespace: RedisNamespace) -> str:
        """Glob matching every key in a namespace (for SCAN-based maintenance)."""
        return f"{self.prefix}{_SEPARATOR}{namespace.value}{_SEPARATOR}*"

    @staticmethod
    def digest(value: str) -> str:
        """Stable short token for arbitrary values (URLs, origins)."""
        return hashlib.sha256(value.encode()).hexdigest()[:32]


class RedisConnection:
    """Owns one async connection pool to the application Redis DB."""

    def __init__(self, config: Settings = settings) -> None:
        self._config = config
        self._client: Redis | None = None
        self.keys = RedisKeyspace(config.REDIS_KEY_PREFIX)

    @property
    def client(self) -> Redis:
        if self._client is None:
            raise RedisUnavailableError("Redis connection is not open")
        return self._client

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = Redis.from_url(
            self._config.redis_url,
            decode_responses=True,
            socket_timeout=self._config.REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_connect_timeout=self._config.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
            max_connections=self._config.REDIS_MAX_CONNECTIONS,
            health_check_interval=30,
            client_name=self._config.PROJECT_NAME.replace(" ", "_").lower(),
        )
        try:
            await self.ping()
        except RedisUnavailableError:
            await self.close()
            raise
        logger.info("connected to Redis db {}", self._config.REDIS_DB)

    async def ping(self) -> None:
        try:
            await self.client.ping()
        except (RedisError, OSError) as exc:
            raise RedisUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def delete_namespace(self, namespace: RedisNamespace) -> int:
        """Remove every key of a namespace with SCAN + UNLINK (never KEYS/FLUSHDB)."""
        deleted = 0
        batch: list[str] = []
        async for key in self.client.scan_iter(match=self.keys.pattern(namespace), count=1000):
            batch.append(key)
            if len(batch) >= _DELETE_BATCH:
                deleted += await self.client.unlink(*batch)
                batch.clear()
        if batch:
            deleted += await self.client.unlink(*batch)
        return deleted

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
