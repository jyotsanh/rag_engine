"""Cancelling a running crawl job: a Redis flag the engine polls (Step 52).

MongoDB stays the source of truth: `JobRepository.request_cancel` cancels a queued job at
once and stamps `cancel_requested_at` on a running one. The Redis flag
(`job_cancel:<job_id>`) only makes the engine notice fast and cheaply, every
`CRAWL_CANCEL_POLL_SECONDS`, instead of reading the job document on every poll.

Flow:
1. API (Step 87): `request_cancel(jobs, flags, tenant_id, job_id)`: Mongo first, then the
   flag for a running job. If Redis is down the request still succeeds; the worker's
   heartbeat sees `cancel_requested_at` and calls `CrawlEngine.cancel()` (Step 77).
2. Engine: the poller sees the flag and stops the crawl (`CrawlStopReason.CANCELLED`, a hard
   stop). In-flight pages drain, everything still queued becomes `skipped/job_cancelled`.
3. Worker (Step 77): on `summary.stop.reason is CANCELLED`, `finish_cancelled(...)` moves the
   job to `cancelled` and clears the flag.
"""

from typing import Protocol

from loguru import logger
from redis.asyncio import Redis
from redis.exceptions import RedisError

from rag_engine.config import Settings, settings
from rag_engine.domain import CrawlJob, JobStatus
from rag_engine.infrastructure import RedisKeyspace, RedisNamespace, RedisUnavailableError
from rag_engine.repositories import JobRepository


class CancelSignal(Protocol):
    """What the engine polls; `RedisCancelFlags` in production."""

    async def is_cancelled(self, job_id: str) -> bool: ...


class RedisCancelFlags:
    """`job_cancel:<job_id>` = "1", expiring after `CRAWL_CANCEL_FLAG_TTL_SECONDS`.

    The TTL outlives any running job (`>= CRAWL_MAX_JOB_SECONDS`), so a flag can't vanish
    under its job, and flags whose job never cleaned up expire on their own.
    """

    def __init__(self, redis: Redis, keys: RedisKeyspace, config: Settings = settings) -> None:
        self._redis = redis
        self._keys = keys
        self._ttl = config.CRAWL_CANCEL_FLAG_TTL_SECONDS

    def key(self, job_id: str) -> str:
        return self._keys.key(RedisNamespace.JOB_CANCEL, job_id)

    async def request(self, job_id: str) -> None:
        try:
            await self._redis.set(self.key(job_id), "1", ex=self._ttl)
        except (RedisError, OSError) as exc:
            raise RedisUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def is_cancelled(self, job_id: str) -> bool:
        try:
            return bool(await self._redis.exists(self.key(job_id)))
        except (RedisError, OSError) as exc:
            raise RedisUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def clear(self, job_id: str) -> None:
        try:
            await self._redis.delete(self.key(job_id))
        except (RedisError, OSError) as exc:
            raise RedisUnavailableError(details={"reason": type(exc).__name__}) from exc


async def request_cancel(
    jobs: JobRepository, flags: RedisCancelFlags, tenant_id: str, job_id: str
) -> CrawlJob:
    """The API's cancel: queued → cancelled at once; running → flagged in Mongo and Redis.

    Idempotent (an already-cancelled job is returned as is). Finalizing and finished jobs
    raise `InvalidJobTransitionError`. A Redis outage doesn't fail the request: the
    durable `cancel_requested_at` still reaches the worker through its heartbeat.
    """
    job = await jobs.request_cancel(tenant_id, job_id)
    if job.status is JobStatus.RUNNING:
        try:
            await flags.request(job.id)
        except RedisUnavailableError:
            logger.bind(job_id=job.id).warning(
                "cancel flag not set (Redis unavailable); the worker's heartbeat will see it"
            )
    return job


async def finish_cancelled(
    jobs: JobRepository, flags: RedisCancelFlags, job_id: str, run_id: str
) -> CrawlJob:
    """After the engine drained a cancelled crawl: `running → cancelled`, then drop the flag.

    The flag is cleared best-effort (it expires anyway).
    """
    job = await jobs.cancel(job_id, run_id)
    try:
        await flags.clear(job_id)
    except RedisUnavailableError:
        logger.bind(job_id=job_id).warning("cancel flag not cleared (Redis unavailable)")
    return job
