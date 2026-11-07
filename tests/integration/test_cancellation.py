"""Cancel flags in real Redis, and the cancel helpers over real MongoDB."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import RedisCancelFlags, finish_cancelled, request_cancel
from rag_engine.domain import CrawlJob, JobStatus
from rag_engine.infrastructure import MongoConnection, RedisConnection, RedisNamespace
from rag_engine.repositories import InvalidJobTransitionError, JobNotFoundError, JobRepository

_TENANT = "ten_" + "1" * 32
_KB = "kb_" + "1" * 32
_SOURCE = "src_" + "1" * 32


@dataclass
class _Env:
    jobs: JobRepository
    redis: RedisConnection
    flags: RedisCancelFlags

    async def job(self) -> CrawlJob:
        return await self.jobs.insert(
            CrawlJob(tenant_id=_TENANT, kb_id=_KB, source_id=_SOURCE, seed_url="https://a.example/")
        )

    async def running(self, run_id: str = "run-1") -> CrawlJob:
        return await self.jobs.start((await self.job()).id, run_id)


@pytest.fixture
async def env(integration_settings: Settings) -> AsyncIterator[_Env]:
    mongo = MongoConnection(integration_settings)
    redis = RedisConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([CrawlJob])
    await redis.connect()
    yield _Env(
        JobRepository(mongo.db),
        redis,
        RedisCancelFlags(redis.client, redis.keys, integration_settings),
    )
    await redis.close()
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestCancelFlags_S52:
    """S52 — Cancellation: Redis flags and the API/worker cancel helpers."""

    async def test_flag_round_trip_with_ttl(
        self, env: _Env, integration_settings: Settings
    ) -> None:
        job_id = "job_" + "a" * 32

        before = await env.flags.is_cancelled(job_id)
        await env.flags.request(job_id)
        after = await env.flags.is_cancelled(job_id)
        ttl = await env.redis.client.ttl(env.flags.key(job_id))
        await env.flags.clear(job_id)

        assert (before, after) == (False, True)
        assert 0 < ttl <= integration_settings.CRAWL_CANCEL_FLAG_TTL_SECONDS
        assert env.flags.key(job_id) == env.redis.keys.key(RedisNamespace.JOB_CANCEL, job_id)
        assert await env.flags.is_cancelled(job_id) is False
        assert await env.flags.is_cancelled("job_" + "b" * 32) is False  # per job

    async def test_cancelling_a_running_job_flags_mongo_and_redis(self, env: _Env) -> None:
        job = await env.running()

        flagged = await request_cancel(env.jobs, env.flags, _TENANT, job.id)
        again = await request_cancel(env.jobs, env.flags, _TENANT, job.id)

        assert flagged.status is JobStatus.RUNNING and flagged.cancel_requested_at is not None
        assert again.cancel_requested_at == flagged.cancel_requested_at  # the first one is kept
        assert await env.flags.is_cancelled(job.id) is True

    async def test_a_queued_job_is_cancelled_at_once_without_a_flag(self, env: _Env) -> None:
        job = await env.job()

        cancelled = await request_cancel(env.jobs, env.flags, _TENANT, job.id)
        again = await request_cancel(env.jobs, env.flags, _TENANT, job.id)

        assert cancelled.status is again.status is JobStatus.CANCELLED
        assert await env.flags.is_cancelled(job.id) is False

    async def test_finalizing_jobs_and_other_tenants_are_refused(self, env: _Env) -> None:
        job = await env.running()
        await env.jobs.begin_finalizing(job.id, "run-1")
        other = await env.running("run-2")

        with pytest.raises(InvalidJobTransitionError):
            await request_cancel(env.jobs, env.flags, _TENANT, job.id)
        with pytest.raises(JobNotFoundError):
            await request_cancel(env.jobs, env.flags, "ten_" + "9" * 32, other.id)

        assert await env.flags.is_cancelled(job.id) is False
        assert await env.flags.is_cancelled(other.id) is False

    async def test_finish_cancelled_ends_the_job_and_clears_the_flag(self, env: _Env) -> None:
        job = await env.running()
        await request_cancel(env.jobs, env.flags, _TENANT, job.id)

        done = await finish_cancelled(env.jobs, env.flags, job.id, "run-1")

        assert done.status is JobStatus.CANCELLED and done.finished_at is not None
        assert await env.flags.is_cancelled(job.id) is False
