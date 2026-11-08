"""Usage metering in real MongoDB: per-job usage and per-tenant monthly rollups."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from rag_engine.config import EmbeddingModelName, Settings
from rag_engine.domain import CrawlJob, EmbeddingUsage, TenantUsage, UsagePurpose, usage_period
from rag_engine.embeddings import (
    CachedEmbedder,
    EmbeddingBatch,
    EmbeddingUsageCounts,
    resolve_space,
)
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import JobRepository, JobRunSupersededError, UsageRepository
from rag_engine.services import UsageMeter
from tests.support.fakes import FakeEmbedder, InMemoryEmbeddingCache

_TENANT = "ten_" + "1" * 32
_OTHER = "ten_" + "2" * 32
_SMALL = resolve_space(EmbeddingModelName.TEXT_EMBEDDING_3_SMALL, 256)
_LARGE = resolve_space(EmbeddingModelName.TEXT_EMBEDDING_3_LARGE, 256)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 8, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@dataclass
class _Env:
    jobs: JobRepository
    usage: UsageRepository
    meter: UsageMeter
    clock: _Clock

    async def running_job(self, tenant: str = _TENANT) -> CrawlJob:
        job = await self.jobs.insert(
            CrawlJob(tenant_id=tenant, kb_id="kb_" + "1" * 32, source_id="src_" + "1" * 32,
                     seed_url="https://acme.example/")
        )  # fmt: skip
        return await self.jobs.start(job.id, "run-1")


def _batch(requests: int, tokens: int, cached: int = 0, space=_SMALL) -> EmbeddingBatch:
    return EmbeddingBatch([], space, EmbeddingUsageCounts(requests, tokens), cached=cached)


@pytest.fixture
async def env(integration_settings: Settings) -> AsyncIterator[_Env]:
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([CrawlJob, TenantUsage])
    clock = _Clock()
    jobs, usage = JobRepository(mongo.db), UsageRepository(mongo.db, clock=clock)
    yield _Env(jobs, usage, UsageMeter(jobs, usage), clock)
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestUsageMetering_S65:
    """S65 — Usage metering: per job (job detail) and per tenant per month/purpose/model."""

    async def test_ingest_usage_goes_on_the_job_and_the_tenant(self, env: _Env) -> None:
        job = await env.running_job()

        await env.meter.record_ingest(job, "run-1", _batch(2, 900, cached=5))
        await env.meter.record_ingest(job, "run-1", _batch(1, 100))

        stored = await env.jobs.get_or_raise(job.id)
        assert stored.usage == EmbeddingUsage(requests=3, tokens=1000, cached=5)
        (row,) = await env.usage.for_tenant(_TENANT)
        assert (row.period, row.purpose) == ("2026-11", UsagePurpose.INGEST)
        assert row.model is EmbeddingModelName.TEXT_EMBEDDING_3_SMALL
        assert (row.requests, row.tokens, row.cached) == (3, 1000, 5)

    async def test_a_superseded_run_bills_nothing(self, env: _Env) -> None:
        job = await env.running_job()
        await env.jobs.requeue(job.id, stale_run_id="run-1")

        with pytest.raises(JobRunSupersededError):
            await env.meter.record_ingest(job, "run-1", _batch(1, 50))

        assert await env.usage.for_tenant(_TENANT) == []

    async def test_queries_go_on_the_tenant_only(self, env: _Env) -> None:
        await env.meter.record_query(_TENANT, _batch(1, 12))
        await env.meter.record_query(_TENANT, _batch(1, 8))

        (row,) = await env.usage.for_tenant(_TENANT)
        assert (row.purpose, row.requests, row.tokens) == (UsagePurpose.QUERY, 2, 20)

    async def test_empty_batches_are_not_written(self, env: _Env) -> None:
        job = await env.running_job()

        await env.meter.record_ingest(job, "run-1", _batch(0, 0))
        await env.meter.record_query(_TENANT, _batch(0, 0))

        assert await env.usage.for_tenant(_TENANT) == []
        assert (await env.jobs.get_or_raise(job.id)).usage == EmbeddingUsage()

    async def test_rows_split_by_month_purpose_model_and_tenant(self, env: _Env) -> None:
        env.clock.now = datetime(2026, 10, 31, 23, 59, tzinfo=UTC)
        await env.usage.record(_TENANT, UsagePurpose.INGEST, _SMALL.model, requests=1, tokens=10)
        env.clock.now = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
        await env.usage.record(_TENANT, UsagePurpose.INGEST, _SMALL.model, requests=2, tokens=20)
        await env.usage.record(_TENANT, UsagePurpose.QUERY, _SMALL.model, requests=3, tokens=30)
        await env.usage.record(_TENANT, UsagePurpose.INGEST, _LARGE.model, requests=4, tokens=40,
                               cached=7)  # fmt: skip
        await env.usage.record(_OTHER, UsagePurpose.INGEST, _SMALL.model, requests=9, tokens=90)

        rows = await env.usage.for_tenant(_TENANT)
        november = await env.usage.totals(_TENANT, "2026-11")
        ingest = await env.usage.totals(_TENANT, "2026-11", purpose=UsagePurpose.INGEST)
        october = await env.usage.totals(_TENANT, "2026-10")

        assert [(r.period, r.purpose.value, r.model.value) for r in rows] == [
            ("2026-11", "ingest", "text-embedding-3-large"),
            ("2026-11", "ingest", "text-embedding-3-small"),
            ("2026-11", "query", "text-embedding-3-small"),
            ("2026-10", "ingest", "text-embedding-3-small"),
        ]
        assert (november.requests, november.tokens, november.cached) == (9, 90, 7)
        assert (ingest.requests, ingest.tokens) == (6, 60)
        assert (october.requests, october.tokens) == (1, 10)
        assert len(await env.usage.for_tenant(_OTHER)) == 1

    async def test_concurrent_records_land_on_one_row(self, env: _Env) -> None:
        await asyncio.gather(
            *(
                env.usage.record(_TENANT, UsagePurpose.INGEST, _SMALL.model, requests=1,
                                 tokens=10, cached=2)
                for _ in range(20)
            )
        )  # fmt: skip

        (row,) = await env.usage.for_tenant(_TENANT)
        assert (row.requests, row.tokens, row.cached) == (20, 200, 40)

    async def test_increments_are_validated(self, env: _Env) -> None:
        job = await env.running_job()

        with pytest.raises(ValueError, match=">= 0"):
            await env.usage.record(_TENANT, UsagePurpose.QUERY, _SMALL.model, requests=-1,
                                   tokens=0)  # fmt: skip
        with pytest.raises(ValueError, match=">= 0"):
            await env.jobs.add_usage(job.id, "run-1", requests=0, tokens=0, cached=-1)

    async def test_a_recrawl_through_the_cache_books_the_savings(self, env: _Env) -> None:
        job = await env.running_job()
        embedder = CachedEmbedder(FakeEmbedder(_SMALL), InMemoryEmbeddingCache())
        texts = [f"Guide\n\nSection {i}." for i in range(4)]

        await env.meter.record_ingest(job, "run-1", await embedder.embed_documents(texts))
        await env.meter.record_ingest(job, "run-1", await embedder.embed_documents(texts))

        usage = (await env.jobs.get_or_raise(job.id)).usage
        assert (usage.requests, usage.cached) == (1, 4)  # the second pass cost nothing
        assert usage.tokens == sum(len(t.split()) for t in texts)

    def test_usage_period_is_the_utc_month(self) -> None:
        assert usage_period(datetime(2026, 1, 5, tzinfo=UTC)) == "2026-01"
        assert usage_period(datetime(2026, 12, 31, 23, 59, tzinfo=UTC)) == "2026-12"
