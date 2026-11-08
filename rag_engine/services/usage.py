"""Embedding usage metering (Step 65): every embedding call is booked where it belongs.

- **Ingest** (`record_ingest`, Step 76): a crawl job's batch goes on the job
  (`CrawlJob.usage`, shown in the job detail, Step 85) *and* on the tenant's monthly
  rollup. The job write is guarded by the run (`JobRunSupersededError` for a stale run),
  and the tenant row is only bumped after it succeeded, so a superseded run never bills.
- **Queries** (`record_query`, Step 89): searches have no job; they go on the tenant's
  rollup only.

Batches with no provider requests and no cache hits (empty batches) are not written.
`tokens` are the provider-billed input tokens; `cached` counts inputs the embedding cache
served (Step 64), i.e. what re-crawls saved.
"""

from rag_engine.domain import CrawlJob, UsagePurpose
from rag_engine.embeddings import EmbeddingBatch
from rag_engine.repositories import JobRepository
from rag_engine.repositories.usage import UsageRepository


class UsageMeter:
    def __init__(self, jobs: JobRepository, usage: UsageRepository) -> None:
        self._jobs = jobs
        self._usage = usage

    async def record_ingest(self, job: CrawlJob, run_id: str, batch: EmbeddingBatch) -> None:
        usage = batch.usage
        if not (usage.requests or usage.tokens or batch.cached):
            return
        await self._jobs.add_usage(
            job.id, run_id, requests=usage.requests, tokens=usage.tokens, cached=batch.cached
        )
        await self._usage.record(
            job.tenant_id,
            UsagePurpose.INGEST,
            batch.space.model,
            requests=usage.requests,
            tokens=usage.tokens,
            cached=batch.cached,
        )

    async def record_query(self, tenant_id: str, batch: EmbeddingBatch) -> None:
        usage = batch.usage
        if not (usage.requests or usage.tokens or batch.cached):
            return
        await self._usage.record(
            tenant_id,
            UsagePurpose.QUERY,
            batch.space.model,
            requests=usage.requests,
            tokens=usage.tokens,
            cached=batch.cached,
        )
