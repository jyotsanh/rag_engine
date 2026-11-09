"""Concurrency guards in real MongoDB: one active job per source, tenant concurrent-job quota,
duplicate submissions returning the existing job."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.domain import (
    ACTIVE_JOB_FILTER,
    CrawlConfig,
    CrawlJob,
    JobError,
    JobStatus,
    Source,
    Tenant,
    TenantQuotas,
    new_id,
    utc_now,
)
from rag_engine.exceptions import QuotaExceededError
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import (
    DuplicateDocumentError,
    JobRepository,
    JobSlotsExhaustedError,
    TenantRepository,
)
from rag_engine.repositories.sources import SourceRepository
from rag_engine.services import CrawlJobService, JobSubmission

_KB = "kb_" + "1" * 32
_ERROR = JobError(code="engine_crashed", message="boom")


@pytest.fixture
async def mongo(integration_settings: Settings) -> AsyncIterator[MongoConnection]:
    connection = MongoConnection(integration_settings)
    await connection.connect()
    await connection.ensure_indexes([Tenant, Source, CrawlJob])
    yield connection
    await connection.close()


@pytest.fixture
def jobs(mongo: MongoConnection) -> JobRepository:
    return JobRepository(mongo.db)


@pytest.fixture
def sources(mongo: MongoConnection) -> SourceRepository:
    return SourceRepository(mongo.db)


@pytest.fixture
def service(jobs: JobRepository, sources: SourceRepository) -> CrawlJobService:
    return CrawlJobService(jobs, sources)


async def _tenant(mongo: MongoConnection, *, max_jobs: int = 2, name: str = "Acme") -> Tenant:
    quotas = TenantQuotas(max_concurrent_jobs=max_jobs)
    return await TenantRepository(mongo.db).create(Tenant(name=name, quotas=quotas))


async def _source(sources: SourceRepository, tenant: Tenant, n: int = 0) -> Source:
    source, _ = await sources.get_or_create(tenant.id, _KB, f"https://site{n}.example/")
    return source


def _job(source_id: str, tenant_id: str = "ten_" + "1" * 32) -> CrawlJob:
    return CrawlJob(
        tenant_id=tenant_id, kb_id=_KB, source_id=source_id, seed_url="https://acme.example/"
    )


async def _finish(jobs: JobRepository, job: CrawlJob, status: JobStatus) -> CrawlJob:
    """Drive `job` to a terminal `status` through the state machine."""
    if status is JobStatus.CANCELLED:
        return await jobs.request_cancel(job.tenant_id, job.id)
    if status is JobStatus.FAILED:
        return await jobs.fail(job.id, _ERROR)
    run_id = new_id("run")
    await jobs.start(job.id, run_id)
    await jobs.begin_finalizing(job.id, run_id)
    return await jobs.succeed(job.id, run_id)


# ---- #
# Tests
# ---- #


class TestActiveJobIndex_S79:
    """S79 — the `one_active_job_per_source` partial unique index and the repository's
    `create_active` / `find_active_for_source` / `count_active_for_tenant`."""

    async def test_index_is_unique_and_partial_on_active_statuses(
        self, mongo: MongoConnection
    ) -> None:
        info = await mongo.db[CrawlJob.collection_name].index_information()

        index = info["one_active_job_per_source"]
        assert index["key"] == [("source_id", 1)]
        assert index["unique"] is True
        assert index["partialFilterExpression"] == {
            "status": {"$in": ["finalizing", "queued", "running"]}
        }
        assert index["partialFilterExpression"] == ACTIVE_JOB_FILTER
        assert info["tenant_active_jobs"]["partialFilterExpression"] == ACTIVE_JOB_FILTER
        slot = info["tenant_job_slot"]
        assert slot["key"] == [("tenant_id", 1), ("slot", 1)] and slot["unique"] is True
        assert slot["partialFilterExpression"] == {**ACTIVE_JOB_FILTER, "slot": {"$type": "int"}}

    async def test_database_refuses_a_second_active_job_for_a_source(
        self, jobs: JobRepository
    ) -> None:
        source_id = new_id("src")
        await jobs.insert(_job(source_id))

        with pytest.raises(DuplicateDocumentError) as exc:
            await jobs.insert(_job(source_id))

        assert exc.value.details == {"collection": "crawl_jobs", "fields": ["source_id"]}
        assert await jobs.insert(_job(new_id("src")))  # another source is unaffected

    @pytest.mark.parametrize("status", [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED])
    async def test_finished_jobs_free_the_source(
        self, jobs: JobRepository, status: JobStatus
    ) -> None:
        source_id = new_id("src")
        first = await _finish(jobs, await jobs.insert(_job(source_id)), status)
        second = await _finish(jobs, await jobs.insert(_job(source_id)), status)

        third = await jobs.insert(_job(source_id))

        assert first.status is second.status is status
        assert third.status is JobStatus.QUEUED
        assert await jobs.count({"source_id": source_id}) == 3

    async def test_running_finalizing_and_requeued_jobs_keep_the_slot(
        self, jobs: JobRepository
    ) -> None:
        source_id = new_id("src")
        job = await jobs.insert(_job(source_id))

        await jobs.start(job.id, "run-1")
        with pytest.raises(DuplicateDocumentError):
            await jobs.insert(_job(source_id))
        await jobs.begin_finalizing(job.id, "run-1")
        with pytest.raises(DuplicateDocumentError):
            await jobs.insert(_job(source_id))
        requeued = await jobs.requeue(job.id, stale_run_id="run-1")

        assert requeued.status is JobStatus.QUEUED
        with pytest.raises(DuplicateDocumentError):
            await jobs.insert(_job(source_id))

    async def test_create_active_returns_the_existing_active_job(self, jobs: JobRepository) -> None:
        source_id = new_id("src")
        first, first_created = await jobs.create_active(_job(source_id))
        await jobs.start(first.id, "run-1")

        again, again_created = await jobs.create_active(_job(source_id))

        assert first_created is True and again_created is False
        assert again.id == first.id and again.status is JobStatus.RUNNING
        assert await jobs.count({"source_id": source_id}) == 1

    async def test_racing_creates_insert_exactly_one_job(self, jobs: JobRepository) -> None:
        source_id = new_id("src")

        results = await asyncio.gather(*(jobs.create_active(_job(source_id)) for _ in range(10)))

        assert [created for _, created in results].count(True) == 1
        assert len({job.id for job, _ in results}) == 1
        assert await jobs.count({"source_id": source_id}) == 1

    async def test_create_active_retries_when_the_active_job_finishes_in_between(
        self, jobs: JobRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the blocking job is cancelled right as the loser looks it up
        source_id = new_id("src")
        blocker = await jobs.insert(_job(source_id))
        lookup = jobs.find_active_for_source
        calls = 0

        async def finishing_lookup(tenant_id: str, source: str) -> CrawlJob | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                await jobs.request_cancel(blocker.tenant_id, blocker.id)
            return await lookup(tenant_id, source)

        monkeypatch.setattr(jobs, "find_active_for_source", finishing_lookup)

        # Act
        job, created = await jobs.create_active(_job(source_id))

        # Assert
        assert created is True and job.id != blocker.id
        assert (await jobs.get_or_raise(job.id)).status is JobStatus.QUEUED
        assert calls == 1

    async def test_create_active_only_accepts_queued_jobs(self, jobs: JobRepository) -> None:
        job = _job(new_id("src")).model_copy(
            update={"status": JobStatus.CANCELLED, "finished_at": utc_now()}
        )

        with pytest.raises(ValueError, match="queued"):
            await jobs.create_active(job)

    async def test_active_reads_are_tenant_scoped(self, jobs: JobRepository) -> None:
        mine, theirs = "ten_" + "1" * 32, "ten_" + "2" * 32
        queued = await jobs.insert(_job(new_id("src"), mine))
        running = await jobs.start((await jobs.insert(_job(new_id("src"), mine))).id, "run-1")
        finalizing = await jobs.insert(_job(new_id("src"), mine))
        await jobs.start(finalizing.id, "run-2")
        await jobs.begin_finalizing(finalizing.id, "run-2")
        for status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
            await _finish(jobs, await jobs.insert(_job(new_id("src"), mine)), status)
        other = await jobs.insert(_job(new_id("src"), theirs))

        assert await jobs.count_active_for_tenant(mine) == 3
        assert await jobs.count_active_for_tenant(theirs) == 1
        found_queued = await jobs.find_active_for_source(mine, queued.source_id)
        found_running = await jobs.find_active_for_source(mine, running.source_id)
        assert found_queued is not None and found_queued.id == queued.id
        assert found_running is not None and found_running.id == running.id
        assert await jobs.find_active_for_source(mine, other.source_id) is None

    # ---- concurrency slots ---- #

    async def test_database_refuses_two_active_jobs_in_one_slot(self, jobs: JobRepository) -> None:
        first = _job(new_id("src"))
        first.slot = 0
        await jobs.insert(first)
        clash, other_tenant, unslotted = (
            _job(new_id("src")),
            _job(new_id("src"), "ten_" + "2" * 32),
            _job(new_id("src")),
        )
        clash.slot = other_tenant.slot = 0

        with pytest.raises(DuplicateDocumentError) as exc:
            await jobs.insert(clash)

        assert exc.value.details == {"collection": "crawl_jobs", "fields": ["slot", "tenant_id"]}
        assert await jobs.insert(other_tenant)  # slots are per tenant
        assert await jobs.insert(unslotted)  # a job without a slot is outside the index
        await _finish(jobs, first, JobStatus.SUCCEEDED)
        assert await jobs.insert(clash)  # a finished job frees its slot

    async def test_create_active_claims_distinct_slots_up_to_the_limit(
        self, jobs: JobRepository
    ) -> None:
        created = [await jobs.create_active(_job(new_id("src")), slots=3) for _ in range(3)]

        with pytest.raises(JobSlotsExhaustedError):
            await jobs.create_active(_job(new_id("src")), slots=3)

        assert {job.slot for job, _ in created} == {0, 1, 2}
        assert all(was_created for _, was_created in created)

    async def test_exhausted_slots_still_return_the_sources_active_job(
        self, jobs: JobRepository
    ) -> None:
        source_id = new_id("src")
        first, _ = await jobs.create_active(_job(source_id), slots=1)

        again, created = await jobs.create_active(_job(source_id), slots=1)

        assert created is False and again.id == first.id

    async def test_a_lowered_limit_counts_jobs_in_higher_slots(self, jobs: JobRepository) -> None:
        # Arrange: three jobs fill slots 0-2 under a limit of 3; then 0 and 1 finish
        held = [(await jobs.create_active(_job(new_id("src")), slots=3))[0] for _ in range(3)]
        for job in held:
            if job.slot != 2:
                await _finish(jobs, job, JobStatus.FAILED)

        # Act + Assert: limit lowered to 1, slot 0 is free but one job is already active
        with pytest.raises(JobSlotsExhaustedError):
            await jobs.create_active(_job(new_id("src")), slots=1)
        job, created = await jobs.create_active(_job(new_id("src")), slots=2)
        assert created is True and job.slot in {0, 1}

    async def test_racing_slot_claims_fill_exactly_the_free_slots(
        self, jobs: JobRepository
    ) -> None:
        results = await asyncio.gather(
            *(jobs.create_active(_job(new_id("src")), slots=4) for _ in range(12)),
            return_exceptions=True,
        )

        won = [r for r in results if isinstance(r, tuple)]
        lost = [r for r in results if isinstance(r, JobSlotsExhaustedError)]
        assert (len(won), len(lost)) == (4, 8)
        assert {job.slot for job, _ in won} == {0, 1, 2, 3}


class TestCrawlJobSubmission_S79:
    """S79 — `CrawlJobService.submit`: duplicate submissions return the active job, the
    tenant's `max_concurrent_jobs` quota, and both under concurrency."""

    async def test_submit_queues_a_job_for_the_source(
        self,
        service: CrawlJobService,
        mongo: MongoConnection,
        sources: SourceRepository,
    ) -> None:
        tenant = await _tenant(mongo)
        source = await _source(sources, tenant)
        config = CrawlConfig(max_pages=50, max_depth=2, path_prefix="/docs/")

        submission = await service.submit(tenant, source, config)

        job = submission.job
        assert submission.created is True
        assert job.status is JobStatus.QUEUED
        assert (job.tenant_id, job.kb_id, job.source_id) == (tenant.id, _KB, source.id)
        assert job.seed_url == source.seed_url and job.config == config
        assert (await sources.get_or_raise(source.id)).last_job_id == job.id

    async def test_duplicate_submission_returns_the_existing_job(
        self,
        service: CrawlJobService,
        mongo: MongoConnection,
        sources: SourceRepository,
        jobs: JobRepository,
    ) -> None:
        tenant = await _tenant(mongo)
        source = await _source(sources, tenant)
        first = await service.submit(tenant, source, CrawlConfig(max_pages=10))

        queued_again = await service.submit(tenant, source, CrawlConfig(max_pages=99))
        await jobs.start(first.job.id, "run-1")
        running_again = await service.submit(tenant, source, CrawlConfig())

        assert queued_again == JobSubmission(job=first.job, created=False)
        assert queued_again.job.config.max_pages == 10  # the existing job keeps its config
        assert running_again.created is False
        assert running_again.job.id == first.job.id
        assert running_again.job.status is JobStatus.RUNNING
        assert await jobs.count({"source_id": source.id}) == 1

    async def test_duplicate_submission_at_the_quota_returns_the_existing_job(
        self, service: CrawlJobService, mongo: MongoConnection, sources: SourceRepository
    ) -> None:
        tenant = await _tenant(mongo, max_jobs=1)
        source = await _source(sources, tenant)
        first = await service.submit(tenant, source, CrawlConfig())

        again = await service.submit(tenant, source, CrawlConfig())

        assert again.created is False and again.job.id == first.job.id

    async def test_resubmission_after_the_job_finished_starts_a_new_job(
        self,
        service: CrawlJobService,
        mongo: MongoConnection,
        sources: SourceRepository,
        jobs: JobRepository,
    ) -> None:
        tenant = await _tenant(mongo)
        source = await _source(sources, tenant)
        first = await service.submit(tenant, source, CrawlConfig())
        await _finish(jobs, first.job, JobStatus.SUCCEEDED)

        second = await service.submit(tenant, source, CrawlConfig())

        assert second.created is True and second.job.id != first.job.id
        assert (await sources.get_or_raise(source.id)).last_job_id == second.job.id

    async def test_tenant_concurrent_job_quota(
        self,
        service: CrawlJobService,
        mongo: MongoConnection,
        sources: SourceRepository,
        jobs: JobRepository,
    ) -> None:
        tenant = await _tenant(mongo, max_jobs=2)
        a, b, c = [await _source(sources, tenant, n) for n in range(3)]
        first = await service.submit(tenant, a, CrawlConfig())
        await jobs.start(first.job.id, "run-1")  # running and queued jobs both count
        await service.submit(tenant, b, CrawlConfig())

        with pytest.raises(QuotaExceededError) as exc:
            await service.submit(tenant, c, CrawlConfig())

        assert exc.value.status_code == 403 and exc.value.code == "quota_exceeded"
        assert exc.value.details == {"quota": "max_concurrent_jobs", "limit": 2, "active": 2}
        assert await jobs.count({"source_id": c.id}) == 0
        assert (await sources.get_or_raise(c.id)).last_job_id is None

        # a finished job frees a slot
        await jobs.cancel(first.job.id, "run-1")
        third = await service.submit(tenant, c, CrawlConfig())
        assert third.created is True

    async def test_quota_is_per_tenant(
        self, service: CrawlJobService, mongo: MongoConnection, sources: SourceRepository
    ) -> None:
        acme = await _tenant(mongo, max_jobs=1, name="Acme")
        globex = await _tenant(mongo, max_jobs=1, name="Globex")
        await service.submit(acme, await _source(sources, acme, 1), CrawlConfig())

        other = await service.submit(globex, await _source(sources, globex, 2), CrawlConfig())

        assert other.created is True

    async def test_racing_submissions_never_exceed_the_quota(
        self,
        service: CrawlJobService,
        mongo: MongoConnection,
        sources: SourceRepository,
        jobs: JobRepository,
    ) -> None:
        tenant = await _tenant(mongo, max_jobs=3)
        targets = [await _source(sources, tenant, n) for n in range(12)]

        results = await asyncio.gather(
            *(service.submit(tenant, source, CrawlConfig()) for source in targets),
            return_exceptions=True,
        )

        accepted = [r for r in results if isinstance(r, JobSubmission)]
        refused = [r for r in results if isinstance(r, QuotaExceededError)]
        assert (len(accepted), len(refused)) == (3, 9)  # the free slots, no more, no fewer
        assert all(r.created for r in accepted)
        assert {r.job.slot for r in accepted} == {0, 1, 2}
        assert await jobs.count_active_for_tenant(tenant.id) == 3
        assert await jobs.count({"tenant_id": tenant.id}) == 3  # refused ones left nothing

    async def test_racing_duplicate_submissions_share_one_job(
        self,
        service: CrawlJobService,
        mongo: MongoConnection,
        sources: SourceRepository,
        jobs: JobRepository,
    ) -> None:
        tenant = await _tenant(mongo, max_jobs=5)
        source = await _source(sources, tenant)

        results = await asyncio.gather(
            *(service.submit(tenant, source, CrawlConfig()) for _ in range(10))
        )

        assert [r.created for r in results].count(True) == 1
        assert len({r.job.id for r in results}) == 1
        assert await jobs.count({"source_id": source.id}) == 1
        assert await jobs.count_active_for_tenant(tenant.id) == 1

    async def test_a_foreign_source_is_refused(
        self, service: CrawlJobService, mongo: MongoConnection, sources: SourceRepository
    ) -> None:
        acme = await _tenant(mongo, name="Acme")
        globex = await _tenant(mongo, name="Globex")
        theirs = await _source(sources, globex)

        with pytest.raises(ValueError, match="another tenant"):
            await service.submit(acme, theirs, CrawlConfig())
