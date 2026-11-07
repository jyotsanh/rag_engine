"""Content-hash claims in real MongoDB: atomic first-page-wins, per job, released at finalize."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import ContentDeduper, CrawlScope, DedupeOutcome, content_hash
from rag_engine.domain import CLAIM_TTL_SECONDS, ContentHashClaim, new_id
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import ContentClaimRepository

_JOB_A = "job_" + "a" * 32
_JOB_B = "job_" + "b" * 32
_HASH = content_hash("# Widgets\n\nSame content under many URLs.\n")


@pytest.fixture
async def repo(integration_settings: Settings) -> AsyncIterator[ContentClaimRepository]:
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([ContentHashClaim])
    yield ContentClaimRepository(mongo.db)
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestContentClaims_S46:
    """S46 — Content dedupe in MongoDB: unique (job, hash) claims, race-safe, releasable."""

    async def test_first_claim_wins(self, repo: ContentClaimRepository) -> None:
        first, second = new_id("page"), new_id("page")

        owner = await repo.claim(_JOB_A, _HASH, first, "https://acme.example/a")
        again = await repo.claim(_JOB_A, _HASH, second, "https://acme.example/b")

        assert owner.page_id == first
        assert (again.page_id, again.url) == (first, "https://acme.example/a")

    async def test_concurrent_claims_have_one_winner(self, repo: ContentClaimRepository) -> None:
        pages = [new_id("page") for _ in range(25)]

        owners = await asyncio.gather(
            *(
                repo.claim(_JOB_A, _HASH, p, f"https://acme.example/p{i}")
                for i, p in enumerate(pages)
            )
        )

        assert len({o.page_id for o in owners}) == 1
        assert owners[0].page_id in pages
        assert await repo.count({"job_id": _JOB_A, "content_hash": _HASH}) == 1

    async def test_reclaim_by_the_owner_is_idempotent(self, repo: ContentClaimRepository) -> None:
        page = new_id("page")

        first = await repo.claim(_JOB_A, _HASH, page, "https://acme.example/a")
        again = await repo.claim(_JOB_A, _HASH, page, "https://acme.example/a")

        assert first.id == again.id

    async def test_jobs_are_isolated_and_released_separately(
        self, repo: ContentClaimRepository
    ) -> None:
        a = await repo.claim(_JOB_A, _HASH, new_id("page"), "https://acme.example/a")
        b = await repo.claim(_JOB_B, _HASH, new_id("page"), "https://acme.example/a")

        released = await repo.release_job(_JOB_A)

        assert a.page_id != b.page_id
        assert released == 1
        assert await repo.count({"job_id": _JOB_A}) == 0
        assert await repo.count({"job_id": _JOB_B}) == 1

    async def test_indexes(self, repo: ContentClaimRepository) -> None:
        indexes = {i["name"]: i async for i in await repo.collection.list_indexes()}

        assert indexes["job_content_unique"]["unique"] is True
        assert indexes["claim_ttl"]["expireAfterSeconds"] == CLAIM_TTL_SECONDS

    async def test_deduper_on_mongo(self, repo: ContentClaimRepository) -> None:
        scope = CrawlScope.from_config("https://acme.example/", max_depth=5)
        deduper = ContentDeduper(_JOB_A, scope, repo)
        markdown = "# Widgets\n\nSame content under many URLs.\n"

        decisions = await asyncio.gather(
            *(
                deduper.check(
                    page_id=new_id("page"),
                    url=f"https://acme.example/{n}",
                    markdown=markdown,
                    depth=1,
                )
                for n in range(10)
            )
        )

        outcomes = [d.outcome for d in decisions]
        assert outcomes.count(DedupeOutcome.UNIQUE) == 1
        assert outcomes.count(DedupeOutcome.DUPLICATE) == 9
