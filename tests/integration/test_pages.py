"""Page repository in real MongoDB: discovery, outcomes, counts, ordered reads, scoping."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rag_engine.config import Settings
from rag_engine.domain import (
    ContentKind,
    DeferredLink,
    Page,
    PageReason,
    PageStatus,
    RenderedWith,
)
from rag_engine.exceptions import AppValidationError
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import PageNotFoundError, PageOutcome, PageRepository
from tests.conftest import OverrideSettings

_TENANT = "ten_" + "1" * 32
_OTHER_TENANT = "ten_" + "2" * 32
_KB = "kb_" + "1" * 32
_SOURCE = "src_" + "1" * 32
_JOB = "job_" + "1" * 32
_OTHER_JOB = "job_" + "2" * 32


def _page(n: int, *, job_id: str = _JOB, tenant_id: str = _TENANT, depth: int = 1) -> Page:
    return Page(
        tenant_id=tenant_id,
        kb_id=_KB,
        source_id=_SOURCE,
        job_id=job_id,
        url=f"https://acme.example/p{n}",
        url_hash=f"{n:064x}",
        depth=depth,
    )


def _success(markdown: str = "# Title\n\nBody text.\n") -> PageOutcome:
    return PageOutcome(
        status=PageStatus.SUCCEEDED,
        http_status=200,
        final_url="https://acme.example/final",
        content_type="text/html",
        content_kind=ContentKind.HTML,
        rendered_with=RenderedWith.BROWSER,
        js_signals=("empty_mount",),
        extractor="html",
        title="Title",
        lang="en",
        doc_metadata={"author": "Dana", "pages": 3},
        markdown=markdown,
        content_hash="a" * 64,
        bytes_downloaded=1234,
        fetch_ms=80,
        extract_ms=12,
        fetched_at=datetime(2026, 11, 7, tzinfo=UTC),
    )


@pytest.fixture
async def repo(
    integration_settings: Settings, override_settings: OverrideSettings
) -> AsyncIterator[PageRepository]:
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Page])
    yield PageRepository(mongo.db, override_settings(PAGE_MAX_MARKDOWN_BYTES=2000))
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestPageRepository_S47:
    """S47 — Page repository: `(job_id, url_hash)` unique, outcomes, counts, reads."""

    # ---- discovery ---- #

    async def test_add_discovered_is_idempotent(self, repo: PageRepository) -> None:
        page, created = await repo.add_discovered(_page(1))
        again, created_again = await repo.add_discovered(_page(1))

        assert (created, created_again) == (True, False)
        assert again.id == page.id
        assert await repo.count({"job_id": _JOB}) == 1

    async def test_same_url_in_another_job_is_a_new_page(self, repo: PageRepository) -> None:
        await repo.add_discovered(_page(1))
        _, created = await repo.add_discovered(_page(1, job_id=_OTHER_JOB))

        assert created

    async def test_bulk_discovery_skips_known_urls(self, repo: PageRepository) -> None:
        await repo.add_discovered(_page(1))

        inserted = await repo.add_many_discovered(
            [_page(1), _page(2), _page(3), _page(2)]  # one known, one repeated in the batch
        )

        assert inserted == 2
        assert await repo.count({"job_id": _JOB}) == 3
        assert await repo.add_many_discovered([]) == 0

    async def test_concurrent_discovery_of_one_url(self, repo: PageRepository) -> None:
        results = await asyncio.gather(*(repo.add_discovered(_page(7)) for _ in range(20)))

        assert sum(created for _, created in results) == 1
        assert len({page.id for page, _ in results}) == 1

    # ---- outcomes ---- #

    async def test_record_success(self, repo: PageRepository) -> None:
        page, _ = await repo.add_discovered(_page(1))

        stored = await repo.record_outcome(page.id, _success())

        reloaded = await repo.get_or_raise(page.id)
        assert stored == reloaded
        assert reloaded.status is PageStatus.SUCCEEDED
        assert (reloaded.rendered_with, reloaded.js_signals) == (
            RenderedWith.BROWSER,
            ["empty_mount"],
        )
        assert reloaded.doc_metadata == {"author": "Dana", "pages": 3}
        assert (reloaded.fetch_ms, reloaded.extract_ms, reloaded.http_status) == (80, 12, 200)
        assert reloaded.fetched_at == datetime(2026, 11, 7, tzinfo=UTC)

    async def test_retry_success_clears_the_earlier_failure(self, repo: PageRepository) -> None:
        page, _ = await repo.add_discovered(_page(1))
        await repo.record_outcome(
            page.id,
            PageOutcome(
                status=PageStatus.FAILED,
                reason=PageReason.HTTP_SERVER_ERROR,
                reason_detail="http_503",
                http_status=503,
            ),
        )

        stored = await repo.record_outcome(page.id, _success())

        assert (stored.reason, stored.reason_detail) == (None, None)
        assert stored.http_status == 200

    async def test_skip_clears_earlier_content(self, repo: PageRepository) -> None:
        page, _ = await repo.add_discovered(_page(1))
        await repo.record_outcome(page.id, _success())

        stored = await repo.record_outcome(
            page.id,
            PageOutcome(status=PageStatus.SKIPPED, reason=PageReason.CANONICAL_ELSEWHERE),
        )

        assert (stored.markdown, stored.content_kind, stored.title) == (None, None, None)

    async def test_duplicate_outcome(self, repo: PageRepository) -> None:
        original, _ = await repo.add_discovered(_page(1))
        duplicate, _ = await repo.add_discovered(_page(2))

        stored = await repo.record_outcome(
            duplicate.id,
            PageOutcome(
                status=PageStatus.DUPLICATE, duplicate_of=original.id, content_hash="b" * 64
            ),
        )

        assert (stored.status, stored.duplicate_of) == (PageStatus.DUPLICATE, original.id)

    @pytest.mark.parametrize(
        "outcome",
        [
            PageOutcome(status=PageStatus.SUCCEEDED),  # no markdown / content kind
            PageOutcome(status=PageStatus.FAILED),  # no reason
            PageOutcome(status=PageStatus.DUPLICATE),  # no duplicate_of
            PageOutcome(
                status=PageStatus.SUCCEEDED,
                reason=PageReason.TIMEOUT,
                content_kind=ContentKind.HTML,
                markdown="x",
            ),  # reason on success
        ],
        ids=[
            "success-without-content",
            "failure-without-reason",
            "dup-without-original",
            "reason-on-success",
        ],
    )
    async def test_inconsistent_outcomes_are_rejected(
        self, repo: PageRepository, outcome: PageOutcome
    ) -> None:
        page, _ = await repo.add_discovered(_page(1))

        with pytest.raises(AppValidationError):
            await repo.record_outcome(page.id, outcome)

        assert (await repo.get_or_raise(page.id)).status is PageStatus.QUEUED  # untouched

    async def test_oversized_markdown_is_stored_truncated(self, repo: PageRepository) -> None:
        page, _ = await repo.add_discovered(_page(1))
        markdown = "\n\n".join(f"Paragraph {n}: " + "x" * 90 for n in range(100))  # ~10 KB

        stored = await repo.record_outcome(page.id, _success(markdown))

        assert stored.markdown_truncated is True
        assert stored.markdown is not None and len(stored.markdown.encode()) <= 2000

    # ---- reads ---- #

    async def test_counts_by_status_and_reason(self, repo: PageRepository) -> None:
        pages = [(await repo.add_discovered(_page(n)))[0] for n in range(6)]
        outcomes = [
            _success(),
            PageOutcome(status=PageStatus.FAILED, reason=PageReason.TIMEOUT),
            PageOutcome(status=PageStatus.FAILED, reason=PageReason.TIMEOUT),
            PageOutcome(status=PageStatus.SKIPPED, reason=PageReason.ROBOTS_DISALLOWED),
            PageOutcome(status=PageStatus.DUPLICATE, duplicate_of=pages[0].id),
        ]
        for page, outcome in zip(pages, outcomes, strict=False):  # the last stays queued
            await repo.record_outcome(page.id, outcome)
        await repo.add_discovered(_page(99, job_id=_OTHER_JOB))

        by_status = await repo.count_by_status(_JOB)
        by_reason = await repo.count_by_reason(_JOB)

        assert by_status == {
            PageStatus.SUCCEEDED: 1,
            PageStatus.FAILED: 2,
            PageStatus.SKIPPED: 1,
            PageStatus.DUPLICATE: 1,
            PageStatus.QUEUED: 1,
        }
        assert by_reason == {PageReason.TIMEOUT: 2, PageReason.ROBOTS_DISALLOWED: 1}

    async def test_iter_succeeded_in_discovery_order(self, repo: PageRepository) -> None:
        pages = [(await repo.add_discovered(_page(n)))[0] for n in range(7)]
        for page in reversed(pages):  # outcomes recorded in reverse order
            if pages.index(page) != 3:
                await repo.record_outcome(page.id, _success())

        seen = [page.id async for page in repo.iter_succeeded(_JOB, batch_size=2)]

        assert seen == [p.id for i, p in enumerate(pages) if i != 3]

    async def test_list_for_job_is_tenant_scoped_and_filterable(self, repo: PageRepository) -> None:
        pages = [(await repo.add_discovered(_page(n)))[0] for n in range(5)]
        await repo.record_outcome(
            pages[1].id, PageOutcome(status=PageStatus.FAILED, reason=PageReason.TIMEOUT)
        )

        first = await repo.list_for_job(_TENANT, _JOB, limit=3)
        second = await repo.list_for_job(_TENANT, _JOB, limit=3, cursor=first.next_cursor)
        failed = await repo.list_for_job(_TENANT, _JOB, status=PageStatus.FAILED)
        timeouts = await repo.list_for_job(_TENANT, _JOB, reason=PageReason.TIMEOUT)
        foreign = await repo.list_for_job(_OTHER_TENANT, _JOB)

        assert [p.id for p in first.items + second.items] == [p.id for p in pages]
        assert second.next_cursor is None
        assert [p.id for p in failed.items] == [pages[1].id] == [p.id for p in timeouts.items]
        assert foreign.items == []  # another tenant never sees this job's pages

    async def test_delete_job(self, repo: PageRepository) -> None:
        for n in range(3):
            await repo.add_discovered(_page(n))
        await repo.add_discovered(_page(0, job_id=_OTHER_JOB))

        deleted = await repo.delete_job(_JOB)

        assert deleted == 3
        assert await repo.count({"job_id": _OTHER_JOB}) == 1

    async def test_indexes(self, repo: PageRepository) -> None:
        indexes = {i["name"]: i async for i in await repo.collection.list_indexes()}

        assert indexes["job_url_unique"]["unique"] is True
        assert list(indexes["job_url_unique"]["key"]) == ["job_id", "url_hash"]
        assert list(indexes["job_status"]["key"]) == ["job_id", "status", "_id"]


class TestPageRepositoryEngineSupport_S50:
    """S50 — Page repository additions for the engine: lease-guarded outcomes, deferred links."""

    async def test_outcome_only_lands_while_the_run_holds_the_page(
        self, repo: PageRepository
    ) -> None:
        page, _ = await repo.add_discovered(_page(1))
        await repo.collection.update_one(
            {"_id": page.id}, {"$set": {"status": "in_progress", "claimed_by": "run-new"}}
        )

        with pytest.raises(PageNotFoundError):
            await repo.record_outcome(page.id, _success(), held_by="run-old")
        stored = await repo.record_outcome(page.id, _success(), held_by="run-new")

        assert stored.status is PageStatus.SUCCEEDED
        with pytest.raises(PageNotFoundError):  # done: no longer in progress
            await repo.record_outcome(page.id, _success(), held_by="run-new")

    async def test_deferred_links_round_trip_and_stream(self, repo: PageRepository) -> None:
        links = (
            DeferredLink(url="https://offsite.example/x", reason=PageReason.OUT_OF_SCOPE,
                         detail="other_host"),
            DeferredLink(url="https://acme.example/held", reason=PageReason.NOFOLLOW),
        )  # fmt: skip
        with_links, _ = await repo.add_discovered(_page(1, depth=2))
        await repo.add_discovered(_page(2))  # no deferred links: not streamed
        await repo.record_outcome(with_links.id, replace(_success(), deferred_links=links))

        streamed = [row async for row in repo.iter_deferred_links(_JOB)]

        assert streamed == [(with_links.url, 2, list(links))]
        retried = await repo.record_outcome(with_links.id, _success())  # cleared on rewrite
        assert retried.deferred_links == []

    def test_deferred_link_validation(self) -> None:
        with pytest.raises(ValidationError):
            DeferredLink(url="mailto:x@acme.example", reason=PageReason.OUT_OF_SCOPE)
        with pytest.raises(ValidationError):
            DeferredLink(url="https://a.example/", reason=PageReason.NOFOLLOW, detail="x" * 101)
