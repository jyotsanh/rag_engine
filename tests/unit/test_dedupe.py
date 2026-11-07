"""Content dedupe: per-job content-hash claims, duplicates, canonical URLs respected."""

import asyncio

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import (
    ContentDeduper,
    CrawlScope,
    DedupeOutcome,
    LinkExtractor,
    content_hash,
)
from rag_engine.crawling.fetchers.content import decode_html
from rag_engine.domain import ContentKind, Page, PageReason, PageStatus, new_id
from rag_engine.extraction import ExtractionInput, HtmlExtractor, TextCleaner
from tests.conftest import OverrideSettings
from tests.support.fakes import InMemoryContentClaimStore
from tests.support.fixture_site import EXPECTED_PAGE_OUTCOME, SITE_PAGES, SITE_ROOT

_SEED = "https://acme.example/"
_TEXT = "# Widgets\n\nThe WX-2048 sorter handles 1200 items per minute.\n"
_JOB = "job_" + "0" * 32


def _deduper(
    store: InMemoryContentClaimStore,
    job_id: str = _JOB,
    config: Settings | None = None,
    path_prefix: str | None = None,
) -> ContentDeduper:
    crawl_scope = CrawlScope.from_config(_SEED, max_depth=5, path_prefix=path_prefix)
    if config is None:
        return ContentDeduper(job_id, crawl_scope, store)
    return ContentDeduper(job_id, crawl_scope, store, config=config)


def _page_id() -> str:
    return new_id("page")


# ---- #
# Tests
# ---- #


class TestContentDedupe_S46:
    """S46 — Content dedupe: content hash per job, `duplicate_of`, canonical URLs respected."""

    # ---- hashing ---- #

    def test_content_hash(self) -> None:
        assert content_hash(_TEXT) == content_hash(f"\n\n{_TEXT}  \n")  # surrounding space
        assert content_hash(_TEXT) != content_hash(_TEXT.replace("1200", "1300"))
        assert len(content_hash(_TEXT)) == 64

    # ---- duplicates ---- #

    async def test_first_page_is_unique_later_ones_duplicate_it(self) -> None:
        deduper = _deduper(InMemoryContentClaimStore())
        first, second = _page_id(), _page_id()

        a = await deduper.check(page_id=first, url=f"{_SEED}a", markdown=_TEXT, depth=1)
        b = await deduper.check(page_id=second, url=f"{_SEED}b", markdown=_TEXT, depth=1)

        assert (a.outcome, a.page_status, a.reason) == (
            DedupeOutcome.UNIQUE,
            PageStatus.SUCCEEDED,
            None,
        )
        assert (b.outcome, b.page_status, b.reason) == (
            DedupeOutcome.DUPLICATE,
            PageStatus.DUPLICATE,
            None,
        )
        assert (b.duplicate_of, b.duplicate_of_url) == (first, f"{_SEED}a")
        assert a.content_hash == b.content_hash

    async def test_rechecking_the_original_is_idempotent(self) -> None:
        deduper = _deduper(InMemoryContentClaimStore())
        page = _page_id()

        await deduper.check(page_id=page, url=f"{_SEED}a", markdown=_TEXT, depth=1)
        again = await deduper.check(page_id=page, url=f"{_SEED}a", markdown=_TEXT, depth=1)

        assert again.outcome is DedupeOutcome.UNIQUE  # a resumed job reprocesses its pages

    async def test_jobs_are_isolated(self) -> None:
        store = InMemoryContentClaimStore()
        other_job = "job_" + "1" * 32

        a = await _deduper(store).check(page_id=_page_id(), url=_SEED, markdown=_TEXT, depth=0)
        b = await _deduper(store, other_job).check(
            page_id=_page_id(), url=_SEED, markdown=_TEXT, depth=0
        )

        assert (a.outcome, b.outcome) == (DedupeOutcome.UNIQUE, DedupeOutcome.UNIQUE)

    async def test_concurrent_checks_produce_one_original(self) -> None:
        deduper = _deduper(InMemoryContentClaimStore())
        pages = [_page_id() for _ in range(20)]

        decisions = await asyncio.gather(
            *(
                deduper.check(page_id=p, url=f"{_SEED}p{i}", markdown=_TEXT, depth=1)
                for i, p in enumerate(pages)
            )
        )

        uniques = [d for d in decisions if d.outcome is DedupeOutcome.UNIQUE]
        assert len(uniques) == 1
        original = pages[decisions.index(uniques[0])]
        assert {d.duplicate_of for d in decisions if d is not uniques[0]} == {original}

    # ---- canonical URLs ---- #

    async def test_canonical_elsewhere_skips_without_claiming(self) -> None:
        store = InMemoryContentClaimStore()
        deduper = _deduper(store)

        variant = await deduper.check(
            page_id=_page_id(),
            url=f"{_SEED}guide?print=1",
            markdown=_TEXT,
            depth=2,
            canonical_url=f"{_SEED}guide",
        )
        target = await deduper.check(
            page_id=_page_id(), url=f"{_SEED}guide", markdown=_TEXT, depth=1
        )

        assert (variant.outcome, variant.page_status, variant.reason) == (
            DedupeOutcome.CANONICAL_ELSEWHERE,
            PageStatus.SKIPPED,
            PageReason.CANONICAL_ELSEWHERE,
        )
        assert variant.canonical_url == f"{_SEED}guide"
        assert target.outcome is DedupeOutcome.UNIQUE  # the hash was left for the canonical

    @pytest.mark.parametrize(
        ("canonical", "aliases"),
        [
            (f"{_SEED}guide", ()),  # itself
            ("HTTPS://ACME.example:443/guide?utm_source=x#top", ()),  # itself, other spelling
            (f"{_SEED}old-guide", (f"{_SEED}old-guide",)),  # the URL that redirected here
            ("https://other.example/guide", ()),  # out of scope: ignored, content kept
            ("not a url", ()),  # unparseable: ignored
            (None, ()),
        ],
        ids=["self", "self-normalized", "redirect-alias", "out-of-scope", "invalid", "none"],
    )
    async def test_canonical_that_does_not_point_elsewhere(
        self, canonical: str | None, aliases: tuple[str, ...]
    ) -> None:
        decision = await _deduper(InMemoryContentClaimStore()).check(
            page_id=_page_id(),
            url=f"{_SEED}guide",
            markdown=_TEXT,
            depth=1,
            canonical_url=canonical,
            aliases=aliases,
        )

        assert decision.outcome is DedupeOutcome.UNIQUE

    async def test_canonical_outside_the_path_prefix_is_ignored(self) -> None:
        deduper = _deduper(InMemoryContentClaimStore(), path_prefix="/docs/")

        decision = await deduper.check(
            page_id=_page_id(),
            url=f"{_SEED}docs/a",
            markdown=_TEXT,
            depth=1,
            canonical_url=f"{_SEED}blog/a",
        )

        assert decision.outcome is DedupeOutcome.UNIQUE

    async def test_canonical_can_be_ignored_by_config(
        self, override_settings: OverrideSettings
    ) -> None:
        config: Settings = override_settings(CRAWL_RESPECT_CANONICAL=False)

        decision = await _deduper(InMemoryContentClaimStore(), config=config).check(
            page_id=_page_id(),
            url=f"{_SEED}variant",
            markdown=_TEXT,
            depth=1,
            canonical_url=f"{_SEED}guide",
        )

        assert decision.outcome is DedupeOutcome.UNIQUE
        assert decision.canonical_url == f"{_SEED}guide"  # still recorded

    # ---- page records & fixtures ---- #

    @pytest.mark.parametrize("outcome", list(DedupeOutcome))
    async def test_decisions_produce_valid_page_records(self, outcome: DedupeOutcome) -> None:
        store = InMemoryContentClaimStore()
        deduper = _deduper(store)
        original = _page_id()
        await deduper.check(page_id=original, url=f"{_SEED}a", markdown=_TEXT, depth=1)
        page_id = original if outcome is DedupeOutcome.UNIQUE else _page_id()
        canonical = f"{_SEED}a" if outcome is DedupeOutcome.CANONICAL_ELSEWHERE else None

        decision = await deduper.check(
            page_id=page_id, url=f"{_SEED}b", markdown=_TEXT, depth=1, canonical_url=canonical
        )
        page = Page(
            id=page_id,
            tenant_id="ten_" + "0" * 32,
            kb_id="kb_" + "0" * 32,
            source_id="src_" + "0" * 32,
            job_id=_JOB,
            url=f"{_SEED}b",
            url_hash="0" * 64,
            status=decision.page_status,
            reason=decision.reason,
            duplicate_of=decision.duplicate_of,
            content_hash=decision.content_hash,
            canonical_url=decision.canonical_url,
            content_kind=ContentKind.HTML if outcome is DedupeOutcome.UNIQUE else None,
            markdown=_TEXT if outcome is DedupeOutcome.UNIQUE else None,
        )

        assert decision.outcome is outcome
        assert page.status is decision.page_status

    async def test_fixture_site_expectations(self) -> None:
        extractor, cleaner, links = HtmlExtractor(), TextCleaner(), LinkExtractor()
        base = "http://127.0.0.1:8000"
        scope = CrawlScope.from_config(f"{base}/", max_depth=10)
        deduper = ContentDeduper(_JOB, scope, InMemoryContentClaimStore())
        decisions = {}
        for path in (
            "/dup-a.html",
            "/dup-b.html",
            "/docs/getting-started.html",
            "/canonical-variant.html",
        ):
            raw = (SITE_ROOT / path.lstrip("/")).read_bytes().replace(b"{{BASE}}", base.encode())
            html, _ = decode_html(raw, None)
            url = base + path
            source = ExtractionInput(url, "text/html", raw, html)
            markdown = cleaner.clean((await extractor.extract(source)).markdown).markdown
            canonical = links.extract(html, url, scope=scope, depth=1).canonical_url
            decisions[path] = await deduper.check(
                page_id=_page_id(), url=url, markdown=markdown, depth=1, canonical_url=canonical
            )

        for path in ("/dup-b.html", "/canonical-variant.html"):
            expected = EXPECTED_PAGE_OUTCOME[SITE_PAGES[path]]
            assert (decisions[path].page_status, decisions[path].reason) == expected, path
        assert decisions["/dup-a.html"].outcome is DedupeOutcome.UNIQUE
        assert decisions["/dup-b.html"].duplicate_of_url == f"{base}/dup-a.html"
        assert decisions["/docs/getting-started.html"].outcome is DedupeOutcome.UNIQUE
