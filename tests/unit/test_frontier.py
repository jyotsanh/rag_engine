"""Frontier ordering and configuration (no database)."""

import pytest
from pydantic import ValidationError

from rag_engine.config import settings
from rag_engine.crawling import frontier_priority
from rag_engine.domain import DiscoveredVia, Page, PageStatus
from rag_engine.repositories import PageOutcome
from tests.conftest import OverrideSettings

# ---- #
# Tests
# ---- #


class TestFrontierOrder_S48:
    """S48 — Frontier: priority order, page defaults, lease setting, outcome clears lease."""

    def test_priority_orders_by_depth_then_discovery_source(self) -> None:
        # Arrange: deliberately shuffled
        entries = [
            (DiscoveredVia.LINK, 1),
            (DiscoveredVia.LINK, 2),
            (DiscoveredVia.SITEMAP, 0),
            (DiscoveredVia.CANONICAL, 1),
            (DiscoveredVia.SEED, 0),
            (DiscoveredVia.CANONICAL, 2),
        ]

        # Act
        ordered = sorted(entries, key=lambda e: frontier_priority(*e))

        # Assert
        assert ordered == [
            (DiscoveredVia.SEED, 0),
            (DiscoveredVia.SITEMAP, 0),
            (DiscoveredVia.CANONICAL, 1),
            (DiscoveredVia.LINK, 1),
            (DiscoveredVia.CANONICAL, 2),
            (DiscoveredVia.LINK, 2),
        ]

    def test_every_source_ranks_below_the_next_depth(self) -> None:
        worst_at_depth_3 = max(frontier_priority(via, 3) for via in DiscoveredVia)
        best_at_depth_4 = min(frontier_priority(via, 4) for via in DiscoveredVia)

        assert worst_at_depth_3 < best_at_depth_4

    def test_negative_depth_rejected(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            frontier_priority(DiscoveredVia.LINK, -1)

    def test_page_defaults_to_an_unclaimed_queued_link(self) -> None:
        page = Page(
            tenant_id="ten_" + "1" * 32,
            kb_id="kb_" + "1" * 32,
            source_id="src_" + "1" * 32,
            job_id="job_" + "1" * 32,
            url="https://acme.example/",
            url_hash="0" * 64,
        )

        assert page.status is PageStatus.QUEUED
        assert page.discovered_via is DiscoveredVia.LINK
        assert (page.priority, page.attempts) == (0, 0)
        assert page.retry_at is page.claimed_by is page.lease_expires_at is None

    def test_lease_must_outlast_fetch_and_render(self, override_settings: OverrideSettings) -> None:
        page_seconds = settings.CRAWL_FETCH_TIMEOUT_SECONDS + settings.CRAWL_RENDER_TIMEOUT_SECONDS

        with pytest.raises(ValidationError, match="CRAWL_FRONTIER_LEASE_SECONDS"):
            override_settings(CRAWL_FRONTIER_LEASE_SECONDS=int(page_seconds))
        assert page_seconds < settings.CRAWL_FRONTIER_LEASE_SECONDS

    def test_recording_an_outcome_clears_the_lease(self) -> None:
        changes = PageOutcome(status=PageStatus.DUPLICATE, duplicate_of="page_" + "1" * 32).changes(
            1000
        )

        assert changes["lease_expires_at"] is None and changes["retry_at"] is None
        assert "claimed_by" not in changes  # kept as an audit of the run that finished it
