"""Per-domain rate limiter: politeness key and interval/burst rules (no Redis needed)."""

import pytest
from redis.asyncio import Redis

from rag_engine.crawling import DomainRateLimiter, rate_limit_host
from rag_engine.infrastructure import RedisKeyspace
from tests.conftest import OverrideSettings


@pytest.fixture
def limiter(override_settings: OverrideSettings) -> DomainRateLimiter:
    config = override_settings(
        CRAWL_PER_DOMAIN_RPS=2.0, CRAWL_PER_DOMAIN_BURST=3, CRAWL_MAX_CRAWL_DELAY_SECONDS=30
    )
    # registering scripts does not connect, so an unconnected client is enough here
    return DomainRateLimiter(Redis(), RedisKeyspace("test"), config)


# ---- #
# Tests
# ---- #


class TestDomainRateLimitRules_S30:
    """S30 — Per-domain rate limiter: host key, RPS vs crawl-delay interval, burst."""

    @pytest.mark.parametrize(
        ("url", "host"),
        [
            ("https://Docs.Example.com/a", "docs.example.com"),
            ("http://docs.example.com:8080/b", "docs.example.com"),  # ports share a bucket
            ("http://bücher.example/", "xn--bcher-kva.example"),
            ("http://[::1]:8000/", "::1"),
        ],
    )
    def test_rate_limit_host(self, url: str, host: str) -> None:
        assert rate_limit_host(url) == host

    @pytest.mark.parametrize(
        ("crawl_delay", "interval", "burst"),
        [
            (None, 0.5, 3),  # configured RPS and burst
            (0, 0.5, 3),
            (0.1, 0.5, 1),  # a shorter crawl delay never speeds us up, but disables bursts
            (2.0, 2.0, 1),  # a longer crawl delay wins
            (86_400, 30.0, 1),  # hostile values are clamped
        ],
    )
    def test_interval_and_burst(
        self, limiter: DomainRateLimiter, crawl_delay: float | None, interval: float, burst: int
    ) -> None:
        assert limiter.interval_for(crawl_delay) == interval
        assert limiter.burst_for(crawl_delay) == burst

    async def test_negative_max_wait_rejected(self, limiter: DomainRateLimiter) -> None:
        with pytest.raises(ValueError, match="max_wait"):
            await limiter.reserve("https://a.example/", max_wait=-1)
