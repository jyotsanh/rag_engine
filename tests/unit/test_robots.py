"""robots.txt: parsing, RFC 9309 status semantics, per-origin caching, single flight."""

import asyncio
from collections.abc import AsyncIterator

import pytest

import rag_engine.crawling.robots as robots_module
from rag_engine.config import Settings, settings
from rag_engine.crawling import (
    CachedRobots,
    RetryPolicy,
    RobotsCache,
    RobotsRules,
    RobotsStatus,
    origin_of,
    robots_token,
)
from rag_engine.crawling.fetchers import (
    FetchError,
    FetchTimeoutError,
    HttpFetcher,
    HttpStatusError,
    NetworkFetchError,
    RawResponse,
    RedirectLoopFetchError,
    SsrfFetchError,
    TooManyRedirectsFetchError,
)
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fakes import InMemoryRobotsStore
from tests.support.fixture_site import FixtureSite

_ORIGIN = "https://acme.example"
_UA = "RagEngineBot"
_TTL = settings.ROBOTS_CACHE_TTL_SECONDS
_UNREACHABLE_TTL = settings.ROBOTS_UNREACHABLE_CACHE_TTL_SECONDS
_STALE_MAX = settings.ROBOTS_STALE_MAX_SECONDS

_SITE_ROBOTS = """\
User-agent: *
Disallow: /private/
Allow: /private/press/
Disallow: /*.pdf$
Crawl-delay: 2

User-agent: RagEngineBot
Disallow: /no-rag/
Allow: /no-rag/ok.html

User-agent: github
Disallow: /

Sitemap: https://acme.example/sitemap.xml
Sitemap: https://acme.example/news-sitemap.xml
"""


def _raw(body: str | bytes, *, truncated: bool = False) -> RawResponse:
    data = body.encode() if isinstance(body, str) else body
    return RawResponse(
        requested_url=_ORIGIN + "/robots.txt",
        final_url=_ORIGIN + "/robots.txt",
        redirect_chain=(),
        status_code=200,
        mime_type="text/plain",
        charset=None,
        body=data,
        truncated=truncated,
        bytes_downloaded=len(data),
        elapsed_ms=1,
    )


def _status(code: int) -> HttpStatusError:
    return HttpStatusError(url=_ORIGIN + "/robots.txt", status_code=code)


class _ScriptedFetcher:
    """`RawFetcher` that plays outcomes in order; the last one repeats forever."""

    def __init__(self, *outcomes: str | bytes | RawResponse | FetchError) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None

    async def fetch_raw(
        self, url: str, *, accept: str, max_bytes: int, truncate: bool = False
    ) -> RawResponse:
        self.calls.append(url)
        if self.gate is not None:
            await self.gate.wait()
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, FetchError):
            raise outcome
        return outcome if isinstance(outcome, RawResponse) else _raw(outcome)


class _CountingStore(InMemoryRobotsStore):
    def __init__(self) -> None:
        super().__init__()
        self.loads: list[str] = []

    async def load(self, origin: str) -> CachedRobots | None:
        self.loads.append(origin)
        return await super().load(origin)


class _Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _fast_policy(max_attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts, base_delay=0.001, max_delay=0.001, max_retry_after=60
    )


def _cache(
    fetcher: _ScriptedFetcher,
    store: InMemoryRobotsStore | None = None,
    clock: _Clock | None = None,
) -> RobotsCache:
    return RobotsCache(
        fetcher,
        store if store is not None else InMemoryRobotsStore(),
        policy=_fast_policy(),
        clock=clock or _Clock(),
    )


def _rules(body: str, status: RobotsStatus = RobotsStatus.OK) -> RobotsRules:
    entry = CachedRobots(status, body, fetched_at=0, expires_at=1)
    return RobotsRules.from_cached(_ORIGIN, entry, _UA)


@pytest.fixture
async def http_fetcher(override_settings: OverrideSettings) -> AsyncIterator[HttpFetcher]:
    config: Settings = override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True,
        CRAWL_READ_TIMEOUT_SECONDS=2,
        CRAWL_FETCH_TIMEOUT_SECONDS=3,
    )
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


# ---- #
# Tests
# ---- #


class TestRobotsTxt_S29:
    """S29 — robots.txt: protego rules, RFC 9309 status semantics, Redis-backed cache."""

    # ---- helpers ---- #

    @pytest.mark.parametrize(
        ("url", "origin"),
        [
            ("HTTPS://Docs.Example.com:443/a/b?x=1#f", "https://docs.example.com"),
            ("http://example.com:8080/", "http://example.com:8080"),
            ("http://bücher.example/katalog", "http://xn--bcher-kva.example"),
            ("http://[::1]:8000/x", "http://[::1]:8000"),
        ],
    )
    def test_origin_of(self, url: str, origin: str) -> None:
        assert origin_of(url) == origin

    @pytest.mark.parametrize(
        ("user_agent", "token"),
        [
            ("RagEngineBot/0.1 (+https://github.com/x)", "RagEngineBot"),
            ("RagEngineBot", "RagEngineBot"),
            ("  Crawler 2.0 compatible", "Crawler"),
        ],
    )
    def test_robots_token(self, user_agent: str, token: str) -> None:
        assert robots_token(user_agent) == token

    def test_robots_token_rejects_empty_agent(self) -> None:
        with pytest.raises(ValueError):
            robots_token("   ")

    # ---- rules ---- #

    @pytest.mark.parametrize(
        ("path", "allowed"),
        [
            ("/", True),
            ("/no-rag/page.html", False),  # our own group
            ("/no-rag/ok.html", True),  # longest match: Allow beats Disallow
            ("/private/x.html", True),  # our group replaces `*`, which disallowed it
            ("/robots.txt", True),
        ],
    )
    def test_specific_group_overrides_wildcard(self, path: str, allowed: bool) -> None:
        assert _rules(_SITE_ROBOTS).can_fetch(_ORIGIN + path) is allowed

    @pytest.mark.parametrize(
        ("path", "allowed"),
        [
            ("/private/x.html", False),
            ("/private/press/release.html", True),  # longest match
            ("/docs/spec.pdf", False),  # `*` and `$` patterns
            ("/docs/spec.pdf?download=1", True),
            ("/docs/", True),
        ],
    )
    def test_wildcard_group_patterns(self, path: str, allowed: bool) -> None:
        body = _SITE_ROBOTS.replace("User-agent: RagEngineBot", "User-agent: SomeoneElse")

        assert _rules(body).can_fetch(_ORIGIN + path) is allowed

    def test_ua_token_is_not_matched_inside_our_url(self) -> None:
        # our full UA contains "github"; the `User-agent: github` group must not apply
        rules = RobotsRules.from_cached(
            _ORIGIN, CachedRobots(RobotsStatus.OK, "User-agent: github\nDisallow: /\n", 0, 1), _UA
        )

        assert rules.can_fetch(_ORIGIN + "/page")

    def test_robots_txt_is_always_allowed(self) -> None:
        rules = _rules("User-agent: *\nDisallow: /\n")

        assert not rules.can_fetch(_ORIGIN + "/")
        assert rules.can_fetch(_ORIGIN + "/robots.txt")

    @pytest.mark.parametrize(
        ("body", "delay"),
        [
            ("User-agent: *\nCrawl-delay: 2.5\n", 2.5),
            ("User-agent: *\nRequest-rate: 1/5\n", 5.0),
            ("User-agent: *\nRequest-rate: 10/5\n", 0.5),
            ("User-agent: *\nCrawl-delay: 1\nRequest-rate: 1/4\n", 4.0),  # the larger wins
            ("User-agent: *\nDisallow: /x\n", None),
            ("User-agent: RagEngineBot\nCrawl-delay: 7\n\nUser-agent: *\nCrawl-delay: 1\n", 7.0),
        ],
    )
    def test_crawl_delay(self, body: str, delay: float | None) -> None:
        assert _rules(body).crawl_delay == delay

    def test_sitemaps_are_exposed(self) -> None:
        assert _rules(_SITE_ROBOTS).sitemaps == (
            "https://acme.example/sitemap.xml",
            "https://acme.example/news-sitemap.xml",
        )

    def test_unavailable_and_unreachable_rules(self) -> None:
        unavailable = _rules("", RobotsStatus.UNAVAILABLE)
        unreachable = _rules("", RobotsStatus.UNREACHABLE)

        assert unavailable.can_fetch(_ORIGIN + "/anything")
        assert not unreachable.can_fetch(_ORIGIN + "/")
        assert unreachable.can_fetch(_ORIGIN + "/robots.txt")
        assert unavailable.crawl_delay is None and unavailable.sitemaps == ()

    # ---- status semantics ---- #

    async def test_ok_robots_is_parsed_and_cached(self) -> None:
        fetcher, store = _ScriptedFetcher(_SITE_ROBOTS), InMemoryRobotsStore()
        cache = _cache(fetcher, store)

        rules = await cache.rules_for(_ORIGIN + "/no-rag/page.html")

        assert rules.status is RobotsStatus.OK
        assert fetcher.calls == [_ORIGIN + "/robots.txt"]
        assert not await cache.can_fetch(_ORIGIN + "/no-rag/page.html")
        assert store.ttls[("current", _ORIGIN)] == _TTL
        assert store.ttls[("last_good", _ORIGIN)] == _STALE_MAX
        assert store.last_good[_ORIGIN].body == _SITE_ROBOTS

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 410, 451])
    async def test_4xx_means_no_restrictions(self, code: int) -> None:
        fetcher, store = _ScriptedFetcher(_status(code)), InMemoryRobotsStore()

        rules = await _cache(fetcher, store).rules_for(_ORIGIN + "/admin/")

        assert rules.status is RobotsStatus.UNAVAILABLE
        assert rules.can_fetch(_ORIGIN + "/admin/")
        assert len(fetcher.calls) == 1  # permanent: not retried
        assert store.ttls[("current", _ORIGIN)] == _TTL
        assert _ORIGIN not in store.last_good

    @pytest.mark.parametrize(
        "error",
        [
            RedirectLoopFetchError("loop", url=_ORIGIN),
            TooManyRedirectsFetchError("too many", url=_ORIGIN),
            SsrfFetchError("blocked", url=_ORIGIN),
        ],
        ids=["loop", "too_many", "ssrf"],
    )
    async def test_redirect_failures_mean_unavailable(self, error: FetchError) -> None:
        rules = await _cache(_ScriptedFetcher(error)).rules_for(_ORIGIN + "/")

        assert rules.status is RobotsStatus.UNAVAILABLE
        assert rules.can_fetch(_ORIGIN + "/")

    @pytest.mark.parametrize(
        "error",
        [
            _status(500),
            _status(503),
            _status(429),
            FetchTimeoutError("timed out", url=_ORIGIN),
            NetworkFetchError("refused", url=_ORIGIN),
        ],
        ids=["500", "503", "429", "timeout", "network"],
    )
    async def test_unreachable_means_complete_disallow(self, error: FetchError) -> None:
        fetcher, store = _ScriptedFetcher(error), InMemoryRobotsStore()

        rules = await _cache(fetcher, store).rules_for(_ORIGIN + "/")

        assert rules.status is RobotsStatus.UNREACHABLE
        assert not rules.can_fetch(_ORIGIN + "/")
        assert not rules.can_fetch(_ORIGIN + "/about.html")
        assert len(fetcher.calls) == 3  # retried up to the attempt limit
        assert "gave up after 3 attempts" in (rules.detail or "")
        assert store.ttls[("current", _ORIGIN)] == _UNREACHABLE_TTL

    async def test_transient_failure_then_success(self) -> None:
        fetcher = _ScriptedFetcher(_status(503), _SITE_ROBOTS)

        rules = await _cache(fetcher).rules_for(_ORIGIN + "/")

        assert rules.status is RobotsStatus.OK
        assert len(fetcher.calls) == 2

    async def test_unreachable_falls_back_to_last_good_copy(self) -> None:
        clock, store = _Clock(), InMemoryRobotsStore()
        good = CachedRobots(RobotsStatus.OK, _SITE_ROBOTS, clock.now - 3600, clock.now - 1)
        store.last_good[_ORIGIN] = good

        rules = await _cache(_ScriptedFetcher(_status(503)), store, clock).rules_for(_ORIGIN)

        assert rules.status is RobotsStatus.STALE
        assert rules.can_fetch(_ORIGIN + "/about.html")
        assert not rules.can_fetch(_ORIGIN + "/no-rag/page.html")  # the old rules apply
        assert store.current[_ORIGIN].source_fetched_at == good.fetched_at
        assert store.ttls[("current", _ORIGIN)] == _UNREACHABLE_TTL

    async def test_too_old_last_good_copy_is_not_used(self) -> None:
        clock, store = _Clock(), InMemoryRobotsStore()
        old = clock.now - _STALE_MAX - 1
        store.last_good[_ORIGIN] = CachedRobots(RobotsStatus.OK, _SITE_ROBOTS, old, old + 1)

        rules = await _cache(_ScriptedFetcher(_status(503)), store, clock).rules_for(_ORIGIN)

        assert rules.status is RobotsStatus.UNREACHABLE

    async def test_truncated_robots_keeps_the_parsed_prefix(self) -> None:
        fetcher = _ScriptedFetcher(_raw("User-agent: *\nDisallow: /x/\n", truncated=True))

        rules = await _cache(fetcher).rules_for(_ORIGIN)

        assert rules.status is RobotsStatus.OK
        assert rules.detail == "truncated"
        assert not rules.can_fetch(_ORIGIN + "/x/page")

    async def test_bom_and_invalid_utf8_are_tolerated(self) -> None:
        body = b"\xef\xbb\xbfUser-agent: *\nDisallow: /x/\n# caf\xe9\n"

        rules = await _cache(_ScriptedFetcher(_raw(body))).rules_for(_ORIGIN)

        assert not rules.can_fetch(_ORIGIN + "/x/page")
        assert rules.can_fetch(_ORIGIN + "/y")

    # ---- caching ---- #

    async def test_one_fetch_per_origin(self) -> None:
        fetcher = _ScriptedFetcher(_SITE_ROBOTS)
        cache = _cache(fetcher)

        for path in ("/", "/a", "/b/c", "/no-rag/x"):
            await cache.can_fetch(_ORIGIN + path)

        assert len(fetcher.calls) == 1

    async def test_origins_are_cached_separately(self) -> None:
        fetcher = _ScriptedFetcher(_SITE_ROBOTS)
        cache = _cache(fetcher)

        for url in (
            "https://acme.example/a",
            "http://acme.example/a",
            "https://acme.example:8443/a",
            "https://docs.acme.example/a",
            "HTTPS://ACME.example:443/b",  # same origin as the first
        ):
            await cache.rules_for(url)

        assert fetcher.calls == [
            "https://acme.example/robots.txt",
            "http://acme.example/robots.txt",
            "https://acme.example:8443/robots.txt",
            "https://docs.acme.example/robots.txt",
        ]

    async def test_expired_entries_are_refetched(self) -> None:
        clock = _Clock()
        fetcher = _ScriptedFetcher("User-agent: *\nDisallow: /old/\n", _SITE_ROBOTS)
        cache = _cache(fetcher, clock=clock)
        assert not await cache.can_fetch(_ORIGIN + "/old/page")

        clock.now += _TTL + 1

        assert await cache.can_fetch(_ORIGIN + "/old/page")
        assert len(fetcher.calls) == 2

    async def test_unreachable_verdict_is_rechecked_sooner(self) -> None:
        clock = _Clock()
        fetcher = _ScriptedFetcher(_status(503), _status(503), _status(503), _SITE_ROBOTS)
        cache = _cache(fetcher, clock=clock)
        assert (await cache.rules_for(_ORIGIN)).status is RobotsStatus.UNREACHABLE

        clock.now += _UNREACHABLE_TTL + 1

        assert (await cache.rules_for(_ORIGIN)).status is RobotsStatus.OK

    async def test_shared_store_entry_is_used_without_fetching(self) -> None:
        # another worker already fetched it
        clock, store = _Clock(), InMemoryRobotsStore()
        store.current[_ORIGIN] = CachedRobots(
            RobotsStatus.OK, _SITE_ROBOTS, clock.now - 10, clock.now + 100
        )
        fetcher = _ScriptedFetcher(_status(500))

        rules = await _cache(fetcher, store, clock).rules_for(_ORIGIN + "/")

        assert rules.status is RobotsStatus.OK
        assert fetcher.calls == []

    async def test_expired_shared_store_entry_is_refetched(self) -> None:
        clock, store = _Clock(), InMemoryRobotsStore()
        store.current[_ORIGIN] = CachedRobots(RobotsStatus.OK, "", clock.now - 100, clock.now)
        fetcher = _ScriptedFetcher(_SITE_ROBOTS)

        await _cache(fetcher, store, clock).rules_for(_ORIGIN + "/")

        assert len(fetcher.calls) == 1

    async def test_concurrent_lookups_share_one_fetch(self) -> None:
        fetcher = _ScriptedFetcher(_SITE_ROBOTS)
        fetcher.gate = asyncio.Event()
        cache = _cache(fetcher)

        waiters = [asyncio.create_task(cache.rules_for(f"{_ORIGIN}/p{i}")) for i in range(20)]
        await asyncio.sleep(0)
        fetcher.gate.set()
        results = await asyncio.gather(*waiters)

        assert len(fetcher.calls) == 1
        assert all(r.status is RobotsStatus.OK for r in results)

    async def test_cancelled_caller_does_not_cancel_the_shared_fetch(self) -> None:
        fetcher = _ScriptedFetcher(_SITE_ROBOTS)
        fetcher.gate = asyncio.Event()
        cache = _cache(fetcher)
        first = asyncio.create_task(cache.rules_for(_ORIGIN + "/a"))
        second = asyncio.create_task(cache.rules_for(_ORIGIN + "/b"))
        await asyncio.sleep(0)

        first.cancel()
        fetcher.gate.set()

        assert (await second).status is RobotsStatus.OK
        with pytest.raises(asyncio.CancelledError):
            await first
        assert len(fetcher.calls) == 1

    async def test_local_cache_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(robots_module, "_LOCAL_CACHE_SIZE", 2)
        fetcher, store = _ScriptedFetcher(_SITE_ROBOTS), _CountingStore()
        cache = _cache(fetcher, store)

        for host in ("a", "b", "c", "b", "a"):  # "b" is still local; "a" was evicted by "c"
            await cache.rules_for(f"https://{host}.example/")

        assert [o.removeprefix("https://") for o in store.loads] == [
            "a.example",
            "b.example",
            "c.example",
            "a.example",
        ]
        assert len(fetcher.calls) == 3  # the evicted origin came back from the shared store

    # ---- through the real fetcher ---- #

    async def test_fixture_site_robots(
        self, http_fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        cache = RobotsCache(http_fetcher, InMemoryRobotsStore(), policy=_fast_policy())

        rules = await cache.rules_for(fixture_site.url("/"))

        assert rules.status is RobotsStatus.OK
        assert rules.origin == fixture_site.base_url
        assert not await cache.can_fetch(fixture_site.url("/private/secret.html"))
        assert await cache.can_fetch(fixture_site.url("/about.html"))
        assert rules.crawl_delay == 0.1
        assert rules.sitemaps == (fixture_site.url("/sitemap_index.xml"),)
        assert fixture_site.hits("/robots.txt") == 1

    async def test_unreachable_host_disallows_everything(self, http_fetcher: HttpFetcher) -> None:
        cache = RobotsCache(http_fetcher, InMemoryRobotsStore(), policy=_fast_policy(2))

        rules = await cache.rules_for("http://127.0.0.1:1/page")

        assert rules.status is RobotsStatus.UNREACHABLE
        assert not rules.can_fetch("http://127.0.0.1:1/page")
