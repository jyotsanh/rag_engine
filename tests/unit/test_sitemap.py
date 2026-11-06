"""Sitemap discovery: parsing, gzip, index recursion, caps, scope filtering, fixture site."""

import gzip
import random
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest
from redis.asyncio import Redis

from rag_engine.config import Settings
from rag_engine.crawling import (
    MAX_ENTRIES_PER_FILE,
    CrawlScope,
    DomainRateLimiter,
    Reservation,
    RetryPolicy,
    RobotsCache,
    SitemapCap,
    SitemapDiscoverer,
    SitemapParseError,
    SitemapProblem,
    inflate,
    parse_lastmod,
    parse_sitemap,
)
from rag_engine.crawling.fetchers import FetchError, HttpFetcher, HttpStatusError, RawResponse
from rag_engine.infrastructure import RedisKeyspace
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fakes import InMemoryRobotsStore
from tests.support.fixture_site import FixtureSite

_NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'

type MakeDiscoverer = Callable[..., SitemapDiscoverer]


def _policy() -> RetryPolicy:
    return RetryPolicy(max_attempts=2, base_delay=0.001, max_delay=0.001, max_retry_after=1)


def _scope(seed: str, path_prefix: str | None = None) -> CrawlScope:
    return CrawlScope.from_config(seed, max_depth=10, path_prefix=path_prefix)


def _urlset(*locs: str) -> bytes:
    body = "".join(f"<url><loc>{loc}</loc></url>" for loc in locs)
    return f'<?xml version="1.0"?><urlset {_NS}>{body}</urlset>'.encode()


class _RecordingLimiter(DomainRateLimiter):
    def __init__(self, config: Settings) -> None:
        super().__init__(Redis(), RedisKeyspace("test"), config)  # never connects
        self.calls: list[tuple[str, float | None]] = []

    async def acquire(
        self, url: str, *, crawl_delay: float | None = None, max_wait: float | None = None
    ) -> Reservation:
        self.calls.append((url, crawl_delay))
        return Reservation(url, True, 0.0, 0.0, 0.5, 1)


class _MapFetcher:
    """`RawFetcher` answering from a {url: body | error} map."""

    def __init__(self, responses: dict[str, bytes | FetchError]) -> None:
        self._responses = responses

    async def fetch_raw(
        self, url: str, *, accept: str, max_bytes: int, truncate: bool = False
    ) -> RawResponse:
        outcome = self._responses.get(url, HttpStatusError(url=url, status_code=404))
        if isinstance(outcome, FetchError):
            raise outcome
        return RawResponse(url, url, (), 200, "text/html", None, outcome, False, len(outcome), 1)


@pytest.fixture
def config(override_settings: OverrideSettings) -> Settings:
    return override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True,
        CRAWL_READ_TIMEOUT_SECONDS=2,
        CRAWL_FETCH_TIMEOUT_SECONDS=3,
    )


@pytest.fixture
async def fetcher(config: Settings) -> AsyncIterator[HttpFetcher]:
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


@pytest.fixture
def make_discoverer(
    fetcher: HttpFetcher, override_settings: OverrideSettings, config: Settings
) -> MakeDiscoverer:
    def make(*, robots: bool = True, **overrides: object) -> SitemapDiscoverer:
        cfg = override_settings(**overrides) if overrides else config
        cache = RobotsCache(fetcher, InMemoryRobotsStore(), config=cfg, policy=_policy())
        return SitemapDiscoverer(
            fetcher, robots=cache if robots else None, config=cfg, policy=_policy()
        )

    return make


# ---- #
# Tests
# ---- #


class TestSitemapDiscovery_S31:
    """S31 — Sitemap discovery: robots + /sitemap.xml, index recursion, gzip, caps, scope."""

    # ---- parsing ---- #

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2024-05-01", datetime(2024, 5, 1, tzinfo=UTC)),
            ("2024-05-01T10:30:00Z", datetime(2024, 5, 1, 10, 30, tzinfo=UTC)),
            ("2024-05-01T10:30:00+02:00", datetime(2024, 5, 1, 8, 30, tzinfo=UTC)),
            ("2024-05-01T10:30", datetime(2024, 5, 1, 10, 30, tzinfo=UTC)),
            ("  2024-05-01 ", datetime(2024, 5, 1, tzinfo=UTC)),
            ("yesterday", None),
            ("", None),
            (None, None),
        ],
    )
    def test_parse_lastmod(self, value: str | None, expected: datetime | None) -> None:
        assert parse_lastmod(value) == expected

    def test_urlset_and_index(self) -> None:
        index = (
            f"<sitemapindex {_NS}><sitemap><loc>https://a.example/s1.xml</loc>"
            "<lastmod>2024-01-02</lastmod></sitemap></sitemapindex>"
        ).encode()

        urlset = parse_sitemap(_urlset("https://a.example/", "https://a.example/b"))
        parsed_index = parse_sitemap(index)

        assert urlset.kind == "urlset"
        assert [loc for loc, _ in urlset.items] == ["https://a.example/", "https://a.example/b"]
        assert parsed_index.kind == "sitemapindex"
        assert parsed_index.items == [
            ("https://a.example/s1.xml", datetime(2024, 1, 2, tzinfo=UTC))
        ]

    def test_namespace_is_optional_and_bom_tolerated(self) -> None:
        data = b"\xef\xbb\xbf\n<urlset><url><loc>https://a.example/x</loc></url></urlset>"

        assert parse_sitemap(data).items == [("https://a.example/x", None)]

    def test_plain_text_sitemap(self) -> None:
        parsed = parse_sitemap(b"https://a.example/1\n\n  https://a.example/2  \n")

        assert parsed.kind == "urlset"
        assert [loc for loc, _ in parsed.items] == ["https://a.example/1", "https://a.example/2"]

    @pytest.mark.parametrize(
        ("data", "problem"),
        [
            (b'<rss version="2.0"><channel/></rss>', SitemapProblem.UNSUPPORTED_FORMAT),
            (b'<feed xmlns="http://www.w3.org/2005/Atom"/>', SitemapProblem.UNSUPPORTED_FORMAT),
            (b"<!DOCTYPE html><html><body>x</body></html>", SitemapProblem.UNSUPPORTED_FORMAT),
            (b"just some words", SitemapProblem.UNSUPPORTED_FORMAT),
            (b"<urlset><url><loc>x</loc></url>", SitemapProblem.INVALID_XML),
            (b"<", SitemapProblem.INVALID_XML),
            (
                b'<!DOCTYPE x [<!ENTITY a "aaaa">]><urlset><url><loc>&a;</loc></url></urlset>',
                SitemapProblem.FORBIDDEN_XML,
            ),
            (
                b'<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><urlset>&e;</urlset>',
                SitemapProblem.FORBIDDEN_XML,
            ),
        ],
        ids=["rss", "atom", "html", "words", "truncated", "garbage", "entities", "xxe"],
    )
    def test_unusable_documents(self, data: bytes, problem: SitemapProblem) -> None:
        with pytest.raises(SitemapParseError) as exc:
            parse_sitemap(data)

        assert exc.value.problem is problem

    def test_entries_per_file_cap(self) -> None:
        locs = [f"https://a.example/p{i}" for i in range(MAX_ENTRIES_PER_FILE + 10)]

        parsed = parse_sitemap(_urlset(*locs))

        assert parsed.capped
        assert len(parsed.items) == MAX_ENTRIES_PER_FILE

    def test_inflate(self) -> None:
        raw = _urlset("https://a.example/")

        assert inflate(raw, 1000) == raw  # not gzip: untouched
        assert inflate(gzip.compress(raw), 1000) == raw

    def test_inflate_caps_gzip_bombs(self) -> None:
        bomb = gzip.compress(b" " * 10_000_000, compresslevel=9)

        with pytest.raises(SitemapParseError) as exc:
            inflate(bomb, 1_000_000)

        assert exc.value.problem is SitemapProblem.TOO_LARGE
        assert len(bomb) < 20_000

    @pytest.mark.parametrize(
        "data",
        [b"\x1f\x8b\x08\x00garbage", gzip.compress(random.Random(0).randbytes(5000))[:-200]],
        ids=["garbage", "truncated"],
    )
    def test_corrupt_gzip(self, data: bytes) -> None:
        with pytest.raises(SitemapParseError) as exc:
            inflate(data, 100_000)

        assert exc.value.problem is SitemapProblem.INVALID_XML

    # ---- discovery on the fixture site ---- #

    async def test_discovers_the_fixture_site(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer().discover(seed, _scope(seed))

        assert set(report.urls) == {
            fixture_site.url(p)
            for p in (
                "/",
                "/about.html",
                "/orphan.html",
                "/blog/post-1.html",
                "/docs/",
                "/docs/getting-started.html",
                "/docs/configuration.html",
                "/docs/api/reference.html",
            )
        }
        assert report.sitemaps_read == [
            fixture_site.url("/sitemap_index.xml"),
            fixture_site.url("/sitemap-pages.xml"),
            fixture_site.url("/sitemap-docs.xml.gz"),  # application/gzip body, inflated
        ]
        assert report.out_of_scope == 1  # offsite.example
        assert report.failures == []  # the absent /sitemap.xml fallback is not a failure
        assert fixture_site.hits("/sitemap.xml") == 1
        docs = next(e for e in report.entries if e.url.endswith("/docs/"))
        assert docs.sitemap == fixture_site.url("/sitemap-docs.xml.gz")

    async def test_scope_filters_sitemap_urls(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/docs/")

        report = await make_discoverer().discover(seed, _scope(seed, path_prefix="/docs/"))

        assert len(report.urls) == 4
        assert all("/docs/" in u for u in report.urls)
        assert report.out_of_scope == 5

    async def test_max_urls_stops_early(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer().discover(seed, _scope(seed), max_urls=3)

        assert len(report.urls) == 3
        assert SitemapCap.MAX_URLS in report.caps_hit
        assert fixture_site.hits("/sitemap-docs.xml.gz") == 0  # never fetched

    async def test_max_files_cap(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer(SITEMAP_MAX_FILES=2).discover(seed, _scope(seed))

        assert len(report.sitemaps_read) == 2
        assert SitemapCap.MAX_FILES in report.caps_hit
        assert fixture_site.hits("/sitemap-docs.xml.gz") == 0

    async def test_without_robots_only_the_fallback_is_tried(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer(robots=False).discover(seed, _scope(seed))

        assert report.urls == [] and report.failures == []
        assert fixture_site.hits("/sitemap.xml") == 1
        assert fixture_site.hits("/robots.txt") == 0

    async def test_index_loop_duplicates_and_invalid_locs(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer().read(
            [fixture_site.url("/sitemaps/loop-index.xml")], _scope(seed)
        )

        assert fixture_site.hits("/sitemaps/loop-index.xml") == 1  # self-reference ignored
        assert report.urls == [fixture_site.url("/terms.html"), fixture_site.url("/privacy.html")]
        assert [e.lastmod for e in report.entries] == [
            datetime(2024, 5, 1, tzinfo=UTC),
            datetime(2024, 5, 1, 8, 30, tzinfo=UTC),
        ]
        assert report.duplicates == 1  # terms.html?utm_source=x normalizes to terms.html
        assert report.invalid == 2  # mailto: and a host-less URL

    async def test_gzip_served_as_x_gzip(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer().read(
            [fixture_site.url("/sitemaps/mixed.xml.gz")], _scope(seed)
        )

        assert len(report.urls) == 2

    async def test_nesting_depth_cap(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed, start = fixture_site.url("/"), [fixture_site.url("/sitemaps/nested-1.xml")]

        deep = await make_discoverer().read(start, _scope(seed))
        shallow = await make_discoverer(SITEMAP_MAX_DEPTH=1).read(start, _scope(seed))

        assert deep.urls == [fixture_site.url("/docs/")]
        assert shallow.urls == []
        assert [(f.sitemap, f.problem) for f in shallow.failures] == [
            (fixture_site.url("/sitemaps/nested-3.xml"), SitemapProblem.TOO_DEEP)
        ]

    async def test_index_children_must_be_in_scope_and_allowed_by_robots(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer().read(
            [fixture_site.url("/sitemaps/foreign-index.xml")], _scope(seed)
        )

        problems = {f.sitemap: f.problem for f in report.failures}
        assert problems == {
            "https://offsite.example/sitemap.xml": SitemapProblem.OUT_OF_SCOPE_HOST,
            fixture_site.other_host_url + "/sitemaps/mixed.xml": SitemapProblem.OUT_OF_SCOPE_HOST,
            fixture_site.url("/private/sitemap.xml"): SitemapProblem.ROBOTS_DISALLOWED,
        }
        assert fixture_site.hits("/private/sitemap.xml") == 0
        assert report.urls == [fixture_site.url("/docs/")]  # the allowed child was read

    async def test_plain_text_sitemap_on_the_site(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer().read(
            [fixture_site.url("/sitemaps/urls.txt")], _scope(seed)
        )

        assert report.urls == [
            fixture_site.url("/about.html"),
            fixture_site.url("/blog/post-2.html"),
        ]
        assert report.out_of_scope == 1

    @pytest.mark.parametrize(
        ("path", "problem"),
        [
            ("/sitemaps/feed.xml", SitemapProblem.UNSUPPORTED_FORMAT),
            ("/sitemaps/soft-404.html", SitemapProblem.UNSUPPORTED_FORMAT),
            ("/sitemaps/broken.xml", SitemapProblem.INVALID_XML),
            ("/sitemaps/laughs.xml", SitemapProblem.FORBIDDEN_XML),
            ("/sitemaps/bomb.xml.gz", SitemapProblem.TOO_LARGE),  # inflates to 4 MiB
            ("/sitemaps/big.xml?n=20000", SitemapProblem.TOO_LARGE),  # download over 1 MiB
            ("/sitemaps/missing.xml", SitemapProblem.FETCH_FAILED),
            ("/error", SitemapProblem.FETCH_FAILED),
        ],
    )
    async def test_broken_sitemaps_are_reported_and_skipped(
        self,
        make_discoverer: MakeDiscoverer,
        fixture_site: FixtureSite,
        path: str,
        problem: SitemapProblem,
    ) -> None:
        seed = fixture_site.url("/")
        good = fixture_site.url("/sitemap-pages.xml")
        discoverer = make_discoverer(SITEMAP_MAX_BYTES=1024 * 1024)

        report = await discoverer.read([fixture_site.url(path), good], _scope(seed))

        assert [(f.sitemap, f.problem) for f in report.failures] == [
            (fixture_site.url(path), problem)
        ]
        assert report.sitemaps_read == [good]  # discovery carried on
        assert len(report.urls) == 4

    async def test_failure_detail_names_the_fetch_reason(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer().read([fixture_site.url("/error")], _scope(seed))

        assert report.failures[0].detail is not None
        assert report.failures[0].detail.startswith("http_server_error")
        assert fixture_site.hits("/error") == 2  # retried per the policy

    async def test_entries_per_file_cap_on_the_site(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")

        report = await make_discoverer().read(
            [fixture_site.url(f"/sitemaps/big.xml?n={MAX_ENTRIES_PER_FILE + 100}")],
            _scope(seed),
            max_urls=MAX_ENTRIES_PER_FILE * 2,  # above the default (CRAWL_HARD_MAX_PAGES)
        )

        assert len(report.urls) == MAX_ENTRIES_PER_FILE
        assert report.caps_hit == {SitemapCap.ENTRIES_PER_FILE}

    async def test_same_sitemap_listed_twice_is_read_once(
        self, make_discoverer: MakeDiscoverer, fixture_site: FixtureSite
    ) -> None:
        seed, url = fixture_site.url("/"), fixture_site.url("/sitemap-pages.xml")

        await make_discoverer().read([url, url + "#frag", url], _scope(seed))

        assert fixture_site.hits("/sitemap-pages.xml") == 1

    async def test_sitemap_fetches_go_through_the_rate_limiter(
        self, fetcher: HttpFetcher, config: Settings, fixture_site: FixtureSite
    ) -> None:
        seed = fixture_site.url("/")
        limiter = _RecordingLimiter(config)
        robots = RobotsCache(fetcher, InMemoryRobotsStore(), config=config, policy=_policy())
        discoverer = SitemapDiscoverer(
            fetcher, robots=robots, limiter=limiter, config=config, policy=_policy()
        )

        await discoverer.discover(seed, _scope(seed))

        assert [url for url, _ in limiter.calls] == [
            fixture_site.url("/sitemap_index.xml"),
            fixture_site.url("/sitemap.xml"),
            fixture_site.url("/sitemap-pages.xml"),
            fixture_site.url("/sitemap-docs.xml.gz"),
        ]
        assert {delay for _, delay in limiter.calls} == {0.1}  # robots.txt Crawl-delay

    # ---- the guessed /sitemap.xml ---- #

    async def test_fallback_soft_404_is_silent_but_server_errors_are_reported(
        self, config: Settings
    ) -> None:
        seed = "https://acme.example/"
        soft_404 = _MapFetcher({"https://acme.example/sitemap.xml": b"<!doctype html><p>nope"})
        down = _MapFetcher(
            {
                "https://acme.example/sitemap.xml": HttpStatusError(
                    url="https://acme.example/sitemap.xml", status_code=503
                )
            }
        )

        quiet = await SitemapDiscoverer(soft_404, config=config, policy=_policy()).discover(
            seed, _scope(seed)
        )
        loud = await SitemapDiscoverer(down, config=config, policy=_policy()).discover(
            seed, _scope(seed)
        )

        assert quiet.failures == [] and quiet.urls == []
        assert [f.problem for f in loud.failures] == [SitemapProblem.FETCH_FAILED]

    async def test_fallback_sitemap_is_used(self, config: Settings) -> None:
        seed = "https://acme.example/"
        fetcher = _MapFetcher(
            {
                "https://acme.example/sitemap.xml": _urlset(
                    "https://acme.example/a", "https://www.acme.example/b"
                )
            }
        )

        report = await SitemapDiscoverer(fetcher, config=config, policy=_policy()).discover(
            seed, _scope(seed)
        )

        assert report.urls == ["https://acme.example/a", "https://www.acme.example/b"]
