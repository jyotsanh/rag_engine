"""Crawl scope: hosts, subdomains, path prefix, globs, assets, depth."""

from typing import Any

import pytest
from pydantic import ValidationError

from rag_engine.crawling import (
    ASSET_EXTENSIONS,
    CrawlScope,
    ScopeRejection,
    normalize_url,
    registrable_domain,
)
from rag_engine.domain import CrawlConfig, PageReason
from rag_engine.domain.types import compile_glob
from tests.support.fixture_site import SITE_PAGES, Expect, FixtureSite

SEED = "https://acme.example/"


def _scope(**options: Any) -> CrawlScope:
    options.setdefault("max_depth", 5)
    return CrawlScope.from_config(SEED, **options)


# ---- #
# Tests
# ---- #


class TestCrawlScope_S24:
    """S24 — Scope rules: same host / subdomains, path prefix, globs, assets, max depth."""

    @pytest.mark.parametrize(
        ("url", "allowed"),
        [
            ("https://acme.example/docs", True),
            ("http://acme.example/docs", True),  # scheme is not part of the site
            ("https://acme.example:8443/docs", True),  # nor is the port
            ("https://www.acme.example/docs", True),  # www ≡ bare domain
            ("https://docs.acme.example/", False),
            ("https://acme.example.evil.example/", False),
            ("https://notacme.example/", False),
            ("https://other.example/", False),
        ],
    )
    def test_same_host_policy(self, url: str, allowed: bool) -> None:
        decision = _scope().check(url, depth=1)

        assert decision.allowed is allowed
        if not allowed:
            assert (decision.reason, decision.detail) == (
                PageReason.OUT_OF_SCOPE,
                ScopeRejection.OTHER_HOST,
            )

    def test_www_seed_accepts_bare_domain(self) -> None:
        scope = CrawlScope.from_config("https://www.acme.example/", max_depth=3)

        assert scope.check("https://acme.example/a", 0).allowed

    @pytest.mark.parametrize(
        ("seed", "url", "allowed"),
        [
            ("https://acme.co.uk/", "https://docs.acme.co.uk/x", True),
            ("https://docs.acme.co.uk/", "https://blog.acme.co.uk/x", True),
            ("https://docs.acme.co.uk/", "https://acme.co.uk/x", True),
            ("https://acme.co.uk/", "https://evil.co.uk/x", False),
            ("https://foo.github.io/", "https://bar.github.io/x", False),  # PSL private section
            ("https://foo.github.io/", "https://docs.foo.github.io/x", True),
            ("https://acme.example/", "https://docs.acme.example/x", True),  # unknown TLD fallback
            ("http://127.0.0.1:8000/", "http://127.0.0.2:8000/x", False),
            ("http://localhost:8000/", "http://other:8000/x", False),
        ],
    )
    def test_include_subdomains(self, seed: str, url: str, allowed: bool) -> None:
        scope = CrawlScope.from_config(seed, max_depth=3, include_subdomains=True)

        assert scope.check(url, 0).allowed is allowed

    def test_registrable_domain(self) -> None:
        assert registrable_domain("a.b.acme.co.uk") == "acme.co.uk"
        assert registrable_domain("foo.github.io") == "foo.github.io"
        assert registrable_domain("x.acme.internal") == "acme.internal"
        assert registrable_domain("10.0.0.1") == "10.0.0.1"

    def test_path_prefix(self) -> None:
        scope = _scope(path_prefix="/docs/")

        assert scope.check("https://acme.example/docs/a", 0).allowed
        assert scope.check("https://acme.example/docs/", 0).allowed
        decision = scope.check("https://acme.example/blog/a", 0)
        assert decision.detail is ScopeRejection.OUTSIDE_PATH_PREFIX
        assert not scope.check("https://acme.example/docs", 0).allowed  # prefix is literal

    @pytest.mark.parametrize(
        ("pattern", "path", "matches"),
        [
            ("/docs/*", "/docs/a", True),
            ("/docs/*", "/docs/a/b", False),
            ("/docs/**", "/docs/a/b/c", True),
            ("/docs/**", "/docs/", True),
            ("/**/*.html", "/a/b/page.html", True),
            ("/v?/api", "/v2/api", True),
            ("/v?/api", "/v10/api", False),
            ("/a.b", "/aXb", False),  # dots are literal
            ("/blog/[draft]", "/blog/[draft]", True),  # brackets are literal
        ],
    )
    def test_glob_semantics(self, pattern: str, path: str, matches: bool) -> None:
        assert bool(compile_glob(pattern).match(path)) is matches

    def test_exclude_wins_over_include(self) -> None:
        scope = _scope(include_patterns=["/docs/**"], exclude_patterns=["/docs/internal/**"])

        assert scope.check("https://acme.example/docs/guide", 0).allowed
        assert scope.check("https://acme.example/docs/internal/x", 0).detail is (
            ScopeRejection.EXCLUDED_PATTERN
        )
        assert scope.check("https://acme.example/blog/x", 0).detail is ScopeRejection.NOT_INCLUDED

    def test_no_includes_means_everything(self) -> None:
        assert _scope(exclude_patterns=["/private/**"]).check("https://acme.example/x", 0).allowed

    @pytest.mark.parametrize(
        "path", ["/logo.PNG", "/app.js", "/style.css", "/font.woff2", "/a/video.mp4", "/x.zip"]
    )
    def test_assets_are_skipped(self, path: str) -> None:
        assert _scope().check(f"https://acme.example{path}", 0).detail is (
            ScopeRejection.ASSET_EXTENSION
        )

    @pytest.mark.parametrize(
        "path", ["/spec.pdf", "/guide.docx", "/deck.pptx", "/prices.xlsx", "/page.html", "/a"]
    )
    def test_documents_and_pages_are_in_scope(self, path: str) -> None:
        assert _scope().check(f"https://acme.example{path}", 0).allowed

    def test_asset_list_does_not_contain_documents(self) -> None:
        assert not {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"} & ASSET_EXTENSIONS

    def test_depth_limit(self) -> None:
        scope = _scope(max_depth=2)

        assert scope.check("https://acme.example/a", 2).allowed
        decision = scope.check("https://acme.example/a", 3)
        assert (decision.reason, decision.detail) == (PageReason.MAX_DEPTH, ScopeRejection.TOO_DEEP)

    def test_out_of_scope_is_reported_before_depth(self) -> None:
        decision = _scope(max_depth=0).check("https://other.example/", 9)

        assert decision.reason is PageReason.OUT_OF_SCOPE

    def test_non_http_scheme(self) -> None:
        assert _scope().check("mailto:x@acme.example", 0).detail is (
            ScopeRejection.UNSUPPORTED_SCHEME
        )

    def test_reanchor_after_seed_redirect(self) -> None:
        scope = CrawlScope.from_config("https://acme.example/", max_depth=3)

        moved = scope.with_seed_host("acme-docs.example")

        assert moved.check("https://acme-docs.example/x", 0).allowed
        assert not moved.check("https://acme.example/x", 0).allowed
        assert moved.max_depth == scope.max_depth

    def test_for_job_uses_crawl_config(self) -> None:
        config = CrawlConfig(
            max_depth=1,
            path_prefix="/docs/",
            include_subdomains=True,
            exclude_patterns=["/docs/old/**"],
        )

        scope = CrawlScope.for_job(SEED, config)

        assert scope.check("https://api.acme.example/docs/a", 1).allowed
        assert not scope.check("https://acme.example/docs/old/a", 0).allowed
        assert not scope.check("https://acme.example/docs/a", 2).allowed

    @pytest.mark.parametrize(
        "config",
        [
            {"include_patterns": ["docs/**"]},
            {"exclude_patterns": ["no-leading-slash"]},
            {"path_prefix": "docs/"},
            {"path_prefix": "/docs/*"},
            {"path_prefix": "/docs?x=1"},
        ],
    )
    def test_crawl_config_rejects_bad_patterns(self, config: dict) -> None:
        with pytest.raises(ValidationError):
            CrawlConfig(**config)

    def test_fixture_site_scope_matches_expectations(self, fixture_site: FixtureSite) -> None:
        scope = CrawlScope.from_config(fixture_site.base_url + "/", max_depth=10)

        for path, expect in SITE_PAGES.items():
            url = fixture_site.url(path)
            if expect is Expect.OUT_OF_SCOPE:
                try:
                    url = normalize_url(url)
                except ValueError:
                    continue  # mailto:/tel:/javascript: never become URLs at all
                assert not scope.check(url, 1).allowed, path
            elif expect is not Expect.UNSUPPORTED_TYPE:
                assert scope.check(normalize_url(url), 1).allowed, path
