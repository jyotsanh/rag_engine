"""Engine politeness hook for browser renders, and `AutoFetcher(before_render=...)`."""

from dataclasses import dataclass, field

import pytest

from rag_engine.config import JsRenderMode
from rag_engine.crawling import (
    CachedRobots,
    Reservation,
    RobotsRules,
    RobotsStatus,
    origin_of,
    render_politeness,
)
from rag_engine.crawling.fetchers import AutoFetcher, FetchResult, RenderedPage
from rag_engine.domain import ContentKind, RenderedWith

_URL = "https://acme.example/app/"
_SHELL = "<html><body><div id='root'></div><script src='/a.js'></script></body></html>"
_PARAGRAPHS = "<p>Plenty of server-rendered text here.</p>" * 40
_RICH = f"<html><body><main>{_PARAGRAPHS}</main></body></html>"


def _result(html: str) -> FetchResult:
    return FetchResult(
        requested_url=_URL,
        final_url=_URL,
        redirect_chain=(),
        status_code=200,
        mime_type="text/html",
        content_kind=ContentKind.HTML,
        body=html.encode(),
        text=html,
        encoding="utf-8",
        bytes_downloaded=len(html),
        elapsed_ms=1,
        headers={},
    )


@dataclass
class _Http:
    html: str

    async def fetch(self, url: str) -> FetchResult:
        return _result(self.html)


@dataclass
class _Browser:
    events: list[str]

    async def render(self, url: str, **_: object) -> RenderedPage:
        self.events.append(f"render {url}")
        return RenderedPage(url, url, 200, _RICH, elapsed_ms=5)


class _Robots:
    async def rules_for(self, url: str) -> RobotsRules:
        entry = CachedRobots(
            status=RobotsStatus.OK, body="User-agent: *\nCrawl-delay: 2\n", fetched_at=0,
            expires_at=1e12,
        )  # fmt: skip
        return RobotsRules.from_cached(origin_of(url), entry, "ragenginebot")


@dataclass
class _Limiter:
    events: list[str] = field(default_factory=list)

    async def acquire(
        self, url: str, *, crawl_delay: float | None = None, max_wait: float | None = None
    ) -> Reservation:
        self.events.append(f"acquire {url} delay={crawl_delay} max_wait={max_wait}")
        return Reservation("acme.example", True, 0.0, 0.0, 2.0, 1)

    async def pause(self, url: str, seconds: float) -> float:
        return seconds


def _auto(html: str, events: list[str], mode: JsRenderMode, hook: object) -> AutoFetcher:
    return AutoFetcher(
        _Http(html),  # ty: ignore[invalid-argument-type]
        _Browser(events),  # ty: ignore[invalid-argument-type]
        mode=mode,
        before_render=hook,  # ty: ignore[invalid-argument-type]
    )


# ---- #
# Tests
# ---- #


class TestRenderPoliteness_S50:
    """S50 — Crawl engine: the browser's second request books a rate-limit slot too."""

    async def test_hook_books_a_slot_with_the_sites_crawl_delay(self) -> None:
        limiter = _Limiter()

        await render_politeness(_Robots(), limiter)(_URL)

        # no max_wait: the page already holds a crawl slot and must be rendered now
        assert limiter.events == [f"acquire {_URL} delay=2.0 max_wait=None"]

    async def test_auto_fetcher_books_before_rendering(self) -> None:
        events: list[str] = []
        limiter = _Limiter(events)
        fetcher = _auto(_SHELL, events, JsRenderMode.AUTO, render_politeness(_Robots(), limiter))

        result = await fetcher.fetch(_URL)

        assert result.rendered_with is RenderedWith.BROWSER
        assert events == [f"acquire {_URL} delay=2.0 max_wait=None", f"render {_URL}"]

    @pytest.mark.parametrize(
        ("html", "mode"),
        [(_RICH, JsRenderMode.AUTO), (_SHELL, JsRenderMode.NEVER)],
        ids=["no-render-needed", "rendering-off"],
    )
    async def test_no_render_no_extra_slot(self, html: str, mode: JsRenderMode) -> None:
        events: list[str] = []
        fetcher = _auto(html, events, mode, render_politeness(_Robots(), _Limiter(events)))

        result = await fetcher.fetch(_URL)

        assert result.rendered_with is RenderedWith.HTTP
        assert events == []
