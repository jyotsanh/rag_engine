"""A local website that exercises every crawler edge case.

Static content lives in `tests/fixtures/site/`. Text files may contain `{{BASE}}`
(e.g. `http://127.0.0.1:54321`) and `{{OTHER_HOST}}` (same server reached as
`localhost`, i.e. a *different host* for scope rules); both are substituted per
request. Dynamic behaviour (redirects, errors, flakiness, slowness, oversized
bodies, gzip sitemap) is implemented by the routes below.

`SITE_PAGES` documents what a correct crawler should conclude about each URL;
the crawl E2E test (Step 54) asserts against it.
"""

import asyncio
import gzip
import mimetypes
import socket
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

from rag_engine.domain import PageReason, PageStatus

SITE_ROOT = Path(__file__).parent.parent / "fixtures" / "site"

LARGE_BODY_BYTES = 25 * 1024 * 1024  # above the 20 MiB CRAWL_MAX_PAGE_BYTES default
FLAKY_FAILURES = 2  # /flaky answers 503 this many times after each reset()
SLOW_DEFAULT_SECONDS = 5.0
METADATA_URL = "http://169.254.169.254/latest/meta-data/"

_TEMPLATED_SUFFIXES = {".html", ".xml", ".txt"}

# ---- sitemap edge cases, served (unlinked) at /sitemaps/<name>; Step 31 tests ---- #
SITEMAP_BOMB_BYTES = 4 * 1024 * 1024  # inflated size of /sitemaps/bomb.xml.gz
_SM_NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'


def _urlset(*locs: str, namespace: bool = True) -> str:
    entries = "".join(f"<url><loc>{loc}</loc></url>" for loc in locs)
    return f'<?xml version="1.0"?><urlset {_SM_NS if namespace else ""}>{entries}</urlset>'


def _sitemap_index(*locs: str) -> str:
    entries = "".join(f"<sitemap><loc>{loc}</loc></sitemap>" for loc in locs)
    return f'<?xml version="1.0"?><sitemapindex {_SM_NS}>{entries}</sitemapindex>'


SITEMAP_EDGE_CASES: dict[str, str] = {
    # index listing itself (loop) and one child
    "loop-index.xml": _sitemap_index(
        "{{BASE}}/sitemaps/loop-index.xml", "{{BASE}}/sitemaps/mixed.xml"
    ),
    # no namespace; lastmod variants; duplicate, invalid and non-http locs
    "mixed.xml": (
        '<?xml version="1.0"?><urlset>'
        "<url><loc> {{BASE}}/terms.html </loc><lastmod>2024-05-01</lastmod></url>"
        "<url><loc>{{BASE}}/privacy.html</loc><lastmod>2024-05-01T10:30:00+02:00</lastmod></url>"
        "<url><loc>{{BASE}}/terms.html?utm_source=x</loc><lastmod>yesterday</lastmod></url>"
        "<url><loc>mailto:support@acme.example</loc></url>"
        "<url><loc>http://</loc></url>"
        "<url><lastmod>2024-01-01</lastmod></url>"
        "</urlset>"
    ),
    "nested-1.xml": _sitemap_index("{{BASE}}/sitemaps/nested-2.xml"),
    "nested-2.xml": _sitemap_index("{{BASE}}/sitemaps/nested-3.xml"),
    "nested-3.xml": _urlset("{{BASE}}/docs/"),
    # children on hosts outside a 127.0.0.1 crawl, and one robots.txt disallows
    "foreign-index.xml": _sitemap_index(
        "https://offsite.example/sitemap.xml",
        "{{OTHER_HOST}}/sitemaps/mixed.xml",
        "{{BASE}}/private/sitemap.xml",
        "{{BASE}}/sitemaps/nested-3.xml",
    ),
    "urls.txt": "{{BASE}}/about.html\n\n{{BASE}}/blog/post-2.html\nhttps://offsite.example/x\n",
    "feed.xml": '<?xml version="1.0"?><rss version="2.0"><channel><title>t</title></channel></rss>',
    "broken.xml": '<?xml version="1.0"?><urlset><url><loc>{{BASE}}/about.html</loc></url>',
    "laughs.xml": (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>'
        "<urlset><url><loc>&lol2;</loc></url></urlset>"
    ),
    "soft-404.html": "<!doctype html><html><body>Not here</body></html>",
}
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".xml": "application/xml",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
# served without a charset so the encoding must be detected from <meta>
_NO_CHARSET = {"latin1.html": "text/html"}


class Expect(StrEnum):
    """What a correct crawl concludes about a URL."""

    HTML = "html"  # fetched and extracted as HTML
    JS_RENDERED = "js_rendered"  # empty without JavaScript; must be rendered
    DOCUMENT = "document"  # PDF/DOCX/PPTX/XLSX extracted
    DOCUMENT_UNREADABLE = "document_unreadable"  # no text layer / encrypted
    REDIRECT = "redirect"  # followed to an in-scope target
    REDIRECT_LOOP = "redirect_loop"
    BLOCKED_SSRF = "blocked_ssrf"  # redirect into a forbidden network
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"  # permanent 5xx after retries
    RETRY_THEN_OK = "retry_then_ok"
    RATE_LIMITED = "rate_limited"  # 429 on every attempt
    TOO_LARGE = "too_large"
    UNSUPPORTED_TYPE = "unsupported_type"
    DUPLICATE = "duplicate"  # same content as another URL
    CANONICAL_ELSEWHERE = "canonical_elsewhere"
    ROBOTS_DISALLOWED = "robots_disallowed"
    OUT_OF_SCOPE = "out_of_scope"  # other host / off-site / non-http scheme
    NOFOLLOW = "nofollow"  # only reachable through rel=nofollow
    SITEMAP_ONLY = "sitemap_only"  # not linked; discoverable via sitemap
    TIMEOUT = "timeout"  # slower than the read timeout


# path (relative to BASE unless absolute) → expected outcome
SITE_PAGES: dict[str, Expect] = {
    "/": Expect.HTML,
    "/about.html": Expect.HTML,
    "/privacy.html": Expect.HTML,
    "/terms.html": Expect.HTML,
    "/docs/": Expect.HTML,
    "/docs/getting-started.html": Expect.HTML,
    "/docs/configuration.html": Expect.HTML,
    "/docs/api/reference.html": Expect.HTML,
    "/docs/api/deep/level4.html": Expect.HTML,
    "/docs/api/deep/level5.html": Expect.HTML,
    "/blog/": Expect.HTML,
    "/blog/post-1.html": Expect.HTML,
    "/blog/post-2.html": Expect.HTML,
    "/latin1.html": Expect.HTML,
    "/orphan.html": Expect.SITEMAP_ONLY,
    "/app/": Expect.JS_RENDERED,
    "/app/advanced": Expect.HTML,  # only linked from JS-rendered content
    "/files/widget-spec.pdf": Expect.DOCUMENT,
    "/files/install-guide.docx": Expect.DOCUMENT,
    "/files/roadmap.pptx": Expect.DOCUMENT,
    "/files/pricing.xlsx": Expect.DOCUMENT,
    "/files/scanned.pdf": Expect.DOCUMENT_UNREADABLE,
    "/files/encrypted.pdf": Expect.DOCUMENT_UNREADABLE,
    "/redirect-me": Expect.REDIRECT,
    "/redirect-chain": Expect.REDIRECT,
    "/loop-a": Expect.REDIRECT_LOOP,
    "/redirect-metadata": Expect.BLOCKED_SSRF,
    "/missing.html": Expect.NOT_FOUND,
    "/error": Expect.SERVER_ERROR,
    "/flaky": Expect.RETRY_THEN_OK,
    "/rate-limited": Expect.RATE_LIMITED,
    "/slow": Expect.TIMEOUT,
    "/large.html": Expect.TOO_LARGE,
    "/files/huge-catalog.pdf": Expect.TOO_LARGE,  # declares its size up front
    "/api/widgets": Expect.UNSUPPORTED_TYPE,  # no extension; JSON body
    "/data.json": Expect.OUT_OF_SCOPE,  # asset extension: skipped before fetching
    "/images/logo.png": Expect.OUT_OF_SCOPE,
    "/dup-a.html": Expect.HTML,
    "/dup-b.html": Expect.DUPLICATE,  # whichever of dup-a/dup-b is fetched second
    "/canonical-variant.html": Expect.CANONICAL_ELSEWHERE,
    "/private/secret.html": Expect.ROBOTS_DISALLOWED,
    "/nofollow-target.html": Expect.NOFOLLOW,
    "{{OTHER_HOST}}/docs/": Expect.OUT_OF_SCOPE,
    "https://offsite.example/page": Expect.OUT_OF_SCOPE,
    "https://offsite.example/from-sitemap": Expect.OUT_OF_SCOPE,
    "mailto:support@acme.example": Expect.OUT_OF_SCOPE,
    "tel:+15550100": Expect.OUT_OF_SCOPE,
    "javascript:void(0)": Expect.OUT_OF_SCOPE,
}


# how each expectation is recorded on a `Page` (status, reason)
EXPECTED_PAGE_OUTCOME: dict[Expect, tuple[PageStatus, PageReason | None]] = {
    Expect.HTML: (PageStatus.SUCCEEDED, None),
    Expect.JS_RENDERED: (PageStatus.SUCCEEDED, None),
    Expect.DOCUMENT: (PageStatus.SUCCEEDED, None),
    Expect.DOCUMENT_UNREADABLE: (PageStatus.FAILED, PageReason.DOCUMENT_UNREADABLE),
    Expect.REDIRECT: (PageStatus.SUCCEEDED, None),
    Expect.REDIRECT_LOOP: (PageStatus.FAILED, PageReason.REDIRECT_LOOP),
    Expect.BLOCKED_SSRF: (PageStatus.FAILED, PageReason.SSRF_BLOCKED),
    Expect.NOT_FOUND: (PageStatus.FAILED, PageReason.HTTP_CLIENT_ERROR),
    Expect.SERVER_ERROR: (PageStatus.FAILED, PageReason.HTTP_SERVER_ERROR),
    Expect.RETRY_THEN_OK: (PageStatus.SUCCEEDED, None),
    Expect.RATE_LIMITED: (PageStatus.FAILED, PageReason.RATE_LIMITED),
    Expect.TOO_LARGE: (PageStatus.FAILED, PageReason.TOO_LARGE),
    Expect.UNSUPPORTED_TYPE: (PageStatus.SKIPPED, PageReason.UNSUPPORTED_CONTENT_TYPE),
    Expect.DUPLICATE: (PageStatus.DUPLICATE, None),
    Expect.CANONICAL_ELSEWHERE: (PageStatus.SKIPPED, PageReason.CANONICAL_ELSEWHERE),
    Expect.ROBOTS_DISALLOWED: (PageStatus.SKIPPED, PageReason.ROBOTS_DISALLOWED),
    Expect.OUT_OF_SCOPE: (PageStatus.SKIPPED, PageReason.OUT_OF_SCOPE),
    Expect.NOFOLLOW: (PageStatus.SKIPPED, PageReason.NOFOLLOW),
    Expect.SITEMAP_ONLY: (PageStatus.SUCCEEDED, None),
    Expect.TIMEOUT: (PageStatus.FAILED, PageReason.TIMEOUT),
}


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    method: str
    host: str
    path: str  # includes the query string
    user_agent: str | None
    at: float  # time.monotonic()


@dataclass
class _State:
    requests: list[RecordedRequest] = field(default_factory=list)
    flaky_hits: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _base_urls(request: Request) -> tuple[str, str]:
    port = request.url.port
    return f"http://127.0.0.1:{port}", f"http://localhost:{port}"


def _render(request: Request, raw: bytes) -> bytes:
    base, other = _base_urls(request)
    return raw.replace(b"{{BASE}}", base.encode()).replace(b"{{OTHER_HOST}}", other.encode())


def _html(title: str, body: str) -> str:
    return f"<!doctype html><html><head><title>{title}</title></head><body>{body}</body></html>"


def build_app(state: _State) -> Starlette:
    async def record(request: Request) -> None:
        query = f"?{request.url.query}" if request.url.query else ""
        with state.lock:
            state.requests.append(
                RecordedRequest(
                    method=request.method,
                    host=request.headers.get("host", ""),
                    path=request.url.path + query,
                    user_agent=request.headers.get("user-agent"),
                    at=time.monotonic(),
                )
            )

    def redirect(target: str, status: int):
        async def endpoint(request: Request) -> Response:
            await record(request)
            return RedirectResponse(target, status_code=status)

        return endpoint

    async def redirect_to(request: Request) -> Response:
        """Open redirect (unlinked): lets tests point a hop at any host."""
        await record(request)
        return RedirectResponse(request.query_params["url"], status_code=302)

    async def error(request: Request) -> Response:
        await record(request)
        return HTMLResponse(_html("Error", "<h1>Internal error</h1>"), status_code=500)

    async def flaky(request: Request) -> Response:
        await record(request)
        with state.lock:
            state.flaky_hits += 1
            failing = state.flaky_hits <= FLAKY_FAILURES
        if failing:
            return HTMLResponse(
                _html("Unavailable", "<h1>Try again</h1>"),
                status_code=503,
                headers={"Retry-After": "1"},
            )
        return HTMLResponse(
            _html("Flaky page", "<h1>Status board</h1><p>All widget lines operational.</p>")
        )

    async def rate_limited(request: Request) -> Response:
        await record(request)
        return PlainTextResponse("slow down", status_code=429, headers={"Retry-After": "2"})

    async def slow(request: Request) -> Response:
        await record(request)
        await asyncio.sleep(float(request.query_params.get("delay", SLOW_DEFAULT_SECONDS)))
        return HTMLResponse(_html("Slow", "<h1>Finally</h1>"))

    async def _chunks() -> AsyncIterator[bytes]:
        chunk = b"<p>" + b"widget " * 9_000 + b"</p>\n"
        sent = 0
        yield b"<!doctype html><html><head><title>Huge</title></head><body>"
        while sent < LARGE_BODY_BYTES:
            yield chunk
            sent += len(chunk)
        yield b"</body></html>"

    async def large_html(request: Request) -> Response:
        await record(request)
        # no Content-Length: the size limit must be enforced while streaming
        return StreamingResponse(_chunks(), media_type="text/html; charset=utf-8")

    async def huge_pdf(request: Request) -> Response:
        await record(request)

        async def zeros() -> AsyncIterator[bytes]:
            block = b"\0" * (1024 * 1024)
            for _ in range(LARGE_BODY_BYTES // len(block)):
                yield block

        return StreamingResponse(
            zeros(),
            media_type="application/pdf",
            headers={"Content-Length": str(LARGE_BODY_BYTES)},
        )

    # ---- unlinked routes for fetcher tests (not part of SITE_PAGES) ---- #

    async def gzip_page(request: Request) -> Response:
        await record(request)
        html = _html("Compressed", "<h1>Compressed page</h1><p>Served with gzip.</p>").encode()
        return Response(
            gzip.compress(html, mtime=0),
            headers={"Content-Type": "text/html; charset=utf-8", "Content-Encoding": "gzip"},
        )

    async def gzip_bomb(request: Request) -> Response:
        await record(request)
        # ~25 KiB on the wire, 25 MiB once decoded
        bomb = gzip.compress(b"<p>" + b"A" * LARGE_BODY_BYTES, compresslevel=9, mtime=0)
        return Response(bomb, headers={"Content-Type": "text/html", "Content-Encoding": "gzip"})

    async def untyped_pdf(request: Request) -> Response:
        await record(request)
        response = Response((SITE_ROOT / "files" / "widget-spec.pdf").read_bytes())
        if request.path_params["kind"] == "octet":
            response.headers["Content-Type"] = "application/octet-stream"
        elif "content-type" in response.headers:
            del response.headers["content-type"]
        return response

    async def bom_page(request: Request) -> Response:
        await record(request)
        html = _html("BOM", "<h1>Café</h1><p>naïve façade</p>").encode("utf-8")
        return Response(b"\xef\xbb\xbf" + html, headers={"Content-Type": "text/html"})

    async def drip(request: Request) -> Response:
        """Sends a few bytes every 0.2 s forever: never trips a read timeout."""
        await record(request)

        async def chunks() -> AsyncIterator[bytes]:
            yield b"<!doctype html><html><body>"
            while True:
                await asyncio.sleep(0.2)
                yield b"<p>.</p>"

        return StreamingResponse(chunks(), media_type="text/html")

    async def api_widgets(request: Request) -> Response:
        await record(request)
        return Response(b'{"widgets": ["WX-2048", "WX-4096"]}', media_type="application/json")

    async def gz_sitemap(request: Request) -> Response:
        await record(request)
        raw = _render(request, (SITE_ROOT / "sitemap-docs.xml").read_bytes())
        return Response(gzip.compress(raw, mtime=0), media_type="application/gzip")

    async def sitemap_edge(request: Request) -> Response:
        await record(request)
        name = request.path_params["name"]
        if name == "big.xml":  # ?n=<count> generated page URLs
            base, _ = _base_urls(request)
            count = int(request.query_params.get("n", "10"))
            locs = (f"{base}/generated/page-{i}.html" for i in range(count))
            return Response(_urlset(*locs).encode(), media_type="application/xml")
        if name == "bomb.xml.gz":
            payload = b"<urlset>" + b" " * SITEMAP_BOMB_BYTES + b"</urlset>"
            return Response(
                gzip.compress(payload, compresslevel=9, mtime=0), media_type="application/gzip"
            )
        if name == "mixed.xml.gz":
            raw = _render(request, SITEMAP_EDGE_CASES["mixed.xml"].encode())
            return Response(gzip.compress(raw, mtime=0), media_type="application/x-gzip")
        if name not in SITEMAP_EDGE_CASES:
            return HTMLResponse(_html("Not found", "<h1>Not found</h1>"), status_code=404)
        body = _render(request, SITEMAP_EDGE_CASES[name].encode())
        media = (
            "text/html"
            if name.endswith(".html")
            else "text/plain"
            if name.endswith(".txt")
            else "application/xml"
        )
        return Response(body, media_type=media)

    async def app_advanced(request: Request) -> Response:
        await record(request)
        return HTMLResponse(
            _html(
                "Advanced options",
                "<h1>Advanced options</h1><p>Firmware channels: stable, beta and nightly.</p>",
            )
        )

    async def static(request: Request) -> Response:
        await record(request)
        rel = request.path_params["path"]
        target = (SITE_ROOT / rel).resolve()
        if not target.is_relative_to(SITE_ROOT.resolve()):
            return HTMLResponse(_html("Not found", "<h1>Not found</h1>"), status_code=404)
        if target.is_dir():
            if rel and not rel.endswith("/"):
                return RedirectResponse(f"/{rel}/", status_code=301)
            target = target / "index.html"
        if not target.is_file():
            return HTMLResponse(_html("Not found", "<h1>Not found</h1>"), status_code=404)

        body = target.read_bytes()
        if target.suffix in _TEMPLATED_SUFFIXES:
            body = _render(request, body)
        relative = target.relative_to(SITE_ROOT.resolve()).as_posix()
        media_type = (
            _NO_CHARSET.get(relative)
            or _CONTENT_TYPES.get(target.suffix)
            or mimetypes.guess_type(target.name)[0]
            or "application/octet-stream"
        )
        return Response(body, headers={"Content-Type": media_type})

    routes = [
        Route("/redirect-me", redirect("/docs/getting-started.html", 301)),
        Route("/redirect-chain", redirect("/redirect-me", 302)),
        Route("/loop-a", redirect("/loop-b", 302)),
        Route("/loop-b", redirect("/loop-a", 302)),
        Route("/redirect-metadata", redirect(METADATA_URL, 302)),
        Route("/redirect-to", redirect_to),
        Route("/error", error),
        Route("/flaky", flaky),
        Route("/rate-limited", rate_limited),
        Route("/slow", slow),
        Route("/large.html", large_html),
        Route("/files/huge-catalog.pdf", huge_pdf),
        Route("/api/widgets", api_widgets),
        Route("/gzip-page", gzip_page),
        Route("/gzip-bomb", gzip_bomb),
        Route("/files/{kind:str}/widget-spec.pdf", untyped_pdf),
        Route("/bom-page", bom_page),
        Route("/drip", drip),
        Route("/sitemap-docs.xml.gz", gz_sitemap),
        Route("/sitemaps/{name:str}", sitemap_edge),
        Route("/app/advanced", app_advanced),
        Route("/{path:path}", static),
    ]
    return Starlette(routes=routes)


class FixtureSite:
    """Runs the fixture app with uvicorn on a free loopback port in a daemon thread."""

    def __init__(self) -> None:
        self._state = _State()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self.port: int = self._sock.getsockname()[1]
        config = uvicorn.Config(
            build_app(self._state),
            log_config=None,
            access_log=False,
            lifespan="off",
            timeout_graceful_shutdown=1,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._sock]}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def other_host_url(self) -> str:
        """Same server under a different hostname: out of scope for a 127.0.0.1 crawl."""
        return f"http://localhost:{self.port}"

    def url(self, path: str) -> str:
        if path.startswith("{{OTHER_HOST}}"):
            return self.other_host_url + path.removeprefix("{{OTHER_HOST}}")
        if "://" in path or ":" in path.split("/", 1)[0]:
            return path  # absolute or non-http scheme (mailto:, tel:, javascript:)
        return self.base_url + path

    @property
    def requests(self) -> list[RecordedRequest]:
        with self._state.lock:
            return list(self._state.requests)

    def hits(self, path: str) -> int:
        return sum(1 for r in self.requests if r.path == path)

    def reset(self) -> None:
        """Forget recorded requests and re-arm /flaky."""
        with self._state.lock:
            self._state.requests.clear()
            self._state.flaky_hits = 0

    def start(self) -> FixtureSite:
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("fixture site failed to start")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
        self._sock.close()
