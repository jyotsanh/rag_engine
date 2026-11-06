"""Render self-check for a worker host/image: the full fetch path on one URL.

    python -m rag_engine.crawling.fetchers.smoke URL [--mode MODE] [--expect TEXT] [--allow-http]

HTTP fetch → JS-shell detector → headless Chromium (SSRF-enforced) → the better result,
exactly as a crawl does. Prints one JSON line. Exits 0 when the page was fetched, rendered
by the browser (unless `--allow-http`) and contains the `--expect` text; 1 otherwise.
Settings come from the environment as usual (secrets must be set);
`SSRF_ALLOW_PRIVATE_NETWORKS` decides whether private/loopback targets are reachable.
"""

import argparse
import asyncio
import json
import sys
from typing import Any

from rag_engine.config import JsRenderMode, settings
from rag_engine.crawling.fetchers.auto import AutoFetcher
from rag_engine.crawling.fetchers.browser import BrowserPool
from rag_engine.crawling.fetchers.errors import FetchError
from rag_engine.crawling.fetchers.http import HttpFetcher
from rag_engine.domain import RenderedWith
from rag_engine.security import SafeHttpClient

_TEXT_PREVIEW = 300


async def run(url: str, mode: JsRenderMode, expect: str | None = None) -> dict[str, Any]:
    pool = BrowserPool(settings)
    try:
        async with SafeHttpClient(settings) as client, pool.job_context(job_id="smoke") as job:
            fetcher = AutoFetcher(HttpFetcher(client, settings), job, mode=mode)
            try:
                result = await fetcher.fetch(url)
            except FetchError as exc:
                return {"ok": False, "url": url, "error": exc.reason, "detail": exc.detail}
            browser_version = (await pool.browser()).version
    finally:
        await pool.stop()
    return {
        "ok": True,
        "url": url,
        "final_url": result.final_url,
        "status": result.status_code,
        "rendered_with": result.rendered_with,
        "js_signals": list(result.js_signals),
        "render_note": result.render_note,
        "chromium": browser_version,
        "html_preview": (result.text or "")[:_TEXT_PREVIEW],
        "html_chars": len(result.text or ""),
        "expect_found": None if expect is None else expect in (result.text or ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument(
        "--mode", choices=[m.value for m in JsRenderMode], default=JsRenderMode.AUTO.value
    )
    parser.add_argument(
        "--allow-http", action="store_true", help="succeed even if the browser was not used"
    )
    parser.add_argument("--expect", help="fail unless the final HTML contains this text")
    args = parser.parse_args(argv)

    report = asyncio.run(run(args.url, JsRenderMode(args.mode), args.expect))
    print(json.dumps(report))
    rendered = report.get("rendered_with") == RenderedWith.BROWSER
    expected = report.get("expect_found") is not False
    return 0 if report["ok"] and (rendered or args.allow_http) and expected else 1


if __name__ == "__main__":
    sys.exit(main())
