"""Auto fetcher: plain HTTP first, a headless-browser re-render only when the page needs it.

1. `HttpFetcher.fetch(url)`. Its typed `FetchError`s propagate unchanged, so the retry
   policy (Step 28) applies as usual.
2. Non-HTML content, `CRAWL_JS_RENDERING=never`, or no browser context → the HTTP result.
3. HTML: the JS-shell detector (Step 33) runs, and its signals are recorded on the result.
   With `auto`, only flagged pages are rendered; with `always`, every HTML page is.
4. Render in the job's browser context (SSRF-enforced, Step 35): load, then network idle
   (capped), then auto-scroll for lazy content (capped), then idle again.
5. Keep the better of the two:
   - rendering failed, or the rendered document is not 2xx → HTTP result, `render_note` says why
   - the rendered DOM has less visible text than the raw HTML (e.g. a consent wall replaced
     the page) → HTTP result, `render_note="render_not_better"`
   - otherwise → the rendered HTML, `rendered_with=browser`

A render failure never fails the page: the HTTP body is still a valid, if thinner, result.
"""

from dataclasses import replace

from loguru import logger

from rag_engine.config import JsRenderMode, Settings, settings
from rag_engine.crawling.fetchers.browser import BrowserJobContext, RenderedPage, RenderError
from rag_engine.crawling.fetchers.http import FetchResult, HttpFetcher
from rag_engine.crawling.js_shell import JsShellDetector, JsShellVerdict
from rag_engine.domain import ContentKind, RenderedWith


class AutoFetcher:
    def __init__(
        self,
        http: HttpFetcher,
        browser: BrowserJobContext | None,
        *,
        config: Settings = settings,
        mode: JsRenderMode | None = None,
        detector: JsShellDetector | None = None,
    ) -> None:
        self._http = http
        self._browser = browser
        self._mode = mode or config.CRAWL_JS_RENDERING
        self._detector = detector or JsShellDetector()
        self._idle_timeout = config.CRAWL_RENDER_NETWORKIDLE_SECONDS
        self._scroll_seconds = config.CRAWL_RENDER_SCROLL_SECONDS
        self._scroll_steps = config.CRAWL_RENDER_SCROLL_MAX_STEPS
        if self._browser is None and self._mode is not JsRenderMode.NEVER:
            logger.warning("no browser available: JS-rendered pages will be fetched over HTTP")

    async def fetch(self, url: str) -> FetchResult:
        result = await self._http.fetch(url)
        if result.content_kind is not ContentKind.HTML or self._mode is JsRenderMode.NEVER:
            return result

        verdict = self._detector.detect(result.text or "")
        result = replace(result, js_signals=tuple(str(s) for s in verdict.signals))
        if self._browser is None:
            return result
        if self._mode is JsRenderMode.AUTO and not verdict.needs_rendering:
            return result

        try:
            rendered = await self._browser.render(
                result.final_url,
                idle_timeout=self._idle_timeout,
                scroll_seconds=self._scroll_seconds,
                scroll_max_steps=self._scroll_steps,
            )
        except RenderError as exc:
            logger.bind(url=result.final_url).warning("render failed, keeping HTTP: {}", exc)
            return replace(result, render_note=f"render_failed: {exc.detail}"[:200])
        return self._better(result, verdict, rendered)

    def _better(
        self, result: FetchResult, verdict: JsShellVerdict, rendered: RenderedPage
    ) -> FetchResult:
        status = rendered.status_code
        if status is not None and not 200 <= status < 300:
            return replace(result, render_note=f"render_status_{status}")
        rendered_text = self._detector.detect(rendered.html).text_chars
        if rendered_text < verdict.text_chars:
            return replace(result, render_note="render_not_better")

        html_bytes = rendered.html.encode("utf-8")
        chain = result.redirect_chain
        if rendered.final_url != result.final_url:  # e.g. a JS redirect during render
            chain = (*chain, rendered.final_url)
        logger.bind(url=result.final_url).debug(
            "rendered: {} -> {} visible chars, {} scroll steps, idle={}",
            verdict.text_chars,
            rendered_text,
            rendered.scroll_steps,
            rendered.network_idle,
        )
        return replace(
            result,
            final_url=rendered.final_url,
            redirect_chain=chain,
            body=html_bytes,
            text=rendered.html,
            encoding="utf-8",
            mime_type="text/html",
            bytes_downloaded=result.bytes_downloaded + len(html_bytes),
            elapsed_ms=result.elapsed_ms + rendered.elapsed_ms,
            rendered_with=RenderedWith.BROWSER,
        )
