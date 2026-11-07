"""HTML → Markdown: main content via trafilatura, markdownify as the fallback.

1. trafilatura extracts the main content (navigation, footers, cookie banners and
   sidebars dropped) as Markdown with headings, lists, tables, inline code and code blocks.
   `favor_recall` and a low minimum size keep short documentation pages.
2. If it finds nothing (or crashes), markdownify converts the whole `<body>` minus
   obvious chrome (`nav`, `header`, `footer`, `aside`, forms, scripts…). Cross-page
   boilerplate that survives is removed later (Step 40).
3. Post-processing: a dropped `<h1>` is restored as the top heading, code fences get their
   language back from the source
   `<pre><code class="language-…">`, intra-word `\\_` escapes are undone, blank lines are
   normalized around headings and fences.

`<noscript>` and `<template>` are dropped first: a rendered page never shows them, and in
an unrendered JS shell the noscript text would otherwise be the only "content".

Metadata: `title` (trafilatura's cleaned title, else `<title>`, else the first `<h1>`),
`description` (meta description, else `og:description`), `lang` (`<html lang>`, else
`Content-Language`, else the `http-equiv` meta; primary subtag only). Author, site name
and publication date (from markup only, no guessing from body text) go to `metadata`.

Outlinks are **not** collected here: HTML links need the crawl scope and nofollow rules,
which `LinkExtractor` (Step 32) applies to the same HTML.
"""

import re
from typing import Any
from urllib.parse import urlsplit

import lxml.html
from lxml import etree
from markdownify import MarkdownConverter
from trafilatura import extract as trafilatura_extract
from trafilatura import extract_metadata
from trafilatura.settings import use_config

from rag_engine.config import logger
from rag_engine.crawling.fetchers.content import decode_html
from rag_engine.domain import ContentKind
from rag_engine.extraction.base import BaseExtractor, ExtractedDocument, ExtractionInput
from rag_engine.extraction.exceptions import EmptyContentError

_TRAFILATURA_CONFIG = use_config()
_TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")  # signal-based: main thread only
_TRAFILATURA_CONFIG.set("DEFAULT", "MIN_EXTRACTED_SIZE", "40")  # short docs pages are content
_TRAFILATURA_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "1")
_TRAFILATURA_CONFIG.set("DEFAULT", "EXTENSIVE_DATE_SEARCH", "off")  # no dates guessed from text

# dropped before the markdownify fallback converts the body
_CHROME_TAGS = (
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "embed",
    "nav", "header", "footer", "aside", "form", "button", "input", "select", "textarea",
)  # fmt: skip
_LANG = re.compile(r"^([a-z]{2,3})(?:[-_][a-z0-9]{1,8})*$", re.I)
_CODE_CLASS = re.compile(r"(?:^|\s)(?:language|lang|highlight-source|highlight)-([\w+#.-]+)", re.I)
_FENCE = re.compile(r"^```[^\n]*\n(.*?)\n?^```[ \t]*$", re.S | re.M)
_INTRAWORD_ESCAPED_UNDERSCORE = re.compile(r"(?<=\w)\\_(?=\w)")
_BLANK_RUNS = re.compile(r"\n{3,}")


class _Markdown(MarkdownConverter):
    """markdownify with ATX headings, `-` bullets and code-block languages from classes."""

    def __init__(self, **options: Any) -> None:
        super().__init__(
            heading_style="ATX",
            bullets="-",
            code_language_callback=_code_language_of,
            strip=["img"],
            **options,
        )


def _code_language_of(element: Any) -> str | None:
    code = element.find("code") if element.name == "pre" else element
    classes = " ".join((code or element).get("class") or [])
    match = _CODE_CLASS.search(classes)
    return match.group(1).lower() if match else None


class HtmlExtractor(BaseExtractor):
    name = "html"
    kind = ContentKind.HTML
    mime_types = frozenset({"text/html", "application/xhtml+xml"})
    extensions = frozenset({".html", ".htm", ".xhtml"})

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        html = source.text
        if html is None:  # e.g. HTML sniffed from a body labelled as something else
            html, _ = decode_html(source.body, None)
        return await self.run_sync(self._extract, html, source)

    def _extract(self, html: str, source: ExtractionInput) -> ExtractedDocument:
        root = _parse(html)
        content_html = html
        if root is not None:
            # <noscript>/<template> never show in a rendered page; in a raw JS shell the
            # noscript text ("enable JavaScript…") would otherwise become the "content"
            for element in list(root.iter("noscript", "template")):
                element.drop_tree()
            content_html = etree.tostring(root, encoding="unicode", method="html")
        markdown, method = self._main_content(content_html, source.url, root)
        if not markdown.strip():
            raise EmptyContentError("no text content in the HTML", detail="no_text")
        if root is not None:
            markdown = _restore_code_languages(markdown, root)
            markdown = _ensure_main_heading(markdown, root)
        markdown = _tidy(markdown)

        meta = _trafilatura_metadata(html, source.url)
        if meta.get("sitename", "").lower() == (urlsplit(source.url).hostname or ""):
            meta.pop("sitename")  # trafilatura's fallback guess from the URL, not markup
        title = meta.get("title") or (_first_text(root, "title") if root is not None else None)
        if not title and root is not None:
            title = _first_text(root, "h1")
        return ExtractedDocument(
            markdown=markdown,
            content_kind=ContentKind.HTML,
            extractor=self.name,
            title=title,
            lang=_language(root, source.headers.get("content-language")),
            description=_description(root) or meta.get("description"),
            metadata={
                "extraction": method,
                "author": meta.get("author"),
                "site_name": meta.get("sitename"),
                "published": meta.get("date"),
            },
        )

    @staticmethod
    def _main_content(html: str, url: str, root: lxml.html.HtmlElement | None) -> tuple[str, str]:
        try:
            extracted = trafilatura_extract(
                html,
                url=url,
                output_format="markdown",
                include_tables=True,
                include_formatting=True,
                include_links=False,
                include_images=False,
                include_comments=False,
                favor_recall=True,
                deduplicate=False,
                config=_TRAFILATURA_CONFIG,
            )
        except Exception as exc:  # a trafilatura bug must not lose the page: fall back
            logger.bind(url=url).warning("trafilatura failed ({}): using markdownify", exc)
            extracted = None
        if extracted and extracted.strip():
            return extracted, "trafilatura"
        return _markdownify_body(root), "markdownify"


def _parse(html: str) -> lxml.html.HtmlElement | None:
    if not html.strip():
        return None
    # bytes + explicit encoding: lxml refuses str carrying an XML encoding declaration
    parser = lxml.html.HTMLParser(encoding="utf-8")
    try:
        return lxml.html.document_fromstring(html.encode("utf-8", errors="replace"), parser=parser)
    except etree.ParserError, ValueError:
        return None


def _markdownify_body(root: lxml.html.HtmlElement | None) -> str:
    if root is None:
        return ""
    body = root.find("body")
    if body is None:
        body = root
    for element in list(body.iter(*_CHROME_TAGS)):
        parent = element.getparent()
        if parent is not None:
            element.drop_tree()
    for comment in body.xpath("//comment()"):
        comment.drop_tree()
    html = etree.tostring(body, encoding="unicode", method="html")
    return _Markdown().convert(html)


def _restore_code_languages(markdown: str, root: lxml.html.HtmlElement) -> str:
    """Give fenced blocks their language back, matching the source `<pre>` by content."""
    languages: dict[str, str] = {}
    for pre in root.iter("pre"):
        code = pre.find("code")
        classes = " ".join(
            filter(None, [pre.get("class"), code.get("class") if code is not None else None])
        )
        match = _CODE_CLASS.search(classes)
        if match:
            languages.setdefault(pre.text_content().strip(), match.group(1).lower())
    if not languages:
        return markdown

    def relabel(fence: re.Match[str]) -> str:
        opening = fence.group(0).split("\n", 1)[0]
        body = fence.group(1)
        language = languages.get(body.strip())
        if opening.strip() != "```" or language is None:
            return fence.group(0)  # already labelled, or no match
        return f"```{language}\n{body}\n```"

    return _FENCE.sub(relabel, markdown)


def _ensure_main_heading(markdown: str, root: lxml.html.HtmlElement) -> str:
    """trafilatura sometimes drops a page's only `<h1>` (it treats it as the title); headings
    drive chunking, so put it back on top when its text is missing entirely."""
    heading = _first_text(root, "h1")
    if heading is None or heading in markdown:
        return markdown
    return f"# {heading}\n\n{markdown}"


def _tidy(markdown: str) -> str:
    markdown = _INTRAWORD_ESCAPED_UNDERSCORE.sub("_", markdown)
    lines: list[str] = []
    in_fence = False
    for line in markdown.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence and lines and lines[-1].strip():
                lines.append("")  # blank line before an opening fence
            lines.append(line.rstrip())
            if in_fence:
                lines.append("")  # and after a closing one
            in_fence = not in_fence
            continue
        if not in_fence and stripped.startswith("#") and lines and lines[-1].strip():
            lines.append("")  # blank line before a heading
        lines.append(line if in_fence else line.rstrip())
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip() + "\n"


def _first_text(root: lxml.html.HtmlElement, tag: str) -> str | None:
    for element in root.iter(tag):
        text = " ".join(element.text_content().split())
        if text:
            return text
    return None


def _meta_content(
    root: lxml.html.HtmlElement, *, name: str | None = None, prop: str | None = None
) -> str | None:
    for meta in root.iter("meta"):
        if name and (meta.get("name") or "").strip().lower() == name:
            return (meta.get("content") or "").strip() or None
        if prop and (meta.get("property") or "").strip().lower() == prop:
            return (meta.get("content") or "").strip() or None
    return None


def _description(root: lxml.html.HtmlElement | None) -> str | None:
    if root is None:
        return None
    return _meta_content(root, name="description") or _meta_content(root, prop="og:description")


def _language(root: lxml.html.HtmlElement | None, header: str | None) -> str | None:
    candidates: list[str | None] = []
    if root is not None:
        # the HTML parser keeps `xml:lang` as a literal attribute name
        candidates.append(root.get("lang") or root.get("xml:lang"))
    candidates.append(header.split(",")[0] if header else None)
    if root is not None:
        for meta in root.iter("meta"):
            if (meta.get("http-equiv") or "").strip().lower() == "content-language":
                candidates.append((meta.get("content") or "").split(",")[0])
    for value in candidates:
        match = _LANG.match((value or "").strip())
        if match:
            return match.group(1).lower()
    return None


def _trafilatura_metadata(html: str, url: str) -> dict[str, str]:
    try:
        document = extract_metadata(html, default_url=url, extensive=False)
    except Exception:  # metadata is a bonus; never fail the page over it
        return {}
    if document is None:
        return {}
    fields = ("title", "description", "author", "sitename", "date")
    return {
        f: v.strip()
        for f in fields
        if isinstance(v := getattr(document, f, None), str) and v.strip()
    }
