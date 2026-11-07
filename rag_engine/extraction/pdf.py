"""PDF → Markdown with pymupdf4llm (PyMuPDF; AGPL accepted, see the Decisions table).

- **Caps**: bodies over `PDF_MAX_BYTES` fail as `too_large`; only the first
  `PDF_MAX_PAGES` pages are extracted (`metadata.truncated` says so; not a failure).
- **Unreadable** (`document_unreadable`, recorded on the page):
  - `encrypted`: a user password is required (owner-password-only PDFs open normally)
  - `no_text_layer`: no extractable text on any page, i.e. a scan without OCR. OCR is
    deliberately off: slow, heavy, and out of scope.
- Corrupt files are `extraction_failed`.
- **Markdown**: pymupdf4llm's layout engine (headings, lists, tables as Markdown tables),
  running page headers and footers dropped, images ignored. Each page starts with a
  marker line `<!-- page N -->` so chunks can cite their page (Step 56 reads it). If
  pymupdf4llm fails on a file, the plain text layer is used instead
  (`metadata.extraction="pymupdf_text"`).
- **Metadata**: `title` (document info, else the first heading), `lang` (catalog `/Lang`),
  `description` (subject), plus author, creation date, page counts.
- **Outlinks**: http(s) URI link annotations, for the engine to scope.

PyMuPDF must not be used from several threads at once, and extraction runs in worker
threads (`run_sync`), so a process-wide lock serializes PDF parsing. Crawl workers
parallelize across processes.
"""

import re
import threading
from datetime import UTC, datetime
from typing import Any

import pymupdf
import pymupdf4llm

from rag_engine.config import Settings, logger, settings
from rag_engine.domain import ContentKind
from rag_engine.extraction.base import (
    BaseExtractor,
    ExtractedDocument,
    ExtractionInput,
    MetadataValue,
)
from rag_engine.extraction.exceptions import (
    DocumentTooLargeError,
    DocumentUnreadableError,
    EmptyContentError,
    ExtractionError,
)

# MuPDF reports recoverable damage on stderr by default; failures surface as exceptions
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

_PYMUPDF_LOCK = threading.Lock()
_PDF_DATE = re.compile(r"^D:(\d{4})(\d{2})?(\d{2})?")
_LANG = re.compile(r"^([a-z]{2,3})(?:[-_][a-z0-9]{1,8})*$", re.I)
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)
# document-info titles that are really file names or tool defaults
_JUNK_TITLE = re.compile(
    r"^(untitled|microsoft (word|powerpoint|excel) - .*|.*\.(docx?|pdf|pptx?))$", re.I
)
_TRAILING_SPACES = re.compile(r"[ \t]+$", re.M)
_BLANK_RUNS = re.compile(r"\n{3,}")


def page_marker(number: int) -> str:
    return f"<!-- page {number} -->"


class PdfExtractor(BaseExtractor):
    name = "pdf"
    kind = ContentKind.PDF
    mime_types = frozenset({"application/pdf", "application/x-pdf"})
    extensions = frozenset({".pdf"})

    def __init__(self, config: Settings = settings) -> None:
        self._max_bytes = config.PDF_MAX_BYTES
        self._max_pages = config.PDF_MAX_PAGES

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        if len(source.body) > self._max_bytes:
            raise DocumentTooLargeError(
                f"PDF is {len(source.body)} bytes, over {self._max_bytes}", detail="bytes"
            )
        return await self.run_sync(self._extract_locked, source)

    def _extract_locked(self, source: ExtractionInput) -> ExtractedDocument:
        with _PYMUPDF_LOCK:
            return self._extract(source)

    def _extract(self, source: ExtractionInput) -> ExtractedDocument:
        try:
            doc = pymupdf.open(stream=source.body, filetype="pdf")
        except Exception as exc:  # pymupdf raises FileDataError / RuntimeError variants
            raise ExtractionError(f"cannot open PDF: {exc}", detail="corrupt") from exc
        with doc:
            if doc.needs_pass and not doc.authenticate(""):  # owner-only PDFs open with ""
                raise DocumentUnreadableError("PDF needs a password", detail="encrypted")
            page_count = doc.page_count
            if page_count == 0:
                raise EmptyContentError("PDF has no pages", detail="no_pages")
            pages = list(range(min(page_count, self._max_pages)))
            if not any(doc[i].get_text("text").strip() for i in pages):
                raise DocumentUnreadableError(
                    "PDF has no text layer (scanned without OCR?)", detail="no_text_layer"
                )

            texts, method = self._page_texts(doc, pages, source.url)
            markdown = _assemble(texts)
            if not markdown:
                raise EmptyContentError("no text extracted from the PDF", detail="no_text")
            info = doc.metadata or {}
            title = (info.get("title") or "").strip()
            if not title or _JUNK_TITLE.match(title):
                heading = _HEADING.search(markdown)
                title = heading.group(1).strip() if heading else ""
            metadata: dict[str, MetadataValue] = {
                "extraction": method,
                "author": (info.get("author") or "").strip() or None,
                "created": _pdf_date(info.get("creationDate")),
                "page_count": page_count,
                "pages_extracted": len(pages),
                "truncated": len(pages) < page_count,
            }
            return ExtractedDocument(
                markdown=markdown,
                content_kind=ContentKind.PDF,
                extractor=self.name,
                title=title or None,
                lang=_catalog_language(doc),
                description=(info.get("subject") or "").strip() or None,
                metadata=metadata,
                outlinks=_uri_links(doc, pages),
            )

    @staticmethod
    def _page_texts(doc: Any, pages: list[int], url: str) -> tuple[list[tuple[int, str]], str]:
        try:
            chunks = pymupdf4llm.to_markdown(
                doc,
                pages=pages,
                page_chunks=True,
                use_ocr=False,
                header=False,  # running page headers / footers repeat on every page
                footer=False,
                write_images=False,
                embed_images=False,
                show_progress=False,
            )
            texts = [
                (int(chunk["metadata"].get("page_number") or i + 1), chunk["text"])
                for i, chunk in zip(pages, chunks, strict=False)
            ]
            return texts, "pymupdf4llm"
        except Exception as exc:  # layout engine failure: the text layer is still good
            logger.bind(url=url).warning("pymupdf4llm failed ({}): using the text layer", exc)
            return [(i + 1, doc[i].get_text("text")) for i in pages], "pymupdf_text"


def _assemble(texts: list[tuple[int, str]]) -> str:
    parts = []
    for number, text in texts:
        cleaned = _BLANK_RUNS.sub("\n\n", _TRAILING_SPACES.sub("", text)).strip()
        if cleaned:
            parts.append(f"{page_marker(number)}\n\n{cleaned}")
    return "\n\n".join(parts) + "\n" if parts else ""


def _pdf_date(value: str | None) -> str | None:
    """`D:20240314093000+01'00'` → `2024-03-14` (date precision is enough here)."""
    match = _PDF_DATE.match(value or "")
    if not match:
        return None
    year, month, day = match.group(1), match.group(2) or "01", match.group(3) or "01"
    try:
        return datetime(int(year), int(month), int(day), tzinfo=UTC).date().isoformat()
    except ValueError:
        return None


def _catalog_language(doc: Any) -> str | None:
    try:
        kind, value = doc.xref_get_key(doc.pdf_catalog(), "Lang")
    except Exception:  # not a PDF catalog we can read: language is optional
        return None
    if kind not in ("string", "name"):
        return None
    match = _LANG.match(value.strip().lstrip("/"))
    return match.group(1).lower() if match else None


def _uri_links(doc: Any, pages: list[int]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for i in pages:
        for link in doc[i].get_links():
            uri = (link.get("uri") or "").strip()
            if uri.lower().startswith(("http://", "https://")):
                seen.setdefault(uri, None)
    return tuple(seen)
