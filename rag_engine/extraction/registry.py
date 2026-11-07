"""Pick the extractor for a fetched resource, and run it safely.

Resolution order (bytes beat labels, labels beat guesses):
1. **Magic bytes** proving a binary format (PDF, DOCX, PPTX, XLSX): that kind, whatever the
   `Content-Type` says. Servers label PDFs `text/html` and Office files `octet-stream`.
2. Bytes that look like **HTML**: HTML, even if labelled `application/pdf` (the classic
   "PDF link that returns a login page").
3. **OLE2 container** where an Office format was expected (by MIME or extension): an
   encrypted OOXML file or a legacy `.doc/.xls/.ppt` → `DocumentUnreadableError`.
4. Nothing proven: a `text/html` label (or `.html` URL on a generic type) with a body that
   is text → HTML (plain-text bodies served as HTML still have words to index).
5. Otherwise → `UnsupportedContentError` (page skipped), including a body that
   contradicts its declared binary type (a "PDF" without `%PDF-`).

`extract()` adds a time cap (`EXTRACTION_TIMEOUT_SECONDS`) and turns unexpected parser
exceptions into `ExtractionError` so one malformed file never crashes the crawl.
"""

import asyncio
from collections.abc import Iterable
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from loguru import logger

from rag_engine.config import Settings, settings
from rag_engine.crawling.fetchers.content import EXTENSION_KINDS, GENERIC_MIME_TYPES, MIME_KINDS
from rag_engine.domain import ContentKind
from rag_engine.extraction.base import BaseExtractor, ExtractedDocument, ExtractionInput
from rag_engine.extraction.docx import DocxExtractor
from rag_engine.extraction.exceptions import (
    DocumentUnreadableError,
    ExtractionError,
    ExtractorRegistrationError,
    UnsupportedContentError,
)
from rag_engine.extraction.html import HtmlExtractor
from rag_engine.extraction.pdf import PdfExtractor
from rag_engine.extraction.pptx import PptxExtractor
from rag_engine.extraction.sniff import Sniffed, looks_like_text, sniff
from rag_engine.extraction.xlsx import XlsxExtractor

_OFFICE_KINDS = frozenset({ContentKind.DOCX, ContentKind.PPTX, ContentKind.XLSX})
_BINARY_KINDS = _OFFICE_KINDS | {ContentKind.PDF}


class ExtractorRegistry:
    def __init__(
        self, extractors: Iterable[BaseExtractor] = (), *, config: Settings = settings
    ) -> None:
        self._by_kind: dict[ContentKind, BaseExtractor] = {}
        self._mime_kinds: dict[str, ContentKind] = dict(MIME_KINDS)
        self._extension_kinds: dict[str, ContentKind] = dict(EXTENSION_KINDS)
        self._timeout = config.EXTRACTION_TIMEOUT_SECONDS
        for extractor in extractors:
            self.register(extractor)

    def register(self, extractor: BaseExtractor) -> None:
        if extractor.kind in self._by_kind:
            existing = self._by_kind[extractor.kind].name
            raise ExtractorRegistrationError(
                f"{extractor.kind} already handled by {existing!r}; cannot add {extractor.name!r}"
            )
        self._by_kind[extractor.kind] = extractor
        # an extractor may teach the registry extra labels for its kind
        for mime in extractor.mime_types:
            self._mime_kinds.setdefault(mime, extractor.kind)
        for extension in extractor.extensions:
            self._extension_kinds.setdefault(extension, extractor.kind)

    @property
    def kinds(self) -> frozenset[ContentKind]:
        return frozenset(self._by_kind)

    def detect_kind(self, source: ExtractionInput) -> ContentKind:
        """The content kind of `source` (see module docstring), or a typed error."""
        sniffed = sniff(source.body)
        declared = self._mime_kinds.get(source.mime_type)
        suffix = PurePosixPath(urlsplit(source.url).path).suffix.lower()
        by_extension = self._extension_kinds.get(suffix)
        expected = declared or (by_extension if source.mime_type in GENERIC_MIME_TYPES else None)

        if isinstance(sniffed, ContentKind):
            return sniffed
        if sniffed is Sniffed.OLE and (expected in _OFFICE_KINDS or by_extension in _OFFICE_KINDS):
            raise DocumentUnreadableError(
                "Office file in an OLE container: encrypted, or a legacy binary format",
                detail="encrypted_or_legacy_office",
            )
        if expected is ContentKind.HTML and (
            looks_like_text(source.body) or (source.encoding or "").startswith("utf-16")
        ):
            return ContentKind.HTML
        if expected in _BINARY_KINDS:
            raise UnsupportedContentError(
                f"body does not look like the declared {expected}",
                detail=f"not_{expected}",
            )
        label = source.mime_type or "missing content-type"
        detail = f"{label} ({sniffed})" if sniffed is not None else label
        raise UnsupportedContentError(f"unsupported content: {detail}", detail=detail)

    def resolve(self, source: ExtractionInput) -> BaseExtractor:
        kind = self.detect_kind(source)
        extractor = self._by_kind.get(kind)
        if extractor is None:
            raise UnsupportedContentError(
                f"no extractor registered for {kind}", detail=f"no_extractor:{kind}"
            )
        return extractor

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        extractor = self.resolve(source)
        try:
            async with asyncio.timeout(self._timeout):
                return await extractor.extract(source)
        except ExtractionError:
            raise
        except TimeoutError as exc:
            # a thread running a parser cannot be killed; it finishes in the background
            raise ExtractionError(
                f"{extractor.name} exceeded {self._timeout}s", detail="timeout"
            ) from exc
        except Exception as exc:  # parser bugs and corrupt files must not crash the crawl
            logger.bind(url=source.url, extractor=extractor.name).exception("extractor crashed")
            raise ExtractionError(
                f"{extractor.name} failed: {exc}", detail=f"{type(exc).__name__}: {exc}"
            ) from exc


def default_registry(config: Settings = settings) -> ExtractorRegistry:
    """Every built-in extractor: HTML, PDF, DOCX, PPTX, XLSX."""
    return ExtractorRegistry(
        [
            HtmlExtractor(),
            PdfExtractor(config),
            DocxExtractor(config),
            PptxExtractor(config),
            XlsxExtractor(config),
        ],
        config=config,
    )
