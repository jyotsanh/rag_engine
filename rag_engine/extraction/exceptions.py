"""Extraction failures. Each carries the `PageReason` (and page status) it is recorded as.

These describe a page's outcome during a crawl; they never reach an API client, so they
are plain exceptions like the fetch errors, not `AppBaseException`s.
"""

from rag_engine.domain import PageReason, PageStatus

_SKIP_REASONS = frozenset({PageReason.UNSUPPORTED_CONTENT_TYPE, PageReason.EMPTY_CONTENT})


class ExtractionError(Exception):
    """The extractor could not turn the bytes into a document (corrupt file, parser bug)."""

    reason: PageReason = PageReason.EXTRACTION_FAILED

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = (detail or message)[:500]

    @property
    def page_status(self) -> PageStatus:
        return PageStatus.SKIPPED if self.reason in _SKIP_REASONS else PageStatus.FAILED


class UnsupportedContentError(ExtractionError):
    """No extractor handles this content: the page is skipped, not failed."""

    reason = PageReason.UNSUPPORTED_CONTENT_TYPE


class DocumentUnreadableError(ExtractionError):
    """A supported format we cannot read: encrypted, no text layer, legacy binary Office."""

    reason = PageReason.DOCUMENT_UNREADABLE


class DocumentTooLargeError(ExtractionError):
    """Over the per-format size cap (bytes); page-count caps truncate instead."""

    reason = PageReason.TOO_LARGE


class EmptyContentError(ExtractionError):
    """Readable, but there is no text worth indexing."""

    reason = PageReason.EMPTY_CONTENT


class ExtractorRegistrationError(ValueError):
    """Programming error: a kind registered twice, or an extractor misdeclared."""
