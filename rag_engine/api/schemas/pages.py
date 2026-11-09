"""The per-page report (Step 86): what happened to every URL a crawl job saw."""

from typing import Any

from pydantic import Field

from rag_engine.domain import Page, PageSummary


class PageResponse(PageSummary):
    """A page in the list: its outcome and fetch facts, without the extracted Markdown."""

    id: str  # the API field (the domain reads Mongo's `_id`)

    @classmethod
    def from_summary(cls, summary: PageSummary) -> PageResponse:
        return cls.model_validate(summary.model_dump())


class PageDetailResponse(PageResponse):
    """One page in full: the extracted Markdown and metadata, and how many chunks it gave."""

    description: str | None = None
    doc_metadata: dict[str, Any] = Field(default_factory=dict)
    extractor: str | None = None
    js_signals: list[str] = Field(default_factory=list)
    render_note: str | None = None
    content_hash: str | None = None
    markdown: str | None = Field(
        default=None, description="The extracted, cleaned Markdown (before chunking)."
    )
    chunks: int = Field(description="Chunks this page produced for this job's version.")

    @classmethod
    def of(cls, page: Page, *, chunks: int) -> PageDetailResponse:
        values = page.model_dump()
        values["markdown_chars"] = len(page.markdown or "")
        return cls.model_validate({**values, "chunks": chunks})
