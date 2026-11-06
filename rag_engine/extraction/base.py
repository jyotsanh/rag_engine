"""Extractor contract: `ExtractionInput` in, `ExtractedDocument` out.

Every content family (HTML, PDF, DOCX, PPTX, XLSX) has one `BaseExtractor` subclass,
registered by `ContentKind` in the `ExtractorRegistry`. An extractor gets the fetched
bytes (and, for HTML, the decoded text) and returns Markdown plus metadata. Failures are
typed `ExtractionError`s that say how the page is recorded.

Parsers are CPU-bound and synchronous; `run_sync` moves them off the event loop so one
large PDF never stalls the crawl's other fetches.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from rag_engine.crawling.fetchers import FetchResult
from rag_engine.domain import ContentKind

type MetadataValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ExtractionInput:
    """The fetched resource, independent of how it was fetched."""

    url: str  # final URL (after redirects / JS navigation)
    mime_type: str  # from Content-Type, lower-case, no parameters ("" if missing)
    body: bytes = field(repr=False)
    text: str | None = field(default=None, repr=False)  # decoded HTML when the fetcher has it
    encoding: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_fetch(cls, result: FetchResult) -> ExtractionInput:
        return cls(
            url=result.final_url,
            mime_type=result.mime_type,
            body=result.body,
            text=result.text,
            encoding=result.encoding,
            headers=dict(result.headers),
        )


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    markdown: str
    content_kind: ContentKind
    extractor: str  # name of the extractor that produced it
    title: str | None = None
    lang: str | None = None  # BCP 47 primary tag when known, e.g. "en"
    description: str | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    outlinks: tuple[str, ...] = ()  # absolute URLs found in the document, not yet scoped

    def __post_init__(self) -> None:
        # normalize the optional text fields so "" and whitespace mean "absent"
        for name in ("title", "lang", "description"):
            value = getattr(self, name)
            object.__setattr__(self, name, value.strip() or None if value else None)
        if self.lang:
            object.__setattr__(self, "lang", self.lang.lower())
        for key, value in self.metadata.items():
            if not isinstance(value, str | int | float | bool | None):
                raise TypeError(f"metadata[{key!r}] must be a scalar, got {type(value).__name__}")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "outlinks", tuple(self.outlinks))


class BaseExtractor(ABC):
    """One content family. Subclasses declare what they handle as class attributes."""

    name: ClassVar[str]
    kind: ClassVar[ContentKind]
    mime_types: ClassVar[frozenset[str]] = frozenset()
    extensions: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return  # intermediate abstract bases may leave the declarations to subclasses
        for attr in ("name", "kind"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must declare `{attr}`")
        if not isinstance(cls.kind, ContentKind):
            raise TypeError(f"{cls.__name__}.kind must be a ContentKind")
        if any(not ext.startswith(".") or ext != ext.lower() for ext in cls.extensions):
            raise TypeError(f"{cls.__name__}.extensions must be lower-case and start with '.'")

    @abstractmethod
    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        """Markdown and metadata for `source`, or a typed `ExtractionError`."""

    @staticmethod
    async def run_sync[T](func: Callable[..., T], /, *args: object) -> T:
        """Run a blocking parser in a worker thread."""
        return await asyncio.to_thread(func, *args)
