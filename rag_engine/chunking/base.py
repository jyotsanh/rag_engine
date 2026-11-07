"""Chunker contract: a page's Markdown in, ordered `ChunkDraft`s out (Step 55).

A `BaseChunker` subclass only decides **where to cut** (`split` → `ChunkPiece`s with their
heading path). Everything every strategy must get right lives in the final `chunk()`:

- blank lines around a piece are trimmed (indentation is kept: code) and empty pieces dropped;
- ordinals are dense, in document order, from 0;
- `embed_text` = bounded context (page title + heading path, Step 58) + the text, from one
  hook (`embed_text`); `text` stays clean for display;
- `token_count` is counted on `embed_text`, the string the embedder receives, with the
  model's tokenizer (`Tokenizer`, tiktoken in Step 57), so embedding batches can be packed
  by token budget (Step 62) without recounting;
- no chunk exceeds `CHUNK_MAX_TOKENS` (`ChunkTooLargeError`: a splitter bug, never silently
  truncated) and a non-empty chunk never counts 0 tokens (`TokenizerError`);
- `content_hash = sha256(embed_text)`: identical embedded text, identical hash (re-embedding
  and the embedding cache key on it, Step 64);
- heading paths are capped at `MAX_HEADING_DEPTH` levels (the outermost are kept).

`ChunkDraft.to_document(source)` turns a draft into the persisted `Chunk` (Step 59).
"""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Protocol

from rag_engine.chunking.context import build_context, compose_embed_text
from rag_engine.chunking.exceptions import ChunkTooLargeError, TokenizerError
from rag_engine.config import Settings, settings
from rag_engine.domain import Chunk, ContentKind, Page, PageStatus

MAX_HEADING_DEPTH = 12  # `Chunk.heading_path` limit
DEFAULT_CONTEXT_TOKENS = 100  # mirrors the CHUNK_CONTEXT_MAX_TOKENS default


class Tokenizer(Protocol):
    """Counts tokens the way the embedding model does (tiktoken encoding, Step 57)."""

    @property
    def name(self) -> str: ...

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int  # the splitter's aim for a chunk's text (Step 57)
    max_tokens: int  # hard cap on a chunk's embedded text
    overlap_tokens: int  # carried over between consecutive prose chunks (Step 57)
    # cap on the embed-text context (title + heading path, Step 58); None: the default,
    # min(DEFAULT_CONTEXT_TOKENS, max_tokens - target_tokens)
    context_tokens: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.target_tokens <= self.max_tokens:
            raise ValueError("need 0 < target_tokens <= max_tokens")
        if not 0 <= self.overlap_tokens < self.target_tokens:
            raise ValueError("need 0 <= overlap_tokens < target_tokens")
        room = self.max_tokens - self.target_tokens
        if self.context_tokens is None:
            object.__setattr__(self, "context_tokens", min(DEFAULT_CONTEXT_TOKENS, room))
        elif not 0 <= self.context_tokens <= room:
            raise ValueError("need 0 <= context_tokens <= max_tokens - target_tokens")

    @property
    def context_budget(self) -> int:
        return self.context_tokens or 0

    @classmethod
    def from_settings(cls, config: Settings = settings) -> ChunkingConfig:
        return cls(
            target_tokens=config.CHUNK_TARGET_TOKENS,
            max_tokens=config.CHUNK_MAX_TOKENS,
            overlap_tokens=config.CHUNK_OVERLAP_TOKENS,
            context_tokens=config.CHUNK_CONTEXT_MAX_TOKENS,
        )


@dataclass(frozen=True, slots=True)
class ChunkInput:
    """One page to chunk, with the ids its chunks are stored under."""

    tenant_id: str
    kb_id: str
    source_id: str
    version: str  # the job that crawled the page (= the vector version it writes)
    page_id: str
    url: str
    markdown: str
    content_kind: ContentKind
    title: str | None = None
    lang: str | None = None

    @classmethod
    def from_page(cls, page: Page, *, markdown: str | None = None) -> ChunkInput:
        """A succeeded page; `markdown` overrides the stored text (e.g. after the job-level
        boilerplate strip, Step 40)."""
        if page.status is not PageStatus.SUCCEEDED or page.markdown is None:
            raise ValueError(f"only succeeded pages are chunked ({page.url} is {page.status})")
        if page.content_kind is None:
            raise ValueError(f"{page.url} has no content_kind")
        return cls(
            tenant_id=page.tenant_id,
            kb_id=page.kb_id,
            source_id=page.source_id,
            version=page.job_id,
            page_id=page.id,
            url=page.url,
            markdown=page.markdown if markdown is None else markdown,
            content_kind=page.content_kind,
            title=page.title,
            lang=page.lang,
        )


@dataclass(frozen=True, slots=True)
class ChunkPiece:
    """Where a subclass cut: the text plus its place in the document."""

    text: str
    heading_path: tuple[str, ...] = ()
    page_number: int | None = None  # PDF page the piece starts on (Step 56)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    text: str  # clean, for display
    embed_text: str  # what gets embedded
    heading_path: tuple[str, ...]
    token_count: int  # tokens of embed_text
    content_hash: str  # sha256(embed_text)
    page_number: int | None = None

    def to_document(self, source: ChunkInput) -> Chunk:
        return Chunk(
            tenant_id=source.tenant_id,
            kb_id=source.kb_id,
            source_id=source.source_id,
            version=source.version,
            page_id=source.page_id,
            url=source.url,
            title=source.title,
            heading_path=list(self.heading_path),
            ordinal=self.ordinal,
            text=self.text,
            embed_text=self.embed_text,
            token_count=self.token_count,
            content_hash=self.content_hash,
            page_number=self.page_number,
        )


def content_hash(embed_text: str) -> str:
    return hashlib.sha256(embed_text.encode("utf-8")).hexdigest()


def _trim(text: str) -> str:
    """Drop blank lines around the text and trailing whitespace; keep indentation."""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).rstrip()


class BaseChunker(ABC):
    name: ClassVar[str]

    def __init__(self, tokenizer: Tokenizer, config: ChunkingConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

    @abstractmethod
    def split(self, source: ChunkInput) -> Iterable[ChunkPiece]:
        """Cut the page into pieces, in document order. Empty pieces are dropped later."""

    def context(self, source: ChunkInput, heading_path: tuple[str, ...]) -> str:
        """The bounded title + heading-path lines put in front of a chunk's text."""
        return build_context(
            source.title,
            heading_path,
            tokenizer=self.tokenizer,
            max_tokens=self.config.context_budget,
        )

    def embed_text(self, source: ChunkInput, piece: ChunkPiece, text: str) -> str:
        """The string the embedder receives for a piece: context + blank line + text."""
        return compose_embed_text(self.context(source, piece.heading_path), text)

    def chunk(self, source: ChunkInput) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for piece in self.split(source):
            text = _trim(piece.text)
            if not text:
                continue
            embed_text = self.embed_text(source, piece, text)
            tokens = self.tokenizer.count(embed_text)
            ordinal = len(drafts)
            if tokens <= 0:
                raise TokenizerError(
                    f"{self.tokenizer.name} counted {tokens} tokens for chunk {ordinal} of "
                    f"{source.url}"
                )
            if tokens > self.config.max_tokens:
                raise ChunkTooLargeError(
                    url=source.url,
                    ordinal=ordinal,
                    token_count=tokens,
                    max_tokens=self.config.max_tokens,
                )
            drafts.append(
                ChunkDraft(
                    ordinal=ordinal,
                    text=text,
                    embed_text=embed_text,
                    heading_path=tuple(piece.heading_path[:MAX_HEADING_DEPTH]),
                    token_count=tokens,
                    content_hash=content_hash(embed_text),
                    page_number=piece.page_number,
                )
            )
        return drafts
