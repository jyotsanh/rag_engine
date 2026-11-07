"""Chunker factory: `ChunkerType` (settings `CHUNKER`) → `BaseChunker` implementation.

Implementations register themselves with `@register_chunker(ChunkerType.X)` in their module;
the structure chunker (Steps 56-58) is `ChunkerType.STRUCTURE`. `get_chunker` builds one
with the model's tokenizer and the configured token budget.
"""

from collections.abc import Callable

from rag_engine.chunking.base import BaseChunker, ChunkingConfig, Tokenizer
from rag_engine.chunking.exceptions import ChunkerRegistrationError, UnknownChunkerError
from rag_engine.config import ChunkerType, Settings, settings

_CHUNKERS: dict[ChunkerType, type[BaseChunker]] = {}


def register_chunker[C: type[BaseChunker]](kind: ChunkerType) -> Callable[[C], C]:
    def register(cls: C) -> C:
        if not (isinstance(cls, type) and issubclass(cls, BaseChunker)):
            raise ChunkerRegistrationError(f"{cls!r} is not a BaseChunker subclass")
        if kind in _CHUNKERS and _CHUNKERS[kind] is not cls:
            raise ChunkerRegistrationError(
                f"chunker {str(kind)!r} is already registered ({_CHUNKERS[kind].__name__})"
            )
        _CHUNKERS[kind] = cls
        return cls

    return register


def registered_chunkers() -> dict[ChunkerType, type[BaseChunker]]:
    return dict(_CHUNKERS)


def get_chunker(
    tokenizer: Tokenizer,
    kind: ChunkerType | None = None,
    *,
    config: Settings = settings,
    chunking: ChunkingConfig | None = None,
) -> BaseChunker:
    """The `kind` chunker (default: `CHUNKER`) using `tokenizer` and the token budget."""
    kind = kind or config.CHUNKER
    cls = _CHUNKERS.get(kind)
    if cls is None:
        known = ", ".join(sorted(_CHUNKERS)) or "none"
        raise UnknownChunkerError(f"no chunker registered for {str(kind)!r} (registered: {known})")
    return cls(tokenizer, chunking or ChunkingConfig.from_settings(config))
