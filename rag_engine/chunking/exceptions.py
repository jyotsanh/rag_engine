"""Chunking failures. These are bugs or misconfiguration, never a property of the page."""


class ChunkingError(Exception):
    """Base for chunking errors."""


class UnknownChunkerError(ChunkingError):
    """`CHUNKER` names a strategy with no registered implementation."""


class ChunkerRegistrationError(ChunkingError):
    """A strategy was registered twice, or with a class that is not a `BaseChunker`."""


class ChunkTooLargeError(ChunkingError):
    """A chunker produced a chunk over `CHUNK_MAX_TOKENS`: its splitter is broken."""

    def __init__(self, *, url: str, ordinal: int, token_count: int, max_tokens: int) -> None:
        super().__init__(
            f"{url} chunk {ordinal}: {token_count} tokens exceeds the {max_tokens}-token cap"
        )
        self.url = url
        self.ordinal = ordinal
        self.token_count = token_count
        self.max_tokens = max_tokens


class TokenizerError(ChunkingError):
    """The tokenizer returned an impossible count (e.g. 0 for non-empty text)."""
