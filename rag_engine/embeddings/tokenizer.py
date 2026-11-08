"""Token counting with the embedding model's own encoding (tiktoken), Step 57.

Lives with the embedding models (moved from `chunking` in Step 61): the chunker sizes
chunks with it and the embedder validates inputs with it; `rag_engine.chunking`
re-exports it.

`tokenizer_for(model)` returns the shared `TiktokenTokenizer` for the model's
`EmbeddingModelSpec.tokenizer_encoding` (`cl100k_base` for the OpenAI 3-series), so chunk
sizes are measured exactly as the embedding API will count them.

tiktoken downloads an encoding's BPE file on first use and caches it on disk
(`TIKTOKEN_CACHE_DIR`, default: the system temp dir). Production images bake the cache in
(Step 95), so workers never download at runtime.
"""

from functools import cache

import tiktoken

from rag_engine.config import EmbeddingModelName
from rag_engine.embeddings.models import get_model_spec


class TiktokenTokenizer:
    """A `Tokenizer` over one tiktoken encoding; thread-safe and cheap to share."""

    def __init__(self, encoding_name: str) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name

    def count(self, text: str) -> int:
        # special-token text (e.g. "<|endoftext|>" on a page) is counted as ordinary text,
        # exactly as the embeddings API treats input
        return len(self._encoding.encode(text, disallowed_special=()))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    def __repr__(self) -> str:
        return f"TiktokenTokenizer({self.name!r})"


@cache
def _encoding_tokenizer(encoding_name: str) -> TiktokenTokenizer:
    return TiktokenTokenizer(encoding_name)


def tokenizer_for(model: EmbeddingModelName | str) -> TiktokenTokenizer:
    """The tokenizer the embedding model counts with (one instance per encoding)."""
    return _encoding_tokenizer(get_model_spec(model).tokenizer_encoding)
