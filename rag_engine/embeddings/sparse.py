"""Sparse (lexical) embeddings for hybrid search (Step 66).

Dense vectors capture meaning; sparse BM25 vectors catch exact terms (model numbers like
"WX-2048", error codes, names) that dense retrieval blurs. Qdrant fuses both (Step 72).

- **`SparseVector`**: sorted, unique uint32 `indices` (hashed terms) with positive finite
  `values`. A text made only of stopwords is a valid *empty* vector.
- **`BaseSparseEmbedder`**: subclasses implement the synchronous `_embed_documents` /
  `_embed_query`. The final async `embed_documents` / `embed_query` validate inputs (non-
  empty strings), make no model call for an empty batch, run the CPU-bound model in a
  worker thread (never on the event loop), and check one well-formed vector per input.
- **`FastembedBm25Embedder`** (`SparseEmbedderType.BM25`, fastembed `Qdrant/bm25`):
  documents get BM25 term weights (term frequency saturated by document length); queries
  get weight 1 per term. **IDF is applied by Qdrant** at query time (the collection's
  sparse vector uses the IDF modifier, Step 68), so corpus statistics never go stale as
  sources change. Stemming and stopwords follow `SPARSE_EMBEDDING_LANGUAGE`.
- The model loads lazily, once per process (`get_sparse_embedder` caches the instance).
  Its files are downloaded on first use into `FASTEMBED_CACHE_DIR`; images bake them in
  at build time (`python -m rag_engine.embeddings.download`, Step 95) and run with
  `HF_HUB_OFFLINE=1`.
"""

import asyncio
import math
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar

from rag_engine.config import Settings, SparseEmbedderType, settings
from rag_engine.embeddings.exceptions import EmbeddingInputError, EmbeddingResponseError

_UINT32 = 2**32


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")
        if any(not 0 <= i < _UINT32 for i in self.indices):
            raise ValueError("sparse indices must be uint32")
        if any(b <= a for a, b in zip(self.indices, self.indices[1:], strict=False)):
            raise ValueError("sparse indices must be sorted and unique")
        if any(not (math.isfinite(v) and v > 0) for v in self.values):
            raise ValueError("sparse values must be finite and positive")

    @classmethod
    def from_pairs(cls, indices: Sequence[int], values: Sequence[float]) -> SparseVector:
        """Build from unsorted (index, value) pairs, summing duplicates, dropping zeros."""
        merged: dict[int, float] = {}
        for index, value in zip(indices, values, strict=True):
            merged[int(index)] = merged.get(int(index), 0.0) + float(value)
        ordered = sorted((i, v) for i, v in merged.items() if v > 0)
        return cls(tuple(i for i, _ in ordered), tuple(v for _, v in ordered))

    def __len__(self) -> int:
        return len(self.indices)


class BaseSparseEmbedder(ABC):
    kind: ClassVar[SparseEmbedderType]

    def __init__(self, *, config: Settings = settings) -> None:
        self.config = config

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        texts = list(texts)
        if not texts:
            return []
        _validate(texts)
        vectors = await asyncio.to_thread(self._embed_documents, texts)
        self._check(vectors, len(texts))
        return vectors

    async def embed_query(self, text: str) -> SparseVector:
        _validate([text])
        vector = await asyncio.to_thread(self._embed_query, text)
        self._check([vector], 1)
        return vector

    def _check(self, vectors: Sequence[object], expected: int) -> None:
        if len(vectors) != expected or not all(isinstance(v, SparseVector) for v in vectors):
            raise EmbeddingResponseError(
                f"{self.model_name}: expected {expected} sparse vectors, got {len(vectors)}",
                details={"model": self.model_name, "expected": expected},
            )

    @abstractmethod
    def _embed_documents(self, texts: list[str]) -> list[SparseVector]: ...

    @abstractmethod
    def _embed_query(self, text: str) -> SparseVector: ...


def _validate(texts: list[str]) -> None:
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError("Cannot embed an empty text", details={"index": index})


# ---- fastembed BM25 ---- #


class FastembedBm25Embedder(BaseSparseEmbedder):
    kind = SparseEmbedderType.BM25

    def __init__(
        self,
        *,
        config: Settings = settings,
        loader: Callable[[], Any] | None = None,
    ) -> None:
        """`loader` builds the fastembed model (tests inject one); default: from settings."""
        super().__init__(config=config)
        self._loader = loader or self._load
        self._model: Any = None
        self._lock = threading.Lock()  # one load even when threads race on first use

    @property
    def model_name(self) -> str:
        return self.config.SPARSE_EMBEDDING_MODEL

    def _load(self) -> Any:
        from fastembed import SparseTextEmbedding  # heavy import, only when actually used

        return SparseTextEmbedding(
            self.model_name,
            cache_dir=self.config.FASTEMBED_CACHE_DIR,
            language=self.config.SPARSE_EMBEDDING_LANGUAGE,
        )

    def model(self) -> Any:
        """The loaded fastembed model (downloads its files on first use if not cached)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._loader()
        return self._model

    def _embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [_convert(e) for e in self.model().embed(texts)]

    def _embed_query(self, text: str) -> SparseVector:
        return _convert(next(iter(self.model().query_embed(text))))


def _convert(embedding: Any) -> SparseVector:
    """fastembed's numpy `SparseEmbedding` → `SparseVector`."""
    return SparseVector.from_pairs(embedding.indices.tolist(), embedding.values.tolist())


# ---- dispatcher ---- #

_SPARSE_EMBEDDERS: dict[SparseEmbedderType, type[BaseSparseEmbedder]] = {
    SparseEmbedderType.BM25: FastembedBm25Embedder,
}


@cache
def _shared(kind: SparseEmbedderType, model: str, language: str, cache_dir: str | None) -> Any:
    return _SPARSE_EMBEDDERS[kind](config=settings)


def get_sparse_embedder(
    kind: SparseEmbedderType | None = None, *, config: Settings = settings
) -> BaseSparseEmbedder:
    """The process-wide sparse embedder (the model is loaded once, on first use)."""
    kind = kind or config.SPARSE_EMBEDDER
    if kind not in _SPARSE_EMBEDDERS:
        raise ValueError(f"no sparse embedder for {str(kind)!r}")
    if config is not settings:
        return _SPARSE_EMBEDDERS[kind](config=config)
    return _shared(
        kind,
        config.SPARSE_EMBEDDING_MODEL,
        config.SPARSE_EMBEDDING_LANGUAGE,
        config.FASTEMBED_CACHE_DIR,
    )


def download_model(config: Settings = settings) -> str:
    """Fetch the sparse model's files into the cache (image build, Step 95); returns its name."""
    embedder = get_sparse_embedder(config=config)
    if isinstance(embedder, FastembedBm25Embedder):
        embedder.model()
    return embedder.model_name
