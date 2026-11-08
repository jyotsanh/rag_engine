"""Embedder contract: texts in, one vector per text out (Step 61).

A provider implementation subclasses `BaseEmbedder` and implements `_embed(texts,
input_type)` only, returning vectors plus usage. The final `embed_documents` /
`embed_query` guarantee for every provider:

- an empty batch costs nothing (no provider call);
- every input is non-empty and within the model's per-input token limit
  (`EmbeddingInputError` names the offending index before anything is sent);
- the response has **one vector per input, in input order, of the space's dimensions, all
  finite** (`EmbeddingResponseError` otherwise), so a misbehaving provider can never store
  misaligned or wrong-sized vectors;
- usage (requests, input tokens) is reported with every result, for job/tenant metering
  (Step 65).

An embedder is bound to one `EmbeddingSpace` (model + dimensions): a knowledge base's
vectors are only comparable within its space. Batching by input count / token budget,
retries and caching are layered on by Steps 62-64.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol

from rag_engine.config import EmbeddingProvider, Settings, settings
from rag_engine.embeddings.exceptions import (
    EmbeddingInputError,
    EmbeddingResponseError,
    UnsupportedEmbeddingProviderError,
)
from rag_engine.embeddings.models import EmbeddingModelSpec, EmbeddingSpace
from rag_engine.embeddings.tokenizer import tokenizer_for

type Vector = list[float]


class InputType(StrEnum):
    """Some providers embed documents and queries differently; OpenAI ignores it."""

    DOCUMENT = "document"
    QUERY = "query"


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class EmbeddingUsageCounts:
    requests: int = 0
    tokens: int = 0  # input tokens billed by the provider

    def __add__(self, other: EmbeddingUsageCounts) -> EmbeddingUsageCounts:
        return EmbeddingUsageCounts(self.requests + other.requests, self.tokens + other.tokens)


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """What `_embed` returns: vectors in input order, plus what the call(s) cost."""

    vectors: list[Vector]
    usage: EmbeddingUsageCounts


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: list[Vector]
    space: EmbeddingSpace
    usage: EmbeddingUsageCounts = field(default_factory=EmbeddingUsageCounts)

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def vector(self) -> Vector:
        """The only vector of a single-text batch (a query)."""
        if len(self.vectors) != 1:
            raise ValueError(f"batch has {len(self.vectors)} vectors, not 1")
        return self.vectors[0]


class BaseEmbedder(ABC):
    provider: ClassVar[EmbeddingProvider]

    def __init__(
        self,
        space: EmbeddingSpace,
        *,
        config: Settings = settings,
        tokenizer: TokenCounter | None = None,
    ) -> None:
        if space.spec.provider is not self.provider:
            raise UnsupportedEmbeddingProviderError(
                f"{space.model} is a {space.spec.provider} model, not {self.provider}",
                details={"model": space.model.value, "provider": str(self.provider)},
            )
        self.space = space
        self.config = config
        self._tokenizer: TokenCounter = tokenizer or tokenizer_for(space.model)

    @property
    def spec(self) -> EmbeddingModelSpec:
        return self.space.spec

    @property
    def dimensions(self) -> int:
        return self.space.dimensions

    # ---- the contract ---- #

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Vectors for chunks' `embed_text`, in input order."""
        return await self._run(list(texts), InputType.DOCUMENT)

    async def embed_query(self, text: str) -> EmbeddingBatch:
        """One vector for a search query (`.vector`)."""
        return await self._run([text], InputType.QUERY)

    async def aclose(self) -> None:  # noqa: B027 (an optional hook, not abstract)
        """Release network clients (implementations override)."""

    async def __aenter__(self) -> BaseEmbedder:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @abstractmethod
    async def _embed(self, texts: list[str], input_type: InputType) -> ProviderResult:
        """Call the provider for already-validated, non-empty `texts`."""

    # ---- checks ---- #

    async def _run(self, texts: list[str], input_type: InputType) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch([], self.space)
        self._validate(texts)
        result = await self._embed(texts, input_type)
        self._check(result, len(texts))
        return EmbeddingBatch(result.vectors, self.space, result.usage)

    def _validate(self, texts: list[str]) -> None:
        limit = self.spec.max_input_tokens
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingInputError("Cannot embed an empty text", details={"index": index})
            tokens = self._tokenizer.count(text)
            if tokens > limit:
                raise EmbeddingInputError(
                    f"Text {index} has {tokens} tokens; {self.space.model} accepts {limit}",
                    details={"index": index, "tokens": tokens, "max_tokens": limit},
                )

    def _check(self, result: ProviderResult, expected: int) -> None:
        details = {"model": self.space.model.value, "provider": str(self.provider)}
        if len(result.vectors) != expected:
            raise EmbeddingResponseError(
                f"expected {expected} vectors, got {len(result.vectors)}",
                details={**details, "expected": expected, "received": len(result.vectors)},
            )
        for index, vector in enumerate(result.vectors):
            if len(vector) != self.dimensions:
                raise EmbeddingResponseError(
                    f"vector {index} has {len(vector)} dimensions, expected {self.dimensions}",
                    details={**details, "index": index, "dimensions": len(vector)},
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingResponseError(
                    f"vector {index} contains non-finite values",
                    details={**details, "index": index},
                )
