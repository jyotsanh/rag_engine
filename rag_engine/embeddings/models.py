"""Static facts about every embedding model the service can use.

A knowledge base locks one `EmbeddingSpace` (model + dimensions) at creation.
Vectors from different spaces are never comparable, so the space key also
names the Qdrant collection that holds them (Step 68).
"""

from dataclasses import dataclass
from types import MappingProxyType

from rag_engine.config.enums import EmbeddingModelName, EmbeddingProvider
from rag_engine.embeddings.exceptions import (
    UnsupportedEmbeddingDimensionsError,
    UnsupportedEmbeddingModelError,
)

# Below this, shortened OpenAI embeddings lose too much retrieval quality to be worth storing.
MIN_REDUCED_DIMENSIONS = 256


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingModelSpec:
    name: EmbeddingModelName
    provider: EmbeddingProvider
    native_dimensions: int
    # Matryoshka-trained models can be truncated via the API's `dimensions` parameter
    supports_dimension_reduction: bool
    max_input_tokens: int  # per input text
    max_batch_inputs: int  # texts per request
    max_batch_tokens: int  # summed tokens per request
    tokenizer_encoding: str  # tiktoken encoding name used to count tokens

    @property
    def min_dimensions(self) -> int:
        if not self.supports_dimension_reduction:
            return self.native_dimensions
        return min(MIN_REDUCED_DIMENSIONS, self.native_dimensions)


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """A model at a fixed output size; the unit of vector compatibility."""

    model: EmbeddingModelName
    dimensions: int

    @property
    def key(self) -> str:
        """Stable identifier, safe for collection names: `text-embedding-3-small__1536`."""
        return f"{self.model.value}__{self.dimensions}"

    @property
    def spec(self) -> EmbeddingModelSpec:
        return get_model_spec(self.model)


# OpenAI limits: 8191 tokens per input, 2048 inputs and 300k summed tokens per request.
_OPENAI_LIMITS = {
    "provider": EmbeddingProvider.OPENAI,
    "supports_dimension_reduction": True,
    "max_input_tokens": 8191,
    "max_batch_inputs": 2048,
    "max_batch_tokens": 300_000,
    "tokenizer_encoding": "cl100k_base",
}

EMBEDDING_MODELS: MappingProxyType[EmbeddingModelName, EmbeddingModelSpec] = MappingProxyType(
    {
        EmbeddingModelName.TEXT_EMBEDDING_3_SMALL: EmbeddingModelSpec(
            name=EmbeddingModelName.TEXT_EMBEDDING_3_SMALL,
            native_dimensions=1536,
            **_OPENAI_LIMITS,
        ),
        EmbeddingModelName.TEXT_EMBEDDING_3_LARGE: EmbeddingModelSpec(
            name=EmbeddingModelName.TEXT_EMBEDDING_3_LARGE,
            native_dimensions=3072,
            **_OPENAI_LIMITS,
        ),
    }
)


def get_model_spec(model: EmbeddingModelName | str) -> EmbeddingModelSpec:
    try:
        return EMBEDDING_MODELS[EmbeddingModelName(model)]
    except (ValueError, KeyError) as exc:
        raise UnsupportedEmbeddingModelError(
            details={"model": str(model), "supported": [m.value for m in EMBEDDING_MODELS]}
        ) from exc


def resolve_space(model: EmbeddingModelName | str, dimensions: int | None = None) -> EmbeddingSpace:
    """Validate a (model, requested dimensions) pair; `None` means the model's native size."""
    spec = get_model_spec(model)
    if dimensions is None:
        return EmbeddingSpace(spec.name, spec.native_dimensions)
    if not spec.min_dimensions <= dimensions <= spec.native_dimensions:
        raise UnsupportedEmbeddingDimensionsError(
            details={
                "model": spec.name.value,
                "requested": dimensions,
                "min": spec.min_dimensions,
                "max": spec.native_dimensions,
            }
        )
    return EmbeddingSpace(spec.name, dimensions)


def request_dimensions(space: EmbeddingSpace) -> int | None:
    """Value for the provider's `dimensions` parameter; `None` = send nothing (native size)."""
    return None if space.dimensions == space.spec.native_dimensions else space.dimensions
