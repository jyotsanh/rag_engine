"""Embedder dispatcher: `EmbeddingProvider` → `BaseEmbedder` implementation (Step 61).

Implementations register with `@register_embedder(EmbeddingProvider.X)` in their module
(OpenAI: Step 62). `get_embedder(space)` builds the embedder for a knowledge base's space
with the active provider (`ACTIVE_EMBEDDING_PROVIDER`), refusing a model that belongs to
another provider.
"""

from collections.abc import Callable

from rag_engine.config import EmbeddingProvider, Settings, settings
from rag_engine.embeddings.base import BaseEmbedder
from rag_engine.embeddings.exceptions import UnsupportedEmbeddingProviderError
from rag_engine.embeddings.models import EmbeddingSpace

_EMBEDDERS: dict[EmbeddingProvider, type[BaseEmbedder]] = {}


def register_embedder[E: type[BaseEmbedder]](provider: EmbeddingProvider) -> Callable[[E], E]:
    def register(cls: E) -> E:
        if not (isinstance(cls, type) and issubclass(cls, BaseEmbedder)):
            raise TypeError(f"{cls!r} is not a BaseEmbedder subclass")
        if getattr(cls, "provider", None) is not provider:
            raise TypeError(f"{cls.__name__}.provider must be {provider!r}")
        if provider in _EMBEDDERS and _EMBEDDERS[provider] is not cls:
            raise TypeError(
                f"embedder {str(provider)!r} is already registered "
                f"({_EMBEDDERS[provider].__name__})"
            )
        _EMBEDDERS[provider] = cls
        return cls

    return register


def registered_embedders() -> dict[EmbeddingProvider, type[BaseEmbedder]]:
    return dict(_EMBEDDERS)


def get_embedder(
    space: EmbeddingSpace,
    *,
    provider: EmbeddingProvider | None = None,
    config: Settings = settings,
) -> BaseEmbedder:
    provider = provider or config.ACTIVE_EMBEDDING_PROVIDER
    if space.spec.provider is not provider:
        raise UnsupportedEmbeddingProviderError(
            f"{space.model} is not a {provider} model",
            details={"model": space.model.value, "provider": provider.value},
        )
    cls = _EMBEDDERS.get(provider)
    if cls is None:
        raise UnsupportedEmbeddingProviderError(
            f"no embedder is registered for {str(provider)!r}",
            details={"provider": provider.value, "registered": sorted(_EMBEDDERS)},
        )
    return cls(space, config=config)
