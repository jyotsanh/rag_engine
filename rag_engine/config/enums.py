"""Backend selectors for the swappable infrastructure concerns.

Each dispatcher (vector store, embeddings, reranker) resolves its implementation
from one of these enums; settings pick the active member.
"""

from enum import StrEnum


class Environment(StrEnum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class VectorStoreType(StrEnum):
    QDRANT = "qdrant"


class EmbeddingProvider(StrEnum):
    OPENAI = "openai"


class RerankerType(StrEnum):
    CROSS_ENCODER = "cross_encoder"


class EmbeddingModelName(StrEnum):
    """Embedding models a tenant/KB may use. Step 14 maps each to its spec (dims, limits)."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
