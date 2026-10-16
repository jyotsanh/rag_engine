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
