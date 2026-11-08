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


class SparseEmbedderType(StrEnum):
    """Sparse (lexical) embedders for hybrid search (Step 66)."""

    BM25 = "bm25"  # fastembed `Qdrant/bm25`; Qdrant applies IDF at query time


class RerankerType(StrEnum):
    CROSS_ENCODER = "cross_encoder"


class ChunkerType(StrEnum):
    """Chunking strategies (Step 55 factory). `structure` is built in Steps 56-58."""

    STRUCTURE = "structure"  # Markdown structure (headings, atomic blocks) + token budget


class JsRenderMode(StrEnum):
    """When the crawler re-renders HTML pages in headless Chromium (Step 36)."""

    AUTO = "auto"  # only pages the JS-shell detector flags
    ALWAYS = "always"  # every HTML page (slow; for SPAs the detector misses)
    NEVER = "never"  # plain HTTP only


class EmbeddingModelName(StrEnum):
    """Embedding models a tenant/KB may use. Step 14 maps each to its spec (dims, limits)."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
