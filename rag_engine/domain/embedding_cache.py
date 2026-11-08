import hashlib
import struct
from datetime import datetime

from pydantic import Field
from pymongo import ASCENDING, IndexModel

from rag_engine.config import EmbeddingModelName, settings
from rag_engine.domain.base import MongoDocument, utc_now

# entries not used (hit or written) for this long are evicted by Mongo's TTL monitor
EMBEDDING_CACHE_TTL_SECONDS = settings.EMBEDDING_CACHE_TTL_DAYS * 24 * 60 * 60


def embedding_cache_key(model: EmbeddingModelName | str, dimensions: int, embed_text: str) -> str:
    """sha256(model|dims|embed_text): the same text in the same space is the same vector."""
    raw = f"{EmbeddingModelName(model).value}|{dimensions}|{embed_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pack_vector(vector: list[float]) -> bytes:
    """Little-endian float32: half the size of BSON doubles, the model's own precision."""
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(data: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(data) // 4}f", data))


class EmbeddingCacheEntry(MongoDocument):
    """One embedded text in one embedding space (Step 64). `_id` is the cache key."""

    collection_name = "embedding_cache"
    id_prefix = "emb"  # unused: entries are always created with their cache key as `_id`
    indexes = (
        IndexModel([("last_used_at", ASCENDING)], expireAfterSeconds=EMBEDDING_CACHE_TTL_SECONDS,
                   name="embedding_cache_ttl"),
    )  # fmt: skip

    model: EmbeddingModelName
    dimensions: int = Field(gt=0)
    vector: bytes  # pack_vector(); 4 bytes per dimension
    last_used_at: datetime = Field(default_factory=utc_now)
