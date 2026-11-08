"""Vector store data model (Step 67): collections, points, payloads, filters, hits.

- **One collection per embedding space** (`chunks__<model>__<dims>`): vectors of
  different spaces are never comparable, and knowledge bases sharing a space share a
  collection, separated by payload filters (tenant first).
- A point is one chunk: a named dense vector `dense` (cosine) plus, for hybrid search, a
  named sparse vector `bm25` (Qdrant applies IDF, Step 66/68), and a `ChunkPayload` with
  everything a hit needs to be shown and cited without a Mongo round trip.
- Point ids are **deterministic** UUID5s of `(version, chunk_id)` (`point_id`): re-ingesting
  a page overwrites its points instead of duplicating them, and a version's points never
  collide with another version's.
- **`VectorFilter` always carries a tenant**: no search, count or delete can cross tenants
  by construction. Empty tuples mean "no constraint on this field".
"""

import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.domain import Chunk
from rag_engine.embeddings import EmbeddingSpace, SparseVector

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
COLLECTION_PREFIX = "chunks"
# a fixed namespace: point ids are stable across processes, machines and releases
_POINT_NAMESPACE = uuid.UUID("7d3c6a2e-5d0f-4c2b-9a51-3b8e4e1f6c90")


def point_id(version: str, chunk_id: str) -> str:
    """Deterministic point id of a chunk in a version (Qdrant ids must be UUIDs or ints)."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{version}:{chunk_id}"))


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    name: str
    dimensions: int
    sparse: bool = True  # a `bm25` sparse vector for hybrid search

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        if not self.name or not all(c.isalnum() or c in "_-" for c in self.name):
            raise ValueError(f"invalid collection name {self.name!r}")

    @classmethod
    def for_space(cls, space: EmbeddingSpace, *, sparse: bool = True) -> CollectionSpec:
        model = space.model.value.replace(".", "-")
        return cls(f"{COLLECTION_PREFIX}__{model}__{space.dimensions}", space.dimensions, sparse)


class ChunkPayload(BaseModel):
    """What a point carries (indexed fields are used by filters, Step 68)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    tenant_id: str
    kb_id: str
    source_id: str
    version: str
    chunk_id: str
    page_id: str
    url: str
    ordinal: int = Field(ge=0)
    text: str
    title: str | None = None
    heading_path: tuple[str, ...] = ()
    page_number: int | None = None

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkPayload:
        return cls(
            tenant_id=chunk.tenant_id,
            kb_id=chunk.kb_id,
            source_id=chunk.source_id,
            version=chunk.version,
            chunk_id=chunk.id,
            page_id=chunk.page_id,
            url=chunk.url,
            ordinal=chunk.ordinal,
            text=chunk.text,
            title=chunk.title,
            heading_path=tuple(chunk.heading_path),
            page_number=chunk.page_number,
        )


@dataclass(frozen=True, slots=True)
class VectorPoint:
    id: str
    dense: list[float]
    payload: ChunkPayload
    sparse: SparseVector | None = None

    @classmethod
    def from_chunk(
        cls, chunk: Chunk, dense: list[float], sparse: SparseVector | None = None
    ) -> VectorPoint:
        return cls(point_id(chunk.version, chunk.id), dense, ChunkPayload.from_chunk(chunk), sparse)


@dataclass(frozen=True, slots=True)
class VectorFilter:
    tenant_id: str
    kb_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    page_ids: tuple[str, ...] = ()
    url_prefix: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("a vector filter needs a tenant_id")
        if self.url_prefix is not None and not self.url_prefix:
            raise ValueError("url_prefix must be non-empty when given")

    def matches(self, payload: ChunkPayload) -> bool:
        """Reference semantics every implementation must reproduce."""
        return (
            payload.tenant_id == self.tenant_id
            and (not self.kb_ids or payload.kb_id in self.kb_ids)
            and (not self.source_ids or payload.source_id in self.source_ids)
            and (not self.versions or payload.version in self.versions)
            and (not self.page_ids or payload.page_id in self.page_ids)
            and (self.url_prefix is None or payload.url.startswith(self.url_prefix))
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float  # cosine similarity for dense search; fused rank score for hybrid
    payload: ChunkPayload = field(compare=False)
