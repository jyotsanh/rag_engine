from pydantic import Field
from pymongo import IndexModel

from rag_engine.domain.base import MongoDocument
from rag_engine.domain.types import (
    HttpUrlStr,
    JobId,
    KnowledgeBaseId,
    NonNegativeInt,
    PageId,
    Sha256Hex,
    SourceId,
    TenantId,
)


class Chunk(MongoDocument):
    """A retrievable piece of a page, stored per (source, version) for audit and re-embedding.

    `text` is shown to users; `embed_text` (title + heading path + text) is what gets
    embedded (Step 58).
    """

    collection_name = "chunks"
    id_prefix = "chk"
    indexes = (
        IndexModel([("source_id", 1), ("version", 1)], name="source_version"),
        IndexModel([("page_id", 1), ("ordinal", 1)], unique=True, name="page_ordinal_unique"),
    )

    tenant_id: TenantId
    kb_id: KnowledgeBaseId
    source_id: SourceId
    version: JobId
    page_id: PageId
    url: HttpUrlStr
    title: str | None = Field(default=None, max_length=1000)
    heading_path: list[str] = Field(default_factory=list, max_length=12)
    ordinal: NonNegativeInt
    text: str = Field(min_length=1)
    embed_text: str = Field(min_length=1)
    token_count: int = Field(gt=0)  # tokens of `embed_text` (what the embedder receives)
    page_number: int | None = Field(default=None, gt=0)  # PDF page the chunk starts on (S56)
    content_hash: Sha256Hex
