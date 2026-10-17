from pydantic import Field
from pymongo import IndexModel

from rag_engine.domain.base import MongoDocument
from rag_engine.domain.enums import EmbeddingModelName, ResourceStatus
from rag_engine.domain.types import Description, Name, TenantId


class KnowledgeBase(MongoDocument):
    """A searchable collection of sources. Its embedding model is locked at creation."""

    collection_name = "knowledge_bases"
    id_prefix = "kb"
    indexes = (
        # names unique per tenant among live KBs; a KB being deleted frees its name
        IndexModel(
            [("tenant_id", 1), ("name", 1)],
            unique=True,
            partialFilterExpression={"status": ResourceStatus.ACTIVE.value},
            name="tenant_name_active_unique",
        ),
        IndexModel([("tenant_id", 1), ("_id", -1)], name="tenant_kbs"),
    )

    tenant_id: TenantId
    name: Name
    description: Description = ""
    status: ResourceStatus = ResourceStatus.ACTIVE
    embedding_model: EmbeddingModelName = Field(frozen=True)
    embedding_dimensions: int = Field(gt=0, le=4096, frozen=True)
