from datetime import datetime

from pydantic import BaseModel
from pymongo import IndexModel

from rag_engine.domain.base import MongoDocument
from rag_engine.domain.enums import ResourceStatus
from rag_engine.domain.types import HttpUrlStr, JobId, KnowledgeBaseId, NonNegativeInt, TenantId


class SourceStats(BaseModel):
    """Figures for the currently active version."""

    pages: NonNegativeInt = 0
    chunks: NonNegativeInt = 0
    last_succeeded_at: datetime | None = None


class Source(MongoDocument):
    """One crawled site inside a knowledge base, identified by its normalized seed URL.

    `active_version` is the job id whose vectors search currently serves; a new
    crawl writes under its own version and flips this on success (atomic swap).
    """

    collection_name = "sources"
    id_prefix = "src"
    indexes = (
        IndexModel(
            [("kb_id", 1), ("seed_url", 1)],
            unique=True,
            partialFilterExpression={"status": ResourceStatus.ACTIVE.value},
            name="kb_seed_active_unique",
        ),
        IndexModel([("tenant_id", 1), ("kb_id", 1), ("_id", -1)], name="tenant_kb_sources"),
        # Step 81: the deletion sweep finds sources whose cascade stalled
        IndexModel(
            [("updated_at", 1)],
            partialFilterExpression={"status": ResourceStatus.DELETING.value},
            name="deleting_since",
        ),
    )

    tenant_id: TenantId
    kb_id: KnowledgeBaseId
    seed_url: HttpUrlStr
    status: ResourceStatus = ResourceStatus.ACTIVE
    active_version: JobId | None = None
    last_job_id: JobId | None = None
    stats: SourceStats = SourceStats()
