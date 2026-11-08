import hashlib
from datetime import datetime

from pydantic import Field
from pymongo import ASCENDING, DESCENDING, IndexModel

from rag_engine.config import EmbeddingModelName
from rag_engine.domain.base import MongoDocument
from rag_engine.domain.enums import UsagePurpose
from rag_engine.domain.types import NonNegativeInt, TenantId

_PERIOD = r"^\d{4}-(0[1-9]|1[0-2])$"


def usage_period(at: datetime) -> str:
    """The UTC calendar month usage is billed to: `2026-11`."""
    return f"{at.year:04d}-{at.month:02d}"


def tenant_usage_id(
    tenant_id: str, period: str, purpose: UsagePurpose, model: EmbeddingModelName
) -> str:
    """Deterministic: concurrent workers `$inc` the same document instead of racing inserts."""
    raw = f"{tenant_id}|{period}|{purpose.value}|{model.value}"
    return f"{TenantUsage.id_prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:32]}"


class TenantUsage(MongoDocument):
    """A tenant's embedding usage in one month, for one purpose and model (Step 65).

    Ingest usage is also recorded on each job (`CrawlJob.usage`); query usage only lives
    here (queries have no job).
    """

    collection_name = "tenant_usage"
    id_prefix = "use"
    indexes = (
        IndexModel([("tenant_id", ASCENDING), ("period", DESCENDING)], name="tenant_period"),
    )

    tenant_id: TenantId
    period: str = Field(pattern=_PERIOD)
    purpose: UsagePurpose
    model: EmbeddingModelName
    requests: NonNegativeInt = 0
    tokens: NonNegativeInt = 0
    cached: NonNegativeInt = 0
