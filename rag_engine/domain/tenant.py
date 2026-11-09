from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator
from pymongo import IndexModel

from rag_engine.config import settings
from rag_engine.domain.base import MongoDocument
from rag_engine.domain.enums import EmbeddingModelName, TenantStatus
from rag_engine.domain.types import Name

# tenant names are unique case-insensitively; queries on `name` must pass this collation
TENANT_NAME_COLLATION = {"locale": "en", "strength": 2}


class TenantQuotas(BaseModel):
    max_knowledge_bases: int = Field(
        default_factory=lambda: settings.TENANT_DEFAULT_MAX_KNOWLEDGE_BASES, gt=0
    )
    max_concurrent_jobs: int = Field(
        default_factory=lambda: settings.TENANT_DEFAULT_MAX_CONCURRENT_JOBS, gt=0
    )
    max_pages_per_job: int = Field(
        default_factory=lambda: settings.TENANT_DEFAULT_MAX_PAGES_PER_JOB, gt=0
    )
    api_rate_limit_per_minute: int = Field(
        default_factory=lambda: settings.TENANT_DEFAULT_API_RATE_LIMIT_PER_MINUTE, gt=0
    )

    @model_validator(mode="after")
    def _within_hard_limits(self) -> Self:
        if self.max_pages_per_job > settings.CRAWL_HARD_MAX_PAGES:
            raise ValueError(
                f"max_pages_per_job cannot exceed the service limit {settings.CRAWL_HARD_MAX_PAGES}"
            )
        return self


MAX_WEBHOOK_SECRETS = 2  # the current one, plus the previous one while receivers switch


class WebhookSecret(BaseModel):
    """A tenant's webhook signing secret (Step 82), encrypted at rest (`SecretBox`)."""

    sealed: str = Field(repr=False, min_length=1, max_length=500)
    prefix: str = Field(max_length=32)  # non-secret head, e.g. `whsec_AbC123`
    created_at: datetime


class Tenant(MongoDocument):
    collection_name = "tenants"
    id_prefix = "ten"
    indexes = (
        IndexModel([("name", 1)], unique=True, collation=TENANT_NAME_COLLATION, name="name_unique"),
    )

    name: Name
    status: TenantStatus = TenantStatus.ACTIVE
    default_embedding_model: EmbeddingModelName = Field(
        default_factory=lambda: settings.DEFAULT_EMBEDDING_MODEL
    )
    quotas: TenantQuotas = Field(default_factory=TenantQuotas)
    # current first; webhooks are signed with every one of them (Step 82)
    webhook_secrets: list[WebhookSecret] = Field(
        default_factory=list, max_length=MAX_WEBHOOK_SECRETS
    )

    @property
    def is_active(self) -> bool:
        return self.status is TenantStatus.ACTIVE
