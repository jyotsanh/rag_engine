"""Request/response bodies for `/admin/*`. Internal fields (key hashes) never appear here."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_engine.domain import (
    ApiKey,
    EmbeddingModelName,
    Tenant,
    TenantQuotas,
    TenantStatus,
    utc_now,
)
from rag_engine.domain.types import Name
from rag_engine.security import display_hint


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typos in admin payloads should fail loudly


class TenantQuotasPatch(_Request):
    """Partial quota change; omitted fields keep their current value."""

    max_knowledge_bases: int | None = Field(default=None, gt=0)
    max_concurrent_jobs: int | None = Field(default=None, gt=0)
    max_pages_per_job: int | None = Field(default=None, gt=0)
    api_rate_limit_per_minute: int | None = Field(default=None, gt=0)


class TenantCreate(_Request):
    name: Name
    default_embedding_model: EmbeddingModelName | None = Field(
        default=None, description="Defaults to the service's DEFAULT_EMBEDDING_MODEL."
    )
    quotas: TenantQuotasPatch | None = Field(
        default=None, description="Overrides on top of the service default quotas."
    )


class TenantPatch(_Request):
    name: Name | None = None
    status: TenantStatus | None = None
    default_embedding_model: EmbeddingModelName | None = Field(
        default=None,
        description="Applies to knowledge bases created later; existing ones keep theirs.",
    )
    quotas: TenantQuotasPatch | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("provide at least one field to change")
        nulls = sorted(f for f in self.model_fields_set if getattr(self, f) is None)
        if nulls:
            raise ValueError(f"fields cannot be null: {nulls}")
        return self


class TenantResponse(BaseModel):
    id: str
    name: str
    status: TenantStatus
    default_embedding_model: EmbeddingModelName
    quotas: TenantQuotas
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, tenant: Tenant) -> TenantResponse:
        return cls.model_validate(tenant.model_dump())


class ApiKeyState(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ApiKeyCreate(_Request):
    name: Name = "default"
    expires_at: datetime | None = Field(default=None, description="Omit for a non-expiring key.")

    @field_validator("expires_at")
    @classmethod
    def _in_future(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("expires_at must include a timezone")
        if value <= utc_now():
            raise ValueError("expires_at must be in the future")
        return value


class ApiKeyResponse(BaseModel):
    id: str
    tenant_id: str
    name: str
    hint: str = Field(description="Non-secret identifier, e.g. `rk_AbCdEf123456_…`.")
    state: ApiKeyState
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None

    @classmethod
    def of(cls, key: ApiKey) -> ApiKeyResponse:
        if key.revoked_at is not None:
            state = ApiKeyState.REVOKED
        elif not key.is_usable():
            state = ApiKeyState.EXPIRED
        else:
            state = ApiKeyState.ACTIVE
        return cls(
            id=key.id,
            tenant_id=key.tenant_id,
            name=key.name,
            hint=display_hint(key.key_prefix),
            state=state,
            created_at=key.created_at,
            last_used_at=key.last_used_at,
            expires_at=key.expires_at,
            revoked_at=key.revoked_at,
        )


class ApiKeyCreatedResponse(ApiKeyResponse):
    api_key: str = Field(
        description="The full key. Shown only in this response; it cannot be retrieved again."
    )
