"""Operator actions on tenants and their API keys (backs `/admin/*`)."""

from typing import Any

from pydantic import ValidationError

from rag_engine.config import Settings, logger, settings
from rag_engine.domain import ApiKey, Tenant, TenantQuotas
from rag_engine.exceptions import AppValidationError, QuotaExceededError
from rag_engine.repositories import ApiKeyRepository, Page, TenantRepository
from rag_engine.security import ApiKeyHasher, GeneratedApiKey


class TenantAdminService:
    def __init__(
        self,
        tenants: TenantRepository,
        keys: ApiKeyRepository,
        hasher: ApiKeyHasher,
        config: Settings = settings,
    ) -> None:
        self._tenants = tenants
        self._keys = keys
        self._hasher = hasher
        self._config = config

    @staticmethod
    def _merge_quotas(current: TenantQuotas, patch: dict[str, Any]) -> TenantQuotas:
        try:
            return TenantQuotas.model_validate(current.model_dump() | patch)
        except ValidationError as exc:
            # cross-field limits (e.g. the service page cap) surface as a 422
            raise AppValidationError(
                "Invalid quotas",
                details={
                    "errors": exc.errors(
                        include_input=False, include_url=False, include_context=False
                    )
                },
            ) from exc

    async def create_tenant(
        self, name: str, *, default_embedding_model: str | None, quota_patch: dict[str, Any]
    ) -> Tenant:
        quotas = self._merge_quotas(TenantQuotas(), quota_patch)
        fields: dict[str, Any] = {"name": name, "quotas": quotas}
        if default_embedding_model is not None:
            fields["default_embedding_model"] = default_embedding_model
        tenant = await self._tenants.create(Tenant(**fields))
        logger.bind(tenant_id=tenant.id).info("admin created tenant {}", tenant.name)
        return tenant

    async def update_tenant(self, tenant_id: str, changes: dict[str, Any]) -> Tenant:
        tenant = await self._tenants.get_or_raise(tenant_id)
        if "quotas" in changes:
            quota_patch = {k: v for k, v in (changes["quotas"] or {}).items() if v is not None}
            changes = changes | {"quotas": self._merge_quotas(tenant.quotas, quota_patch)}
        updated = await self._tenants.update(tenant_id, changes)
        logger.bind(tenant_id=tenant_id, fields=sorted(changes)).info("admin updated tenant")
        return updated

    async def issue_key(
        self, tenant_id: str, *, name: str, expires_at: Any = None
    ) -> tuple[ApiKey, GeneratedApiKey]:
        await self._tenants.get_or_raise(tenant_id)
        active = await self._keys.count_active(tenant_id)
        if active >= self._config.TENANT_MAX_ACTIVE_API_KEYS:
            raise QuotaExceededError(
                "Too many active API keys; revoke one first",
                details={
                    "quota": "max_active_api_keys",
                    "limit": self._config.TENANT_MAX_ACTIVE_API_KEYS,
                },
            )
        record, generated = await self._keys.issue(
            tenant_id, self._hasher, name=name, expires_at=expires_at
        )
        logger.bind(tenant_id=tenant_id, key_id=record.id).info("admin issued API key")
        return record, generated

    async def list_keys(
        self, tenant_id: str, *, include_revoked: bool, limit: int, cursor: str | None
    ) -> Page[ApiKey]:
        await self._tenants.get_or_raise(tenant_id)
        return await self._keys.list_for_tenant(
            tenant_id, include_revoked=include_revoked, limit=limit, cursor=cursor
        )

    async def revoke_key(self, tenant_id: str, key_id: str) -> ApiKey:
        await self._tenants.get_or_raise(tenant_id)
        key = await self._keys.revoke(key_id, tenant_id)
        logger.bind(tenant_id=tenant_id, key_id=key_id).info("admin revoked API key")
        return key
