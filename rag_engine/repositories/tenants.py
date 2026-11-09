from collections.abc import Mapping
from typing import Any, ClassVar

from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from rag_engine.domain import (
    TENANT_NAME_COLLATION,
    MongoDocument,
    Tenant,
    TenantStatus,
    WebhookSecret,
    utc_now,
)
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.repositories.base import BaseRepository, Filter
from rag_engine.repositories.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    TenantNameTakenError,
    TenantNotFoundError,
)


class TenantRepository(BaseRepository[Tenant]):
    model: ClassVar[type[MongoDocument]] = Tenant
    not_found_error: ClassVar[type[DocumentNotFoundError]] = TenantNotFoundError

    async def create(self, tenant: Tenant) -> Tenant:
        try:
            return await self.insert(tenant)
        except DuplicateDocumentError as exc:
            raise TenantNameTakenError(details={"name": tenant.name}) from exc

    async def get_by_name(self, name: str) -> Tenant | None:
        """Case-insensitive, using the same collation as the unique index."""
        raw = await self.collection.find_one(
            {"name": name.strip()}, collation=TENANT_NAME_COLLATION
        )
        return self._load(raw) if raw is not None else None

    async def update(
        self, id_: str, changes: Mapping[str, Any], *, scope: Filter | None = None
    ) -> Tenant:
        try:
            return await super().update(id_, changes, scope=scope)
        except DuplicateDocumentError as exc:
            raise TenantNameTakenError(details={"name": changes.get("name")}) from exc

    async def set_status(self, id_: str, status: TenantStatus) -> Tenant:
        return await self.update(id_, {"status": status})

    async def replace_webhook_secrets(
        self, tenant: Tenant, secrets: list[WebhookSecret]
    ) -> Tenant | None:
        """Compare-and-swap the tenant's webhook secrets (Step 82).

        Applied only if they are still what `tenant` (as read) holds, so two concurrent
        rotations can't silently drop a secret; None when they changed in between.
        """
        expected = [s.model_dump() for s in tenant.webhook_secrets]
        # tenants stored before Step 82 have no field at all
        current: dict[str, Any] = (
            {"webhook_secrets": expected}
            if expected
            else {"$or": [{"webhook_secrets": {"$exists": False}}, {"webhook_secrets": []}]}
        )
        try:
            raw = await self.collection.find_one_and_update(
                {"_id": tenant.id, **current},
                {
                    "$set": {
                        "webhook_secrets": [s.model_dump() for s in secrets],
                        "updated_at": utc_now(),
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return self._load(raw) if raw is not None else None
