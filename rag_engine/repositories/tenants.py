from collections.abc import Mapping
from typing import Any, ClassVar

from rag_engine.domain import TENANT_NAME_COLLATION, MongoDocument, Tenant, TenantStatus
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
