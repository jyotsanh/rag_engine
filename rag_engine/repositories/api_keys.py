from datetime import datetime, timedelta
from typing import ClassVar

from pydantic import SecretStr

from rag_engine.domain import ApiKey, MongoDocument, utc_now
from rag_engine.repositories.base import BaseRepository
from rag_engine.repositories.exceptions import (
    ApiKeyNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentError,
)
from rag_engine.repositories.pagination import DEFAULT_PAGE_SIZE, Page
from rag_engine.security import ApiKeyHasher, GeneratedApiKey

# a fresh prefix collision needs ~2^35 keys; more than a few retries means something is wrong
_MAX_GENERATE_ATTEMPTS = 3
# avoid one write per request: last_used_at only moves if it is older than this
LAST_USED_RESOLUTION = timedelta(minutes=1)


class ApiKeyRepository(BaseRepository[ApiKey]):
    model: ClassVar[type[MongoDocument]] = ApiKey
    not_found_error: ClassVar[type[DocumentNotFoundError]] = ApiKeyNotFoundError

    async def issue(
        self,
        tenant_id: str,
        hasher: ApiKeyHasher,
        *,
        name: str = "default",
        expires_at: datetime | None = None,
    ) -> tuple[ApiKey, GeneratedApiKey]:
        """Generate and store a key; the returned plaintext must be shown once and dropped."""
        for attempt in range(1, _MAX_GENERATE_ATTEMPTS + 1):
            generated = hasher.generate()
            record = ApiKey(
                tenant_id=tenant_id,
                name=name,
                key_prefix=generated.prefix,
                key_hash=generated.key_hash,
                expires_at=expires_at,
            )
            try:
                return await self.insert(record), generated
            except DuplicateDocumentError:
                if attempt == _MAX_GENERATE_ATTEMPTS:
                    raise
        raise AssertionError("unreachable")

    async def get_by_prefix(self, prefix: str) -> ApiKey | None:
        return await self.find_one({"key_prefix": prefix})

    async def list_for_tenant(
        self,
        tenant_id: str,
        *,
        include_revoked: bool = False,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> Page[ApiKey]:
        filter_ = {} if include_revoked else {"revoked_at": None}
        return await self.paginate(
            filter_, scope={"tenant_id": tenant_id}, limit=limit, cursor=cursor
        )

    async def count_active(self, tenant_id: str, now: datetime | None = None) -> int:
        now = now or utc_now()
        return await self.count(
            {
                "revoked_at": None,
                "$or": [{"expires_at": None}, {"expires_at": {"$gt": now}}],
            },
            scope={"tenant_id": tenant_id},
        )

    async def revoke(self, key_id: str, tenant_id: str) -> ApiKey:
        """Idempotent: revoking an already-revoked key keeps its original revocation time."""
        scope = {"tenant_id": tenant_id}
        await self.collection.update_one(
            self._where({"_id": key_id, "revoked_at": None}, scope),
            {"$set": {"revoked_at": utc_now(), "updated_at": utc_now()}},
        )
        return await self.get_or_raise(key_id, scope=scope)

    async def touch_last_used(self, key_id: str, now: datetime | None = None) -> bool:
        """Record use at most once per `LAST_USED_RESOLUTION`; returns True if it wrote."""
        now = now or utc_now()
        result = await self.collection.update_one(
            {
                "_id": key_id,
                "$or": [
                    {"last_used_at": None},
                    {"last_used_at": {"$lt": now - LAST_USED_RESOLUTION}},
                ],
            },
            {"$set": {"last_used_at": now}},
        )
        return result.modified_count == 1

    async def rehash(self, key_id: str, plaintext: SecretStr, hasher: ApiKeyHasher) -> None:
        """Store the current-pepper hash after a key verified against a retired pepper."""
        await self.collection.update_one(
            {"_id": key_id},
            {"$set": {"key_hash": hasher.hash(plaintext), "updated_at": utc_now()}},
        )
