from datetime import datetime
from typing import Self

from pydantic import Field, model_validator
from pymongo import IndexModel

from rag_engine.domain.base import MongoDocument, utc_now
from rag_engine.domain.types import Name, Sha256Hex, TenantId


class ApiKey(MongoDocument):
    """A tenant credential. Only the HMAC of the secret is stored (Step 15 generates keys).

    `key_prefix` is the public, non-secret part shown in the key string
    (`rk_<key_prefix>_<secret>`) and is how a presented key is looked up.
    """

    collection_name = "api_keys"
    id_prefix = "key"
    indexes = (
        IndexModel([("key_prefix", 1)], unique=True, name="key_prefix_unique"),
        IndexModel([("tenant_id", 1), ("_id", -1)], name="tenant_keys"),
    )

    tenant_id: TenantId
    name: Name = "default"
    key_prefix: str = Field(pattern=r"^[A-Za-z0-9]{8,16}$")
    key_hash: Sha256Hex = Field(repr=False)
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    @model_validator(mode="after")
    def _expiry_after_creation(self) -> Self:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self

    def is_usable(self, now: datetime | None = None) -> bool:
        now = now or utc_now()
        if self.revoked_at is not None and self.revoked_at <= now:
            return False
        return self.expires_at is None or now < self.expires_at
