"""Base class for every document persisted in MongoDB.

Subclasses set `collection_name`, `id_prefix` and (optionally) `indexes`.
Concrete subclasses register themselves so startup can ensure all indexes.
IDs are prefixed, time-ordered UUIDv7 strings (e.g. `kb_0199...`), which keeps
`_id` index inserts append-only and lets `_id` double as a pagination cursor.
"""

import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field
from pymongo import IndexModel

_REGISTRY: dict[str, type[MongoDocument]] = {}


def utc_now() -> datetime:
    # Mongo stores milliseconds; truncate so a round-trip compares equal
    now = datetime.now(UTC)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid7().hex}"


class MongoDocument(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True,
        validate_by_alias=True,
        validate_assignment=True,
        extra="ignore",  # tolerate fields written by newer code during rolling deploys
    )

    collection_name: ClassVar[str]
    id_prefix: ClassVar[str]
    indexes: ClassVar[tuple[IndexModel, ...]] = ()

    id: str = Field(default="", alias="_id")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("collection_name")
        if name is None:
            return  # abstract intermediate class
        existing = _REGISTRY.get(name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise TypeError(f"collection {name!r} already owned by {existing.__qualname__}")
        _REGISTRY[name] = cls

    def model_post_init(self, context: Any, /) -> None:
        if not self.id:
            self.id = new_id(self.id_prefix)

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="python")

    @classmethod
    def from_mongo(cls, raw: dict[str, Any]) -> Self:
        return cls.model_validate(raw)


def registered_documents() -> list[type[MongoDocument]]:
    return list(_REGISTRY.values())


__all__ = ["MongoDocument", "new_id", "registered_documents", "utc_now"]
