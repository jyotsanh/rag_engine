"""Abstract base for MongoDB-backed domain documents.

Built up across the sub-issues of #4:

- identity: auto-generated ``id`` + equality/hash — #70 ✅
- serialization (``to_mongo`` / ``from_mongo``) — #71 ✅
- collection-name resolution via an inner ``Settings`` — #72
- write ops (``save`` / ``bulk_insert``) — #73
- read ops (``find`` / ``bulk_find``) — #74
- ``get_or_create`` — #75

Unlike the reference implementation, no database handle is bound at import time;
the persistence sub-issues reach the backend through
:class:`rag_engine.infrastructure.db.DatabaseDispatcher`.
"""

import uuid
from abc import ABC
from typing import Any, Self

from pydantic import UUID4, BaseModel, ConfigDict, Field


class NoSQLBaseDocument(BaseModel, ABC):
    """Base class every MongoDB document model inherits from.

    Classmethods return ``Self``, so calling e.g. ``from_mongo`` on a concrete
    subclass yields that subclass, not the base.

    The ``id`` field aliases to Mongo's ``_id``. ``populate_by_name`` keeps
    construction by field name working too (``Doc(id=...)`` as well as the
    ``_id`` alias that ``from_mongo`` supplies).
    """

    model_config = ConfigDict(populate_by_name=True)

    id: UUID4 = Field(default_factory=uuid.uuid4, alias="_id")

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump to a plain dict, stringifying any ``UUID`` values.

        Overridden so Mongo stores UUIDs as strings while other native types
        (datetimes, etc.) are left untouched, keeping the round-trip lossless.
        """
        data = super().model_dump(**kwargs)

        for key, value in data.items():
            if isinstance(value, uuid.UUID):
                data[key] = str(value)

        return data

    def to_mongo(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to a Mongo-ready document: ``id`` becomes a stringified ``_id``."""
        kwargs.setdefault("by_alias", True)

        return self.model_dump(**kwargs)

    @classmethod
    def from_mongo(cls, data: dict[str, Any]) -> Self:
        """Rebuild a typed model from a Mongo document (``_id`` maps back to ``id``)."""
        if not data:
            raise ValueError("Data is empty.")

        return cls(**data)
