"""Abstract base for MongoDB-backed domain documents.

Phase 2 (#70): the skeleton only — a Pydantic v2 generic base with an
auto-generated ``id`` and identity semantics. The persistence surface builds on
top in later sub-issues of #4:

- serialization (``to_mongo`` / ``from_mongo``) — #71
- collection-name resolution via an inner ``Settings`` — #72
- write ops (``save`` / ``bulk_insert``) — #73
- read ops (``find`` / ``bulk_find``) — #74
- ``get_or_create`` — #75

Unlike the reference implementation, no database handle is bound at import time;
later sub-issues reach the backend through
:class:`rag_engine.infrastructure.db.DatabaseDispatcher`.
"""

import uuid
from abc import ABC

from pydantic import UUID4, BaseModel, ConfigDict, Field


class NoSQLBaseDocument[T: "NoSQLBaseDocument"](BaseModel, ABC):
    """Base class every MongoDB document model inherits from.

    ``T`` is bound to the subclass so the classmethods added in later sub-issues
    (``find``, ``get_or_create``, ...) can be typed as returning the concrete
    subtype rather than the base.

    ``populate_by_name`` is enabled so the ``_id`` alias introduced with
    serialization (#71) can be populated by field name too.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)
