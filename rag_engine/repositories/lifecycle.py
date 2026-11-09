"""Queries for documents being deleted (`ResourceStatus.DELETING`), shared by the KB and
source repositories: the cascade (Step 81) touches what it works on, and the deletion sweep
finds cascades nobody touched for a while (a lost task message, a dead worker)."""

from datetime import datetime
from typing import Any, ClassVar

from pymongo import ASCENDING
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import PyMongoError

from rag_engine.domain import MongoDocument, ResourceStatus, utc_now
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError

_DELETING = {"status": ResourceStatus.DELETING.value}


class DeletingQueries[T: MongoDocument]:
    """Mixed into a `BaseRepository` (which provides `model` and `collection`)."""

    model: ClassVar[type[MongoDocument]]
    collection: AsyncCollection[dict]

    async def find_deleting_since(self, before: datetime, *, limit: int = 100) -> list[T]:
        """Documents marked deleting and untouched since `before` (oldest first)."""
        try:
            cursor = self.collection.find(
                {**_DELETING, "updated_at": {"$lt": before}},
                sort=[("updated_at", ASCENDING)],
                limit=limit,
            )
            return [self.model.from_mongo(raw) async for raw in cursor]  # type: ignore
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def touch_deleting(
        self, id_: str, seen_updated_at: datetime | None = None, *, now: datetime | None = None
    ) -> bool:
        """Bump a deleting document's `updated_at` ("a cascade is on it").

        With `seen_updated_at`, only if nothing touched it since that read, so of two
        sweeps only one restarts its cascade. False when it isn't deleting (any more).
        """
        filter_: dict[str, Any] = {"_id": id_, **_DELETING}
        if seen_updated_at is not None:
            filter_["updated_at"] = seen_updated_at
        try:
            result = await self.collection.update_one(
                filter_, {"$set": {"updated_at": now or utc_now()}}
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return result.modified_count == 1
