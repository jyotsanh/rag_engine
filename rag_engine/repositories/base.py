"""Generic async repository over one `MongoDocument` collection."""

from collections.abc import Mapping
from typing import Any, ClassVar

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import DuplicateKeyError, PyMongoError

from rag_engine.domain.base import MongoDocument, utc_now
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.infrastructure.mongo import MongoDatabase
from rag_engine.repositories.exceptions import DocumentNotFoundError, DuplicateDocumentError
from rag_engine.repositories.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    Page,
    SortOrder,
    decode_cursor,
    encode_cursor,
)

type Filter = Mapping[str, Any]


class BaseRepository[T: MongoDocument]:
    """CRUD + keyset pagination. Subclasses set `model` and add domain queries.

    Every public method accepts an optional `scope` filter that is AND-ed into
    the query; tenant-owned repositories pass `{"tenant_id": ...}` so one tenant
    can never read or modify another's documents.
    """

    model: ClassVar[type[MongoDocument]]
    not_found_error: ClassVar[type[DocumentNotFoundError]] = DocumentNotFoundError

    def __init__(self, db: MongoDatabase) -> None:
        self.collection: AsyncCollection[dict] = db[self.model.collection_name]

    def _load(self, raw: Mapping[str, Any]) -> T:
        return self.model.from_mongo(dict(raw))  # type: ignore

    @staticmethod
    def _where(filter_: Filter | None, scope: Filter | None) -> dict[str, Any]:
        return {**(filter_ or {}), **(scope or {})}

    async def insert(self, document: T) -> T:
        try:
            await self.collection.insert_one(document.to_mongo())
        except DuplicateKeyError as exc:
            raise DuplicateDocumentError(
                details={"collection": self.model.collection_name, "key": exc.details}
            ) from exc
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        return document

    async def get(self, id_: str, *, scope: Filter | None = None) -> T | None:
        raw = await self.collection.find_one(self._where({"_id": id_}, scope))
        return self._load(raw) if raw is not None else None

    async def get_or_raise(self, id_: str, *, scope: Filter | None = None) -> T:
        document = await self.get(id_, scope=scope)
        if document is None:
            raise self.not_found_error(details={"id": id_})
        return document

    async def find_one(self, filter_: Filter, *, scope: Filter | None = None) -> T | None:
        raw = await self.collection.find_one(self._where(filter_, scope))
        return self._load(raw) if raw is not None else None

    async def find(
        self,
        filter_: Filter | None = None,
        *,
        scope: Filter | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 0,
    ) -> list[T]:
        cursor = self.collection.find(self._where(filter_, scope), sort=sort, limit=limit)
        return [self._load(raw) async for raw in cursor]

    async def count(self, filter_: Filter | None = None, *, scope: Filter | None = None) -> int:
        return await self.collection.count_documents(self._where(filter_, scope))

    async def update(
        self, id_: str, changes: Mapping[str, Any], *, scope: Filter | None = None
    ) -> T:
        """`$set` the given fields (validated against the model) and bump `updated_at`."""
        forbidden = {"_id", "id", "created_at"} & set(changes)
        if forbidden:
            raise ValueError(f"immutable fields cannot be updated: {sorted(forbidden)}")
        current = await self.get_or_raise(id_, scope=scope)
        # validate the merged document so invalid values never reach the database
        merged = self.model.model_validate({**current.to_mongo(), **changes})
        stored = merged.to_mongo()
        update_set = {field: stored[field] for field in changes} | {"updated_at": utc_now()}
        try:
            raw = await self.collection.find_one_and_update(
                self._where({"_id": id_}, scope),
                {"$set": update_set},
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError as exc:
            raise DuplicateDocumentError(
                details={"collection": self.model.collection_name, "key": exc.details}
            ) from exc
        if raw is None:  # deleted between read and write
            raise self.not_found_error(details={"id": id_})
        return self._load(raw)

    async def delete(self, id_: str, *, scope: Filter | None = None) -> bool:
        result = await self.collection.delete_one(self._where({"_id": id_}, scope))
        return result.deleted_count == 1

    async def delete_many(self, filter_: Filter, *, scope: Filter | None = None) -> int:
        if not filter_ and not scope:
            raise ValueError("refusing to delete every document; pass a filter")
        result = await self.collection.delete_many(self._where(filter_, scope))
        return result.deleted_count

    async def paginate(
        self,
        filter_: Filter | None = None,
        *,
        scope: Filter | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        order: SortOrder = "desc",
    ) -> Page[T]:
        """Keyset pagination on `_id` (newest first by default)."""
        if not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        where = self._where(filter_, scope)
        if cursor is not None:
            after = decode_cursor(cursor, order)
            bound = {"$lt": after} if order == "desc" else {"$gt": after}
            where = {"$and": [where, {"_id": bound}]} if where else {"_id": bound}
        direction = DESCENDING if order == "desc" else ASCENDING
        # fetch one extra row to know whether another page exists
        rows = await self.find(where, sort=[("_id", direction)], limit=limit + 1)
        has_more = len(rows) > limit
        items = rows[:limit]
        next_cursor = encode_cursor(items[-1].id, order) if has_more and items else None
        return Page[T](items=items, next_cursor=next_cursor)
