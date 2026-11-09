from typing import ClassVar

from pymongo import ReturnDocument
from pymongo.errors import PyMongoError

from rag_engine.domain import MongoDocument, ResourceStatus, Source, SourceStats, utc_now
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.repositories.base import BaseRepository
from rag_engine.repositories.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    SourceNotFoundError,
    SourceVersionConflictError,
)
from rag_engine.repositories.pagination import DEFAULT_PAGE_SIZE, Page

_ACTIVE = {"status": ResourceStatus.ACTIVE.value}


class SourceRepository(BaseRepository[Source]):
    """Tenant- and KB-scoped. Sources being deleted are invisible to reads.

    A source is identified by its seed URL within a KB; callers pass the
    *normalized* URL (Step 23) so trivially different spellings map to one source.
    """

    model: ClassVar[type[MongoDocument]] = Source
    not_found_error: ClassVar[type[DocumentNotFoundError]] = SourceNotFoundError

    @staticmethod
    def _scope(tenant_id: str, kb_id: str) -> dict[str, str]:
        return {"tenant_id": tenant_id, "kb_id": kb_id}

    async def get_or_create(self, tenant_id: str, kb_id: str, seed_url: str) -> tuple[Source, bool]:
        """Return the active source for this seed URL, creating it if needed.

        Race-safe: if a concurrent request inserts the same seed first, the unique
        index rejects ours and we return theirs. Returns `(source, created)`.
        """
        scope = self._scope(tenant_id, kb_id)
        existing = await self.find_one({"seed_url": seed_url, **_ACTIVE}, scope=scope)
        if existing is not None:
            return existing, False
        try:
            created = await self.insert(Source(tenant_id=tenant_id, kb_id=kb_id, seed_url=seed_url))
            return created, True
        except DuplicateDocumentError:
            winner = await self.find_one({"seed_url": seed_url, **_ACTIVE}, scope=scope)
            if winner is None:  # the winner was deleted in between; extremely unlikely
                raise
            return winner, False

    async def get_active(self, source_id: str, tenant_id: str, kb_id: str) -> Source:
        source = await self.find_one(
            {"_id": source_id, **_ACTIVE}, scope=self._scope(tenant_id, kb_id)
        )
        if source is None:
            raise SourceNotFoundError(details={"id": source_id})
        return source

    async def list_active(
        self,
        tenant_id: str,
        kb_id: str,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> Page[Source]:
        return await self.paginate(
            _ACTIVE, scope=self._scope(tenant_id, kb_id), limit=limit, cursor=cursor
        )

    async def count_active(self, tenant_id: str, kb_id: str) -> int:
        return await self.count(_ACTIVE, scope=self._scope(tenant_id, kb_id))

    async def mark_deleting(self, source_id: str, tenant_id: str, kb_id: str) -> Source:
        """Flip active → deleting atomically (frees the seed URL); repeat → 404."""
        scope = self._scope(tenant_id, kb_id)
        result = await self.collection.update_one(
            self._where({"_id": source_id, **_ACTIVE}, scope),
            {"$set": {"status": ResourceStatus.DELETING.value, "updated_at": utc_now()}},
        )
        if result.modified_count != 1:
            raise SourceNotFoundError(details={"id": source_id})
        return await self.get_or_raise(source_id, scope=scope)

    async def set_last_job(self, source_id: str, job_id: str) -> None:
        """Record the source's most recent crawl job (Step 79's submission)."""
        try:
            await self.collection.update_one(
                {"_id": source_id}, {"$set": {"last_job_id": job_id, "updated_at": utc_now()}}
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc

    # ---- versions (Step 73) ---- #

    async def activate_version(
        self,
        source_id: str,
        version: str,
        *,
        stats: SourceStats,
        check_previous: bool = False,
        expected_previous: str | None = None,
    ) -> str | None:
        """**The atomic swap**: make `version` the one search serves; returns the previous one.

        A single conditional update, so a search sees either the old version or the new one,
        never neither. Refused (`SourceVersionConflictError`) for a source being deleted, and,
        with `check_previous`, when the active version is no longer `expected_previous`
        (compare-and-swap: a stale finalizer can't overwrite a newer flip).
        """
        condition: dict[str, object] = {"_id": source_id, **_ACTIVE}
        if check_previous:
            condition["active_version"] = expected_previous
        now = utc_now()
        try:
            before = await self.collection.find_one_and_update(
                condition,
                {
                    "$set": {
                        "active_version": version,
                        "stats": stats.model_dump(),
                        "updated_at": now,
                    }
                },
                return_document=ReturnDocument.BEFORE,
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
        if before is None:
            current = await self.get(source_id)
            if current is None:
                raise SourceNotFoundError(details={"id": source_id})
            raise SourceVersionConflictError(
                details={
                    "id": source_id,
                    "status": current.status.value,
                    "active_version": current.active_version,
                    "expected_previous": expected_previous if check_previous else None,
                }
            )
        return before.get("active_version")

    async def active_versions(
        self, tenant_id: str, kb_id: str, *, source_ids: tuple[str, ...] = ()
    ) -> dict[str, str]:
        """`source_id → active version` for the KB's live sources that have one."""
        filter_: dict[str, object] = {**_ACTIVE, "active_version": {"$ne": None}}
        if source_ids:
            filter_["_id"] = {"$in": list(source_ids)}
        try:
            cursor = self.collection.find(
                self._where(filter_, self._scope(tenant_id, kb_id)),
                projection={"active_version": 1},
            )
            return {raw["_id"]: raw["active_version"] async for raw in cursor}
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc
