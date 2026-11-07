from typing import ClassVar

from rag_engine.domain import ContentHashClaim, MongoDocument
from rag_engine.repositories.base import BaseRepository
from rag_engine.repositories.exceptions import DuplicateDocumentError


class ContentClaimRepository(BaseRepository[ContentHashClaim]):
    """Atomic "first page with this content in this job" claims (Step 46)."""

    model: ClassVar[type[MongoDocument]] = ContentHashClaim

    async def claim(
        self, job_id: str, content_hash: str, page_id: str, url: str
    ) -> ContentHashClaim:
        """The owning claim: ours if we were first, otherwise the existing one.

        Race-safe through the unique index: of two concurrent claims exactly one insert
        succeeds, and the other reads the winner. Re-claiming with the same page (a job
        resumed after a crash) returns that page's own claim.
        """
        filter_ = {"job_id": job_id, "content_hash": content_hash}
        existing = await self.find_one(filter_)
        if existing is not None:
            return existing
        candidate = ContentHashClaim(
            job_id=job_id, content_hash=content_hash, page_id=page_id, url=url
        )
        try:
            return await self.insert(candidate)
        except DuplicateDocumentError:
            winner = await self.find_one(filter_)
            if winner is None:  # deleted between the two calls (job released): retry once
                return await self.insert(candidate)
            return winner

    async def release_job(self, job_id: str) -> int:
        """Drop a finished job's claims (finalize); the TTL index is only a safety net."""
        return await self.delete_many({"job_id": job_id})
