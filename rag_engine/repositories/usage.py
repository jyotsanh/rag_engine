"""Tenant embedding usage per month, purpose and model (Step 65)."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from rag_engine.config import EmbeddingModelName
from rag_engine.domain import (
    MongoDocument,
    TenantUsage,
    UsagePurpose,
    tenant_usage_id,
    usage_period,
    utc_now,
)
from rag_engine.infrastructure.exceptions import DatabaseUnavailableError
from rag_engine.infrastructure.mongo import MongoDatabase
from rag_engine.repositories.base import BaseRepository


@dataclass(frozen=True, slots=True)
class UsageTotals:
    requests: int = 0
    tokens: int = 0
    cached: int = 0


class UsageRepository(BaseRepository[TenantUsage]):
    model: ClassVar[type[MongoDocument]] = TenantUsage

    def __init__(self, db: MongoDatabase, *, clock: Callable[[], datetime] = utc_now) -> None:
        super().__init__(db)
        self._clock = clock

    async def record(
        self,
        tenant_id: str,
        purpose: UsagePurpose,
        model: EmbeddingModelName,
        *,
        requests: int,
        tokens: int,
        cached: int = 0,
    ) -> None:
        """`$inc` this month's row (created on first use; concurrent writers share it)."""
        if requests < 0 or tokens < 0 or cached < 0:
            raise ValueError("usage increments must be >= 0")
        now = self._clock()
        period = usage_period(now)
        try:
            await self.collection.update_one(
                {"_id": tenant_usage_id(tenant_id, period, purpose, model)},
                {
                    "$inc": {"requests": requests, "tokens": tokens, "cached": cached},
                    "$set": {"updated_at": now},
                    "$setOnInsert": {
                        "tenant_id": tenant_id,
                        "period": period,
                        "purpose": purpose.value,
                        "model": model.value,
                        "created_at": now,
                    },
                },
                upsert=True,
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def for_tenant(self, tenant_id: str, *, period: str | None = None) -> list[TenantUsage]:
        """A tenant's rows, newest month first (one month with `period`)."""
        filter_ = {"tenant_id": tenant_id, **({"period": period} if period else {})}
        return await self.find(
            filter_,
            sort=[("period", DESCENDING), ("purpose", ASCENDING), ("model", ASCENDING)],
        )

    async def totals(
        self, tenant_id: str, period: str, *, purpose: UsagePurpose | None = None
    ) -> UsageTotals:
        """Summed over models (and purposes unless one is given) for a month."""
        rows = await self.for_tenant(tenant_id, period=period)
        rows = [r for r in rows if purpose is None or r.purpose is purpose]
        return UsageTotals(
            requests=sum(r.requests for r in rows),
            tokens=sum(r.tokens for r in rows),
            cached=sum(r.cached for r in rows),
        )
