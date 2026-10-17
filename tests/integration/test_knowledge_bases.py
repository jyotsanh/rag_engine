"""KnowledgeBaseRepository + KnowledgeBaseService against real MongoDB."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.domain import (
    EmbeddingModelName,
    KnowledgeBase,
    ResourceStatus,
    Tenant,
    TenantQuotas,
)
from rag_engine.embeddings import (
    UnsupportedEmbeddingDimensionsError,
    UnsupportedEmbeddingModelError,
)
from rag_engine.exceptions import AppValidationError, QuotaExceededError
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import (
    KnowledgeBaseNameTakenError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRepository,
    TenantRepository,
)
from rag_engine.services import KnowledgeBaseService

SMALL = EmbeddingModelName.TEXT_EMBEDDING_3_SMALL
LARGE = EmbeddingModelName.TEXT_EMBEDDING_3_LARGE


@pytest.fixture
async def mongo(integration_settings: Settings) -> AsyncIterator[MongoConnection]:
    connection = MongoConnection(integration_settings)
    await connection.connect()
    await connection.ensure_indexes([Tenant, KnowledgeBase])
    yield connection
    await connection.close()


@pytest.fixture
def repo(mongo: MongoConnection) -> KnowledgeBaseRepository:
    return KnowledgeBaseRepository(mongo.db)


@pytest.fixture
def service(repo: KnowledgeBaseRepository) -> KnowledgeBaseService:
    return KnowledgeBaseService(repo)


async def _tenant(mongo: MongoConnection, name: str = "Acme", **fields) -> Tenant:
    return await TenantRepository(mongo.db).create(Tenant(name=name, **fields))


# ---- #
# Tests
# ---- #


class TestKnowledgeBaseService_S19:
    """S19 — KnowledgeBase repo + service: model locking, quotas, tenant scoping, lifecycle."""

    async def test_create_uses_tenant_default_model_at_native_size(
        self, service: KnowledgeBaseService, mongo: MongoConnection
    ) -> None:
        tenant = await _tenant(mongo, default_embedding_model=LARGE)

        kb = await service.create(tenant, name="docs", description="Product docs")

        assert (kb.embedding_model, kb.embedding_dimensions) == (LARGE, 3072)
        assert kb.tenant_id == tenant.id and kb.status is ResourceStatus.ACTIVE

    async def test_explicit_model_and_reduced_dimensions(
        self, service: KnowledgeBaseService, mongo: MongoConnection
    ) -> None:
        tenant = await _tenant(mongo, default_embedding_model=SMALL)

        kb = await service.create(
            tenant, name="docs", embedding_model=LARGE, embedding_dimensions=1024
        )

        assert kb.embedding_space.key == "text-embedding-3-large__1024"

    async def test_invalid_model_or_dimensions_are_typed_422s(
        self, service: KnowledgeBaseService, mongo: MongoConnection, repo: KnowledgeBaseRepository
    ) -> None:
        tenant = await _tenant(mongo)

        with pytest.raises(UnsupportedEmbeddingDimensionsError) as dims:
            await service.create(tenant, name="a", embedding_model=SMALL, embedding_dimensions=3072)
        with pytest.raises(UnsupportedEmbeddingModelError):
            await service.create(tenant, name="b", embedding_model="text-embedding-ada-002")

        assert dims.value.status_code == 422
        assert dims.value.details["max"] == 1536
        assert await repo.count_active(tenant.id) == 0

    async def test_names_unique_per_tenant_and_freed_by_delete(
        self, service: KnowledgeBaseService, mongo: MongoConnection
    ) -> None:
        tenant, other = await _tenant(mongo, "A"), await _tenant(mongo, "B")
        first = await service.create(tenant, name="docs")
        await service.create(other, name="docs")

        with pytest.raises(KnowledgeBaseNameTakenError) as exc:
            await service.create(tenant, name="docs")
        await service.delete(tenant, first.id)
        reused = await service.create(tenant, name="docs")

        assert exc.value.status_code == 409
        assert reused.id != first.id

    async def test_quota_counts_only_active_kbs(
        self, service: KnowledgeBaseService, mongo: MongoConnection
    ) -> None:
        tenant = await _tenant(mongo, quotas=TenantQuotas(max_knowledge_bases=2))
        a = await service.create(tenant, name="a")
        await service.create(tenant, name="b")

        with pytest.raises(QuotaExceededError) as exc:
            await service.create(tenant, name="c")
        await service.delete(tenant, a.id)
        await service.create(tenant, name="c")

        assert exc.value.details == {"quota": "max_knowledge_bases", "limit": 2}

    async def test_concurrent_creates_never_exceed_quota(
        self, service: KnowledgeBaseService, mongo: MongoConnection, repo: KnowledgeBaseRepository
    ) -> None:
        tenant = await _tenant(mongo, quotas=TenantQuotas(max_knowledge_bases=2))

        results = await asyncio.gather(
            *(service.create(tenant, name=f"kb{i}") for i in range(8)), return_exceptions=True
        )

        failures = [r for r in results if isinstance(r, BaseException)]
        assert all(isinstance(f, QuotaExceededError) for f in failures)
        assert 1 <= await repo.count_active(tenant.id) <= 2
        assert await repo.count_active(tenant.id) == len(results) - len(failures)

    async def test_get_is_tenant_scoped_and_hides_deleting(
        self, service: KnowledgeBaseService, mongo: MongoConnection
    ) -> None:
        tenant, other = await _tenant(mongo, "A"), await _tenant(mongo, "B")
        kb = await service.create(tenant, name="docs")

        assert (await service.get(tenant, kb.id)).id == kb.id
        with pytest.raises(KnowledgeBaseNotFoundError):
            await service.get(other, kb.id)
        await service.delete(tenant, kb.id)
        with pytest.raises(KnowledgeBaseNotFoundError):
            await service.get(tenant, kb.id)

    async def test_list_is_active_only_tenant_scoped_and_paginated(
        self, service: KnowledgeBaseService, mongo: MongoConnection
    ) -> None:
        tenant, other = await _tenant(mongo, "A"), await _tenant(mongo, "B")
        kbs = [await service.create(tenant, name=f"kb{i}") for i in range(4)]
        await service.create(other, name="foreign")
        await service.delete(tenant, kbs[0].id)

        first = await service.list(tenant, limit=2, cursor=None)
        second = await service.list(tenant, limit=2, cursor=first.next_cursor)

        assert [k.id for k in first.items + second.items] == [k.id for k in kbs[1:]][::-1]
        assert second.next_cursor is None

    async def test_update_name_and_description_only(
        self, service: KnowledgeBaseService, mongo: MongoConnection
    ) -> None:
        tenant = await _tenant(mongo)
        kb = await service.create(tenant, name="docs")
        await service.create(tenant, name="taken")

        updated = await service.update(tenant, kb.id, {"name": "handbook", "description": "HR"})
        with pytest.raises(AppValidationError) as locked:
            await service.update(tenant, kb.id, {"embedding_dimensions": 512})
        with pytest.raises(KnowledgeBaseNameTakenError):
            await service.update(tenant, kb.id, {"name": "taken"})

        assert (updated.name, updated.description) == ("handbook", "HR")
        assert locked.value.details == {"fields": ["embedding_dimensions"]}

    async def test_deleting_kb_cannot_be_updated_or_deleted_again(
        self, service: KnowledgeBaseService, mongo: MongoConnection, repo: KnowledgeBaseRepository
    ) -> None:
        tenant = await _tenant(mongo)
        kb = await service.create(tenant, name="docs")

        deleted = await service.delete(tenant, kb.id)

        assert deleted.status is ResourceStatus.DELETING
        with pytest.raises(KnowledgeBaseNotFoundError):
            await service.update(tenant, kb.id, {"description": "x"})
        with pytest.raises(KnowledgeBaseNotFoundError):
            await service.delete(tenant, kb.id)
        assert (await repo.get_or_raise(kb.id)).status is ResourceStatus.DELETING  # data kept

    async def test_other_tenant_cannot_delete(
        self, service: KnowledgeBaseService, mongo: MongoConnection
    ) -> None:
        tenant, other = await _tenant(mongo, "A"), await _tenant(mongo, "B")
        kb = await service.create(tenant, name="docs")

        with pytest.raises(KnowledgeBaseNotFoundError):
            await service.delete(other, kb.id)
        assert (await service.get(tenant, kb.id)).status is ResourceStatus.ACTIVE
