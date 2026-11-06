"""Domain models against real MongoDB: round-trips and unique-index semantics."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pymongo.errors import DuplicateKeyError

from rag_engine.config import Settings
from rag_engine.domain import (
    TENANT_NAME_COLLATION,
    ApiKey,
    Chunk,
    CrawlJob,
    KnowledgeBase,
    MongoDocument,
    Page,
    ResourceStatus,
    Source,
    Tenant,
)
from rag_engine.infrastructure import MongoConnection, MongoDatabase

HASH = "b" * 64


@pytest.fixture
async def db(integration_settings: Settings) -> AsyncIterator[MongoDatabase]:
    connection = MongoConnection(integration_settings)
    await connection.connect()
    await connection.ensure_indexes([Tenant, ApiKey, KnowledgeBase, Source, CrawlJob, Page, Chunk])
    yield connection.db
    await connection.close()


async def _insert(db: MongoDatabase, document: MongoDocument) -> None:
    await db[document.collection_name].insert_one(document.to_mongo())


def _kb(tenant_id: str, name: str, **extra: Any) -> KnowledgeBase:
    return KnowledgeBase(
        tenant_id=tenant_id,
        name=name,
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        **extra,
    )


# ---- #
# Tests
# ---- #


class TestDomainIndexes_S13:
    """S13 — Domain models & enums: BSON round-trips and index constraints in MongoDB."""

    async def test_every_model_round_trips(self, db: MongoDatabase) -> None:
        tenant = Tenant(name="Acme")
        kb = _kb(tenant.id, "docs")
        source = Source(tenant_id=tenant.id, kb_id=kb.id, seed_url="https://acme.example/")
        job = CrawlJob(
            tenant_id=tenant.id, kb_id=kb.id, source_id=source.id, seed_url=source.seed_url
        )
        page = Page(
            tenant_id=tenant.id,
            kb_id=kb.id,
            source_id=source.id,
            job_id=job.id,
            url="https://acme.example/",
            url_hash=HASH,
            status="succeeded",
            content_kind="html",
            markdown="# Acme",
        )
        documents: list[MongoDocument] = [
            tenant,
            ApiKey(tenant_id=tenant.id, key_prefix="AbCd1234", key_hash=HASH),
            kb,
            source,
            job,
            page,
            Chunk(
                tenant_id=tenant.id,
                kb_id=kb.id,
                source_id=source.id,
                version=job.id,
                page_id=page.id,
                url=page.url,
                ordinal=0,
                text="Acme",
                embed_text="Acme\nAcme",
                token_count=2,
                content_hash=HASH,
            ),
        ]

        for document in documents:
            await _insert(db, document)
            raw = await db[document.collection_name].find_one({"_id": document.id})
            assert raw is not None
            assert type(document).from_mongo(raw) == document

    async def test_tenant_names_unique_case_insensitively(self, db: MongoDatabase) -> None:
        await _insert(db, Tenant(name="Acme"))

        with pytest.raises(DuplicateKeyError):
            await _insert(db, Tenant(name="ACME"))
        found = await db["tenants"].find_one({"name": "aCmE"}, collation=TENANT_NAME_COLLATION)
        assert found is not None

    async def test_api_key_prefix_is_globally_unique(self, db: MongoDatabase) -> None:
        t1, t2 = Tenant(name="A"), Tenant(name="B")
        await _insert(db, ApiKey(tenant_id=t1.id, key_prefix="SamePref", key_hash=HASH))

        with pytest.raises(DuplicateKeyError):
            await _insert(db, ApiKey(tenant_id=t2.id, key_prefix="SamePref", key_hash=HASH))

    async def test_kb_name_unique_per_tenant_only_while_active(self, db: MongoDatabase) -> None:
        tenant, other = Tenant(name="A"), Tenant(name="B")
        old = _kb(tenant.id, "docs")
        await _insert(db, old)
        await _insert(db, _kb(other.id, "docs"))  # other tenant: allowed

        with pytest.raises(DuplicateKeyError):
            await _insert(db, _kb(tenant.id, "docs"))
        await db["knowledge_bases"].update_one(
            {"_id": old.id}, {"$set": {"status": ResourceStatus.DELETING.value}}
        )
        await _insert(db, _kb(tenant.id, "docs"))  # name freed once the old KB is deleting

    async def test_source_seed_unique_per_kb_while_active(self, db: MongoDatabase) -> None:
        tenant = Tenant(name="A")
        kb1, kb2 = _kb(tenant.id, "one"), _kb(tenant.id, "two")
        url = "https://acme.example/"
        await _insert(db, Source(tenant_id=tenant.id, kb_id=kb1.id, seed_url=url))
        await _insert(db, Source(tenant_id=tenant.id, kb_id=kb2.id, seed_url=url))

        with pytest.raises(DuplicateKeyError):
            await _insert(db, Source(tenant_id=tenant.id, kb_id=kb1.id, seed_url=url))
        await _insert(
            db,
            Source(tenant_id=tenant.id, kb_id=kb1.id, seed_url=url, status="deleting"),
        )

    async def test_page_url_hash_unique_per_job(self, db: MongoDatabase) -> None:
        tenant = Tenant(name="A")
        kb = _kb(tenant.id, "x")
        source = Source(tenant_id=tenant.id, kb_id=kb.id, seed_url="https://acme.example/")
        job1, job2 = (
            CrawlJob(
                tenant_id=tenant.id, kb_id=kb.id, source_id=source.id, seed_url=source.seed_url
            )
            for _ in range(2)
        )

        def page(job: CrawlJob) -> Page:
            return Page(
                tenant_id=tenant.id,
                kb_id=kb.id,
                source_id=source.id,
                job_id=job.id,
                url="https://acme.example/a",
                url_hash=HASH,
            )

        await _insert(db, page(job1))
        await _insert(db, page(job2))  # same URL in a later crawl: allowed
        with pytest.raises(DuplicateKeyError):
            await _insert(db, page(job1))
