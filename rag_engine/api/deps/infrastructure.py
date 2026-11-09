"""Dependencies that hand routes the process-wide infrastructure and repositories."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from rag_engine.infrastructure import MongoDatabase
from rag_engine.repositories import (
    ApiKeyRepository,
    JobRepository,
    KnowledgeBaseRepository,
    SourceRepository,
    TenantRepository,
)
from rag_engine.security import ApiKeyHasher, SecretBox


def get_db(request: Request) -> MongoDatabase:
    return request.app.state.infrastructure.mongo.db


@lru_cache(maxsize=1)
def get_api_key_hasher() -> ApiKeyHasher:
    """One hasher per process (peppers are read from settings once)."""
    return ApiKeyHasher()


@lru_cache(maxsize=1)
def get_secret_box() -> SecretBox:
    """One box per process (encryption keys are read from settings once)."""
    return SecretBox.from_settings()


Database = Annotated[MongoDatabase, Depends(get_db)]
Hasher = Annotated[ApiKeyHasher, Depends(get_api_key_hasher)]
Box = Annotated[SecretBox, Depends(get_secret_box)]


def get_tenant_repository(db: Database) -> TenantRepository:
    return TenantRepository(db)


def get_api_key_repository(db: Database) -> ApiKeyRepository:
    return ApiKeyRepository(db)


Tenants = Annotated[TenantRepository, Depends(get_tenant_repository)]
ApiKeys = Annotated[ApiKeyRepository, Depends(get_api_key_repository)]


def get_knowledge_base_repository(db: Database) -> KnowledgeBaseRepository:
    return KnowledgeBaseRepository(db)


KnowledgeBases = Annotated[KnowledgeBaseRepository, Depends(get_knowledge_base_repository)]


def get_source_repository(db: Database) -> SourceRepository:
    return SourceRepository(db)


Sources = Annotated[SourceRepository, Depends(get_source_repository)]


def get_job_repository(db: Database) -> JobRepository:
    return JobRepository(db)


Jobs = Annotated[JobRepository, Depends(get_job_repository)]
