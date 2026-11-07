"""Domain models & enums: ids, field validation, cross-field invariants (no database)."""

from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from rag_engine.config import settings
from rag_engine.domain import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    ApiKey,
    Chunk,
    ContentKind,
    CrawlConfig,
    CrawlJob,
    EmbeddingModelName,
    JobError,
    JobStatus,
    KnowledgeBase,
    MongoDocument,
    Page,
    PageReason,
    PageStatus,
    Source,
    Tenant,
    TenantQuotas,
    registered_documents,
    utc_now,
)
from rag_engine.domain.types import MAX_URL_LENGTH
from tests.support.fixture_site import EXPECTED_PAGE_OUTCOME, Expect

TEN = Tenant(name="Acme").id
KB = KnowledgeBase(
    tenant_id=TEN, name="docs", embedding_model="text-embedding-3-small", embedding_dimensions=1536
).id
SRC = Source(tenant_id=TEN, kb_id=KB, seed_url="https://acme.example/").id
JOB = CrawlJob(tenant_id=TEN, kb_id=KB, source_id=SRC, seed_url="https://acme.example/").id
HASH = "a" * 64


def _page(**overrides: Any) -> Page:
    fields: dict[str, Any] = {
        "tenant_id": TEN,
        "kb_id": KB,
        "source_id": SRC,
        "job_id": JOB,
        "url": "https://acme.example/docs",
        "url_hash": HASH,
    }
    return Page(**(fields | overrides))


def _job(**overrides: Any) -> CrawlJob:
    fields: dict[str, Any] = {
        "tenant_id": TEN,
        "kb_id": KB,
        "source_id": SRC,
        "seed_url": "https://acme.example/",
    }
    return CrawlJob(**(fields | overrides))


def _samples() -> list[MongoDocument]:
    page = _page(status="succeeded", content_kind="html", markdown="# Docs")
    return [
        Tenant(name="Acme"),
        ApiKey(tenant_id=TEN, key_prefix="AbCd1234", key_hash=HASH),
        KnowledgeBase(
            tenant_id=TEN,
            name="docs",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
        ),
        Source(tenant_id=TEN, kb_id=KB, seed_url="https://acme.example/"),
        _job(),
        page,
        Chunk(
            tenant_id=TEN,
            kb_id=KB,
            source_id=SRC,
            version=JOB,
            page_id=page.id,
            url=page.url,
            ordinal=0,
            text="t",
            embed_text="Docs\nt",
            token_count=3,
            content_hash=HASH,
        ),
    ]


# ---- #
# Tests
# ---- #


class TestDomainModels_S13:
    """S13 — Domain models & enums: Tenant, ApiKey, KB, Source, CrawlJob, Page, Chunk."""

    def test_all_seven_models_are_registered_with_distinct_names(self) -> None:
        models = {m.__name__: m for m in registered_documents()}
        core = [Tenant, ApiKey, KnowledgeBase, Source, CrawlJob, Page, Chunk]

        assert all(m.__name__ in models for m in core)
        assert len({m.collection_name for m in core}) == 7
        assert len({m.id_prefix for m in core}) == 7

    @pytest.mark.parametrize("document", _samples(), ids=lambda d: type(d).__name__)
    def test_round_trip_through_bson_shape(self, document: MongoDocument) -> None:
        raw = document.to_mongo()

        assert raw["_id"].startswith(f"{document.id_prefix}_")
        assert type(document).from_mongo(raw) == document

    def test_foreign_ids_must_have_the_right_prefix(self) -> None:
        with pytest.raises(ValidationError, match="kb_id"):
            Source(tenant_id=TEN, kb_id=TEN, seed_url="https://acme.example/")
        with pytest.raises(ValidationError, match="tenant_id"):
            KnowledgeBase(
                tenant_id="ten_123",
                name="x",
                embedding_model="text-embedding-3-small",
                embedding_dimensions=8,
            )

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://acme.example/",
            "https:///no-host",
            "/relative/path",
            "https://acme.example/with space",
            "https://user:pw@acme.example/",
            "https://acme.example/" + "a" * MAX_URL_LENGTH,
            "javascript:alert(1)",
        ],
    )
    def test_invalid_urls_rejected(self, url: str) -> None:
        with pytest.raises(ValidationError):
            Source(tenant_id=TEN, kb_id=KB, seed_url=url)

    def test_valid_url_kept_verbatim(self) -> None:
        url = "HTTPS://Acme.Example:443/Docs/?b=2&a=1#frag"

        assert Source(tenant_id=TEN, kb_id=KB, seed_url=url).seed_url == url

    def test_tenant_defaults_come_from_settings(self) -> None:
        tenant = Tenant(name="  Acme  ")

        assert tenant.name == "Acme"
        assert tenant.is_active
        assert tenant.default_embedding_model is settings.DEFAULT_EMBEDDING_MODEL
        assert tenant.quotas.max_pages_per_job == settings.TENANT_DEFAULT_MAX_PAGES_PER_JOB
        assert tenant.quotas.max_concurrent_jobs == settings.TENANT_DEFAULT_MAX_CONCURRENT_JOBS

    def test_tenant_name_and_quotas_validated(self) -> None:
        with pytest.raises(ValidationError):
            Tenant(name="   ")
        with pytest.raises(ValidationError, match="service limit"):
            TenantQuotas(max_pages_per_job=settings.CRAWL_HARD_MAX_PAGES + 1)
        with pytest.raises(ValidationError):
            TenantQuotas(max_concurrent_jobs=0)

    def test_api_key_hash_hidden_from_repr(self) -> None:
        key = ApiKey(tenant_id=TEN, key_prefix="AbCd1234", key_hash=HASH)

        assert HASH not in repr(key)

    @pytest.mark.parametrize("prefix", ["short", "has_underscore1", "x" * 17, "bad-char!"])
    def test_api_key_prefix_format(self, prefix: str) -> None:
        with pytest.raises(ValidationError):
            ApiKey(tenant_id=TEN, key_prefix=prefix, key_hash=HASH)

    def test_api_key_usability(self) -> None:
        now = utc_now()
        active = ApiKey(tenant_id=TEN, key_prefix="AbCd1234", key_hash=HASH)
        revoked = active.model_copy(update={"revoked_at": now - timedelta(seconds=1)})
        expiring = ApiKey(
            tenant_id=TEN,
            key_prefix="AbCd1235",
            key_hash=HASH,
            expires_at=now + timedelta(hours=1),
        )

        assert active.is_usable(now)
        assert not revoked.is_usable(now)
        assert expiring.is_usable(now)
        assert not expiring.is_usable(now + timedelta(hours=2))

    def test_api_key_expiry_must_follow_creation(self) -> None:
        with pytest.raises(ValidationError, match="expires_at"):
            ApiKey(
                tenant_id=TEN,
                key_prefix="AbCd1234",
                key_hash=HASH,
                expires_at=utc_now() - timedelta(days=1),
            )

    def test_knowledge_base_embedding_model_is_locked(self) -> None:
        kb = KnowledgeBase(
            tenant_id=TEN,
            name="docs",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
        )

        with pytest.raises(ValidationError, match="frozen"):
            kb.embedding_model = EmbeddingModelName.TEXT_EMBEDDING_3_SMALL  # type: ignore
        with pytest.raises(ValidationError, match="frozen"):
            kb.embedding_dimensions = 1536  # type: ignore
        with pytest.raises(ValidationError):
            KnowledgeBase(
                tenant_id=TEN, name="x", embedding_model="ada-002", embedding_dimensions=8
            )

    def test_source_active_version_must_be_a_job_id(self) -> None:
        source = Source(tenant_id=TEN, kb_id=KB, seed_url="https://acme.example/")

        assert source.active_version is None and source.stats.pages == 0
        source.active_version = JOB
        with pytest.raises(ValidationError):
            source.active_version = "v2"

    def test_job_defaults_and_version(self) -> None:
        job = _job()

        assert job.status is JobStatus.QUEUED
        assert job.version == job.id
        assert job.config.max_pages == settings.CRAWL_DEFAULT_MAX_PAGES
        assert job.config.max_depth == settings.CRAWL_DEFAULT_MAX_DEPTH
        assert job.counters.discovered == 0 and job.usage.tokens == 0

    @pytest.mark.parametrize(
        "config",
        [
            {"max_pages": settings.CRAWL_HARD_MAX_PAGES + 1},
            {"max_pages": 0},
            {"max_depth": settings.CRAWL_HARD_MAX_DEPTH + 1},
            {"include_patterns": ["/x"] * 51},
            {"exclude_patterns": [""]},
            {"webhook_url": "ftp://hooks.example/"},
        ],
    )
    def test_crawl_config_limits(self, config: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            CrawlConfig(**config)

    def test_job_status_invariants(self) -> None:
        now = utc_now()
        with pytest.raises(ValidationError, match="error"):
            _job(status="failed", finished_at=now)
        with pytest.raises(ValidationError, match="finished_at"):
            _job(status="succeeded")
        with pytest.raises(ValidationError, match="terminal"):
            _job(status="running", finished_at=now)

        failed = _job(status="failed", finished_at=now, error=JobError(code="x", message="y"))
        assert failed.status.is_terminal

    def test_job_status_sets_partition_all_statuses(self) -> None:
        assert set(JobStatus) == ACTIVE_JOB_STATUSES | TERMINAL_JOB_STATUSES
        assert not ACTIVE_JOB_STATUSES & TERMINAL_JOB_STATUSES
        assert JobStatus.FINALIZING in ACTIVE_JOB_STATUSES

    def test_page_outcome_invariants(self) -> None:
        with pytest.raises(ValidationError, match="needs a reason"):
            _page(status="skipped")
        with pytest.raises(ValidationError, match="only set"):
            _page(status="queued", reason="timeout")
        with pytest.raises(ValidationError, match="duplicate_of"):
            _page(status="duplicate")
        with pytest.raises(ValidationError, match="content_kind"):
            _page(status="succeeded", markdown="x")
        with pytest.raises(ValidationError):
            _page(http_status=700)

        assert _page(status="failed", reason="timeout").reason is PageReason.TIMEOUT

    def test_every_fixture_site_outcome_is_representable(self) -> None:
        assert set(EXPECTED_PAGE_OUTCOME) == set(Expect)
        for status, reason in EXPECTED_PAGE_OUTCOME.values():
            extra: dict[str, Any] = {}
            if status is PageStatus.SUCCEEDED:
                extra = {"content_kind": ContentKind.HTML, "markdown": "x"}
            if status is PageStatus.DUPLICATE:
                extra = {"duplicate_of": _page().id}
            _page(status=status, reason=reason, **extra)

    def test_chunk_requires_content(self) -> None:
        base: dict[str, Any] = {
            "tenant_id": TEN,
            "kb_id": KB,
            "source_id": SRC,
            "version": JOB,
            "page_id": _page().id,
            "url": "https://acme.example/",
            "ordinal": 0,
            "embed_text": "x",
            "token_count": 1,
            "content_hash": HASH,
        }
        with pytest.raises(ValidationError):
            Chunk.model_validate(base | {"text": ""})
        with pytest.raises(ValidationError):
            Chunk.model_validate(base | {"token_count": 0, "text": "x"})
        with pytest.raises(ValidationError):
            Chunk.model_validate(base | {"version": SRC, "text": "x"})

    def test_enum_values_are_a_stable_contract(self) -> None:
        # these strings are stored in Mongo and returned by the API: renaming is a migration
        assert [s.value for s in JobStatus] == [
            "queued",
            "running",
            "finalizing",
            "succeeded",
            "failed",
            "cancelled",
        ]
        assert [s.value for s in PageStatus] == [
            "queued",
            "in_progress",
            "succeeded",
            "skipped",
            "duplicate",
            "failed",
        ]
        assert [k.value for k in ContentKind] == ["html", "pdf", "docx", "pptx", "xlsx"]
        assert len(PageReason) == 25
