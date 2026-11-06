"""Settings: required secrets, env parsing, cross-field limits, computed URLs."""

from typing import Any

import pytest
from pydantic import ValidationError

from rag_engine.config import EmbeddingProvider, Environment, Settings, VectorStoreType

_REQUIRED = {
    "ADMIN_API_KEY": "a" * 32,
    "API_KEY_PEPPER": "p" * 32,
    "OPENAI_API_KEY": "sk-test",
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every Settings field from the process env so each test starts blank."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _make(**overrides: Any) -> Settings:
    """Build Settings without reading any `.env` file on disk."""
    return Settings(_env_file=None, **{**_REQUIRED, **overrides})  # type: ignore


# ---- #
# Tests
# ---- #


@pytest.mark.usefixtures("clean_env")
class TestSettings_S3:
    """S3 — Settings: fail-fast secrets, env parsing, limit validation, computed URLs."""

    @pytest.mark.parametrize("missing", sorted(_REQUIRED))
    def test_missing_secret_fails_fast(self, missing: str) -> None:
        values = {k: v for k, v in _REQUIRED.items() if k != missing}

        with pytest.raises(ValidationError) as exc:
            Settings(_env_file=None, **values)  # type: ignore

        assert missing in str(exc.value)

    @pytest.mark.parametrize("name", ["ADMIN_API_KEY", "API_KEY_PEPPER"])
    def test_short_security_secret_rejected(self, name: str) -> None:
        with pytest.raises(ValidationError):
            _make(**{name: "too-short"})

    def test_secrets_are_masked_in_repr(self) -> None:
        settings = _make()

        assert "a" * 32 not in repr(settings)
        assert settings.ADMIN_API_KEY.get_secret_value() == "a" * 32

    def test_defaults_select_active_backends(self) -> None:
        settings = _make()

        assert settings.ENVIRONMENT is Environment.DEV
        assert settings.ACTIVE_VECTOR_STORE is VectorStoreType.QDRANT
        assert settings.ACTIVE_EMBEDDING_PROVIDER is EmbeddingProvider.OPENAI
        assert settings.SSRF_ALLOW_PRIVATE_NETWORKS is False

    def test_env_vars_are_parsed(self, clean_env: pytest.MonkeyPatch) -> None:
        # Arrange
        for key, value in _REQUIRED.items():
            clean_env.setenv(key, value)
        clean_env.setenv("CRAWL_CONCURRENCY", "16")
        clean_env.setenv("SSRF_ALLOWED_PORTS", "[80, 443, 8443]")
        clean_env.setenv("BACKEND_CORS_ORIGINS", '["https://app.example.com"]')
        clean_env.setenv("ENVIRONMENT", "prod")

        # Act
        settings = Settings(_env_file=None)

        # Assert
        assert settings.CRAWL_CONCURRENCY == 16
        assert settings.SSRF_ALLOWED_PORTS == [80, 443, 8443]
        assert settings.BACKEND_CORS_ORIGINS == ["https://app.example.com"]
        assert settings.ENVIRONMENT is Environment.PROD

    def test_empty_env_value_falls_back_to_default(self, clean_env: pytest.MonkeyPatch) -> None:
        for key, value in _REQUIRED.items():
            clean_env.setenv(key, value)
        clean_env.setenv("MONGO_HOST", "")

        settings = Settings(_env_file=None)

        assert settings.MONGO_HOST == "localhost"

    def test_unknown_embedding_model_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make(DEFAULT_EMBEDDING_MODEL="text-embedding-ada-002")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"CRAWL_DEFAULT_MAX_PAGES": 20_000, "CRAWL_HARD_MAX_PAGES": 10_000},
            {"CRAWL_DEFAULT_MAX_DEPTH": 60, "CRAWL_HARD_MAX_DEPTH": 50},
            {"CRAWL_MAX_PAGE_BYTES": 10, "CRAWL_MAX_TOTAL_BYTES": 5},
            {"CELERY_TASK_SOFT_TIME_LIMIT_SECONDS": 10, "CELERY_TASK_TIME_LIMIT_SECONDS": 10},
            {"CRAWL_FETCH_TIMEOUT_SECONDS": 10, "CRAWL_READ_TIMEOUT_SECONDS": 30},
            {"CRAWL_RETRY_BASE_DELAY_SECONDS": 60, "CRAWL_RETRY_MAX_DELAY_SECONDS": 30},
            {"ROBOTS_UNREACHABLE_CACHE_TTL_SECONDS": 600, "ROBOTS_CACHE_TTL_SECONDS": 300},
            {"ROBOTS_CACHE_TTL_SECONDS": 2 * 24 * 60 * 60},
            {"CRAWL_CONCURRENCY": 0},
            {"CRAWL_PER_DOMAIN_BURST": 0},
            {"CRAWL_PER_DOMAIN_RPS": 0},
            {"PROD_PORT": 70_000},
        ],
    )
    def test_inconsistent_limits_rejected(self, overrides: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            _make(**overrides)

    def test_private_networks_forbidden_in_prod(self) -> None:
        with pytest.raises(ValidationError, match="SSRF_ALLOW_PRIVATE_NETWORKS"):
            _make(ENVIRONMENT="prod", SSRF_ALLOW_PRIVATE_NETWORKS=True)

    def test_mongo_url_built_from_parts_with_escaped_credentials(self) -> None:
        settings = _make(
            MONGO_HOST="mongo", MONGO_PORT=27017, MONGO_USER="rag", MONGO_PASSWORD="p@ss/w"
        )

        assert settings.mongo_url == "mongodb://rag:p%40ss%2Fw@mongo:27017"

    def test_mongo_uri_overrides_parts(self) -> None:
        settings = _make(MONGO_HOST="ignored", MONGO_URI="mongodb+srv://u:p@cluster.example")

        assert settings.mongo_url == "mongodb+srv://u:p@cluster.example"

    def test_default_ports_match_compose_host_ports(self) -> None:
        settings = _make()

        assert settings.mongo_url == "mongodb://localhost:27020"
        assert settings.redis_url == "redis://localhost:6380/0"
        assert settings.qdrant_url == "http://localhost:6333"

    def test_redis_and_celery_urls_use_separate_dbs(self) -> None:
        settings = _make(REDIS_HOST="redis", REDIS_PORT=6379, REDIS_PASSWORD="s3cret")

        assert settings.redis_url == "redis://:s3cret@redis:6379/0"
        assert settings.celery_broker_url == "redis://:s3cret@redis:6379/1"
        assert settings.celery_result_backend == "redis://:s3cret@redis:6379/2"

    def test_qdrant_url_scheme_follows_https_flag(self) -> None:
        assert _make().qdrant_url == "http://localhost:6333"
        assert _make(QDRANT_HTTPS=True, QDRANT_HOST="q.example").qdrant_url == (
            "https://q.example:6333"
        )
