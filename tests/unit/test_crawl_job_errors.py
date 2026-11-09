"""How a failed crawl run is classified: (code, transient?)."""

import pytest
from celery.exceptions import SoftTimeLimitExceeded

from rag_engine.embeddings import (
    EmbeddingAuthError,
    EmbeddingRateLimitedError,
    EmbeddingRequestError,
    EmbeddingUnavailableError,
)
from rag_engine.infrastructure import (
    DatabaseUnavailableError,
    RedisUnavailableError,
    VectorStoreUnavailableError,
)
from rag_engine.repositories import SourceVersionConflictError
from rag_engine.worker.crawl_job import _classify, _JobRejectedError

# ---- #
# Tests
# ---- #


class TestCrawlJobErrors_S77:
    """S77 — run_crawl_job error capture: what fails a job, what requeues it."""

    @pytest.mark.parametrize(
        ("error", "code", "transient"),
        [
            (EmbeddingAuthError(), "embedding_auth_failed", False),
            (EmbeddingRequestError(), "embedding_request_rejected", False),
            (EmbeddingUnavailableError(), "embedding_unavailable", True),
            (EmbeddingRateLimitedError(retry_after=900), "embedding_rate_limited", True),
            (DatabaseUnavailableError(), "database_unavailable", True),
            (RedisUnavailableError(), "redis_unavailable", True),
            (VectorStoreUnavailableError(), "vector_store_unavailable", True),
            (SourceVersionConflictError(), "source_version_conflict", False),
            (_JobRejectedError("source_deleted", "gone"), "source_deleted", False),
            (SoftTimeLimitExceeded(), "time_limit", False),
            (KeyError("boom"), "internal_error", False),
        ],
    )
    def test_classification(self, error: Exception, code: str, transient: bool) -> None:
        got_code, message, got_transient = _classify(error)

        assert (got_code, got_transient) == (code, transient)
        assert message
