"""Qdrant upsert retries with a scripted client (no Qdrant needed)."""

import random

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_engine.domain import chunk_id
from rag_engine.embeddings import SparseVector
from rag_engine.vectorstores import (
    ChunkPayload,
    CollectionSpec,
    QdrantVectorStore,
    VectorPoint,
    VectorStoreError,
    VectorStoreUnavailableError,
)
from tests.conftest import OverrideSettings

_SPEC = CollectionSpec("retry_test", 2)


def _response(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status, "status", b'{"status": {"error": "x"}}', httpx.Headers())


class _ScriptedClient:
    """`upsert` raises the scripted errors in order, then succeeds."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def upsert(self, collection_name: str, points, wait: bool) -> None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)


class _Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _point() -> VectorPoint:
    payload = ChunkPayload(
        tenant_id="ten_1", kb_id="kb_1", source_id="src_1", version="job_1",
        chunk_id=chunk_id("page_1", 0), page_id="page_1", url="https://a.example/",
        ordinal=0, text="x",
    )  # fmt: skip
    return VectorPoint("00000000-0000-5000-8000-000000000000", [1.0, 0.0], payload,
                       SparseVector((1,), (1.0,)))  # fmt: skip


def _store(client: _ScriptedClient, sleeps: _Sleeps, config) -> QdrantVectorStore:
    return QdrantVectorStore(client, config=config, sleep=sleeps, rng=random.Random(69))  # ty: ignore[invalid-argument-type]


# ---- #
# Tests
# ---- #


class TestQdrantUpsertRetries_S69:
    """S69 — Qdrant upsert retries: transient failures retried with backoff, others raised."""

    @pytest.fixture
    def config(self, override_settings: OverrideSettings):
        return override_settings(
            QDRANT_MAX_RETRIES=3,
            QDRANT_RETRY_BASE_DELAY_SECONDS=0.5,
            QDRANT_RETRY_MAX_DELAY_SECONDS=8,
        )

    async def test_transient_failures_are_retried(self, config) -> None:
        client = _ScriptedClient(_response(503), ResponseHandlingException(OSError("reset")))
        sleeps = _Sleeps()

        await _store(client, sleeps, config).upsert(_SPEC, [_point()])

        assert client.calls == 3
        assert 0.25 <= sleeps.delays[0] <= 0.5 and 0.5 <= sleeps.delays[1] <= 1.0

    async def test_retries_run_out(self, config) -> None:
        client = _ScriptedClient(*[_response(502)] * 4)
        sleeps = _Sleeps()

        with pytest.raises(VectorStoreUnavailableError) as exc:
            await _store(client, sleeps, config).upsert(_SPEC, [_point()])

        assert client.calls == 4 and len(sleeps.delays) == 3
        assert exc.value.details["attempts"] == 4

    async def test_a_refused_upsert_is_not_retried(self, config) -> None:
        client = _ScriptedClient(_response(400))
        sleeps = _Sleeps()

        with pytest.raises(VectorStoreError, match="400"):
            await _store(client, sleeps, config).upsert(_SPEC, [_point()])

        assert client.calls == 1 and sleeps.delays == []

    def test_retry_settings_are_validated(self, override_settings: OverrideSettings) -> None:
        with pytest.raises(ValueError):
            override_settings(QDRANT_RETRY_BASE_DELAY_SECONDS=10, QDRANT_RETRY_MAX_DELAY_SECONDS=8)
