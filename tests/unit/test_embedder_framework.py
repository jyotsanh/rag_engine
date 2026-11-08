"""Embedder framework: the base contract every provider inherits, the dispatcher, errors."""

import math
from collections.abc import Iterator

import pytest

from rag_engine.config import EmbeddingModelName, EmbeddingProvider
from rag_engine.embeddings import (
    BaseEmbedder,
    EmbeddingAuthError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingRateLimitedError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
    EmbeddingUsageCounts,
    InputType,
    ProviderResult,
    UnsupportedEmbeddingProviderError,
    get_embedder,
    register_embedder,
    registered_embedders,
    resolve_space,
    tokenizer_for,
)
from rag_engine.embeddings import dispatcher as embedder_dispatcher
from tests.support.fakes import FakeEmbedder, fake_vector

_SMALL = resolve_space(EmbeddingModelName.TEXT_EMBEDDING_3_SMALL)
_REDUCED = resolve_space(EmbeddingModelName.TEXT_EMBEDDING_3_SMALL, 256)


def _embedder(space=_SMALL) -> FakeEmbedder:
    return FakeEmbedder(space)


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(embedder_dispatcher, "_EMBEDDERS", {})
    yield


# ---- #
# Tests
# ---- #


class TestEmbedderFramework_S61:
    """S61 — Embedder framework: `BaseEmbedder` contract, dispatcher, typed errors."""

    # ---- the contract ---- #

    async def test_documents_get_one_vector_each_in_order(self) -> None:
        embedder = _embedder()
        texts = ["first chunk", "second chunk", "first chunk"]

        batch = await embedder.embed_documents(texts)

        assert len(batch) == 3
        assert batch.vectors[0] == fake_vector("first chunk", 1536) == batch.vectors[2]
        assert batch.vectors[1] == fake_vector("second chunk", 1536)
        assert batch.space == _SMALL
        assert batch.usage == EmbeddingUsageCounts(requests=1, tokens=6)
        assert embedder.calls == [(texts, InputType.DOCUMENT)]

    async def test_a_query_is_one_vector(self) -> None:
        embedder = _embedder(_REDUCED)

        batch = await embedder.embed_query("how fast is the WX-2048?")

        assert len(batch.vector) == 256 == embedder.dimensions
        assert math.isclose(math.sqrt(sum(v * v for v in batch.vector)), 1.0)
        assert embedder.calls == [(["how fast is the WX-2048?"], InputType.QUERY)]

    async def test_an_empty_batch_never_calls_the_provider(self) -> None:
        embedder = _embedder()

        batch = await embedder.embed_documents([])

        assert batch.vectors == [] and batch.usage == EmbeddingUsageCounts()
        assert embedder.calls == []
        with pytest.raises(ValueError, match="not 1"):
            _ = batch.vector

    async def test_usage_counts_add_up(self) -> None:
        total = EmbeddingUsageCounts(1, 10) + EmbeddingUsageCounts(2, 5)

        assert total == EmbeddingUsageCounts(requests=3, tokens=15)

    # ---- input validation (before anything is sent) ---- #

    @pytest.mark.parametrize("bad", ["", "   \n\t"])
    async def test_empty_inputs_are_refused(self, bad: str) -> None:
        embedder = _embedder()

        with pytest.raises(EmbeddingInputError) as exc:
            await embedder.embed_documents(["fine", bad])

        assert exc.value.details == {"index": 1}
        assert exc.value.status_code == 422
        assert embedder.calls == []

    async def test_an_input_over_the_model_limit_is_refused(self) -> None:
        embedder = _embedder()
        too_long = "word " * 9000  # > 8191 cl100k tokens

        with pytest.raises(EmbeddingInputError) as exc:
            await embedder.embed_documents(["fine", too_long])

        assert exc.value.details["index"] == 1
        assert exc.value.details["max_tokens"] == 8191
        assert exc.value.details["tokens"] == tokenizer_for(_SMALL.model).count(too_long)
        assert embedder.calls == []

    # ---- response validation ---- #

    @pytest.mark.parametrize(
        ("vectors", "match"),
        [
            (lambda d: [[0.0] * d], "expected 2 vectors, got 1"),
            (lambda d: [[0.0] * d, [0.0] * d, [0.0] * d], "expected 2 vectors, got 3"),
            (lambda d: [[0.0] * d, [0.0] * (d - 1)], "vector 1 has 1535 dimensions"),
            (lambda d: [[0.0] * d, [math.nan] * d], "vector 1 contains non-finite"),
            (lambda d: [[math.inf] * d, [0.0] * d], "vector 0 contains non-finite"),
        ],
    )
    async def test_a_broken_provider_response_is_rejected(self, vectors, match: str) -> None:
        embedder = _embedder()
        embedder.respond = lambda texts, _: ProviderResult(
            vectors(embedder.dimensions), EmbeddingUsageCounts(1, 2)
        )

        with pytest.raises(EmbeddingResponseError, match=match) as exc:
            await embedder.embed_documents(["a", "b"])

        assert exc.value.retryable is False and exc.value.status_code == 502
        assert exc.value.details["model"] == "text-embedding-3-small"

    async def test_provider_errors_pass_through(self) -> None:
        embedder = _embedder()

        def fail(texts: list[str], input_type: InputType) -> ProviderResult:
            raise EmbeddingRateLimitedError(retry_after=7.2)

        embedder.respond = fail

        with pytest.raises(EmbeddingRateLimitedError) as exc:
            await embedder.embed_query("q")

        assert exc.value.retryable and exc.value.retry_after == 7.2
        assert exc.value.headers == {"Retry-After": "8"}

    # ---- errors ---- #

    @pytest.mark.parametrize(
        ("error", "status", "code", "retryable"),
        [
            (EmbeddingAuthError, 502, "embedding_auth_failed", False),
            (EmbeddingRequestError, 502, "embedding_request_rejected", False),
            (EmbeddingResponseError, 502, "embedding_response_invalid", False),
            (EmbeddingRateLimitedError, 503, "embedding_rate_limited", True),
            (EmbeddingUnavailableError, 503, "embedding_unavailable", True),
        ],
    )
    def test_error_taxonomy(
        self, error: type[EmbeddingError], status: int, code: str, retryable: bool
    ) -> None:
        exc = error()

        assert isinstance(exc, EmbeddingError)
        assert (exc.status_code, exc.code, exc.retryable) == (status, code, retryable)
        assert exc.headers == {} and exc.retry_after is None

    # ---- lifecycle and binding ---- #

    async def test_embedders_close_as_context_managers(self) -> None:
        async with _embedder() as embedder:
            await embedder.embed_query("q")

        assert isinstance(embedder, FakeEmbedder) and embedder.closed

    def test_an_embedder_is_bound_to_its_providers_models(self) -> None:
        class _OtherProvider(FakeEmbedder):
            provider = "cohere"  # ty: ignore[invalid-assignment]

        with pytest.raises(UnsupportedEmbeddingProviderError, match="not cohere"):
            _OtherProvider(_SMALL)

    # ---- dispatcher ---- #

    def test_dispatcher_builds_the_active_providers_embedder(self, isolated_registry: None) -> None:
        register_embedder(EmbeddingProvider.OPENAI)(FakeEmbedder)

        embedder = get_embedder(_REDUCED)

        assert isinstance(embedder, FakeEmbedder)
        assert (embedder.space, embedder.dimensions) == (_REDUCED, 256)
        assert registered_embedders() == {EmbeddingProvider.OPENAI: FakeEmbedder}

    def test_an_unregistered_provider_is_a_typed_error(self, isolated_registry: None) -> None:
        with pytest.raises(UnsupportedEmbeddingProviderError, match="no embedder") as exc:
            get_embedder(_SMALL)

        assert exc.value.details == {"provider": "openai", "registered": []}

    def test_registration_is_checked(self, isolated_registry: None) -> None:
        class _Other(FakeEmbedder):
            pass

        register_embedder(EmbeddingProvider.OPENAI)(FakeEmbedder)
        register_embedder(EmbeddingProvider.OPENAI)(FakeEmbedder)  # same class: fine

        with pytest.raises(TypeError, match="already registered"):
            register_embedder(EmbeddingProvider.OPENAI)(_Other)
        with pytest.raises(TypeError, match="not a BaseEmbedder"):
            register_embedder(EmbeddingProvider.OPENAI)(dict)  # ty: ignore[invalid-argument-type]

    def test_registration_requires_a_matching_provider(self, isolated_registry: None) -> None:
        class _Mislabelled(BaseEmbedder):
            provider = "cohere"  # ty: ignore[invalid-assignment]

            async def _embed(self, texts: list[str], input_type: InputType) -> ProviderResult:
                raise NotImplementedError

        with pytest.raises(TypeError, match="provider must be"):
            register_embedder(EmbeddingProvider.OPENAI)(_Mislabelled)

    def test_the_tokenizer_moved_but_chunking_still_exports_it(self) -> None:
        from rag_engine.chunking import tokenizer_for as chunking_tokenizer_for

        assert chunking_tokenizer_for is tokenizer_for
        assert tokenizer_for(_SMALL.model).name == "cl100k_base"
