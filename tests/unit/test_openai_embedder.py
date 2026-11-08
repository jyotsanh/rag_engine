"""OpenAI embedder over a mocked REST API (respx): batching, dimensions, order, errors."""

import json
import random
from collections.abc import Iterator
from email.utils import format_datetime

import httpx
import pytest
import respx

from rag_engine.config import Settings
from rag_engine.domain import utc_now
from rag_engine.embeddings import (
    EmbeddingAuthError,
    EmbeddingError,
    EmbeddingRateLimitedError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
    EmbeddingUsageCounts,
    OpenAIEmbedder,
    get_embedder,
    resolve_space,
    retry_after_seconds,
    tokenizer_for,
)
from tests.conftest import OverrideSettings

_URL = "https://api.openai.com/v1/embeddings"
_NATIVE = resolve_space("text-embedding-3-small")
_REDUCED = resolve_space("text-embedding-3-small", 256)
_TOKENIZER = tokenizer_for("text-embedding-3-small")


def _vector(text: str, dimensions: int) -> list[float]:
    rng = random.Random(text)
    return [rng.uniform(-1, 1) for _ in range(dimensions)]


def _ok(request: httpx.Request, *, shuffle: bool = False) -> httpx.Response:
    """A well-formed OpenAI response for the request's inputs (optionally out of order)."""
    body = json.loads(request.content)
    dimensions = body.get("dimensions", 1536)
    data = [
        {"object": "embedding", "index": i, "embedding": _vector(text, dimensions)}
        for i, text in enumerate(body["input"])
    ]
    if shuffle:
        random.Random(0).shuffle(data)
    tokens = sum(_TOKENIZER.count(text) for text in body["input"])
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": data,
            "model": body["model"],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        },
    )


def _sent(route: respx.Route) -> list[dict]:
    return [json.loads(call.request.content) for call in route.calls]


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def config(override_settings: OverrideSettings) -> Settings:
    return override_settings(OPENAI_API_KEY="sk-test-secret-key")


# ---- #
# Tests
# ---- #


class TestOpenAIEmbedder_S62:
    """S62 — OpenAI embedder: request shape, batching by inputs and tokens, order, errors."""

    # ---- request shape ---- #

    async def test_native_space_request(self, api: respx.MockRouter, config: Settings) -> None:
        route = api.post(_URL).mock(side_effect=_ok)

        async with OpenAIEmbedder(_NATIVE, config=config) as embedder:
            batch = await embedder.embed_documents(["alpha", "beta"])

        (request,) = route.calls
        assert request.request.headers["authorization"] == "Bearer sk-test-secret-key"
        assert json.loads(request.request.content) == {
            "model": "text-embedding-3-small",
            "input": ["alpha", "beta"],
            "encoding_format": "float",
        }  # no `dimensions` for the native size
        assert batch.vectors == [_vector("alpha", 1536), _vector("beta", 1536)]
        assert batch.usage == EmbeddingUsageCounts(1, _TOKENIZER.count("alpha") + 1)

    async def test_a_reduced_space_sends_dimensions(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        route = api.post(_URL).mock(side_effect=_ok)

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            batch = await embedder.embed_query("how fast is the WX-2048?")

        assert _sent(route)[0]["dimensions"] == 256
        assert len(batch.vector) == 256

    async def test_base_url_override(
        self, api: respx.MockRouter, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(OPENAI_BASE_URL="https://proxy.internal/openai/v1/")
        route = api.post("https://proxy.internal/openai/v1/embeddings").mock(side_effect=_ok)

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            await embedder.embed_query("q")

        assert route.call_count == 1

    async def test_vectors_come_back_in_input_order(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        api.post(_URL).mock(side_effect=lambda request: _ok(request, shuffle=True))
        texts = [f"text {i}" for i in range(12)]

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            batch = await embedder.embed_documents(texts)

        assert batch.vectors == [_vector(text, 256) for text in texts]

    # ---- batching ---- #

    async def test_batches_by_input_count(
        self, api: respx.MockRouter, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(EMBEDDING_MAX_BATCH_INPUTS=3)
        route = api.post(_URL).mock(side_effect=_ok)
        texts = [f"chunk number {i}" for i in range(7)]

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            batch = await embedder.embed_documents(texts)

        assert [body["input"] for body in _sent(route)] == [texts[0:3], texts[3:6], texts[6:7]]
        assert batch.vectors == [_vector(text, 256) for text in texts]
        assert batch.usage == EmbeddingUsageCounts(
            requests=3, tokens=sum(_TOKENIZER.count(t) for t in texts)
        )

    async def test_batches_by_token_budget(
        self, api: respx.MockRouter, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(EMBEDDING_MAX_BATCH_TOKENS=50)
        route = api.post(_URL).mock(side_effect=_ok)
        texts = [" ".join(["widget"] * n) for n in (10, 20, 15, 30, 5, 49, 1)]

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            batch = await embedder.embed_documents(texts)

        requests = [body["input"] for body in _sent(route)]
        assert [t for request in requests for t in request] == texts  # in order, all sent
        assert all(sum(_TOKENIZER.count(t) for t in r) <= 50 for r in requests)
        expected, current = [], []  # greedy split on the real token counts
        for text in texts:
            if current and sum(map(_TOKENIZER.count, current)) + _TOKENIZER.count(text) > 50:
                expected.append(current)
                current = []
            current.append(text)
        expected.append(current)
        assert requests == expected and len(requests) > 1
        assert len(batch) == len(texts)

    async def test_an_input_over_the_configured_token_budget_goes_alone(
        self, api: respx.MockRouter, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(EMBEDDING_MAX_BATCH_TOKENS=10)
        route = api.post(_URL).mock(side_effect=_ok)
        texts = ["small", " ".join(["widget"] * 40), "small too"]

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            await embedder.embed_documents(texts)

        assert [body["input"] for body in _sent(route)] == [[t] for t in texts]

    def test_model_limits_cap_the_configured_budgets(
        self, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(
            EMBEDDING_MAX_BATCH_INPUTS=10_000, EMBEDDING_MAX_BATCH_TOKENS=1_000_000
        )

        embedder = OpenAIEmbedder(_NATIVE, config=config)

        assert (embedder.max_batch_inputs, embedder.max_batch_tokens) == (2048, 300_000)

    # ---- failures ---- #

    @pytest.mark.parametrize(
        ("status", "body", "headers", "error", "retry_after", "code"),
        [
            (401, {"error": {"message": "Incorrect API key", "code": "invalid_api_key"}}, {},
             EmbeddingAuthError, None, "invalid_api_key"),
            (403, {"error": {"message": "No access", "type": "permission_error"}}, {},
             EmbeddingAuthError, None, "permission_error"),
            (400, {"error": {"message": "Invalid input", "type": "invalid_request_error"}}, {},
             EmbeddingRequestError, None, "invalid_request_error"),
            (404, {"error": {"message": "Model not found", "code": "model_not_found"}}, {},
             EmbeddingRequestError, None, "model_not_found"),
            (429, {"error": {"message": "Slow down", "code": "rate_limit_exceeded"}},
             {"retry-after-ms": "1500"}, EmbeddingRateLimitedError, 1.5, "rate_limit_exceeded"),
            (429, {"error": {"message": "Slow down"}}, {"Retry-After": "20"},
             EmbeddingRateLimitedError, 20.0, None),
            (429, {"error": {"message": "Quota exceeded", "code": "insufficient_quota"}}, {},
             EmbeddingRequestError, None, "insufficient_quota"),
            (500, {"error": {"message": "server error"}}, {},
             EmbeddingUnavailableError, None, None),
            (503, "<html>overloaded</html>", {"Retry-After": "3"},
             EmbeddingUnavailableError, 3.0, None),
        ],
    )  # fmt: skip
    async def test_http_errors_map_to_the_taxonomy(
        self,
        api: respx.MockRouter,
        config: Settings,
        status: int,
        body: object,
        headers: dict[str, str],
        error: type[EmbeddingError],
        retry_after: float | None,
        code: str | None,
    ) -> None:
        content = json.dumps(body) if isinstance(body, dict) else str(body)
        api.post(_URL).mock(return_value=httpx.Response(status, text=content, headers=headers))

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            with pytest.raises(error) as exc:
                await embedder.embed_query("q")

        assert type(exc.value) is error
        assert exc.value.retryable is (error in (EmbeddingRateLimitedError,
                                                 EmbeddingUnavailableError))  # fmt: skip
        assert exc.value.retry_after == retry_after
        assert exc.value.details["status"] == status
        assert exc.value.details.get("error_code") == code
        assert f"OpenAI returned {status}" in exc.value.message
        assert "sk-test-secret-key" not in exc.value.message

    @pytest.mark.parametrize(
        "failure",
        [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), httpx.ConnectError("refused")],
    )
    async def test_transport_failures_are_unavailable(
        self, api: respx.MockRouter, config: Settings, failure: Exception
    ) -> None:
        api.post(_URL).mock(side_effect=failure)

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            with pytest.raises(EmbeddingUnavailableError) as exc:
                await embedder.embed_query("q")

        assert exc.value.retryable

    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            ("not json at all", "not JSON"),
            ({"object": "list"}, "no data list"),
            ({"data": [{"embedding": [0.1] * 256}]}, "without index"),
            ({"data": [{"index": 0, "embedding": [0.1] * 256},
                       {"index": 0, "embedding": [0.1] * 256}]}, "repeated"),
            ({"data": [{"index": 0, "embedding": [0.1] * 256},
                       {"index": 2, "embedding": [0.1] * 256}]}, "not 0..1"),
            ({"data": [{"index": 0, "embedding": ["0.1"] * 256},
                       {"index": 1, "embedding": [0.1] * 256}]}, "not a list of numbers"),
            ({"data": [{"index": 0, "embedding": [True] * 256},
                       {"index": 1, "embedding": [0.1] * 256}]}, "not a list of numbers"),
            ({"data": [{"index": 0, "embedding": [0.1] * 255},
                       {"index": 1, "embedding": [0.1] * 256}]}, "255 dimensions"),
        ],
    )  # fmt: skip
    async def test_malformed_success_responses_are_rejected(
        self, api: respx.MockRouter, config: Settings, payload: object, reason: str
    ) -> None:
        content = json.dumps(payload) if isinstance(payload, dict) else str(payload)
        api.post(_URL).mock(return_value=httpx.Response(200, text=content))

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            with pytest.raises(EmbeddingResponseError, match=reason):
                await embedder.embed_documents(["a", "b"])

    async def test_usage_falls_back_when_missing(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        payload = {"data": [{"index": 0, "embedding": [0.1] * 256}]}
        api.post(_URL).mock(return_value=httpx.Response(200, json=payload))

        async with OpenAIEmbedder(_REDUCED, config=config) as embedder:
            batch = await embedder.embed_query("q")

        assert batch.usage == EmbeddingUsageCounts(requests=1, tokens=0)

    # ---- client ownership and dispatch ---- #

    async def test_a_shared_client_is_not_closed(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        api.post(_URL).mock(side_effect=_ok)
        async with httpx.AsyncClient() as client:
            async with OpenAIEmbedder(_REDUCED, config=config, client=client) as embedder:
                await embedder.embed_query("q")

            assert not client.is_closed  # still usable by its owner

    async def test_an_owned_client_is_closed(self, config: Settings) -> None:
        embedder = OpenAIEmbedder(_REDUCED, config=config)

        await embedder.aclose()

        assert embedder._client.is_closed

    def test_dispatcher_builds_the_openai_embedder(self) -> None:
        embedder = get_embedder(_REDUCED)

        assert isinstance(embedder, OpenAIEmbedder) and embedder.dimensions == 256

    # ---- Retry-After parsing ---- #

    def test_retry_after_headers(self) -> None:
        future = format_datetime(utc_now().replace(microsecond=0)).replace("+0000", "GMT")

        assert retry_after_seconds(httpx.Headers({"retry-after-ms": "250"})) == 0.25
        assert retry_after_seconds(httpx.Headers({"Retry-After": "7"})) == 7.0
        assert retry_after_seconds(httpx.Headers({"Retry-After": "-3"})) == 0.0
        assert retry_after_seconds(httpx.Headers({"Retry-After": future})) == pytest.approx(
            0.0, abs=2
        )
        assert retry_after_seconds(httpx.Headers({"Retry-After": "soon"})) is None
        assert retry_after_seconds(httpx.Headers({"retry-after-ms": "x", "Retry-After": "2"})) == 2
        assert retry_after_seconds(httpx.Headers({})) is None
