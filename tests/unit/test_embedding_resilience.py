"""Embedding resilience: retry policy, provider gate (concurrency + shared pause), ordering."""

import asyncio
import json
import random
from collections.abc import Callable, Iterator

import httpx
import pytest
import respx

from rag_engine.config import Settings
from rag_engine.embeddings import (
    EmbeddingAuthError,
    EmbeddingError,
    EmbeddingRateLimitedError,
    EmbeddingRequestError,
    EmbeddingRetryPolicy,
    EmbeddingUnavailableError,
    OpenAIEmbedder,
    ProviderGate,
    call_with_retries,
    duration_seconds,
    gather_in_order,
    provider_gate,
    rate_limit_reset_seconds,
    resolve_space,
)
from tests.conftest import OverrideSettings

_URL = "https://api.openai.com/v1/embeddings"
_SPACE = resolve_space("text-embedding-3-small", 256)


class _Sleeps:
    """A recording `sleep` that doesn't wait."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        await asyncio.sleep(0)


def _ok(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": i, "embedding": [float(len(t))] + [0.5] * 255}
                for i, t in enumerate(texts)
            ],
            "usage": {"prompt_tokens": len(texts)},
        },
    )


def _error(status: int, headers: dict[str, str] | None = None, code: str | None = None):
    body = {"error": {"message": f"status {status}", **({"code": code} if code else {})}}
    return httpx.Response(status, json=body, headers=headers or {})


type _Reply = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


def _replies(*replies: _Reply) -> Callable[[httpx.Request], httpx.Response]:
    """A respx side effect answering successive calls with `replies` in order."""
    queue = list(replies)

    def reply(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, httpx.Response) else item(request)

    return reply


def _policy(max_retries: int = 4, max_retry_after: float = 300.0) -> EmbeddingRetryPolicy:
    return EmbeddingRetryPolicy(
        max_retries=max_retries,
        base_delay=1.0,
        max_delay=8.0,
        max_retry_after=max_retry_after,
        rng=random.Random(63),
    )


def _embedder(config: Settings, sleeps: _Sleeps, **policy: float) -> OpenAIEmbedder:
    return OpenAIEmbedder(_SPACE, config=config, policy=_policy(**policy), sleep=sleeps)  # ty: ignore[invalid-argument-type]


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def config(override_settings: OverrideSettings) -> Settings:
    return override_settings(OPENAI_MAX_CONCURRENT_REQUESTS=3)


# ---- #
# Tests
# ---- #


class TestOpenAIResilience_S63:
    """S63 — OpenAI resilience: retries with backoff + Retry-After, gate, permanent errors."""

    # ---- the policy ---- #

    def test_backoff_is_jittered_exponential_and_capped(self) -> None:
        policy = _policy()

        for retry, ceiling in [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0), (5, 8.0), (9, 8.0)]:
            delays = [policy.backoff(retry) for _ in range(200)]
            assert all(ceiling / 2 <= d <= ceiling for d in delays), retry
            assert max(delays) - min(delays) > ceiling / 4  # actually jittered

    def test_policy_decisions(self) -> None:
        policy = _policy(max_retries=2, max_retry_after=60)

        assert policy.delay(EmbeddingUnavailableError(), 1) is not None
        assert policy.delay(EmbeddingUnavailableError(), 3) is None  # out of retries
        assert policy.delay(EmbeddingAuthError(), 1) is None  # permanent
        assert policy.delay(EmbeddingRequestError(), 1) is None
        assert policy.delay(EmbeddingRateLimitedError(retry_after=30), 1) == 30  # a floor
        assert policy.delay(EmbeddingRateLimitedError(retry_after=61), 1) is None  # too long

    def test_policy_from_settings_and_validation(self, override_settings: OverrideSettings) -> None:
        config = override_settings(
            OPENAI_MAX_RETRIES=2,
            EMBEDDING_RETRY_BASE_DELAY_SECONDS=0.5,
            EMBEDDING_RETRY_MAX_DELAY_SECONDS=4,
            EMBEDDING_RETRY_AFTER_MAX_SECONDS=30,
        )

        policy = EmbeddingRetryPolicy.from_settings(config)

        assert (policy.max_retries, policy.base_delay, policy.max_delay) == (2, 0.5, 4)
        assert policy.max_retry_after == 30
        with pytest.raises(ValueError, match="base_delay"):
            EmbeddingRetryPolicy(max_retries=1, base_delay=5, max_delay=1, max_retry_after=1)
        with pytest.raises(ValueError):
            override_settings(EMBEDDING_RETRY_BASE_DELAY_SECONDS=90)  # > max 60

    # ---- retries over HTTP ---- #

    async def test_server_errors_are_retried_until_success(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        route = api.post(_URL).mock(side_effect=_replies(_error(503), _error(500), _ok))
        sleeps = _Sleeps()

        async with _embedder(config, sleeps) as embedder:
            batch = await embedder.embed_query("q")

        assert route.call_count == 3 and len(batch.vector) == 256
        assert len(sleeps.delays) == 2
        assert 0.5 <= sleeps.delays[0] <= 1.0 and 1.0 <= sleeps.delays[1] <= 2.0

    async def test_timeouts_are_retried(self, api: respx.MockRouter, config: Settings) -> None:
        route = api.post(_URL).mock(side_effect=_replies(httpx.ReadTimeout("slow"), _ok))

        async with _embedder(config, _Sleeps()) as embedder:
            await embedder.embed_query("q")

        assert route.call_count == 2

    async def test_a_rate_limit_waits_at_least_what_openai_asks(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        api.post(_URL).mock(side_effect=_replies(_error(429, {"retry-after-ms": "2500"}), _ok))
        sleeps = _Sleeps()

        async with _embedder(config, sleeps) as embedder:
            await embedder.embed_query("q")

        # the retry's own wait: the floor (backoff ≤ 1 s); with fake time the shared pause
        # is then still "active" and shows up as a second wait, which real time never has
        assert sleeps.delays[0] == pytest.approx(2.5)

    async def test_reset_headers_set_the_wait_without_retry_after(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        headers = {"x-ratelimit-reset-requests": "20ms", "x-ratelimit-reset-tokens": "1m30s"}
        api.post(_URL).mock(side_effect=_replies(_error(429, headers), _ok))
        sleeps = _Sleeps()

        async with _embedder(config, sleeps) as embedder:
            await embedder.embed_query("q")

        assert 90.0 in [round(d, 3) for d in sleeps.delays]

    async def test_a_wait_longer_than_allowed_fails_at_once(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        route = api.post(_URL).mock(side_effect=_replies(_error(429, {"Retry-After": "600"}), _ok))
        sleeps = _Sleeps()

        async with _embedder(config, sleeps) as embedder:
            with pytest.raises(EmbeddingRateLimitedError) as exc:
                await embedder.embed_query("q")

        assert route.call_count == 1 and sleeps.delays == []
        assert (exc.value.retry_after, exc.value.details["attempts"]) == (600.0, 1)

    async def test_retries_run_out_and_the_last_error_surfaces(
        self, api: respx.MockRouter, config: Settings
    ) -> None:
        route = api.post(_URL).mock(return_value=_error(500))
        sleeps = _Sleeps()

        async with _embedder(config, sleeps, max_retries=3) as embedder:
            with pytest.raises(EmbeddingUnavailableError) as exc:
                await embedder.embed_query("q")

        assert route.call_count == 4  # 1 + 3 retries
        assert len(sleeps.delays) == 3 and exc.value.details["attempts"] == 4

    @pytest.mark.parametrize(
        ("response", "error"),
        [
            (lambda: _error(401, code="invalid_api_key"), EmbeddingAuthError),
            (lambda: _error(400), EmbeddingRequestError),
            (lambda: _error(429, code="insufficient_quota"), EmbeddingRequestError),
        ],
    )
    async def test_permanent_errors_are_never_retried(
        self, api: respx.MockRouter, config: Settings, response, error: type[EmbeddingError]
    ) -> None:
        route = api.post(_URL).mock(return_value=response())
        sleeps = _Sleeps()

        async with _embedder(config, sleeps) as embedder:
            with pytest.raises(error) as exc:
                await embedder.embed_query("q")

        assert route.call_count == 1 and sleeps.delays == []
        assert exc.value.details["attempts"] == 1

    # ---- concurrency ---- #

    async def test_sub_batches_run_concurrently_within_the_limit_and_stay_ordered(
        self, api: respx.MockRouter, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(EMBEDDING_MAX_BATCH_INPUTS=1, OPENAI_MAX_CONCURRENT_REQUESTS=3)
        in_flight = peak = 0

        async def slow(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            text = json.loads(request.content)["input"][0]
            await asyncio.sleep(0.05 if text.startswith("x" * 3) else 0.01)  # early ones slower
            in_flight -= 1
            return _ok(request)

        route = api.post(_URL).mock(side_effect=slow)
        texts = ["x" * n for n in range(1, 11)]

        async with _embedder(config, _Sleeps()) as embedder:
            batch = await embedder.embed_documents(texts)

        assert route.call_count == 10
        assert peak == 3  # concurrent, never more than the limit
        assert [v[0] for v in batch.vectors] == [float(len(t)) for t in texts]  # input order
        assert batch.usage.requests == 10

    async def test_the_limit_is_shared_by_every_embedder_of_the_provider(
        self, api: respx.MockRouter, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(EMBEDDING_MAX_BATCH_INPUTS=1, OPENAI_MAX_CONCURRENT_REQUESTS=2)
        in_flight = peak = 0

        async def slow(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return _ok(request)

        api.post(_URL).mock(side_effect=slow)
        first, second = _embedder(config, _Sleeps()), _embedder(config, _Sleeps())

        await asyncio.gather(
            first.embed_documents([f"a{i}" for i in range(4)]),
            second.embed_documents([f"b{i}" for i in range(4)]),
        )

        assert peak == 2
        await first.aclose()
        await second.aclose()

    async def test_a_permanent_failure_cancels_the_other_sub_batches(
        self, api: respx.MockRouter, override_settings: OverrideSettings
    ) -> None:
        config = override_settings(EMBEDDING_MAX_BATCH_INPUTS=1, OPENAI_MAX_CONCURRENT_REQUESTS=4)
        finished: list[str] = []

        async def respond(request: httpx.Request) -> httpx.Response:
            text = json.loads(request.content)["input"][0]
            if text == "bad":
                return _error(400)
            await asyncio.sleep(0.5)
            finished.append(text)
            return _ok(request)

        api.post(_URL).mock(side_effect=respond)

        async with _embedder(config, _Sleeps()) as embedder:
            with pytest.raises(EmbeddingRequestError):  # the error itself, no ExceptionGroup
                await embedder.embed_documents(["ok1", "bad", "ok2", "ok3"])

        assert finished == []  # the slow siblings were cancelled, not awaited

    # ---- the gate ---- #

    async def test_a_rate_limit_pauses_every_request_of_the_provider(self) -> None:
        gate = ProviderGate(2)
        sleeps = _Sleeps()
        calls = 0

        async def limited() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise EmbeddingRateLimitedError(retry_after=5.0)
            return "ok"

        assert await call_with_retries(limited, policy=_policy(), gate=gate, sleep=sleeps) == "ok"
        assert gate.wait_seconds() == pytest.approx(5.0, abs=0.5)  # the pause is shared

        async def other() -> str:
            return "other"

        await call_with_retries(other, policy=_policy(), gate=gate, sleep=sleeps)
        assert sleeps.delays[-1] == pytest.approx(5.0, abs=0.5)  # another caller waited too

    async def test_the_longest_pause_wins(self) -> None:
        gate = ProviderGate(1)

        gate.pause(10)
        gate.pause(2)

        assert gate.wait_seconds() == pytest.approx(10, abs=0.5)
        with pytest.raises(ValueError):
            ProviderGate(0)

    async def test_gates_are_per_provider_limit_and_event_loop(self) -> None:
        here = provider_gate("openai", 4)

        assert provider_gate("openai", 4) is here
        assert provider_gate("openai", 2) is not here
        assert provider_gate("other", 4) is not here

        async def in_another_loop() -> ProviderGate:
            return provider_gate("openai", 4)

        elsewhere = await asyncio.to_thread(asyncio.run, in_another_loop())
        assert elsewhere is not here  # never share a semaphore across loops

    # ---- helpers ---- #

    async def test_gather_in_order(self) -> None:
        async def value(n: int, delay: float) -> int:
            await asyncio.sleep(delay)
            return n

        results = await gather_in_order([lambda n=n: value(n, 0.03 - n * 0.01) for n in range(3)])

        assert results == [0, 1, 2]
        assert await gather_in_order([lambda: value(7, 0)]) == [7]

    def test_reset_durations(self) -> None:
        assert duration_seconds("20ms") == pytest.approx(0.02)
        assert duration_seconds("1s") == 1.0
        assert duration_seconds("6m0s") == 360.0
        assert duration_seconds("1h2m3.5s") == pytest.approx(3723.5)
        assert duration_seconds("soon") is None
        assert duration_seconds("1s junk") is None
        assert rate_limit_reset_seconds(httpx.Headers({})) is None
        assert (
            rate_limit_reset_seconds(
                httpx.Headers({"x-ratelimit-reset-requests": "2s", "x-ratelimit-reset-tokens": "x"})
            )
            == 2.0
        )
