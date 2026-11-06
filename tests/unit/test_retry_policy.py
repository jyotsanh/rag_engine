"""Retry & failure policy: classification, backoff with jitter, attempt limit, Retry-After."""

import itertools
import random
import socket
import ssl
from collections.abc import AsyncIterator

import pytest

from rag_engine.config import Settings
from rag_engine.crawling import (
    FailureKind,
    RetriesExhaustedError,
    RetryPolicy,
    call_with_retries,
    classify,
)
from rag_engine.crawling.fetchers import (
    FetchError,
    FetchResult,
    FetchTimeoutError,
    HttpFetcher,
    HttpStatusError,
    NetworkFetchError,
    RedirectLoopFetchError,
    SsrfFetchError,
    TooLargeError,
    TooManyRedirectsFetchError,
    UnsupportedContentTypeError,
)
from rag_engine.domain import PageReason, PageStatus
from rag_engine.security import HostResolutionError, SafeHttpClient, SsrfGuard
from tests.conftest import OverrideSettings
from tests.support.fixture_site import FLAKY_FAILURES, FixtureSite

_URL = "https://acme.example/page"


def _policy(
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    max_retry_after: float = 120.0,
    rng: random.Random | None = None,
) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        max_retry_after=max_retry_after,
        rng=rng or random.Random(7),  # seeded: jitter is reproducible
    )


def _status(code: int, retry_after: float | None = None) -> HttpStatusError:
    return HttpStatusError(url=_URL, status_code=code, retry_after=retry_after)


def _chained(*chain: BaseException) -> NetworkFetchError:
    """A `NetworkFetchError` raised from `chain[0]`, itself raised from `chain[1]`, …"""
    for outer, inner in itertools.pairwise(chain):
        outer.__cause__ = inner
    error = NetworkFetchError("connect failed", url=_URL, detail="ConnectError")
    error.__cause__ = chain[0]
    return error


class _RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture
def config(override_settings: OverrideSettings) -> Settings:
    return override_settings(
        SSRF_ALLOW_PRIVATE_NETWORKS=True,
        CRAWL_READ_TIMEOUT_SECONDS=2,
        CRAWL_FETCH_TIMEOUT_SECONDS=3,
    )


@pytest.fixture
async def fetcher(config: Settings) -> AsyncIterator[HttpFetcher]:
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


# ---- #
# Tests
# ---- #


class TestRetryPolicy_S28:
    """S28 — Retry & failure policy: error classifier, backoff + jitter, per-URL attempt limit."""

    # ---- classification ---- #

    @pytest.mark.parametrize(
        ("error", "kind", "reason"),
        [
            (_status(500), FailureKind.RETRYABLE, PageReason.HTTP_SERVER_ERROR),
            (_status(502), FailureKind.RETRYABLE, PageReason.HTTP_SERVER_ERROR),
            (_status(503), FailureKind.RETRYABLE, PageReason.HTTP_SERVER_ERROR),
            (_status(504), FailureKind.RETRYABLE, PageReason.HTTP_SERVER_ERROR),
            (_status(522), FailureKind.RETRYABLE, PageReason.HTTP_SERVER_ERROR),
            (_status(501), FailureKind.PERMANENT, PageReason.HTTP_SERVER_ERROR),
            (_status(505), FailureKind.PERMANENT, PageReason.HTTP_SERVER_ERROR),
            (_status(429), FailureKind.RETRYABLE, PageReason.RATE_LIMITED),
            (_status(408), FailureKind.RETRYABLE, PageReason.HTTP_CLIENT_ERROR),
            (_status(425), FailureKind.RETRYABLE, PageReason.HTTP_CLIENT_ERROR),
            (_status(400), FailureKind.PERMANENT, PageReason.HTTP_CLIENT_ERROR),
            (_status(401), FailureKind.PERMANENT, PageReason.HTTP_CLIENT_ERROR),
            (_status(403), FailureKind.PERMANENT, PageReason.HTTP_CLIENT_ERROR),
            (_status(404), FailureKind.PERMANENT, PageReason.HTTP_CLIENT_ERROR),
            (_status(410), FailureKind.PERMANENT, PageReason.HTTP_CLIENT_ERROR),
            (FetchTimeoutError("t", url=_URL), FailureKind.RETRYABLE, PageReason.TIMEOUT),
            (NetworkFetchError("n", url=_URL), FailureKind.RETRYABLE, PageReason.NETWORK_ERROR),
            (SsrfFetchError("s", url=_URL), FailureKind.PERMANENT, PageReason.SSRF_BLOCKED),
            (TooLargeError("l", url=_URL), FailureKind.PERMANENT, PageReason.TOO_LARGE),
            (
                RedirectLoopFetchError("r", url=_URL),
                FailureKind.PERMANENT,
                PageReason.REDIRECT_LOOP,
            ),
            (
                TooManyRedirectsFetchError("r", url=_URL),
                FailureKind.PERMANENT,
                PageReason.TOO_MANY_REDIRECTS,
            ),
            (
                UnsupportedContentTypeError("u", url=_URL, content_type="application/json"),
                FailureKind.SKIP,
                PageReason.UNSUPPORTED_CONTENT_TYPE,
            ),
            (FetchError("unknown", url=_URL), FailureKind.PERMANENT, PageReason.NETWORK_ERROR),
        ],
        ids=lambda v: getattr(v, "detail", None) or str(v),
    )
    def test_classify_matrix(
        self, error: FetchError, kind: FailureKind, reason: PageReason
    ) -> None:
        result = classify(error)

        assert (result.kind, result.reason) == (kind, reason)

    def test_retry_after_is_kept_only_for_retryable_statuses(self) -> None:
        assert classify(_status(429, retry_after=5)).retry_after == 5
        assert classify(_status(503, retry_after=7)).retry_after == 7
        assert classify(_status(404, retry_after=5)).retry_after is None

    def test_bad_tls_certificate_is_permanent(self) -> None:
        # same shape as the real chain: httpx.NetworkError <- httpcore.ConnectError <- ssl error
        error = _chained(OSError("wrapped"), ssl.SSLCertVerificationError("unknown issuer"))

        result = classify(error)

        assert result.kind is FailureKind.PERMANENT
        assert result.detail == "tls_certificate"
        assert result.reason is PageReason.NETWORK_ERROR

    def test_nonexistent_host_is_permanent(self) -> None:
        error = _chained(HostResolutionError("failed"), socket.gaierror(socket.EAI_NONAME, "x"))

        result = classify(error)

        assert result.kind is FailureKind.PERMANENT
        assert result.detail == "dns_nxdomain"

    @pytest.mark.parametrize(
        "cause",
        [
            socket.gaierror(socket.EAI_AGAIN, "temporary failure in name resolution"),
            TimeoutError(),
            ConnectionResetError(),
        ],
        ids=["eai_again", "dns_timeout", "reset"],
    )
    def test_transient_network_errors_are_retryable(self, cause: BaseException) -> None:
        result = classify(_chained(HostResolutionError("failed"), cause))

        assert result.kind is FailureKind.RETRYABLE

    def test_cause_cycle_does_not_hang(self) -> None:
        a, b = OSError("a"), OSError("b")
        a.__cause__, b.__cause__ = b, a

        assert classify(_chained(a)).kind is FailureKind.RETRYABLE

    # ---- backoff ---- #

    def test_backoff_is_exponential_with_equal_jitter(self) -> None:
        policy = _policy(base_delay=1.0, max_delay=1000.0, rng=random.Random(1))

        for attempts in range(1, 8):
            ceiling = 2 ** (attempts - 1)
            delays = [policy.backoff(attempts) for _ in range(200)]

            assert all(ceiling / 2 <= d <= ceiling for d in delays)
            assert len(set(delays)) > 1  # jittered, not a fixed schedule

    def test_backoff_is_capped(self) -> None:
        policy = _policy(base_delay=1.0, max_delay=5.0)

        delays = [policy.backoff(20) for _ in range(100)]

        assert all(2.5 <= d <= 5.0 for d in delays)

    def test_backoff_is_deterministic_with_a_seeded_rng(self) -> None:
        a, b = _policy(rng=random.Random(42)), _policy(rng=random.Random(42))

        assert [a.backoff(n) for n in range(1, 5)] == [b.backoff(n) for n in range(1, 5)]

    # ---- decisions ---- #

    def test_retryable_error_under_the_limit_is_retried(self) -> None:
        decision = _policy().decide(_status(503), attempts=1)

        assert decision.retry is True
        assert 0.5 <= decision.delay_seconds <= 1.0
        assert decision.final_status is None
        assert decision.reason is PageReason.HTTP_SERVER_ERROR

    def test_attempt_limit_fails_the_page(self) -> None:
        policy = _policy(max_attempts=4)

        retries = [policy.decide(_status(503), attempts=n).retry for n in range(1, 5)]
        final = policy.decide(_status(503), attempts=4)

        assert retries == [True, True, True, False]
        assert final.final_status is PageStatus.FAILED
        assert final.reason is PageReason.HTTP_SERVER_ERROR
        assert final.delay_seconds == 0
        assert "gave up after 4 attempts" in final.detail

    def test_single_attempt_policy_never_retries(self) -> None:
        decision = _policy(max_attempts=1).decide(FetchTimeoutError("t", url=_URL), attempts=1)

        assert decision.retry is False
        assert decision.final_status is PageStatus.FAILED
        assert decision.reason is PageReason.TIMEOUT

    def test_permanent_error_fails_on_first_attempt(self) -> None:
        decision = _policy().decide(_status(404), attempts=1)

        assert decision.retry is False
        assert decision.final_status is PageStatus.FAILED
        assert decision.reason is PageReason.HTTP_CLIENT_ERROR
        assert decision.detail == "http_404"

    def test_unsupported_content_is_skipped_not_failed(self) -> None:
        error = UnsupportedContentTypeError("u", url=_URL, content_type="application/json")

        decision = _policy().decide(error, attempts=1)

        assert decision.retry is False
        assert decision.final_status is PageStatus.SKIPPED
        assert decision.reason is PageReason.UNSUPPORTED_CONTENT_TYPE

    def test_retry_after_is_a_floor_on_the_delay(self) -> None:
        decision = _policy().decide(_status(429, retry_after=10), attempts=1)

        assert decision.retry is True
        assert decision.delay_seconds == 10

    def test_backoff_wins_over_a_shorter_retry_after(self) -> None:
        decision = _policy(base_delay=8.0).decide(_status(503, retry_after=1), attempts=1)

        assert 4.0 <= decision.delay_seconds <= 8.0

    def test_retry_after_beyond_the_cap_gives_up(self) -> None:
        decision = _policy(max_retry_after=60).decide(_status(429, retry_after=3600), attempts=1)

        assert decision.retry is False
        assert decision.final_status is PageStatus.FAILED
        assert decision.reason is PageReason.RATE_LIMITED
        assert "Retry-After 3600s exceeds 60s" in decision.detail

    def test_attempts_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="attempts"):
            _policy().decide(_status(503), attempts=0)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"max_attempts": 0},
            {"base_delay": 0},
            {"base_delay": 10, "max_delay": 5},
            {"max_retry_after": -1},
        ],
    )
    def test_invalid_policy_rejected(self, overrides: dict) -> None:
        with pytest.raises(ValueError):
            _policy(**overrides)

    def test_policy_from_settings(self, override_settings: OverrideSettings) -> None:
        config = override_settings(
            CRAWL_MAX_RETRIES=2,
            CRAWL_RETRY_BASE_DELAY_SECONDS=0.5,
            CRAWL_RETRY_MAX_DELAY_SECONDS=8,
            CRAWL_RETRY_AFTER_MAX_SECONDS=30,
        )

        policy = RetryPolicy.from_settings(config)

        assert policy.max_attempts == 3  # first attempt + 2 retries
        assert (policy.base_delay, policy.max_delay, policy.max_retry_after) == (0.5, 8, 30)

    # ---- call_with_retries against the fixture site ---- #

    async def test_flaky_page_succeeds_after_retries(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        # Arrange: /flaky answers 503 + Retry-After: 1 twice, then 200
        sleep = _RecordingSleep()
        url = fixture_site.url("/flaky")

        # Act
        result: FetchResult = await call_with_retries(
            lambda: fetcher.fetch(url), _policy(base_delay=0.1), sleep=sleep
        )

        # Assert
        assert result.status_code == 200
        assert fixture_site.hits("/flaky") == FLAKY_FAILURES + 1
        assert len(sleep.delays) == FLAKY_FAILURES
        assert all(d >= 1.0 for d in sleep.delays)  # Retry-After honoured

    async def test_server_error_exhausts_the_attempt_limit(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        sleep = _RecordingSleep()

        with pytest.raises(RetriesExhaustedError) as exc:
            await call_with_retries(
                lambda: fetcher.fetch(fixture_site.url("/error")), _policy(), sleep=sleep
            )

        assert fixture_site.hits("/error") == 4
        assert len(sleep.delays) == 3
        assert sleep.delays == sorted(sleep.delays)  # 0.5-1, 1-2, 2-4: never overlap
        assert exc.value.decision.final_status is PageStatus.FAILED
        assert exc.value.decision.reason is PageReason.HTTP_SERVER_ERROR
        assert exc.value.decision.attempts == 4
        assert isinstance(exc.value.error, HttpStatusError)

    @pytest.mark.parametrize(
        ("path", "status", "reason"),
        [
            ("/missing.html", PageStatus.FAILED, PageReason.HTTP_CLIENT_ERROR),
            ("/redirect-metadata", PageStatus.FAILED, PageReason.SSRF_BLOCKED),
            ("/api/widgets", PageStatus.SKIPPED, PageReason.UNSUPPORTED_CONTENT_TYPE),
        ],
    )
    async def test_non_retryable_outcomes_make_one_request(
        self,
        fetcher: HttpFetcher,
        fixture_site: FixtureSite,
        path: str,
        status: PageStatus,
        reason: PageReason,
    ) -> None:
        sleep = _RecordingSleep()

        with pytest.raises(RetriesExhaustedError) as exc:
            await call_with_retries(
                lambda: fetcher.fetch(fixture_site.url(path)), _policy(), sleep=sleep
            )

        assert fixture_site.hits(path) == 1
        assert sleep.delays == []
        assert (exc.value.decision.final_status, exc.value.decision.reason) == (status, reason)
        assert exc.value.decision.attempts == 1

    async def test_rate_limited_page_honours_retry_after(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        # /rate-limited always answers 429 with Retry-After: 2
        sleep = _RecordingSleep()

        with pytest.raises(RetriesExhaustedError) as exc:
            await call_with_retries(
                lambda: fetcher.fetch(fixture_site.url("/rate-limited")), _policy(), sleep=sleep
            )

        assert fixture_site.hits("/rate-limited") == 4
        assert all(d >= 2.0 for d in sleep.delays)
        assert exc.value.decision.reason is PageReason.RATE_LIMITED

    async def test_rate_limited_page_with_a_too_long_retry_after_fails_at_once(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        sleep = _RecordingSleep()

        with pytest.raises(RetriesExhaustedError) as exc:
            await call_with_retries(
                lambda: fetcher.fetch(fixture_site.url("/rate-limited")),
                _policy(max_retry_after=1),
                sleep=sleep,
            )

        assert fixture_site.hits("/rate-limited") == 1
        assert sleep.delays == []
        assert "Retry-After" in exc.value.decision.detail

    async def test_connection_refused_is_retried(self, fetcher: HttpFetcher) -> None:
        sleep = _RecordingSleep()

        with pytest.raises(RetriesExhaustedError) as exc:
            await call_with_retries(
                lambda: fetcher.fetch("http://127.0.0.1:1/"), _policy(max_attempts=2), sleep=sleep
            )

        assert len(sleep.delays) == 1
        assert exc.value.decision.reason is PageReason.NETWORK_ERROR

    @pytest.mark.parametrize(
        ("code", "attempts"),
        [(socket.EAI_NONAME, 1), (socket.EAI_AGAIN, 4)],
        ids=["nxdomain", "eai_again"],
    )
    async def test_dns_failures_through_the_real_fetcher(
        self, config: Settings, code: int, attempts: int
    ) -> None:
        # stub resolver: deterministic regardless of the machine's DNS setup
        async def resolver(host: str, port: int) -> list[str]:
            raise socket.gaierror(code, "lookup failed")

        sleep = _RecordingSleep()
        guard = SsrfGuard(config, resolver=resolver)

        async with SafeHttpClient(config, guard=guard) as client:
            fetcher = HttpFetcher(client, config)
            with pytest.raises(RetriesExhaustedError) as exc:
                await call_with_retries(
                    lambda: fetcher.fetch("http://ghost.example/"), _policy(), sleep=sleep
                )

        assert exc.value.decision.attempts == attempts
        assert len(sleep.delays) == attempts - 1
        assert exc.value.decision.reason is PageReason.NETWORK_ERROR

    async def test_non_fetch_errors_propagate_without_retry(self) -> None:
        calls = 0

        async def broken() -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("bug")

        with pytest.raises(RuntimeError, match="bug"):
            await call_with_retries(broken, _policy(), sleep=_RecordingSleep())

        assert calls == 1

    async def test_real_sleep_is_used_by_default(self) -> None:
        outcomes: list[Exception | str] = [FetchTimeoutError("t", url=_URL), "ok"]

        async def operation() -> str:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        result = await call_with_retries(operation, _policy(base_delay=0.01, max_delay=0.01))

        assert result == "ok"
        assert outcomes == []
