"""Retry & failure policy for fetches.

`classify()` turns a typed `FetchError` into retryable / permanent / skip.
`RetryPolicy.decide()` adds the per-URL attempt limit and the backoff delay.
`call_with_retries()` runs an operation under the policy for callers that retry inline
(robots.txt, sitemaps); the crawl engine can call `decide()` and re-queue instead.
"""

import asyncio
import random
import socket
import ssl
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from loguru import logger

from rag_engine.config import Settings, settings
from rag_engine.crawling.fetchers.errors import (
    FetchError,
    FetchTimeoutError,
    HttpStatusError,
    NetworkFetchError,
    UnsupportedContentTypeError,
)
from rag_engine.domain import PageReason, PageStatus

# 4xx that mean "try again later" rather than "this request is wrong"
RETRYABLE_CLIENT_STATUSES = frozenset({408, 425, 429})
# 5xx that describe the request or server capability, so a retry gets the same answer
PERMANENT_SERVER_STATUSES = frozenset({501, 505})
# getaddrinfo codes meaning "this name does not exist" (EAI_AGAIN and friends stay retryable)
_NXDOMAIN_CODES = frozenset(
    code for code in (socket.EAI_NONAME, getattr(socket, "EAI_NODATA", None)) if code is not None
)


class FailureKind(StrEnum):
    RETRYABLE = "retryable"  # transient: try again after a backoff
    PERMANENT = "permanent"  # retrying would give the same answer: the page fails
    SKIP = "skip"  # not an error of the site: the page is skipped


@dataclass(frozen=True, slots=True)
class Classification:
    kind: FailureKind
    reason: PageReason
    detail: str
    retry_after: float | None = None  # server-requested wait (Retry-After), seconds


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """What to do after attempt number `attempts` failed."""

    retry: bool
    attempts: int  # attempts made so far, including the one that just failed
    reason: PageReason
    detail: str
    delay_seconds: float = 0.0  # wait before the next attempt; 0 when not retrying
    final_status: PageStatus | None = None  # FAILED or SKIPPED when not retrying


class RetriesExhaustedError(Exception):
    """`call_with_retries` gave up. Carries the final decision and the last fetch error."""

    def __init__(self, decision: RetryDecision, error: FetchError) -> None:
        super().__init__(f"{error.url}: {decision.reason} after {decision.attempts} attempt(s)")
        self.decision = decision
        self.error = error


def _causes(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _permanent_network_detail(error: NetworkFetchError) -> str | None:
    """A network error that will not heal on retry: bad certificate or non-existent host."""
    for cause in _causes(error):
        if isinstance(cause, ssl.SSLCertVerificationError):
            return "tls_certificate"
        if isinstance(cause, socket.gaierror) and cause.errno in _NXDOMAIN_CODES:
            return "dns_nxdomain"
    return None


def classify(error: FetchError) -> Classification:
    reason, detail = error.reason, error.detail
    match error:
        case UnsupportedContentTypeError():
            return Classification(FailureKind.SKIP, reason, detail)
        case HttpStatusError(status_code=status):
            retryable = status in RETRYABLE_CLIENT_STATUSES or (
                status >= 500 and status not in PERMANENT_SERVER_STATUSES
            )
            kind = FailureKind.RETRYABLE if retryable else FailureKind.PERMANENT
            return Classification(kind, reason, detail, error.retry_after if retryable else None)
        case FetchTimeoutError():
            return Classification(FailureKind.RETRYABLE, reason, detail)
        case NetworkFetchError():
            if permanent := _permanent_network_detail(error):
                return Classification(FailureKind.PERMANENT, reason, permanent)
            return Classification(FailureKind.RETRYABLE, reason, detail)
        case _:
            # SSRF-blocked, too large, redirect loop / too many redirects, and anything new:
            # fail closed rather than hammer a site with requests that cannot succeed
            return Classification(FailureKind.PERMANENT, reason, detail)


class RetryPolicy:
    """Per-URL attempt limit plus exponential backoff with jitter, honouring `Retry-After`.

    The delay before attempt n+1 is drawn from [d/2, d] with d = min(max, base · 2^(n-1))
    ("equal jitter"): spread out, but never close to zero, so a struggling site gets a real
    pause. A server's `Retry-After` is a floor on that delay. When it asks for longer than
    `max_retry_after` we give up instead of either waiting that long or retrying early.
    """

    def __init__(
        self,
        *,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
        max_retry_after: float,
        rng: random.Random | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if not 0 < base_delay <= max_delay:
            raise ValueError("need 0 < base_delay <= max_delay")
        if max_retry_after < 0:
            raise ValueError("max_retry_after must be >= 0")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retry_after = max_retry_after
        self._rng = rng or random.Random()

    @classmethod
    def from_settings(cls, config: Settings = settings, rng: random.Random | None = None) -> Self:
        return cls(
            max_attempts=config.CRAWL_MAX_RETRIES + 1,
            base_delay=config.CRAWL_RETRY_BASE_DELAY_SECONDS,
            max_delay=config.CRAWL_RETRY_MAX_DELAY_SECONDS,
            max_retry_after=config.CRAWL_RETRY_AFTER_MAX_SECONDS,
            rng=rng,
        )

    def backoff(self, attempts: int) -> float:
        """Jittered delay after `attempts` failed attempts (>= 1)."""
        ceiling = min(self.max_delay, self.base_delay * 2 ** (max(attempts, 1) - 1))
        return ceiling / 2 + self._rng.uniform(0, ceiling / 2)

    def decide(self, error: FetchError, attempts: int) -> RetryDecision:
        if attempts < 1:
            raise ValueError("attempts counts the failed attempt, so it is >= 1")
        c = classify(error)

        if c.kind is FailureKind.SKIP:
            return RetryDecision(
                False, attempts, c.reason, c.detail, final_status=PageStatus.SKIPPED
            )
        if c.kind is FailureKind.PERMANENT:
            return RetryDecision(
                False, attempts, c.reason, c.detail, final_status=PageStatus.FAILED
            )

        if attempts >= self.max_attempts:
            detail = f"{c.detail} (gave up after {attempts} attempts)"
            return RetryDecision(False, attempts, c.reason, detail, final_status=PageStatus.FAILED)
        if c.retry_after is not None and c.retry_after > self.max_retry_after:
            detail = (
                f"{c.detail} (Retry-After {c.retry_after:.0f}s exceeds {self.max_retry_after:.0f}s)"
            )
            return RetryDecision(False, attempts, c.reason, detail, final_status=PageStatus.FAILED)

        delay = max(self.backoff(attempts), c.retry_after or 0.0)
        return RetryDecision(True, attempts, c.reason, c.detail, delay_seconds=delay)


async def call_with_retries[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> T:
    """Run `operation` until it succeeds or the policy gives up (→ `RetriesExhaustedError`).

    Only `FetchError`s are retried; anything else (a bug, cancellation) propagates at once.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            return await operation()
        except FetchError as error:
            decision = policy.decide(error, attempts)
            if not decision.retry:
                raise RetriesExhaustedError(decision, error) from error
            logger.info(
                "retrying {} in {:.2f}s after {} (attempt {}/{})",
                error.url,
                decision.delay_seconds,
                decision.reason,
                attempts,
                policy.max_attempts,
            )
            await sleep(decision.delay_seconds)
