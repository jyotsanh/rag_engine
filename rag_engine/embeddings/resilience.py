"""Embedding resilience: retries with backoff, a per-provider gate, ordered concurrency (Step 63).

- **`EmbeddingRetryPolicy`**: only `retryable` errors (rate limits, 5xx, timeouts) are
  retried, at most `OPENAI_MAX_RETRIES` times. The delay before retry n is drawn from
  [d/2, d], d = min(max, base·2^(n-1)) ("equal jitter"), and a provider's `retry_after`
  is a floor on it. A provider asking to wait longer than `EMBEDDING_RETRY_AFTER_MAX_SECONDS`
  fails the request instead of stalling the job. Permanent errors (auth, bad request,
  billing quota, malformed response) are raised at once, never retried.
- **`ProviderGate`**: one per provider **and event loop** (Celery tasks may each run their
  own loop; an asyncio semaphore is bound to the loop it was first used in):
  - a semaphore of `OPENAI_MAX_CONCURRENT_REQUESTS` around each HTTP attempt (never held
    while sleeping), shared by every embedder of the provider in the process;
  - a shared pause: a 429 with a wait time pauses *all* requests of the provider until
    then, so concurrent requests don't keep hammering a rate limit.
- **`gather_in_order`**: runs sub-requests concurrently and returns their results in input
  order; the first failure cancels the rest and is raised as itself (no ExceptionGroup).
"""

import asyncio
import random
import time
import weakref
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Self

from loguru import logger

from rag_engine.config import Settings, settings
from rag_engine.embeddings.exceptions import EmbeddingError, EmbeddingRateLimitedError

type Sleep = Callable[[float], Awaitable[None]]
type Clock = Callable[[], float]


class EmbeddingRetryPolicy:
    def __init__(
        self,
        *,
        max_retries: int,
        base_delay: float,
        max_delay: float,
        max_retry_after: float,
        rng: random.Random | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if not 0 < base_delay <= max_delay:
            raise ValueError("need 0 < base_delay <= max_delay")
        if max_retry_after < 0:
            raise ValueError("max_retry_after must be >= 0")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retry_after = max_retry_after
        self._rng = rng or random.Random()

    @classmethod
    def from_settings(cls, config: Settings = settings, rng: random.Random | None = None) -> Self:
        return cls(
            max_retries=config.OPENAI_MAX_RETRIES,
            base_delay=config.EMBEDDING_RETRY_BASE_DELAY_SECONDS,
            max_delay=config.EMBEDDING_RETRY_MAX_DELAY_SECONDS,
            max_retry_after=config.EMBEDDING_RETRY_AFTER_MAX_SECONDS,
            rng=rng,
        )

    def backoff(self, retry: int) -> float:
        """Jittered delay before retry number `retry` (>= 1)."""
        ceiling = min(self.max_delay, self.base_delay * 2 ** (max(retry, 1) - 1))
        return ceiling / 2 + self._rng.uniform(0, ceiling / 2)

    def delay(self, error: EmbeddingError, retry: int) -> float | None:
        """Seconds to wait before retry `retry`, or None when the error must be raised."""
        if not error.retryable or retry > self.max_retries:
            return None
        if error.retry_after is not None and error.retry_after > self.max_retry_after:
            return None
        return max(self.backoff(retry), error.retry_after or 0.0)


@dataclass(slots=True)
class ProviderGate:
    """Concurrency + shared rate-limit pause for one provider in one event loop."""

    limit: int
    clock: Clock = time.monotonic
    semaphore: asyncio.Semaphore = field(init=False)
    paused_until: float = 0.0

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
        self.semaphore = asyncio.Semaphore(self.limit)

    def pause(self, seconds: float) -> None:
        """Hold every request of the provider for `seconds` (the longest pause wins)."""
        self.paused_until = max(self.paused_until, self.clock() + max(seconds, 0.0))

    def wait_seconds(self) -> float:
        return max(self.paused_until - self.clock(), 0.0)


_GATES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, int], ProviderGate]
] = weakref.WeakKeyDictionary()


def provider_gate(provider: str, limit: int) -> ProviderGate:
    """The running loop's gate for `(provider, limit)`, shared by all embedders using it."""
    loop = asyncio.get_running_loop()
    gates = _GATES.setdefault(loop, {})
    gate = gates.get((provider, limit))
    if gate is None:
        gate = gates[(provider, limit)] = ProviderGate(limit)
    return gate


async def call_with_retries[T](
    attempt: Callable[[], Awaitable[T]],
    *,
    policy: EmbeddingRetryPolicy,
    gate: ProviderGate,
    sleep: Sleep = asyncio.sleep,
    describe: str = "embedding request",
) -> T:
    """Run `attempt` through the gate, retrying retryable `EmbeddingError`s per `policy`.

    The raised error (permanent, or retryable but out of retries / waiting too long) gets
    `details["attempts"]`.
    """
    retry = 0
    while True:
        if (wait := gate.wait_seconds()) > 0:
            await sleep(wait)  # another request hit a rate limit: everyone waits
        try:
            async with gate.semaphore:
                return await attempt()
        except EmbeddingError as error:
            retry += 1
            delay = policy.delay(error, retry)
            if delay is None:
                error.details["attempts"] = retry
                raise
            if isinstance(error, EmbeddingRateLimitedError) and error.retry_after:
                gate.pause(error.retry_after)
            logger.warning(
                "{} failed ({}), retry {}/{} in {:.2f}s",
                describe,
                error.code,
                retry,
                policy.max_retries,
                delay,
            )
            await sleep(delay)


async def gather_in_order[T](calls: Sequence[Callable[[], Awaitable[T]]]) -> list[T]:
    """Run `calls` concurrently; results in input order; the first failure cancels the rest."""
    if len(calls) == 1:
        return [await calls[0]()]
    tasks = [asyncio.ensure_future(call()) for call in calls]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in tasks:  # the first failure in input order among those finished
            if task in done and (error := task.exception()) is not None:
                raise error
        return [task.result() for task in tasks]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
