"""Delivering a finished job's completion webhook (Step 83).

`WebhookDeliverer.attempt(job_id)` makes **one** delivery attempt; the `webhooks` Celery
task schedules the next one when it says so. Each attempt:

1. Loads the job: only one whose webhook is still open (`pending`/`retrying`) is sent, so a
   duplicated task (late acks) or a sweep racing a retry is a no-op.
2. Signs the body **afresh** with every active secret of the tenant (Step 82): a retry
   carries a new timestamp, so it isn't rejected as a replay. No secret → `failed`
   (`webhook_secret_missing`); there is nothing a receiver could verify.
3. POSTs through `SafeHttpClient.send_once`: the receiver's host is vetted and pinned
   (SSRF, DNS rebinding) and **redirects are not followed** (a redirect could point the
   signed payload anywhere).
4. Classifies the answer:
   - 2xx → `delivered`;
   - 408, 425, 429, 5xx, a timeout, a network or DNS error → retry, after
     `base x 2^(attempt-1)` capped at `WEBHOOK_RETRY_MAX_SECONDS`, with equal jitter, and
     never sooner than the receiver's `Retry-After` (also capped);
   - any other answer (3xx, 4xx) or an SSRF-blocked target → `failed` at once: retrying
     wouldn't change it;
   - out of attempts (`WEBHOOK_MAX_ATTEMPTS`) → `failed`.
5. Records the attempt in the job's delivery log (the latest `MAX_WEBHOOK_LOG`), guarded on
   the attempt number. The receiver's response body is never stored.

Payload (`application/json`, compact, sorted keys)::

    {"id": "evt_…", "type": "job.succeeded", "created_at": "…",
     "data": {"job": {"id", "kb_id", "source_id", "seed_url", "status", "error",
                      "counters", "usage", "attempts", "started_at", "finished_at",
                      "report": {"pages", "stop_reason", "stop_detail"}}}}

Headers: `X-RagEngine-Signature`, `X-RagEngine-Event` (the type), `X-RagEngine-Delivery`
(the event id, the same on every attempt, for receivers to de-duplicate) and
`X-RagEngine-Attempt`.
"""

import json
import random
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

import httpx
from loguru import logger

from rag_engine.config import Settings, settings
from rag_engine.domain import CrawlJob, WebhookAttempt, WebhookStatus, utc_now
from rag_engine.infrastructure import MongoDatabase
from rag_engine.repositories import JobRepository, TenantRepository
from rag_engine.security import HostResolutionError, SafeHttpClient, SecretBox, SsrfBlockedError
from rag_engine.webhooks.secrets import WebhookSecretService
from rag_engine.webhooks.signer import signature_headers

EVENT_HEADER = "X-RagEngine-Event"
DELIVERY_HEADER = "X-RagEngine-Delivery"
ATTEMPT_HEADER = "X-RagEngine-Attempt"
_RETRYABLE_STATUSES = frozenset({408, 425, 429})
_MAX_DRAIN_BYTES = 64 * 1024  # read (and drop) at most this much of a receiver's answer


class DeliveryOutcome(StrEnum):
    DELIVERED = "delivered"
    RETRY = "retry"  # failed transiently; the task must try again after `retry_in`
    FAILED = "failed"  # given up
    SKIPPED = "skipped"  # nothing to send (no open webhook, or another delivery won)


@dataclass(frozen=True, slots=True)
class AttemptResult:
    outcome: DeliveryOutcome
    attempt: int = 0
    status_code: int | None = None
    error: str | None = None
    retry_in: float | None = None


@dataclass(frozen=True, slots=True)
class _Answer:
    status_code: int | None
    error: str | None  # None: a 2xx answer
    detail: str | None
    retryable: bool
    retry_after: float | None = None


def event_id(job: CrawlJob) -> str:
    """The job's completion event id: stable across attempts (receivers de-duplicate)."""
    return "evt_" + job.id.removeprefix(f"{CrawlJob.id_prefix}_")


def build_payload(job: CrawlJob) -> dict[str, Any]:
    report = job.report
    return {
        "id": event_id(job),
        "type": f"job.{job.status.value}",
        "created_at": (job.finished_at or utc_now()).isoformat(),
        "data": {
            "job": {
                "id": job.id,
                "kb_id": job.kb_id,
                "source_id": job.source_id,
                "seed_url": job.seed_url,
                "status": job.status.value,
                "error": job.error.model_dump() if job.error else None,
                "counters": job.counters.model_dump(),
                "usage": job.usage.model_dump(),
                "attempts": job.attempts,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "report": None
                if report is None
                else {
                    "pages": {status.value: n for status, n in report.pages.items()},
                    "stop_reason": report.stop_reason.value if report.stop_reason else None,
                    "stop_detail": report.stop_detail,
                },
            }
        },
    }


def render_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def retry_delay(
    attempt: int, *, retry_after: float | None, config: Settings, rng: Callable[[], float]
) -> float:
    """Seconds before attempt `attempt + 1`: exponential with equal jitter, capped, and
    no sooner than the receiver asked (also capped)."""
    cap = config.WEBHOOK_RETRY_MAX_SECONDS
    exponential = min(cap, config.WEBHOOK_RETRY_BASE_SECONDS * 2 ** (attempt - 1))
    delay = exponential / 2 + rng() * exponential / 2
    if retry_after is not None:
        delay = max(delay, min(retry_after, cap))
    return delay


def _parse_retry_after(value: str | None, now: datetime) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        return max(0.0, (parsedate_to_datetime(value) - now).total_seconds())
    except TypeError, ValueError:
        return None


type ClientFactory = Callable[[], AbstractAsyncContextManager[SafeHttpClient]]


class WebhookDeliverer:
    def __init__(
        self,
        db: MongoDatabase,
        box: SecretBox,
        *,
        client_factory: ClientFactory | None = None,
        config: Settings = settings,
        clock: Callable[[], datetime] = utc_now,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.jobs = JobRepository(db, clock=clock)
        self.secrets = WebhookSecretService(TenantRepository(db), box)
        self.tenants = TenantRepository(db)
        self._config = config
        self._clock = clock
        self._rng = rng
        timeout = httpx.Timeout(config.WEBHOOK_TIMEOUT_SECONDS)
        self._client_factory = client_factory or (
            lambda: SafeHttpClient(config, max_connections=2, timeout=timeout)
        )

    async def attempt(self, job_id: str) -> AttemptResult:
        job = await self.jobs.get(job_id)
        if job is None or job.webhook is None or not job.webhook.status.is_open:
            return AttemptResult(DeliveryOutcome.SKIPPED)
        url = job.config.webhook_url
        if url is None:  # can't happen (the transition only marks jobs with a URL)
            return AttemptResult(DeliveryOutcome.SKIPPED)
        number = job.webhook.attempts + 1
        log = logger.bind(job_id=job.id, tenant_id=job.tenant_id, attempt=number)

        tenant = await self.tenants.get(job.tenant_id)
        secrets = await self.secrets.signing_secrets(tenant) if tenant is not None else []
        started = time.monotonic()
        if not secrets:
            answer = _Answer(
                None, "webhook_secret_missing", "the tenant has no signing secret", retryable=False
            )
        else:
            answer = await self._post(job, url, secrets, number)
        duration_ms = int((time.monotonic() - started) * 1000)

        retry = answer.retryable and number < self._config.WEBHOOK_MAX_ATTEMPTS
        if answer.error is None:
            status = WebhookStatus.DELIVERED
        elif retry:
            status = WebhookStatus.RETRYING
        else:
            status = WebhookStatus.FAILED
        detail = answer.detail
        if answer.retryable and not retry:
            detail = f"gave up after {number} attempts: {detail or answer.error}"
        record = WebhookAttempt(
            attempt=number,
            at=self._clock(),
            status_code=answer.status_code,
            error=answer.error,
            detail=detail[:500] if detail else None,
            duration_ms=duration_ms,
        )
        if await self.jobs.record_webhook_attempt(job.id, record, status) is None:
            log.info("webhook attempt {} of job {} already recorded; skipping", number, job.id)
            return AttemptResult(DeliveryOutcome.SKIPPED, attempt=number)

        if status is WebhookStatus.DELIVERED:
            log.info("webhook of job {} delivered ({})", job.id, answer.status_code)
            return AttemptResult(DeliveryOutcome.DELIVERED, number, answer.status_code)
        if status is WebhookStatus.RETRYING:
            delay = retry_delay(
                number, retry_after=answer.retry_after, config=self._config, rng=self._rng
            )
            log.warning("webhook of job {} failed ({}); retry in {:.0f}s", job.id, answer.error,
                        delay)  # fmt: skip
            return AttemptResult(DeliveryOutcome.RETRY, number, answer.status_code,
                                 answer.error, delay)  # fmt: skip
        log.error("webhook of job {} failed for good: {} ({})", job.id, answer.error, detail)
        return AttemptResult(DeliveryOutcome.FAILED, number, answer.status_code, answer.error)

    async def _post(self, job: CrawlJob, url: str, secrets: list[str], number: int) -> _Answer:
        payload = build_payload(job)
        body = render_body(payload)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self._config.WEBHOOK_USER_AGENT,
            EVENT_HEADER: payload["type"],
            DELIVERY_HEADER: payload["id"],
            ATTEMPT_HEADER: str(number),
            **signature_headers(body, secrets, now=self._clock().timestamp()),
        }
        try:
            async with (
                self._client_factory() as client,
                client.send_once("POST", url, headers=headers, content=body) as response,
            ):
                await _drain(response)
                return self._classify(response)
        except SsrfBlockedError as exc:
            return _Answer(None, "ssrf_blocked", exc.message, retryable=False)
        except HostResolutionError as exc:
            return _Answer(None, "dns_error", str(exc) or "DNS lookup failed", retryable=True)
        except httpx.TimeoutException as exc:
            return _Answer(None, "timeout", type(exc).__name__, retryable=True)
        except httpx.TransportError as exc:  # connection refused/reset, TLS, protocol
            return _Answer(None, "network_error", f"{type(exc).__name__}: {exc}"[:500],
                           retryable=True)  # fmt: skip

    def _classify(self, response: httpx.Response) -> _Answer:
        code = response.status_code
        if 200 <= code < 300:
            return _Answer(code, None, None, retryable=False)
        if code in _RETRYABLE_STATUSES or code >= 500:
            retry_after = _parse_retry_after(response.headers.get("retry-after"), self._clock())
            return _Answer(code, "http_error", f"HTTP {code}", retryable=True,
                           retry_after=retry_after)  # fmt: skip
        if 300 <= code < 400:
            return _Answer(code, "redirect_not_followed",
                           f"HTTP {code} to {response.headers.get('location', '?')}"[:500],
                           retryable=False)  # fmt: skip
        return _Answer(code, "http_error", f"HTTP {code}", retryable=False)


async def _drain(response: httpx.Response) -> None:
    """Read (a bounded amount of) the answer so the connection can be reused; dropped."""
    read = 0
    async for chunk in response.aiter_raw():
        read += len(chunk)
        if read >= _MAX_DRAIN_BYTES:
            break
