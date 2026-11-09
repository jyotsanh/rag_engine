"""`webhooks` queue tasks: completion webhook delivery (Step 83)."""

from dataclasses import asdict
from typing import Any

from celery import Task

from rag_engine.infrastructure import DatabaseUnavailableError
from rag_engine.security import SecretBox
from rag_engine.webhooks.delivery import AttemptResult, DeliveryOutcome, WebhookDeliverer
from rag_engine.worker.celery_app import DELIVER_WEBHOOK_TASK, celery_app
from rag_engine.worker.context import run_async, worker_context

_OUTAGE_RETRY_SECONDS = 60


@celery_app.task(name=DELIVER_WEBHOOK_TASK, bind=True, max_retries=10)
def deliver_job_webhook(self: Task, job_id: str) -> dict[str, Any]:
    """One delivery attempt; the next one is this task retried after the backoff.

    Attempts are counted on the job (`webhook.attempts`), not by Celery: a MongoDB outage
    retries without spending one, and the recovery sweep re-sends a lost retry.
    """
    runtime = worker_context.runtime
    deliverer = WebhookDeliverer(
        runtime.mongo.db, SecretBox.from_settings(runtime.config), config=runtime.config
    )
    try:
        result = run_async(deliverer.attempt(job_id))
    except DatabaseUnavailableError as exc:
        raise self.retry(exc=exc, countdown=_OUTAGE_RETRY_SECONDS) from exc
    return settle_attempt(self, result)


def settle_attempt(self: Task, result: AttemptResult) -> dict[str, Any]:
    if result.outcome is DeliveryOutcome.RETRY:
        assert result.retry_in is not None
        # the job's attempt counter is the limit; lift Celery's own explicitly
        raise self.retry(countdown=result.retry_in, max_retries=self.request.retries + 1)
    return asdict(result)
