"""The Celery application (Step 75): `celery -A rag_engine.worker.celery_app worker`.

- **Broker and result backend**: Redis, on their own DBs (`REDIS_CELERY_BROKER_DB` /
  `REDIS_CELERY_RESULT_DB`); the application's keys never share a DB with Celery's.
- **At-least-once delivery**: `task_acks_late` (a task is acknowledged after it finished)
  plus `task_reject_on_worker_lost` (a task whose worker died is redelivered) and
  `worker_prefetch_multiplier = 1` (a busy worker never sits on queued crawls). Tasks must
  therefore be idempotent; crawl jobs are (the frontier resumes, Step 48).
- **Visibility timeout**: with late acks, Redis redelivers an unacknowledged task after
  the visibility timeout. It must exceed the longest task (the hard time limit), or a
  long crawl would be started a second time while still running.
- **Queues**: `crawl` (long, browser-heavy), `maintenance` (cleanup, cascades, recovery,
  Celery beat) and `webhooks` (short HTTP deliveries), routed by task module, so each
  can get its own workers and concurrency.
- **Beat** (`celery -A rag_engine.worker.celery_app beat`, one process per deployment):
  `recover_jobs` every `CRAWL_RECOVERY_INTERVAL_SECONDS` on the `maintenance` queue (Step
  80) and `sweep_deletions` every `CASCADE_SWEEP_INTERVAL_SECONDS` (Step 81). Each tick
  expires after one interval, so a backlog of ticks never piles up.
- JSON only (no pickle), UTC, time limits from settings, results kept for a day, and
  Celery does not take over the root logger (`setup_logging` runs in the worker init).
"""

from celery import Celery
from kombu import Queue

from rag_engine.config import Settings, settings

CRAWL_QUEUE = "crawl"
MAINTENANCE_QUEUE = "maintenance"
WEBHOOKS_QUEUE = "webhooks"
QUEUES = (CRAWL_QUEUE, MAINTENANCE_QUEUE, WEBHOOKS_QUEUE)
TASK_ROUTES = {
    "rag_engine.worker.tasks.crawl.*": {"queue": CRAWL_QUEUE},
    "rag_engine.worker.tasks.maintenance.*": {"queue": MAINTENANCE_QUEUE},
    "rag_engine.worker.tasks.webhooks.*": {"queue": WEBHOOKS_QUEUE},
}
RUN_CRAWL_JOB_TASK = "rag_engine.worker.tasks.crawl.run_crawl_job"
RECOVER_JOBS_TASK = "rag_engine.worker.tasks.maintenance.recover_jobs"
DELETE_SOURCE_TASK = "rag_engine.worker.tasks.maintenance.delete_source"
DELETE_KNOWLEDGE_BASE_TASK = "rag_engine.worker.tasks.maintenance.delete_knowledge_base"
SWEEP_DELETIONS_TASK = "rag_engine.worker.tasks.maintenance.sweep_deletions"
DELIVER_WEBHOOK_TASK = "rag_engine.worker.tasks.webhooks.deliver_job_webhook"
# a redelivery margin on top of the hard time limit (see the module docstring)
VISIBILITY_MARGIN_SECONDS = 15 * 60
RESULT_EXPIRES_SECONDS = 24 * 60 * 60


def configure(app: Celery, config: Settings = settings) -> Celery:
    """Apply the configuration (tests re-apply it once settings point at containers)."""
    app.conf.update(
        broker_url=config.celery_broker_url,
        result_backend=config.celery_result_backend,
        broker_connection_retry_on_startup=True,
        broker_transport_options={
            "visibility_timeout": config.CELERY_TASK_TIME_LIMIT_SECONDS + VISIBILITY_MARGIN_SECONDS
        },
        result_backend_transport_options={"retry_policy": {"timeout": 5.0}},
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        task_soft_time_limit=config.CELERY_TASK_SOFT_TIME_LIMIT_SECONDS,
        task_time_limit=config.CELERY_TASK_TIME_LIMIT_SECONDS,
        result_expires=RESULT_EXPIRES_SECONDS,
        task_queues=[Queue(name) for name in QUEUES],
        task_default_queue=MAINTENANCE_QUEUE,
        task_routes=TASK_ROUTES,
        worker_concurrency=config.CELERY_WORKER_CONCURRENCY,
        worker_hijack_root_logger=False,
        worker_redirect_stdouts=False,
        beat_schedule={
            "recover-crawl-jobs": {
                "task": RECOVER_JOBS_TASK,
                "schedule": config.CRAWL_RECOVERY_INTERVAL_SECONDS,
                "options": {
                    "queue": MAINTENANCE_QUEUE,
                    "expires": config.CRAWL_RECOVERY_INTERVAL_SECONDS,
                },
            },
            "sweep-deletions": {
                "task": SWEEP_DELETIONS_TASK,
                "schedule": config.CASCADE_SWEEP_INTERVAL_SECONDS,
                "options": {
                    "queue": MAINTENANCE_QUEUE,
                    "expires": config.CASCADE_SWEEP_INTERVAL_SECONDS,
                },
            },
        },
    )
    return app


celery_app = configure(
    Celery(
        "rag_engine",
        include=[
            "rag_engine.worker.tasks.crawl",
            "rag_engine.worker.tasks.maintenance",
            "rag_engine.worker.tasks.webhooks",
        ],
    )
)

from rag_engine.worker import signals  # noqa: E402, F401  (connects the process signals)
