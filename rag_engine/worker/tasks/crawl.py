"""`crawl` queue tasks (Step 77)."""

from rag_engine.worker.celery_app import CRAWL_QUEUE, celery_app
from rag_engine.worker.context import run_async, worker_context
from rag_engine.worker.crawl_job import CrawlJobDeps, CrawlJobRunner
from rag_engine.worker.producer import send_job_webhook
from rag_engine.worker.tasks.maintenance import enqueue_version_cleanup

RUN_CRAWL_JOB = "rag_engine.worker.tasks.crawl.run_crawl_job"


def enqueue_crawl_job(job_id: str, countdown: float = 0) -> None:
    """Send a crawl job to the `crawl` queue by name (producers need not import workers)."""
    celery_app.send_task(
        RUN_CRAWL_JOB, args=[job_id], queue=CRAWL_QUEUE, countdown=countdown or None
    )


@celery_app.task(name=RUN_CRAWL_JOB)
def run_crawl_job(job_id: str) -> str:
    """Run one crawl job end to end; returns the `JobOutcome`. Idempotent (late acks)."""
    from rag_engine.crawling.fetchers.browser import process_browser_pool

    deps = CrawlJobDeps.from_runtime(
        worker_context.runtime,
        enqueue=enqueue_crawl_job,
        enqueue_cleanup=enqueue_version_cleanup,
        enqueue_webhook=send_job_webhook,
        browser_pool=process_browser_pool(),
    )
    return run_async(CrawlJobRunner(deps).run(job_id)).value
