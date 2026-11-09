"""Starting a crawl (Step 84): `POST /knowledge-bases/{kb_id}/crawls`.

`CrawlSubmissionService.submit` checks everything that can be checked before any record
is written, so a refused request leaves nothing behind:

1. The KB is the tenant's and live (404 otherwise; a deleting KB is invisible).
2. **The seed URL** is normalized (Step 23; 422 `invalid_url`), so trivially different
   spellings map to one source, then **SSRF-vetted with DNS** (422 `url_not_allowed`, or
   `host_unresolvable` when it doesn't resolve). The worker re-vets every request anyway;
   this only turns an obviously bad URL into an immediate 422 instead of a failed job.
3. `path_prefix` must contain the seed's own path (or the seed itself is out of scope).
4. **Tenant quotas**: `max_pages` defaults to `min(CRAWL_DEFAULT_MAX_PAGES,
   max_pages_per_job)`; an explicit value above the tenant's `max_pages_per_job` is 403
   `quota_exceeded` (it is never silently lowered).
5. **`webhook_url`**: SSRF-vetted the same way (422 `webhook_url_not_allowed` /
   `host_unresolvable`), and the tenant must have a webhook signing secret (422
   `webhook_secret_missing`, Steps 82-83): an unsigned webhook couldn't be trusted.

A cheap concurrent-job pre-check keeps a crawl that will be refused from creating its
source (only a race can still leave an empty source behind; the next submission of that
seed reuses it). Then the source is upserted (`get_or_create` on the normalized seed) and
the job goes through the concurrency guards (Step 79), which enforce the quota atomically:
a duplicate returns the source's active job (`created=False`, not re-enqueued); a new job
is enqueued on the `crawl` queue. A failed publish leaves the job queued, and crash
recovery re-sends it (Step 80).
"""

from urllib.parse import urlsplit

from rag_engine.config import Settings, logger, settings
from rag_engine.crawling.url import UrlNormalizationError, normalize_url
from rag_engine.domain import CrawlConfig, Tenant
from rag_engine.exceptions import AppValidationError, QuotaExceededError
from rag_engine.repositories import KnowledgeBaseRepository
from rag_engine.repositories.sources import SourceRepository
from rag_engine.security import HostResolutionError, SsrfBlockedError, SsrfGuard
from rag_engine.services.crawl_jobs import CrawlJobService, JobSubmission
from rag_engine.worker.producer import TaskQueue


class InvalidUrlError(AppValidationError):
    code = "invalid_url"
    default_message = "The URL is not a crawlable http(s) URL"


class HostUnresolvableError(AppValidationError):
    code = "host_unresolvable"
    default_message = "The URL's host name does not resolve"


class WebhookUrlNotAllowedError(AppValidationError):
    code = "webhook_url_not_allowed"
    default_message = "The webhook URL points to a non-public network location"


class WebhookSecretMissingError(AppValidationError):
    code = "webhook_secret_missing"
    default_message = (
        "Webhooks are signed with the tenant's webhook secret, and none has been issued; "
        "ask an administrator to issue one"
    )


class SeedOutsidePathPrefixError(AppValidationError):
    code = "seed_outside_path_prefix"
    default_message = "The seed URL's path is outside path_prefix, so nothing would be crawled"


class CrawlSubmissionService:
    def __init__(
        self,
        kbs: KnowledgeBaseRepository,
        sources: SourceRepository,
        jobs: CrawlJobService,
        tasks: TaskQueue,
        *,
        guard: SsrfGuard | None = None,
        config: Settings = settings,
    ) -> None:
        self._kbs = kbs
        self._sources = sources
        self._jobs = jobs
        self._tasks = tasks
        self._guard = guard or SsrfGuard(config)
        self._config = config

    async def submit(
        self, tenant: Tenant, kb_id: str, url: str, config: CrawlConfig
    ) -> JobSubmission:
        """Queue a crawl of `url` into the KB (or return the crawl already running there).

        `config` is already resolved against the tenant's quotas (`resolve_max_pages`).
        """
        kb = await self._kbs.get_active(kb_id, tenant.id)
        seed = await self._vet_seed(url)
        if config.path_prefix and not (urlsplit(seed).path or "/").startswith(config.path_prefix):
            raise SeedOutsidePathPrefixError(
                details={"seed_path": urlsplit(seed).path or "/", "path_prefix": config.path_prefix}
            )
        if config.webhook_url is not None:
            await self._vet_webhook(tenant, config.webhook_url)

        existing = await self._sources.find_active_by_seed(tenant.id, kb.id, seed)
        active = await self._jobs.would_exceed_quota(tenant, existing)
        if active is not None:
            raise self._jobs.quota_error(tenant, active)
        source, created_source = await self._sources.get_or_create(tenant.id, kb.id, seed)
        submission = await self._jobs.submit(tenant, source, config)
        log = logger.bind(kb_id=kb.id, source_id=source.id, job_id=submission.job.id)
        if submission.created:
            await self._tasks.enqueue_crawl_job(submission.job.id)
            log.info("crawl of {} submitted{}", seed, " (new source)" if created_source else "")
        return submission

    def resolve_max_pages(self, tenant: Tenant, requested: int | None) -> int:
        """The job's page cap: the request's (within the tenant's quota) or the default."""
        quota = tenant.quotas.max_pages_per_job
        if requested is None:
            return min(self._config.CRAWL_DEFAULT_MAX_PAGES, quota)
        if requested > quota:
            raise QuotaExceededError(
                "max_pages is above the tenant's page quota per job",
                details={"quota": "max_pages_per_job", "limit": quota, "requested": requested},
            )
        return requested

    async def _vet_seed(self, url: str) -> str:
        try:
            seed = normalize_url(url)
        except UrlNormalizationError as exc:
            raise InvalidUrlError(details={"field": "url", "reason": str(exc)}) from exc
        try:
            await self._guard.vet(seed)
        except HostResolutionError as exc:
            raise HostUnresolvableError(
                details={"field": "url", "host": urlsplit(seed).hostname}
            ) from exc
        return seed

    async def _vet_webhook(self, tenant: Tenant, url: str) -> None:
        try:
            await self._guard.vet(url)
        except SsrfBlockedError as exc:
            details = {"field": "webhook_url", **exc.details}
            raise WebhookUrlNotAllowedError(details=details) from exc
        except HostResolutionError as exc:
            raise HostUnresolvableError(
                details={"field": "webhook_url", "host": urlsplit(url).hostname}
            ) from exc
        if not tenant.webhook_secrets:
            raise WebhookSecretMissingError()
