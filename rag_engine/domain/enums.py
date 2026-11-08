"""Statuses and classifications shared across the domain."""

from enum import StrEnum

from rag_engine.config.enums import EmbeddingModelName


class TenantStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"  # API keys rejected, no new jobs; data kept


class ResourceStatus(StrEnum):
    """Lifecycle of knowledge bases and sources (deletion cascades asynchronously)."""

    ACTIVE = "active"
    DELETING = "deleting"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    FINALIZING = "finalizing"  # crawl done; swapping the active version
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_JOB_STATUSES

    def can_become(self, target: JobStatus) -> bool:
        return target in JOB_TRANSITIONS[self]


TERMINAL_JOB_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})
ACTIVE_JOB_STATUSES = frozenset(set(JobStatus) - TERMINAL_JOB_STATUSES)

# The job state machine (Step 49). Anything not listed is refused.
JOB_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    # a worker picks it up; it can fail before starting (e.g. its source was deleted) or be
    # cancelled outright, since nothing has been written yet
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED}),
    # crawl done → finalize; failed; cancelled once in-flight pages drained (Step 52);
    # back to queued when its worker died (stale heartbeat, Step 80): the frontier resumes
    JobStatus.RUNNING: frozenset(
        {JobStatus.FINALIZING, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.QUEUED}
    ),
    # swapping the active version (Step 78) is not cancellable: it either completes or fails
    # (keeping the old version); a worker dying mid-swap requeues it to finalize again
    JobStatus.FINALIZING: frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.QUEUED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


class PageStatus(StrEnum):
    QUEUED = "queued"  # in the frontier
    IN_PROGRESS = "in_progress"  # claimed by a crawler
    SUCCEEDED = "succeeded"  # fetched and extracted
    SKIPPED = "skipped"  # deliberately not fetched/kept (see PageReason)
    DUPLICATE = "duplicate"  # same content as `duplicate_of`
    FAILED = "failed"  # tried and failed (see PageReason)


class PageReason(StrEnum):
    """Why a page was skipped or failed. Stable codes surfaced in the per-page report."""

    # skipped
    ROBOTS_DISALLOWED = "robots_disallowed"
    OUT_OF_SCOPE = "out_of_scope"
    NOFOLLOW = "nofollow"
    MAX_DEPTH = "max_depth"
    MAX_PAGES = "max_pages"  # the job's page budget was spent before this page (Step 51)
    MAX_TOTAL_BYTES = "max_total_bytes"  # the job's download budget ran out (Step 51)
    MAX_DURATION = "max_duration"  # the job ran out of wall time (Step 51)
    CANONICAL_ELSEWHERE = "canonical_elsewhere"
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    EMPTY_CONTENT = "empty_content"
    JOB_CANCELLED = "job_cancelled"
    REDIRECTED = "redirected"  # its content is recorded under the redirect target's page
    NOINDEX = "noindex"  # robots meta / X-Robots-Tag noindex: links followed, not indexed
    # failed
    TOO_LARGE = "too_large"
    REDIRECT_LOOP = "redirect_loop"
    TOO_MANY_REDIRECTS = "too_many_redirects"
    SSRF_BLOCKED = "ssrf_blocked"
    HTTP_CLIENT_ERROR = "http_client_error"  # 4xx except 429
    HTTP_SERVER_ERROR = "http_server_error"  # 5xx after retries
    RATE_LIMITED = "rate_limited"  # 429 after retries
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"  # DNS, connection refused/reset, TLS
    RENDER_FAILED = "render_failed"
    EXTRACTION_FAILED = "extraction_failed"
    DOCUMENT_UNREADABLE = "document_unreadable"  # encrypted, no text layer
    INTERNAL_ERROR = "internal_error"  # a bug while processing the page (logged)
    SERVICE_UNAVAILABLE = "service_unavailable"  # Redis etc. kept failing on this page
    JOB_FAILED = "job_failed"  # the job failed before this page finished (Step 53)
    WORKER_LOST = "worker_lost"  # claimed the maximum number of times, never reported back


class CrawlStopReason(StrEnum):
    """Why a crawl stopped before its frontier drained (caps: Step 51, cancel: Step 52)."""

    MAX_PAGES = "max_pages"  # soft: no new pages, but pages already counted still finish
    MAX_TOTAL_BYTES = "max_total_bytes"  # hard: nothing more is fetched
    MAX_DURATION = "max_duration"  # hard: nothing more is fetched
    CANCELLED = "cancelled"  # hard, and it outranks the caps: the job ends `cancelled`

    @property
    def page_reason(self) -> PageReason:
        """What the pages left in the frontier are recorded as."""
        return _STOP_PAGE_REASON[self]

    @property
    def is_hard(self) -> bool:
        """A hard stop fetches nothing more; a soft one only stops taking new pages."""
        return self is not CrawlStopReason.MAX_PAGES

    @property
    def rank(self) -> int:
        """A stop only replaces one of a lower rank: soft < hard cap < cancelled."""
        if self is CrawlStopReason.CANCELLED:
            return 2
        return 1 if self.is_hard else 0


_STOP_PAGE_REASON = {
    CrawlStopReason.MAX_PAGES: PageReason.MAX_PAGES,
    CrawlStopReason.MAX_TOTAL_BYTES: PageReason.MAX_TOTAL_BYTES,
    CrawlStopReason.MAX_DURATION: PageReason.MAX_DURATION,
    CrawlStopReason.CANCELLED: PageReason.JOB_CANCELLED,
}


class UsagePurpose(StrEnum):
    """What embedding usage was for (Step 65)."""

    INGEST = "ingest"  # chunks embedded by crawl jobs
    QUERY = "query"  # search queries embedded by the API


class ContentKind(StrEnum):
    """Extractor family chosen for a fetched resource (Step 38 registry keys on this)."""

    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"


class DiscoveredVia(StrEnum):
    """How a URL entered the frontier; ties at one depth are broken in this order (Step 48)."""

    SEED = "seed"  # the job's start URL
    SITEMAP = "sitemap"  # robots.txt / sitemap.xml entry (depth 0)
    CANONICAL = "canonical"  # rel=canonical target of a page skipped as canonical_elsewhere
    LINK = "link"  # outlink of a fetched page (depth + 1)


class RenderedWith(StrEnum):
    HTTP = "http"  # plain HTTP fetch
    BROWSER = "browser"  # re-rendered in headless Chromium


__all__ = [
    "ACTIVE_JOB_STATUSES",
    "JOB_TRANSITIONS",
    "TERMINAL_JOB_STATUSES",
    "ContentKind",
    "CrawlStopReason",
    "DiscoveredVia",
    "EmbeddingModelName",
    "JobStatus",
    "PageReason",
    "PageStatus",
    "RenderedWith",
    "ResourceStatus",
    "TenantStatus",
    "UsagePurpose",
]
