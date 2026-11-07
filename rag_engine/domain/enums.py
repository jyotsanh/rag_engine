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


TERMINAL_JOB_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})
ACTIVE_JOB_STATUSES = frozenset(set(JobStatus) - TERMINAL_JOB_STATUSES)


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
    MAX_PAGES = "max_pages"
    CANONICAL_ELSEWHERE = "canonical_elsewhere"
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    EMPTY_CONTENT = "empty_content"
    JOB_CANCELLED = "job_cancelled"
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
    WORKER_LOST = "worker_lost"  # claimed the maximum number of times, never reported back


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
    "TERMINAL_JOB_STATUSES",
    "ContentKind",
    "DiscoveredVia",
    "EmbeddingModelName",
    "JobStatus",
    "PageReason",
    "PageStatus",
    "RenderedWith",
    "ResourceStatus",
    "TenantStatus",
]
