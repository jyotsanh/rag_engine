from typing import Literal, Self
from urllib.parse import quote

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_engine.config.enums import (
    ChunkerType,
    EmbeddingModelName,
    EmbeddingProvider,
    Environment,
    JsRenderMode,
    RerankerType,
    SparseEmbedderType,
    VectorStoreType,
)

# Playwright `request.resource_type` values that may be blocked (never "document")
type BrowserResourceType = Literal[
    "stylesheet",
    "image",
    "media",
    "font",
    "script",
    "texttrack",
    "xhr",
    "fetch",
    "eventsource",
    "websocket",
    "manifest",
    "other",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,  # ignore empty key
        extra="ignore",  # ignores extra key if provided
    )

    # ---- app ---- #
    ENVIRONMENT: Environment = Environment.DEV
    PROJECT_NAME: str = "RAG Engine"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_JSON: bool = True
    # each dependency ping in GET /health/ready must answer within this budget
    READINESS_CHECK_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0, le=30)

    # Production HOST & PORT
    PROD_HOST: str = "0.0.0.0"
    PROD_PORT: int = Field(default=8000, ge=1, le=65535)

    # Development HOST & PORT
    DEV_HOST: str = "127.0.0.1"
    DEV_PORT: int = Field(default=8000, ge=1, le=65535)

    # cors
    BACKEND_CORS_ORIGINS: list[str] = []
    ALLOW_CREDENTIALS: bool = False

    # ---- security ---- #
    # master key for /admin/* (tenant + API key management)
    ADMIN_API_KEY: SecretStr = Field(min_length=32)
    # server-side pepper mixed into tenant API key hashes
    API_KEY_PEPPER: SecretStr = Field(min_length=32)
    # retired peppers still accepted during rotation; keys verified with one get re-hashed
    API_KEY_PEPPER_PREVIOUS: list[SecretStr] = []
    # outbound fetches (crawler, robots, webhooks) may only target these ports
    SSRF_ALLOWED_PORTS: list[int] = [80, 443]
    # ONLY for tests/local fixture sites — never enable in prod
    SSRF_ALLOW_PRIVATE_NETWORKS: bool = False

    # ---- mongodb (jobs, pages, chunks, tenants) ---- #
    # defaults match the host ports published by docker-compose.yml
    MONGO_HOST: str = "localhost"
    MONGO_PORT: int = Field(default=27020, ge=1, le=65535)
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_DB: str = "rag_engine"
    # full URI overrides HOST/PORT/USER/PASSWORD when set
    MONGO_URI: SecretStr | None = None
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = Field(default=5_000, gt=0)
    MONGO_CONNECT_TIMEOUT_MS: int = Field(default=5_000, gt=0)
    MONGO_MAX_POOL_SIZE: int = Field(default=100, gt=0)

    # ---- redis (rate limits, caches, cancel flags, celery) ---- #
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = Field(default=6380, ge=1, le=65535)
    REDIS_PASSWORD: SecretStr | None = None
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_CELERY_BROKER_DB: int = Field(default=1, ge=0, le=15)
    REDIS_CELERY_RESULT_DB: int = Field(default=2, ge=0, le=15)
    # every app key starts with "<prefix>:" so one Redis can be shared and scanned safely
    REDIS_KEY_PREFIX: str = Field(default="rag", pattern=r"^[a-z0-9_-]{1,32}$")
    REDIS_SOCKET_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0)
    REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0)
    REDIS_MAX_CONNECTIONS: int = Field(default=100, gt=0)

    # ---- celery ---- #
    CELERY_TASK_SOFT_TIME_LIMIT_SECONDS: int = Field(default=3 * 60 * 60, gt=0)
    CELERY_TASK_TIME_LIMIT_SECONDS: int = Field(default=3 * 60 * 60 + 300, gt=0)
    CELERY_WORKER_CONCURRENCY: int = Field(default=2, gt=0)

    # ---- vector store ---- #
    ACTIVE_VECTOR_STORE: VectorStoreType = VectorStoreType.QDRANT
    QDRANT_HOST: str = "localhost"
    QDRANT_HTTP_PORT: int = Field(default=6333, ge=1, le=65535)
    QDRANT_GRPC_PORT: int = Field(default=6334, ge=1, le=65535)
    QDRANT_PREFER_GRPC: bool = True
    QDRANT_HTTPS: bool = False
    QDRANT_API_KEY: SecretStr | None = None
    QDRANT_TIMEOUT_SECONDS: int = Field(default=30, gt=0)
    # points per upsert request (each acknowledged with wait=True)
    QDRANT_UPSERT_BATCH_SIZE: int = Field(default=256, gt=0, le=10_000)
    # retries of an unavailable Qdrant for idempotent writes (upserts by id, deletes by
    # filter): jittered backoff, delay before retry n in [d/2, d], d = min(max, base·2^(n-1))
    QDRANT_MAX_RETRIES: int = Field(default=3, ge=0)
    QDRANT_RETRY_BASE_DELAY_SECONDS: float = Field(default=0.5, gt=0)
    QDRANT_RETRY_MAX_DELAY_SECONDS: float = Field(default=8.0, gt=0)

    # ---- embeddings ---- #
    ACTIVE_EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.OPENAI
    OPENAI_API_KEY: SecretStr = Field(min_length=1)
    OPENAI_BASE_URL: str | None = None
    OPENAI_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    OPENAI_MAX_RETRIES: int = Field(default=6, ge=0)
    OPENAI_MAX_CONCURRENT_REQUESTS: int = Field(default=4, gt=0)
    # one embeddings request carries at most this many inputs / summed input tokens (the
    # model's own limits, 2048 / 300k for OpenAI, cap these); smaller requests keep a
    # retried failure cheap
    EMBEDDING_MAX_BATCH_INPUTS: int = Field(default=256, gt=0)
    EMBEDDING_MAX_BATCH_TOKENS: int = Field(default=100_000, gt=0)
    # retries of 429/5xx/timeouts (OPENAI_MAX_RETRIES of them): delay before retry n is in
    # [d/2, d], d = min(max, base·2^(n-1)); a provider's Retry-After is a floor on it
    EMBEDDING_RETRY_BASE_DELAY_SECONDS: float = Field(default=1.0, gt=0)
    EMBEDDING_RETRY_MAX_DELAY_SECONDS: float = Field(default=60.0, gt=0)
    # a provider asking to wait longer than this fails the request instead of stalling it
    EMBEDDING_RETRY_AFTER_MAX_SECONDS: float = Field(default=300.0, ge=0)
    # reuse vectors of unchanged texts across jobs (Mongo `embedding_cache`, Step 64)
    EMBEDDING_CACHE_ENABLED: bool = True
    # evict cache entries unused for this long (TTL index; applies when indexes are created)
    EMBEDDING_CACHE_TTL_DAYS: int = Field(default=90, ge=1)
    # tenant default when the admin does not pick one; a KB locks its model at creation
    DEFAULT_EMBEDDING_MODEL: EmbeddingModelName = EmbeddingModelName.TEXT_EMBEDDING_3_SMALL

    # ---- sparse embeddings (Step 66) ---- #
    SPARSE_EMBEDDER: SparseEmbedderType = SparseEmbedderType.BM25
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"
    # stemmer + stopword list (fastembed BM25 languages: english, german, french, ...)
    SPARSE_EMBEDDING_LANGUAGE: str = "english"
    # where fastembed keeps model files; images bake them here (Step 95, with HF_HUB_OFFLINE=1).
    # None: fastembed's default (<tmp>/fastembed_cache)
    FASTEMBED_CACHE_DIR: str | None = None

    # ---- ingestion (Step 76) ---- #
    # chunks per embedding batch (dense + sparse); each batch is metered and upserted
    INGEST_EMBED_BATCH_SIZE: int = Field(default=128, gt=0, le=2048)
    # batches buffered between the chunk → embed → upsert stages (backpressure)
    INGEST_QUEUE_SIZE: int = Field(default=4, gt=0, le=64)

    # ---- chunking (Steps 55-58) ---- #
    CHUNKER: ChunkerType = ChunkerType.STRUCTURE
    # the splitter aims for chunks of about CHUNK_TARGET_TOKENS of text...
    CHUNK_TARGET_TOKENS: int = Field(default=400, gt=0)
    # ...never above CHUNK_MAX_TOKENS of embedded text (title + heading path + chunk, Step 58);
    # far below the embedding models' 8191-token input limit, which also bounds it
    CHUNK_MAX_TOKENS: int = Field(default=800, gt=0, le=8000)
    # tokens of the previous chunk repeated at the start of the next (prose only, Step 57)
    CHUNK_OVERLAP_TOKENS: int = Field(default=60, ge=0)
    # cap on the context put in front of each chunk's embedded text (page title + heading
    # path, Step 58); 0 disables it. Must fit what the cap leaves above the target
    CHUNK_CONTEXT_MAX_TOKENS: int = Field(default=100, ge=0)

    # ---- reranker ---- #
    ACTIVE_RERANKER: RerankerType = RerankerType.CROSS_ENCODER

    # ---- tenant quota defaults (admins override per tenant) ---- #
    TENANT_DEFAULT_MAX_KNOWLEDGE_BASES: int = Field(default=20, gt=0)
    TENANT_DEFAULT_MAX_CONCURRENT_JOBS: int = Field(default=2, gt=0)
    TENANT_DEFAULT_MAX_PAGES_PER_JOB: int = Field(default=2_000, gt=0)
    TENANT_DEFAULT_API_RATE_LIMIT_PER_MINUTE: int = Field(default=120, gt=0)
    # hard cap on usable (not revoked, not expired) API keys per tenant
    TENANT_MAX_ACTIVE_API_KEYS: int = Field(default=10, gt=0, le=1000)

    # ---- crawler ---- #
    CRAWL_USER_AGENT: str = "RagEngineBot/0.1 (+https://github.com/jyotsanh/rag_engine)"
    CRAWL_DEFAULT_MAX_PAGES: int = Field(default=500, gt=0)
    CRAWL_HARD_MAX_PAGES: int = Field(default=10_000, gt=0)
    CRAWL_DEFAULT_MAX_DEPTH: int = Field(default=10, ge=0)
    CRAWL_HARD_MAX_DEPTH: int = Field(default=50, ge=0)
    CRAWL_MAX_JOB_SECONDS: int = Field(default=2 * 60 * 60, gt=0)
    CRAWL_MAX_PAGE_BYTES: int = Field(default=20 * 1024 * 1024, gt=0)
    CRAWL_MAX_TOTAL_BYTES: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    CRAWL_CONCURRENCY: int = Field(default=8, gt=0)
    # per-host politeness shared by all workers: at most this many requests per second...
    CRAWL_PER_DOMAIN_RPS: float = Field(default=2.0, gt=0, le=100)
    # ...after an initial burst of this many (1 = strictly spaced; robots crawl-delay forces 1)
    CRAWL_PER_DOMAIN_BURST: int = Field(default=1, ge=1, le=100)
    # cross-page boilerplate (per crawl job): a Markdown unit on >= MIN_RATIO of the job's
    # distinct pages (once at least MIN_PAGES are known) is template and gets stripped;
    # units this long keep one copy on the first page that had them (still searchable once)
    BOILERPLATE_MIN_PAGES: int = Field(default=5, ge=2)
    BOILERPLATE_MIN_RATIO: float = Field(default=0.5, gt=0, le=1)
    BOILERPLATE_KEEP_ONE_MIN_CHARS: int = Field(default=200, ge=0)
    # PDFs: bytes accepted for parsing, and pages extracted (later pages are dropped, flagged)
    PDF_MAX_BYTES: int = Field(default=20 * 1024 * 1024, gt=0)
    PDF_MAX_PAGES: int = Field(default=500, gt=0)
    # Office files (DOCX/PPTX/XLSX) are zips: cap the total *uncompressed* size before parsing
    OOXML_MAX_UNCOMPRESSED_BYTES: int = Field(default=200 * 1024 * 1024, gt=0)
    # PPTX: visible slides extracted (later slides are dropped, flagged)
    PPTX_MAX_SLIDES: int = Field(default=500, gt=0)
    # XLSX: visible sheets, rows per sheet and columns per sheet extracted (rest flagged)
    XLSX_MAX_SHEETS: int = Field(default=50, gt=0)
    XLSX_MAX_ROWS: int = Field(default=2_000, gt=0)
    XLSX_MAX_COLS: int = Field(default=50, gt=0)
    # cleaned pages with fewer letters/digits than this (headings and markers excluded) are
    # skipped as empty_content; catches "Loading…" shells and empty listings, not quality
    CLEAN_MIN_CONTENT_CHARS: int = Field(default=20, ge=0)
    # stored Markdown per page; larger documents are cut at a block boundary and flagged
    # (MongoDB documents max out at 16 MB, and the page row carries more than Markdown)
    PAGE_MAX_MARKDOWN_BYTES: int = Field(default=4 * 1024 * 1024, gt=0, le=12 * 1024 * 1024)
    # wall-clock cap per extracted document (a parser thread over the cap finishes unobserved)
    EXTRACTION_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0)
    # robots.txt Crawl-delay is honoured up to this; a larger value would stall the job
    CRAWL_MAX_CRAWL_DELAY_SECONDS: float = Field(default=30.0, gt=0)
    CRAWL_CONNECT_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0)
    CRAWL_READ_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0)
    # wall-clock cap per fetch incl. redirects; stops servers that drip bytes under the read timeout
    CRAWL_FETCH_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    CRAWL_MAX_REDIRECTS: int = Field(default=5, ge=0)
    # retries after the first attempt; only transient failures (timeouts, 5xx, 429, network) retry
    CRAWL_MAX_RETRIES: int = Field(default=3, ge=0)
    # backoff with jitter: delay before retry n is in [d/2, d], d = min(max, base·2^(n-1))
    CRAWL_RETRY_BASE_DELAY_SECONDS: float = Field(default=1.0, gt=0)
    CRAWL_RETRY_MAX_DELAY_SECONDS: float = Field(default=30.0, gt=0)
    # a longer Retry-After than this makes the page fail instead of waiting (or retrying early)
    CRAWL_RETRY_AFTER_MAX_SECONDS: float = Field(default=120.0, ge=0)
    # skip links marked rel="nofollow" (and every link on meta/X-Robots-Tag nofollow pages)
    CRAWL_RESPECT_NOFOLLOW: bool = True
    # skip pages whose rel=canonical points to another in-scope URL (that URL is crawled)
    CRAWL_RESPECT_CANONICAL: bool = True
    # the engine's deadline for one page's rate-limit wait + fetch + render (the fetchers'
    # own timeouts normally fire first); over it the fetch fails as a retryable timeout
    CRAWL_PAGE_TIMEOUT_SECONDS: float = Field(default=150.0, gt=0)
    # a claimed frontier page whose crawler hasn't reported back within this is reclaimed
    # (a crashed worker); must outlast the page deadline plus extraction
    CRAWL_FRONTIER_LEASE_SECONDS: int = Field(default=10 * 60, gt=0)
    # a running crawl checks its job's Redis cancel flag this often (Step 52)
    CRAWL_CANCEL_POLL_SECONDS: float = Field(default=1.0, gt=0, le=60)
    # cancel flags expire after this; must outlive any running job (>= CRAWL_MAX_JOB_SECONDS)
    CRAWL_CANCEL_FLAG_TTL_SECONDS: int = Field(default=24 * 60 * 60, gt=0)
    # a crawler waits at most this for a host's rate-limit slot, then re-queues the page
    # for that time instead of holding a concurrency slot (slow hosts can't starve others)
    CRAWL_RATE_LIMIT_MAX_WAIT_SECONDS: float = Field(default=5.0, ge=0)
    # link candidates considered per page; link farms beyond this are cut (and flagged)
    CRAWL_MAX_LINKS_PER_PAGE: int = Field(default=5_000, gt=0, le=100_000)
    # re-render HTML in headless Chromium: auto (JS-shell detector) / always / never
    CRAWL_JS_RENDERING: JsRenderMode = JsRenderMode.AUTO
    # per-page navigation/action timeout inside the browser
    CRAWL_RENDER_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0)
    # after "load": wait for network idle at most this long (long-polling pages never idle)
    CRAWL_RENDER_NETWORKIDLE_SECONDS: float = Field(default=5.0, ge=0)
    # auto-scroll for lazy-loaded content: time cap and number of viewport steps (0 = off)
    CRAWL_RENDER_SCROLL_SECONDS: float = Field(default=5.0, ge=0)
    CRAWL_RENDER_SCROLL_MAX_STEPS: int = Field(default=20, ge=0, le=200)
    # headless Chromium (Playwright): pages open at once per worker process, request types
    # aborted before they leave the browser, and a relaunch after this many job contexts
    CRAWL_BROWSER_MAX_PAGES: int = Field(default=4, gt=0, le=64)
    # pydantic copies mutable defaults per instance
    CRAWL_BROWSER_BLOCKED_RESOURCES: list[BrowserResourceType] = ["image", "font", "media"]
    CRAWL_BROWSER_RECYCLE_AFTER_CONTEXTS: int = Field(default=100, gt=0)
    CRAWL_BROWSER_HEADLESS: bool = True
    # sitemaps: per-file byte cap (compressed download and inflated; protocol max is 50 MB),
    # files fetched per discovery, and sitemap-index nesting (the protocol allows 1)
    SITEMAP_MAX_BYTES: int = Field(default=50 * 1024 * 1024, gt=0)
    SITEMAP_MAX_FILES: int = Field(default=50, gt=0, le=1000)
    SITEMAP_MAX_DEPTH: int = Field(default=3, ge=0, le=10)
    # RFC 9309 §2.4: a cached robots.txt should not be used for more than 24 hours
    ROBOTS_CACHE_TTL_SECONDS: int = Field(default=60 * 60, gt=0, le=24 * 60 * 60)
    # 5xx/network failures (complete disallow) are re-checked sooner than good copies
    ROBOTS_UNREACHABLE_CACHE_TTL_SECONDS: int = Field(default=5 * 60, gt=0)
    # an unreachable robots.txt falls back to the last good copy if it is at most this old
    ROBOTS_STALE_MAX_SECONDS: int = Field(default=7 * 24 * 60 * 60, ge=0)

    @model_validator(mode="after")
    def _check_limits(self) -> Self:
        if self.CRAWL_DEFAULT_MAX_PAGES > self.CRAWL_HARD_MAX_PAGES:
            raise ValueError("CRAWL_DEFAULT_MAX_PAGES must be <= CRAWL_HARD_MAX_PAGES")
        if self.CRAWL_DEFAULT_MAX_DEPTH > self.CRAWL_HARD_MAX_DEPTH:
            raise ValueError("CRAWL_DEFAULT_MAX_DEPTH must be <= CRAWL_HARD_MAX_DEPTH")
        if self.TENANT_DEFAULT_MAX_PAGES_PER_JOB > self.CRAWL_HARD_MAX_PAGES:
            raise ValueError("TENANT_DEFAULT_MAX_PAGES_PER_JOB must be <= CRAWL_HARD_MAX_PAGES")
        if self.CRAWL_FETCH_TIMEOUT_SECONDS < self.CRAWL_READ_TIMEOUT_SECONDS:
            raise ValueError("CRAWL_FETCH_TIMEOUT_SECONDS must be >= CRAWL_READ_TIMEOUT_SECONDS")
        if self.CRAWL_RETRY_BASE_DELAY_SECONDS > self.CRAWL_RETRY_MAX_DELAY_SECONDS:
            raise ValueError(
                "CRAWL_RETRY_BASE_DELAY_SECONDS must be <= CRAWL_RETRY_MAX_DELAY_SECONDS"
            )
        if self.ROBOTS_UNREACHABLE_CACHE_TTL_SECONDS > self.ROBOTS_CACHE_TTL_SECONDS:
            raise ValueError(
                "ROBOTS_UNREACHABLE_CACHE_TTL_SECONDS must be <= ROBOTS_CACHE_TTL_SECONDS"
            )
        fetch_seconds = self.CRAWL_FETCH_TIMEOUT_SECONDS + self.CRAWL_RENDER_TIMEOUT_SECONDS
        if fetch_seconds > self.CRAWL_PAGE_TIMEOUT_SECONDS:
            raise ValueError(
                "CRAWL_PAGE_TIMEOUT_SECONDS must be >= "
                "CRAWL_FETCH_TIMEOUT_SECONDS + CRAWL_RENDER_TIMEOUT_SECONDS"
            )
        page_seconds = self.CRAWL_PAGE_TIMEOUT_SECONDS + self.EXTRACTION_TIMEOUT_SECONDS
        if page_seconds >= self.CRAWL_FRONTIER_LEASE_SECONDS:
            raise ValueError(
                "CRAWL_FRONTIER_LEASE_SECONDS must be > "
                "CRAWL_PAGE_TIMEOUT_SECONDS + EXTRACTION_TIMEOUT_SECONDS"
            )
        if self.QDRANT_RETRY_BASE_DELAY_SECONDS > self.QDRANT_RETRY_MAX_DELAY_SECONDS:
            raise ValueError(
                "QDRANT_RETRY_BASE_DELAY_SECONDS must be <= QDRANT_RETRY_MAX_DELAY_SECONDS"
            )
        if self.EMBEDDING_RETRY_BASE_DELAY_SECONDS > self.EMBEDDING_RETRY_MAX_DELAY_SECONDS:
            raise ValueError(
                "EMBEDDING_RETRY_BASE_DELAY_SECONDS must be <= EMBEDDING_RETRY_MAX_DELAY_SECONDS"
            )
        if self.CHUNK_TARGET_TOKENS > self.CHUNK_MAX_TOKENS:
            raise ValueError("CHUNK_TARGET_TOKENS must be <= CHUNK_MAX_TOKENS")
        if self.CHUNK_OVERLAP_TOKENS >= self.CHUNK_TARGET_TOKENS:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be < CHUNK_TARGET_TOKENS")
        if self.CHUNK_CONTEXT_MAX_TOKENS > self.CHUNK_MAX_TOKENS - self.CHUNK_TARGET_TOKENS:
            raise ValueError(
                "CHUNK_CONTEXT_MAX_TOKENS must be <= CHUNK_MAX_TOKENS - CHUNK_TARGET_TOKENS"
            )
        if self.CRAWL_CANCEL_FLAG_TTL_SECONDS < self.CRAWL_MAX_JOB_SECONDS:
            raise ValueError("CRAWL_CANCEL_FLAG_TTL_SECONDS must be >= CRAWL_MAX_JOB_SECONDS")
        if self.CRAWL_MAX_PAGE_BYTES > self.CRAWL_MAX_TOTAL_BYTES:
            raise ValueError("CRAWL_MAX_PAGE_BYTES must be <= CRAWL_MAX_TOTAL_BYTES")
        if self.CELERY_TASK_SOFT_TIME_LIMIT_SECONDS >= self.CELERY_TASK_TIME_LIMIT_SECONDS:
            raise ValueError(
                "CELERY_TASK_SOFT_TIME_LIMIT_SECONDS must be < CELERY_TASK_TIME_LIMIT_SECONDS"
            )
        if any(len(p.get_secret_value()) < 32 for p in self.API_KEY_PEPPER_PREVIOUS):
            raise ValueError("every API_KEY_PEPPER_PREVIOUS entry must be >= 32 characters")
        if self.ENVIRONMENT is Environment.PROD and self.SSRF_ALLOW_PRIVATE_NETWORKS:
            raise ValueError("SSRF_ALLOW_PRIVATE_NETWORKS must be false in prod")
        return self

    @property
    def mongo_url(self) -> str:
        """Resolved Mongo connection string — explicit MONGO_URI wins."""
        if self.MONGO_URI:
            return self.MONGO_URI.get_secret_value()
        auth = ""
        if self.MONGO_USER and self.MONGO_PASSWORD:
            password = quote(self.MONGO_PASSWORD.get_secret_value(), safe="")
            auth = f"{quote(self.MONGO_USER, safe='')}:{password}@"
        return f"mongodb://{auth}{self.MONGO_HOST}:{self.MONGO_PORT}"

    def _redis_url(self, db: int) -> str:
        auth = ""
        if self.REDIS_PASSWORD:
            auth = f":{quote(self.REDIS_PASSWORD.get_secret_value(), safe='')}@"
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{db}"

    @property
    def redis_url(self) -> str:
        return self._redis_url(self.REDIS_DB)

    @property
    def celery_broker_url(self) -> str:
        return self._redis_url(self.REDIS_CELERY_BROKER_DB)

    @property
    def celery_result_backend(self) -> str:
        return self._redis_url(self.REDIS_CELERY_RESULT_DB)

    @property
    def qdrant_url(self) -> str:
        scheme = "https" if self.QDRANT_HTTPS else "http"
        return f"{scheme}://{self.QDRANT_HOST}:{self.QDRANT_HTTP_PORT}"


settings = Settings()
