# rag_engine: 100-Step Build Plan

## Context
The repo was reset to empty (`d5c85d6 step 0: everything deleted`). We're rebuilding `rag_engine` from scratch: a multi-tenant service that takes a website URL, crawls the whole site (HTML plus office documents), turns the content into structure-aware chunks, embeds them with OpenAI, and stores them in a vector store behind a `BaseVectorStore` factory (Qdrant first). It also exposes a search API.

The build runs over **100 prompts, one focused step per prompt**. Each step is built and tested before moving on. We don't take "MVP for now" shortcuts: every step builds what the service actually needs.

**Approval of this plan → Step 1 writes this document to `/plan.md` (repo root)** with an empty `## Progress Log` section. Every later prompt appends an entry to that log.

---

## Decisions (from Q&A, all user-confirmed)
| Area | Decision |
|---|---|
| Language | Python 3.12, uv + hatchling, package `rag_engine`, house style (pydantic-settings singleton, loguru, ruff + ty, `[dependency-groups]`, `uv run dev/server`, no Makefile) |
| Interface | FastAPI REST API (Swagger `/docs`) + async jobs |
| Queue | Celery + Redis (queues: `crawl`, `maintenance`, `webhooks`; Celery beat for periodic tasks) |
| Crawl scope | Whole site: seed URL + sitemap, per-job `max_pages`, `max_depth`, include/exclude globs |
| JS sites | httpx first; if the page looks like a JS shell, re-render it with Playwright Chromium |
| Content | HTML → Markdown (tables and code preserved), PDF (pymupdf4llm, AGPL accepted), DOCX, PPTX, XLSX; `BaseExtractor` registry keyed by MIME type so new formats plug in |
| Raw storage | MongoDB (async PyMongo): tenants, keys, KBs, sources, jobs, pages, chunks, embedding cache |
| Vector store | `BaseVectorStore` ABC + factory/dispatcher (StrEnum); Qdrant impl with named dense vector + BM25 sparse vector (IDF modifier) |
| Embeddings | `BaseEmbedder` factory; OpenAI impl. **Model chosen per tenant (default) and locked onto a KB at creation** |
| Sparse | fastembed `Qdrant/bm25` |
| Reranker | `BaseReranker` factory; local cross-encoder via fastembed `TextCrossEncoder` |
| Search | Modes `dense` / `hybrid` (RRF), optional rerank in either mode |
| Re-crawl | Full replace **per source**, **atomic swap**: ingest under a new `version` (= job id); on success flip `Source.active_version`; search filters on active versions; old points deleted afterwards. A failed or cancelled job leaves the old data serving |
| Tenancy | Tenant → Knowledge Base → Sources. API keys (`rk_<prefix>_<secret>`, stored hashed) resolve the tenant; master admin key (env) manages tenants and keys via `/admin/*`. Every query is filtered by `tenant_id` |
| Crawl safety | robots.txt + crawl-delay, per-domain rate limit (Redis, shared across workers), SSRF guard (DNS-pinned, re-checked on every redirect and on Playwright requests), hard caps (pages, bytes, time), retry/failure policy, URL normalization + dedupe, content-type allowlist |
| Jobs | Cancel, HMAC-signed completion webhook, live progress counters, per-page error report |
| Observability | Structured JSON loguru logs with request_id/job_id; Prometheus `/metrics` |
| Deploy | Docker Compose (api, crawl worker w/ Chromium, maintenance worker, beat, mongo, redis, qdrant) |
| Tests | Unit (respx, fakes) + integration (testcontainers Mongo/Redis/Qdrant, local fixture website). OpenAI mocked except `@pytest.mark.live`. Classes named `Test<Feature>_S<step>` |

## Architecture (target layout)
```
rag_engine/
  __init__.py          # dev()/server() entrypoints
  config/              # settings.py, logging.py
  exceptions.py        # AppBaseException + subclasses
  api/                 # app.py, middlewares/, exception_handlers/, deps/, routers/{admin,knowledge_bases,crawls,jobs,search,health}
  domain/              # pydantic models + enums: tenant, api_key, knowledge_base, source, job, page, chunk
  repositories/        # Mongo repositories per aggregate
  infrastructure/      # mongo.py, redis.py, lifecycle.py
  security/            # api_keys.py, ssrf.py
  crawling/            # url.py, scope.py, robots.py, rate_limit.py, fetchers/{http,browser,auto}.py, sitemap.py, links.py, frontier.py, engine.py
  extraction/          # base.py, registry.py, html.py, pdf.py, docx.py, pptx.py, xlsx.py, cleaning.py
  chunking/            # base.py, factory.py, markdown_parser.py, structure_chunker.py
  embeddings/          # base.py, dispatcher.py, models.py, openai.py, sparse.py, cache.py
  vectorstores/        # base.py, dispatcher.py, qdrant.py, exceptions.py
  reranking/           # base.py, dispatcher.py, cross_encoder.py
  ingestion/           # pipeline.py, finalize.py
  search/              # service.py
  worker/              # celery_app.py, tasks/{crawl,maintenance,webhooks}.py, signals.py
  webhooks/            # signer.py, delivery.py
  observability/       # metrics.py
tests/ (unit/, integration/, fixtures/site/, conftest.py, README.md)
docker/  docker-compose.yml  Dockerfile  .env.example  plan.md  README.md
```
Every swappable concern follows the house dispatcher pattern: `base.py` (ABC) + `dispatcher.py` (StrEnum → impl, typed error on unknown) + impls + `exceptions.py`.

## Per-prompt workflow (each of the 100 steps)
1. Read `plan.md` (decisions + last Progress Log entry) and confirm which step is next.
2. Build only that step's scope. Reuse earlier modules and don't duplicate them.
3. Write tests (`Test<Feature>_S<NN>`) and run `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest -q` (plus integration markers when relevant).
4. Append a Progress Log entry to `plan.md`:
   ```
   ### Step NN: <title>  (YYYY-MM-DD)
   - Status: ✅ done | ⚠️ partial (spills into next prompt) | ❌ blocked
   - Built: <files/modules>
   - Tests: <what's covered>; checks: ruff ✅ ty ✅ pytest ✅ N passed
   - Decisions/notes: <anything a future step must know>
   - Next: Step NN+1
   ```
5. Hand back. **No commit or push**; the user reviews and commits.

If a step spills over, the next prompt finishes it and logs it. Later steps then shift, and the tail steps (97–100) absorb the slack.

---

## The 100 Steps

### Phase A: Foundation (1–12)
1. **plan.md + repo hygiene**: write this plan to `/plan.md` with an empty Progress Log, plus `.gitignore`, `.python-version` (3.12), README stub, `tests/README.md` (test conventions).
2. **uv scaffold**: `pyproject.toml` (hatchling, `rag_engine` package, ruff/ty/pytest config, dev group), `rag_engine/__init__.py` with `dev()`/`server()`, `uv.lock`. Test: package imports, scripts are registered.
3. **Settings**: `config/settings.py` singleton with sections for app, mongo, redis, qdrant, openai, crawler limits, security, celery; secrets have no default; `.env.example`. Tests: fail-fast on missing secrets, env parsing, computed URLs.
4. **Logging**: loguru JSON sink, stdlib/uvicorn/celery intercept, contextvars binding for `request_id`/`job_id`/`tenant_id`. Tests: JSON shape, context propagation.
5. **Exceptions**: `AppBaseException(message, status_code, details)` + base subclasses (NotFound, Conflict, Unauthorized, Forbidden, Validation, UpstreamError); error response schema. Tests.
6. **FastAPI app**: app factory, lifespan, `/api/v1` router aggregator, `GET /health`, exception handlers, request-id + process-time middleware, CORS. Tests with httpx ASGI client.
7. **Dev compose**: `docker-compose.yml` with mongo, redis, qdrant (healthchecks, volumes). Verify all three are healthy.
8. **Test infrastructure**: pytest-asyncio, markers (`unit`, `integration`, `browser`, `live`), testcontainers fixtures for Mongo/Redis/Qdrant (session-scoped), settings override fixture.
9. **Fixture website**: `tests/fixtures/site/` served by a local server fixture: nested pages, robots.txt, sitemap index, PDF/DOCX/PPTX/XLSX, JS-only SPA page, redirects (including a loop), 404/500, oversized file, duplicate-content URLs, tracking-param links, off-site links.
10. **Mongo infrastructure**: async PyMongo client lifecycle, `MongoDocument` base (pydantic ↔ BSON, `_id`, timestamps), `BaseRepository` (CRUD, pagination helpers), index registration at startup. Integration tests.
11. **Redis infrastructure**: async redis client lifecycle + helper for namespaced keys. Tests.
12. **Readiness**: `GET /health/ready` pings Mongo/Redis/Qdrant with timeouts; returns per-dependency status, 503 if any is down. Tests.

### Phase B: Domain & Tenancy (13–22)
13. **Domain models & enums**: Tenant, ApiKey, KnowledgeBase, Source, CrawlJob, Page, Chunk; `JobStatus`, `PageStatus`, `ContentKind`, `EmbeddingModelName`. Validation tests.
14. **Embedding model registry**: `EmbeddingModelSpec` (provider, name, dims, max_input_tokens, tokenizer encoding) for OpenAI 3-small/3-large (+ optional dimensions). Tests.
15. **API key service**: generate `rk_<prefix>_<secret>`, HMAC-SHA256 with server pepper, constant-time verify. Unit tests.
16. **Tenant & ApiKey repositories**: unique indexes, lookup by key prefix, revoke. Integration tests.
17. **Auth dependencies**: `require_admin` (master key) and `require_tenant` (API key → Tenant, disabled tenants rejected, `last_used_at` updated). Tests for 401/403.
18. **Admin API**: `POST/GET/PATCH /admin/tenants` (default embedding model, quotas: max concurrent jobs, max pages per job, max KBs), `POST /admin/tenants/{id}/keys` (plaintext shown once), `DELETE .../keys/{key_id}`. Tests.
19. **KnowledgeBase repo + service**: create (locks the embedding model from the tenant default or an allowed explicit choice), list, get, quota enforcement. Tests.
20. **KB API**: `POST/GET /knowledge-bases`, `GET/DELETE /knowledge-bases/{id}` (delete marks the KB deleting and enqueues cascade later). Tenant-isolation tests (A can't see B).
21. **Source repo + API**: source identity = normalized seed URL within a KB; `active_version`, `last_job_id`, stats; `GET /knowledge-bases/{id}/sources`, `GET/DELETE .../sources/{sid}`. Tests.
22. **Per-tenant API rate limiting**: Redis sliding window dependency with 429 + `Retry-After`. Tests.

### Phase C: URL & Network Safety (23–32)
23. **URL normalization**: lowercase scheme and host, IDNA, default ports, dot-segments, fragment drop, sorted query, tracking-param strip, trailing-slash policy, `url_hash`. Hypothesis property tests.
24. **Scope rules**: same-host / allowed-subdomain policy, path-prefix, include/exclude globs, max depth, skip non-http(s) schemes and asset extensions. Tests.
25. **SSRF guard**: scheme/port allowlist; resolve all A/AAAA records; reject private, loopback, link-local, CGNAT, metadata, multicast, reserved, IPv4-mapped IPv6. Tests matrix.
26. **SSRF-safe HTTP client**: httpx transport that pins the validated IP (SNI/Host preserved), validates every redirect hop, enforces max redirects. Tests with local server + DNS stubs.
27. **HTTP fetcher**: streaming download with max-bytes cutoff, connect/read/total timeouts, content-type allowlist, charset detection, `FetchResult` model, configurable User-Agent. Tests.
28. **Retry & failure policy**: error classifier (retryable: timeouts, 5xx, 429 + Retry-After; permanent: 4xx, SSRF-blocked, too-large, disallowed type), exponential backoff + jitter, per-URL attempt limit. Tests.
29. **robots.txt**: fetch via the safe client, parse (protego), Redis cache per host with TTL, missing/5xx semantics, crawl-delay exposed. Tests.
30. **Per-domain rate limiter**: Redis token bucket shared across workers; takes the larger of configured RPS and crawl-delay; async acquire. Tests incl. concurrency.
31. **Sitemap discovery**: robots `Sitemap:` + `/sitemap.xml`, sitemap index recursion, gzip, size/count caps, scope filtering. Tests on the fixture site.
32. **Link extraction**: `<a href>`, `<base href>`, canonical link, `rel=nofollow` policy, document links (pdf/docx/…), normalization + scope applied. Tests.

### Phase D: JS Rendering (33–37)
33. **JS-shell detector**: heuristics (visible text length, empty root mount nodes, `<noscript>` hints, script/text ratio). Tests on fixture HTML.
34. **Playwright browser pool**: one browser per worker process (Celery signals), context per job, blocked resource types (image/font/media), page timeout. `browser` marker tests.
35. **Browser SSRF enforcement**: `context.route` validates every request host through the SSRF guard; downloads, websockets, and service workers blocked. Tests.
36. **Auto fetcher**: httpx → detector → Playwright re-render (networkidle with cap, auto-scroll for lazy content); records `rendered_with` on the page. Tests.
37. **Chromium in the worker image**: worker Dockerfile stage with `playwright install --with-deps chromium`; container smoke test renders the fixture SPA.

### Phase E: Extraction (38–46)
38. **Extractor framework**: `BaseExtractor`, `ExtractorRegistry` (MIME + extension + magic-byte sniffing), `ExtractedDocument` (markdown, title, lang, metadata, outlinks); unknown types → `UnsupportedContent` page status. Tests.
39. **HTML extractor**: trafilatura main content → Markdown with tables/code/headings preserved; markdownify fallback when the result is empty; title/description/lang metadata. Fixture tests.
40. **Cross-page boilerplate removal**: per-job block-frequency index (hashed blocks seen on more than N% of pages, e.g. nav/footer/cookie banners) stripped from Markdown. Tests.
41. **PDF extractor**: pymupdf4llm → Markdown (page markers, tables), page-count/size caps, encrypted/no-text-layer handling recorded as failure reason. Tests.
42. **DOCX extractor**: python-docx → Markdown (heading styles, lists, tables). Tests.
43. **PPTX extractor**: python-pptx → a section per slide (title, body, tables, speaker notes). Tests.
44. **XLSX extractor**: openpyxl read-only → Markdown table per sheet with row/col caps. Tests.
45. **Text cleaning**: NFKC, zero-width/control chars, whitespace collapse (outside code), empty-section pruning, min-content threshold. Tests.
46. **Content dedupe**: content hash per job; duplicates (same content under different URLs) recorded as `duplicate_of`, canonical URLs respected. Tests.

### Phase F: Crawl Engine (47–54)
47. **Page repository**: Page docs keyed `(job_id, url_hash)` unique; status, depth, http status, content kind, failure reason, timings, `rendered_with`. Tests.
48. **Frontier**: Mongo-backed queue with claim via `find_one_and_update` (sitemap/shallow-first priority), dedupe through the unique index, resumable after a crash. Tests.
49. **CrawlJob repository + state machine**: `queued → running → finalizing → succeeded | failed | cancelled`, guarded transitions, atomic `$inc` counters (discovered/fetched/failed/skipped/chunked/embedded), heartbeat field. Tests.
50. **Crawl engine core**: async loop with bounded concurrency; per URL: scope → robots → rate limit → fetch (auto) → extract → clean → dedupe → persist page → enqueue outlinks. Integration test crawls the fixture site.
51. **Caps & timeouts**: max pages, max total bytes, max job wall-time, per-page timeouts; on cap hit the job stops gracefully with a reason. Tests.
52. **Cancellation**: Redis cancel flag polled by the engine; in-flight pages drain; status `cancelled`. Tests.
53. **Engine error accounting**: every failure ends as a page row with a classified reason; job summary of failures by reason. Tests.
54. **Crawl engine E2E**: full fixture-site crawl asserting every page's expected status (SPA rendered, loop blocked, 404 failed, PDF extracted, off-site skipped, robots-disallowed skipped).

### Phase G: Chunking (55–60)
55. **Chunker framework**: `BaseChunker` + factory, `Chunk` model (text, embed_text, heading_path, ordinal, token_count, page_id, url, content hash). Tests.
56. **Markdown structure parser**: markdown-it-py AST → sections by heading hierarchy, atomic blocks (table, code fence, list). Tests.
57. **Token-aware splitter**: tiktoken (per-model encoding), target/max tokens + overlap, recursive (section → paragraph → sentence). Atomic blocks are never split unless oversized; oversized tables split by rows with the header repeated, code split by lines. Tests.
58. **Contextual embed text**: `embed_text` = page title + heading path + chunk; `text` kept clean for display. Tests.
59. **Chunk persistence**: chunks stored in Mongo per `(source_id, version)` for audit and re-embedding without re-crawl. Integration tests.
60. **Chunking golden tests**: snapshot tests over every fixture document type.

### Phase H: Embeddings (61–66)
61. **Embedder framework**: `BaseEmbedder` (async `embed_documents`, `embed_query`), dispatcher on provider StrEnum, typed errors. Tests.
62. **OpenAI embedder**: async client, batching by input count and token budget, `dimensions` support, order-preserving results. respx tests.
63. **OpenAI resilience**: 429/5xx/timeouts retried with backoff honoring rate-limit headers, a concurrency semaphore, permanent errors surfaced. Tests.
64. **Embedding cache**: Mongo collection keyed `sha256(model|dims|embed_text)` so a full replace of unchanged content doesn't re-pay. Tests.
65. **Usage metering**: tokens/requests recorded per job and aggregated per tenant; shown in job detail. Tests.
66. **Sparse embedder**: `BaseSparseEmbedder` + fastembed BM25 impl (documents and queries), model cached in the image. Tests.

### Phase I: Vector Store (67–74)
67. **Vector store framework**: `BaseVectorStore` ABC (ensure_collection, upsert, delete_by_filter, count, search_dense, search_hybrid), `VectorStoreDispatcher` (StrEnum `QDRANT`), `VectorPoint`/`SearchHit`/`VectorFilter` models, exceptions. Tests with a fake impl.
68. **Qdrant collection management**: one collection per embedding spec (`chunks__<model>__<dims>`), named dense vector (cosine) + `bm25` sparse with IDF modifier, payload indexes (`tenant_id` is_tenant, `kb_id`, `source_id`, `version`, `url`). Integration tests.
69. **Qdrant upsert**: deterministic UUID5 point IDs (version + chunk id), batching, `wait=True`, retries. Tests.
70. **Qdrant delete & count**: by version/source/KB/tenant filters. Tests.
71. **Qdrant dense search**: tenant/KB/active-version/source/url-prefix filters, score threshold. Tests.
72. **Qdrant hybrid search**: Query API with dense + sparse prefetch fused by RRF. Tests.
73. **Version swap semantics**: `Source.active_version` flip in Mongo; search filter uses `version ∈ active_versions(kb)`; a concurrent-ingest test shows search keeps serving the old version until the flip.
74. **Vector store bootstrap**: ensure collections at startup for all specs in use; lifespan wiring + readiness integration. Tests.

### Phase J: Ingestion & Workers (75–83)
75. **Celery app**: Redis broker/backend, `acks_late`, prefetch 1, queues/routes (`crawl`, `maintenance`, `webhooks`), JSON serializer, worker-process init signals (Mongo/Redis/Qdrant/browser clients). Tests with a real worker fixture.
76. **Ingestion pipeline**: bounded async queue from engine → chunk → batch embed (dense + sparse) → upsert under the job's version; counters updated. Tests with a fake embedder.
77. **`run_crawl_job` task**: loads job/KB/source, runs engine + pipeline, state transitions, heartbeat, error capture → `failed` with reason. Integration test.
78. **Finalization**: success → atomic `active_version` flip + enqueue old-version cleanup; failure/cancel → delete the new version's points and keep the old version active. Tests.
79. **Concurrency guards**: one active job per source (Mongo partial unique index), tenant max concurrent jobs, a duplicate submission returns the existing job. Tests.
80. **Crash recovery**: beat task finds `running` jobs with a stale heartbeat → requeue (frontier resumes) up to N times, then `failed`. Tests.
81. **Cascade deletes**: source delete and KB delete tasks remove Mongo pages/chunks and Qdrant points, then the records. Tests.
82. **Webhook signer**: HMAC-SHA256 over timestamp + body, `X-RagEngine-Signature` header, verification helper for receivers. Tests.
83. **Webhook delivery**: `webhooks` queue task through the SSRF-safe client, retries with backoff, delivery log on the job. Tests.

### Phase K: Crawl & Job API (84–88)
84. **Start crawl**: `POST /knowledge-bases/{kb_id}/crawls` (url, max_pages, max_depth, include/exclude, webhook_url); validated against tenant quotas + SSRF precheck; upserts Source; enqueues; returns 202 + job. Tests.
85. **Job read API**: `GET /jobs/{id}` (status, live counters, usage, failure summary), `GET /jobs` (cursor pagination, filters by KB/source/status). Tests.
86. **Per-page report**: `GET /jobs/{id}/pages` (status/reason filters, pagination). Tests.
87. **Cancel**: `DELETE /jobs/{id}` → cancel flag + status; idempotent; old version stays active. Tests.
88. **API E2E**: admin creates tenant → key → KB → crawl fixture site → real Celery worker → job succeeded → sources/pages consistent (fake embedder). Re-crawl → atomic swap verified.

### Phase L: Search (89–93)
89. **Reranker framework + cross-encoder**: `BaseReranker` + dispatcher; fastembed `TextCrossEncoder` impl (model baked into the image, lazy singleton). Tests.
90. **Search service**: query embedding with the KB's locked model (dense + sparse as needed), active-version filter, dense/hybrid modes, optional rerank over `candidate_k → top_k`. Tests.
91. **Search API**: `POST /knowledge-bases/{kb_id}/search` (query, mode, rerank, top_k, source_ids, url_prefix, score_threshold) → hits with text, heading path, url, title, source, scores. Tests incl. tenant isolation.
92. **Query embedding cache**: Redis cache of query vectors per model with TTL; search latency logged. Tests.
93. **Retrieval eval harness**: labeled Q→expected-URL set over the fixture site; recall@k / MRR for dense vs hybrid vs +rerank; `eval` marker with a baseline threshold.

### Phase M: Observability & Ops (94–97)
94. **Prometheus metrics**: API request metrics; worker metrics (pages by status, fetch latency, render count, embed tokens, upsert latency, job durations) with multiprocess-safe collection; `/metrics`. Tests.
95. **Production images**: multi-stage Dockerfile → `api` image and `worker` image (Chromium, fastembed models baked), non-root, `CMD ["server"]` / celery. Build verified.
96. **Full compose**: api, worker-crawl, worker-maintenance (+webhooks), beat, mongo, redis, qdrant; healthchecks, restart policies, resource limits; smoke script runs the E2E flow against compose.
97. **Retention & sweepers**: TTL on old job page docs, orphan-point sweeper (points whose version is neither active nor in-flight), old embedding-cache pruning. Tests.

### Phase N: Hardening & Release (98–100)
98. **Security & failure-injection pass**: SSRF matrix re-run end-to-end, request size limits, pip-audit; Qdrant down mid-job, OpenAI 429 storm, worker kill → resume. Fix findings.
99. **Load test + tuning + docs**: crawl a generated 1k-page fixture site, tune concurrency/batch sizes, record numbers; README (architecture, API walkthrough, ops runbook), complete `.env.example`.
100. **Acceptance run**: live run (`live` marker) on a real public website with real OpenAI → search quality spot-check, full check suite green, final Progress Log summary + known follow-ups.

---

## Verification (applies to every step)
- `uv run ruff check . && uv run ruff format --check . && uv run ty check`
- `uv run pytest -q -m "not integration and not browser and not live"` (fast path)
- `uv run pytest -q -m integration` (testcontainers / compose services) when the step touches Mongo/Redis/Qdrant/Celery
- `uv run pytest -q -m browser` when the step touches Playwright
- System checkpoints: Step 54 (crawl E2E), Step 88 (API E2E + atomic swap), Step 96 (compose smoke), Step 100 (live acceptance)

## Progress Log
### Step 1: plan.md + repo hygiene  (2026-10-16)
- Status: ✅ done
- Built: `plan.md` (decisions, 100 steps, this log), `.gitignore` (standard Python + project entries), `.python-version` (3.12), `README.md` stub, `tests/README.md` (test conventions: markers, `Test<Feature>_S<step>` naming, AAA layout).
- Tests: none. This step has no Python package; the first tests arrive in Step 2.
- Decisions/notes: all stack decisions are recorded in the Decisions table above. pymupdf4llm (AGPL) is accepted for PDFs. Test class suffix is the plan step number (`_S<NN>`), not GitHub issue numbers.
- Next: Step 2: uv scaffold

### Step 2: uv scaffold  (2026-10-16)
- Status: ✅ done
- Built: `pyproject.toml` (hatchling, `rag_engine` package, `dev`/`server` scripts, ruff with E/W/F/I/B/UP/SIM/ASYNC/RUF, ty `[tool.ty.src]` over `rag_engine` + `tests`, pytest `--strict-markers --strict-config`), `uv.lock`, `rag_engine/__init__.py` (`dev()` reload / `server()` → `uvicorn.run("rag_engine.api.app:app")`), `tests/unit/test_scaffold.py`.
- Deps: runtime `uvicorn[standard]`, `loguru`. Dev group: `pytest`, `pytest-asyncio`, `ruff`, `ty`. Python pinned `>=3.12,<3.13`. Other deps get added by the step that needs them.
- Tests: `TestUvScaffold_S2`: import, console scripts registered, dev/server call uvicorn with the right args. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 5 passed.
- Decisions/notes: host/port are module constants in `rag_engine/__init__.py`, and Step 3 must swap them for `settings.DEV_*`/`PROD_*`. `rag_engine.api.app:app` doesn't exist until Step 6, so `uv run dev` won't serve until then. The old ty keys `root`/`python` were dropped (ty 0.0.89 uses `[tool.ty.src] include` and finds `.venv` itself).
- Next: Step 3: Settings

### Step 3: Settings  (2026-10-16)
- Status: ✅ done
- Built: `rag_engine/config/settings.py` (one flat uppercase `Settings` singleton, grouped app / security / mongo / redis / celery / vector store / embeddings / reranker / crawler), `rag_engine/config/enums.py` (`Environment`, `VectorStoreType`, `EmbeddingProvider`, `RerankerType`), `rag_engine/config/__init__.py` re-exports, `.env.example`, `tests/conftest.py` (injects test secrets before import). `rag_engine/__init__.py` now reads `settings.DEV_*`/`PROD_*`.
- Required secrets (no default): `ADMIN_API_KEY` (≥32), `API_KEY_PEPPER` (≥32), `OPENAI_API_KEY`, all `SecretStr`.
- Computed: `mongo_url` (`MONGO_URI` wins, credentials URL-escaped), `redis_url`, `celery_broker_url`, `celery_result_backend` (separate Redis DBs 0/1/2), `qdrant_url`.
- Validation: default ≤ hard max for pages/depth, page bytes ≤ total bytes, Celery soft < hard time limit, `SSRF_ALLOW_PRIVATE_NETWORKS` forbidden when `ENVIRONMENT=prod`, port ranges, positive limits.
- Tests: `TestSettings_S3`: missing/short secrets, masked repr, defaults, env parsing (JSON lists, enums), empty-env fallback, unknown model rejected, inconsistent limits, prod SSRF guard, URL builders. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 26 passed.
- Decisions/notes: `DEFAULT_EMBEDDING_MODEL` is a `Literal` for now, and Step 14's model registry should become its source of truth. Backend selector enums live in `config/enums.py`, and the later dispatchers import them from there. ty 0.0.89 does not flag missing required fields on `Settings()`, so no `# type: ignore` is needed. `.env` comments must sit on their own lines. Tests build `Settings(_env_file=None, ...)` so a developer's local `.env` never leaks in.
- Next: Step 4: Logging

### Step 4: Logging  (2026-10-16)
- Status: ✅ done
- Built: `rag_engine/config/logging.py`:
  - `setup_logging(level, json_logs, stream)` is idempotent. JSON sink fields: timestamp, level, message, logger, function, line, request_id, job_id, tenant_id, extra, exception.
  - Human format is used when `LOG_JSON=false`.
  - `InterceptHandler` routes stdlib logging into loguru and keeps the original stdlib logger name in the `logger` field. uvicorn, fastapi, celery, kombu, httpx and pymongo loggers are reset to propagate.
  - Context IDs live in contextvars, bound with `log_context(request_id=…, job_id=…, tenant_id=…)` and read with `get_log_context()`. A loguru patcher fills them in, and an explicit `logger.bind` takes precedence.
  - Re-exported from `rag_engine.config`.
  - `dev()`/`server()` call `setup_logging()` and pass `log_config=None` to uvicorn.
- Tests: `TestLogging_S4`: JSON shape, context set and cleared, nesting restore, isolation across concurrent asyncio tasks, bind precedence, exception serialization, stdlib interception (with context), level filtering, human format, unknown-field guard. Scaffold tests now stub `setup_logging`. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 36 passed.
- Decisions/notes:
  - `diagnose` (variable values in tracebacks) is always off for JSON logs and on for human logs outside prod only, so secrets don't leak.
  - With `reload=True`, uvicorn serves from a child process, so Step 6's app factory/lifespan must also call `setup_logging()`.
  - Step 75 must set Celery `worker_hijack_root_logger=False` and call `setup_logging()` in the worker init signal.
  - Use `traceback.format_exception`; ty rejects `Formatter.formatException` with loguru's optional exception value.
- Next: Step 5: Exceptions

### Step 5: Exceptions  (2026-10-16)
- Status: ✅ done
- Built: `rag_engine/exceptions.py`:
  - `AppBaseException(message, status_code, details, headers)`. Class-level `status_code` / `code` / `default_message` defaults mean subclasses need no `__init__` boilerplate.
  - Subclasses: `BadRequestError` 400, `UnauthorizedError` 401 (adds `WWW-Authenticate: Bearer`), `ForbiddenError` 403, `QuotaExceededError` 403, `NotFoundError` 404, `ConflictError` 409, `AppValidationError` 422, `TooManyRequestsError` 429 (`retry_after` → `Retry-After` header + details), `UpstreamError` 502, `ServiceUnavailableError` 503.
  - Error contract: `ErrorResponse{error: ErrorBody{code, message, details, request_id}}`, produced by `exc.to_response(request_id)`.
- Tests: `TestExceptions_S5`: defaults for all 11 types, overrides without touching class defaults, details copied (no shared mutable state), feature-subclass pattern, auth/rate-limit headers, `to_response` contract round-trip, repr, catch-as-base. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 56 passed.
- Decisions/notes:
  - `code` is the stable client-facing identifier. Feature packages subclass a base type and override `code`/`default_message`, e.g. `knowledge_base_not_found(NotFoundError)`.
  - Named `AppValidationError` to avoid clashing with pydantic's `ValidationError`.
  - `UpstreamError` (502) is for external services we call (OpenAI, target sites); `ServiceUnavailableError` (503) is for our own infrastructure.
  - Step 6's handlers must render `exc.to_response(get_log_context()["request_id"])` with `exc.headers`, map FastAPI `RequestValidationError` and unhandled exceptions into the same `ErrorResponse` shape, and document `ErrorResponse` in OpenAPI.
- Next: Step 6: FastAPI app

### Step 6: FastAPI app  (2026-10-16)
- Status: ✅ done
- Built:
  - `rag_engine/api/app.py`: `create_app()` factory, module-level `app`, lifespan hook, OpenAPI at `/api/v1/openapi.json`.
  - `api/middlewares/request_id.py` (pure ASGI): honors a safe incoming `X-Request-ID` (`[A-Za-z0-9._-]{1,128}`), else generates uuid4 hex; binds `log_context(request_id)`, stores it on `scope["state"]`, echoes the header.
  - `api/middlewares/process_time.py` (pure ASGI): `X-Process-Time` header plus one access-log line per request (method, path, status_code, duration_ms).
  - `api/exception_handlers/handlers.py`: `AppBaseException`, `RequestValidationError` (strips `input` so secrets aren't echoed), Starlette `HTTPException` (404/405…), and unhandled `Exception` (generic 500, traceback only in logs). All render `ErrorResponse`, and the request ID header is added even on 500s.
  - `api/routers/api.py`: `api_router` aggregator documenting 422/500 `ErrorResponse`.
  - `api/routers/health.py`: `GET /api/v1/health` → `{"message": "app is healthy!!!"}`.
  - CORS is only installed when `BACKEND_CORS_ORIGINS` is set, and exposes `X-Request-ID`, `X-Process-Time` and `Retry-After`.
  - `rag_engine.__version__` added.
- Deps: runtime `fastapi` (0.143, Starlette 1.8). Dev: `httpx`, `asgi-lifespan`. pytest `asyncio_mode = "auto"`.
- Tests: `TestFastApiApp_S6` (16): health, generated/echoed/sanitized request IDs, timing header, access log carries request_id, app-error shape and headers (Retry-After), validation error hides input, 404/405 shape, 500 doesn't leak and is logged, OpenAPI path + `ErrorResponse` schema, lifespan logs, CORS allow/deny. Plus `test_uvicorn_access_log_is_silenced` in `TestLogging_S4`. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 73 passed.
- Smoke: `uv run server`, then curl `/api/v1/health` → 200 with the headers, and `/nope` → 404 in `ErrorResponse` shape. Logs are JSON with one access line per request.
- Decisions/notes:
  - Middleware order (outermost first): Starlette ServerError → RequestId → ProcessTime → CORS → app. Unhandled 500s are rendered by ServerErrorMiddleware outside our middleware, so handlers read `request.state.request_id` and set the header themselves.
  - `uvicorn.access` is silenced in `setup_logging` (`create_app()` re-runs it, which used to re-enable the logger after uvicorn's `access_log=False`). `dev()`/`server()` pass `access_log=False`.
  - Feature routers get included in `api/routers/api.py`. Lifespan is where Steps 10–12 attach Mongo/Redis/Qdrant clients to `app.state`.
- Next: Step 7: Dev compose

### Step 7: Dev compose  (2026-10-16)
- Status: ✅ done
- Built: `docker-compose.yml` (project `rag_engine`):
  - `mongo:8.0` with a root user from `MONGO_USER`/`MONGO_PASSWORD`, `mongosh ping` healthcheck.
  - `redis:8.2` with `--requirepass`, AOF on, `redis-cli ping` healthcheck via `REDISCLI_AUTH`.
  - `qdrant/qdrant:v1.19.2-unprivileged` (non-root) with an API key and telemetry off; the healthcheck hits `/readyz` over bash `/dev/tcp` because the image has no curl.
  - Named volumes. Every port binds to `127.0.0.1` only. Credentials are required (`${VAR:?}`) from `.env`.
- Settings/env: default `MONGO_PORT` changed to **27020** and `REDIS_PORT` to **6380**, matching the compose host ports (this machine's 27017/6379 are taken by other stacks). Compose reads the same `.env` variables as `Settings`, so the ports have one source of truth. `.env.example` now lists `MONGO_USER`, `MONGO_PASSWORD`, `REDIS_PASSWORD` and `QDRANT_API_KEY` as required by compose.
- Tests: `test_default_ports_match_compose_host_ports` added to `TestSettings_S3`. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 74 passed.
- Verified: `docker compose up -d --wait` → all 3 `healthy`. On fresh volumes: Mongo unauthenticated write → `Unauthorized`, authenticated → ok; Redis without password → `NOAUTH`, with → `PONG`, `appendonly yes`; Qdrant without key → 401, with key → 200 (v1.19.2). Torn down with `down -v`.
- ⚠️ Found a stale volume `rag_engine_mongo_data` (created 2026-07-21 by the pre-reset project). Mongo only creates the root user on an empty data dir, so with that volume present the new credentials are never applied and auth fails. **Left untouched for the user to decide** (`docker volume rm rag_engine_mongo_data` before first `up`). Verification ran under a separate project name (`rag_engine_verify`) on fresh volumes.
- Decisions/notes: Step 10's `qdrant-client` should pin to the 1.19 line to match the server. Step 96 adds api/worker/beat services to this file; those containers must override `MONGO_PORT=27017`, `REDIS_PORT=6379` and the `*_HOST` values to the service names.
- Next: Step 8: Test infrastructure
