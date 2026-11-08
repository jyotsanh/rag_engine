# rag_engine: 100-Step Build Plan

## Context
The repo was reset to empty (`d5c85d6 step 0: everything deleted`). We're rebuilding `rag_engine` from scratch: a multi-tenant service that takes a website URL, crawls the whole site (HTML plus office documents), turns the content into structure-aware chunks, embeds them with OpenAI, and stores them in a vector store behind a `BaseVectorStore` factory (Qdrant first). It also exposes a search API.

The build runs over **100 prompts, one focused step per prompt**. Each step is built and tested before moving on. We don't take "MVP for now" shortcuts: every step builds what the service actually needs.

**Approval of this plan → Step 1 writes this document to `/plan.md` (repo root)** with an empty `## Progress Log` section. Every later prompt appends an entry to that log.

---

## Decisions (from Q&A, all user-confirmed)
| Area | Decision |
|---|---|
| Language | Python 3.14 (changed from 3.12 by the user on 2026-10-17), uv + hatchling, package `rag_engine`, house style (pydantic-settings singleton, loguru, ruff + ty, `[dependency-groups]`, `uv run dev/server`, no Makefile) |
| Interface | FastAPI REST API (Swagger `/docs`) + async jobs |
| ASGI server | uvicorn 0.54 with `loop="zuvloop"` (Zig libuv loop) and `http="zttp"` (Zig HTTP/1.1/2/3 parser), both by the uvicorn maintainer (Kludex), 0.0.x pre-release; user decision 2026-10-17 |
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

### Step 8: Test infrastructure  (2026-10-16)
- Status: ✅ done
- Built:
  - `tests/conftest.py`: test secrets injected before import; directory auto-markers (`tests/unit` → `unit`, `tests/integration` → `integration`); `--run-live` gate (live tests are skipped by default); `override_settings(**fields)` fixture that validates via a full `Settings(...)` build, monkeypatches the singleton and auto-restores; `pytester` plugin enabled.
  - `tests/support/services.py`: `MongoService` / `RedisService` / `QdrantService` connection dataclasses, plus image constants pinned to the same images as `docker-compose.yml`.
  - `tests/integration/conftest.py`: `docker_available` (skips when there's no daemon), session-scoped lazy testcontainers for Mongo (root auth), Redis (`requirepass`) and Qdrant (API key), and an `integration_settings` fixture (settings aimed at the containers, unique `MONGO_DB` per test, teardown drops the DB, flushes the app/broker/result Redis DBs and deletes all Qdrant collections).
  - `pyproject.toml`: registered markers `unit`, `integration`, `browser`, `live`, `eval`; filtered the expected Qdrant "api key over http" warning.
  - `tests/README.md`: run commands and fixture table.
- Deps (dev): `testcontainers[mongodb,redis,qdrant]` 4.15, which brings sync `pymongo` 4.19, `redis` 8.1, `qdrant-client` 1.19.1 and `docker` as test-side helpers (used only in fixtures and teardown for now).
- Tests: `TestTestInfrastructure_S8` (8, unit): auto marker, live skipped / enabled with `--run-live` (via pytester), override apply / restore / validate / coerce / unknown-field. `TestIntegrationServices_S8` (7, integration): marker, auth enforced on all 3 services (Qdrant reports 1.19.2), settings point at the containers, cross-test isolation (data written in one test is gone in the next). Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 89 passed (82 unit + 7 integration, ~15–20s with container start). No leftover containers.
- Decisions/notes:
  - Integration tests run sequentially; the isolation test relies on file order (no pytest-randomly or xdist).
  - Use the `testcontainers.community.*` imports; the top-level module paths are deprecated.
  - When runtime clients arrive (Steps 10/11/68), add `pymongo`, `redis` and `qdrant-client` as **runtime** deps (`qdrant-client` on the 1.19 line).
- Next: Step 9: Fixture website

### Step 9: Fixture website  (2026-10-16)
- Status: ✅ done
- Built:
  - `tests/fixtures/site/` (~240 KB): a fictional "Acme Widgets" site.
    - Pages: home, about/privacy/terms, docs (getting-started with code blocks, configuration with a table, api/reference → deep/level4 → level5 for depth limits), blog (2 posts), a sitemap-only orphan page.
    - Every page shares a header nav, cookie banner and footer (for Step 40 boilerplate removal).
    - Edge-case pages: dup-a/dup-b (same content), a canonical variant, a latin-1 page with the charset only in `<meta>`, a robots-disallowed `/private/`, a nofollow target, a JS-only SPA at `/app/`.
    - `robots.txt` (Disallow `/private/`, `Crawl-delay: 0.1`, Sitemap), `sitemap_index.xml` → `sitemap-pages.xml` + gzip `sitemap-docs.xml.gz` (includes an off-site URL).
    - Documents: `files/` widget-spec.pdf (2 pages), scanned.pdf (no text layer), encrypted.pdf, install-guide.docx (headings, bullets, table), roadmap.pptx (3 slides, notes, table), pricing.xlsx (2 sheets); `images/logo.png`; `data.json`.
  - `tests/fixtures/build_documents.py` regenerates the binaries (pymupdf, python-docx, python-pptx, openpyxl) with pinned metadata dates.
  - `tests/support/fixture_site.py`: a Starlette app run by uvicorn in a daemon thread on a free `127.0.0.1` port.
    - `{{BASE}}`/`{{OTHER_HOST}}` templating (`localhost:<port>` = same server, different host).
    - Dynamic routes: 301/302 redirects and a chain, a `/loop-a`↔`/loop-b` loop, a redirect to `169.254.169.254`, 500, `/flaky` (503 + Retry-After ×2 then 200), `/rate-limited` (429 always), `/slow?delay=` (default 5s), `/large.html` (25 MiB chunked, no Content-Length), `/files/large.bin` (25 MiB with Content-Length), the gzip sitemap, `/app/advanced` (linked only from JS-rendered content), dir → slash redirect, path-traversal guard.
    - Request log (`requests`, `hits(path)`, `reset()`).
    - `SITE_PAGES`: URL → `Expect` outcome (HTML, JS_RENDERED, DOCUMENT, DOCUMENT_UNREADABLE, REDIRECT, REDIRECT_LOOP, BLOCKED_SSRF, NOT_FOUND, SERVER_ERROR, RETRY_THEN_OK, RATE_LIMITED, TIMEOUT, TOO_LARGE, UNSUPPORTED_TYPE, DUPLICATE, CANONICAL_ELSEWHERE, ROBOTS_DISALLOWED, NOFOLLOW, SITEMAP_ONLY, OUT_OF_SCOPE), the oracle for Step 54.
  - Root `conftest.py`: session `fixture_site_server` plus per-test `fixture_site` (resets the log and flaky counter).
- Deps (dev): `python-docx`, `python-pptx`, `openpyxl`, `pymupdf`. Steps 41–44 will promote the needed ones to runtime (with `pymupdf4llm`).
- Tests: `TestFixtureSite_S9` (52): templating, manifest covers every home-page link, all HTML pages served, shared boilerplate, robots, sitemap index + gzip child, orphan not linked, all redirects, 404/500, flaky + reset, 429, timeout, streaming size without Content-Length, Content-Length on the binary, unsupported types, SPA empty without JS, latin-1 meta charset, duplicates, canonical, nofollow, other hostname, traversal, request log, and every document parses (PDF text / no text / encrypted, DOCX, PPTX, XLSX). Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 141 passed (~23s incl. containers).
- Decisions/notes:
  - The fixture-site tests live under `tests/unit` because they need loopback only (no Docker); the marker text now says so.
  - SPA content lives inside the `<script>`; a JS-shell detector must ignore script text.
  - `pymupdf.PDF_ENCRYPT_AES_256` exists only at runtime, hence the blanket ignore in the builder.
- Next: Step 10: Mongo infrastructure

### Step 10: Mongo infrastructure  (2026-10-17)
- Status: ✅ done
- Pre-step: the user switched to **Python 3.14** and uvicorn `loop="zuvloop"` / `http="zttp"` (uncommitted when this step started).
  - Checked that both packages are genuine: `zuvloop` wheels carry a PyPI trusted-publishing attestation from GitHub `Kludex/zuvloop`; `zttp` has a real `Kludex/zttp` repo with a publish workflow but no attestations; uvicorn 0.54 supports both natively. The user confirmed the change was intentional.
  - Follow-ups: scaffold tests now expect `loop`/`http`; ruff `target-version = "py314"` (dropped quoted annotations, since PEP 649 makes them lazy); Decisions table updated. Baseline suite green on 3.14.5.
- Built:
  - `rag_engine/domain/base.py`:
    - `MongoDocument` base with `collection_name` / `id_prefix` / `indexes` ClassVars.
    - IDs are `"<prefix>_<uuid7 hex>"`: time-ordered, so `_id` doubles as the pagination key.
    - `created_at` / `updated_at` are UTC, truncated to milliseconds so BSON round-trips compare equal.
    - `to_mongo()` / `from_mongo()`; `validate_assignment`; `extra="ignore"` for rolling deploys.
    - Concrete subclasses auto-register (`registered_documents()`); two models can't claim one collection.
  - `rag_engine/infrastructure/mongo.py`: `MongoConnection` (one `AsyncMongoClient` per process, tz-aware codec options, `connect()` pings and fails fast, `ping()`, `ensure_indexes()` (idempotent, all registered models), `close()`), plus type alias `MongoDatabase`.
  - `infrastructure/exceptions.py`: `DatabaseUnavailableError` (503).
  - `infrastructure/lifecycle.py`: `Infrastructure` protocol plus `AppInfrastructure` (start = connect + ensure indexes; Redis and Qdrant attach in Steps 11 and 74).
  - `rag_engine/repositories/base.py`: `BaseRepository[T]` with insert (`DuplicateKeyError` → `DuplicateDocumentError` 409), get / get_or_raise (per-repo `not_found_error`), find_one / find (sort, limit), count, update, delete, delete_many (refuses an empty filter), paginate.
    - `update` validates the merged document before `$set`, bumps `updated_at` and rejects `_id`/`created_at`.
    - Every method takes `scope=` (AND-ed filter); tenant repos pass `{"tenant_id": ...}`.
  - `repositories/pagination.py`: `Page[T]` (items, next_cursor); opaque base64url cursor `{after, order}` (wrong order or garbage → `InvalidCursorError` 422); limit 1–200; fetches limit+1 rows so there's never an empty trailing page.
  - `repositories/exceptions.py`.
  - `api/app.py`: `create_app(infrastructure_factory=AppInfrastructure)`. The lifespan starts the infrastructure before serving, stores it on `app.state.infrastructure` and stops it on shutdown. Startup fails if Mongo is down (verified: `uv run server` with Mongo unreachable exits code 3 after ~2s with "Application startup failed").
  - Settings: `MONGO_SERVER_SELECTION_TIMEOUT_MS`, `MONGO_CONNECT_TIMEOUT_MS`, `MONGO_MAX_POOL_SIZE` (+ `.env.example`).
  - `tests/support/fakes.py`: `NoopInfrastructure`; `test_app.py` uses it.
- Deps: runtime `pymongo` 4.19 (native asyncio API).
- Tests: `TestMongoDocument_S10` (17, unit): uuid7 prefixed ids, sort order, BSON mapping, round-trip ignoring unknown fields, ms UTC timestamps, assignment validation, registry and collision guard, cursor round-trip / malformed / wrong order, lifespan start/stop, lifespan aborts on failure. `TestMongoInfrastructure_S10` (17, integration): connect/ping/close, unreachable fails fast without leaving a client, idempotent indexes, CRUD round-trip, unique → 409, custom not-found error, tenant scope on get/count/update/delete, update validation and immutables, update into a unique conflict, find sort/limit, delete_many guard, pagination both orders across pages, exact multiple has no trailing empty page, bad limit/cursor, real app lifespan creates indexes. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 175 passed.
- Decisions/notes:
  - Test-only document models must use unique `collection_name`s (prefix with `s<step>_`) because of the global registry.
  - Repositories own query logic and raise typed errors; services/routers never touch `collection` directly.
  - Step 12's readiness endpoint should call `app.state.infrastructure.mongo.ping()`.
- Next: Step 11: Redis infrastructure

### Step 11: Redis infrastructure  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/infrastructure/redis.py`:
    - `RedisConnection`: one async pool on the app DB (`REDIS_DB`), `decode_responses=True`, socket/connect timeouts, max connections, health check every 30s, client name. `connect()` pings and fails fast, leaving no half-open client. `ping()` maps any `RedisError`/`OSError` (incl. bad password) to `RedisUnavailableError` (503). `delete_namespace(ns)` uses SCAN + batched UNLINK and never runs KEYS/FLUSHDB. `close()`.
    - `RedisKeyspace`: `key(namespace, *parts)` → `<prefix>:<namespace>:<part>...`. Parts can't be empty or contain whitespace, control chars or `:`; the error points to `digest()` (sha256, 32 hex chars) for URLs and origins. `pattern(namespace)` gives the SCAN glob.
    - `RedisNamespace` StrEnum, one member per key family with the step that uses it: `api_rl` (22), `robots` (29), `domain_rl` (30), `job_cancel` (52), `query_emb` (92), `lock`.
  - `infrastructure/lifecycle.py`: `AppInfrastructure(mongo, redis)` starts Mongo → indexes → Redis and stops in reverse. If any start fails it closes whatever was opened and re-raises.
  - `infrastructure/exceptions.py`: `RedisUnavailableError`.
  - Settings: `REDIS_KEY_PREFIX` (validated `^[a-z0-9_-]{1,32}$`, default `rag`), `REDIS_SOCKET_TIMEOUT_SECONDS`, `REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS`, `REDIS_MAX_CONNECTIONS` (+ `.env.example`).
- Deps: runtime `redis` 8.1 (redis-py asyncio).
- Tests: `TestRedisKeyspace_S11` (16, unit): key format, unsafe parts, separator hint, required part, digest stable/distinct/short, pattern, prefix validation, start/stop ordering, failed start closes opened services. `TestRedisInfrastructure_S11` (7, integration): connect/ping/close, unreachable fails fast, wrong password → 503, decoded strings + TTL, uses the configured DB, `delete_namespace` over 1,234 keys (multiple batches) leaves other namespaces and other prefixes alone, real app lifespan connects Redis. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 198 passed.
- Decisions/notes:
  - Code must build Redis keys only through `connection.keys.key(...)`. A new key family gets a `RedisNamespace` member first.
  - Celery broker/result DBs (1/2) are separate and owned by Celery (Step 75); this client never touches them.
  - If a later step needs binary values (e.g. packed vectors in Step 92), add a second client with `decode_responses=False` rather than changing this one.
- Next: Step 12: Readiness

### Step 12: Readiness  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/infrastructure/qdrant.py`: `QdrantConnection` (one `AsyncQdrantClient` per process from the `QDRANT_*` settings; gRPC by default; `check_compatibility=False` to skip the blocking handshake; `connect()` pings and fails fast without leaving a half-open client; `ping()` uses the authenticated `get_collections()`, so a wrong key fails at startup; `close()`).
    - Pulled forward from Step 74 because readiness must ping Qdrant. Step 74 still owns collection bootstrap; Steps 67–73 receive this client.
  - `infrastructure/health.py`: `check_readiness(checks, budget_seconds)` runs every ping concurrently, each under `asyncio.timeout`.
    - Returns `ReadinessReport{ready, dependencies: {name: DependencyStatus{status ok|error, latency_ms, error}}}`.
    - Errors are stable codes only (`timeout`, `database_unavailable`, `redis_unavailable`, `vector_store_unavailable`, `unexpected_error`), never exception messages. Failures are logged as warnings with dependency/latency.
  - `infrastructure/lifecycle.py`: `Infrastructure` protocol gained `health_checks()`. `AppInfrastructure(mongo, redis, qdrant)` starts Mongo → indexes → Redis → Qdrant, stops in reverse, cleans up on failed start, and exposes `{"mongo", "redis", "qdrant"}` pings.
  - `infrastructure/exceptions.py`: `VectorStoreUnavailableError` (503).
  - `api/routers/health.py`: `GET /api/v1/health/ready` → 200 when all checks pass, else 503 with the same `ReadinessReport` body (documented in OpenAPI). Unauthenticated (for probes). Liveness `GET /health` stays independent of dependencies.
  - Settings: `READINESS_CHECK_TIMEOUT_SECONDS` (default 2, max 30) + `.env.example`.
  - `tests/support/fakes.py`: `NoopInfrastructure(checks)`.
- Tests: `TestReadiness_S12` (11, unit): all ok, error code per dependency, three hanging checks time out concurrently (<0.6s for a 0.2s budget), crash contained with no leaked message, no checks = ready, endpoint 200/503 bodies, liveness unaffected, configured budget honored, OpenAPI documents the 503 body, `AppInfrastructure` exposes 3 checks. `TestReadinessIntegration_S12` (6, integration): Qdrant connect/ping/close; wrong API key fails at connect over both gRPC and REST; unreachable Qdrant fails; real app → 200 with all `ok`; Qdrant key rotated plus Redis unreachable after startup → 503 with Mongo `ok`, `redis_unavailable`, `vector_store_unavailable`. `TestRedisKeyspace_S11` start/stop ordering now includes Qdrant. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 215 passed.
- Decisions/notes:
  - redis-py's default retry/backoff on connection errors can exceed the 2s budget, so a Redis outage may show as `timeout` rather than `redis_unavailable`. Both mean not ready; the real client keeps its retries for transient blips.
  - Readiness has no caching; each probe pings all three (cheap calls). Revisit only if probe frequency becomes a load issue.
- Next: Step 13: Domain models & enums

### Step 13: Domain models & enums  (2026-10-17)
- Status: ✅ done
- Built (`rag_engine/domain/`):
  - `enums.py`:
    - `TenantStatus` (active/disabled), `ResourceStatus` (active/deleting, for KBs and sources).
    - `JobStatus` (queued → running → finalizing → succeeded/failed/cancelled, `.is_terminal`) with `ACTIVE_JOB_STATUSES` / `TERMINAL_JOB_STATUSES`.
    - `PageStatus` (queued, in_progress, succeeded, skipped, duplicate, failed).
    - `PageReason`: 21 stable skip/failure codes — robots_disallowed, out_of_scope, nofollow, max_depth, max_pages, canonical_elsewhere, unsupported_content_type, empty_content, job_cancelled, too_large, redirect_loop, too_many_redirects, ssrf_blocked, http_client_error, http_server_error, rate_limited, timeout, network_error, render_failed, extraction_failed, document_unreadable.
    - `ContentKind` (html, pdf, docx, pptx, xlsx), `RenderedWith` (http, browser).
    - `EmbeddingModelName` lives in `config/enums.py` and is re-exported here; settings' `DEFAULT_EMBEDDING_MODEL` now uses it instead of the `Literal`.
  - `types.py`:
    - Prefixed id types `TenantId`/`ApiKeyId`/`KnowledgeBaseId`/`SourceId`/`JobId`/`PageId`/`ChunkId` (`<prefix>_<32 hex>`), so a `kb_id` can't hold a tenant id.
    - `HttpUrlStr`: absolute http(s), host required, no whitespace or embedded credentials, ≤2048 chars, kept verbatim (normalization is Step 23).
    - `Name`, `Description`, `PathPattern`, `Sha256Hex`, `NonNegativeInt`.
  - Models (collection / id prefix / indexes):
    - `Tenant` (`tenants`/`ten`; name unique case-insensitively via `TENANT_NAME_COLLATION`). Embedded `TenantQuotas` (max KBs, max concurrent jobs, max pages per job ≤ `CRAWL_HARD_MAX_PAGES`, API rate/min) defaulting from settings; `default_embedding_model`.
    - `ApiKey` (`api_keys`/`key`; `key_prefix` globally unique; tenant index): `key_hash` (sha256 hex, hidden from repr), `last_used_at` / `expires_at` (must follow creation) / `revoked_at`, `is_usable(now)`.
    - `KnowledgeBase` (`knowledge_bases`/`kb`; name unique per tenant **among active KBs** via a partial index, so a deleting KB frees its name): `embedding_model` + `embedding_dimensions` are **frozen** (locked at creation).
    - `Source` (`sources`/`src`; seed_url unique per KB among active sources): `active_version` / `last_job_id` (job ids), `SourceStats`.
    - `CrawlJob` (`crawl_jobs`/`job`; tenant, source and status+heartbeat indexes):
      - `CrawlConfig` (max_pages/max_depth bounded by hard limits, ≤50 include/exclude patterns, optional webhook URL).
      - `JobCounters`, `EmbeddingUsage`, `JobError`, attempts, cancel/started/heartbeat/finished timestamps.
      - `version` = job id.
      - Invariants: failed ⇒ error; terminal ⇔ finished_at.
    - `Page` (`pages`/`page`; `(job_id, url_hash)` unique; job+status index): outcome fields (status, reason, reason_detail, http_status, final_url, content_type/kind, rendered_with, title, markdown, content_hash, canonical_url, duplicate_of, bytes, fetch_ms, attempts). Invariants: skipped/failed ⇔ reason; duplicate ⇒ duplicate_of; succeeded ⇒ content_kind + markdown.
    - `Chunk` (`chunks`/`chk`; source+version index, `(page_id, ordinal)` unique): text, embed_text, heading_path (≤12), ordinal, token_count > 0, content_hash, version (job id).
  - Settings: `TENANT_DEFAULT_MAX_KNOWLEDGE_BASES` (20), `_MAX_CONCURRENT_JOBS` (2), `_MAX_PAGES_PER_JOB` (2000, validated ≤ `CRAWL_HARD_MAX_PAGES`), `_API_RATE_LIMIT_PER_MINUTE` (120) + `.env.example`.
  - `tests/support/fixture_site.py`: `EXPECTED_PAGE_OUTCOME` maps every fixture-site `Expect` to a `(PageStatus, PageReason)` pair, the oracle Step 54 asserts against.
- Tests: `TestDomainModels_S13` (41, unit): registration with distinct names and prefixes, BSON round-trip per model, id-prefix enforcement, URL rejection matrix and verbatim storage, tenant defaults and quota limits, key repr/prefix/usability/expiry, frozen KB model, source version type, job defaults/config limits/status invariants, status sets partition, page invariants, every fixture outcome representable, chunk content rules, enum values snapshot. `TestDomainIndexes_S13` (6, integration): all 7 models round-trip in Mongo; case-insensitive tenant names; global key prefix; KB name freed when deleting; source seed per KB; page URL per job. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 262 passed.
- Decisions/notes:
  - Enum values are a stored and API contract (snapshot-tested). Renaming one needs a data migration.
  - `Page.markdown` lives on the page document, so Mongo's 16 MB document limit applies. Step 47 must cap stored markdown (or move large bodies out) before persisting.
  - The one-active-job-per-source partial unique index is deliberately left to Step 79.
  - Queries on `Tenant.name` must pass `collation=TENANT_NAME_COLLATION` to use the unique index.
- Next: Step 14: Embedding model registry

### Step 14: Embedding model registry  (2026-10-17)
- Status: ✅ done
- Built (`rag_engine/embeddings/`):
  - `models.py`:
    - `EmbeddingModelSpec` (frozen): name, provider, `native_dimensions`, `supports_dimension_reduction`, `max_input_tokens` (8191), `max_batch_inputs` (2048), `max_batch_tokens` (300k), `tokenizer_encoding` (`cl100k_base`), and a derived `min_dimensions`.
    - `EMBEDDING_MODELS`: read-only registry for `text-embedding-3-small` (1536) and `text-embedding-3-large` (3072).
    - `EmbeddingSpace(model, dimensions)` is the unit of vector compatibility. Its `.key` (e.g. `text-embedding-3-large__1024`) is collection-name safe and is what Step 68 names Qdrant collections by.
    - `get_model_spec()` accepts an enum or a string.
    - `resolve_space(model, dims=None)`: `None` means native size. Reduced sizes are allowed between `MIN_REDUCED_DIMENSIONS` (256) and native.
    - `request_dimensions(space)`: the value to send as OpenAI's `dimensions` parameter, `None` at native size.
  - `exceptions.py`: `UnsupportedEmbeddingModelError` (lists the supported models) and `UnsupportedEmbeddingDimensionsError` (requested/min/max), both 422.
  - `domain/knowledge_base.py`: a validator checks that `embedding_dimensions` is valid for the model (re-raised as `ValueError` so pydantic reports a normal `ValidationError`); new `embedding_space` property.
- Tests: `TestEmbeddingModelRegistry_S14` (22, unit): registry covers every model name and provider, default model registered, OpenAI spec values, immutability, string lookup, unknown model error, native default, reduced range accepted/rejected, stable distinct space keys, `dimensions` only sent when reduced, KB validation + `embedding_space`. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 284 passed.
- Decisions/notes:
  - 256 as the minimum reduced size is a product decision (below it retrieval quality drops sharply); change `MIN_REDUCED_DIMENSIONS` if needed.
  - `tiktoken` is not added yet; it downloads encoding files on first use. Step 57 adds it, and Step 95 must bake the encoding cache (`TIKTOKEN_CACHE_DIR`) into the images.
  - Limits are OpenAI's published ones; Step 62's batcher must respect both `max_batch_inputs` and `max_batch_tokens`.
  - Step 19 should call `resolve_space(model, requested_dims)` when creating a KB so API clients get the typed 422 with min/max details.
- Next: Step 15: API key service

### Step 15: API key service  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/security/api_keys.py`:
    - Key format `rk_<prefix>_<secret>`: 12-char base62 public prefix (lookup) plus 43-char base62 secret (~256 bits), from `secrets`.
    - `ApiKeyHasher`:
      - `generate()` → `GeneratedApiKey(plaintext: SecretStr, prefix, key_hash)`; plaintext and hash hidden from repr.
      - `hash()` = `HMAC-SHA256(API_KEY_PEPPER, full key)` hex. The pepper lives only in settings, so a DB leak alone can't be brute-forced.
      - `verify(plaintext, stored_hash)` → `VerificationResult(valid, needs_rehash)`: `hmac.compare_digest` against the current pepper **and** every retired pepper (always all of them, so timing doesn't reveal which matched). A match on a retired pepper sets `needs_rehash`.
      - `burn()` performs equivalent hashing work when a prefix isn't found, so unknown-prefix and wrong-secret responses take the same time.
    - `parse_api_key(raw)` → `ParsedApiKey(prefix, plaintext)` or `None` (trims whitespace; accepts prefix 8–16 and secret 32–64 to match the `ApiKey` model; rejects other schemes, bad characters, extra parts, a `Bearer ` prefix).
    - `display_hint(prefix)` → `rk_<prefix>_…` for listings.
  - Settings: `API_KEY_PEPPER_PREVIOUS: list[SecretStr]` (each ≥32 chars) for zero-downtime pepper rotation; `.env.example` documents the procedure (move the old pepper into the list, set a new one; keys re-hash on next use).
- Tests: `TestApiKeyService_S15` (24, unit): exact format, 500 unique keys/prefixes, secrets hidden from repr, the hash equals the HMAC spec, pepper changes the hash, a generated key fits the `ApiKey` model, valid/wrong-secret/other-key verify, rotation accepted with `needs_rehash` then a clean re-hash, old keys rejected without the previous pepper, previous-pepper validation, burn, parse ok + 10 malformed cases, display hint. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 308 passed.
- Decisions/notes:
  - Step 16 must retry `generate()` on `DuplicateDocumentError` for the (astronomically unlikely) prefix collision.
  - Step 17 must: `parse_api_key` the header (None → 401) → look up by prefix → if no key, `burn()` and 401 → `verify` → if `needs_rehash`, store `hash()` → check `is_usable()` + tenant active. Every failure gets the same generic 401 so callers can't tell which check failed.
- Next: Step 16: Tenant & ApiKey repositories

### Step 16: Tenant & ApiKey repositories  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/repositories/tenants.py`: `TenantRepository`
    - `create()` (duplicate → `TenantNameTakenError` 409, details `{name}`).
    - `get_by_name()` (trimmed, case-insensitive via `TENANT_NAME_COLLATION`, so it uses the unique index).
    - `update()` (rename into a taken name → `TenantNameTakenError`).
    - `set_status()`; not found → `TenantNotFoundError` (`tenant_not_found`).
  - `rag_engine/repositories/api_keys.py`: `ApiKeyRepository`
    - `issue(tenant_id, hasher, name, expires_at)` → `(ApiKey, GeneratedApiKey)`; retries a prefix collision up to 3 times, then raises `DuplicateDocumentError`.
    - `get_by_prefix()`; `list_for_tenant()` (paginated, hides revoked unless `include_revoked`); `count_active()` (excludes revoked and expired).
    - `revoke(key_id, tenant_id)`: tenant-scoped, idempotent (keeps the first `revoked_at`).
    - `touch_last_used()`: atomic and throttled to once per `LAST_USED_RESOLUTION` (1 min), so auth doesn't write on every request.
    - `rehash()`: stores the current-pepper hash after a retired-pepper match.
  - `repositories/exceptions.py`: `TenantNotFoundError`, `TenantNameTakenError`, `ApiKeyNotFoundError`.
  - **Fix in `BaseRepository`**: `DuplicateDocumentError.details` used to embed Mongo's raw error (conflicting values plus server message), which reached API clients. Now it's `{collection, fields}` with field names only (shared `_duplicate()` for insert and update).
- Tests: `TestTenantRepository_S16` (6, integration): case-insensitive lookup, duplicate → typed 409, rename conflict, status and quotas update, typed not-found, duplicate details carry field names but no values. `TestApiKeyRepository_S16` (8, integration): only the hash is stored (plaintext absent from the raw doc), issue → parse → lookup → verify round-trip, retry on 2 collisions, give up after repeated collisions, list/count exclude revoked/expired/other tenants, revoke is tenant-scoped and idempotent, last-used throttling, rehash after pepper rotation. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 322 passed.
- Decisions/notes:
  - No per-tenant cap on the number of API keys yet; add one in Step 18's admin service if wanted.
  - `touch_last_used` and `rehash` aren't tenant-scoped: they're called by auth (Step 17) after the key was already resolved from its prefix.
- Next: Step 17: Auth dependencies

### Step 17: Auth dependencies  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/api/deps/infrastructure.py`: `get_db` (from `app.state.infrastructure.mongo`), `get_api_key_hasher` (one per process, overridable), repository deps; Annotated aliases `Database`, `Hasher`, `Tenants`, `ApiKeys`.
  - `rag_engine/api/deps/auth.py`:
    - `require_admin` (`AdminAuth`): `Authorization: Bearer <ADMIN_API_KEY>`, compared in constant time over SHA-256 digests so the comparison time doesn't depend on input length.
    - `require_tenant` (`CurrentTenant` → `TenantContext{tenant, api_key, tenant_id, scope}`): parse → lookup by prefix (unknown → `burn()`) → verify → usable (not revoked/expired) → tenant exists → binds `tenant_id` to the log context → tenant active → re-hash if a retired pepper matched → `touch_last_used`.
    - Every authentication failure is the same 401 `invalid_credentials` / "Missing or invalid API key" / empty details / `WWW-Authenticate: Bearer`.
    - **A valid key whose tenant is disabled → 403 `tenant_disabled`.** This changes the Step 15 note: the holder has already proven the secret, so 403 leaks nothing and tells operators why a working key stopped working.
    - OpenAPI security schemes `AdminKey` and `TenantApiKey` (HTTP bearer), so Swagger's Authorize button works.
  - `config/logging.py`: `bind_log_context(**ids)` sets context IDs without a reset (safe in FastAPI dependencies; each request has its own context copy).
- Tests: `TestAdminAuth_S17` (11, unit): correct admin key → 200; 9 wrong shapes (missing, empty, wrong, ±1 char, Basic, no scheme, tenant-format key, `X-API-Key`) → uniform 401 without echoing the key; OpenAPI schemes and per-route security. `TestTenantAuth_S17` (11, integration, real app + Mongo): valid key → tenant and `tenant_id` in the log context; missing/malformed, unknown prefix (`burn` called), wrong secret, revoked, expired, deleted tenant → identical 401; disabled tenant → 403; `last_used_at` set; retired-pepper key accepted and re-hashed; admin key rejected as a tenant key. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 344 passed.
- Decisions/notes:
  - FastAPI resolves `require_tenant`'s sub-dependencies (repositories → `get_db`) before the header is parsed, so tenant routes need `app.state.infrastructure.mongo`. Unit tests of tenant routes must override `get_api_key_repository`/`get_tenant_repository` (or use integration tests).
  - Model invariants also run when loading from Mongo: a stored document that violates one (e.g. `expires_at` before `created_at`) makes reads fail. Writes always go through validation, so this only happens with manual DB edits.
  - Use `ctx.scope` as the `scope=` argument on every tenant-owned repository call (Steps 19+).
- Next: Step 18: Admin API

### Step 18: Admin API  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/api/routers/admin.py` (prefix `/admin`, tag `admin`, every route behind `AdminAuth`, 401/403/404/409 documented as `ErrorResponse`):
    - `POST /admin/tenants` (201)
    - `GET /admin/tenants` (cursor pagination, `?status=`)
    - `GET /admin/tenants/{tenant_id}`
    - `PATCH /admin/tenants/{tenant_id}` (name, status, default model, partial quotas merged into the current ones)
    - `POST /admin/tenants/{tenant_id}/keys` (201, plaintext `api_key` returned once, `Cache-Control: no-store`)
    - `GET /admin/tenants/{tenant_id}/keys` (`?include_revoked=`)
    - `DELETE /admin/tenants/{tenant_id}/keys/{key_id}` (idempotent revoke, returns the key)
    - Path IDs are validated by prefix (malformed → 422).
  - `rag_engine/api/schemas/admin.py`: `TenantCreate`, `TenantPatch` (≥1 field, no explicit nulls), `TenantQuotasPatch`, `TenantResponse`, `ApiKeyCreate` (`expires_at` must be timezone-aware and in the future), `ApiKeyResponse` (hint, `state` active/revoked/expired; never the hash), `ApiKeyCreatedResponse`. Requests use `extra="forbid"`.
  - `rag_engine/services/tenant_admin.py`: `TenantAdminService` (quota merge with typed 422, per-tenant key cap → 403 `quota_exceeded {quota, limit}`, tenant existence checks, audit log lines with tenant/key ids).
  - Settings: `TENANT_MAX_ACTIVE_API_KEYS` (default 10) + `.env.example`.
  - `BaseRepository.update`: model validation failures now raise `AppValidationError` (422) instead of a raw pydantic `ValueError` (which surfaced as 500). Error details drop input/ctx/url so they stay JSON-serializable and never echo values. The Step 10 test was updated accordingly.
- **Incident fixed in this step:** the full suite hung. Root cause: the Mongo test container ran out of file descriptors (Docker default `nofile` = 1024). WiredTiger keeps a file per collection and index, every test gets its own DB, and Step 18's tests crossed the limit → `EMFILE` → mongod fatal assertion (msgid 50853), exit 139. Every later test then waited 30 s in teardown on pymongo's default server-selection timeout.
  - Test Mongo container: `nofile` 64000 (MongoDB's recommendation).
  - **`docker-compose.yml` Mongo: `ulimits.nofile` 64000**, since production would hit the same wall as tenants, KBs and indexes grow.
  - Teardown clients use short timeouts (Mongo 2 s, Redis 2/5 s, Qdrant 5 s).
  - Added `pytest-timeout` (120 s per test) so future hangs fail by name.
  - Removed the containers left behind by the killed runs.
- Tests: `TestAdminTenantsApi_S18` (18, integration): every route rejects a wrong key, create with defaults / model / partial quotas, 6 invalid bodies → 422, duplicate → 409, get / 404 / malformed id 422, pagination and status filter, patch merges quotas and bumps `updated_at`, 4 invalid patches → 422, rename conflict 409 / missing 404. `TestAdminKeysApi_S18` (8, integration): plaintext once + no-store and never in listings, tenant key rejected on admin routes, expiry validation, key cap 403 then OK after revoke, revoked hidden unless asked, revoke idempotent and tenant-scoped, unknown tenant 404, OpenAPI shows 7 admin ops under `AdminKey`. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 370 passed (52 s).
- Decisions/notes:
  - The key-cap check is count-then-insert, so two concurrent admin requests could exceed the cap by one. Acceptable for an operator-only endpoint.
  - Changing a tenant's `default_embedding_model` affects only KBs created afterwards (documented in the schema).
- Next: Step 19: KnowledgeBase repo + service

### Step 19: KnowledgeBase repo + service  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/repositories/knowledge_bases.py`: `KnowledgeBaseRepository`. Every method takes the tenant id, and KBs being deleted are invisible to reads.
    - `create()` (duplicate → `KnowledgeBaseNameTakenError` 409), `get_active()`, `list_active()` (paginated), `count_active()`.
    - `update_active()` (rename conflict → 409; a deleting KB → 404).
    - `mark_deleting()`: atomic active → deleting, frees the name; a second call → 404.
    - `remove()`: hard delete, for create rollback and the Step 81 cascade.
  - `rag_engine/services/knowledge_bases.py`: `KnowledgeBaseService`
    - `create(tenant, name, description, embedding_model?, embedding_dimensions?)`: model = explicit or the tenant default; `resolve_space()` gives typed 422s (`unsupported_embedding_model` / `unsupported_embedding_dimensions` with min/max); native size when dimensions are omitted.
    - Quota check counts → inserts → recounts and rolls back its own insert if concurrent creates overshot, so `max_knowledge_bases` holds under races (403 `quota_exceeded {quota, limit}`).
    - `get`, `list`, `update` (name/description only; model, dimensions, status or tenant → 422 `AppValidationError` with the field list), `delete` (marks deleting; data stays until Step 81's cascade).
  - `repositories/exceptions.py`: `KnowledgeBaseNotFoundError` (`knowledge_base_not_found`), `KnowledgeBaseNameTakenError` (`knowledge_base_name_taken`).
- Tests: `TestKnowledgeBaseService_S19` (11, integration): tenant default model at native size, explicit model + reduced dimensions (space key), invalid model/dimensions → typed 422 with nothing stored, name unique per tenant and freed by delete, quota counts only active KBs, 8 concurrent creates against a quota of 2 never exceed it (repeated 5× to check for flakiness), tenant-scoped get that hides deleting KBs, active-only paginated listing, name/description update, locked-field 422, rename conflict, a deleting KB can't be updated or deleted again (data kept), another tenant can't delete. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 381 passed.
- Decisions/notes:
  - Under a race the recount can roll back *both* overshooting creates (fail-safe: never above the quota, may briefly reject one extra request). The caller just retries.
  - Deleting a KB only hides it and frees the name. Step 20 enqueues the cascade task once Celery exists (Step 75); until then deleting KBs stay as hidden records.
- Next: Step 20: KB API

### Step 20: KB API  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/api/routers/knowledge_bases.py` (`/knowledge-bases`, tag `knowledge-bases`, every route needs a tenant key, 401/403/404/409 documented):
    - `POST /knowledge-bases` (201)
    - `GET /knowledge-bases` (paginated, newest first)
    - `GET /knowledge-bases/{kb_id}`
    - `PATCH /knowledge-bases/{kb_id}` (name/description only)
    - `DELETE /knowledge-bases/{kb_id}` → **202 Accepted** with status `deleting` (hidden immediately, name freed; Step 81 cascades the data).
    - IDs are validated by prefix (malformed or wrong-prefix → 422).
  - `rag_engine/api/schemas/knowledge_bases.py`: `KnowledgeBaseCreate` (`embedding_model`/`embedding_dimensions` optional, documented as permanent), `KnowledgeBasePatch` (≥1 field, no nulls), `KnowledgeBaseResponse` (no `tenant_id`). Requests use `extra="forbid"`.
  - `rag_engine/api/params.py`: shared `Limit` / `Cursor` query params, now used by the admin and KB routers.
  - `api/deps`: `get_knowledge_base_repository` / `KnowledgeBases`.
- Tests: `TestKnowledgeBaseApi_S20` (14, integration, two tenants bootstrapped through the real admin API): no key / admin key → 401, create with explicit space, tenant default at native size, 6 invalid bodies (schema 422 vs typed `unsupported_embedding_dimensions`), duplicate 409 + quota 403 with details, list pagination + get, patch rename keeps the space, locked/empty/null patches → 422, delete 202 → 404 on repeat/get, hidden from list, name reusable, malformed ids 422. `TestKnowledgeBaseTenantIsolation_S20` (4): another tenant gets 404 on get/patch/delete, indistinguishable from a missing KB and the owner's KB untouched; listings tenant-scoped even with identical names; quota is per tenant; disabled tenant → 403. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 399 passed (72 s).
- Decisions/notes:
  - Cross-tenant access returns 404 (never 403) so a caller can't probe which KB ids exist.
  - `DELETE` returns the KB body (status `deleting`) rather than an empty 204, so clients can show what is being removed.
- Next: Step 21: Source repo + API

### Step 21: Source repo + API  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/repositories/sources.py`: `SourceRepository`, scoped by `(tenant_id, kb_id)`; sources being deleted are invisible.
    - `get_or_create(tenant_id, kb_id, seed_url)` → `(source, created)`: finds the active source with that seed URL or inserts one. **Race-safe**: on a unique-index conflict it returns the winner's source.
    - `get_active`, `list_active` (paginated), `count_active`.
    - `mark_deleting` (atomic, frees the seed URL; a repeat → 404).
  - `rag_engine/services/sources.py`: `SourceService` checks first that the KB is active and owned by the caller, so a missing, foreign or deleting KB hides all its sources (404 `knowledge_base_not_found`). Then list / get / delete.
  - `rag_engine/api/routers/sources.py` (`/knowledge-bases/{kb_id}/sources`, tag `sources`):
    - `GET` (paginated)
    - `GET /{source_id}`
    - `DELETE /{source_id}` → 202 with status `deleting`
    - IDs validated by prefix.
  - `rag_engine/api/schemas/sources.py`: `SourceResponse` (seed_url, status, `active_version`, `last_job_id`, stats; no tenant_id).
  - `SourceNotFoundError` (`source_not_found`); deps `get_source_repository` / `Sources`.
- Tests: `TestSourceRepository_S21` (4, integration): idempotent get_or_create, 10 concurrent calls → exactly one source and one `created=True` (repeated 3× to check for flakiness), same seed in another KB/tenant is a separate source, deleting frees the seed (a repeat delete → typed 404; record kept as `deleting`). `TestSourceApi_S21` (8): list pagination + get shape, delete 202 then hidden, a source is only reachable through its own KB (wrong KB path → 404, the source untouched), other tenant → 404 `knowledge_base_not_found` everywhere, deleted KB hides its sources, malformed ids 422, tenant key required. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 411 passed (86 s).
- Decisions/notes:
  - No create endpoint: sources come into being when a crawl is submitted (Step 84), which must **normalize the seed URL (Step 23) before `get_or_create`** so trivially different spellings map to one source.
  - `set_active_version` / `last_job_id` / stats updates belong to Steps 78 and 84. Deleting a source with a crawl in flight must cancel that crawl (Step 87); deletion data cascade is Step 81.
  - The suite now takes ~86 s; integration tests dominate. If it keeps growing, consider `pytest-xdist` with per-worker databases (currently fine).
- Next: Step 22: Per-tenant API rate limiting

### Step 22: Per-tenant API rate limiting  (2026-10-17)
- Status: ✅ done
- Built:
  - `rag_engine/security/rate_limit.py`: `SlidingWindowRateLimiter`, an **exact sliding window**: one Lua script per hit trims entries older than the window, counts, and only records the request if under the limit (denied requests don't consume capacity). It sets `PEXPIRE` = window and uses the **Redis server clock** (`TIME`), so API replicas with skewed clocks agree. Returns `RateLimitDecision{allowed, limit, remaining, retry_after (s, ≥1 when denied), window_seconds}` with `.headers()`.
  - `api/deps/auth.py`: `require_tenant_within_rate_limit`. **`CurrentTenant` now means authenticated + rate limited**, so every tenant route is covered automatically.
    - Limit = `tenant.quotas.api_rate_limit_per_minute` over `API_RATE_LIMIT_WINDOW_SECONDS` = 60, keyed `rag:api_rl:<tenant_id>`.
    - Success responses carry `X-RateLimit-Limit` / `-Remaining` / `-Reset`.
    - Over quota → 429 `rate_limited` with `Retry-After` + the same headers + details `{limit, window_seconds, retry_after}`.
    - **Fails open** if Redis errors (logged at error level): readiness already reports the outage, and the limiter must not turn a Redis blip into a full API outage.
    - Admin routes and unauthenticated requests aren't counted.
  - `TooManyRequestsError` accepts extra `headers`.
- **Test-infra fix:** an intermittent teardown error (2 of 6 runs) was a container readiness race. The official Mongo image starts a temporary `mongod` to create the root user and then restarts it, so "Waiting for connections" is logged **twice**; testcontainers returned after the first. Step 18's 2 s teardown timeout exposed it. The fixture now waits for the 2nd occurrence and performs an authenticated ping before yielding (8/8 clean runs afterwards).
- Tests: `TestSlidingWindowLimiter_S22` (7, integration, real Redis): allow-then-deny with remaining counts and retry_after, window slides after 1 s, denials don't consume capacity, 60 concurrent hits against a limit of 10 → exactly 10 allowed, independent subjects + key TTL, bad parameters rejected. `TestTenantApiRateLimit_S22` (6, real app): success headers, 4th request with quota 3 → 429 with Retry-After/headers/details, tenants limited independently, admin routes unlimited, unauthenticated requests don't count, fails open with Redis down. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 424 passed (92 s).
- Decisions/notes:
  - The sliding log stores one ZSET entry per allowed request (≤ the per-minute quota per tenant), which is negligible memory at these limits.
  - `X-RateLimit-Reset` on allowed requests reports the window length (an upper bound), not the exact time until the oldest entry expires.
  - The limiter instance is cached on `app.state.rate_limiter` (built lazily from the Redis client).
- Next: Step 23: URL normalization

### Step 23: URL normalization  (2026-10-17)
- Status: ✅ done
- Built: `rag_engine/crawling/url.py` (new `crawling` package):
  - `normalize_url(url, base=None)` → canonical absolute URL, or `UrlNormalizationError` (a `ValueError`). Rules, in order:
    1. resolve against `base`; strip outer whitespace and remove internal tab/CR/LF
    2. scheme lowercased, http/https only
    3. **credentials rejected** (not silently dropped)
    4. host lowercased, IDNA 2008/UTS #46 via the `idna` package, trailing dot removed, IPv6 kept bracketed and lowercased
    5. default ports dropped; invalid ports rejected
    6. path: RFC 3986 dot-segment removal; empty → `/`; percent-encoding normalized (uppercase hex, unreserved decoded, unsafe and non-ASCII → UTF-8 escapes)
    7. query: tracking params stripped (`utm_*`, `pk_*`, `mtm_*` prefixes plus ~27 click IDs such as `gclid`, `fbclid`, `msclkid`, `mc_cid`, HubSpot `__hs*`), the rest sorted by key then value with blank values kept, `?` dropped when empty
    8. fragment dropped
  - `url_hash(normalized)` = sha256 hex (the `Page.url_hash` / frontier key); `normalize_and_hash()`.
  - **Policy choices**: trailing and repeated slashes are **preserved** (servers may serve different resources; duplicates are caught by redirects and content hashing in Step 46). Ambiguous params like `ref` are **kept** (some sites use them for content), so the fixture site's tracked link normalizes to `/docs/getting-started.html?ref=home`.
- Deps: runtime `idna` 3.20; dev `hypothesis` 6.170.
- Tests: `TestUrlNormalization_S23` (58, unit): 21 canonical-form cases, 11 query cases (sorting, duplicate keys, tracking removal, blank values, `+` vs `%2B`, encoded `&`), fixture-site tracked link, 7 relative resolutions, 14 rejections (non-http schemes, credentials, missing host, bad ports/hosts, broken IPv6), hash consistency. **Hypothesis properties** (300 examples each over unicode hosts, odd path segments including `.`/`..`, query pairs, ports, fragments): idempotent; output is ASCII, lowercase host, absolute path, no fragment and accepted by the `Source` model; invariant under shuffled params, injected tracking params, upper-cased scheme/host and fragments. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 482 passed.
- Decisions/notes:
  - Step 84 must run the submitted seed URL through `normalize_url` before `SourceRepository.get_or_create`; reject with 422 on `UrlNormalizationError`.
  - `+` in queries is read as a space (form encoding) and re-emitted as `%20`; a literal plus must be `%2B`. This is the common server interpretation.
  - The tracking list is a code constant. Make it a setting later only if tenants need per-site overrides.
- Next: Step 24: Scope rules

### Step 24: Scope rules  (2026-10-23)
- Status: ✅ done
- Built:
  - `rag_engine/crawling/scope.py`: `CrawlScope` (built per job via `CrawlScope.for_job(seed_url, CrawlConfig)` or `from_config`) → `check(normalized_url, depth)` → `ScopeDecision{allowed, reason: PageReason, detail: ScopeRejection}`. Checks run in this order:
    1. scheme (http/https)
    2. **host**: same host as the seed, with `www.` ≡ the bare domain; scheme and port are ignored. With `include_subdomains`, any host under the seed's **registrable domain** via the Public Suffix List (`tldextract`, **bundled snapshot, never downloaded**, **private section on**, so `foo.github.io` ≠ `bar.github.io`). Unknown suffixes fall back to the last two labels; IPs and single-label hosts match exactly only.
    3. asset extensions (`ASSET_EXTENSIONS`: images, CSS/JS/JSON/XML/feeds/wasm, fonts, audio/video, archives/binaries; PDF/Office documents stay in scope)
    4. `path_prefix` (literal)
    5. exclude globs (they win)
    6. include globs (none = everything)
    7. depth > `max_depth` → `max_depth`
  - Rejections use `PageReason.OUT_OF_SCOPE` / `MAX_DEPTH` plus a `ScopeRejection` detail for `Page.reason_detail`: `other_host`, `asset_extension`, `outside_path_prefix`, `excluded_pattern`, `not_included`, `unsupported_scheme`, `too_deep`.
  - `with_seed_host(host)` re-anchors the scope when the seed redirects to another host (Step 50 must call it after the seed fetch).
  - `registrable_domain()` (cached).
  - `domain/types.py`: `compile_glob` lives here (`*` within a segment, `**` across, `?` one char, everything else literal) to avoid a domain↔crawling import cycle. `PathPattern` now **validates** globs (must start with `/`). New `PathPrefix` (starts with `/`, no query/fragment/globs/spaces).
  - `CrawlConfig`: new `include_subdomains: bool = False` and `path_prefix: PathPrefix | None`, so bad patterns get a 422 at submission.
- **Fixture site adjusted** (scope now skips asset extensions before fetching):
  - `/files/large.bin` → `/files/huge-catalog.pdf` (25 MiB `application/pdf` with Content-Length; still `TOO_LARGE`).
  - New `/api/widgets` (no extension, JSON body) is now the `UNSUPPORTED_TYPE` case.
  - `/data.json` and `/images/logo.png` → `OUT_OF_SCOPE` (asset).
  - The Step 9 log entry above still says `large.bin`; this note supersedes it.
- Deps: runtime `tldextract` 5.4.
- Tests: `TestCrawlScope_S24` (55, unit): same-host matrix (scheme/port/www ignored; subdomains, suffix tricks and look-alikes rejected), www seed, 9 `include_subdomains` cases (co.uk, github.io private suffix, unknown TLD, IPs, single label), registrable domain, literal path prefix, 9 glob-semantics cases, exclude beats include, no includes = all, 6 asset paths skipped, documents/pages kept, asset list excludes documents, depth limit, scope reported before depth, non-http scheme, re-anchoring, `for_job`, 5 bad `CrawlConfig` patterns → `ValidationError`, and **every fixture-site URL's scope outcome matches `SITE_PAGES`**. Fixture-site tests updated for the new routes. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 538 passed (127 s; integration timing varies 90–130 s run to run).
- Decisions/notes:
  - `.json` / `.xml` count as assets for link-following. Sitemaps (XML) are fetched by Step 31's discovery, not via link scope.
  - Robots rules are not part of scope (Step 29); the engine checks scope first, then robots.
- Next: Step 25: SSRF guard

### Step 25: SSRF guard  (2026-10-23)
- Status: ✅ done
- Built: `rag_engine/security/ssrf.py`
  - `SsrfGuard(config, resolver, dns_timeout_seconds)`:
    - `check_static(url)`: scheme http/https; port in `SSRF_ALLOWED_PORTS`; blocked hostnames (`localhost`, `*.localhost`, `metadata`, `metadata.google.internal`, `instance-data[.ec2.internal]`).
    - `vet(url)` → `VettedTarget{scheme, host, port, addresses}`: IP literals skip DNS; otherwise **all** A/AAAA records are resolved (with a timeout) and **every one must pass**, so one private record blocks the host (defeats mixed public/private DNS answers). Returns the de-duplicated vetted IPs for Step 26 to pin. IPv6 zone ids are stripped.
  - `classify(ip, allow_private)` returns `None` or an `SsrfRejection`:
    - **Cloud metadata IPs are always blocked**: 169.254.169.254, 169.254.170.2, 169.254.169.123, 100.100.100.200, 192.0.0.192, fd00:ec2::254/23.
    - Embedded IPv4 is unwrapped and checked: IPv4-mapped, **NAT64** (`64:ff9b::/96`, which reports `is_global=True` even for 127.0.0.1), 6to4, Teredo. 6to4/Teredo are rejected as reserved even when the inner IP is public.
    - Unspecified and multicast are always blocked (`ff0e::` also reports `is_global=True`).
    - Global unicast is allowed.
    - Loopback, link-local and **explicitly defined** private ranges (RFC 1918, CGNAT 100.64/10, ULA fc00::/7) are allowed only with `SSRF_ALLOW_PRIVATE_NETWORKS`.
    - Everything else (240/4, broadcast, TEST-NETs, benchmarking, 2001:db8::/32…) is reserved and always blocked.
  - `SSRF_ALLOW_PRIVATE_NETWORKS` (forbidden in prod) also lifts the port allowlist so tests can reach the fixture site on its random loopback port, but metadata names and IPs stay blocked.
  - Errors: `SsrfBlockedError` (`AppValidationError`, 422 `url_not_allowed`, details `{reason, host}`, **never the resolved IPs**, so the guard can't be used as a DNS oracle; `.rejection` for `Page.reason_detail`). `HostResolutionError` for DNS failure/timeout is a network error, not an SSRF verdict (→ `PageReason.NETWORK_ERROR`).
- **Bug caught while writing tests:** the first draft used `ipaddress.is_private` for "private networks", but it also covers reserved, broadcast and documentation ranges, so with private networks allowed `240.0.0.1` / `255.255.255.255` / TEST-NETs would have passed. Replaced with the explicit `PRIVATE_NETWORKS` list and covered by tests.
- Tests: `TestSsrfGuard_S25` (70, unit): 35-address classification matrix (public v4/v6, loopback, RFC 1918, CGNAT, ULA, link-local, 3 metadata, unspecified, multicast incl. `ff0e::1`, broadcast/240/TEST-NET/benchmarking/documentation, IPv4-mapped, NAT64 → loopback and → metadata, 6to4, Teredo); 13-case `allow_private` matrix (private/loopback allowed; metadata, reserved, multicast, unspecified still blocked); vetted addresses de-duplicated; one private record blocks a host (no IP in error details); IP literals skip DNS; DNS failure and timeout → `HostResolutionError`; no records → blocked; zone id stripped; 11 static rejections (file/ftp/gopher, ports 22/6379/8443, localhost names, metadata names incl. upper-case + trailing dot); configured extra port allowed; allow-private permits loopback/localhost and any port but not metadata by IP or name; real system resolver on localhost; fixture site reachable only with private networks allowed, and its metadata redirect target blocked even then. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 608 passed (91 s).
- Decisions/notes:
  - The guard decides once per URL. **Step 26 must connect only to `VettedTarget.addresses`** (DNS-rebinding defence) and re-vet every redirect hop. Playwright requests are checked in Step 35, webhooks in Step 83.
  - Step 84 uses `vet()` on the submitted seed URL to return a 422 `url_not_allowed` up front.
- Next: Step 26: SSRF-safe HTTP client

### Step 26: SSRF-safe HTTP client  (2026-10-23)
- Status: ✅ done
- Built: `rag_engine/security/safe_http.py`
  - `PinnedNetworkBackend` (httpcore `AsyncNetworkBackend`): `connect_tcp(host, port)` connects **only to the IPs pinned for that host by the guard**, trying them in vetted order (falls back on connect failure). Unvetted hosts and unix sockets raise `UnvettedHostError`. httpcore still passes the hostname for `Host`, **TLS SNI and certificate verification**, so pinning doesn't weaken TLS. Pins are a bounded LRU (10k).
  - `_PinnedTransport`: a minimal httpx transport over an `httpcore.AsyncConnectionPool` with that backend (HTTP/1.1, default certifi SSL context, pool size `max(2×CRAWL_CONCURRENCY, 10)`, keep-alive 30 s), mapping httpcore errors to httpx `TimeoutException` / `NetworkError` / `ProtocolError`.
  - `SafeHttpClient.stream_get(url, headers)` (async context manager → `SafeResponse{response (unread), chain, url}`):
    - every hop is `SsrfGuard.vet()`-ed then pinned before connecting
    - redirects (301/302/303/307/308) are followed manually with intermediate responses closed
    - loops raise `RedirectLoopError`; more than `CRAWL_MAX_REDIRECTS` raises `TooManyRedirectsError` (both carry `.chain`)
    - `User-Agent: CRAWL_USER_AGENT`, `Accept-Encoding: gzip, deflate`, connect/read timeouts from settings
  - **Deliberately not `httpx.AsyncClient`**: no cookie jar (it would leak cookies between tenants' crawls), no auth, no `HTTP(S)_PROXY` env handling (a proxy would bypass pinning).
- Fixture site: unlinked open redirect `/redirect-to?url=…` for tests (doesn't affect `SITE_PAGES`).
- Deps: dev `trustme` (test CA + certificates).
- Tests: `TestSafeHttpClient_S26` (14, unit, loopback): an OS-unresolvable `acme.test` stub-pinned to 127.0.0.1 is fetched (proves pinning) with the `Host` header and User-Agent preserved and DNS resolved exactly once; redirect chain reported; loop detected; redirect cap; redirect to the metadata IP blocked **before any connection**; open redirect to a host resolving to metadata blocked (second hop never contacted); redirect to `file://` blocked; strict guard blocks hop 1 with zero requests reaching the server; backend refuses unvetted hosts and unix sockets; bounded pin LRU; fallback from a refusing pinned IP to the next. **Real TLS** (uvicorn + trustme CA): connection pinned to 127.0.0.1 verifies the cert for `secure.test` and sends `Host: secure.test`; the same server reached as `other.test` fails verification (proves SNI/verification use the hostname, not the IP); an untrusted CA fails. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 622 passed (104 s).
- Decisions/notes:
  - HTTP/2 off for now (crawling many origins gets little from multiplexing; fewer moving parts). Revisit only with evidence.
  - Only GET is exposed. Step 83's webhook delivery adds a POST path that does **not** follow redirects.
  - Step 27 builds on `stream_get` for streaming size caps, content-type checks and charset detection, and must map `RedirectLoopError` → `redirect_loop`, `TooManyRedirectsError` → `too_many_redirects`, `SsrfBlockedError` → `ssrf_blocked`, `HostResolutionError`/`httpx.NetworkError` → `network_error`, `httpx.TimeoutException` → `timeout`.
- Next: Step 27: HTTP fetcher

### Step 27: HTTP fetcher  (2026-10-23)
- Status: ✅ done
- Built (`rag_engine/crawling/fetchers/`):
  - `http.py`: `HttpFetcher(SafeHttpClient, config).fetch(url)` returns a `FetchResult` **only for 2xx**; everything else raises a typed `FetchError`.
    - `FetchResult` (frozen): requested/final URL, redirect chain, status, MIME, `content_kind`, `body` bytes, `text` (decoded HTML only), `encoding`, `bytes_downloaded`, `elapsed_ms`, kept headers (`content-language`, `link`, `x-robots-tag`, `last-modified`, `etag`), `rendered_with=http`.
    - Content type is decided **from headers before reading the body** (unsupported → no download).
    - A declared `Content-Length` over `CRAWL_MAX_PAGE_BYTES` fails immediately. Streaming is capped on **decoded** bytes, so gzip bombs are stopped too.
    - A wall-clock cap `CRAWL_FETCH_TIMEOUT_SECONDS` (new, default 60, validated ≥ the read timeout) covers the whole fetch incl. redirects, which stops servers dripping bytes under the read timeout.
    - `Accept` advertises the supported types. `parse_retry_after` handles delta-seconds and HTTP dates.
  - `content.py`:
    - `MIME_KINDS` allowlist: HTML/XHTML, PDF, DOCX, PPTX, XLSX.
    - `kind_for(mime, url)`: for `application/octet-stream` or a missing type, the URL extension decides (Step 38 confirms by magic bytes); an explicit wrong type wins over the extension.
    - `parse_content_type`.
    - `decode_html`: header charset → BOM (beats a missing/Unicode header) → `<meta charset>`/`http-equiv` in the first 4 KB → `charset-normalizer` detection → UTF-8 with replacement.
  - `errors.py`: each error carries its `PageReason`, so Step 28 classifies retries in one place:
    - `NetworkFetchError` (connection/DNS/TLS/protocol → `network_error`)
    - `FetchTimeoutError` (`timeout`)
    - `SsrfFetchError` (`ssrf_blocked`, detail = the SSRF rejection)
    - `RedirectLoopFetchError`, `TooManyRedirectsFetchError`
    - `TooLargeError` (detail `content_length` / `streamed`)
    - `UnsupportedContentTypeError` (`unsupported_content_type`, a **skip**)
    - `HttpStatusError` (status + `retry_after`; reason 429 → `rate_limited`, 5xx → `http_server_error`, else `http_client_error`)
- Fixture site: unlinked `/gzip-page`, `/gzip-bomb` (~25 KB → 25 MiB), `/files/octet/…pdf` and `/files/untyped/…pdf`, `/bom-page`, `/drip` (bytes every 0.2 s forever).
- Deps: runtime `charset-normalizer`.
- Tests: `TestHttpFetcher_S27` (40, unit, fixture site): HTML result shape; redirect chain; 6 document kinds incl. octet-stream and untyped PDF; JSON rejected without download; declared 25 MiB fails in <1 s; streamed cap; gzip decoded; gzip bomb capped; latin-1 via meta; UTF-8 BOM stripped; 404/500/429 (+`Retry-After` 2 s); read timeout; drip stopped by the total timeout (2.5–5 s for a 3 s cap); redirect loop; metadata redirect → `ssrf_blocked`; too many redirects; connection refused and DNS failure → `network_error`; `parse_content_type`, `kind_for` (6 cases), `decode_html` (incl. cp1251 detection), `Retry-After` parsing. Settings test for the new timeout ordering. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 663 passed.
- Decisions/notes:
  - Bodies are held in memory up to `CRAWL_MAX_PAGE_BYTES` (20 MiB). With `CRAWL_CONCURRENCY`=8 that's ≤160 MiB per worker in the worst case, so keep these two settings in mind together.
  - `text/plain`, JSON, XML, images etc. are skips (`unsupported_content_type`), not failures. Sitemaps (XML) are fetched by Step 31 through its own path.
- Next: Step 28: Retry & failure policy

### Step 28: Retry & failure policy  (2026-11-06)
- Status: ✅ done
- Built: `rag_engine/crawling/retry.py` (exported from `rag_engine.crawling`)
  - `classify(FetchError) -> Classification{kind, reason, detail, retry_after}`, with `FailureKind` = `retryable` / `permanent` / `skip`:
    - **retryable**: timeouts (connect/read/total), network errors (refused/reset, DNS timeout or `EAI_AGAIN`), 5xx except 501/505, 408/425/429. `Retry-After` is kept only for retryable statuses.
    - **permanent**: other 4xx, 501/505, SSRF-blocked, too large, redirect loop, too many redirects. Also two network errors that a retry can't fix, found by walking the exception's `__cause__` chain: a bad TLS certificate (`ssl.SSLCertVerificationError` → detail `tls_certificate`) and a host that doesn't exist (`gaierror` `EAI_NONAME`/`EAI_NODATA` → detail `dns_nxdomain`). Both were checked against real chains (trustme server, stub resolver). The `reason` stays `network_error`.
    - **skip**: unsupported content type (page status `skipped`, not `failed`).
    - An unknown `FetchError` subclass is **permanent** (fail closed: don't hammer a site).
  - `RetryPolicy(max_attempts, base_delay, max_delay, max_retry_after, rng)` with `from_settings()`:
    - `backoff(n)`: equal jitter. The delay is uniform in [d/2, d] with d = min(max, base·2^(n-1)), so a struggling site always gets a real pause.
    - `decide(error, attempts) -> RetryDecision{retry, attempts, reason, detail, delay_seconds, final_status}`. `Retry-After` is a **floor** on the delay. If `Retry-After` exceeds `CRAWL_RETRY_AFTER_MAX_SECONDS`, the page fails at once (detail says so) rather than waiting that long or retrying early. When the attempt limit is hit, the detail gets "gave up after N attempts".
  - `call_with_retries(operation, policy, sleep=asyncio.sleep)`: retries only `FetchError`. Anything else (bugs, cancellation) propagates at once. Giving up raises `RetriesExhaustedError{decision, error}`, which carries everything a `Page` row needs (status, reason, detail, attempts).
- Settings: `CRAWL_MAX_RETRIES` (existing, 3) is the number of retries, so the per-URL attempt limit is 4. New: `CRAWL_RETRY_BASE_DELAY_SECONDS` (1), `CRAWL_RETRY_MAX_DELAY_SECONDS` (30, validated ≥ base), `CRAWL_RETRY_AFTER_MAX_SECONDS` (120). `.env.example` updated.
- Tests: `TestRetryPolicy_S28` (59, unit):
  - 23-case classification matrix; `Retry-After` kept only when retryable; TLS-cert and NXDOMAIN chains permanent; `EAI_AGAIN`/DNS timeout/reset retryable; a cause cycle doesn't hang.
  - Backoff bounds per attempt (200 draws each), cap, jitter varies, seeded determinism.
  - Decisions: retry under the limit, give up at the limit, single-attempt policy, permanent and skip outcomes, `Retry-After` floor, backoff beating a shorter `Retry-After`, oversized `Retry-After` gives up, invalid policies rejected, `from_settings`.
  - Against the fixture site through the real fetcher (fake sleep):
    - `/flaky`: 3 requests, 2 sleeps ≥ 1 s
    - `/error`: exactly 4 requests, increasing delays
    - `/missing.html`, `/redirect-metadata`, `/api/widgets`: 1 request, no sleep
    - `/rate-limited`: 4 requests, all sleeps ≥ 2 s; with a 1 s cap, gives up after 1 request
    - connection refused: retried
    - stub-resolver NXDOMAIN: 1 attempt; `EAI_AGAIN`: 4 attempts
  - Non-fetch errors are not retried; the real `asyncio.sleep` default path works. New settings-validator case in `TestSettings_S3`. Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 723 passed (161 s).
- Decisions/notes:
  - `Page.attempts` should be `RetryDecision.attempts`.
  - **Step 50 (engine)** should call `policy.decide()` itself. On `retry`, it re-queues the URL in the frontier with a not-before time (Step 48 needs a `retry_at` field) instead of sleeping inside a concurrency slot. Otherwise it records `final_status`/`reason`/`detail`.
  - Steps 29/31 (robots.txt, sitemaps) can use `call_with_retries` inline.
  - All timeouts are retryable as the plan says, including the total-timeout "drip" case. Step 51's job wall-clock cap bounds the worst case (4 × `CRAWL_FETCH_TIMEOUT_SECONDS` per URL).
  - The rate-limit step (30) should also treat a 429's `Retry-After` as a per-domain signal, not just per URL.
- Next: Step 29: robots.txt

### Step 29: robots.txt  (2026-11-06)
- Status: ✅ done
- Built:
  - `rag_engine/crawling/robots.py` (exported from `rag_engine.crawling`):
    - `RobotsCache(fetcher, store, config, policy, clock)`: `rules_for(url) -> RobotsRules` and `can_fetch(url)`.
      - Two cache levels: the shared `RobotsStore` (Redis), plus an in-process LRU (1024 origins) of **parsed** rules, so per-URL checks neither hit Redis nor re-parse.
      - **Single flight** per origin inside a process. Waiters share one task under `asyncio.shield`, so a cancelled caller doesn't cancel the fetch the others are waiting on.
    - `RobotsRules`: `status`, `can_fetch(url)` (protego), `crawl_delay` (the larger of `Crawl-delay` and `Request-rate` seconds/requests, for our group), `sitemaps`, `detail`. `/robots.txt` itself is always allowed (RFC 9309 §2.2.2).
    - Status semantics (RFC 9309 §2.3.1), after Step 28's `call_with_retries`:
      - 2xx → `ok`, the parsed body. Only the first `ROBOTS_MAX_BYTES` = 500 KiB is read; the rest is truncated (`detail="truncated"`), not failed.
      - 4xx except 429, and redirect loop / too many redirects / SSRF-blocked redirect → `unavailable` (allow all).
      - 5xx, 429, timeout, network error → `unreachable` (**complete disallow**). If the last good copy is at most `ROBOTS_STALE_MAX_SECONDS` old, it becomes `stale` and those rules are used instead. Unreachable/stale verdicts are cached for only `ROBOTS_UNREACHABLE_CACHE_TTL_SECONDS`.
    - Body decoded as UTF-8 with replacement, BOM stripped.
    - `robots_token(user_agent)`: rules match on the UA **product token** (`RagEngineBot`), not the full UA string. Protego substring-matches, so `User-agent: github` would otherwise hit our UA because of its URL.
    - `origin_of(url)`: normalized `scheme://host[:port]` (default port dropped, IDNA).
    - `RedisRobotsStore`: `robots:<digest(origin)>` holds the current verdict as JSON with the raw body, using `PX` = verdict TTL. `robots:<digest>:last_good` keeps the last 2xx copy for `ROBOTS_STALE_MAX_SECONDS`. Unreadable entries are ignored (refetch). Shared across tenants: robots.txt is public, and its content comes from the site itself.
    - `RawFetcher` / `RobotsStore` protocols for the fetcher and store dependencies.
  - `HttpFetcher` refactor (`fetchers/http.py`):
    - The error translation + total timeout became `_fetch_errors()` (an async context manager) and the capped read became `_read_capped()`, so `fetch()` is unchanged in behaviour.
    - New `fetch_raw(url, accept, max_bytes, truncate=False) -> RawResponse`: any 2xx body, **no content-type allowlist**. Over `max_bytes` it fails (`TooLargeError`) or, with `truncate=True`, is cut. Step 31 sitemaps should use it.
- Settings: `ROBOTS_CACHE_TTL_SECONDS` now `le=86400` (RFC 9309 §2.4). New: `ROBOTS_UNREACHABLE_CACHE_TTL_SECONDS` (300, validated ≤ the cache TTL) and `ROBOTS_STALE_MAX_SECONDS` (7 days). `.env.example` updated.
- Deps: runtime `protego>=0.7.0`. Test fake: `tests/support/fakes.py::InMemoryRobotsStore`.
- Tests:
  - `TestRobotsTxt_S29` (59, unit; scripted fetcher + in-memory store, plus the fixture site):
    - Origins incl. IDN/IPv6, UA token.
    - Group selection (our group replaces `*`), longest-match Allow, `*`/`$` patterns, `github` group not matching, `/robots.txt` allowed under `Disallow: /`, crawl-delay/request-rate matrix, sitemaps.
    - Status semantics: 6 4xx codes → unavailable, not retried; redirect/SSRF → unavailable; 500/503/429/timeout/network → unreachable after 3 attempts with the short TTL; 503→200 retry; stale fallback (old rules apply); too-old copy not used; truncated and BOM/invalid UTF-8.
    - Caching: one fetch per origin, origins kept apart, TTL expiry refetch, unreachable rechecked sooner, store hit with no fetch, expired store entry refetched, 20 concurrent lookups → 1 fetch, cancelled caller, bounded local LRU falls back to the store.
    - Real fetcher: fixture site rules / crawl-delay 0.1 / sitemap with one request; connection refused → unreachable.
  - `TestFetchRaw_S29` (9, unit, fixture site): text/plain; JSON allowed; truncation; `TooLargeError`; gzip bomb truncated on decoded bytes; 404/loop/SSRF errors; total timeout.
  - `TestRobotsRedisStore_S29` (5, integration): round trip + PTTL; separate last-good key; corrupt entries ignored; two workers share one fetch; deleted entry refetched.
  - 2 new settings-validator cases.
  - Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 798 passed (99 s).
- Decisions/notes:
  - **Step 30 (rate limiter)** gets `RobotsRules.crawl_delay` raw, with no clamp. A hostile `Crawl-delay: 86400` must be handled there or by Step 51's job time cap (e.g. clamp, plus log or report it).
  - **Step 31 (sitemaps)**: take `RobotsRules.sitemaps` and fetch through `fetch_raw` with `call_with_retries`.
  - **Step 50 (engine)**: check `rules_for(url)` per URL (cheap after the first). An `unreachable` seed origin skips every page as `robots_disallowed`; the job summary should surface `rules.status`/`detail` so users see "robots.txt unreachable", not a silent empty crawl.
  - Cross-worker duplicate fetches are possible when two workers miss at the same instant (no Redis lock). That's harmless: at most one extra robots.txt request per worker.
- Next: Step 30: Per-domain rate limiter

### Step 30: Per-domain rate limiter  (2026-11-06)
- Status: ✅ done
- Built: `rag_engine/crawling/rate_limit.py` (exported from `rag_engine.crawling`)
  - `DomainRateLimiter(redis, keys, config, sleep)`: a **GCRA token bucket** in one Lua script on the Redis clock. The state is a hash `domain_rl:<digest(host)>` = `{tat, paused_until}` in **integer microseconds**, written with `string.format('%d')` (Lua's `tostring` would turn 16-digit values into `1.7e+15`). The key expires once the bucket is idle.
    - `reserve(url, crawl_delay=None, max_wait=None) -> Reservation{host, granted, wait_seconds, start_at, interval_seconds, burst}` **books** the next slot in a single round-trip. Concurrent callers across workers get distinct, evenly spaced start times (FIFO, no polling or thundering herd). If the wait would exceed `max_wait`, nothing is booked and `granted=False`.
    - `acquire(...)`: reserve, then sleep until the slot. Past `max_wait` it raises `RateLimitWaitExceededError` (with the reservation). A caller cancelled mid-sleep leaves its slot unused, which costs one interval at most.
    - `interval_for(crawl_delay)` = max(1/`CRAWL_PER_DOMAIN_RPS`, min(crawl_delay, `CRAWL_MAX_CRAWL_DELAY_SECONDS`)). `burst_for()` is 1 whenever the site sets a crawl delay, else `CRAWL_PER_DOMAIN_BURST`.
    - `pause(url, seconds)`: holds **every worker** off a host (for 429/503 `Retry-After`). It never shortens an existing pause, is capped at `CRAWL_RETRY_AFTER_MAX_SECONDS`, and returns the remaining pause.
  - `rate_limit_host(url)`: the politeness key is the normalized **host** (IDNA, lower-case). Scheme and port are ignored, so http/https/:8080 of one host share a bucket. The key is not per tenant: two tenants crawling one site share its budget.
- Settings: `CRAWL_PER_DOMAIN_RPS` now `le=100`. New: `CRAWL_PER_DOMAIN_BURST` (1, 1..100) and `CRAWL_MAX_CRAWL_DELAY_SECONDS` (30), which clamps hostile `Crawl-delay` values, as flagged in Step 29. `.env.example` updated.
- Tests:
  - `TestDomainRateLimitRules_S30` (10, unit): host key incl. ports/IDN/IPv6; interval and burst matrix (RPS, shorter delay ignored but no bursts, longer delay wins, 86400 clamped to 30); negative `max_wait` rejected.
  - `TestDomainRateLimiter_S30` (14, integration, real Redis):
    - Spacing: first request immediate, then exactly one interval apart. **40 concurrent bookings from two independent connection pools ("workers") get 40 distinct slots spaced exactly 0.5 s**. `acquire` really waits (10 requests ≥ 0.4 s).
    - Burst of 3 then spaced; crawl delay overrides both RPS and burst; hosts independent, ports shared; an idle bucket refills.
    - `max_wait`: a refused reservation books nothing; `acquire` raises past `max_wait`.
    - Pause: holds the other worker and other requests queue behind it; never shortened; capped; zero is a no-op.
    - Redis state: key TTL ≤ interval with only the `tat` field; microsecond state stored as exact digits (333 333 µs spacing).
  - 2 new settings-validator cases.
  - Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 824 passed (136 s).
- Decisions/notes:
  - **Step 50 (engine)** order per URL: scope → `robots.rules_for(url)` → `limiter.acquire(url, crawl_delay=rules.crawl_delay, max_wait=…)` → fetch. With a `max_wait` (e.g. a few seconds), a slow host can't pin a concurrency slot: on `RateLimitWaitExceededError`, re-queue the URL with `retry_at = reservation.start_at`. On an `HttpStatusError` with `retry_after`, call `limiter.pause(url, retry_after)` as well as the Step 28 per-URL decision.
  - Retries (Step 28) must also go through `acquire`, so retry traffic counts against the host budget.
  - Robots.txt and sitemap fetches are **not** rate-limited for now. They are one request per origin per TTL; Step 31 may put sitemap fetches through the limiter if a site has many child sitemaps.
- Next: Step 31: Sitemap discovery

### Step 31: Sitemap discovery  (2026-11-06)
- Status: ✅ done
- Built: `rag_engine/crawling/sitemap.py` (exported from `rag_engine.crawling`)
  - `SitemapDiscoverer(fetcher, robots=None, limiter=None, config, policy)`:
    - `discover(seed_url, scope, max_urls=None)`: the seed origin's robots.txt `Sitemap:` lines (if robots is `ok`/`stale`), then `/sitemap.xml`.
    - `read(sitemap_urls, scope, max_urls=None)`: explicitly given sitemaps.
    - Traversal: breadth-first over sitemap indexes, sitemaps de-duplicated by normalized URL (loops and self-references are ignored), fetched **sequentially** through `call_with_retries` and, when given, `DomainRateLimiter.acquire(crawl_delay=robots.crawl_delay)`.
    - Which sitemaps may be fetched:
      - robots-declared ones may be on any host (cross-submission).
      - Index children and the fallback must pass `scope.allows_host` (else `out_of_scope_host`) and that host's robots.txt (else `robots_disallowed`, not fetched).
    - The guessed `/sitemap.xml` is silent when it returns 4xx or a soft-404 page. A 5xx there is reported.
  - Page URLs: normalized, de-duplicated, then `scope.check(url, depth=0)`. `<loc>` whitespace is stripped; non-crawlable locs are counted as `invalid`. robots.txt rules for pages are **not** applied here (the engine does that per URL).
  - `SitemapReport`: `entries` (`SitemapEntry{url, lastmod (aware UTC), sitemap}`), `urls`, `sitemaps_read`, `failures` (`SitemapFailure{sitemap, problem, detail}`), `out_of_scope`, `invalid`, `duplicates`, `caps_hit` (`max_urls` / `max_files` / `entries_per_file`). A broken sitemap is reported and skipped; discovery carries on.
  - `SitemapProblem`: `fetch_failed` (detail `"<PageReason>: <detail>"`), `too_large`, `invalid_xml`, `forbidden_xml`, `unsupported_format`, `too_deep`, `out_of_scope_host`, `robots_disallowed`.
  - `parse_sitemap(bytes) -> ParsedSitemap{kind, items, capped}`:
    - `urlset` / `sitemapindex` XML is parsed **incrementally** with defusedxml `iterparse(forbid_dtd=True)`, so DTDs and entities are refused (no billion laughs, no XXE). The root is cleared per entry, so 50k-entry files stay flat in memory.
    - Namespace-agnostic (missing or wrong `xmlns` is fine). BOM tolerated. Plain-text URL-list sitemaps are supported.
    - RSS/Atom/HTML → `unsupported_format`. Capped at `MAX_ENTRIES_PER_FILE` = 50 000 (protocol).
  - `inflate(body, max_bytes)`: gzip detected by **magic bytes**, since `.gz` sitemaps are usually served as `application/gzip`, not `Content-Encoding`. Inflated through `zlib.decompressobj(max_length)`, so a bomb never allocates more than the cap. A **truncated stream** raises `invalid_xml`: zlib returns the partial data without error, and a test caught this.
  - `parse_lastmod`: W3C datetime/date → aware UTC; garbage → `None`.
- Refactor: the `RawFetcher` protocol moved from `robots.py` to `fetchers/http.py` (next to `RawResponse`). Re-exported unchanged.
- Settings: `SITEMAP_MAX_BYTES` (50 MiB, applies to the download and to the inflated body), `SITEMAP_MAX_FILES` (50, ≤1000), `SITEMAP_MAX_DEPTH` (3, index nesting; the protocol allows 1, real sites nest). `.env.example` updated.
- Deps: runtime `defusedxml`.
- Fixture site: unlinked `/sitemaps/<name>` edge cases: `loop-index.xml`, `mixed.xml` (no namespace, lastmod variants, duplicate/invalid/missing locs) and `mixed.xml.gz` (`application/x-gzip`), `nested-1..3.xml`, `foreign-index.xml`, `urls.txt`, `feed.xml`, `broken.xml`, `laughs.xml`, `soft-404.html`, `bomb.xml.gz` (4 MiB inflated), `big.xml?n=N`.
- Tests: `TestSitemapDiscovery_S31` (48, unit):
  - Parsing: lastmod matrix; urlset/index; optional namespace + BOM; text sitemap; 8 unusable documents (rss, atom, html, words, truncated, garbage, entities, XXE); per-file cap; inflate passthrough, bomb, garbage and truncated gzip.
  - Fixture site:
    - Full discovery: the 8 expected URLs, the 3 sitemaps read in order incl. gzip, 1 offsite out of scope, no failure for the absent fallback.
    - Caps: `path_prefix` scope; `max_urls` stops before fetching the next sitemap; `max_files`; per-file cap at 50 000 on a 50 100-URL file.
    - No robots → fallback only; loop / duplicates / invalid with lastmods; x-gzip; nesting depth cap; foreign children (offsite, other host, robots-disallowed child never requested).
    - Broken sitemaps: 8 kinds reported while the next sitemap is still read. `/error` retried per the policy with the reason in the detail. A sitemap listed twice is fetched once.
    - Every sitemap fetch goes through the limiter with the robots crawl delay.
  - Scripted fetcher: fallback soft-404 silent vs 503 reported; fallback urlset used.
- **Fix to a Step 30 test:** `test_concurrent_workers_get_evenly_spaced_slots` failed once under full-suite load. It asserted the last booking's wait as ≈19.5 s ± 0.2, but waits are measured from each booking's own arrival. I removed that assertion and widened the interval to 2 s so all 40 bookings land inside one slot. It still asserts exact spacing and the total span.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 872 passed (124 s).
- Decisions/notes:
  - **Step 48/50**: seed the frontier with `report.urls` as sitemap-priority entries at depth 0 (scope was checked with `depth=0`). `lastmod` is available for later incremental crawls. Store `report.failures`/`caps_hit` on the job for the per-page/job report (Step 53).
  - Sitemap bodies are held in memory up to `SITEMAP_MAX_BYTES` (50 MiB) one at a time. Inflated output is the same order, since `iterparse` reads from an in-memory buffer.
  - Discovery is sequential by design (politeness, simplicity). A site with 50 child sitemaps at a 0.5 s interval takes ~25 s.
- Next: Step 32: Link extraction

### Step 32: Link extraction  (2026-11-06)
- Status: ✅ done
- Built: `rag_engine/crawling/links.py` (exported from `rag_engine.crawling`)
  - `LinkExtractor(config).extract(html, page_url, *, scope, depth, headers=None) -> PageLinks`:
    - Sources: `<a href>`, `<area href>`, and the `<meta http-equiv="refresh">` target (`LinkSource`).
    - Relative URLs resolve against the **first `<base>` with an href** (itself resolved against the page), else the page's final URL. Then `normalize_url`, de-dupe (first anchor text kept), self-links (incl. `#fragment`) dropped.
    - `scope.check(url, depth + 1)`. Rejections are kept with their `ScopeDecision`, so `MAX_DEPTH` stays distinct from `OUT_OF_SCOPE`. Non-http(s) schemes (mailto/tel/javascript/data/ftp) are `rejected` as `unsupported_scheme` with the raw URL, matching the fixture's `OUT_OF_SCOPE` expectations. Hrefs that aren't URLs count as `invalid`.
    - **nofollow policy** (`CRAWL_RESPECT_NOFOLLOW`, default on):
      - `rel~=nofollow`, and page-wide nofollow from `<meta name="robots|ragenginebot">` or `X-Robots-Tag`, move in-scope links to `nofollow` instead of `follow`.
      - A URL linked both ways on a page is followed.
      - `noindex`/`none` are reported as `PageLinks.noindex` for extraction.
      - `ugc`/`sponsored` are hints, not directives: followed.
    - `parse_x_robots_tag(value, token)`: directives with no UA prefix or with our token. A `ua:` prefix scopes the following directives. Valued directives (`unavailable_after:` …) aren't mistaken for UA prefixes.
    - Canonical: `<link rel="canonical">` wins over a `Link: <…>; rel="canonical"` header (`canonical_from_link_header`). It is resolved against the base and normalized, but **not scope-checked**. `canonical_is_self` is provided.
    - `Link{url, text (collapsed, ≤200 chars), source, is_document}`: `is_document` comes from the URL extension (`EXTENSION_KINDS`). Document links are followed like pages; the content type decides later.
    - `CRAWL_MAX_LINKS_PER_PAGE` (5000) caps the candidates; `capped` is flagged.
    - Parsing: lxml `document_fromstring` on **bytes** with an explicit UTF-8 parser. lxml rejects `str` that carries an XML encoding declaration (XHTML). Empty/garbage HTML gives an empty result, never an exception.
  - `filter_urls(urls, page_url, scope, depth)`: the same normalize/scope/dedupe for links found outside HTML (Step 41 PDF link annotations, etc.).
- **Bug fixes found while testing:**
  - `normalize_url(url, base=…)` (Step 23) leaked a raw `ValueError` from `urljoin` for malformed IPv6 netlocs (`https://[bad`). A page with such a `<base>`/canonical/href would have crashed extraction. It now raises `UrlNormalizationError`. The link classifier also guards its own `urljoin`/`urlsplit`.
  - `TestHttpFetcher_S27::test_dns_failure_is_a_network_error` used real DNS and failed once when the resolver was slower than the 3 s fetch cap (it got `timeout` instead of `network_error`). It now uses a stub resolver.
- Settings: `CRAWL_RESPECT_NOFOLLOW` (true), `CRAWL_MAX_LINKS_PER_PAGE` (5000, ≤100 000). `.env.example` updated.
- Deps: runtime `lxml` (was transitive via python-docx/pptx; Step 39 trafilatura needs it too). Dev `types-lxml` (ty can't see the compiled `lxml.etree` otherwise).
- Tests: `TestLinkExtraction_S32` (62, unit):
  - Resolution: relative/`..`/protocol-relative/query/fragment resolution; normalization + dedupe; page URL normalized.
  - `<base>`: 4 cases (absolute path, relative, first href wins, empty) and a base on another host moving links out of scope.
  - Scope and URL validity: depth+1 with reasons; 5 non-http schemes rejected, not invalid; invalid/empty/missing hrefs.
  - nofollow: rel nofollow incl. a multi-token rel; follow wins in both orders; policy off; 7 meta robots cases; 9 `X-Robots-Tag` parses; header applied.
  - Canonical: relative + tracking-stripped, self-canonical, Link header and HTML precedence, 5 header parses.
  - Other sources and limits: 3 meta-refresh formats and the no-URL refresh; `<area>`; document flags; anchor text; link cap.
  - Robustness: 5 malformed inputs; XHTML with an encoding declaration; malformed base/canonical ignored (regression); `normalize_url` raises its own error (regression); `filter_urls`.
  - Fixture site: home page (pages, the `?ref=home` variant, 7 document links, only `nofollow-target` held back, all 7 rejections documented `OUT_OF_SCOPE` in `SITE_PAGES`, `#downloads` dropped); `reference.html`'s relative `deep/level4.html`; `canonical-variant.html` → getting-started.
  - Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 934 passed (124 s).
- Decisions/notes:
  - **Step 50 (engine)**: enqueue `follow` at `depth + 1`. Record `nofollow` URLs as `skipped/nofollow` **only if no followable link reaches them by the end of the crawl** (the frontier's unique index makes "followed later" win). Record `rejected` per `decision.reason/detail` if the job wants them in the report (they can be numerous: consider counts only).
  - **Step 39 (HTML extractor)** should reuse `PageLinks.noindex` (skip indexing, still follow links unless nofollow), and Step 46 uses `canonical_url` for `canonical_elsewhere`.
  - The JS-rendered DOM (Step 36) goes through the same `extract()` with the rendered HTML.
- Next: Step 33: JS-shell detector

### Step 33: JS-shell detector  (2026-11-06)
- Status: ✅ done
- Built: `rag_engine/crawling/js_shell.py` (exported from `rag_engine.crawling`)
  - `JsShellDetector(thresholds).detect(html) -> JsShellVerdict{needs_rendering, score, signals, text_chars, script_count, script_bytes, mount}`. It works on the raw HTML only (lxml, same bytes+UTF-8 parsing as Step 32), with no network.
  - Weighted signals (`JsSignal`), render when score ≥ `threshold` (3):

    | Signal | Weight | Condition |
    |---|---|---|
    | `very_low_text` | +3 | < 50 visible chars |
    | `low_text` | +2 | < 250 visible chars |
    | `empty_mount` | +3 | an SPA mount has no visible text: `#root #app #__next #__nuxt #___gatsby #svelte …`, `<app-root>`, `[ng-app] [data-reactroot] [data-v-app]`; a spinner or only `<style>`/`<svg>` inside still counts as empty |
    | `noscript_hint` | +2 | `<noscript>` text like "enable/requires/turn on JavaScript", "JavaScript is disabled" |
    | `template_markers` | +2 | ≥ 3 unrendered `{{ … }}` in the visible text |
    | `ajax_fragment_meta` | +2 | `<meta name="fragment" content="!">` |
    | `script_heavy` | +1 | inline code bytes + 5000 per external script ≥ 10 × visible text |
    | `rich_text` | −2 | ≥ 2000 visible chars |

  - **Guards towards not rendering** (Playwright is expensive): a page with **no executable script never renders**, whatever the score. Rich text offsets a stray empty widget mount.
  - Visible text excludes `script/style/noscript/template/svg/head`. JSON data blocks (`application/ld+json`, `application/json`, `__NEXT_DATA__`, text templates) are not counted as scripts.
  - `JsShellThresholds` (frozen dataclass) holds the tunables. They aren't in settings yet; Step 36 can expose them if tuning is needed.
- Tests: `TestJsShellDetector_S33` (32, unit):
  - Shells: CRA shell (noscript, `#root`, text 0); Next.js client-only (`__NEXT_DATA__` not counted as code); Angular `<app-root>`; 4 empty-mount variants (spinner, style+svg, Gatsby, `data-reactroot`); AngularJS `{{ }}` templates; AJAX fragment meta; "Loading…" placeholder.
  - Not shells: SSR Next.js with content; article with analytics scripts; script-less short page (score ≥ 3 but never rendered); short page with a small script; rich article with a stray `#app`; JSON-LD-only page.
  - Mechanics: non-visible text excluded; 7 noscript texts (5 hints, pixel and video fallbacks not); 5 empty/garbage inputs; configurable threshold.
  - Fixture site: **every static fixture page is judged not-a-shell except `/app/`**, which is cross-checked against `SITE_PAGES` `JS_RENDERED`; `/app/` fetched over HTTP → render, mount `#root`.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 966 passed (105 s).
- Decisions/notes:
  - **Step 36 (auto fetcher)**: run `detect(result.text)` only for `ContentKind.HTML`. If `needs_rendering`, re-render and record `rendered_with=browser`. Log `signals` on the page (useful when tuning). If the rendered DOM still has `very_low_text`, keep the better of the two results rather than failing.
  - **False negatives** are possible for SPAs that server-render a large shell (nav/footer ≥ 250 chars) around a mount with a non-standard id. A per-job "always render" switch (Step 36/84) is the escape hatch, not more heuristics.
- Next: Step 34: Playwright browser pool

### Step 34: Playwright browser pool  (2026-11-06)
- Status: ✅ done
- Built: `rag_engine/crawling/fetchers/browser.py` (exported from `rag_engine.crawling.fetchers`)
  - `BrowserPool(config)`: **one Chromium per worker process**, launched **lazily** on first use (`browser()` / `start()`), `stop()` idempotent.
    - **Fork-safe**: the pool records the pid. A pool inherited through `fork()` (Celery prefork) resets and launches its own browser, never touching the parent's Playwright objects.
    - **Bound to one event loop**: Playwright objects belong to the loop that created them, so use from another loop raises `BrowserPoolError` instead of hanging. ⇒ **Step 75 must run each worker process on one persistent `asyncio.Runner`** (not `asyncio.run` per task).
    - **Self-healing**: a disconnected or crashed browser is relaunched on next use (`generation` counts launches). The browser is **recycled** after `CRAWL_BROWSER_RECYCLE_AFTER_CONTEXTS` contexts, but only while no context is open.
    - `CRAWL_BROWSER_MAX_PAGES` (4) is a per-process semaphore on open pages.
    - Launch args `--disable-dev-shm-usage` etc. are container-friendly; `CRAWL_BROWSER_HEADLESS`.
  - `job_context(job_id=, request_guard=)` → `BrowserJobContext`: a **fresh context per job** (isolated cookies/storage/cache), our `CRAWL_USER_AGENT`, a 1280×900 viewport, `en-US`, HTTPS errors not ignored. Closed on exit, including when the job raises.
    - One `context.route("**/*")` handler sees **every** request (documents included). It aborts `CRAWL_BROWSER_BLOCKED_RESOURCES` (default `image/font/media`; validated `Literal` list, `document` not allowed) and then asks the optional `RequestGuard` (`async (Request) -> abort reason | None`). That hook is where **Step 35 plugs in the SSRF guard**. `ContextStats` counts requests and blocks by `resource:<type>` / `guard:<reason>`.
    - `page()`: a page with `CRAWL_RENDER_TIMEOUT_SECONDS` default/navigation timeouts, holding a page slot.
    - `render(url, wait_until="load") -> RenderedPage{requested_url, final_url, status_code, html, elapsed_ms}`, with an asyncio backstop over Playwright's timeouts. Failures raise `RenderError` (`FetchError`, reason `render_failed`, detail `timeout` or the net error). Step 36 adds networkidle caps and auto-scroll on top.
  - `process_browser_pool()` / `shutdown_process_browser_pool()`: the per-process singleton and the shutdown hook, for Step 75's `worker_process_shutdown` signal. Init needs no hook (lazy launch happens after the fork).
  - ⚠️ Until Step 35, browser requests are **not SSRF-guarded**: only point the pool at trusted URLs (tests use the loopback fixture site).
- Settings: `CRAWL_BROWSER_MAX_PAGES` (4, ≤64), `CRAWL_BROWSER_BLOCKED_RESOURCES` (`["image","font","media"]`, type `BrowserResourceType` Literal list), `CRAWL_BROWSER_RECYCLE_AFTER_CONTEXTS` (100), `CRAWL_BROWSER_HEADLESS` (true). `.env.example` updated.
- Deps: runtime `playwright` (1.64.1 → Chromium build 1248). Local dev setup: `uv run playwright install chromium`. The cached builds 1228/1254 didn't match, so 1248 was downloaded into `~/.cache/ms-playwright`.
- Tests: new `tests/browser/` directory, auto-marked `browser` (`_DIRECTORY_MARKERS`). It is **skipped when Chromium can't launch** (session probe in `tests/browser/conftest.py`). `tests/README.md` updated.
  - Fixture site: unlinked `/browser/<name>` assets (`resources.html` with web font, img, video, stylesheet and script).
  - `TestBrowserPool_S34` (17):
    - Rendering: the fixture SPA renders (JS content + `/app/advanced` link); UA seen by the server.
    - Blocking: image/font/media blocked (server never hit) while CSS/JS load; configurable blocked types; the guard aborts requests, sees documents, and never sees already-blocked types.
    - Lifecycle: per-job cookie isolation (shared within a job); contexts/pages closed incl. on job exception; lazy + idempotent lifecycle; crashed browser relaunched; recycle waits for idle; page concurrency peak = limit.
    - Failures: page timeout (1 s) → `RenderError("timeout")`; connection refused → `RenderError`. Chromium refuses "unsafe" port 1 itself, so the test uses a freed port.
    - Binding: second event loop refused; fresh launch after a (simulated) fork; process singleton and shutdown.
- **Test-robustness fix (Step 30)**: `test_acquire_really_waits` failed once under heavy machine load (load average ~11 from other apps), because it asserted a minimum gap between *wake-up* times. It now asserts only load-proof facts: elapsed ≥ 9 intervals (sleeps never end early) and booked slots never closer than one interval.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 983 passed (344 s on a heavily loaded machine; the browser tests alone take ~53 s, and the suite without them took 105 s at Step 33). No leftover Chromium processes after the run.
- Next: Step 35: Browser SSRF enforcement

### Step 35: Browser SSRF enforcement  (2026-11-06)
- Status: ✅ done
- Design decision: **Chromium never opens a connection itself.** The plan text says "`context.route` validates every request host". But validating then `route.continue_()` leaves a DNS-rebinding window: Chromium re-resolves after our check. So the router **performs** every allowed request through the pinned client and `route.fulfill`s the response.
- Built:
  - `SafeHttpClient.send_once(method, url, headers=, content=)` (`security/safe_http.py`): one vetted + pinned hop, with redirects **returned, not followed**. `Accept-Encoding` is forced to `gzip, deflate` (what httpx decodes; browsers advertise br/zstd).
  - `fetchers/browser.py` router, per request:
    1. Blocked resource type → abort.
    2. Method not in `ALLOWED_METHODS` = GET/HEAD/POST/OPTIONS → abort `method:<M>`. POST stays allowed because GraphQL reads use it.
    3. Optional `request_guard`.
    4. `_fetch` via `send_once`, with the `CRAWL_MAX_PAGE_BYTES` cap on decoded bytes and the `CRAWL_FETCH_TIMEOUT_SECONDS` cap. Then `route.fulfill(status, headers, body)`. Content-encoding/length/hop-by-hop headers are dropped, and repeated `Set-Cookie` values are newline-joined (`_fulfill_headers`).
    - Failures abort with Chromium error codes and a `ContextStats.blocked` key: `ssrf:<rejection>` (blockedbyclient), `dns` (namenotresolved), `timeout`, `network`, `too_large`, `too_many_redirects`.
    - Request headers from the browser are forwarded (incl. `cookie`), minus host/hop-by-hop/content-length/accept-encoding/pseudo-headers.
  - **Redirects (found by the backstop test):** a fulfilled 3xx is followed by Chromium *outside* interception. The dead proxy caught it (`ERR_PROXY_CONNECTION_FAILED`). Now:
    - **Main-frame navigation:** the router parks the vetted `Location` and aborts. The new `BrowserJobContext.goto(page, url)` (used by `render`) re-navigates, so `page.url`/relative links/`location` stay right and each hop is vetted. Loops and `CRAWL_MAX_REDIRECTS` raise `RenderError`.
    - **Subresources and iframes:** followed server-side hop by hop, each vetted. 303 (and POST on 301/302) becomes GET. **`Cookie`/`Authorization` are dropped when a hop changes origin.**
  - **Backstop:** Chromium is launched with `--proxy-server=http://127.0.0.1:<port>` on a socket that is **bound but never listening** (connections refused instantly) plus `--proxy-bypass-list=<-loopback>`. Any traffic that escapes routing (preconnect, prefetch, a stray `continue_`) dies. Also `--dns-prefetch-disable` and `--force-webrtc-ip-handling-policy=disable_non_proxied_udp`.
  - Context options: `accept_downloads=False`, `service_workers="block"` (`register()` then resolves to `undefined` and the script is never fetched). `context.route_web_socket("**/*")` closes every WebSocket (1008) without connecting.
  - The pool owns a `SafeHttpClient` (created on launch, closed on `stop()`, dropped on fork) and the dead-proxy socket. `BrowserPool(config, ssrf_guard=...)` accepts a guard (stub resolvers in tests).
- Fixture site: `/browser/ssrf.html` probe page (metadata IP, metadata name, `evil.test`, same-origin GET/POST/PUT/DELETE, WebSocket, service worker → `data-*` results); `/browser/echo` (method/cookie/body size, all methods); `/browser/set-cookie` (two cookies); `/browser/sw.js`; `/browser/download.bin` (attachment).
- Tests:
  - `TestBrowserSsrf_S35` (14, browser; stub resolver with `site.test`, `evil.test` → metadata, and `rebind.test` → loopback then metadata):
    - The browser never resolves or connects itself: an OS-unknown `site.test` page and its subresources load only via our client, with the Host header preserved.
    - The strict guard blocks a loopback document with 0 server hits.
    - In-page probes: metadata IP/name/`evil.test` blocked, PUT/DELETE blocked (never reach the server), GET/POST ok, WebSocket closed, service worker blocked.
    - Redirects: every navigation hop vetted (chain followed; metadata hop blocked after hop 1 hit); navigation loop and redirect cap; subresource redirects followed and vetted.
    - **DNS rebinding**: the document is pinned to the first answer, and the rebound subresources are blocked with no connection.
    - Credentials not carried across origins (same-origin hop keeps the cookie).
    - Download refused; traffic outside routing dies at the dead proxy; cookies round-trip; gzip decoded; oversized response aborted; `_fulfill_headers`.
  - `TestSendOnce_S35` (3, unit): 301 returned not followed; method+body sent; hop vetted before connecting.
  - Step 34 browser tests now set `SSRF_ALLOW_PRIVATE_NETWORKS=True` (the fixture is loopback). The refused-connection test now asserts the forwarded `network` abort.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1000 passed (192 s). The browser suite went 53 s → 41 s for 31 tests. No leftover Chromium processes.
- Decisions/notes:
  - All browser traffic is buffered in Python per resource (≤ `CRAWL_MAX_PAGE_BYTES`). Fine for documents/scripts/XHR with images/fonts/media blocked. Streaming media would need a different path, and it's blocked anyway.
  - Main-frame redirects triggered by JS *after* load (`location.href = …` during the networkidle wait) are parked and not followed. Step 36 decides whether to follow them, and meta-refresh targets are already outlinks (Step 32).
  - Intermediate `Set-Cookie` headers on server-side-followed subresource redirects are not applied (rarely matters for crawling).
- Next: Step 36: Auto fetcher

### Step 36: Auto fetcher  (2026-11-06)
- Status: ✅ done
- Built:
  - `rag_engine/crawling/fetchers/auto.py`: `AutoFetcher(http, browser: BrowserJobContext | None, config, mode=, detector=)`; `fetch(url) -> FetchResult`.
    1. `HttpFetcher.fetch`. Its `FetchError`s propagate unchanged (Step 28 retries apply).
    2. Non-HTML, mode `never`, or no browser → the HTTP result. Without a browser, a warning is logged once at construction.
    3. HTML: the JS-shell detector runs and its signals land in **`FetchResult.js_signals`**. `auto` renders only flagged pages; `always` renders every HTML page.
    4. `BrowserJobContext.render(final_url, idle_timeout, scroll…)`, SSRF-enforced per Step 35.
    5. **Keep the better**:
       - `RenderError` → HTTP result with `render_note="render_failed: <detail>"`
       - non-2xx rendered document → `render_status_<code>`
       - rendered visible text < raw visible text → `render_not_better` (e.g. a script wiping the page)
       - otherwise the rendered HTML, with `rendered_with=browser`, `mime=text/html`, `encoding=utf-8`, `body=html bytes`, `bytes_downloaded` and `elapsed_ms` summed, and a JS redirect's final URL appended to `redirect_chain`
    - **A render failure never fails the page.**
  - `FetchResult` gained `js_signals: tuple[str, ...] = ()` and `render_note: str | None = None`.
  - `BrowserJobContext.render(url, wait_until="load", idle_timeout=None, scroll_seconds=0, scroll_max_steps=0)`:
    - After load, **network idle is waited for up to `idle_timeout`**. It's a cap, not an error: long-polling pages never idle (`RenderedPage.network_idle` = True/False/None).
    - Then **auto-scroll** one viewport per step (150 ms pause) until the page stops growing at the bottom, bounded by `scroll_max_steps` and `scroll_seconds`, scrolls back to top, and settles the network again (`scroll_steps`).
    - `page.content()` tolerates a JS navigation that replaced the document meanwhile. The whole render has an asyncio backstop.
- Settings:
  - `CRAWL_JS_RENDERING` (`JsRenderMode` auto/always/never in `config/enums.py`, default auto)
  - `CRAWL_RENDER_NETWORKIDLE_SECONDS` (5)
  - `CRAWL_RENDER_SCROLL_SECONDS` (5)
  - `CRAWL_RENDER_SCROLL_MAX_STEPS` (20, 0 = off)
  - `.env.example` updated
- Fixture site: `/browser/lazy.html` (+ `lazy-chunk`, IntersectionObserver at the bottom of a 4000 px page), `late.html` (content after load once `/slow?delay=1` answers), `polling.html` (fetch every 100 ms forever), `wipe.html` (script empties the body), `js-redirect.html` (`location.replace("/docs/")`).
- Tests: `TestAutoFetcher_S36` (16, browser):
  - auto: the SPA is rendered (signals, result shape); **links only in the rendered DOM (`/app/advanced`) are found** by the Step 32 extractor; a server-rendered page never touches the browser (0 browser requests, 1 server hit); a PDF is never rendered even in `always`; HTTP errors propagate with no render.
  - never: HTTP only, the detector isn't run. always: every HTML page rendered (2 server hits). No browser: HTTP result, signals still recorded.
  - Fallbacks: render failure (guard denies documents) → HTTP + `render_failed: …`; wipe page → `render_not_better`; non-2xx render → `render_status_404`.
  - JS redirect recorded in `final_url` + chain.
  - Settling: network idle waits for late content (absent with load only); idle cap ends a never-idle page in < 5 s; auto-scroll loads lazy content (chunk fetched once) and is absent without scrolling; scroll step cap stops early.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1016 passed (187 s; browser suite 47 tests, ~2 s Chromium launch per test on a loaded machine). No leftover Chromium processes.
- Decisions/notes:
  - **Step 50 (engine)**: one `BrowserPool.job_context(job_id=…)` per crawl job, wrapped in one `AutoFetcher`. Use `result.final_url` for scope re-checks and link extraction (a JS redirect may change it). Persist `rendered_with`, `js_signals` and `render_note` on the `Page` (the `Page` model has `rendered_with`; Step 47 should add the other two).
  - Rendering re-fetches the document in the browser (2 requests per rendered page). Both go through the per-domain rate limiter only if the engine acquires twice: **Step 50 should `acquire()` before the render too**, or count it via the browser guard hook.
  - A per-job render-mode override (`CrawlConfig.js_rendering`) belongs to Step 84's API; the `mode=` parameter is ready for it.
  - Browser tests launch Chromium per test (~2 s). If the suite grows, a module-scoped pool with a module-scoped event loop would cut this.
- Next: Step 37: Chromium in the worker image

### Step 37: Chromium in the worker image  (2026-11-06)
- Status: ✅ done
- Built:
  - `Dockerfile` (BuildKit, pinned `python:3.14-slim-bookworm` + `ghcr.io/astral-sh/uv:0.11.14`):
    - `base`: Python + uv; venv at `/app/.venv` on `PATH`.
    - `deps`: `uv sync --frozen --no-dev --no-install-project` with a uv cache mount and bind-mounted lock/pyproject, so the layer only changes with `uv.lock`.
    - **`worker`**:
      - copies the deps venv, then `playwright install --with-deps --only-shell chromium`. That layer depends only on the playwright version. Only the headless shell is installed, which is all `headless=True` needs; headed mode is unsupported in the image.
      - then installs the project non-editable (`uv sync --no-dev --no-editable`), so code changes don't re-download Chromium.
      - runs as non-root user `app` (uid 10001), with `PLAYWRIGHT_BROWSERS_PATH=/ms-playwright`.
      - `CMD` is a placeholder (smoke `--help`); **Step 75 sets the Celery worker command**, and **Step 95 adds the `api` stage** and hardening on the same `deps` stage.
    - Build ≈ 1m50 cold; image 1.57 GB, mostly Chromium's system libraries plus the Python deps. Step 95 can slim it.
  - `.dockerignore`: `.env*` (except `.env.example`), `.git`, `.venv`, caches, `tests`, `plan.md`… **Secrets never enter the build context.**
  - `rag_engine/crawling/fetchers/smoke.py`: `python -m rag_engine.crawling.fetchers.smoke URL [--mode auto|always|never] [--expect TEXT] [--allow-http]`. It runs the real fetch path (`SafeHttpClient` → `AutoFetcher` → SSRF-guarded `BrowserPool`), prints one JSON line (ok, final_url, status, rendered_with, js_signals, render_note, chromium version, html preview/size, expect_found), and exits 0 only if rendered by the browser (unless `--allow-http`) and the `--expect` text is present. It doubles as an ops self-check for a worker host.
- **Bug found by the container test: runtime deps declared as dev-only.**
  - `httpx` was in the dev group. `httpcore` (imported directly by `safe_http`) and `qdrant-client` (`infrastructure/qdrant.py`) came only transitively through dev tools (`testcontainers[qdrant]`).
  - A `--no-dev` install (the image) failed with `ModuleNotFoundError: httpx`.
  - Moved `httpx>=0.28.1`, `httpcore>=1.0.9` and `qdrant-client>=1.19.1` to runtime dependencies (locked versions unchanged).
  - Regression guard **`TestRuntimeDependencies_S37`** (unit, 2): every third-party import under `rag_engine/` must be provided by the `uv export --frozen --no-dev` closure, and dev tools (pytest/ruff/ty/testcontainers/trustme) must not be in it. Verified it fails on the pre-fix lock.
- Test infra:
  - New `tests/container/`, auto-marked **`container`** (registered in pyproject) and **skipped unless `--run-container`** (new conftest option), and skipped if no Docker daemon.
  - A session fixture builds `rag-engine-worker:test` once.
  - `tests/README.md` updated.
- Tests: `TestWorkerImage_S37` (4, container, `--network host` to reach the host's fixture site, dummy secrets via `-e`):
  - **The image renders the fixture SPA** (`--expect "Widget configurator"`, `rendered_with=browser`, `empty_mount` signal, Chromium version reported, 2 server hits = HTTP + browser via our client).
  - A server-rendered page needs no browser.
  - **SSRF enforced in the image**: default settings block loopback (`ssrf_blocked`, 0 server hits).
  - Runs as uid 10001, with no `/app/.env` and no `/app/tests`, and the headless shell present in `PLAYWRIGHT_BROWSERS_PATH`.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1018 passed, 4 skipped (container) in 150 s; `--run-container`: 4 passed (147 s incl. build).
- Decisions/notes:
  - The image has no `.env` and never will. Configuration is environment variables, which compose (Step 96) supplies.
  - Chromium runs with Playwright's default `--no-sandbox` (the container is the isolation boundary, as a non-root user). Step 95/98 may revisit with a seccomp profile if the sandbox is wanted.
- Next: Step 38: Extractor framework

### Step 38: Extractor framework  (2026-11-06)
- Status: ✅ done
- Built `rag_engine/extraction/`:
  - `base.py`:
    - `ExtractionInput{url (final), mime_type, body, text, encoding, headers}` with `from_fetch(FetchResult)`.
    - `ExtractedDocument{markdown, content_kind, extractor, title, lang, description, metadata, outlinks}`, frozen. Blank title/lang/description → None, lang lower-cased, metadata values must be scalars (`MetadataValue`), outlinks frozen to a tuple. Outlinks are raw absolute URLs; the engine scopes them with `LinkExtractor.filter_urls` (Step 32).
    - `BaseExtractor` (ABC): class attrs `name`, `kind: ContentKind`, optional extra `mime_types`/`extensions`, validated in `__init_subclass__`; `async extract(source)`; `run_sync(func, *args)` moves CPU-bound parsers to a thread.
  - `sniff.py` → `sniff(body) -> ContentKind | Sniffed | None`:
    - PDF: `%PDF-` in the first 1024 bytes.
    - OOXML: ZIP with `[Content_Types].xml` + a `word/`|`ppt/`|`xl/` part. Only the central directory is read, never decompressed, so zip bombs are free.
    - `Sniffed.OLE`: CFB container, i.e. legacy `.doc/.xls/.ppt` **or encrypted OOXML**.
    - `Sniffed.ZIP`: a non-Office zip.
    - HTML markers after BOM/whitespace/XML prolog/comments.
    - `looks_like_text(body)` is encoding-agnostic (no NUL, ≤5 % control bytes), so Latin-1 and Shift-JIS bodies are text.
  - `registry.py` → `ExtractorRegistry(extractors, config)`: `register` (one extractor per kind, else `ExtractorRegistrationError`; extractors may add MIME/extension labels for their kind), `kinds`, `detect_kind`, `resolve`, `extract`.
    - **Resolution (bytes beat labels, labels beat guesses):**
      1. magic-proven binary kind wins over any label (PDF served as `text/html`, PPTX as `application/zip`)
      2. HTML-looking bytes → HTML (a "PDF" link returning a login page)
      3. OLE where Office was expected by MIME or extension → `DocumentUnreadableError(detail="encrypted_or_legacy_office")`
      4. a `text/html` label (or an `.html`-family extension on a generic/missing type) with a text body → HTML; UTF-16 is trusted via the fetcher's detected encoding
      5. a body contradicting its declared binary type → `UnsupportedContentError(detail="not_<kind>")`
      6. anything else, or a kind with no registered extractor (`no_extractor:<kind>`) → `UnsupportedContentError`
    - `extract()` caps time with `EXTRACTION_TIMEOUT_SECONDS` (new setting, 120 s; detail `timeout`; a parser thread past the cap finishes unobserved). Typed errors pass through. **Any other exception becomes `ExtractionError`** (detail `"<Type>: <msg>"`, cause chained, logged with a traceback), so a malformed file never crashes the crawl.
  - `exceptions.py`: plain exceptions (page outcomes, not API errors), each with `reason: PageReason` and `page_status`:
    - `ExtractionError` → failed/`extraction_failed`
    - `UnsupportedContentError` → **skipped**/`unsupported_content_type`
    - `DocumentUnreadableError` → failed/`document_unreadable`
    - `EmptyContentError` → skipped/`empty_content`
- Tests: `TestExtractorFramework_S38` (59, unit):
  - Sniffing: the 6 real fixture documents (incl. scanned and encrypted PDFs, which keep `%PDF-`); 5 HTML variants; 9 other bodies (PDF with a junk prefix, OLE, plain zip, OOXML with an unknown part, minimal DOCX, broken zip, PNG, JSON, empty); 6 text/binary cases.
  - Resolution: 9 kind cases (incl. PDF-as-HTML, PPTX-as-zip, HTML-as-PDF, text-as-HTML, Latin-1, octet-stream + `.html`); UTF-16 HTML; 9 error cases (encrypted DOCX by MIME and XLSX by extension, legacy `.doc`, fake PDF, image, binary-as-HTML, zip, JSON, no content type).
  - Registry: resolve+extract; missing extractor; duplicate kind; extra labels (incl. an explicit wrong type beating the URL); typed errors pass through with reason/status; crash → `ExtractionError`; timeout.
  - Contract: unsupported is a skip; subclass declarations enforced; document normalization + scalar metadata; `from_fetch`; `run_sync` uses a worker thread.
- Also: `tests/container/test_worker_image.py` reformatted. Step 37's last edit had skipped `ruff format`; it was only the 3.14 unparenthesized `except` style.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1077 passed, 4 skipped (container) in 142 s.
- Decisions/notes:
  - The registry starts empty. **Steps 39 and 41–44 each add their extractor**, and a `default_registry()` factory listing them lands with the last one (or the engine builds it in Step 50).
  - Engine contract (Step 50/53): `registry.extract(ExtractionInput.from_fetch(result))`. On an `ExtractionError`, record `page_status`/`reason`/`detail`. On success, `Page.content_kind = document.content_kind`, which may differ from the header-based `FetchResult.content_kind`.
- Next: Step 39: HTML extractor

### Step 39: HTML extractor  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/extraction/html.py` → `HtmlExtractor` (`name="html"`, `kind=HTML`, exported from `rag_engine.extraction`). Parsing runs in a worker thread (`run_sync`).
  1. `<noscript>`/`<template>` are dropped first. A rendered page never shows them, and in an unrendered JS shell the noscript text ("enable JavaScript…") would otherwise be the "content". Such a shell now ends as `EmptyContentError` (skipped, `empty_content`).
  2. **trafilatura 2.3.1** main content → Markdown (`include_tables`, `include_formatting`, `favor_recall`, no links/images/comments). A module-level config sets:
     - `EXTRACTION_TIMEOUT=0`: its signal-based timeout only works in the main thread; the registry's cap covers this.
     - `MIN_EXTRACTED_SIZE=40`: short docs pages are content.
     - `EXTENSIVE_DATE_SEARCH=off`: it had "found" 2026-01-01 from a footer `© 2026`.
  3. **markdownify fallback** when trafilatura returns nothing or raises: the `<body>` minus chrome (`nav/header/footer/aside/form/script/style/iframe/svg…`, comments) with ATX headings, `-` bullets and code languages from classes. `metadata["extraction"]` says which path ran.
  4. Post-processing:
     - **Restore the dropped `<h1>`**: trafilatura sometimes drops a page's only h1. Headings drive chunking, so it's put back on top when missing.
     - **Code fences get their language back** from the source `<pre><code class="language-…|lang-…|highlight-…">`, matched by content. trafilatura loses it.
     - Intra-word `\_` escapes are undone (`listen_port`, not `listen\_port`).
     - Blank lines are normalized around fences and headings.
  - Metadata:
    - `title`: trafilatura's (cleaned `<title>` without the site suffix; it prefers the `<h1>` when there is one), else `<title>`, else the first `<h1>`.
    - `description`: meta description → `og:description` → trafilatura.
    - `lang`: `<html lang>`/`xml:lang` (the HTML parser keeps the literal attribute name) → `Content-Language` header → `http-equiv` meta; primary subtag only, validated.
    - `metadata`: `{extraction, author, site_name, published}`, from markup only. trafilatura's site-name guess from the URL host is discarded.
  - Body decoding when `source.text` is missing (e.g. HTML sniffed from a mislabelled response) reuses `decode_html`.
  - **Outlinks stay empty for HTML**: links need scope and nofollow rules, which `LinkExtractor` (Step 32) applies to the same HTML.
- Deps: runtime `trafilatura` (2.3.1) and `markdownify`. Pytest filter for a `codecs.open` DeprecationWarning raised inside third-party `tld` (trafilatura → courlan).
- Tests: `TestHtmlExtractor_S39` (29, unit):
  - Main content: structure (h1/h2, ordered list, ```python fence restored, table with an unescaped underscore header); nav/footer dropped.
  - Fixture site through the real fetcher: `configuration.html` (title/description/lang, table row, ```toml, no outlinks); `getting-started.html` (inline code, bold, h3, two ```bash fences); `latin1.html` (Café, `fr`, accents).
  - Edge content: short page kept; rendered SPA DOM keeps its h1; unrendered JS shell → `EmptyContentError`; 3 no-text inputs → `EmptyContentError`.
  - Fallback: trafilatura returns nothing → markdownify (headings, code language, table, nav/footer/script/form gone); trafilatura crash → markdownify.
  - Metadata: 3 title cases (cleaned `<title>`, h1 preferred, h1 without a title); og:description fallback; 8 language cases (region stripped, `pt_BR`, `xml:lang`, header, http-equiv, markup beats header, invalid/missing → None); author/date/site name from markup only, with no date guessed from body text and no URL-host site name.
  - Inputs: Latin-1 body decoded when text is missing; XHTML with an XML encoding declaration; dispatch through `ExtractorRegistry`.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1106 passed, 4 skipped (container) in 211 s.
- Decisions/notes:
  - Repeated chrome that trafilatura keeps (the fixture's cookie banner on some pages) is **Step 40's** job (cross-page block frequency), not per-page heuristics.
  - Link-list sections lose their anchor text (`include_links=False`), which can leave empty headings (fixture home page). **Step 45**'s empty-section pruning removes those.
  - `PageLinks.noindex` (Step 32) is not consulted here. **Step 50** decides whether to index a `noindex` page; extraction stays a pure function of the HTML.
- Next: Step 40: Cross-page boilerplate removal

### Step 40: Cross-page boilerplate removal  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/extraction/boilerplate.py` (exported from `rag_engine.extraction`)
  - `BoilerplateIndex(min_pages, min_ratio, keep_one_min_chars, config)`: one per crawl job, in memory.
    - `add(page_key, markdown) -> bool`: counts each unit **once per page**. It returns False for a known key or **duplicate content**: twenty URLs serving one page count once, so duplicates can't make real content look like boilerplate.
    - `strip(page_key, markdown) -> StrippedMarkdown{markdown, removed_units, removed_chars}`.
    - `is_active()`, `pages`, `boilerplate_count()`.
  - **Units**: a paragraph, a table, or **one list item**. Nav lists differ per page by an "active" item, but their other items repeat exactly. **Headings and code fences are never units**: a heading is structure (Step 45 prunes it if its section empties), and code repeated across docs pages is usually real content.
  - `split_blocks(markdown) -> list[MarkdownBlock]`: blank-line blocks, with a fenced block (blank lines included, unterminated fences tolerated) as one block. It reassembles fences byte for byte.
  - `normalize_unit`: NFKC, casefold, list markers and Markdown punctuation ignored, whitespace collapsed. Hashed to 8 bytes (blake2b), so memory is ~100 B per distinct unit.
  - **Rule**: a unit is boilerplate when the index has ≥ `BOILERPLATE_MIN_PAGES` distinct pages and the unit is on ≥ `BOILERPLATE_MIN_RATIO` of them (and on ≥ 2). Inactive below `min_pages`: small sites are left alone.
  - **Keep one copy**: a boilerplate unit with ≥ `BOILERPLATE_KEEP_ONE_MIN_CHARS` normalized chars (a shared warranty paragraph, not a nav link) stays on the **first page that added it**. The engine adds hub pages first. Shorter chrome (nav items, cookie banners) is removed everywhere, including the keeper page.
  - A page made only of boilerplate becomes `""`. Step 45's min-content check turns it into an `empty_content` skip.
- Settings: `BOILERPLATE_MIN_PAGES` (5, ≥2), `BOILERPLATE_MIN_RATIO` (0.5), `BOILERPLATE_KEEP_ONE_MIN_CHARS` (200). `.env.example` updated.
- Tests: `TestBoilerplate_S40` (23, unit):
  - Blocks and normalization: block splitting (heading, multi-line text, list with continuation, fence with an inner blank line, table); 4 normalization variants (emphasis/case/whitespace, list markers, ordered markers, full-width NFKC).
  - Stripping: banner + 3 nav items stripped and the unique content and heading kept; inactive below min pages; a unit on 4/10 pages kept while one on 10/10 goes; nav with a varying active item; headings and shared code never stripped; fences reassembled byte for byte.
  - Duplicates and keeper: 10 duplicate URLs count once; `add` idempotent per key; a long shared block keeps one copy on the first page only; keep-one can be disabled; pages added later are stripped; an all-boilerplate page → `""`.
  - Configuration: 4 invalid threshold combinations; defaults from settings.
  - **Fixture site**: all static pages through `HtmlExtractor` → index (19 distinct; dup-a/dup-b counted once) → **the cookie banner trafilatura kept on ≥ 5 pages is gone everywhere**, while Latin-1 text, the getting-started code block and the h3 survive.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1129 passed, 4 skipped (container) in 188 s.
- Decisions/notes:
  - **Where it runs**: frequency is a whole-job statistic, so stripping can't happen page by page during the crawl. **Step 50** stores each page's extracted Markdown as-is. The **finalize/ingest stage** (before Step 55's chunking) builds the index from the job's pages in crawl order, which picks the keeper, then strips each page. After a crash it is rebuilt from Mongo, so nothing is persisted. Step 46's dedupe hash stays on the unstripped Markdown (duplicates are duplicates either way).
  - Recommended order at ingest: boilerplate strip (40) → text cleaning (45; prunes headings whose sections emptied, applies min content) → chunking (55+).
- Next: Step 41: PDF extractor

### Step 41: PDF extractor  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/extraction/pdf.py` → `PdfExtractor(config)` (`name="pdf"`, `kind=PDF`, exported).
  - **Caps**: a body over `PDF_MAX_BYTES` (20 MiB) → new `DocumentTooLargeError` (`too_large`, failed). Only the first `PDF_MAX_PAGES` (500) pages are extracted; `metadata.truncated` / `pages_extracted` / `page_count` record it (not a failure).
  - **Unreadable** (`DocumentUnreadableError`, `document_unreadable`, failed):
    - `encrypted`: a user password is required. Owner-password-only PDFs are opened with `authenticate("")` and extracted normally.
    - `no_text_layer`: no text on any extracted page, i.e. a scan without OCR. **OCR is off** (`use_ocr=False`) by design.
  - Corrupt → `ExtractionError` (`extraction_failed`, detail `corrupt`). No pages → `EmptyContentError`.
  - **Markdown** via pymupdf4llm 2.1's layout engine (`pymupdf-layout`, ONNX GNN): headings, lists, **tables as Markdown tables** (verified with a ruled grid), running page headers/footers dropped (`header=False, footer=False`), images ignored.
    - Each page is prefixed with the marker line **`<!-- page N -->`** (`page_marker()`), which Step 56 uses for page citations. Trailing spaces and blank runs are tidied.
    - If pymupdf4llm raises, the plain text layer is used (`metadata.extraction="pymupdf_text"`).
  - **Metadata**:
    - `title`: document info, unless empty or junk (`Untitled`, `Microsoft Word - …`, file names), then the first heading.
    - `lang`: catalog `/Lang`, primary subtag.
    - `description`: subject.
    - `metadata`: `{extraction, author, created (from `D:YYYYMMDD…`), page_count, pages_extracted, truncated}`.
  - **Outlinks**: http(s) URI link annotations on the extracted pages, de-duplicated in order (mailto etc. dropped), for `LinkExtractor.filter_urls`.
  - **Thread safety**: PyMuPDF must not run in several threads at once, and extraction runs in `run_sync` threads, so a process-wide `threading.Lock` serializes PDF parsing. Crawl workers parallelize across processes. MuPDF's stderr warnings/errors are switched off (failures still raise).
  - Never prints to stdout: with OCR off, pymupdf4llm's "Document parser messages" block (the RapidOCR warning) is not emitted.
- Deps: runtime `pymupdf4llm` (2.1) and `pymupdf` (moved from the dev group; imported directly). pymupdf 2.1 brings `pymupdf-layout` and **`onnxruntime`** into the runtime closure, so the worker image grows; Step 95 should measure it.
- Settings: `PDF_MAX_BYTES` (20 MiB), `PDF_MAX_PAGES` (500); `.env.example` updated.
- Also: removed a now-unused `type: ignore` in `tests/fixtures/build_documents.py` (pymupdf 2.1 declares `PDF_ENCRYPT_AES_256` statically; ty flagged it).
- Tests: `TestPdfExtractor_S41` (21, unit; PDFs generated with pymupdf where the fixtures don't cover a case):
  - Content: fixture spec sheet (markers, headings, title/author, counts); ruled table → Markdown table; running header/footer dropped; page markers in order.
  - Metadata: title/subject/`/Lang`/creation date; 4 junk titles fall back to the first heading; URI outlinks de-duplicated, mailto dropped.
  - Caps and failures: page cap truncates and flags; byte cap → `too_large`; scanned fixture → `no_text_layer`; password fixture → `encrypted`; owner-only encryption readable; 2 corrupt bodies → `extraction_failed`; layout failure → text-layer fallback.
  - Integration: 4 concurrent extractions (lock) correct; dispatched by bytes through the registry from `application/octet-stream`; fixture PDF fetched over HTTP.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1150 passed, 4 skipped (container) in 171 s.
- Decisions/notes:
  - **Step 56** must treat `<!-- page N -->` lines as page boundaries (attach `page` to chunks), not as content.
  - The lock means one PDF at a time per worker process. If PDF-heavy crawls are slow, the fix is more worker processes (or a dedicated process pool), not threads.
- Next: Step 42: DOCX extractor

### Step 42: DOCX extractor  (2026-11-07)
- Status: ✅ done
- Built:
  - `rag_engine/extraction/docx.py` → `DocxExtractor(config)` (`name="docx"`, `kind=DOCX`, exported). It walks the body **in document order** (`iter_inner_content`: paragraphs and tables interleaved).
    - **Headings**: `Title` → `#`. `Heading N` → N `#`s, capped at 6, **including custom styles whose base-style chain reaches a heading**. For localized templates ("Überschrift 1"), the paragraph/style **outline level** decides (0 → `#`; 9 = body text).
    - **Lists**: Word numbering, direct `numPr` or inherited from the style. Bullet vs ordered comes from `numbering.xml` (`numId` → `abstractNum` → `numFmt` per `ilvl`); nesting from `ilvl`, rendered with **3 spaces per level** so it nests under both `-` and `1.` parents in CommonMark. `numId=0` (numbering removed) → plain paragraph. Style names `List Number/Bullet [N]` are the fallback. Ordered items all render as `1.` (Markdown renumbers).
    - **Inline**: bold → `**…**`, italic → `*…*`, both → `***…***`; adjacent runs with equal formatting merged; whitespace kept outside the markers. Hyperlink text kept; **http(s) hyperlink URLs → `outlinks`** (de-duplicated; mailto dropped).
    - **Code**: consecutive paragraphs in code-like styles (`Code`, `Source Code`, `HTML Preformatted`, `…verbatim/monospace`) → one fenced block with indentation preserved. Quote styles → `> `.
    - **Tables** → Markdown tables (first row = header). **Merged cells appear once** (horizontal and vertical repeats blank), `|` escaped, multi-paragraph cells joined with spaces, nested tables flattened into their cell, empty rows dropped.
    - Ignored: page headers/footers, footnotes, comments (boilerplate).
    - **Metadata**: `title` (core properties, else the first heading), `lang` (core `language`, else `docDefaults` run language), `description` (subject), `metadata {author, created, modified}` (ISO dates).
    - Failures: corrupt/unopenable → `ExtractionError(detail="corrupt")`; no text → `EmptyContentError`. Encrypted DOCX (OLE container) is already `document_unreadable` from the registry (Step 38).
  - `rag_engine/extraction/ooxml.py` → `check_archive(body, max_uncompressed)`: the **zip-bomb guard shared by DOCX/PPTX/XLSX**. It sums declared uncompressed sizes from the central directory only (nothing inflated) and raises `DocumentTooLargeError(detail="uncompressed")` over `OOXML_MAX_UNCOMPRESSED_BYTES` (new setting, 200 MiB). A non-zip is `ExtractionError("corrupt")`.
- Deps: `python-docx` moved from the dev group to runtime.
- Tests: `TestDocxExtractor_S42` (17, unit; documents generated with python-docx):
  - Fixture `install-guide.docx`: exact Markdown (h1, paragraph, h2, bullets, h2, table), title, metadata.
  - Headings: Title/Heading 1-3, Heading 7 capped at 6, a custom style based on Heading 2; localized "Überschrift" styles via outline level.
  - Lists and inline: ordered list with a nested bullet, then a paragraph; bold/italic with merged runs; hyperlinks (text kept, https outlink once, mailto dropped).
  - Code/quote: code-style paragraphs → one fence with indentation, Quote → `>`.
  - Tables: in document order, with escaped pipe, multi-paragraph cell, vertical + horizontal merges; nested table flattened.
  - Metadata: core properties (empty title → first heading, subject, `de-DE` → `de`); language from template defaults.
  - Failures and guards: empty → `EmptyContentError`; 2 corrupt bodies (not a zip, missing parts); **zip bomb** (< 50 KB on the wire, 5 MB declared, 1 MB cap) refused before parsing.
  - Integration: registry dispatch from `application/octet-stream`; fixture fetched over HTTP.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1167 passed, 4 skipped (container) in 158 s.
- Next: Step 43: PPTX extractor

### Step 43: PPTX extractor  (2026-11-07)
- Status: ✅ done
- Built:
  - `rag_engine/extraction/pptx.py` → `PptxExtractor(config)` (`name="pptx"`, `kind=PPTX`, exported). **One section per visible slide**:
    ```
    <!-- slide N -->          (slide_marker(); mirrors the PDF page markers for citations)
    ## <slide title>          ("## Slide N" when untitled)
    - body bullet             (body/content placeholders; 3 spaces per paragraph level)
    text box paragraphs
    | table |                 (merged cells shown once via `is_spanned`)
    Chart: <chart title>
    > speaker notes           (one quoted line per notes paragraph)
    ```
    - Shapes are read **top-to-bottom, left-to-right**, and group shapes are flattened in place.
    - Footer/date/slide-number/header placeholders are skipped (template text).
    - Soft line breaks (`\v`) become spaces.
    - A slide with no text at all is omitted. A deck with no text → `EmptyContentError`.
    - **Hidden slides** (`show="0"`) are skipped (drafts/backups), counted in `metadata.hidden_skipped`.
    - `PPTX_MAX_SLIDES` (new setting, 500) caps visible slides, flagged via `metadata.truncated`.
    - **Metadata**: `title` (core properties, else the first slide title), `lang` (core `language`, else the first run's `lang`), `description` (subject), `metadata {author, created, modified, slide_count, slides_extracted, hidden_skipped, truncated}`.
    - **Outlinks**: http(s) hyperlinks from text runs and shape click actions.
    - The OOXML zip-bomb guard (`check_archive`) runs first. Corrupt files → `ExtractionError("corrupt")`.
  - `rag_engine/extraction/tables.py` → `render_table(rows)` / `cell_text()`: the **shared GFM table renderer** (whitespace collapsed, `|` escaped, rows padded, empty rows dropped). DOCX was refactored onto it (its tests unchanged and green); XLSX (Step 44) uses it too.
- Deps: `python-pptx` moved from the dev group to runtime. The re-lock bumped two of its dependencies: `xlsxwriter` 3.2.9 → 3.3.1 and **`pillow` 12.3.0 → 13.0.0 (major)**. Nothing in `rag_engine` uses Pillow directly, and the full suite is green.
- Tests: `TestPptxExtractor_S43` (16, unit; decks generated with python-pptx):
  - Fixture `roadmap.pptx`: exact Markdown (3 slide sections, bullets, notes, release table), title, slide count, created date.
  - Slide content: nested bullets with a soft line break and multi-paragraph notes; text boxes in reading order despite insertion order; table with an escaped pipe and a merged row; group-shape text and a chart title; untitled slide → `## Slide 3`, an empty slide omitted.
  - Visibility and caps: hidden slide skipped and counted; slide cap truncates and flags.
  - Metadata and links: core properties incl. `fr-FR` → `fr`; language from a run's `lang`; run hyperlink and click-action outlinks, mailto dropped.
  - Failures: textless deck; corrupt archive; zip-bomb guard.
  - Integration: registry dispatch from `application/zip`; fixture fetched over HTTP.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1183 passed, 4 skipped (container) in 163 s.
- Decisions/notes:
  - **Step 56** treats `<!-- slide N -->` like `<!-- page N -->` (a location marker, not content).
  - Pictures' alt text and SmartArt text are not extracted yet (rarely matter for docs sites). This is a known gap.
- Next: Step 44: XLSX extractor

### Step 44: XLSX extractor  (2026-11-07)
- Status: ✅ done
- Built:
  - `rag_engine/extraction/xlsx.py` → `XlsxExtractor(config)` (`name="xlsx"`, `kind=XLSX`, exported). openpyxl `load_workbook(read_only=True, data_only=True)` streams rows; the workbook is always closed.
    - **One section per visible sheet**: `<!-- sheet N -->` (`sheet_marker()`, like the page/slide markers) + `## <sheet name>` + a GFM table via the shared `render_table` (first row = header).
    - **Values**: Excel's **cached results**, not formulas (a file without cached values shows those cells empty, never `=A2+B2`).
      - integral floats → `1899`; other floats → up to 10 significant digits
      - percent number formats → `12.5%`
      - dates → ISO, with the time kept unless midnight; `time` → ISO
      - booleans → `TRUE/FALSE`
      - text whitespace collapsed, `|` escaped
    - **Tidying**: entirely empty rows and **entirely empty columns** are dropped (data starting at C5 with an empty column D, or formatting far past the data, becomes a clean table). A **lone title row** above a wider header becomes a caption paragraph instead of a fake one-column header.
    - Hidden and very-hidden sheets are skipped (`metadata.hidden_skipped`). Chart sheets have no cells. Empty sheets are omitted; no values anywhere → `EmptyContentError`.
    - **Caps**:
      - `XLSX_MAX_SHEETS` (50) visible sheets, `XLSX_MAX_ROWS` (2000) and `XLSX_MAX_COLS` (50) per sheet.
      - One extra row/column is read, so a sheet **exactly** at the cap is not marked truncated.
      - A truncated sheet gets an `*Truncated: showing the first R rows and C columns.*` line (so an answer can say the table was cut) and counts in `metadata.truncated_sheets`; `metadata.truncated` covers the sheet cap too.
    - **Metadata**: `title`/`description` (subject, else description)/`lang` from workbook properties; `metadata {author (creator), created, modified, sheet_count, sheets_extracted, hidden_skipped, truncated_sheets, truncated}`.
    - **No outlinks**: read-only mode doesn't load hyperlinks (known gap; links in spreadsheets rarely matter for crawling).
    - The zip-bomb guard runs first. Corrupt → `ExtractionError("corrupt")`.
  - **`default_registry(config)`** (in `registry.py`, exported): all five extractors (HTML, PDF, DOCX, PPTX, XLSX), as promised in Step 38. The crawl engine (Step 50) uses it.
- Deps: `openpyxl` moved from the dev group to runtime (no version changes in the lock).
- Settings: `XLSX_MAX_SHEETS` (50), `XLSX_MAX_ROWS` (2000), `XLSX_MAX_COLS` (50); `.env.example` updated.
- Tests:
  - `TestXlsxExtractor_S44` (17, unit; workbooks generated with openpyxl):
    - Fixture `pricing.xlsx`: exact Markdown for both sheets.
    - Values: a 12-value formatting matrix; formulas show cached values only.
    - Table shape: sparse sheet trimmed (offset start, empty column, gap rows, far formatting); title row → caption; single-column sheet keeps its header.
    - Sheets: hidden/very-hidden/empty/chart sheets; no-values workbook → `EmptyContentError`.
    - Caps: row cap truncates with a note; exactly-at-cap not truncated; column cap; sheet cap.
    - Metadata: workbook properties.
    - Failures and integration: corrupt archive; zip-bomb guard; registry dispatch with an empty content type; fixture fetched over HTTP.
  - `TestDefaultRegistry_S44` (6, unit): every `ContentKind` has an extractor, and all 5 fixture document types dispatch by bytes (labelled `application/octet-stream`) to the right extractor with non-empty Markdown.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1206 passed, 4 skipped (container) in 229 s.
- Decisions/notes:
  - **Extraction phase (38–44) complete.** Location markers in Markdown are `<!-- page N -->` (PDF), `<!-- slide N -->` (PPTX) and `<!-- sheet N -->` (XLSX). **Step 56** must parse all three as location metadata, not content.
- Next: Step 45: Text cleaning

### Step 45: Text cleaning  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/extraction/cleaning.py` (exported from `rag_engine.extraction`)
  - `TextCleaner(min_content_chars=, config).clean(markdown) -> CleanedText{markdown, content_chars, pruned_headings}`, or `EmptyContentError(detail="below_min_content")` (skipped, `empty_content`). **Idempotent**: it runs twice per page (see notes).
  - `normalize_characters(line, code=False)`:
    - **NFKC** outside code (ligatures, full-width forms, NBSP/thin/ideographic spaces → space).
    - **Invisible characters removed**: ZWSP, word joiner, BOM, soft hyphen, U+180E, LRM/RLM and bidi embeddings/overrides/isolates (also the "Trojan source" trick). **ZWJ/ZWNJ kept**: they carry meaning in Persian, Indic scripts and emoji sequences.
    - **Control characters removed everywhere** (C0 except tab/newline, DEL, C1). Inside code only controls go; tabs, NBSP and ligatures stay.
  - Lines: CRLF/CR → LF; outside fences, inner whitespace runs → one space while **leading indentation is kept** (list nesting, tabs → 4 spaces), trailing whitespace stripped. **Code fences untouched** (blank lines, tabs, spacing), including an unterminated fence through to the end.
  - Structure:
    - Empty list items and all-blank table body rows dropped (separator rows like `|---|:---:|` kept).
    - **Empty sections pruned**: a heading goes when no content follows before the next heading of the same or a higher level. A parent whose subsections have content stays. Page markers don't end a section (a heading at the bottom of a PDF page with its text on the next is kept).
    - **Location markers** (`<!-- page|slide|sheet N -->`, other standalone comments) are kept only if a kept block follows before the next marker. This runs as a second pass after headings, so a kept heading never appears to move to the previous page.
    - Blocks are rejoined with one blank line (via Step 40's `split_blocks`).
  - **Min content**: letters/digits in content blocks (headings and markers excluded) < `CLEAN_MIN_CONTENT_CHARS` → `EmptyContentError`.
- Settings: `CLEAN_MIN_CONTENT_CHARS` (**20**). It's deliberately low, aimed at "Loading…" shells and empty listings rather than quality. The fixture's `/flaky` page ("All widget lines operational.", ~25 chars), which Step 54 expects to succeed, must pass. `.env.example` updated.
- Tests: `TestTextCleaning_S45` (27, unit):
  - Characters: 7 normalization cases (ligature, full-width, compatibility spaces, zero-width, soft hyphen, bidi, controls); ZWNJ (Persian) and ZWJ (emoji) kept; code keeps NBSP/ligature/tab but loses controls.
  - Whitespace and code: collapse outside code with list indentation kept and CRLF handled; code block byte-identical; an unterminated fence protects the rest.
  - Structure: empty list items and blank table rows; pruning hierarchy (4 headings pruned, parent kept); a heading with only code kept; blank-page markers pruned; **a heading continued on the next page keeps its page marker**.
  - Min content: 5 below-threshold inputs → `empty_content` skip; headings and markers don't count; threshold from settings.
  - Idempotence and fixtures: idempotent on messy input; **every fixture document (all HTML pages except the SPA shell, plus PDF, DOCX, PPTX, XLSX) extracts, cleans, and is unchanged by a second clean**; **ingest order on the whole fixture site** (Step 40 boilerplate strip → clean) removes the cookie banner and the home page's empty link-list headings while the content stays.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1233 passed, 4 skipped (container) in 147 s.
  - One earlier full run (load average ~39 from other processes, 401 s) failed `TestJsShellDetector_S33::test_every_static_fixture_page_but_the_spa_is_not_a_shell`. It passed alone right after, and the run meant to capture its error was cut off by a machine reboot. It has passed in the full suite since. The cause is **unconfirmed** (most likely load-related); watch for a repeat.
- Decisions/notes:
  - **Where it runs** (twice, which is why it's idempotent):
    1. **During the crawl** (Step 50: extract → **clean** → dedupe), so empty pages are skipped early and Step 46 hashes clean text.
    2. **At ingest** after Step 40's boilerplate strip, which can empty sections or whole pages.
  - Indented (4-space, unfenced) code blocks get inner whitespace collapsed. All our extractors emit fenced code, so this only matters for hand-written Markdown.
- Next: Step 46: Content dedupe

### Step 46: Content dedupe  (2026-11-07)
- Status: ✅ done
- Built:
  - `rag_engine/crawling/dedupe.py` (exported from `rag_engine.crawling`):
    - `content_hash(markdown)`: sha256 hex of the **cleaned** Markdown (Step 45), surrounding whitespace ignored. This is `Page.content_hash`.
    - `ContentDeduper(job_id, scope, store, config).check(page_id=, url=, markdown=, depth=, canonical_url=, aliases=) -> DedupeDecision{outcome, content_hash, canonical_url, duplicate_of, duplicate_of_url}`, with `page_status`/`reason` properties that map straight onto `Page`:
      1. **canonical_elsewhere** → `skipped`/`canonical_elsewhere`. The page's normalized canonical (Step 32) is **in scope** (`scope.check` at the page's depth) and is not the page itself. **"Itself" = final URL + `aliases`** (requested URL, redirect chain), so "A redirects to B, B's canonical is A" never skips the only copy. Its hash is **not claimed**, so the canonical target claims it when fetched (the engine enqueues `canonical_url`). Out-of-scope, unparseable or self canonicals are ignored, so content is kept.
      2. **duplicate** → status `duplicate`, `duplicate_of` = the original page id (plus its URL): another page of this job already claimed the hash.
      3. **unique** → `succeeded`: this page owns the hash and is the one to chunk/embed. Re-checking the owner (a resumed job) is unique again (idempotent).
    - `ContentClaimStore` protocol (one `claim()` method), satisfied by the repository below and by `tests/support/fakes.py::InMemoryContentClaimStore`.
  - `rag_engine/domain/content_claim.py` → `ContentHashClaim` (collection `content_hash_claims`, prefix `chc`, fields `job_id/content_hash/page_id/url`). **Unique index `(job_id, content_hash)`** makes "first page wins" atomic across concurrent fetches and crash-safe on resume. A **TTL index** on `created_at` (`CLAIM_TTL_SECONDS` = 30 days) is the safety net for jobs that never finalize. It is registered, so startup's `ensure_indexes` creates both.
  - `rag_engine/repositories/content_claims.py` → `ContentClaimRepository.claim(job_id, hash, page_id, url) -> owner claim` (find, else insert; on a duplicate-key race read the winner; retry once if it vanished) and `release_job(job_id)` (finalize cleanup).
- Settings: `CRAWL_RESPECT_CANONICAL` (true). With false, the canonical is still recorded on the decision but never skips. `.env.example` updated.
- Tests:
  - `TestContentDedupe_S46` (18, unit, in-memory store):
    - Hashing.
    - Duplicates: first unique/second duplicate with status mapping; owner re-check idempotent; jobs isolated; **20 concurrent checks → exactly one original**.
    - Canonical: canonical elsewhere skips **without claiming** (the target is then unique); 6 non-elsewhere canonicals (self, self with tracking/case/port/fragment, redirect alias, out of scope, invalid, none); a canonical outside the path prefix is ignored; respect disabled.
    - Every outcome produces a **valid `Page` record** (model validators pass).
    - **Fixture site**: real extraction → cleaning → `LinkExtractor` canonical → dedupe gives exactly `SITE_PAGES`/`EXPECTED_PAGE_OUTCOME` for `dup-b.html` (duplicate of dup-a) and `canonical-variant.html` (`canonical_elsewhere`), and getting-started is unique.
  - `TestContentClaims_S46` (6, integration, real MongoDB): first claim wins; **25 concurrent claims → one owner, one document**; owner re-claim returns the same document; jobs isolated and released separately; unique + TTL indexes exist; `ContentDeduper` on the repository (10 concurrent pages → 1 unique, 9 duplicates).
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1257 passed, 4 skipped (container) in 148 s.
- Decisions/notes:
  - **Step 50 (engine)** per page: extract → clean (45) → `deduper.check(...)` with `aliases=(requested_url, *redirect_chain)` → persist the page with `status/reason/duplicate_of/content_hash/canonical_url`. On `canonical_elsewhere`, enqueue `decision.canonical_url` (frontier dedupes it). Only `unique` pages keep their Markdown for chunking.
  - **Finalize (Step 49 state machine / maintenance)** calls `ContentClaimRepository.release_job(job_id)`.
  - First claim wins. With hub/shallow-first crawl order (Step 48), originals tend to be the better-linked URLs. Only exact duplicates of cleaned Markdown are caught; near-duplicates (e.g. a differing "last updated" line) are a known gap.
- Next: Step 47: Page repository

### Step 47: Page repository  (2026-11-07)
- Status: ✅ done
- Built:
  - `Page` model (`domain/page.py`) gained the fields earlier steps deferred here:
    - `js_signals` and `render_note` (Step 36)
    - `extractor`, `description`, `lang`, `doc_metadata` (scalar dict) from `ExtractedDocument`
    - `markdown_truncated`, and the `extract_ms` timing (`fetch_ms` covers fetch + render)
    - Existing: the `(job_id, url_hash)` unique index `job_url_unique` and the `(job_id, status, _id)` index `job_status`.
  - `rag_engine/repositories/pages.py`:
    - `PageOutcome` (frozen dataclass, kw-only): every outcome field. `changes()` writes **all** of them, so unset fields are **cleared**. Re-recording a retried page can't leave a stale `reason`, and a skip after a success drops the old Markdown. `reason_detail` is capped at 500.
    - `truncate_markdown(md, max_bytes)`: cuts at the last `\n\n` block boundary within the UTF-8 byte cap, never splitting a multi-byte character, and flags it.
    - `PageRepository(db, config)`:
      - **Discovery**: `add_discovered(page) -> (page, created)` is race-safe; a repeat returns the existing page. `add_many_discovered(pages) -> inserted` is an unordered bulk insert where duplicate-key errors (known or repeated-in-batch URLs) count as "already there" and any other write error raises `DatabaseUnavailableError`.
      - `get_by_url_hash(job_id, url_hash)`.
      - **Outcomes**: `record_outcome(page_id, PageOutcome) -> Page` goes through `BaseRepository.update`, so the merged document is **validated by the `Page` consistency rules first** (`AppValidationError`, page untouched). Markdown over `PAGE_MAX_MARKDOWN_BYTES` is truncated and flagged.
      - **Reads**: `count_by_status(job_id)` and `count_by_reason(job_id)` (aggregations, for Step 53's summary); `iter_succeeded(job_id, batch_size)` streams succeeded pages in **discovery order** (time-ordered `_id`; Step 40's keeper order at ingest); `list_for_job(tenant_id, job_id, status=, reason=, limit, cursor)` is keyset-paginated oldest first and **tenant-scoped** (API).
      - `delete_job(job_id)`.
  - `PageNotFoundError` (`page_not_found`).
- Settings: `PAGE_MAX_MARKDOWN_BYTES` (4 MiB, ≤ 12 MiB; MongoDB's 16 MB document limit). `.env.example` updated.
- Also: `tests/unit/test_dedupe.py` builds its `Page` with explicit kwargs. ty rejected the `**{...}` unpack once `Page` gained a dict-typed field.
- Tests:
  - `TestPageOutcome_S47` (9, unit): every field written and cleared; `reason_detail` cap; oversized Markdown truncated on a paragraph and flagged; 6 truncation cases (none, short, exactly at the cap, block boundary at 10 vs 9 bytes, no boundary, multi-byte safety).
  - `TestPageRepository_S47` (19, integration, real MongoDB):
    - Discovery: idempotent `add_discovered`; same URL in another job is new; bulk insert skips known and in-batch repeats; **20 concurrent discoveries of one URL → one page**.
    - Outcomes: success round-trips every field; **retry success clears the earlier failure**; skip clears earlier content; duplicate outcome; 4 inconsistent outcomes rejected with the page untouched; oversized Markdown stored truncated.
    - Reads: counts by status and reason (other job excluded); `iter_succeeded` in discovery order across batches; `list_for_job` pagination, status/reason filters and **tenant isolation**; `delete_job`; indexes.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1285 passed, 4 skipped (container) in 150 s.
- Decisions/notes:
  - **Step 48 (frontier)** claims `QUEUED` pages with `find_one_and_update` (sets `IN_PROGRESS`, increments `attempts`, sets a lease time) and needs a claim index plus priority fields (`discovered_via`, depth). Those belong to Step 48, together with the claim query they serve.
  - The engine (Step 50) builds `PageOutcome`s from `FetchResult` + `ExtractedDocument` + `DedupeDecision`. The repository stays free of crawling imports.
- Next: Step 48: Frontier

### Step 48: Frontier  (2026-11-07)
- Status: ✅ done
- Built:
  - `rag_engine/crawling/frontier.py` (exported from `rag_engine.crawling`). `Frontier(pages, FrontierJob(tenant/kb/source/job ids), run_id=, config, clock)` is one job's queue of `Page` documents, bound to one run (a new `run_id` per attempt of the job):
    - **Enqueue**: `add(url, depth=, via=, discovered_from=) -> created` and `add_many(entries) -> inserted`. URLs are normalized + hashed (Step 23). **Dedupe is the `(job_id, url_hash)` unique index** (Step 47's race-safe inserts). `add_many` skips invalid URLs and keeps the best entry for a URL listed twice in one batch.
    - **Promotion**: a URL rediscovered at a better priority **while still queued** gets the new `priority/depth/discovered_via/discovered_from`, so `max_depth` sees its shortest known path. Claimed or finished pages are never touched.
    - **Order**: `frontier_priority(via, depth) = depth * 10 + rank`, rank seed 0 < sitemap 1 < canonical 2 < link 3; ties in discovery order (`_id`). Sitemap entries (depth 0) and shallow pages go first; a canonical target is claimed before the links at its depth.
    - **Claim**: one `find_one_and_update` on the new `job_frontier` index: `queued → in_progress`, `attempts += 1`, `claimed_by = run_id`, `lease_expires_at = now + CRAWL_FRONTIER_LEASE_SECONDS`. Pages with a future `retry_at` aren't claimable. Only when nothing is ready, pages whose **lease expired** (a dead crawler) are reclaimed.
    - **Give back**: `retry_later(page_id, delay_seconds)` re-queues with a not-before time (Step 28's note: retries don't sleep inside a concurrency slot); `release(page_id)` re-queues and refunds the attempt (stopping early). Both only work for the run holding the page (False otherwise, e.g. after its lease was reclaimed).
    - **Crash recovery**: `recover() -> RecoveryReport(requeued, abandoned)` re-queues every page held by another (dead) run, immediately, without waiting for the lease. A page that has used every attempt (`CRAWL_MAX_RETRIES + 1`) without reporting back fails as **`worker_lost`** (in `recover()` and on lease reclaim), so a page that crashes the worker every time can't loop forever.
    - `stats() -> FrontierStats(ready, delayed, in_flight, next_retry_at)` with `drained` (nothing to claim or wait for).
  - `Page` gained `discovered_via` (`DiscoveredVia`: seed/sitemap/canonical/link), `priority`, `retry_at`, `claimed_by`, `lease_expires_at`, and the index `job_frontier (job_id, status, priority, _id)`. `attempts` now means claims so far (= fetch attempts started, what `RetryPolicy.decide` takes).
  - `PageReason.WORKER_LOST` (failed).
  - `PageOutcome.changes()` also clears `lease_expires_at` and `retry_at`. `claimed_by` stays as an audit of the run that finished the page.
- Settings: `CRAWL_FRONTIER_LEASE_SECONDS` (600; must be > `CRAWL_FETCH_TIMEOUT_SECONDS + CRAWL_RENDER_TIMEOUT_SECONDS`, validated). `.env.example` updated.
- Also: `TestDomainModels_S13` count of `PageReason` 21 → 22; new `TestSettings_S3` invalid-limits case for the lease.
- Tests:
  - `TestFrontierOrder_S48` (6, unit): priority order (depth, then source) on a shuffled list; every source at depth n ranks before depth n+1; negative depth rejected; `Page` defaults; lease validator; outcome clears the lease but keeps `claimed_by`.
  - `TestFrontier_S48` (20, integration, real MongoDB):
    - Enqueue: normalized URL stored once with job ids and priority; `add_many` counts, in-batch best duplicate, invalid URL skipped; **20 concurrent adds of one URL → one page**; promotion (worse ignored, better applied); claimed/finished pages never promoted; jobs isolated.
    - Claims: **seed → sitemap → canonical → link, by depth, then discovery order** on a shuffled insert order; the claim sets status/run/attempts/lease; **8 concurrent crawlers drain 40 pages with no page claimed twice**.
    - Retries: not claimable before `retry_at`, claimable after (attempts 2), stats show it delayed; a retry doesn't block the queue; only the holder can give a page back; release refunds the attempt; negative delay rejected.
    - Recovery: an expired lease is reclaimed only when nothing else is ready, and the dead run then can't give it back; a page crashing every crawler fails `worker_lost` after 3 attempts (valid `Page`); `recover()` re-queues a dead run's pages and leaves its own alone; `recover()` abandons an exhausted page; **crash mid-crawl → resume: finished pages not redone, in-flight pages redone, job drained, leases cleared**.
    - `job_frontier` index exists.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1312 passed, 4 skipped (container) in 217 s.
- Decisions/notes:
  - **Step 50 (engine)** loop: `recover()` at start → seed `add(seed, depth=0, via=SEED)` + sitemap `add_many(..., via=SITEMAP, depth=0)` → workers `claim()` until `stats().drained` (sleep until `next_retry_at` when only delayed pages remain) → per page either `record_outcome` or, on a retryable failure, `retry_later(page.id, decision.delay_seconds)` with `policy.decide(error, page.attempts)`. Outlinks → `add_many(via=LINK, depth+1)`; `canonical_elsewhere` → `add(canonical_url, via=CANONICAL, depth=page.depth)`.
  - **Zombie writes**: `record_outcome` doesn't check `claimed_by`, so a crawler that outlived its lease could still write an outcome for a page another run reclaimed. The lease validator makes this unlikely; Step 50 can guard by checking `claimed_by == run_id` in the outcome filter if it shows up.
  - **Step 51 (caps)**: `max_pages` should count claims (or succeeded pages) and stop claiming; remaining queued pages become `skipped/max_pages` at finalize. **Step 52 (cancel)**: in-flight pages finish or `release()`; the rest become `skipped/job_cancelled`.
  - **Step 80 (crash recovery task)** requeues the job with a new run id; the new run's `recover()` does the page side.
  - Mutual canonicals (A → B, B → A both skipped, found in Step 46's review) are still open for Step 50.
- Next: Step 49: CrawlJob repository + state machine

### Step 49: CrawlJob repository + state machine  (2026-11-07)
- Status: ✅ done
- Built:
  - **State machine** `JOB_TRANSITIONS` (`domain/enums.py`, plus `JobStatus.can_become()`):
    - `queued → running | failed | cancelled`
    - `running → finalizing | failed | cancelled | queued` (queued = its worker died, Step 80)
    - `finalizing → succeeded | failed | queued`. **Not cancellable**: the version swap (Step 78) completes or fails, never half-cancels.
    - Terminal statuses have no exits; success is reachable only through finalizing.
  - `CrawlJob` gained `run_id` (the worker run holding it, also the frontier's `run_id` from Step 48). New invariant: a running/finalizing job has `run_id` and `started_at`.
  - `rag_engine/repositories/jobs.py` → `JobRepository(db, clock=)`:
    - **Transitions**: each one is a single conditional `find_one_and_update` on the allowed source statuses, so racing transitions can't both win. `start(job, run_id)` (pipeline update: `attempts + 1`, `started_at` kept from the first start, `heartbeat_at`, `run_id` set via `$literal`); `begin_finalizing`, `succeed`, `cancel` (worker, after draining); `fail(job, JobError, run_id=None)` (worker with its run, or system code without one: a job that never started, or Step 80 giving up); `requeue(job, stale_run_id=)` (Step 80). A pending cancel request makes `requeue` **cancel** a running job instead, since its worker is gone and there's nothing to drain.
    - **Run ownership**: after `start`, everything (heartbeat, counters, usage, finalize, finish, cancel) is guarded on `run_id`. A run whose job was requeued to another run, finished or cancelled gets `JobRunSupersededError` (409 `job_run_superseded`) and must stop. A transition that's wrong for the current status raises `InvalidJobTransitionError` (409 `invalid_job_transition`, details `from`/`to`). `JobNotFoundError` (404 `job_not_found`).
    - `request_cancel(tenant_id, job)` (API): queued → `cancelled` at once; running → `cancel_requested_at` flagged (first time kept) for the worker to see; already cancelled → no-op; finalizing/finished → `InvalidJobTransitionError`. Tenant-scoped.
    - **Counters**: `increment(job, run, **deltas)` is one atomic `$inc` (names checked against `JobCounters`, deltas ≥ 0, all-zero is a read). It's allowed while running **and finalizing**, since ingest may still be embedding, and refused once finished. `set_counters(job, run, JobCounters)` overwrites them for reconciling. `add_usage(job, run, requests=, tokens=)` for embedding usage.
    - `heartbeat(job, run)` sets `heartbeat_at` and returns the job, so the worker sees `cancel_requested_at`. `find_stale(heartbeat_before, limit)` lists held jobs with an old heartbeat (status+heartbeat index), for Step 80.
    - Reads: `get_for_tenant`, `list_for_tenant(status=, source_id=, limit, cursor)` (newest first, keyset).
- Tests:
  - `TestJobStateMachine_S49` (43, unit): the table equals the documented edge set; `can_become` over all 36 pairs; terminal final / every active status can fail; finalizing not cancellable; success only via finalizing; held-job invariant; counter names.
  - `TestJobRepository_S49` (26, integration, real MongoDB):
    - Happy path queued → running → finalizing → succeeded with timestamps and attempts.
    - Refused: double start, skipped steps, finished jobs final, unknown job; `run_id` validated and stored literally (`"$status"`); system fail from each active status.
    - Run guard: another run is superseded on heartbeat, increment, finalize, cancel and fail; **10 concurrent starts → one winner**.
    - Recovery: requeue cuts off the dead run, the restart keeps the first `started_at` with attempts 2, no double requeue; requeue from finalizing; requeue with a pending cancel → cancelled; `find_stale` and a heartbeat clearing it.
    - Cancel: queued cancelled immediately and idempotent; running flagged, seen via heartbeat, worker cancels; finalizing/succeeded refused; tenant-scoped.
    - Counters: **100 concurrent increments exact**; validation; frozen once finished; counting while finalizing; `set_counters` and usage; heartbeat time.
    - Tenant-scoped list/get with status/source filters and pagination.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1381 passed, 4 skipped (container) in 237 s.
- Decisions/notes:
  - **Step 46's "finalize releases content claims"** moves to **Step 78 (finalization)**, which drives the terminal transitions: on any terminal status call `ContentClaimRepository.release_job(job_id)`. The job repository stays free of other collections.
  - **Step 50/77 (worker)**: `run_id = new_id("run")` per attempt → `jobs.start(job, run_id)` → `Frontier(..., run_id=run_id).recover()` → crawl, with `heartbeat()` on an interval (stop on `cancel_requested_at` or `JobRunSupersededError`) → `begin_finalizing` → Step 78 → `succeed`. Exceptions → `fail(job, JobError(code, message), run_id=run_id)`. Heartbeat interval and stale threshold settings belong to Steps 77/80.
  - Counter semantics for Step 50: `discovered` += `frontier.add_many(...)` new count; `fetched/failed/skipped/duplicates` once per recorded page outcome. A crash between `record_outcome` and `increment` loses a count, so the resumed run (and Step 53's summary) should `set_counters` from `PageRepository.count_by_status`.
  - Step 79 adds the one-active-job-per-source partial unique index and the tenant concurrency check.
- Next: Step 50: Crawl engine core

### Step 50: Crawl engine core  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/crawling/engine.py` (exported from `rag_engine.crawling`). `CrawlEngine(job, run_id=, pages, jobs, claims, fetcher, robots, limiter, sitemaps=, registry=, cleaner=, links=, policy=, config).run() -> CrawlSummary`. It needs a job already started by this run (Step 49); status transitions, heartbeat, caps (51) and cancel (52) stay with the caller. Collaborators are typed as protocols (`PageFetcher`, `RobotsSource`, `RateLimiter`, `SitemapSource`), so tests script them.
  - **Start**:
    - `frontier.recover()`, then queue the seed (`SEED`).
    - The seed is crawled **alone first**. If it redirects to another host, the scope is re-anchored (`with_seed_host`, as Step 31 asked) and the deduper rebuilt. On a resume, the re-anchor is derived from the seed row.
    - Sitemap discovery runs in the background (depth 0, `SITEMAP`) while `CRAWL_CONCURRENCY` workers drain the frontier.
  - **Per page**: scope → robots.txt (an unreachable robots.txt skips with its status/detail) → rate limit → fetch → redirect? → extract (`default_registry`) → clean (Step 45) → links (Step 32, or `filter_urls` for document outlinks) → dedupe (Step 46) → enqueue outlinks (`LINK`, depth+1) / canonical target (`CANONICAL`) → persist the outcome → `$inc` counters.
    - **Rate limit**: wait at most `CRAWL_RATE_LIMIT_MAX_WAIT_SECONDS` (new, 5 s). Past that, `frontier.release(page, delay_seconds=wait)` refunds the claim and the page returns when its slot opens (Step 30's note).
    - **Fetch failures**: `policy.decide(error, page.attempts)`. A retry goes through `frontier.retry_later` (not-before time). A `Retry-After` also calls `limiter.pause` for the host. A final outcome records reason/detail/`http_status`.
    - **Redirects**: the URL that serves the content owns it. The redirecting URL → `skipped/redirected` (new reason, with `final_url`). The target URL is queued and **claimed directly** (`frontier.claim_url`, new), gated by scope/robots, and gets the content, so it isn't fetched again and a linked target can't become a duplicate of its own redirect. **The fixture oracle changed** to `Expect.REDIRECT → (skipped, redirected)`.
    - **noindex** (robots meta / `X-Robots-Tag`) → `skipped/noindex` (new reason); links still followed.
    - **Canonical cycles** (the open item from Step 46): a canonical whose target row is itself `skipped/canonical_elsewhere` is ignored, so A↔B keeps one copy.
    - Outlinks are enqueued **before** the outcome is written, so the frontier never looks drained mid-page.
    - Any unexpected exception fails only that page: `failed/internal_error` (new reason) with `Type: message`, logged. `JobRunSupersededError`/`DatabaseUnavailableError` stop the engine (TaskGroup); pages it held are recovered by the next run.
    - **Zombie guard** (Step 48 note): `PageRepository.record_outcome(..., held_by=run_id)` only writes while that run holds the page (`claimed_by` + `in_progress`), otherwise `PageNotFoundError`, which the engine logs and drops.
  - **End**: drained (nothing ready/delayed/in flight) and sitemap discovery finished. Workers idle on a wake event (1 s cap, or until the next retry). Then **never-crawled links** (rejected by scope: out of scope / too deep; nofollow) become `skipped` rows via `frontier.add_skipped` (new), only for URLs the job doesn't know: "followed later wins" (Step 32's note). Each page keeps them on its row (`Page.deferred_links`, new, `DeferredLink{url, reason, detail}`, ≤ 1000 per page) and the flush streams `PageRepository.iter_deferred_links`, so links seen by a crashed run are reported too. Non-http(s) links (mailto:, tel:) can't be page rows; they're counted in `skipped_links_dropped`.
  - `CrawlSummary`: anchored seed URL, pages by status, reasons, recovery report, seed robots status/detail, the `SitemapReport` (or its error), skipped/dropped link counts.
  - Counters: `discovered` = queued pages created; `fetched` once per successful fetch (counted on the redirect target when it uses the content); `failed/skipped/duplicates` per recorded outcome, deferred skips included.
  - **Render politeness** (Step 36's note): `AutoFetcher(before_render=...)` (new hook) + `render_politeness(robots, limiter)`. The browser's second request books its own slot with the site's crawl delay.
- Settings: `CRAWL_RATE_LIMIT_MAX_WAIT_SECONDS` (5). `.env.example` updated.
- Also:
  - `PageReason` + `REDIRECTED`, `NOINDEX`, `INTERNAL_ERROR` (`TestDomainModels_S13` count 22 → 25).
  - `Frontier.release(page, delay_seconds=)`; `Frontier.claim_url`; `Frontier.add_skipped` + `SkippedEntry`.
  - `PageOutcome.deferred_links` (cleared on every rewrite like the other outcome fields).
- Tests:
  - `TestCrawlEngine_S50` (12, integration):
    - **Fixture site over real HTTP** (real Mongo + Redis limiter, `CRAWL_JS_RENDERING=never`): every `SITE_PAGES` URL ends with its `EXPECTED_PAGE_OUTCOME` (SPA shell excluded: Step 54 renders it). dup-a/dup-b = one original + one duplicate pointing at it. No page left queued/in progress; all four document formats extracted. Requests: `/private/` never requested, assets never fetched, `/flaky` 3, `/error` 3, `/rate-limited` 1, no page refetched beyond retries and redirect hops, ≤ 11 requests in any 1 s window (robots `Crawl-delay: 0.1`). Summary/counters agree with the page table.
    - **Crash and resume on the fixture site**: run-1 cancelled after 12 fetches, job requeued, run-2 resumes; the same final outcomes, and no page finished by run-1 is requested again.
    - Scripted (fake fetcher/robots/limiter, real Mongo): redirect content under the target (fetched at most once); seed redirect to another host re-anchors the scope (old host's link → out_of_scope); mutual canonicals keep one copy; noindex skipped with links followed; never-crawled links (max_depth, nofollow, offsite recorded; a too-deep URL reached at an allowed depth is crawled; mailto dropped); retries/rate-limit release (attempt refunded)/404/429 with host pause; a bug fails one page; counters exact (13 discovered, 11 fetched, 6 duplicates, 2 failed); a superseded run stops; engine refuses an unstarted job.
  - `TestFrontierEngineSupport_S50` (3, integration): `claim_url`, delayed release, `add_skipped` never overrides a known URL.
  - `TestPageRepositoryEngineSupport_S50` (3): lease-guarded outcomes; deferred links round-trip, stream and clear; `DeferredLink` validation.
  - `TestRenderPoliteness_S50` (4, unit): the hook books with the crawl delay (no max wait); `AutoFetcher` books before rendering; no slot when no render.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1403 passed, 4 skipped (container) in 197 s. The engine tests passed 3 runs in a row (~28 s).
- Decisions/notes:
  - Test fixes during the step, not engine bugs: the HTTP client follows redirects, so the server sees the target requested once per redirect hop; and request arrival jitters by tens of ms around the 0.1 s slots, so politeness is asserted as a rate (requests per 1 s window), not a minimum gap.
  - A redirect into a robots-disallowed or out-of-scope path is fetched (the client follows it) but not kept: the target is gated before its content is used.
  - **Step 51 (caps)**: the engine has no `max_pages`/bytes/time cap yet. Stop claiming at the cap and mark the rest `skipped/max_pages`. `CrawlConfig.max_pages` already bounds sitemap discovery.
  - **Step 52 (cancel)**: check the cancel flag in the worker loop (between claims), `release()` held pages or let them finish, then mark the rest `skipped/job_cancelled`.
  - **Step 53**: `CrawlSummary` + `count_by_reason` are the inputs for the job report; reconcile counters with `set_counters` there.
  - **Step 77 (task)**: wire `HttpFetcher(SafeHttpClient)`, `BrowserPool.job_context` → `AutoFetcher(http, browser, before_render=render_politeness(robots, limiter))`, `RobotsCache(RedisRobotsStore)`, `DomainRateLimiter`, `SitemapDiscoverer`, `ContentClaimRepository`, then `jobs.start` → `engine.run()` → `begin_finalizing`.
  - Step 54 asserts the browser-rendered SPA (`/app/`, `/app/advanced`) on top of this test.
- Next: Step 51: Caps & timeouts

### Step 51: Caps & timeouts  (2026-11-07)
- Status: ✅ done
- Built: caps enforced inside `CrawlEngine` (`rag_engine/crawling/engine.py`). Each cap stops the crawl **gracefully with a reason**: `CrawlSummary.stop: CrawlStop | None` (`reason: CrawlStopReason`, `detail`), with `CrawlStop` exported from `rag_engine.crawling`. `None` means the frontier drained.
  - **`max_pages`** (job config) is a **page budget**:
    - A page is charged once, right before its first fetch (`Page.charged`, new), and only after scope and robots allow it. Retries and rate-limit releases don't charge it again.
    - A resumed run reads the budget back with `frontier.count_charged()`. A redirect target reuses the alias's fetch and is not charged.
    - Spending the last page is a **soft** stop. Queued pages that aren't charged become `skipped/max_pages` in bulk. Pages found later are skipped one at a time as they're claimed, with no robots lookup. Charged pages that are waiting for a retry still finish. Sitemap discovery is cancelled.
  - **`CRAWL_MAX_TOTAL_BYTES`** is a **hard** stop. It's tracked by the new `JobCounters.bytes_downloaded`, which is incremented on every successful fetch and persisted, so a resume continues from it. It's checked after each fetch, so it can be overshot by the pages in flight.
  - **`CRAWL_MAX_JOB_SECONDS`** is a **hard** stop, measured from the job's first `started_at` (kept across requeues, Step 49). Idle workers wake at the deadline, so a long pending retry doesn't hold the job open.
  - **On a hard stop** nothing more is fetched:
    - Workers exit and in-flight pages finish (bounded by the page deadline).
    - A page claimed after the stop, or whose time ran out while it waited for a rate-limit slot, is released.
    - `_skip_rest` records everything still queued, pending retries included, as `skipped/max_total_bytes` or `skipped/max_duration`.
  - Every cap is re-checked when a run starts, so a resumed job that's already over a cap stops right away without fetching.
  - Precedence: the first cap hit is the reason, except that a hard stop overrides a soft one.
  - **Per-page deadline** `CRAWL_PAGE_TIMEOUT_SECONDS` (new, 150 s) covers the rate-limit wait plus fetch plus render. It sits on top of the fetchers' own timeouts, so the browser's `before_render` slot wait is bounded too. When it fires, the fetch fails with `FetchTimeoutError` (detail `page deadline of N s exceeded in the fetch | the rate-limit wait`) and goes through the retry policy like any other timeout. Extraction keeps its own `EXTRACTION_TIMEOUT_SECONDS`. Releases and outcome writes run outside the deadline, so a timeout never cuts a DB write.
- Frontier: `charge(page_id)` (only while this run holds the page), `count_charged()`, `skip_queued(reason, detail, *, uncharged_only=False)`, and a shared `_held()` filter.
- Domain:
  - `CrawlStopReason` (`max_pages` is soft; `max_total_bytes` and `max_duration` are hard; `.page_reason`, `.is_hard`).
  - `PageReason` adds `MAX_TOTAL_BYTES` and `MAX_DURATION` (27 reasons).
  - `JobCounters.bytes_downloaded` and `Page.charged`.
- Settings: `CRAWL_PAGE_TIMEOUT_SECONDS`. Validators: page ≥ fetch + render, and lease > page + extraction. This replaces the old "lease > fetch + render" rule. `.env.example` updated.
- Tests:
  - `TestCrawlCaps_S51` (8, integration, scripted):
    - The page budget gives exactly 4 fetches, the other 7 are `max_pages` with no robots lookup, and the counters agree.
    - A charged page's retry still succeeds after the budget is spent, and its new link is skipped.
    - One budget holds across a crash and resume (5 distinct URLs fetched over both runs).
    - Byte budget: the crawl stops on the fetch that crosses it, `bytes_downloaded` equals the sum fetched, a retry pending 60 s is skipped, and the job doesn't wait for it.
    - Wall time (1 s): the job stops in under 3 s despite a pending 30 s retry, and nothing is left queued or in progress.
    - A resumed job past its time or over its byte budget stops with zero fetches.
    - Page deadline: a hanging fetch and a hanging rate-limit wait each fail `timeout` after 2 attempts with the stage in the detail, and the other pages succeed.
  - `TestFrontierCaps_S51` (3, integration): `charge` is held-only and survives a release; `count_charged` is per job; `skip_queued` (soft keeps charged/in-flight pages, hard takes pending retries, detail capped at 500 characters).
  - `TestCrawlCapsSettings_S51` (3, unit): default consistency and both validators.
  - Updated: `TestCrawlEngine_S50` counters include `bytes_downloaded`, `TestJobState` counter names, and `TestDomainModels_S13` (25 → 27 reasons).
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1417 passed, 4 skipped (container) in 196 s. The S50 + S51 engine tests passed 3 runs in a row (~33 s).
- Decisions/notes:
  - The byte and time caps are global settings. Per-tenant limits arrive with the quotas (Step 84), which only need to feed `CrawlEngine` a different value. `max_pages` is already per job.
  - The engine **does not persist** the stop reason on the job. A resume re-detects every cap from persisted state (charged pages, the bytes counter, `started_at`). **Step 53** must put `summary.stop` into the job report. **Step 77** decides the job outcome: a capped crawl still finalizes (it's a partial success, not a failure).
  - When a stop cancels sitemap discovery before it reported, `summary.sitemap_error = "stopped early: <reason>"`.
  - **Step 52 (cancel)**: add `CrawlStopReason.CANCELLED` (hard, `page_reason` → `job_cancelled`) and call `engine._stop_crawl(...)` from the cancel poll (expose it publicly then). `_skip_rest` then records the queue as `skipped/job_cancelled`. In-flight pages drain exactly as with the other hard stops.
- Next: Step 52: Cancellation

### Step 52: Cancellation  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/crawling/cancellation.py` (exported from `rag_engine.crawling`), plus a cancel poll in `CrawlEngine`.
  - **MongoDB stays the source of truth** (`cancel_requested_at`, Step 49). The Redis flag `job_cancel:<job_id>` (`RedisNamespace.JOB_CANCEL`) only lets the engine notice quickly and cheaply.
  - `RedisCancelFlags(redis, keys, config)`: `request` (SET with `CRAWL_CANCEL_FLAG_TTL_SECONDS`), `is_cancelled` (EXISTS), `clear` (DEL), `key()`. Redis/OS errors raise `RedisUnavailableError`.
  - `CancelSignal` protocol (`is_cancelled(job_id)`), so tests can script the signal.
  - `request_cancel(jobs, flags, tenant_id, job_id)` (the API helper, for Step 87): calls `jobs.request_cancel` first, so a queued job is cancelled immediately with no flag. A running job is stamped in Mongo, then flagged in Redis. It's idempotent, and finalizing/finished jobs are refused. **A Redis outage doesn't fail the request**: it logs a warning and the worker's heartbeat picks up `cancel_requested_at` instead.
  - `finish_cancelled(jobs, flags, job_id, run_id)` (the worker helper, for Step 77): `jobs.cancel` (running → `cancelled`), then clears the flag best-effort (it expires anyway).
  - **Engine**:
    - New `cancel: CancelSignal | None` collaborator.
    - A poller task runs alongside the workers and checks the signal every `CRAWL_CANCEL_POLL_SECONDS`. `run()` waits on the workers and then cancels the poller, so polling never delays the end of a crawl.
    - Public `engine.cancel(detail=...)` (True the first time) is for the heartbeat path in Step 77.
    - A job whose cancel was requested before the run started (`job.cancel_requested_at`, or the flag) stops without fetching.
    - A Redis outage during polls is logged once per outage and polling continues.
  - **Stop semantics**: `CrawlStopReason.CANCELLED` is hard, and `page_reason` now comes from a mapping (→ `job_cancelled`). The new `rank` orders soft (0) < hard cap (1) < cancelled (2), and a stop replaces only a lower-ranked one: a cancel wins over any cap, so the job ends `cancelled`. The drain reuses Step 51's hard-stop path:
    - Idle workers wake, in-flight pages finish normally (bounded by the page deadline), and their outlinks get queued.
    - `_skip_rest` records everything still queued, pending retries included, as `skipped/job_cancelled` with the stop detail.
    - Sitemap discovery is cancelled (`sitemap_error = "stopped early: cancelled"`).
- Settings:
  - `CRAWL_CANCEL_POLL_SECONDS` (1.0, in (0, 60]).
  - `CRAWL_CANCEL_FLAG_TTL_SECONDS` (86400; the validator requires ≥ `CRAWL_MAX_JOB_SECONDS`).
  - `.env.example` updated.
- Tests:
  - `TestCrawlCancellation_S52` (6, integration):
    - Full flow with real Redis: the API helper fires during a slow fetch.
      - Every page whose fetch had started finishes `succeeded`, including the slow one with its Markdown.
      - The slow page's late link and all unfetched pages become `job_cancelled`; nothing is left queued or in progress.
      - `finish_cancelled` leaves the job `cancelled`, the skipped counter matches, and the flag is cleared.
    - A cancel requested before the run (Mongo only) gives zero fetches.
    - A cancel outranks a spent page budget and wakes workers waiting on a 30 s retry (run < 3 s).
    - `engine.cancel()` returns True, then False.
    - 3 failed polls are tolerated and the cancel still lands.
    - With a 30 s poll interval the crawl still ends at once (one poll: the start check).
  - `TestCancelFlags_S52` (5, integration: Redis + Mongo): flag round trip with TTL and the namespaced key; running job → Mongo + flag (first `cancel_requested_at` kept); queued job → cancelled with no flag; finalizing and other tenants refused with no flag; `finish_cancelled` ends the job and clears the flag.
  - `TestCancellation_S52` (5, unit): stop ranking and page reasons; settings validators; typed flag errors; `request_cancel` and `finish_cancelled` survive a Redis outage.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1433 passed, 4 skipped (container) in 230 s. The engine and cancellation integration tests passed 3 runs in a row (~40 s).
- Decisions/notes:
  - The engine still makes no job transitions. **Step 77 (worker)**:
    - Build `RedisCancelFlags` and pass it as `cancel=`.
    - On every heartbeat, if `job.cancel_requested_at` is set, call `engine.cancel()` (this covers a lost or expired flag).
    - After `run()`: if `summary.stop.reason is CANCELLED`, or the job has `cancel_requested_at` (a cancel that arrived just as the frontier drained), call `finish_cancelled`. Otherwise call `begin_finalizing`.
  - **Step 78**: a cancelled job deletes its new version's points; the old version stays active.
  - **Step 87**: `DELETE /jobs/{id}` → `request_cancel(...)`. It's already idempotent and tenant-scoped, and maps `InvalidJobTransitionError` to 409 and `JobNotFoundError` to 404.
  - Step 80's `requeue` already cancels (instead of requeuing) a dead run's job when a cancel is pending, so the flag can be left to expire there.
- Next: Step 53: Engine error accounting

### Step 53: Engine error accounting  (2026-11-07)
- Status: ✅ done
- Built: every page ends as a row with a classified reason, and each run stores a `CrawlReport` (the failure summary) on the job.
  - **Gaps found and closed**:
    - **Redis outages** (robots store, rate limiter, the render-slot hook) reached the engine as raw `RedisError` and became `internal_error`, as if they were bugs. `CrawlEngine._guarded` now classifies `RedisError`/`RedisUnavailableError`: the page is retried with the policy's backoff while it has attempts left (`frontier.max_attempts`), then fails as **`service_unavailable`** (new) with `... (gave up after N attempts)`. The seed's robots verdict for the report tolerates an outage (`robots_status=None`, detail `robots.txt verdict unavailable: ...`), so the run carries on.
    - **Redirect targets** processed with their alias's content had no guard of their own: an exception left the target `in_progress` until its 10-minute lease expired, which kept the frontier undrained. They now go through `_guarded` too (`_redirect_target`).
    - **`render_failed`** was never recorded: AutoFetcher falls back to the HTTP shell, which then showed up as `skipped/empty_content`. An `EMPTY_CONTENT` extraction on a result whose `render_note` starts with `render_failed` is now `failed/render_failed` (detail: the note plus "the HTTP response has no content").
    - **Counter drift**: pages the frontier fails as `worker_lost` (`recover()`, expired-lease abandonment) never passed through `_finish`, so `failed` undercounted. A crash between an outcome and its `$inc` drifts the counters too. At the end of each run the engine recounts `failed`/`skipped`/`duplicates` from the page table and writes them with the new `JobRepository.reconcile_counters` (an `$set` of the named fields only, so `chunked`/`embedded` from the pipeline are untouched). It logs any drift and only writes when it finds some.
    - **Failed jobs**: the new `PageRepository.close_out_job(job_id, detail) -> (skipped, failed)` turns queued pages into `skipped/job_failed` and in-progress pages into `failed/job_failed` (new reason, 29 in total). Finished pages are untouched. It's for Step 77/80 when a job fails.
  - **Report**:
    - Domain `rag_engine/domain/report.py`: `CrawlReport` holds `generated_at`, `seed_url`, `stop_reason`/`stop_detail`, `pages` by status, and `failures` and `skips` (each a list of `ReasonCount{status, reason, count, samples≤5}`, largest first; `PageSample{url, detail, http_status}`). It also holds `robots_status`/`detail`, `sitemap_urls`/`sitemap_error`, `skipped_links`(`_dropped`), `recovered_pages`/`abandoned_pages`, and the properties `failed` and `unfinished`. `CrawlJob.report` defaults to None.
    - `PageRepository.reason_breakdown(job_id, samples=5)` is one aggregation that groups by (status, reason) and samples the first pages found (`$topN` by `_id`, Mongo ≥ 5.2; we run 8.0).
    - `rag_engine/crawling/report.py`: `build_report(pages, job_id, ...)` is exported, so the worker can build a report without the engine.
    - At the end of `run()`, the engine reconciles, builds and stores the report (`JobRepository.set_report`, guarded by the run, replacing the earlier run's) and returns it as `CrawlSummary.report`.
- Tests:
  - `TestEngineErrorAccounting_S53` (5, integration):
    - One page per failure kind gives the exact (status, reason) and a non-empty detail. The kinds: 404, 500 exhausted, 429, timeout, network, SSRF, too large, redirect loop, too many redirects, unsupported type (skip), empty (skip), render failed, corrupt PDF → extraction_failed, a bug → internal_error, Redis down → service_unavailable after 2 attempts. The report's failures and skips match per reason, the samples carry URL and HTTP status, the stored report equals the returned one, and the counters match.
    - A one-off Redis blip is retried and the page succeeds on attempt 2.
    - A redirect target whose content crashes ends `internal_error` in under 5 s, with nothing unfinished.
    - Reconciliation: a `worker_lost` page counts as failed, an inflated `skipped` is corrected, and `abandoned_pages=1`.
    - A robots-store outage fails the seed as `service_unavailable` with no fetch, and the report notes the robots verdict was unavailable.
  - `TestPageAccounting_S53` (2, integration): breakdown order, counts, the first 5 samples, `samples=` override, per job; `close_out_job` reasons, detail cap, finished/other-job pages untouched, idempotent.
  - `TestJobReport_S53` (3, integration): the report round trips and is replaced, it's guarded by the run, and `reconcile_counters` only touches the named counters (plus validation).
  - `TestCrawlReport_S53` (3, unit): totals, sample cap, job round trip.
  - `TestDomainModels_S13`: 27 → 29 reasons.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1446 passed, 4 skipped (container) in 218 s. The integration suite passed twice in a row after the fix below.
- Decisions/notes:
  - A pre-existing race (Step 50 design, left as is): with several workers, another worker can claim a redirect target between `_redirected`'s `add` and `claim_url`, and then fetches it directly. The outcome is still correct (the alias stays `skipped/redirected` and the target keeps its own content), at the cost of one extra request. The redirect-target test pins `CRAWL_CONCURRENCY=1` for this reason.
  - The report is per run and replaced on a resume. Counts come from the page table, so they cover every run.
  - **Step 77**:
    - On a job failure, call `jobs.fail(...)`, then `pages.close_out_job(job_id, detail)`, then optionally `build_report(...)` + `set_report` (only while the run still holds the job; `fail` releases it, so build and store the report *before* `fail`, or add a system-level setter then).
    - `CrawlSummary.report` is also what the completion webhook can carry.
  - **Step 85**: `GET /jobs/{id}` serves `job.report` as the failure summary. **Step 86** lists every page by status and reason.
- Next: Step 54: Crawl engine E2E

### Step 54: Crawl engine E2E  (2026-11-07)
- Status: ✅ done
- Built: `tests/browser/test_crawl_e2e.py`. It crawls the whole fixture site through `CrawlEngine`, **wired as production will be**: real headless Chromium (`BrowserPool.job_context` around the job, then `AutoFetcher(http, browser, before_render=render_politeness(robots, limiter))`), `RobotsCache(RedisRobotsStore)`, `DomainRateLimiter`, `SitemapDiscoverer`, `ContentClaimRepository` and `RedisCancelFlags`, over real MongoDB and Redis. The `_Crawl.run()` helper is effectively a draft of Step 77's wiring.
  - Test infra:
    - The container fixtures moved from `tests/integration/conftest.py` to the pytest plugin `tests/support/containers.py`, registered in `tests/conftest.py` (`pytest_plugins`). Browser tests can now use `integration_settings`, and containers still start lazily once per session.
    - The S50 outcome check became the shared `assert_crawl_outcomes(site, outcomes, *, rendered)` in `tests/support/fixture_site.py`. S50 uses `rendered=False`; the E2E uses `rendered=True`, which also requires the SPA and its JS-only link.
    - `tests/README.md` updated.
  - The module is marked `integration` as well as `browser` (from its directory). It's skipped when Docker or Chromium is missing.
- Tests: `TestCrawlEngineE2E_S54` (2):
  - **Full site**:
    - Every `SITE_PAGES` URL matches `EXPECTED_PAGE_OUTCOME` and nothing is left queued or in progress.
    - `/app/` is rendered by the browser (`rendered_with=browser`, "Widget configurator" in its Markdown, the `empty_mount` signal), and it's the **only** rendered page. `/app/advanced`, linked only from the rendered DOM, succeeds with `discovered_from=/app/`.
    - The plan's named cases: the loop fails `redirect_loop`, `/missing.html` fails 404, the PDF is extracted ("1200 items per minute"), the off-site page is `out_of_scope`, and `/private/` is `robots_disallowed`.
    - Requests: `/private/` is never requested, `/app/` is requested twice (HTTP plus the browser render) and `/app/advanced` once.
    - The stored report equals the summary's, matches the page table and has no unfinished pages. The robots verdict is OK, sitemap URLs were found, failures by reason exactly match the failures expected by the fixture contract, the counters match, and `bytes_downloaded` > 0.
  - **Cancel through the API mid-crawl**: after about 12 requests, `request_cancel` (Mongo + Redis flag) gives `stop=cancelled`. Then `finish_cancelled` leaves the job `cancelled` and clears the flag. Nothing is left queued or in progress, the `job_cancelled` pages were never fetched, in-flight pages drained to `succeeded`, and `report.stop_reason=cancelled`.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1448 passed, 4 skipped (container) in 216 s. The E2E passed 3 runs in a row (~31–37 s for both tests).
- Decisions/notes:
  - The E2E needed no engine changes: Steps 50–53 handled the browser-rendered site as is.
  - **Step 77**: lift `_Crawl.run()`'s wiring into the task. Enter `BrowserPool.job_context(job_id=...)` around the engine run, and pass `render_politeness(robots, limiter)` as `before_render`.
  - Phase F (Crawl Engine, 47–54) is complete. Phase G (Chunking) starts next.
- Next: Step 55: Chunker framework

### Step 55: Chunker framework  (2026-11-07)
- Status: ✅ done
- Built: the `rag_engine/chunking/` package.
  - `base.py`:
    - `Tokenizer` protocol (`name`, `count(text)`). tiktoken per model encoding is Step 57.
    - `ChunkingConfig(target_tokens, max_tokens, overlap_tokens)`, validated, with `.from_settings()`.
    - `ChunkInput.from_page(page, markdown=None)`: succeeded pages only (`ValueError` otherwise). It carries the storage ids, with `version = page.job_id`. `markdown=` takes the job-level boilerplate-stripped text (Step 40).
    - `ChunkPiece(text, heading_path, page_number)`: what a strategy outputs.
    - `ChunkDraft(ordinal, text, embed_text, heading_path, token_count, content_hash, page_number)` with `.to_document(source) -> Chunk`.
    - `content_hash()`.
  - **`BaseChunker`** is a template method. Subclasses implement only `split(source) -> Iterable[ChunkPiece]` (plus the `embed_text(source, piece, text)` hook, which defaults to the text and is finalized by Step 58). The final `chunk()` guarantees, for every strategy:
    - blank lines around a piece are trimmed (indentation kept, for code) and empty pieces dropped;
    - dense ordinals from 0;
    - `token_count` is counted on **embed_text**, the string the embedder receives, so Step 62 can pack batches by token budget without recounting;
    - over `max_tokens` raises `ChunkTooLargeError` (a splitter bug, never silently truncated); 0 tokens for non-empty text raises `TokenizerError`;
    - `content_hash = sha256(embed_text)`, which the embedding cache can key on (Step 64);
    - heading paths are capped at `MAX_HEADING_DEPTH=12`, keeping the outermost levels.
  - `factory.py`: `@register_chunker(ChunkerType.X)` (refuses non-`BaseChunker` classes and a second class for the same kind; registering the same class again is fine), `registered_chunkers()`, `get_chunker(tokenizer, kind=None, *, config, chunking=None)`. An unregistered kind raises `UnknownChunkerError`, listing the registered ones.
  - `exceptions.py`: `ChunkingError` → `UnknownChunkerError`, `ChunkerRegistrationError`, `ChunkTooLargeError(url, ordinal, token_count, max_tokens)`, `TokenizerError`. These are plain exceptions, not API errors: they mean a bug or misconfiguration.
- Config:
  - `ChunkerType` StrEnum (`structure`), exported from `rag_engine.config`.
  - Settings `CHUNKER` (structure), `CHUNK_TARGET_TOKENS` (400), `CHUNK_MAX_TOKENS` (800, ≤ 8000, under the models' 8191 input limit) and `CHUNK_OVERLAP_TOKENS` (60). Validators: target ≤ max and overlap < target. `.env.example` has a chunking section.
- Domain: `Chunk.page_number: int | None` (> 0), for the PDF page a chunk starts on (Step 56). `token_count` is documented as tokens of `embed_text`.
- Tests: `TestChunkerFramework_S55` (20, unit):
  - input from a succeeded page / override / non-succeeded refused;
  - dense ordinals with empty pieces dropped, indentation kept, page numbers passed through;
  - tokens and hash describe embed_text via a contextual subclass; same embed_text gives the same hash;
  - heading-path cap; oversized → `ChunkTooLargeError` with its fields; zero-token tokenizer refused; no pieces → no chunks;
  - draft → a valid `Chunk` that round-trips through BSON; `page_number` must be positive;
  - config from settings plus validation; 5 inconsistent settings rejected;
  - factory default/explicit budget, unregistered strategy error, registration checks.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1468 passed, 4 skipped (container) in 215 s.
- Decisions/notes:
  - **No strategy is registered yet**: `get_chunker()` raises `UnknownChunkerError` until Step 57 registers `StructureChunker` under `ChunkerType.STRUCTURE`. Tests register a fake under an isolated registry.
  - **Step 56** (parser) should output sections with heading paths and atomic blocks, and map `<!-- page N -->` markers to `ChunkPiece.page_number` (never to content).
  - **Step 57**: a tiktoken `Tokenizer` per `EmbeddingModelSpec.tokenizer_encoding`. The splitter budgets text against `target_tokens` but must keep embed_text ≤ `max_tokens`, so it should reserve room for Step 58's context (title + heading path).
  - **Step 58**: override `embed_text` in the base class (or the structure chunker); `text` stays clean for display.
  - **Step 59**: persist with `draft.to_document(source)`. `(page_id, ordinal)` is already unique.
- Next: Step 56: Markdown structure parser

### Step 56: Markdown structure parser  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/chunking/markdown_parser.py`, exported from `rag_engine.chunking`. `MarkdownParser().parse(md)` / `parse_markdown(md)` return a `ParsedDocument(sections, location_kind)`.
  - Parsing uses **markdown-it-py** (CommonMark plus GFM `table` and `strikethrough`). Only **top-level tokens and their source line ranges** (`token.map`) are used, so a block's `text` is its exact Markdown source (blank lines around it trimmed), and a `#` in code or a table cell is never a heading.
  - **`Section(heading_path, level, heading, heading_markdown, location, blocks)`**:
    - The path is the titles of the headings above, outermost first, as plain text: `heading_text()` drops markup, link targets and image URLs, and keeps alt text.
    - A deeper heading nests; an equal or shallower one closes the open sections. Skipped levels are fine, and setext headings work.
    - Content before the first heading is the preamble (level 0, empty path), dropped when empty. Headings without content are kept as empty sections, so the chunker decides what to do with them.
  - **`Block(kind, text, location, line, header, footer, parts, info)`**:
    - Kinds `paragraph`, `quote`, `html` (raw HTML that isn't a comment) are prose. **Atomic** kinds are `table`, `code` (fenced or indented) and `list`.
    - `parts` tells Step 57 where an oversized atomic block may be cut:
      - table: `header` (header + separator rows) and `parts` (one per row);
      - fenced code: `header` (the opening fence), `footer` (the closing fence, or None when the fence never closed, which runs to the end), `parts` (body lines, blank lines kept) and `info` (language);
      - indented code: just `parts` (its lines);
      - list: `parts` (top-level items with their nested content).
  - **Location markers**: `<!-- page|slide|sheet N -->` (case-insensitive) are never content. They set `location` on the blocks and headings that follow, and `ParsedDocument.location_kind` records which kind the document uses. Other HTML comments and thematic breaks (`---`, `***`) are dropped. CRLF/CR are normalized.
- Deps: `markdown-it-py` 4.2.0 (runtime; it adds `mdurl`, nothing else changed in the lock).
- Tests: `TestMarkdownParser_S56` (20, unit):
  - Sections: hierarchy with sibling closing, skipped levels, setext, plain heading text, empty headings kept / empty preamble dropped / empty doc, `#` in code and tables.
  - Blocks: all 8 kind cases with atomicity; table header and rows (escaped `\|` kept); `~~~~python` fence with header, footer, info and lines; unclosed fence; indented code; list items with nested children.
  - Non-content: page markers → locations (blocks and headings, none leak into text); the slide/sheet/upper-case variants; comments and `---`/`***` dropped; line numbers with CRLF.
  - **Real extractor output** (fixture documents through `default_registry`): the PDF has pages {1,2} with the spec line on page 1; PPTX slide sections are in location order; XLSX tables carry sheet locations {1,2}, each with a header and rows.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1488 passed, 4 skipped (container) in 204 s.
- Decisions/notes:
  - **`Chunk.page_number` is the document location** in general: PDF page, PPTX slide or XLSX sheet number (`ParsedDocument.location_kind` says which). Step 57 sets `ChunkPiece.page_number` from the location of a piece's first block, or the section's location for a heading-only piece.
  - **Step 57**: when splitting, re-emit `header` (+ `footer`) around each slice of `parts`. For an unclosed fence, close it with the opening fence's backticks/tildes.
  - **Step 58**: `heading_path` is plain text ready for the embed-text prefix. `heading_markdown` is available if `text` should show the heading line.
- Next: Step 57: Token-aware splitter

### Step 57: Token-aware splitter  (2026-11-07)
- Status: ✅ done
- Built:
  - `rag_engine/chunking/tokenizer.py`:
    - `TiktokenTokenizer(encoding)` implements `Tokenizer` and adds `encode`/`decode`. Special-token text on a page (`<|endoftext|>`) is counted as plain text, the way the embeddings API counts it.
    - `tokenizer_for(model)` returns **one shared instance per encoding**, from `EmbeddingModelSpec.tokenizer_encoding` (`cl100k_base` for both 3-series models). An unknown model raises `UnsupportedEmbeddingModelError`.
  - `rag_engine/chunking/structure_chunker.py`: **`StructureChunker`**, registered as `ChunkerType.STRUCTURE`, so `get_chunker(tokenizer)` now works with the default settings. `sentences()` is exported. Per section of the Step 56 parse:
    1. **Units**:
       - A block that fits is one unit, so paragraphs, tables, code and lists stay whole.
       - A prose block over the target is cut **sentences → lines → words → characters**, recursively, until each piece fits the cap. Characters are cut by a binary search on the token count, which handles e.g. a base64 blob.
       - An **atomic** block is cut only when it's over the cap, and then along its parser `parts`: a table by rows with the header and separator repeated on every piece; fenced code by lines re-wrapped in the original fence (an unclosed fence gets a matching closing fence; an over-long line is cut inside the fence); indented code by lines; a list by top-level items (an item over the cap falls back to prose cutting).
       - Pathological wrappers (a header or fence that leaves no room) fall back to prose cutting.
    2. **Packing**: greedy up to `target_tokens`. An atomic block between the target and the cap becomes its own chunk. **Sections are never merged** (one heading path per chunk), and a heading-only section yields no chunk.
    3. **Overlap**: only where a cut falls **inside a prose block**. The next chunk starts with the previous chunk's last whole sentences, up to `overlap_tokens`, and only if that doesn't push it over the cap. Block, paragraph and section boundaries are never overlapped, and neither is atomic content.
    - **Budgets**: the target is measured on `text`. The cap reserves the room the `embed_text` context takes for the section (measured through the `embed_text` hook), so text + context ≤ `max_tokens` holds by construction. If the context alone is over the cap, it raises `ChunkingError`.
    - `page_number` = the location of a chunk's first own (non-overlap) unit, else the section's. Heading lines aren't repeated in `text`.
    - `sentences()` handles common abbreviations (e.g., i.e., etc., Fig., Mr., months, …), initials, decimals, closing quotes/brackets, ellipses and the CJK full stops. Its regexes use `\u` escapes (ruff RUF001).
- Deps: **`tiktoken` 0.14.0** (runtime; nothing else in the lock). The first use downloads the encoding (~4 s) into `TIKTOKEN_CACHE_DIR` (default: the system temp dir).
- Tests: `TestTokenAwareSplitter_S57` (24, unit, real tiktoken):
  - Tokenizer: counts match cl100k_base, round trip, shared instance, unknown model, special-token text.
  - Structure is the default chunker; the sentence cases.
  - Sections and packing: small sections one chunk each with no headings in the text; paragraphs packed ≤ target, whole, in order, no duplication; long prose split at sentences with whole-sentence overlap ≤ budget and nothing lost; no overlap across blocks or sections; overlap off.
  - Atomic blocks: a table between target and cap stays whole; an oversized table repeats its header with every row exactly once; oversized code stays fenced with indentation intact; an unclosed `~~~~` gets closed; a list splits between items.
  - Last resorts: unpunctuated lines split by line; a 3000-char blob cut by characters and losslessly rejoined.
  - Budgets and locations: a long context prefix is reserved under the cap; chunks carry their first block's page.
  - 25 seeded random documents: ordinals dense, every chunk within 0 < tokens ≤ cap, every source word kept.
  - The 4 fixture document types chunk within the default budget, with no markers leaking and page numbers exactly for PDF/PPTX/XLSX.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1512 passed, 4 skipped (container) in 207 s.
- Decisions/notes:
  - Overlap is limited to cuts inside a block on purpose: overlapping across a paragraph boundary duplicated whole short paragraphs into the next chunk, which costs embedding tokens and adds near-duplicate results for no lost context.
  - Small sections stay small chunks (an FAQ entry, a slide): better precision and exact locations. Step 60's golden tests will show whether merging tiny sibling sections is worth adding.
  - **Step 58**: override `embed_text` (title + heading path + text). The splitter already reserves its room. Keep the context bounded (e.g. cut long titles/paths) so `_limit` never goes below a useful size.
  - **Step 95**: bake the tiktoken cache (`TIKTOKEN_CACHE_DIR`) into the images.
  - **Step 76**: chunk with `get_chunker(tokenizer_for(kb.embedding_model))`.
- Next: Step 58: Contextual embed text

### Step 58: Contextual embed text  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/chunking/context.py` (`build_context`, `compose_embed_text`, exported), wired into **`BaseChunker`**, so every strategy gets it.
  - `BaseChunker.context(source, heading_path)` builds the context. The default `embed_text` hook returns `context + "\n\n" + text`, or just the text when there's no context. **`text` stays clean** for display.
    ```
    Install guide | Acme Widgets      <- page title
    Getting started > Linux           <- heading path, outermost first (" > ")

    Run the installer.                <- chunk text
    ```
  - **Title repeats are dropped**: a leading heading equal to the title, or to one whole part of it, is not repeated. The comparison is case/space-insensitive, and parts are split on `|`, `:`, `·` or a *spaced* hyphen/en/em dash, so a hyphen inside "WX-2048" never splits.
  - **Bounded** by `ChunkingConfig.context_tokens`. Over budget, it collapses the middle of a deep path to `first > … > last`, then shortens the title to word boundaries with "…", then shortens the headings, then falls back to the title alone, else nothing. A budget of 0 disables the context.
  - `ChunkingConfig.context_tokens` defaults to `min(100, max - target)`, and the validator requires ≤ max − target. So the context can never squeeze the chunk text below the target, and Step 57's reservation always leaves room for a full target-sized chunk.
- Settings: `CHUNK_CONTEXT_MAX_TOKENS` (100, ≥ 0, ≤ `CHUNK_MAX_TOKENS − CHUNK_TARGET_TOKENS`). `.env.example` updated.
- Tests: `TestContextualEmbedText_S58` (26, unit, real tiktoken):
  - Format; missing title, path or blank headings; 8 title-repeat cases (including the WX-2048 hyphen and a partial-phrase non-match).
  - Budget: a deep path collapses its middle first, a long title is shortened with "…", long headings are shortened, budget 0 is off, and **200 random title/path/budget combinations always fit**.
  - In the chunker: exact embed vs display text with the h1 dedupe; `token_count` == tokens of embed_text; the same text in different sections gets different hashes; no title or headings → embed_text == text; long headings and title never break the cap; the stored `Chunk` keeps both texts.
  - Config default/derived/explicit/validation; settings validation.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1538 passed, 4 skipped (container) in 212 s. The Step 55/57 tests needed no changes.
- Decisions/notes:
  - `content_hash` covers the context too, so identical text under different headings is embedded and cached separately (Step 64), which is correct because the vectors differ.
  - No location ("page 3") goes into the context: the embedder gains little from it. It stays in `Chunk.page_number` for citations.
  - **Step 59**: persist with `draft.to_document(source)`; both texts are already on the draft.
  - **Step 60** (golden tests): snapshot `embed_text` as well as `text`, so changes to the context format show up.
- Next: Step 59: Chunk persistence

### Step 59: Chunk persistence  (2026-11-07)
- Status: ✅ done
- Built: `rag_engine/repositories/chunks.py` → **`ChunkRepository`** (exported). Chunks are stored per `(source_id, version)` for audit, for re-embedding without re-crawling, and for cleanup by version.
  - **Deterministic chunk ids**: `chunk_id(page_id, ordinal)` = `chk_` + sha256(`page_id:ordinal`)[:32] (domain, exported). A page row belongs to one job, so the page id already implies the version. `ChunkDraft.to_document` sets it, and the new `BaseChunker.chunk_documents(source)` does chunk() + to_document() in one call.
  - `save_page_chunks(page_id, chunks) -> n`: **idempotent per page**. It `ReplaceOne(upsert)`s by id (an unordered bulk write), then deletes ordinals ≥ n, so re-chunking after a crash or with a new chunker leaves no duplicates or stale tails. An empty list removes the page's chunks. Validation runs before any write: all chunks of that page, one (source, version), ordinals 0..n-1 in order, and ids == `chunk_id`.
  - Reads: `for_page(page_id)` (in order), `iter_version(source_id, version, batch_size)` (streamed in page/ordinal order; carries `embed_text` and `token_count`, ready to re-embed), `count_version`, `versions(source_id)` (distinct, sorted; finds leftovers).
  - Cleanup: `delete_version(source_id, version)` (Step 78: failed/cancelled job or old version), `delete_source(source_id)` (Step 81). Mongo errors raise `DatabaseUnavailableError`.
  - Index: `source_version` was replaced by **`source_version_order`** `(source_id, version, page_id, ordinal)`, which serves the version filter and the streaming sort. `page_ordinal_unique` is unchanged.
- Tests: `TestChunkPersistence_S59` (9, integration, real chunker + Mongo):
  - A chunked page is stored and read back exactly (deterministic ids, contextual embed_text).
  - A re-save is idempotent.
  - Re-chunking replaces the page and drops a stale tail; an empty chunking removes only that page.
  - A version streams in page/ordinal order with small batches when saved out of order, isolated from another version.
  - Cleanup by version and by source, with `versions()`, leaves another source untouched.
  - 4 invalid writes are refused with nothing written; the indexes; chunk-id determinism.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1547 passed, 4 skipped (container) in 223 s.
- Decisions/notes:
  - A replace rewrites `created_at`/`updated_at` with the new chunking (the chunk was regenerated), which is intentional.
  - The old `source_version` index name may linger in an existing dev database. It's harmless; drop it by hand if wanted.
  - **Step 69**: derive Qdrant point ids as UUID5(version + chunk id). With deterministic chunk ids, re-ingesting a page upserts the same points. When a page re-chunks to fewer chunks, delete the points for the dropped ordinals too (mirror `save_page_chunks`).
  - **Step 76**: per succeeded page, `chunker.chunk_documents(ChunkInput.from_page(page, markdown=stripped))` → `save_page_chunks` → embed → upsert, then `jobs.increment(chunked=n)`.
  - **Step 81** (KB delete): iterate the KB's sources and call `delete_source`; a `kb_id` index isn't needed for that.
- Next: Step 60: Chunking golden tests

### Step 60: Chunking golden tests  (2026-11-07)
- Status: ✅ done
- Built:
  - `tests/unit/test_chunking_golden.py` → `TestChunkingGolden_S60`. **Snapshots of every fixture document type through the real ingest pipeline.** The fixture site is treated as one crawl job:
    1. every HTML page and document is extracted and cleaned (what the engine stores);
    2. the **job-level `BoilerplateIndex`** (Step 40) is built over all of them, hub page first;
    3. each golden document is stripped, cleaned again (Step 45) and chunked with `StructureChunker`.
  - Documents: 6 HTML pages (home, getting-started, configuration, API reference, blog post, Latin-1), the PDF, DOCX, PPTX and XLSX. Each is chunked at **two budgets**: the default settings, and a small one (60/160/20) that forces splitting. 20 golden files.
  - Golden format `tests/fixtures/golden/chunks/<doc>.<budget>.txt`: a header (URL, title, budget, chunk count), then per chunk `=== chunk N | heading path | location | tokens ===`, the embed **context** and the **text**, so diffs read like a review.
  - `tests/support/golden.py`: `assert_golden(path, actual, update=)` fails with a unified diff, or says the file is missing. The new root option `--update-golden` rewrites the files (root conftest plus `tests/README.md`).
- Tests (24): the 20 snapshots, each also asserting the contract (dense ordinals, 0 < tokens ≤ cap, embed_text = context + text, no markers); determinism; **site chrome never reaches a chunk** (the boilerplate index is active and the cookie banner is gone); the helper's diff/missing behaviour; every fixture type covered.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1571 passed, 4 skipped (container) in 218 s.
- Review of the snapshots (the point of the step):
  - **Found**: per-page extraction keeps the cookie banner on the home page (trafilatura dropped it elsewhere). My first harness chunked pages on their own and indexed "We use cookies…" as the home page's first chunk. The fix was in the test: the real pipeline runs the cross-page strip before chunking, and the harness now does too. It also confirms that **Step 76 must run the strip** (index over the job's pages, then `strip()` → clean → chunk); chunking stored page Markdown directly would index site chrome.
  - Everything else looked right: whole paragraphs, code and tables; tables a little over the target kept whole (pricing: 62 tokens at target 60); paragraph + code in one chunk at the default budget and separate at the small one; h1s that repeat the title are deduped from the context; PDF/PPTX/XLSX chunks carry their page/slide/sheet; Latin-1 decoded correctly.
  - **Small sections stay small** (FAQ-like sections, slides), and I'm not adding section merging. The chunks are precise and carry exact locations, and the cost (more, shorter vectors) is small. Revisit only if Step 100's retrieval evaluation shows a problem.
- **Phase G (Chunking, 55–60) is complete.**
- Next: Step 61: Embedder framework

### Step 61: Embedder framework  (2026-11-08)
- Status: ✅ done
- Built (in `rag_engine/embeddings/`, all exported):
  - **`base.py`**:
    - `BaseEmbedder(space, *, config, tokenizer=None)`, bound to one `EmbeddingSpace` (refuses a model of another provider with `UnsupportedEmbeddingProviderError`). Subclasses implement only `_embed(texts, input_type) -> ProviderResult(vectors, usage)`. The final `embed_documents(texts)` / `embed_query(text)` return an `EmbeddingBatch(vectors, space, usage)` (`.vector` for a query) and guarantee, for every provider:
      - an empty batch makes no provider call;
      - every input is non-empty and ≤ the model's `max_input_tokens` (counted with the model's tokenizer), else `EmbeddingInputError` (422) with the index, before anything is sent;
      - the response has **exactly one vector per input, in order, of the space's dimensions, all finite**, else `EmbeddingResponseError`;
      - usage (`EmbeddingUsageCounts(requests, tokens)`, addable) comes with every result.
    - `InputType` (document/query, for providers that distinguish them; OpenAI doesn't). `aclose()` and async context manager. `Vector` and `TokenCounter` types.
  - **`dispatcher.py`**: `@register_embedder(provider)` (checks the class is a `BaseEmbedder` whose `provider` matches, and refuses a second class for the same provider), `registered_embedders()`, `get_embedder(space, *, provider=None, config)`. It uses `ACTIVE_EMBEDDING_PROVIDER` and refuses a model of another provider or an unregistered provider (`UnsupportedEmbeddingProviderError`, 500).
  - **`exceptions.py`**:
    - `EmbeddingInputError` (422) and `UnsupportedEmbeddingProviderError`.
    - Provider failures are `EmbeddingError(UpstreamError)` with **`retryable`** and `retry_after` (which also sets a `Retry-After` header): `EmbeddingAuthError`, `EmbeddingRequestError` and `EmbeddingResponseError` are permanent (502); `EmbeddingRateLimitedError` and `EmbeddingUnavailableError` are retryable (503).
  - **Tokenizer moved** to `rag_engine/embeddings/tokenizer.py` (git mv). It's a model concern and the embedder validates with it. `rag_engine.chunking` re-exports `tokenizer_for`/`TiktokenTokenizer`, so nothing else changed.
- Test support: `tests/support/fakes.py` → **`FakeEmbedder`** (deterministic unit vectors per text via `fake_vector`; records calls; `respond` hook to return broken responses or raise; counts 1 request per call and 1 token per word; `closed` flag). Steps 64 and 76 use it.
- Tests: `TestEmbedderFramework_S61` (25, unit):
  - One vector per document in order (repeats are equal), the space, usage; a query `.vector` (256-d reduced space, unit norm); empty batch makes no call; usage adds up.
  - Empty/blank inputs and an over-limit input (> 8191 cl100k tokens, with exact token details) are refused with no call.
  - 5 broken responses (too few, too many, wrong dimensions, NaN, inf) → `EmbeddingResponseError`; provider errors pass through (Retry-After rounded up); the error taxonomy (status, code, retryable).
  - Context manager closes; a mismatched provider is refused.
  - Dispatcher builds/lists, unregistered is a typed error, registration checks, provider must match.
  - The moved tokenizer is still exported from chunking.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1596 passed, 4 skipped (container) in 229 s.
- Decisions/notes:
  - No provider is registered yet. `get_embedder()` raises until **Step 62** registers `OpenAIEmbedder` (`@register_embedder(EmbeddingProvider.OPENAI)`, imported from `embeddings/__init__`).
  - **Step 62**: batch inside `_embed` by `spec.max_batch_inputs` and `max_batch_tokens` (the base already guarantees each input fits on its own), pass `dimensions` only for reduced spaces, return vectors in input order, and sum the usage over sub-requests.
  - **Step 63**: map HTTP/transport failures to the taxonomy (401/403 → auth, 429 → rate limited with `Retry-After`/rate-limit headers, 5xx/timeouts → unavailable, other 4xx → request) and retry only `retryable` errors.
  - **Step 64**: the cache wraps an embedder, keyed `sha256(model|dims|embed_text)`, and only calls `embed_documents` for misses.
- Next: Step 62: OpenAI embedder

### Step 62: OpenAI embedder  (2026-11-08)
- Status: ✅ done
- Built: `rag_engine/embeddings/openai_embedder.py` → **`OpenAIEmbedder`**, registered for `EmbeddingProvider.OPENAI`, so `get_embedder(space)` now works out of the box. `retry_after_seconds` is exported.
  - **Plain async `httpx`, no SDK**: `POST {OPENAI_BASE_URL or https://api.openai.com/v1}/embeddings` with a Bearer key. Retries, backoff and concurrency stay entirely ours (Step 63). It takes an optional shared `client` (the caller owns and closes it); otherwise it creates one with `OPENAI_TIMEOUT_SECONDS` and closes it in `aclose()`.
  - **Batching**: in order, greedy, at most `min(EMBEDDING_MAX_BATCH_INPUTS, spec.max_batch_inputs)` inputs and `min(EMBEDDING_MAX_BATCH_TOKENS, spec.max_batch_tokens)` summed tokens per request, counted with the model's tokenizer. An input larger than a lowered token budget goes in a request of its own. Requests are sequential here; usage is summed (1 request each, billed `usage.prompt_tokens`, falling back to `total_tokens`, else 0).
  - **Request**: `{model, input, encoding_format: "float"}`, plus **`dimensions` only for reduced spaces**, so native-size spaces also work with proxies that lack the parameter.
  - **Order**: vectors are reassembled by `index`, which must be exactly 0..n-1 with no repeats. Items must be lists of numbers (bools refused). The Step 61 base then checks the dimensions and that values are finite.
  - **Failures → Step 61 taxonomy** (single attempt):
    - timeouts and transport errors → `EmbeddingUnavailableError`; 5xx → `EmbeddingUnavailableError` (with Retry-After when given);
    - 429 → `EmbeddingRateLimitedError` with `retry_after` from `retry-after-ms`, else `Retry-After` (seconds or HTTP date); **429 `insufficient_quota`** (billing) → `EmbeddingRequestError`, which is permanent;
    - 401/403 → `EmbeddingAuthError`; other 4xx → `EmbeddingRequestError`;
    - details carry `status` and OpenAI's `error_code`; messages include OpenAI's message (whitespace-collapsed, ≤ 300 chars) and never the key;
    - a malformed 2xx → `EmbeddingResponseError`.
- Settings: `EMBEDDING_MAX_BATCH_INPUTS` (256) and `EMBEDDING_MAX_BATCH_TOKENS` (100k), capped by the model's 2048 / 300k. `.env.example` updated.
- Deps: **`respx` 0.23.1** (dev; nothing else in the lock). The test README already named it.
- Tests: `TestOpenAIEmbedder_S62` (33, unit, respx):
  - Request shape: auth header, exact JSON with no `dimensions` for native and with it for 256-d; base-URL override; order restored from a shuffled response.
  - Batching: by input count (7 → 3/3/1, vectors stitched, usage summed); by token budget (each request ≤ 50 real tokens, greedy split matches); an over-budget input goes alone; model limits cap the settings.
  - Failures: 9 HTTP cases (401, 403, 400, 404, 429 ms, 429 seconds, 429 quota, 500, 503 HTML with Retry-After) checked for exact type, `retryable`, `retry_after`, status/code details and no key in the message; 3 transport failures; 8 malformed 200s; usage fallback.
  - A shared client isn't closed, an owned one is; the dispatcher builds `OpenAIEmbedder`; Retry-After parsing (ms, seconds, negative, HTTP date, garbage, precedence, absent).
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1629 passed, 4 skipped (container) in 232 s.
- Decisions/notes:
  - The HTTP → error mapping landed here rather than in Step 63: every response needs a typed outcome. **Step 63** keeps the resilience policy: retry only `retryable` errors (up to `OPENAI_MAX_RETRIES`) with exponential backoff and jitter, using `retry_after` as a floor; send sub-batches concurrently under a semaphore of `OPENAI_MAX_CONCURRENT_REQUESTS` (one per process, shared across embedders); and keep results in order. `request()` is the unit to wrap.
  - `encoding_format=float` is chosen for simplicity. Base64 would cut response size by about 3×; revisit if bandwidth shows up in Step 94's metrics.
- Next: Step 63: OpenAI resilience

### Step 63: OpenAI resilience  (2026-11-08)
- Status: ✅ done
- Built: `rag_engine/embeddings/resilience.py` (provider-agnostic, exported), wired into `OpenAIEmbedder`.
  - **`EmbeddingRetryPolicy`** (`from_settings`):
    - Only `retryable` errors (429, 5xx, timeouts/transport) are retried, at most `OPENAI_MAX_RETRIES`.
    - The delay before retry n is in [d/2, d], d = min(max, base·2^(n-1)) (equal jitter, like the crawl policy), and the provider's `retry_after` is a **floor** on it.
    - A requested wait over `EMBEDDING_RETRY_AFTER_MAX_SECONDS` fails at once instead of stalling the job.
    - Permanent errors (auth, request, billing quota, malformed response) are raised immediately.
  - **`call_with_retries(attempt, policy, gate, sleep)`**: the raised error gets `details["attempts"]`; each retry logs a warning (code, n/max, delay).
  - **`ProviderGate`** via `provider_gate(provider, limit)`, **one per (provider, limit) per event loop**: Celery tasks may each run their own loop, and an asyncio semaphore is bound to its first loop; the gates are held in a `WeakKeyDictionary` keyed by loop.
    - A semaphore of `OPENAI_MAX_CONCURRENT_REQUESTS` around each HTTP attempt only (never held while sleeping), shared by every embedder in the process.
    - A **shared pause**: a 429's wait pauses every request of the provider (the longest pause wins), so concurrent requests stop hammering a rate limit.
  - **`gather_in_order(calls)`**: runs the calls concurrently with results in input order. The first failure cancels the in-flight siblings and is raised **as itself** (no ExceptionGroup).
  - **OpenAIEmbedder**: sub-batches now go out concurrently through the gate, each wrapped in `call_with_retries`, and are stitched back in order (`policy=` and `sleep=` are injectable). On a 429 without `Retry-After`, the wait comes from the longest **`x-ratelimit-reset-requests/-tokens`** (`duration_seconds` parses "20ms", "6m0s", "1h2m3.5s"; `rate_limit_reset_seconds`).
- Settings: `EMBEDDING_RETRY_BASE_DELAY_SECONDS` (1), `EMBEDDING_RETRY_MAX_DELAY_SECONDS` (60; validator base ≤ max) and `EMBEDDING_RETRY_AFTER_MAX_SECONDS` (300). The existing `OPENAI_MAX_RETRIES` (6) and `OPENAI_MAX_CONCURRENT_REQUESTS` (4) are now in effect. `.env.example` updated.
- Tests:
  - `TestOpenAIResilience_S63` (20, unit, respx plus a recording fake `sleep`):
    - Policy: backoff ranges, growth, cap and real jitter; the decision table (permanent, exhausted, floor, too-long wait); `from_settings` and validation.
    - HTTP: 503 → 500 → 200 retried with the expected delays; timeouts retried; `retry-after-ms` 2.5 s honoured over backoff; reset headers → 90 s; `Retry-After: 600` fails at once (1 call, attempts=1); retries exhaust (4 calls, attempts=4); 401/400/insufficient-quota never retried.
    - Concurrency: 10 sub-batches with peak in-flight exactly 3 and input order kept despite uneven latency; the limit is shared across two embedders (peak 2); a permanent failure cancels slow siblings, which never finish.
    - Gate: a 429 pauses another caller; the longest pause wins; one gate per provider/limit/loop (a second loop gets its own).
    - Helpers: `gather_in_order`, duration parsing.
  - `TestOpenAIEmbedder_S62` now runs single-attempt (an autouse `OPENAI_MAX_RETRIES=0`): it covers one request's mapping, and the retries are tested here.
- Checks: ruff ✅ format ✅ ty ✅ pytest ✅ 1649 passed, 4 skipped (container) in 235 s.
- Decisions/notes:
  - Retry budget: with the defaults, a batch can spend up to ~2 min in backoff (1 + 2 + 4 + … capped at 60 s, 6 retries) plus provider waits ≤ 300 s each. That fits the 2 h job cap. Step 76 should treat a surfaced `EmbeddingError` as **job-fatal** only when it's permanent (auth, request); a retryable one that ran out of retries can fail the job with `embedding_unavailable` so Step 80's requeue can try later.
  - **Step 64** (cache) wraps an embedder and only sends misses through it, so cached chunks cost no requests or retries.
- Next: Step 64: Embedding cache
