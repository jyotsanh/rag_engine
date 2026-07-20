# RAG Engine

Give it a website link. It extracts, transforms, and loads the site's text, then answers questions over it with cited sources.

RAG Engine is a website-to-answers pipeline: a client (or an upstream service) supplies a URL, the engine crawls and cleans the text, embeds it into a vector store, and serves a retrieval-augmented question-answering API. Generation is grounded in the ingested content and returns citations back to the source pages.

> **Status:** greenfield / in active design. The architecture and roadmap below are tracked as issues on the [RAG Engine Roadmap](https://github.com/users/jyotsanh/projects/7) board. Nothing here is implemented yet — this README is the north star.

---

## How it works

```
                         POST /crawl {url}
   Swagger UI (/docs) ─────────────────────▶ FastAPI ──enqueue──▶ Redis
        ▲                                        │                  │
        │  GET /jobs/{id}  (poll progress)       └─ 202 {job_id}     ▼
        │                                                       Celery worker
        │                                                            │
        │                                          ┌─────────────────┘
        │                                          ▼
        │                          CrawlerDispatcher.get_crawler(url)
        │                          (regex on domain → concrete crawler,
        │                           fallback → generic article crawler)
        │                                          │
        │                                   extract text
        │                                          ▼
        │                           MongoDB  (raw document warehouse)
        │                                          │
        │                     clean → chunk → embed (feature pipeline)
        │                                          ▼
        │                             Qdrant  (vector store)
        │                                          │
        │  POST /query {question}                  ▼
        └────────────────────────  retrieve top-k → (rerank) →
                                    generate with Claude + citations
                                          → { answer, sources[] }
```

1. **Extract** — the client sends only a link. A `CrawlerDispatcher` matches the URL's domain to a concrete crawler, falling back to a generic article crawler for anything unregistered, so any site works out of the box.
2. **Transform** — raw HTML is normalized to clean text, split into overlapping chunks, and embedded into vectors.
3. **Load** — raw documents land in MongoDB (the data warehouse); embedded chunks land in Qdrant (the vector store).
4. **Ask** — `POST /query` embeds the question, retrieves the most relevant chunks, and asks Claude to answer using only that context, returning citations that resolve back to source URLs.

Crawling is long-running, so it never blocks an HTTP request: `POST /crawl` enqueues a job and returns immediately with a `job_id`; a background worker does the work; you poll `GET /jobs/{id}` for progress. Swagger UI at `/docs` is the operator interface — there is no bundled frontend (a dashboard lives in a separate service).

## Extraction strategy

Different sites need different tools. The dispatcher routes each URL to the right crawler:

| Crawler | Technique | Handles |
|---|---|---|
| **Generic article** (fallback) | LangChain `AsyncHtmlLoader` + `Html2TextTransformer` | Any static page / article — the default for a bare link |
| **Full-site crawler** | `sitemap.xml` discovery, else same-domain BFS link-following | Ingesting a whole site from one root link |
| **Selenium** | Headless Chrome + scroll + BeautifulSoup | JS-rendered / infinite-scroll pages |
| **GitHub** (optional) | `git clone` + tree walk | Repository docs/code as a text source |

New crawlers register a domain pattern and implement a single `extract()` method — the base abstraction and dispatcher handle the rest.

## Tech stack

| Concern | Choice |
|---|---|
| Language / packaging | Python, managed with **[uv](https://docs.astral.sh/uv/)** |
| API | **FastAPI** (Swagger UI at `/docs` is the operator surface) |
| Async jobs | **Celery + Redis** (crawls run off the request thread) |
| Raw document store | **MongoDB** (NoSQL warehouse of crawled text) |
| Vector store | **Qdrant** |
| Crawling | Selenium + headless Chrome, BeautifulSoup, LangChain loaders |
| Embeddings | **Voyage AI** (default) with a local `sentence-transformers` fallback |
| Generation | **Claude** via the official Anthropic SDK (`claude-opus-4-8`), with native citations |
| Orchestration | ZenML pipelines (ETL + feature engineering) |
| Runtime | Docker Compose (`api`, `worker`, `redis`, `mongo`, `qdrant`) |

> Anthropic has no first-party embedding model, so embeddings use Voyage AI (Anthropic's recommended provider) behind an interface — swap in the local model for offline or cost-sensitive runs without touching call sites.

## Getting started

> The steps below describe the intended developer setup. They'll work once the corresponding roadmap issues land.

```bash
# 1. Install dependencies into a virtualenv
uv sync

# 2. Configure secrets
cp .env.example .env      # set ANTHROPIC_API_KEY, VOYAGE_API_KEY, DB URLs, etc.

# 3. Bring up MongoDB, Redis, Qdrant, the API, and the worker
docker compose up -d

# 4. Ingest a site — from Swagger UI at http://localhost:8000/docs,
#    or via CLI:
uv run python -m rag_engine.run --url https://example.com --full-site

# 5. Ask a question over what you ingested (Swagger: POST /query)
```

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /crawl` `{url, full_site?}` | Enqueue a crawl; returns `202 {job_id}` immediately |
| `GET /jobs/{id}` | Job status + progress (`pages_done` / `pages_total`) |
| `GET /jobs` | List / filter crawl jobs |
| `DELETE /jobs/{id}` | Cancel a running crawl |
| `POST /query` `{question, site?, top_k?}` | Retrieve + generate a cited answer over the ingested corpus |
| `GET /health` · `GET /ready` | Liveness / dependency readiness |

Because the service fetches arbitrary URLs server-side, `POST /crawl` is guarded against SSRF (internal/loopback targets rejected) and rate-limited.

## Project layout

```
rag_engine/
├── domain/                  # Pydantic ODM documents + DataCategory types
│   └── base/                #   NoSQLBaseDocument (Mongo), VectorBaseDocument (Qdrant)
├── application/
│   ├── crawlers/            # BaseCrawler, CrawlerDispatcher, concrete crawlers
│   └── preprocessing/       # Cleaning / Chunking / Embedding dispatchers
├── infrastructure/db/       # Mongo + Qdrant + Redis connectors
├── api/                     # FastAPI app, routes, middleware
├── worker/                  # Celery app + crawl/feature tasks
steps/                       # ZenML steps (crawl_links, feature engineering)
pipelines/                   # ZenML pipelines (digital_data_etl, feature_engineering)
```

## Roadmap

Work is organized into phases on the [RAG Engine Roadmap](https://github.com/users/jyotsanh/projects/7) board (69 issues across 9 phases):

| Phase | Focus |
|---|---|
| 1 — Foundation | uv scaffold, settings/secrets, MongoDB via Docker Compose |
| 2 — Storage | NoSQL ODM, Mongo connector, document models |
| 3 — Crawler framework | `BaseCrawler`, Selenium base, `CrawlerDispatcher` |
| 4 — Concrete crawlers | generic-article fallback, full-site, Selenium, GitHub |
| 5 — Transform | text cleaning & normalization |
| 6 — Orchestration | ETL pipeline (link → crawler → Mongo), CLI |
| 7 — Quality | tests, logging, error handling, CI |
| 8 — API + async jobs | FastAPI, Celery/Redis, crawl endpoints, SSRF hardening |
| 9 — Retrieval & generation | Qdrant, chunking, embeddings, `POST /query` with Claude + citations |

## Acknowledgements

The extraction architecture (dispatcher + pluggable crawlers, ODM over MongoDB, ZenML ETL and feature-engineering pipelines) is modeled on the patterns in the [LLM Engineer's Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook), adapted here for a link-only, website-focused RAG service using `uv`.
