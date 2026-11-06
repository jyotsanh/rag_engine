# rag_engine

Multi-tenant service that takes a website URL, crawls the whole site (HTML, PDF, DOCX, PPTX, XLSX), converts content into structure-aware chunks, embeds them with OpenAI, and stores them in a vector store (Qdrant, behind a `BaseVectorStore` factory). Exposes dense / hybrid / reranked search over each knowledge base.

**Stack:** Python 3.12 · uv · FastAPI · Celery + Redis · MongoDB · Qdrant · Playwright · OpenAI embeddings · fastembed (BM25, cross-encoder rerank)

> 🚧 Under construction. The build follows a 100-step plan — see [`plan.md`](plan.md) for decisions, the step list, and the Progress Log.
