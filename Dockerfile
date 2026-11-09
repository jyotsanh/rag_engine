# syntax=docker/dockerfile:1.7
#
# Images for rag_engine. Step 37 adds the crawl `worker` stage (headless Chromium);
# Step 95 adds the `api` stage and production hardening on top of the same `deps` stage.
#
#   docker build --target worker -t rag-engine-worker .
#   docker run --rm --network host --env-file .env rag-engine-worker \
#       python -m rag_engine.crawling.fetchers.smoke http://127.0.0.1:8000/
#
# Pinned so builds are reproducible; bump together with uv.lock / playwright.
ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.11.14

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---- base: Python + uv, no project code ---- #
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app

# ---- deps: third-party packages only (cached until uv.lock changes) ---- #
FROM base AS deps
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# ---- worker: crawl worker with headless Chromium ---- #
FROM base AS worker
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
COPY --from=deps /app/.venv /app/.venv
# Chromium + its system libraries and fonts, in a layer that only changes with playwright.
# --only-shell: the headless shell is all `headless=True` needs (≈ half the size of full
# Chromium); a headed browser (CRAWL_BROWSER_HEADLESS=false) is not supported in the image.
RUN playwright install --with-deps --only-shell chromium \
    && rm -rf /var/lib/apt/lists/* /tmp/*
# the project itself, installed non-editable into the venv
COPY pyproject.toml uv.lock README.md ./
COPY rag_engine ./rag_engine
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
USER app
# the Celery worker (Step 75): prefork children, one async runtime each; split the queues
# across deployments with `--queues` (e.g. crawl workers vs maintenance/webhooks workers)
CMD ["celery", "--app", "rag_engine.worker.celery_app", "worker", "--loglevel", "INFO", \
     "--pool", "prefork", "--queues", "crawl,maintenance,webhooks"]
