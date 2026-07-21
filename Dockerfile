# ============================================
# Stage 1: Builder — install dependencies
# ============================================
FROM python:3.12-slim-bookworm AS builder

# Install system deps needed for the uv installer
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /code

# Copy only dependency files first (layer caching)
COPY pyproject.toml uv.lock* ./

# Install production dependencies only (no dev group)
# --frozen: don't update lockfile; --no-dev: skip dev dependencies
RUN uv sync --frozen --no-install-project --no-dev || uv sync --no-install-project --no-dev

# Copy source code and install the project itself
COPY . .
RUN uv sync --frozen --no-dev || uv sync --no-dev


# ============================================
# Stage 2: Runtime — slim final image
# ============================================
FROM python:3.12-slim-bookworm AS runtime

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /code

# Copy the full project (with venv) from builder — no uv needed at runtime
COPY --from=builder /code /code

# Put the venv on PATH so the installed `server` script is found directly
ENV VIRTUAL_ENV=/code/.venv
ENV PATH="/code/.venv/bin:$PATH"

# Transfer ownership to the non-root user
RUN chown -R appuser:appgroup /code
USER appuser

EXPOSE 8000

# Calls the `server` console script installed by pyproject.toml → rag_engine:server
CMD ["server"]
