FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

# Install build/runtime requirements for documentation conversion
RUN apt-get update && apt-get install -y --no-install-recommends pandoc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifests and README first for better caching
COPY pyproject.toml uv.lock README.md ./

# Create project virtual environment
ENV UV_PROJECT_ENVIRONMENT=.venv
RUN uv sync --frozen --no-dev

# Ensure the environment is on PATH for runtime commands
ENV PATH="/app/.venv/bin:${PATH}"

# Copy the remainder of the application source
COPY . .

# Pre-populate documentation cache during build for faster cold starts
ENV DOCS_SEED_DIR=/app/docs_seed
RUN mkdir -p "${DOCS_SEED_DIR}" && \
    DOCS_DIR="${DOCS_SEED_DIR}" python scripts/sync_docs.py && \
    rm -f sync.log

# Set defaults for runtime configuration
ENV DOCS_DIR=/data/docs
ENV MEDIAWIKI_API_URL=https://docs.alliancecan.ca/mediawiki/api.php
ENV USER_AGENT=AllianceDocsMCP/1.0
ENV PYTHONUNBUFFERED=1

# Create volume mount point for documentation cache
VOLUME ["/data"]

# Make the entrypoint executable (just in case git permissions are lost)
RUN chmod +x docker-entrypoint.sh

EXPOSE 8080

CMD ["/app/docker-entrypoint.sh"]
