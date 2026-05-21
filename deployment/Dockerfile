# Agent Memory Toolkit API Server Dockerfile
# Multi-stage build for production-ready deployment

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim AS builder

# Set build arguments
ARG INSTALL_EMBEDDINGS=false
ARG INSTALL_COMPRESSION=true

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy project files
WORKDIR /app
COPY pyproject.toml README.md ./
COPY agent_memory ./agent_memory

# Install the package with API dependencies
RUN pip install --no-cache-dir ".[api]"

# Optionally install embeddings support (heavier)
RUN if [ "$INSTALL_EMBEDDINGS" = "true" ]; then \
        pip install --no-cache-dir ".[embeddings]"; \
    fi

# Optionally install compression support
RUN if [ "$INSTALL_COMPRESSION" = "true" ]; then \
        pip install --no-cache-dir ".[compression]"; \
    fi

# ============================================
# Stage 2: Production Runtime
# ============================================
FROM python:3.11-slim AS production

# Labels
LABEL org.opencontainers.image.title="Agent Memory Toolkit API"
LABEL org.opencontainers.image.description="Local-first memory layer for AI agents"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="Agent Memory Toolkit"
LABEL org.opencontainers.image.source="https://github.com/agent-memory-toolkit/agent-memory-toolkit"

# Create non-root user for security
RUN groupadd --gid 1000 amt \
    && useradd --uid 1000 --gid amt --shell /bin/bash --create-home amt

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --from=builder /app/agent_memory ./agent_memory
COPY --chown=amt:amt deployment/docker-entrypoint.sh /usr/local/bin/

# Create data directory for SQLite database
RUN mkdir -p /data && chown -R amt:amt /data

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Application settings
    AMT_HOST=0.0.0.0 \
    AMT_PORT=8000 \
    AMT_DB_PATH=/data/agent_memory.db \
    AMT_WORKERS=4 \
    AMT_DEBUG=false \
    # Health check settings
    AMT_HEALTH_CHECK_ENABLED=true

# Switch to non-root user
USER amt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command
CMD ["uvicorn", "agent_memory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
