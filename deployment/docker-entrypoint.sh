#!/bin/bash
set -e

# Agent Memory Toolkit Docker Entrypoint Script
# Handles initialization, environment setup, and process management

echo "=========================================="
echo "Agent Memory Toolkit API Server"
echo "Version: ${AMT_VERSION:-1.0.0}"
echo "=========================================="

# Create data directory if it doesn't exist
if [ -n "$AMT_DB_PATH" ]; then
    DB_DIR=$(dirname "$AMT_DB_PATH")
    if [ ! -d "$DB_DIR" ]; then
        echo "Creating database directory: $DB_DIR"
        mkdir -p "$DB_DIR"
    fi
fi

# Generate JWT secret if not provided
if [ -z "$AMT_JWT_SECRET" ]; then
    echo "Warning: AMT_JWT_SECRET not set, generating random secret"
    echo "Note: This will invalidate tokens on restart"
    export AMT_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
fi

# Wait for Redis if configured
if [ -n "$AMT_REDIS_URL" ]; then
    echo "Waiting for Redis..."
    REDIS_HOST=$(echo "$AMT_REDIS_URL" | sed -n 's/.*:\/\/\([^:]*\).*/\1/p')
    REDIS_PORT=$(echo "$AMT_REDIS_URL" | sed -n 's/.*:\([0-9]*\)$/\1/p')
    REDIS_PORT=${REDIS_PORT:-6379}
    
    MAX_RETRIES=${AMT_REDIS_WAIT_RETRIES:-30}
    RETRY_INTERVAL=${AMT_REDIS_WAIT_INTERVAL:-1}
    
    for i in $(seq 1 $MAX_RETRIES); do
        if nc -z "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; then
            echo "Redis is available"
            break
        fi
        if [ $i -eq $MAX_RETRIES ]; then
            echo "Warning: Redis not available after $MAX_RETRIES attempts"
        fi
        echo "Waiting for Redis... ($i/$MAX_RETRIES)"
        sleep $RETRY_INTERVAL
    done
fi

# Print configuration (non-sensitive)
echo ""
echo "Configuration:"
echo "  Host: ${AMT_HOST:-0.0.0.0}"
echo "  Port: ${AMT_PORT:-8000}"
echo "  Workers: ${AMT_WORKERS:-1}"
echo "  Debug: ${AMT_DEBUG:-false}"
echo "  Database: ${AMT_DB_PATH:-agent_memory.db}"
echo "  Rate Limit: ${AMT_RATE_LIMIT:-true}"
echo "  Redis: ${AMT_REDIS_URL:-not configured}"
echo ""

# Set uvicorn worker count
WORKERS=${AMT_WORKERS:-1}

# Run database migrations or initialization if needed
if [ "$AMT_RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    # Add migration commands here if needed
    echo "Migrations complete"
fi

# Execute the command
echo "Starting API server..."
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    exec uvicorn agent_memory.api.app:app \
        --host "${AMT_HOST:-0.0.0.0}" \
        --port "${AMT_PORT:-8000}" \
        --workers "$WORKERS" \
        --log-level "${AMT_LOG_LEVEL:-info}" \
        --access-log \
        --proxy-headers \
        --forwarded-allow-ips='*'
fi
