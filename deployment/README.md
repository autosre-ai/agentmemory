# Agent Memory Toolkit - Deployment Guide

This directory contains all deployment configurations for the Agent Memory Toolkit API server, including Docker, Docker Compose, and Kubernetes manifests.

## Quick Start

### Docker Compose (Recommended for Development)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env and set a secure JWT secret
# Generate a secret: openssl rand -base64 32

# 3. Start services
docker-compose up -d

# 4. Check status
docker-compose ps
docker-compose logs -f api

# 5. Test the API
curl http://localhost:8000/health
```

### Docker Build Only

```bash
# Build the image
docker build -t agent-memory-toolkit:latest -f deployment/Dockerfile .

# Run standalone
docker run -d \
  -p 8000:8000 \
  -e AMT_JWT_SECRET="your-secret-key" \
  -v amt-data:/data \
  agent-memory-toolkit:latest
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
│                   (Ingress/LB)                          │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  API Server                             │
│              (FastAPI + Uvicorn)                        │
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ Memory  │  │ Branch  │  │Extract  │  │Security │   │
│  │ Routes  │  │ Routes  │  │ Routes  │  │ Routes  │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
│                                                         │
└─────────┬────────────────────────────┬──────────────────┘
          │                            │
┌─────────▼─────────┐      ┌───────────▼────────────────┐
│    SQLite DB      │      │        Redis Cache         │
│  (Persistent Vol) │      │    (Session/Rate Limit)    │
└───────────────────┘      └────────────────────────────┘
```

## Directory Structure

```
deployment/
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Full stack with Redis
├── docker-entrypoint.sh    # Container entrypoint script
├── .env.example            # Environment variable template
├── prometheus.yml          # Prometheus scrape config
├── README.md               # This file
└── kubernetes/
    ├── configmap.yaml      # Non-sensitive configuration
    ├── secret.yaml         # Sensitive configuration
    ├── deployment.yaml     # API and Redis deployments
    ├── service.yaml        # Kubernetes services
    ├── ingress.yaml        # Ingress configuration
    ├── storage.yaml        # Persistent volume claims
    └── hpa.yaml            # Autoscaling & policies
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AMT_HOST` | API server bind address | `0.0.0.0` |
| `AMT_PORT` | API server port | `8000` |
| `AMT_WORKERS` | Uvicorn worker count | `4` |
| `AMT_DEBUG` | Enable debug mode | `false` |
| `AMT_LOG_LEVEL` | Logging level | `info` |
| `AMT_DB_PATH` | SQLite database path | `/data/agent_memory.db` |
| `AMT_JWT_SECRET` | **REQUIRED** JWT signing key | - |
| `AMT_JWT_EXPIRATION` | Token expiration (minutes) | `60` |
| `AMT_ADMIN_PASSWORD` | Admin user password | `admin` |
| `AMT_AGENT_PASSWORD` | Agent user password | `agent` |
| `AMT_RATE_LIMIT` | Enable rate limiting | `true` |
| `AMT_RATE_LIMIT_REQUESTS` | Requests per window | `100` |
| `AMT_RATE_LIMIT_WINDOW` | Window size (seconds) | `60` |
| `AMT_REDIS_URL` | Redis connection URL | - |
| `AMT_CACHE_TTL` | Cache TTL (seconds) | `3600` |

## Docker Compose Profiles

```bash
# Default (API + Redis)
docker-compose up -d

# With Redis Commander admin UI
docker-compose --profile admin up -d

# With Prometheus + Grafana monitoring
docker-compose --profile monitoring up -d

# Full stack with all services
docker-compose --profile admin --profile monitoring up -d
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- NGINX Ingress Controller (optional)
- cert-manager (for TLS, optional)

### Deploy

```bash
# 1. Create namespace and base resources
kubectl apply -f kubernetes/storage.yaml

# 2. Create secrets (edit first!)
kubectl apply -f kubernetes/secret.yaml

# 3. Create config
kubectl apply -f kubernetes/configmap.yaml

# 4. Deploy services
kubectl apply -f kubernetes/service.yaml

# 5. Deploy applications
kubectl apply -f kubernetes/deployment.yaml

# 6. Configure ingress (optional)
kubectl apply -f kubernetes/ingress.yaml

# 7. Enable autoscaling (optional)
kubectl apply -f kubernetes/hpa.yaml
```

### Verify Deployment

```bash
# Check pods
kubectl get pods -n agent-memory

# Check services
kubectl get svc -n agent-memory

# View logs
kubectl logs -f deployment/amt-api -n agent-memory

# Test health endpoint
kubectl port-forward svc/amt-api 8000:80 -n agent-memory
curl http://localhost:8000/health
```

### Production Checklist

- [ ] Set strong `AMT_JWT_SECRET` in secret.yaml
- [ ] Change default passwords
- [ ] Configure proper storage class
- [ ] Set up TLS certificates
- [ ] Configure resource limits
- [ ] Enable network policies
- [ ] Set up monitoring and alerting
- [ ] Configure backup for SQLite database

## Health Check Endpoints

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `/health` | Basic health check | No |
| `/health/live` | Kubernetes liveness probe | No |
| `/health/ready` | Kubernetes readiness probe | No |
| `/health/detailed` | Detailed component status | No |
| `/health/startup` | Kubernetes startup probe | No |

### Health Check Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600.5,
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 1.23,
      "message": "Connected"
    },
    {
      "name": "redis",
      "status": "healthy",
      "latency_ms": 0.45,
      "message": "Connected"
    }
  ]
}
```

## API Authentication

```bash
# Get JWT token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" | jq -r '.access_token')

# Use token
curl http://localhost:8000/api/v1/memories \
  -H "Authorization: Bearer $TOKEN"
```

## Scaling

### Docker Compose

```bash
# Scale API containers
docker-compose up -d --scale api=3
```

### Kubernetes

```bash
# Manual scaling
kubectl scale deployment amt-api --replicas=5 -n agent-memory

# Autoscaling is configured in hpa.yaml
# Scales based on CPU (70%) and memory (80%) utilization
```

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics` (when enabled):
- Request latency
- Request count by endpoint
- Error rates
- Active connections

### Grafana Dashboards

Import the following dashboard IDs:
- FastAPI: 11074
- Redis: 11835

## Backup and Recovery

### SQLite Database

```bash
# Backup
docker cp amt-api:/data/agent_memory.db ./backup.db

# Restore
docker cp ./backup.db amt-api:/data/agent_memory.db
docker restart amt-api
```

### Kubernetes

```bash
# Backup PVC
kubectl exec -n agent-memory deployment/amt-api -- \
  sqlite3 /data/agent_memory.db ".backup /tmp/backup.db"
kubectl cp agent-memory/amt-api-xxx:/tmp/backup.db ./backup.db
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs api

# Check health
docker inspect amt-api | jq '.[0].State.Health'
```

### Database locked errors

SQLite doesn't support concurrent writes from multiple processes.
Solutions:
- Use single writer with multiple readers
- Use connection pooling
- Consider PostgreSQL for high-write scenarios

### Redis connection refused

```bash
# Check Redis is running
docker-compose ps redis
docker-compose logs redis

# Test connection
docker exec amt-redis redis-cli ping
```

## Security Recommendations

1. **JWT Secret**: Always use a strong, random JWT secret in production
2. **TLS**: Enable HTTPS in production (use cert-manager with Let's Encrypt)
3. **Network Policies**: Restrict pod-to-pod communication
4. **Secrets Management**: Use a secrets manager (Vault, AWS Secrets Manager)
5. **Image Scanning**: Scan Docker images for vulnerabilities
6. **RBAC**: Configure proper Kubernetes RBAC
7. **Audit Logging**: Enable API audit logging

## License

MIT License - see LICENSE file for details.
