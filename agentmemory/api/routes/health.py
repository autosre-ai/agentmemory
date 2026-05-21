"""
Enhanced health check endpoints for production deployment.

Provides comprehensive health checks for:
- API server status
- Database connectivity
- Redis cache connectivity
- System resource monitoring
"""

from __future__ import annotations

import os
import time
import sqlite3
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(BaseModel):
    """Basic health check response."""
    status: str
    version: str
    timestamp: datetime


class ComponentHealth(BaseModel):
    """Health status for a single component."""
    name: str
    status: str  # healthy, unhealthy, degraded
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class DetailedHealthStatus(BaseModel):
    """Detailed health check response with component status."""
    status: str  # healthy, unhealthy, degraded
    version: str
    timestamp: datetime
    uptime_seconds: float
    components: list[ComponentHealth]


class ReadinessStatus(BaseModel):
    """Readiness probe response."""
    ready: bool
    checks: dict[str, bool]


class LivenessStatus(BaseModel):
    """Liveness probe response."""
    alive: bool


# Track server start time
_start_time = time.time()


def get_version() -> str:
    """Get API version from environment or default."""
    return os.environ.get("AMT_VERSION", "1.0.0")


def check_database() -> ComponentHealth:
    """Check database connectivity."""
    db_path = os.environ.get("AMT_DB_PATH", "agent_memory.db")
    start = time.time()
    
    try:
        if db_path == ":memory:":
            return ComponentHealth(
                name="database",
                status="healthy",
                latency_ms=0,
                message="Using in-memory database"
            )
        
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("SELECT 1")
        conn.close()
        
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="database",
            status="healthy",
            latency_ms=round(latency, 2),
            message=f"Connected to {db_path}"
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status="unhealthy",
            message=str(e)
        )


def check_redis() -> ComponentHealth:
    """Check Redis connectivity if configured."""
    redis_url = os.environ.get("AMT_REDIS_URL")
    
    if not redis_url:
        return ComponentHealth(
            name="redis",
            status="healthy",
            message="Redis not configured (optional)"
        )
    
    start = time.time()
    try:
        # Try to import and connect to Redis
        import redis
        r = redis.from_url(redis_url, socket_timeout=5)
        r.ping()
        
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="redis",
            status="healthy",
            latency_ms=round(latency, 2),
            message="Connected"
        )
    except ImportError:
        return ComponentHealth(
            name="redis",
            status="degraded",
            message="Redis client not installed"
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status="unhealthy",
            message=str(e)
        )


def check_disk_space() -> ComponentHealth:
    """Check available disk space."""
    try:
        import shutil
        data_path = os.environ.get("AMT_DB_PATH", ".")
        data_dir = os.path.dirname(data_path) or "."
        
        total, used, free = shutil.disk_usage(data_dir)
        free_percent = (free / total) * 100
        
        status = "healthy"
        if free_percent < 10:
            status = "unhealthy"
        elif free_percent < 20:
            status = "degraded"
        
        return ComponentHealth(
            name="disk",
            status=status,
            message=f"{free_percent:.1f}% free ({free // (1024**3)}GB)"
        )
    except Exception as e:
        return ComponentHealth(
            name="disk",
            status="degraded",
            message=str(e)
        )


def check_memory() -> ComponentHealth:
    """Check memory usage."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        status = "healthy"
        if memory.percent > 95:
            status = "unhealthy"
        elif memory.percent > 85:
            status = "degraded"
        
        return ComponentHealth(
            name="memory",
            status=status,
            message=f"{memory.percent:.1f}% used ({memory.available // (1024**2)}MB available)"
        )
    except ImportError:
        return ComponentHealth(
            name="memory",
            status="healthy",
            message="psutil not installed (skipping check)"
        )
    except Exception as e:
        return ComponentHealth(
            name="memory",
            status="degraded",
            message=str(e)
        )


@router.get("", response_model=HealthStatus)
@router.get("/", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Basic health check endpoint.
    
    Returns 200 if the API is running.
    Used by load balancers and basic monitoring.
    """
    return HealthStatus(
        status="healthy",
        version=get_version(),
        timestamp=datetime.utcnow()
    )


@router.get("/live", response_model=LivenessStatus)
async def liveness_probe() -> LivenessStatus:
    """
    Kubernetes liveness probe.
    
    Returns 200 if the process is alive and should continue running.
    If this fails, Kubernetes will restart the pod.
    """
    return LivenessStatus(alive=True)


@router.get("/ready", response_model=ReadinessStatus)
async def readiness_probe() -> ReadinessStatus:
    """
    Kubernetes readiness probe.
    
    Returns 200 if the service is ready to receive traffic.
    Checks database connectivity and other dependencies.
    """
    db_health = check_database()
    redis_health = check_redis()
    
    checks = {
        "database": db_health.status == "healthy",
        "redis": redis_health.status in ("healthy", "degraded"),
    }
    
    ready = all(checks.values())
    
    if not ready:
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "checks": checks
            }
        )
    
    return ReadinessStatus(ready=ready, checks=checks)


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check() -> DetailedHealthStatus:
    """
    Detailed health check with component status.
    
    Checks all components and returns comprehensive status.
    Useful for debugging and monitoring dashboards.
    """
    components = [
        check_database(),
        check_redis(),
        check_disk_space(),
        check_memory(),
    ]
    
    # Determine overall status
    statuses = [c.status for c in components]
    if "unhealthy" in statuses:
        overall = "unhealthy"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"
    
    return DetailedHealthStatus(
        status=overall,
        version=get_version(),
        timestamp=datetime.utcnow(),
        uptime_seconds=round(time.time() - _start_time, 2),
        components=components
    )


@router.get("/startup")
async def startup_probe() -> dict:
    """
    Kubernetes startup probe.
    
    Returns 200 when the application has fully started.
    Used to delay liveness/readiness probes during slow startup.
    """
    # Check if critical components are initialized
    db_health = check_database()
    
    if db_health.status == "unhealthy":
        raise HTTPException(
            status_code=503,
            detail="Database not ready"
        )
    
    return {
        "started": True,
        "uptime_seconds": round(time.time() - _start_time, 2)
    }
