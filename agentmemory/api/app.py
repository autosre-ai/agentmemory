"""
Agent Memory Toolkit REST API Server.

A FastAPI-based REST API for the Agent Memory Toolkit, providing
HTTP endpoints for all memory operations with JWT authentication,
rate limiting, and OpenAPI documentation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .config import get_config, APIConfig
from .middleware import RateLimitMiddleware
from .dependencies import close_memory_store
from .models import HealthResponse, InfoResponse, ErrorResponse, ValidationErrorResponse
from .routes import (
    auth_router,
    memories_router,
    branches_router,
    extraction_router,
    security_router,
    compression_router,
    health_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    yield
    # Shutdown
    close_memory_store()


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Optional API configuration. Uses global config if not provided.
        
    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = get_config()
    
    app = FastAPI(
        title=config.api_title,
        description=config.api_description,
        version=config.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=config.cors_allow_methods,
        allow_headers=config.cors_allow_headers,
    )
    
    # Add rate limiting middleware
    if config.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware)
    
    # Add exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "detail": exc.errors(),
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(exc),
            },
        )
    
    # Health check endpoint (no auth required)
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check",
        description="Check if the API server is running.",
    )
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version=config.api_version,
            timestamp=datetime.utcnow(),
        )
    
    # Info endpoint (no auth required)
    @app.get(
        "/info",
        response_model=InfoResponse,
        tags=["Info"],
        summary="API information",
        description="Get information about the API and available modules.",
    )
    async def api_info() -> InfoResponse:
        return InfoResponse(
            name=config.api_title,
            version=config.api_version,
            description="Local-first memory layer for AI agents",
            modules=["store", "extraction", "team", "security", "compression"],
        )
    
    # Include routers
    app.include_router(auth_router, prefix=config.api_prefix)
    app.include_router(memories_router, prefix=config.api_prefix)
    app.include_router(branches_router, prefix=config.api_prefix)
    app.include_router(extraction_router, prefix=config.api_prefix)
    app.include_router(security_router, prefix=config.api_prefix)
    app.include_router(compression_router, prefix=config.api_prefix)
    
    # Health routes at root level (no prefix, no auth)
    app.include_router(health_router)
    
    return app


# Default application instance
app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info",
):
    """
    Run the API server using uvicorn.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        workers: Number of worker processes
        log_level: Logging level
    """
    import uvicorn
    
    uvicorn.run(
        "agent_memory.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_server()
