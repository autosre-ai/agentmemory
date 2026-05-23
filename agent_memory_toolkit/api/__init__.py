"""
Agent Memory Toolkit REST API Module.

Provides a FastAPI-based REST API for the Agent Memory Toolkit,
with JWT authentication, rate limiting, and comprehensive endpoints
for all memory operations.

Usage:
    # Start the server
    amt api serve
    
    # Or programmatically
    from agent_memory_toolkit.api import create_app, run_server
    app = create_app()
    run_server(port=8000)
"""

from .app import app, create_app, run_server
from .config import APIConfig, get_config, set_config
from .auth import (
    create_access_token,
    verify_token,
    authenticate_user,
    get_current_user,
    AuthenticationError,
)
from .middleware import RateLimiter, RateLimitMiddleware
from .dependencies import get_memory_store, close_memory_store, MemoryStoreManager

__all__ = [
    # Application
    "app",
    "create_app",
    "run_server",
    # Configuration
    "APIConfig",
    "get_config",
    "set_config",
    # Authentication
    "create_access_token",
    "verify_token",
    "authenticate_user",
    "get_current_user",
    "AuthenticationError",
    # Middleware
    "RateLimiter",
    "RateLimitMiddleware",
    # Dependencies
    "get_memory_store",
    "close_memory_store",
    "MemoryStoreManager",
]
