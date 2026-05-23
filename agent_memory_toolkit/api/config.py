"""API configuration and settings."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIConfig:
    """Configuration for the REST API server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    
    # Database settings
    db_path: str = "agent_memory.db"
    
    # Authentication settings
    jwt_secret_key: str = field(default_factory=lambda: os.environ.get(
        "AMT_JWT_SECRET", secrets.token_urlsafe(32)
    ))
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window_seconds: int = 60  # window size
    
    # CORS settings
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: list[str] = field(default_factory=lambda: ["*"])
    
    # API settings
    api_prefix: str = "/api/v1"
    api_title: str = "Agent Memory Toolkit API"
    api_description: str = """
## Agent Memory Toolkit REST API

A comprehensive REST API for managing AI agent memory with SQLite + FTS5,
structured extraction, team collaboration, security validation, and
intelligent context compression.

### Features

- **Memory Operations**: Add, query, update, delete memories
- **Full-Text Search**: FTS5-powered semantic search
- **Branching**: Git-like version control for memories
- **Extraction**: Extract structured data from text
- **Security**: Content validation and audit logging
- **Compression**: Context compression for token budgets

### Authentication

All endpoints (except `/health` and `/auth/token`) require JWT authentication.
Include the token in the `Authorization` header:

```
Authorization: Bearer <your_token>
```
"""
    api_version: str = "1.0.0"
    
    # User settings (simple in-memory users for demo)
    # In production, use a proper user database
    api_users: dict[str, str] = field(default_factory=lambda: {
        "admin": os.environ.get("AMT_ADMIN_PASSWORD", "admin"),
        "agent": os.environ.get("AMT_AGENT_PASSWORD", "agent"),
    })
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.environ.get("AMT_HOST", "0.0.0.0"),
            port=int(os.environ.get("AMT_PORT", "8000")),
            debug=os.environ.get("AMT_DEBUG", "false").lower() == "true",
            workers=int(os.environ.get("AMT_WORKERS", "1")),
            db_path=os.environ.get("AMT_DB_PATH", "agent_memory.db"),
            jwt_secret_key=os.environ.get("AMT_JWT_SECRET", secrets.token_urlsafe(32)),
            jwt_expiration_minutes=int(os.environ.get("AMT_JWT_EXPIRATION", "60")),
            rate_limit_enabled=os.environ.get("AMT_RATE_LIMIT", "true").lower() == "true",
            rate_limit_requests=int(os.environ.get("AMT_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window_seconds=int(os.environ.get("AMT_RATE_LIMIT_WINDOW", "60")),
        )


# Global config instance
_config: Optional[APIConfig] = None


def get_config() -> APIConfig:
    """Get the global API configuration."""
    global _config
    if _config is None:
        _config = APIConfig.from_env()
    return _config


def set_config(config: APIConfig) -> None:
    """Set the global API configuration."""
    global _config
    _config = config
