"""Rate limiting middleware for the REST API."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional, Callable
from dataclasses import dataclass, field

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .config import get_config


@dataclass
class RateLimitState:
    """State for a single client's rate limit window."""
    requests: int = 0
    window_start: float = field(default_factory=time.time)


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window.
    
    For production use, consider using Redis-based rate limiting.
    """
    
    def __init__(
        self,
        requests_per_window: int = 100,
        window_seconds: int = 60,
    ):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._state: dict[str, RateLimitState] = defaultdict(RateLimitState)
    
    def _get_client_id(self, request: Request) -> str:
        """Get a unique identifier for the client."""
        # Use X-Forwarded-For if behind a proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def is_allowed(self, request: Request) -> tuple[bool, dict]:
        """
        Check if a request is allowed under rate limiting.
        
        Args:
            request: The incoming request
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        state = self._state[client_id]
        
        # Check if window has expired
        if current_time - state.window_start >= self.window_seconds:
            # Start new window
            state.requests = 0
            state.window_start = current_time
        
        # Calculate remaining
        remaining = max(0, self.requests_per_window - state.requests)
        reset_time = int(state.window_start + self.window_seconds)
        
        info = {
            "limit": self.requests_per_window,
            "remaining": remaining,
            "reset": reset_time,
        }
        
        if state.requests >= self.requests_per_window:
            return False, info
        
        # Increment request count
        state.requests += 1
        info["remaining"] = max(0, self.requests_per_window - state.requests)
        
        return True, info
    
    def reset(self, request: Request) -> None:
        """Reset rate limit for a client."""
        client_id = self._get_client_id(request)
        if client_id in self._state:
            del self._state[client_id]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Adds rate limit headers to responses and returns 429 when limit exceeded.
    """
    
    def __init__(
        self,
        app,
        limiter: Optional[RateLimiter] = None,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        config = get_config()
        
        if limiter is None:
            limiter = RateLimiter(
                requests_per_window=config.rate_limit_requests,
                window_seconds=config.rate_limit_window_seconds,
            )
        
        self.limiter = limiter
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/redoc"]
        self.enabled = config.rate_limit_enabled
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Check rate limit
        allowed, info = self.limiter.is_allowed(request)
        
        if not allowed:
            # Return 429 Too Many Requests
            return Response(
                content='{"error": "rate_limit_exceeded", "message": "Too many requests. Please retry later."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(info["reset"] - int(time.time())),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])
        
        return response


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        config = get_config()
        _rate_limiter = RateLimiter(
            requests_per_window=config.rate_limit_requests,
            window_seconds=config.rate_limit_window_seconds,
        )
    return _rate_limiter
