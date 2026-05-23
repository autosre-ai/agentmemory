"""Routes package for the REST API."""

from .memories import router as memories_router
from .branches import router as branches_router
from .auth import router as auth_router
from .extraction import router as extraction_router
from .security import router as security_router
from .compression import router as compression_router
from .health import router as health_router

__all__ = [
    "memories_router",
    "branches_router",
    "auth_router",
    "extraction_router",
    "security_router",
    "compression_router",
    "health_router",
]
