"""Security validation routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..models import (
    SecurityCheckRequest,
    SecurityCheckResponse,
)
from ..auth import get_current_user

router = APIRouter(prefix="/security", tags=["Security"])


def get_memory_guard():
    """Lazy import MemoryGuard."""
    from agent_memory_toolkit.security import MemoryGuard, SecurityLevel
    return MemoryGuard, SecurityLevel


@router.post(
    "/check",
    response_model=SecurityCheckResponse,
    summary="Check content for security issues",
    description="Validate content for potential security issues like prompt injection.",
)
async def check_content(
    request: SecurityCheckRequest,
    current_user: str = Depends(get_current_user),
) -> SecurityCheckResponse:
    """Check content for security issues."""
    MemoryGuard, SecurityLevel = get_memory_guard()
    
    level_map = {
        "minimal": SecurityLevel.MINIMAL,
        "low": SecurityLevel.LOW,
        "medium": SecurityLevel.MEDIUM,
        "high": SecurityLevel.HIGH,
        "paranoid": SecurityLevel.PARANOID,
    }
    
    security_level = level_map.get(request.level, SecurityLevel.MEDIUM)
    guard = MemoryGuard(level=security_level)
    result = guard.validate_content(request.content)
    
    detected_patterns = []
    if result.poison_result and result.poison_result.detected_patterns:
        detected_patterns = [p.value for p in result.poison_result.detected_patterns]
    
    return SecurityCheckResponse(
        is_safe=result.is_safe,
        rejection_reason=result.rejection_reason,
        adjusted_confidence=result.adjusted_confidence,
        validation_time_ms=result.validation_time_ms,
        detected_patterns=detected_patterns,
    )


@router.get(
    "/levels",
    summary="List security levels",
    description="Get available security levels and their descriptions.",
)
async def list_security_levels(
    current_user: str = Depends(get_current_user),
):
    """List available security levels."""
    return {
        "levels": [
            {
                "name": "minimal",
                "description": "Minimal validation, basic checks only",
            },
            {
                "name": "low",
                "description": "Low security, catches obvious issues",
            },
            {
                "name": "medium",
                "description": "Balanced security (recommended)",
            },
            {
                "name": "high",
                "description": "High security, stricter validation",
            },
            {
                "name": "paranoid",
                "description": "Maximum security, may have false positives",
            },
        ],
        "default": "medium",
    }
