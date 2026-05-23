"""
Memory Security Layer

Production-ready validation and security module for agent memory systems.
Provides poison detection, confidence scoring, uncertainty detection,
source validation, and audit trail logging.

Usage:
    from agent_memory_toolkit.security import MemoryGuard, SecurityLevel
    
    guard = MemoryGuard(level=SecurityLevel.MEDIUM)
    result = guard.validate(memory)
    
    if result.is_safe:
        # Memory can be trusted
        store.save(memory)
    else:
        logger.warning(f"Rejected: {result.rejection_reason}")
"""

from .guard import (
    MemoryGuard,
    ValidationResult,
    SecurityLevel,
    SecurityConfig,
)
from .detectors import (
    PoisonDetector,
    InjectionPattern,
    DetectionResult,
)
from .confidence import (
    ConfidenceScorer,
    UncertaintyDetector,
    ConfidenceResult,
)
from .sources import (
    SourceValidator,
    SourceTrust,
    SourceValidationResult,
)
from .audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
)

__all__ = [
    # Main guard
    "MemoryGuard",
    "ValidationResult",
    "SecurityLevel",
    "SecurityConfig",
    # Detection
    "PoisonDetector",
    "InjectionPattern",
    "DetectionResult",
    # Confidence
    "ConfidenceScorer",
    "UncertaintyDetector",
    "ConfidenceResult",
    # Sources
    "SourceValidator",
    "SourceTrust",
    "SourceValidationResult",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
]
