"""
Memory Guard - Main Security Validation Module

Provides comprehensive security validation for memory content before
storage or retrieval. Combines poison detection, confidence scoring,
uncertainty detection, and source validation.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, TYPE_CHECKING

from .detectors import PoisonDetector, DetectionResult, InjectionPattern
from .confidence import ConfidenceScorer, UncertaintyDetector, ConfidenceResult
from .sources import SourceValidator, SourceTrust, SourceValidationResult
from .audit import AuditLogger, AuditEventType, MemoryAuditSink, FileAuditSink

# Import Memory type if available
if TYPE_CHECKING:
    from ..extraction.domains import Memory


class SecurityLevel(Enum):
    """
    Security levels for memory validation.
    
    Each level applies different thresholds and checks.
    """
    
    MINIMAL = "minimal"  # Basic checks only
    LOW = "low"  # Light validation
    MEDIUM = "medium"  # Standard validation (default)
    HIGH = "high"  # Strict validation
    PARANOID = "paranoid"  # Maximum security
    
    @property
    def config(self) -> "SecurityConfig":
        """Get configuration for this security level."""
        configs = {
            SecurityLevel.MINIMAL: SecurityConfig(
                poison_check=True,
                confidence_check=False,
                source_check=False,
                min_confidence=0.0,
                poison_threshold=0.9,
                block_unknown_sources=False,
            ),
            SecurityLevel.LOW: SecurityConfig(
                poison_check=True,
                confidence_check=True,
                source_check=False,
                min_confidence=0.2,
                poison_threshold=0.8,
                block_unknown_sources=False,
            ),
            SecurityLevel.MEDIUM: SecurityConfig(
                poison_check=True,
                confidence_check=True,
                source_check=True,
                min_confidence=0.4,
                poison_threshold=0.5,
                block_unknown_sources=False,
            ),
            SecurityLevel.HIGH: SecurityConfig(
                poison_check=True,
                confidence_check=True,
                source_check=True,
                min_confidence=0.6,
                poison_threshold=0.3,
                block_unknown_sources=False,
                require_source=True,
            ),
            SecurityLevel.PARANOID: SecurityConfig(
                poison_check=True,
                confidence_check=True,
                source_check=True,
                min_confidence=0.8,
                poison_threshold=0.2,
                block_unknown_sources=True,
                require_source=True,
                strict_uncertainty=True,
            ),
        }
        return configs[self]


@dataclass
class SecurityConfig:
    """
    Configuration for memory security validation.
    
    Attributes:
        poison_check: Enable poison/injection detection
        confidence_check: Enable confidence scoring
        source_check: Enable source validation
        min_confidence: Minimum confidence to accept
        poison_threshold: Risk score threshold for rejection
        block_unknown_sources: Block memories from unknown sources
        require_source: Require source to be provided
        strict_uncertainty: Use strict uncertainty detection
        audit_all: Log all validations, not just rejections
        quarantine_suspicious: Quarantine instead of reject
    """
    
    poison_check: bool = True
    confidence_check: bool = True
    source_check: bool = True
    min_confidence: float = 0.4
    poison_threshold: float = 0.5
    block_unknown_sources: bool = False
    require_source: bool = False
    strict_uncertainty: bool = False
    audit_all: bool = False
    quarantine_suspicious: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "poison_check": self.poison_check,
            "confidence_check": self.confidence_check,
            "source_check": self.source_check,
            "min_confidence": self.min_confidence,
            "poison_threshold": self.poison_threshold,
            "block_unknown_sources": self.block_unknown_sources,
            "require_source": self.require_source,
            "strict_uncertainty": self.strict_uncertainty,
            "audit_all": self.audit_all,
            "quarantine_suspicious": self.quarantine_suspicious,
        }


@dataclass
class ValidationResult:
    """
    Result of memory security validation.
    
    Attributes:
        is_safe: Whether the memory passed all checks
        is_quarantined: Whether the memory was quarantined
        memory_id: ID of the validated memory
        rejection_reason: Reason for rejection (if applicable)
        adjusted_confidence: Final adjusted confidence score
        poison_result: Detailed poison detection result
        confidence_result: Detailed confidence scoring result
        source_result: Detailed source validation result
        timestamp: When validation occurred
        validation_time_ms: Time taken for validation
    """
    
    is_safe: bool = True
    is_quarantined: bool = False
    memory_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    adjusted_confidence: float = 1.0
    poison_result: Optional[DetectionResult] = None
    confidence_result: Optional[ConfidenceResult] = None
    source_result: Optional[SourceValidationResult] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validation_time_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "is_quarantined": self.is_quarantined,
            "memory_id": self.memory_id,
            "rejection_reason": self.rejection_reason,
            "adjusted_confidence": self.adjusted_confidence,
            "poison_result": self.poison_result.to_dict() if self.poison_result else None,
            "confidence_result": self.confidence_result.to_dict() if self.confidence_result else None,
            "source_result": self.source_result.to_dict() if self.source_result else None,
            "timestamp": self.timestamp.isoformat(),
            "validation_time_ms": self.validation_time_ms,
        }


class MemoryGuard:
    """
    Main security guard for memory validation.
    
    Provides comprehensive security checks for memory content:
    - Poison/injection detection
    - Confidence scoring
    - Uncertainty detection
    - Source validation
    - Audit trail logging
    
    Example:
        guard = MemoryGuard(level=SecurityLevel.MEDIUM)
        
        result = guard.validate(memory)
        if result.is_safe:
            store.save(memory)
        else:
            print(f"Rejected: {result.rejection_reason}")
    
    For high-throughput scenarios:
        if guard.quick_check(content):
            # Fast path: content is likely safe
            store.save(memory)
        else:
            # Slow path: full validation needed
            result = guard.validate(memory)
    """
    
    def __init__(
        self,
        level: SecurityLevel = SecurityLevel.MEDIUM,
        config: Optional[SecurityConfig] = None,
        audit_path: Optional[Path] = None,
        enable_audit: bool = True,
    ):
        """
        Initialize memory guard.
        
        Args:
            level: Security level preset
            config: Custom configuration (overrides level)
            audit_path: Path for audit log file
            enable_audit: Enable audit logging
        """
        self._level = level
        self._config = config or level.config
        
        # Initialize components
        self._poison_detector = PoisonDetector(
            risk_threshold=self._config.poison_threshold,
        )
        
        self._confidence_scorer = ConfidenceScorer()
        
        self._uncertainty_detector = UncertaintyDetector(
            strict_mode=self._config.strict_uncertainty,
        )
        
        self._source_validator = SourceValidator(
            block_unknown=self._config.block_unknown_sources,
            min_trust_threshold=0.3,
        )
        
        # Initialize audit logger
        self._audit_logger: Optional[AuditLogger] = None
        if enable_audit:
            self._audit_logger = AuditLogger()
            self._memory_sink = MemoryAuditSink()
            self._audit_logger.add_sink(self._memory_sink)
            
            if audit_path:
                file_sink = FileAuditSink(audit_path)
                self._audit_logger.add_sink(file_sink)
            
            # Log initialization
            self._audit_logger.log(
                AuditEventType.GUARD_INITIALIZED,
                details={"level": level.value, "config": self._config.to_dict()},
            )
        
        # Statistics
        self._stats = {
            "total_validated": 0,
            "total_passed": 0,
            "total_rejected": 0,
            "total_quarantined": 0,
            "poison_detections": 0,
            "low_confidence": 0,
            "source_rejections": 0,
        }
    
    @property
    def level(self) -> SecurityLevel:
        """Get current security level."""
        return self._level
    
    @property
    def config(self) -> SecurityConfig:
        """Get current configuration."""
        return self._config
    
    @property
    def stats(self) -> dict[str, int]:
        """Get validation statistics."""
        return dict(self._stats)
    
    def set_level(self, level: SecurityLevel) -> None:
        """
        Change security level.
        
        Args:
            level: New security level
        """
        self._level = level
        self._config = level.config
        
        # Update components
        self._poison_detector.risk_threshold = self._config.poison_threshold
        self._source_validator.block_unknown = self._config.block_unknown_sources
        
        if self._audit_logger:
            self._audit_logger.log(
                AuditEventType.CONFIG_CHANGED,
                details={"new_level": level.value, "config": self._config.to_dict()},
            )
    
    def set_config(self, config: SecurityConfig) -> None:
        """
        Set custom configuration.
        
        Args:
            config: Custom configuration
        """
        self._config = config
        self._poison_detector.risk_threshold = config.poison_threshold
        self._source_validator.block_unknown = config.block_unknown_sources
        
        if self._audit_logger:
            self._audit_logger.log(
                AuditEventType.CONFIG_CHANGED,
                details={"config": config.to_dict()},
            )
    
    def validate(
        self,
        memory: Any,  # Memory object or dict
        content: Optional[str] = None,
        source: Optional[str] = None,
        base_confidence: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate a memory for security.
        
        Args:
            memory: Memory object to validate
            content: Override content (if not using Memory object)
            source: Override source (if not using Memory object)
            base_confidence: Override base confidence
            
        Returns:
            ValidationResult with detailed findings
        """
        import time
        start_time = time.perf_counter()
        
        # Extract content and metadata
        if hasattr(memory, "value"):
            # Memory object
            memory_id = getattr(memory, "memory_id", None)
            content = content if content is not None else memory.value
            source = source if source is not None else getattr(memory, "source", None)
            base_confidence = base_confidence if base_confidence is not None else getattr(memory, "confidence", 1.0)
        elif isinstance(memory, dict):
            memory_id = memory.get("memory_id")
            content = content if content is not None else memory.get("value", "")
            source = source if source is not None else memory.get("source")
            base_confidence = base_confidence if base_confidence is not None else memory.get("confidence", 1.0)
        else:
            # Assume content string
            memory_id = None
            content = content if content is not None else str(memory)
            base_confidence = base_confidence if base_confidence is not None else 1.0
        
        # Ensure we have valid values
        content = content or ""
        base_confidence = base_confidence if base_confidence is not None else 1.0
        
        self._stats["total_validated"] += 1
        
        rejection_reason = None
        is_safe = True
        is_quarantined = False
        adjusted_confidence = base_confidence
        
        poison_result: Optional[DetectionResult] = None
        confidence_result: Optional[ConfidenceResult] = None
        source_result: Optional[SourceValidationResult] = None
        
        # 1. Poison Detection
        if self._config.poison_check:
            poison_result = self._poison_detector.analyze(content)
            
            if not poison_result.is_safe:
                self._stats["poison_detections"] += 1
                
                if self._config.quarantine_suspicious:
                    is_quarantined = True
                    rejection_reason = f"Quarantined: Poison detected ({poison_result.risk_score:.2f})"
                else:
                    is_safe = False
                    rejection_reason = f"Poison detected: {', '.join(p.value for p in poison_result.detected_patterns)}"
                
                if self._audit_logger:
                    for pattern in poison_result.detected_patterns:
                        self._audit_logger.log_poison_detected(
                            memory_id or "unknown",
                            pattern.value,
                            poison_result.suspicious_segments[0] if poison_result.suspicious_segments else "",
                            poison_result.risk_score,
                        )
        
        # 2. Confidence Scoring
        if self._config.confidence_check and is_safe:
            confidence_result = self._confidence_scorer.score(
                content,
                source_confidence=1.0,  # Will be adjusted by source check
                base_confidence=base_confidence,
            )
            
            adjusted_confidence = confidence_result.confidence
            
            if adjusted_confidence < self._config.min_confidence:
                self._stats["low_confidence"] += 1
                is_safe = False
                rejection_reason = f"Low confidence: {adjusted_confidence:.2f} < {self._config.min_confidence}"
                
                if self._audit_logger:
                    self._audit_logger.log_low_confidence(
                        memory_id or "unknown",
                        adjusted_confidence,
                        self._config.min_confidence,
                        confidence_result.recommendation,
                    )
            
            if confidence_result.should_defer:
                if self._audit_logger:
                    self._audit_logger.log_uncertainty(
                        memory_id or "unknown",
                        "deferred",
                        confidence_result.uncertainty_indicators,
                    )
        
        # 3. Source Validation
        if self._config.source_check and is_safe:
            if self._config.require_source and not source:
                self._stats["source_rejections"] += 1
                is_safe = False
                rejection_reason = "Source required but not provided"
            else:
                source_result = self._source_validator.validate(source)
                
                if not source_result.is_valid:
                    self._stats["source_rejections"] += 1
                    is_safe = False
                    rejection_reason = f"Source validation failed: {'; '.join(source_result.reasons)}"
                else:
                    # Adjust confidence based on source trust
                    adjusted_confidence = self._confidence_scorer.adjust_for_source(
                        adjusted_confidence,
                        source_result.trust_score,
                    )
                
                if self._audit_logger:
                    self._audit_logger.log_source_validation(
                        memory_id or "unknown",
                        source or "unknown",
                        source_result.source_profile.trust_level.value if source_result.source_profile else "unknown",
                        source_result.is_valid,
                    )
        
        # Calculate validation time
        end_time = time.perf_counter()
        validation_time_ms = (end_time - start_time) * 1000
        
        # Update stats and record with source validator
        if is_safe and not is_quarantined:
            self._stats["total_passed"] += 1
            if source:
                self._source_validator.record_memory(source, rejected=False)
        elif is_quarantined:
            self._stats["total_quarantined"] += 1
        else:
            self._stats["total_rejected"] += 1
            if source:
                self._source_validator.record_memory(source, rejected=True)
        
        # Create result
        result = ValidationResult(
            is_safe=is_safe and not is_quarantined,
            is_quarantined=is_quarantined,
            memory_id=memory_id,
            rejection_reason=rejection_reason,
            adjusted_confidence=adjusted_confidence,
            poison_result=poison_result,
            confidence_result=confidence_result,
            source_result=source_result,
            validation_time_ms=validation_time_ms,
        )
        
        # Log validation result
        if self._audit_logger and (self._config.audit_all or not is_safe):
            self._audit_logger.log_validation(
                memory_id or "unknown",
                passed=is_safe and not is_quarantined,
                details=result.to_dict(),
            )
        
        return result
    
    def validate_content(self, content: str) -> ValidationResult:
        """
        Validate raw content string.
        
        Convenience method for validating content without a Memory object.
        """
        return self.validate(content, content=content)
    
    def quick_check(self, content: str) -> bool:
        """
        Fast check if content is likely safe.
        
        Returns True if content passes quick poison detection.
        Use for high-throughput filtering before full validation.
        
        Args:
            content: Content to check
            
        Returns:
            True if content is likely safe
        """
        return self._poison_detector.quick_check(content)
    
    def batch_validate(
        self,
        memories: list[Any],
        fail_fast: bool = False,
    ) -> list[ValidationResult]:
        """
        Validate multiple memories.
        
        Args:
            memories: List of memories to validate
            fail_fast: Stop on first failure
            
        Returns:
            List of validation results
        """
        results = []
        
        for memory in memories:
            result = self.validate(memory)
            results.append(result)
            
            if fail_fast and not result.is_safe:
                break
        
        return results
    
    def get_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit events."""
        if self._audit_logger and hasattr(self, "_memory_sink"):
            events = self._memory_sink.events[-limit:]
            return [e.to_dict() for e in events]
        return []
    
    def register_source(
        self,
        source_id: str,
        name: str,
        trust_level: SourceTrust = SourceTrust.KNOWN,
    ) -> None:
        """Register a source for validation."""
        self._source_validator.register_source(source_id, name, trust_level)
    
    def trust_source(self, source_id: str) -> None:
        """Mark a source as trusted."""
        self._source_validator.trust_source(source_id)
    
    def block_source(self, source_id: str, reason: str = "") -> None:
        """Block a source."""
        self._source_validator.block_source(source_id, reason)
    
    def get_poison_detector(self) -> PoisonDetector:
        """Get the poison detector for custom configuration."""
        return self._poison_detector
    
    def get_source_validator(self) -> SourceValidator:
        """Get the source validator for custom configuration."""
        return self._source_validator


# Convenience function for quick validation
def validate_memory(
    memory: Any,
    level: SecurityLevel = SecurityLevel.MEDIUM,
) -> ValidationResult:
    """
    Quick validation of a single memory.
    
    Creates a temporary guard for one-off validation.
    For repeated validations, create a MemoryGuard instance.
    
    Args:
        memory: Memory to validate
        level: Security level to use
        
    Returns:
        ValidationResult
    """
    guard = MemoryGuard(level=level, enable_audit=False)
    return guard.validate(memory)
