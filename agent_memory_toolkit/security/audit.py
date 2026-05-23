"""
Audit Trail Logging

Provides comprehensive audit logging for memory security operations.
Supports JSON logging, rotation, and structured event tracking.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Callable
import hashlib
import threading


class AuditEventType(Enum):
    """Types of audit events for memory security."""
    
    # Validation events
    MEMORY_VALIDATED = "memory_validated"
    MEMORY_REJECTED = "memory_rejected"
    MEMORY_QUARANTINED = "memory_quarantined"
    
    # Detection events
    POISON_DETECTED = "poison_detected"
    INJECTION_DETECTED = "injection_detected"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    
    # Confidence events
    LOW_CONFIDENCE = "low_confidence"
    UNCERTAINTY_DETECTED = "uncertainty_detected"
    CONFIDENCE_ADJUSTED = "confidence_adjusted"
    
    # Source events
    SOURCE_VALIDATED = "source_validated"
    SOURCE_REJECTED = "source_rejected"
    SOURCE_UNKNOWN = "source_unknown"
    
    # System events
    CONFIG_CHANGED = "config_changed"
    GUARD_INITIALIZED = "guard_initialized"
    GUARD_ERROR = "guard_error"


@dataclass
class AuditEvent:
    """
    A single audit event for memory security operations.
    
    Attributes:
        event_type: Type of the audit event
        timestamp: When the event occurred
        memory_id: ID of the memory involved (if applicable)
        details: Additional event details
        severity: Event severity (info, warning, critical)
        actor: Who/what triggered the event
        session_id: Optional session identifier
        event_id: Unique event identifier
    """
    
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    memory_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    actor: str = "system"
    session_id: Optional[str] = None
    event_id: str = field(default="")
    
    def __post_init__(self):
        """Generate event_id if not provided."""
        if not self.event_id:
            content = f"{self.event_type.value}:{self.timestamp.isoformat()}:{self.memory_id}"
            self.event_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "memory_id": self.memory_id,
            "details": self.details,
            "severity": self.severity,
            "actor": self.actor,
            "session_id": self.session_id,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        return cls(
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            memory_id=data.get("memory_id"),
            details=data.get("details", {}),
            severity=data.get("severity", "info"),
            actor=data.get("actor", "system"),
            session_id=data.get("session_id"),
            event_id=data.get("event_id", ""),
        )


class AuditSink(Protocol):
    """Protocol for audit event sinks."""
    
    def write(self, event: AuditEvent) -> None:
        """Write an audit event."""
        ...
    
    def flush(self) -> None:
        """Flush any buffered events."""
        ...


class FileAuditSink:
    """Write audit events to a JSON-lines file."""
    
    def __init__(
        self,
        path: Path,
        max_size_mb: float = 100.0,
        rotate_count: int = 5,
    ):
        """
        Initialize file audit sink.
        
        Args:
            path: Path to audit log file
            max_size_mb: Maximum file size before rotation
            rotate_count: Number of rotated files to keep
        """
        self.path = Path(path)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.rotate_count = rotate_count
        self._lock = threading.Lock()
        
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, event: AuditEvent) -> None:
        """Write event to file."""
        with self._lock:
            self._rotate_if_needed()
            with open(self.path, "a") as f:
                f.write(event.to_json() + "\n")
    
    def flush(self) -> None:
        """No buffering, nothing to flush."""
        pass
    
    def _rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        if not self.path.exists():
            return
        
        if self.path.stat().st_size < self.max_size_bytes:
            return
        
        # Rotate existing files
        for i in range(self.rotate_count - 1, 0, -1):
            old_path = self.path.with_suffix(f".{i}.jsonl")
            new_path = self.path.with_suffix(f".{i + 1}.jsonl")
            if old_path.exists():
                old_path.rename(new_path)
        
        # Rotate current file
        self.path.rename(self.path.with_suffix(".1.jsonl"))


class MemoryAuditSink:
    """In-memory audit sink for testing and short-term analysis."""
    
    def __init__(self, max_events: int = 10000):
        """
        Initialize memory sink.
        
        Args:
            max_events: Maximum events to keep in memory
        """
        self.max_events = max_events
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()
    
    def write(self, event: AuditEvent) -> None:
        """Add event to memory."""
        with self._lock:
            self._events.append(event)
            # Trim if needed
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]
    
    def flush(self) -> None:
        """Nothing to flush."""
        pass
    
    @property
    def events(self) -> list[AuditEvent]:
        """Get all events."""
        with self._lock:
            return list(self._events)
    
    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self._events.clear()
    
    def filter_by_type(self, event_type: AuditEventType) -> list[AuditEvent]:
        """Get events of a specific type."""
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]
    
    def filter_by_severity(self, severity: str) -> list[AuditEvent]:
        """Get events of a specific severity."""
        with self._lock:
            return [e for e in self._events if e.severity == severity]


class AuditLogger:
    """
    Main audit logging interface for memory security.
    
    Supports multiple sinks and provides convenient methods for
    logging common security events.
    
    Example:
        logger = AuditLogger()
        logger.add_sink(FileAuditSink(Path("audit.jsonl")))
        logger.log_validation("mem_123", passed=True, details={"score": 0.95})
    """
    
    def __init__(
        self,
        sinks: Optional[list[AuditSink]] = None,
        default_actor: str = "memory_guard",
        session_id: Optional[str] = None,
    ):
        """
        Initialize audit logger.
        
        Args:
            sinks: List of audit sinks to write to
            default_actor: Default actor for events
            session_id: Optional session identifier
        """
        self._sinks = sinks or []
        self._default_actor = default_actor
        self._session_id = session_id
        self._callbacks: list[Callable[[AuditEvent], None]] = []
    
    def add_sink(self, sink: AuditSink) -> None:
        """Add an audit sink."""
        self._sinks.append(sink)
    
    def add_callback(self, callback: Callable[[AuditEvent], None]) -> None:
        """Add a callback to be called for each event."""
        self._callbacks.append(callback)
    
    def log(
        self,
        event_type: AuditEventType,
        memory_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        severity: str = "info",
        actor: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            memory_id: ID of memory involved
            details: Additional details
            severity: Event severity
            actor: Who triggered the event
            
        Returns:
            The created audit event
        """
        event = AuditEvent(
            event_type=event_type,
            memory_id=memory_id,
            details=details or {},
            severity=severity,
            actor=actor or self._default_actor,
            session_id=self._session_id,
        )
        
        # Write to all sinks
        for sink in self._sinks:
            try:
                sink.write(event)
            except Exception as e:
                # Log to Python logger as fallback
                logging.error(f"Failed to write audit event: {e}")
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logging.error(f"Audit callback error: {e}")
        
        return event
    
    def log_validation(
        self,
        memory_id: str,
        passed: bool,
        details: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log a memory validation event."""
        return self.log(
            event_type=AuditEventType.MEMORY_VALIDATED if passed else AuditEventType.MEMORY_REJECTED,
            memory_id=memory_id,
            details=details or {},
            severity="info" if passed else "warning",
        )
    
    def log_poison_detected(
        self,
        memory_id: str,
        pattern_name: str,
        matched_content: str,
        risk_score: float,
    ) -> AuditEvent:
        """Log poison detection event."""
        return self.log(
            event_type=AuditEventType.POISON_DETECTED,
            memory_id=memory_id,
            details={
                "pattern": pattern_name,
                "matched": matched_content[:100],  # Truncate for safety
                "risk_score": risk_score,
            },
            severity="critical" if risk_score > 0.8 else "warning",
        )
    
    def log_injection_detected(
        self,
        memory_id: str,
        injection_type: str,
        content_preview: str,
    ) -> AuditEvent:
        """Log injection detection event."""
        return self.log(
            event_type=AuditEventType.INJECTION_DETECTED,
            memory_id=memory_id,
            details={
                "type": injection_type,
                "preview": content_preview[:50],
            },
            severity="critical",
        )
    
    def log_low_confidence(
        self,
        memory_id: str,
        confidence: float,
        threshold: float,
        reason: str,
    ) -> AuditEvent:
        """Log low confidence event."""
        return self.log(
            event_type=AuditEventType.LOW_CONFIDENCE,
            memory_id=memory_id,
            details={
                "confidence": confidence,
                "threshold": threshold,
                "reason": reason,
            },
            severity="warning",
        )
    
    def log_uncertainty(
        self,
        memory_id: str,
        uncertainty_type: str,
        indicators: list[str],
    ) -> AuditEvent:
        """Log uncertainty detection event."""
        return self.log(
            event_type=AuditEventType.UNCERTAINTY_DETECTED,
            memory_id=memory_id,
            details={
                "type": uncertainty_type,
                "indicators": indicators,
            },
            severity="info",
        )
    
    def log_source_validation(
        self,
        memory_id: str,
        source: str,
        trust_level: str,
        passed: bool,
    ) -> AuditEvent:
        """Log source validation event."""
        return self.log(
            event_type=AuditEventType.SOURCE_VALIDATED if passed else AuditEventType.SOURCE_REJECTED,
            memory_id=memory_id,
            details={
                "source": source,
                "trust_level": trust_level,
            },
            severity="info" if passed else "warning",
        )
    
    def log_error(
        self,
        error: str,
        memory_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log a guard error."""
        return self.log(
            event_type=AuditEventType.GUARD_ERROR,
            memory_id=memory_id,
            details={"error": error, **(details or {})},
            severity="critical",
        )
    
    def flush(self) -> None:
        """Flush all sinks."""
        for sink in self._sinks:
            try:
                sink.flush()
            except Exception as e:
                logging.error(f"Failed to flush audit sink: {e}")
