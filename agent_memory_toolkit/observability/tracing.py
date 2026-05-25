"""
Operation Tracing Module

Distributed tracing for memory operations with support for OpenTelemetry,
Jaeger, and custom exporters. Enables tracking of operation flows across
components and services.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SpanKind(Enum):
    """Type of span in the trace."""
    
    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a span."""
    
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""
    
    service_name: str = "agent-memory-toolkit"
    service_version: str = "1.0.0"
    environment: str = "development"
    enabled: bool = True
    sample_rate: float = 1.0  # 1.0 = trace everything
    max_attributes: int = 128
    max_events: int = 128
    max_links: int = 128
    propagation_format: str = "w3c"  # w3c or b3
    export_timeout_ms: int = 5000
    export_batch_size: int = 512


@dataclass
class SpanContext:
    """
    Context for distributed tracing.
    
    Contains the trace ID, span ID, and propagation flags
    for correlating operations across services.
    """
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # 1 = sampled
    trace_state: str = ""
    
    @classmethod
    def generate(cls, parent: Optional["SpanContext"] = None) -> "SpanContext":
        """Generate a new span context."""
        trace_id = parent.trace_id if parent else uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        parent_span_id = parent.span_id if parent else None
        
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
    
    def to_w3c_traceparent(self) -> str:
        """Export as W3C Trace Context traceparent header."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"
    
    @classmethod
    def from_w3c_traceparent(cls, header: str) -> Optional["SpanContext"]:
        """Parse W3C Trace Context traceparent header."""
        try:
            parts = header.split("-")
            if len(parts) >= 4 and parts[0] == "00":
                return cls(
                    trace_id=parts[1],
                    span_id=parts[2],
                    trace_flags=int(parts[3], 16),
                )
        except (ValueError, IndexError):
            pass
        return None
    
    def to_b3_headers(self) -> Dict[str, str]:
        """Export as B3 propagation headers."""
        headers = {
            "X-B3-TraceId": self.trace_id,
            "X-B3-SpanId": self.span_id,
            "X-B3-Sampled": "1" if self.trace_flags & 1 else "0",
        }
        if self.parent_span_id:
            headers["X-B3-ParentSpanId"] = self.parent_span_id
        return headers
    
    @classmethod
    def from_b3_headers(cls, headers: Dict[str, str]) -> Optional["SpanContext"]:
        """Parse B3 propagation headers."""
        trace_id = headers.get("X-B3-TraceId")
        span_id = headers.get("X-B3-SpanId")
        
        if not trace_id or not span_id:
            return None
        
        sampled = headers.get("X-B3-Sampled", "1")
        parent_span_id = headers.get("X-B3-ParentSpanId")
        
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            trace_flags=1 if sampled == "1" else 0,
        )


@dataclass
class SpanEvent:
    """An event that occurred during a span."""
    
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """A link to another span in the same or different trace."""
    
    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    A span represents a single operation in a trace.
    
    Spans form a tree structure where each span may have
    multiple child spans representing sub-operations.
    """
    
    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set an attribute on the span."""
        self.attributes[key] = value
        return self
    
    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes on the span."""
        self.attributes.update(attributes)
        return self
    
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Add an event to the span."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=datetime.now(timezone.utc),
                attributes=attributes or {},
            )
        )
        return self
    
    def add_link(
        self,
        context: SpanContext,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Add a link to another span."""
        self.links.append(
            SpanLink(context=context, attributes=attributes or {})
        )
        return self
    
    def set_status(
        self, status: SpanStatus, message: str = ""
    ) -> "Span":
        """Set the span status."""
        self.status = status
        self.status_message = message
        return self
    
    def record_exception(
        self,
        exception: Exception,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Record an exception that occurred during the span."""
        exc_attrs = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
        }
        if attributes:
            exc_attrs.update(attributes)
        
        self.add_event("exception", exc_attrs)
        self.set_status(SpanStatus.ERROR, str(exception))
        return self
    
    def end(self) -> "Span":
        """End the span."""
        if self.end_time is None:
            self.end_time = datetime.now(timezone.utc)
        return self
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "status_message": self.status_message,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
            "links": [
                {
                    "trace_id": l.context.trace_id,
                    "span_id": l.context.span_id,
                    "attributes": l.attributes,
                }
                for l in self.links
            ],
        }


class TracingExporter(ABC):
    """Base class for trace exporters."""
    
    @abstractmethod
    def export(self, spans: List[Span]) -> bool:
        """Export spans. Returns True on success."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleExporter(TracingExporter):
    """Export spans to console for debugging."""
    
    def __init__(
        self,
        pretty: bool = True,
        output: Any = None,
    ) -> None:
        self.pretty = pretty
        self.output = output or sys.stdout
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to console."""
        for span in spans:
            data = span.to_dict()
            if self.pretty:
                output = json.dumps(data, indent=2)
            else:
                output = json.dumps(data)
            print(output, file=self.output)
        return True
    
    def shutdown(self) -> None:
        """No cleanup needed for console exporter."""
        pass


class JaegerExporter(TracingExporter):
    """
    Export spans to Jaeger via HTTP/Thrift.
    
    Requires jaeger-client or opentelemetry-exporter-jaeger package.
    Falls back to console export if not available.
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:14268/api/traces",
        timeout_ms: int = 5000,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._fallback = ConsoleExporter(pretty=False)
        
        # Try to import HTTP client
        try:
            import urllib.request
            self._http_available = True
        except ImportError:
            self._http_available = False
            logger.warning(
                "HTTP support not available, falling back to console export"
            )
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans to Jaeger."""
        if not self._http_available:
            return self._fallback.export(spans)
        
        try:
            import urllib.request
            
            # Convert spans to Jaeger format
            jaeger_spans = self._convert_to_jaeger(spans)
            payload = json.dumps(jaeger_spans).encode("utf-8")
            
            req = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            
            with urllib.request.urlopen(
                req, timeout=self.timeout_ms / 1000
            ) as response:
                return response.status == 200 or response.status == 202
        except Exception as e:
            logger.error(f"Failed to export to Jaeger: {e}")
            return False
    
    def _convert_to_jaeger(self, spans: List[Span]) -> Dict[str, Any]:
        """Convert spans to Jaeger Thrift format."""
        # Simplified Jaeger batch format
        return {
            "batch": {
                "process": {
                    "serviceName": "agent-memory-toolkit",
                },
                "spans": [span.to_dict() for span in spans],
            }
        }
    
    def shutdown(self) -> None:
        """No cleanup needed."""
        pass


class OTLPExporter(TracingExporter):
    """
    Export spans using OpenTelemetry Protocol (OTLP).
    
    Supports both gRPC and HTTP/JSON endpoints.
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        protocol: str = "http/json",
        headers: Optional[Dict[str, str]] = None,
        timeout_ms: int = 5000,
    ) -> None:
        self.endpoint = endpoint
        self.protocol = protocol
        self.headers = headers or {}
        self.timeout_ms = timeout_ms
        self._fallback = ConsoleExporter(pretty=False)
    
    def export(self, spans: List[Span]) -> bool:
        """Export spans via OTLP."""
        try:
            import urllib.request
            
            # Convert to OTLP format
            otlp_data = self._convert_to_otlp(spans)
            payload = json.dumps(otlp_data).encode("utf-8")
            
            headers = {
                "Content-Type": "application/json",
                **self.headers,
            }
            
            req = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers=headers,
                method="POST",
            )
            
            with urllib.request.urlopen(
                req, timeout=self.timeout_ms / 1000
            ) as response:
                return response.status in (200, 202)
        except Exception as e:
            logger.error(f"Failed to export via OTLP: {e}")
            return False
    
    def _convert_to_otlp(self, spans: List[Span]) -> Dict[str, Any]:
        """Convert spans to OTLP format."""
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "agent-memory-toolkit"}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "agent_memory_toolkit.observability"},
                            "spans": [
                                self._span_to_otlp(span) for span in spans
                            ],
                        }
                    ],
                }
            ]
        }
    
    def _span_to_otlp(self, span: Span) -> Dict[str, Any]:
        """Convert a single span to OTLP format."""
        return {
            "traceId": span.context.trace_id,
            "spanId": span.context.span_id,
            "parentSpanId": span.context.parent_span_id or "",
            "name": span.name,
            "kind": self._span_kind_to_otlp(span.kind),
            "startTimeUnixNano": int(span.start_time.timestamp() * 1e9),
            "endTimeUnixNano": int(span.end_time.timestamp() * 1e9) if span.end_time else 0,
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in span.attributes.items()
            ],
            "status": {
                "code": 2 if span.status == SpanStatus.ERROR else 1,
                "message": span.status_message,
            },
        }
    
    def _span_kind_to_otlp(self, kind: SpanKind) -> int:
        """Convert SpanKind to OTLP numeric value."""
        mapping = {
            SpanKind.INTERNAL: 1,
            SpanKind.SERVER: 2,
            SpanKind.CLIENT: 3,
            SpanKind.PRODUCER: 4,
            SpanKind.CONSUMER: 5,
        }
        return mapping.get(kind, 1)
    
    def shutdown(self) -> None:
        """No cleanup needed."""
        pass


# Thread-local storage for trace context
_trace_context = threading.local()


class TraceContext:
    """
    Thread-local trace context management.
    
    Manages the current active span for automatic parent-child
    relationship tracking.
    """
    
    @classmethod
    def get_current_span(cls) -> Optional[Span]:
        """Get the current active span."""
        stack = getattr(_trace_context, "span_stack", [])
        return stack[-1] if stack else None
    
    @classmethod
    def get_current_context(cls) -> Optional[SpanContext]:
        """Get the current span context."""
        span = cls.get_current_span()
        return span.context if span else None
    
    @classmethod
    def push_span(cls, span: Span) -> None:
        """Push a span onto the context stack."""
        if not hasattr(_trace_context, "span_stack"):
            _trace_context.span_stack = []
        _trace_context.span_stack.append(span)
    
    @classmethod
    def pop_span(cls) -> Optional[Span]:
        """Pop a span from the context stack."""
        stack = getattr(_trace_context, "span_stack", [])
        return stack.pop() if stack else None
    
    @classmethod
    def clear(cls) -> None:
        """Clear the trace context."""
        _trace_context.span_stack = []


class OperationTracer:
    """
    Main tracer for memory operations.
    
    Provides distributed tracing with automatic context propagation
    and multiple export backends.
    
    Example:
        >>> tracer = OperationTracer(
        ...     config=TracingConfig(service_name="my-agent"),
        ...     exporter=ConsoleExporter(),
        ... )
        >>> 
        >>> with tracer.start_span("memory_add") as span:
        ...     span.set_attribute("memory.size", 1024)
        ...     result = store.add(content)
        ...     span.set_attribute("memory.id", result.id)
        >>> 
        >>> # Nested spans
        >>> with tracer.start_span("search") as parent:
        ...     with tracer.start_span("embedding") as child:
        ...         # child automatically linked to parent
        ...         embeddings = generate_embeddings(query)
    """
    
    def __init__(
        self,
        config: Optional[TracingConfig] = None,
        exporter: Optional[TracingExporter] = None,
    ) -> None:
        self.config = config or TracingConfig()
        self.exporter = exporter or ConsoleExporter()
        
        self._buffer: List[Span] = []
        self._lock = threading.Lock()
        self._shutdown = False
    
    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanContext]] = None,
    ) -> Generator[Span, None, None]:
        """
        Start a new span.
        
        The span is automatically ended when the context manager exits.
        Parent context is automatically propagated from the current trace context.
        """
        if not self.config.enabled:
            # Return a no-op span
            yield Span(
                name=name,
                context=SpanContext(trace_id="", span_id=""),
            )
            return
        
        # Get parent context
        parent_context = TraceContext.get_current_context()
        
        # Check sampling
        if not self._should_sample(parent_context):
            yield Span(
                name=name,
                context=SpanContext(trace_id="", span_id=""),
            )
            return
        
        # Create span context
        context = SpanContext.generate(parent_context)
        
        # Create span
        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
        )
        
        # Add resource attributes
        span.set_attributes({
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment,
        })
        
        # Add links
        if links:
            for link_context in links:
                span.add_link(link_context)
        
        # Push to context stack
        TraceContext.push_span(span)
        
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            TraceContext.pop_span()
            self._buffer_span(span)
    
    def _should_sample(self, parent_context: Optional[SpanContext]) -> bool:
        """Determine if this span should be sampled."""
        if parent_context:
            # Respect parent sampling decision
            return bool(parent_context.trace_flags & 1)
        
        # Make sampling decision based on config
        import random
        return random.random() < self.config.sample_rate
    
    def _buffer_span(self, span: Span) -> None:
        """Buffer a span for export."""
        with self._lock:
            self._buffer.append(span)
            
            # Flush if buffer is full
            if len(self._buffer) >= self.config.export_batch_size:
                self._flush()
    
    def _flush(self) -> None:
        """Flush buffered spans to exporter."""
        with self._lock:
            if not self._buffer:
                return
            
            spans_to_export = self._buffer
            self._buffer = []
        
        try:
            self.exporter.export(spans_to_export)
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
    
    def inject_context(
        self,
        carrier: Dict[str, str],
        context: Optional[SpanContext] = None,
    ) -> None:
        """
        Inject trace context into a carrier (e.g., HTTP headers).
        
        Used for propagating context across service boundaries.
        """
        ctx = context or TraceContext.get_current_context()
        if not ctx:
            return
        
        if self.config.propagation_format == "w3c":
            carrier["traceparent"] = ctx.to_w3c_traceparent()
            if ctx.trace_state:
                carrier["tracestate"] = ctx.trace_state
        else:  # b3
            carrier.update(ctx.to_b3_headers())
    
    def extract_context(
        self,
        carrier: Dict[str, str],
    ) -> Optional[SpanContext]:
        """
        Extract trace context from a carrier (e.g., HTTP headers).
        
        Used for receiving context from upstream services.
        """
        if self.config.propagation_format == "w3c":
            traceparent = carrier.get("traceparent")
            if traceparent:
                return SpanContext.from_w3c_traceparent(traceparent)
        else:  # b3
            return SpanContext.from_b3_headers(carrier)
        
        return None
    
    def force_flush(self) -> None:
        """Force flush all buffered spans."""
        self._flush()
    
    def shutdown(self) -> None:
        """Shutdown the tracer and export remaining spans."""
        self._shutdown = True
        self._flush()
        self.exporter.shutdown()
    
    def get_active_span(self) -> Optional[Span]:
        """Get the currently active span."""
        return TraceContext.get_current_span()
    
    def trace_operation(
        self,
        operation_name: str,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator to trace a function as a span.
        
        Example:
            >>> @tracer.trace_operation("memory_search")
            ... def search(query: str) -> List[Result]:
            ...     return store.search(query)
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args: Any, **kwargs: Any) -> T:
                with self.start_span(operation_name) as span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    return func(*args, **kwargs)
            return wrapper
        return decorator
