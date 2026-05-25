"""
Observability Module for Agent Memory Toolkit

Production-ready observability including metrics, tracing, structured logging,
and dashboard generation for monitoring agent memory systems at scale.

Usage:
    from agent_memory_toolkit.observability import (
        MemoryMetrics,
        OperationTracer,
        StructuredLogger,
        DashboardGenerator,
    )
    
    # Initialize metrics
    metrics = MemoryMetrics(namespace="my_agent")
    
    # Track operations
    with metrics.track_operation("memory_add"):
        store.add("some memory")
    
    # Export Prometheus metrics
    print(metrics.export_prometheus())
    
    # Generate Grafana dashboard
    generator = DashboardGenerator(metrics)
    dashboard_json = generator.generate()
"""

from .metrics import (
    MemoryMetrics,
    MetricsConfig,
    MetricType,
    Counter,
    Gauge,
    Histogram,
    Timer,
    OperationMetrics,
    SearchMetrics,
    StorageMetrics,
    CacheMetrics,
)
from .tracing import (
    OperationTracer,
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    TracingConfig,
    TracingExporter,
    ConsoleExporter,
    JaegerExporter,
    OTLPExporter,
    TraceContext,
)
from .logging import (
    StructuredLogger,
    LogLevel,
    LogConfig,
    LogFormatter,
    JSONFormatter,
    ConsoleFormatter,
    LogHandler,
    FileHandler,
    RotatingFileHandler,
    StreamHandler,
    LogContext,
    CorrelationIdFilter,
)
from .dashboards import (
    DashboardGenerator,
    GrafanaDashboard,
    DashboardPanel,
    PanelType,
    DashboardRow,
    DataSource,
    PrometheusQuery,
    AlertRule,
    AlertCondition,
    DashboardConfig,
)

__all__ = [
    # Metrics
    "MemoryMetrics",
    "MetricsConfig",
    "MetricType",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "OperationMetrics",
    "SearchMetrics",
    "StorageMetrics",
    "CacheMetrics",
    # Tracing
    "OperationTracer",
    "Span",
    "SpanContext",
    "SpanKind",
    "SpanStatus",
    "TracingConfig",
    "TracingExporter",
    "ConsoleExporter",
    "JaegerExporter",
    "OTLPExporter",
    "TraceContext",
    # Logging
    "StructuredLogger",
    "LogLevel",
    "LogConfig",
    "LogFormatter",
    "JSONFormatter",
    "ConsoleFormatter",
    "LogHandler",
    "FileHandler",
    "RotatingFileHandler",
    "StreamHandler",
    "LogContext",
    "CorrelationIdFilter",
    # Dashboards
    "DashboardGenerator",
    "GrafanaDashboard",
    "DashboardPanel",
    "PanelType",
    "DashboardRow",
    "DataSource",
    "PrometheusQuery",
    "AlertRule",
    "AlertCondition",
    "DashboardConfig",
]
