# Observability

Agent Memory Toolkit provides comprehensive production-ready observability including metrics collection, distributed tracing, structured logging, and Grafana dashboard generation.

## Overview

The observability module consists of four main components:

- **Metrics**: Prometheus-compatible metrics for operations, search, storage, and cache
- **Tracing**: Distributed tracing with OpenTelemetry, Jaeger, and OTLP support
- **Logging**: Structured JSON logging with correlation IDs and context propagation
- **Dashboards**: Grafana dashboard and alert rule generation

## Quick Start

```python
from agent_memory_toolkit.observability import (
    MemoryMetrics,
    OperationTracer,
    StructuredLogger,
    DashboardGenerator,
)

# Initialize metrics
metrics = MemoryMetrics(namespace="my_agent")

# Track operations with automatic timing
with metrics.track_operation("add"):
    store.add("my memory content")

# Initialize structured logging
logger = StructuredLogger("my_agent")
logger.info("Operation completed", memory_id="abc123", size=1024)

# Export Prometheus metrics
print(metrics.export_prometheus())
```

## Metrics

### Configuration

```python
from agent_memory_toolkit.observability import (
    MemoryMetrics,
    MetricsConfig,
)

config = MetricsConfig(
    namespace="my_agent",
    subsystem="memory",
    histogram_buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
    enable_timestamps=True,
)

metrics = MemoryMetrics(config=config)
```

### Operation Tracking

```python
# Track operations with automatic timing and error handling
with metrics.track_operation("add") as timer:
    result = store.add(content)
    
print(f"Operation took {timer.elapsed:.3f}s")

# Record search metrics
metrics.record_search(
    method="hybrid",
    duration=0.05,
    results_count=10,
)

# Record embedding generation
metrics.record_embedding(model="all-MiniLM-L6-v2", duration=0.02)
```

### Storage Metrics

```python
# Update storage statistics
metrics.update_storage_stats(
    total_memories=1000,
    storage_bytes=50_000_000,
    branch_count=3,
    memories_by_domain={
        "declarative": 500,
        "procedural": 300,
        "episodic": 200,
    },
)
```

### Cache Metrics

```python
# Track cache performance
metrics.record_cache_access(
    cache_name="embeddings",
    hit=True,
    current_size=100,
)
```

### Prometheus Export

```python
# Export in Prometheus text format
prometheus_output = metrics.export_prometheus()

# Example output:
# # HELP agent_memory_operations_total Total number of memory operations
# # TYPE agent_memory_operations_total counter
# agent_memory_operations_total{operation="add"} 150
# agent_memory_operations_total{operation="search"} 500

# Export as JSON for custom dashboards
json_output = metrics.export_json()
```

### Custom Metrics

```python
from agent_memory_toolkit.observability import Counter, Gauge, Histogram

# Register custom metrics
custom_counter = Counter(
    "custom_events_total",
    "Total custom events",
    labels=["event_type"],
    config=metrics.config,
)
metrics.register_metric(custom_counter)

custom_counter.inc(labels={"event_type": "sync"})
```

## Distributed Tracing

### Basic Usage

```python
from agent_memory_toolkit.observability import (
    OperationTracer,
    TracingConfig,
    ConsoleExporter,
)

tracer = OperationTracer(
    config=TracingConfig(
        service_name="my-agent",
        sample_rate=1.0,
    ),
    exporter=ConsoleExporter(),
)

# Create spans with automatic parent-child relationships
with tracer.start_span("memory_operation") as span:
    span.set_attribute("memory.type", "declarative")
    
    with tracer.start_span("embedding") as child:
        # child is automatically linked to parent
        embeddings = generate_embeddings(content)
        child.set_attribute("embedding.dimension", 384)
```

### Span Attributes and Events

```python
with tracer.start_span("search") as span:
    span.set_attributes({
        "search.query": query,
        "search.method": "hybrid",
        "search.limit": 10,
    })
    
    # Add events for important points
    span.add_event("query_parsed", {"tokens": 5})
    
    results = store.search(query)
    
    span.add_event("search_completed", {"results": len(results)})
    span.set_attribute("search.results_count", len(results))
```

### Error Handling

```python
with tracer.start_span("risky_operation") as span:
    try:
        result = do_risky_thing()
    except Exception as e:
        span.record_exception(e)
        raise
```

### Decorator-based Tracing

```python
@tracer.trace_operation("memory_search")
def search(query: str) -> list:
    return store.search(query)
```

### Exporters

```python
# Console (for development)
from agent_memory_toolkit.observability import ConsoleExporter
exporter = ConsoleExporter(pretty=True)

# Jaeger
from agent_memory_toolkit.observability import JaegerExporter
exporter = JaegerExporter(
    endpoint="http://localhost:14268/api/traces",
)

# OpenTelemetry Protocol (OTLP)
from agent_memory_toolkit.observability import OTLPExporter
exporter = OTLPExporter(
    endpoint="http://localhost:4318/v1/traces",
    headers={"Authorization": "Bearer token"},
)
```

### Context Propagation

```python
# Inject context into outgoing requests
headers = {}
tracer.inject_context(headers)
# headers now contains: {"traceparent": "00-trace_id-span_id-01"}

# Extract context from incoming requests
incoming_headers = {"traceparent": "00-abc123-def456-01"}
context = tracer.extract_context(incoming_headers)

# Start span with extracted context
with tracer.start_span("handle_request", links=[context]) as span:
    process_request()
```

## Structured Logging

### Configuration

```python
from agent_memory_toolkit.observability import (
    StructuredLogger,
    LogConfig,
    LogLevel,
)

config = LogConfig(
    level=LogLevel.INFO,
    format="json",  # "json" or "console"
    include_correlation_id=True,
    include_thread_info=False,
    output_file="/var/log/agent-memory.log",
    max_file_size_bytes=10 * 1024 * 1024,
    backup_count=5,
)

logger = StructuredLogger("my_agent", config=config)
```

### Basic Logging

```python
# Log with extra fields
logger.info("Memory added", memory_id="abc123", size=1024)
logger.debug("Cache lookup", cache_key="key1", hit=True)
logger.warning("High latency", duration_ms=500, threshold_ms=200)

# Log exceptions
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True, error_code="E001")
    # or
    logger.exception("Operation failed", error_code="E001")
```

### Log Context

```python
from agent_memory_toolkit.observability import LogContext

# Set context for all logs in scope
with LogContext.scope(request_id="req-001", user_id="user1"):
    logger.info("Processing request")  # includes request_id and user_id
    
    do_something()
    
    logger.info("Request completed")  # also includes context

# Manual context management
LogContext.set("session_id", "session-123")
logger.info("Session started")  # includes session_id
LogContext.remove("session_id")
```

### Correlation IDs

```python
from agent_memory_toolkit.observability import LogContext

# Auto-generate correlation ID
correlation_id = LogContext.set_correlation_id()
print(f"Correlation ID: {correlation_id}")

# Use existing correlation ID
LogContext.set_correlation_id("existing-correlation-id")

# All logs will include the correlation ID
logger.info("This log has a correlation ID")
```

### Bound Loggers

```python
# Create a logger with preset fields
request_logger = logger.bind(
    request_id="req-001",
    endpoint="/api/memories",
)

# All logs include bound fields
request_logger.info("Request started")
request_logger.info("Processing")
request_logger.info("Request completed", status_code=200)
```

### JSON Output Example

```json
{
  "timestamp": "2024-01-15T10:30:00.123456+00:00",
  "level": "INFO",
  "logger": "my_agent",
  "message": "Memory added",
  "correlation_id": "abc123def456",
  "memory_id": "mem-001",
  "size": 1024,
  "source": {
    "file": "main.py",
    "line": 42,
    "function": "add_memory"
  }
}
```

## Dashboard Generation

### Generate Grafana Dashboard

```python
from agent_memory_toolkit.observability import (
    DashboardGenerator,
    DashboardConfig,
    DataSource,
)

config = DashboardConfig(
    title="My Agent Memory Dashboard",
    uid="my-agent-memory",
    namespace="my_agent",
    datasource=DataSource(
        name="Prometheus",
        url="http://prometheus:9090",
    ),
)

generator = DashboardGenerator(config=config)
dashboard = generator.generate()

# Export to JSON file
generator.export_dashboard("dashboard.json")

# Or get JSON string
json_str = dashboard.to_json()
```

### Dashboard Contents

The generated dashboard includes:

1. **Overview Row**
   - Total memories count
   - Operations per second
   - Error rate gauge
   - Cache hit ratio gauge
   - Storage size
   - Average operation latency

2. **Operations Row**
   - Operations over time by type
   - Operation latency distribution (heatmap)
   - Errors over time by type
   - P95 latency by operation

3. **Search Row**
   - Searches by method (FTS, vector, hybrid)
   - Search latency percentiles (P50, P95, P99)
   - Results per search
   - Embedding generation time
   - Reranking time

4. **Storage Row**
   - Memory count over time
   - Memories by cognitive domain (pie chart)
   - Storage size by component
   - Branch and commit counts
   - Fragmentation ratio

5. **Cache Row**
   - Cache hit/miss rates
   - Cache hit ratio over time
   - Cache size
   - Cache evictions

### Generate Alert Rules

```python
# Get alert rules
rules = generator.generate_alert_rules()

# Export as Prometheus rules YAML
generator.export_prometheus_rules("alert_rules.yml")
```

### Default Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighErrorRate | Error rate > 5% for 5m | warning |
| HighOperationLatency | P95 latency > 1s for 5m | warning |
| LowCacheHitRatio | Cache hit ratio < 50% for 10m | warning |
| HighStorageFragmentation | Fragmentation > 50% for 30m | warning |
| NoRecentOperations | No operations for 15m | info |
| SearchLatencySpike | P99 search latency > 5s for 5m | critical |

### Custom Panels

```python
from agent_memory_toolkit.observability import (
    DashboardPanel,
    DashboardRow,
    PanelType,
    PrometheusQuery,
)

# Create custom panel
custom_panel = DashboardPanel(
    title="Custom Metric",
    panel_type=PanelType.TIMESERIES,
    queries=[
        PrometheusQuery(
            expr="my_custom_metric_total",
            legend_format="{{label}}",
        ),
    ],
    grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
    unit="ops",
)

# Add to dashboard
dashboard.add_row(DashboardRow(
    title="Custom Metrics",
    panels=[custom_panel],
))
```

## Integration with FastAPI

```python
from fastapi import FastAPI, Request
from agent_memory_toolkit.observability import (
    MemoryMetrics,
    StructuredLogger,
    LogContext,
)

app = FastAPI()
metrics = MemoryMetrics()
logger = StructuredLogger("api")

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    # Generate correlation ID
    correlation_id = request.headers.get("X-Correlation-ID")
    if not correlation_id:
        correlation_id = LogContext.set_correlation_id()
    else:
        LogContext.set_correlation_id(correlation_id)
    
    # Track request
    with metrics.track_operation("http_request"):
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
        )
        
        response = await call_next(request)
        
        logger.info(
            "Request completed",
            status_code=response.status_code,
        )
    
    response.headers["X-Correlation-ID"] = correlation_id
    return response

@app.get("/metrics")
async def get_metrics():
    return Response(
        content=metrics.export_prometheus(),
        media_type="text/plain",
    )
```

## Prometheus Scrape Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'agent-memory-toolkit'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## Grafana Setup

1. Import the generated dashboard JSON
2. Configure the Prometheus data source
3. Import the alert rules

Or use the CLI:

```bash
# Generate dashboard
amt observability dashboard --output dashboard.json

# Generate alert rules
amt observability alerts --output alert_rules.yml
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AMT_METRICS_NAMESPACE` | Metrics namespace prefix | `agent_memory` |
| `AMT_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `AMT_LOG_FORMAT` | Log format (json, console) | `json` |
| `AMT_TRACING_ENABLED` | Enable distributed tracing | `true` |
| `AMT_TRACING_SAMPLE_RATE` | Trace sampling rate (0.0-1.0) | `1.0` |
| `AMT_JAEGER_ENDPOINT` | Jaeger collector endpoint | - |
| `AMT_OTLP_ENDPOINT` | OTLP collector endpoint | - |

## Best Practices

1. **Use correlation IDs** for request tracing across services
2. **Set appropriate sample rates** in production (e.g., 0.1 for 10%)
3. **Use structured logging** with JSON format for log aggregation
4. **Monitor the key metrics**: error rate, latency percentiles, cache hit ratio
5. **Set up alerts** for critical conditions before they impact users
6. **Use bound loggers** for consistent context in request handlers
