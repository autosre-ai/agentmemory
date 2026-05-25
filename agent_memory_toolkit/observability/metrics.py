"""
Memory Metrics Module

Comprehensive metrics collection for agent memory systems including operation
timing, storage stats, cache performance, and search analytics.

Supports Prometheus-style metrics export and OpenMetrics format.
"""

from __future__ import annotations

import time
import threading
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

T = TypeVar("T")


class MetricType(Enum):
    """Type of metric being collected."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    
    namespace: str = "agent_memory"
    subsystem: str = ""
    enable_default_metrics: bool = True
    histogram_buckets: List[float] = field(
        default_factory=lambda: [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        ]
    )
    summary_quantiles: List[float] = field(
        default_factory=lambda: [0.5, 0.9, 0.95, 0.99]
    )
    max_label_cardinality: int = 100
    enable_timestamps: bool = True


class Metric(ABC):
    """Base class for all metrics."""
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        config: Optional[MetricsConfig] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.labels = labels or []
        self.config = config or MetricsConfig()
        self._lock = threading.RLock()
        self._values: Dict[tuple, Any] = defaultdict(lambda: 0)
        self._created_at = datetime.now(timezone.utc)
    
    def _make_key(self, label_values: Dict[str, str]) -> tuple:
        """Create a hashable key from label values."""
        return tuple(label_values.get(l, "") for l in self.labels)
    
    @abstractmethod
    def _format_value(self, key: tuple) -> str:
        """Format the metric value for export."""
        pass
    
    def _full_name(self) -> str:
        """Get the full metric name with namespace."""
        parts = [self.config.namespace]
        if self.config.subsystem:
            parts.append(self.config.subsystem)
        parts.append(self.name)
        return "_".join(parts)
    
    def _format_labels(self, key: tuple) -> str:
        """Format labels for Prometheus export."""
        if not self.labels:
            return ""
        pairs = [f'{l}="{v}"' for l, v in zip(self.labels, key)]
        return "{" + ",".join(pairs) + "}"
    
    def export_prometheus(self) -> str:
        """Export metric in Prometheus format."""
        lines = [
            f"# HELP {self._full_name()} {self.description}",
            f"# TYPE {self._full_name()} {self.metric_type.value}",
        ]
        
        with self._lock:
            for key in self._values:
                labels = self._format_labels(key)
                value = self._format_value(key)
                lines.append(f"{self._full_name()}{labels} {value}")
        
        return "\n".join(lines)
    
    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Return the type of this metric."""
        pass


class Counter(Metric):
    """A monotonically increasing counter metric."""
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.COUNTER
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        key = self._make_key(labels or {})
        with self._lock:
            self._values[key] += value
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current counter value."""
        key = self._make_key(labels or {})
        with self._lock:
            return self._values[key]
    
    def _format_value(self, key: tuple) -> str:
        return str(self._values[key])


class Gauge(Metric):
    """A metric that can go up and down."""
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.GAUGE
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge to a specific value."""
        key = self._make_key(labels or {})
        with self._lock:
            self._values[key] = value
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        key = self._make_key(labels or {})
        with self._lock:
            self._values[key] += value
    
    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        key = self._make_key(labels or {})
        with self._lock:
            self._values[key] -= value
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current gauge value."""
        key = self._make_key(labels or {})
        with self._lock:
            return self._values[key]
    
    def _format_value(self, key: tuple) -> str:
        return str(self._values[key])


@dataclass
class HistogramBucket:
    """A histogram bucket with upper bound and count."""
    
    upper_bound: float
    count: int = 0


class Histogram(Metric):
    """A histogram metric for tracking distributions."""
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        config: Optional[MetricsConfig] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        super().__init__(name, description, labels, config)
        self.buckets = sorted(buckets or self.config.histogram_buckets)
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._counts: Dict[tuple, int] = defaultdict(int)
        self._bucket_counts: Dict[tuple, List[int]] = {}
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM
    
    def observe(
        self, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record an observation."""
        key = self._make_key(labels or {})
        with self._lock:
            self._sums[key] += value
            self._counts[key] += 1
            
            if key not in self._bucket_counts:
                self._bucket_counts[key] = [0] * len(self.buckets)
            
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._bucket_counts[key][i] += 1
    
    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """Get the total count of observations."""
        key = self._make_key(labels or {})
        with self._lock:
            return self._counts[key]
    
    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the sum of all observations."""
        key = self._make_key(labels or {})
        with self._lock:
            return self._sums[key]
    
    def _format_value(self, key: tuple) -> str:
        # Histograms are formatted specially in export_prometheus
        return ""
    
    def export_prometheus(self) -> str:
        """Export histogram in Prometheus format."""
        lines = [
            f"# HELP {self._full_name()} {self.description}",
            f"# TYPE {self._full_name()} histogram",
        ]
        
        with self._lock:
            for key in set(list(self._counts.keys()) + list(self._sums.keys())):
                base_labels = self._format_labels(key)
                
                # Export bucket counts
                cumulative = 0
                for i, bound in enumerate(self.buckets):
                    if key in self._bucket_counts:
                        cumulative += self._bucket_counts[key][i]
                    
                    bucket_label = f'le="{bound}"'
                    if base_labels:
                        labels_str = base_labels[:-1] + "," + bucket_label + "}"
                    else:
                        labels_str = "{" + bucket_label + "}"
                    lines.append(
                        f"{self._full_name()}_bucket{labels_str} {cumulative}"
                    )
                
                # +Inf bucket
                inf_label = 'le="+Inf"'
                if base_labels:
                    labels_str = base_labels[:-1] + "," + inf_label + "}"
                else:
                    labels_str = "{" + inf_label + "}"
                lines.append(
                    f"{self._full_name()}_bucket{labels_str} {self._counts[key]}"
                )
                
                # Sum and count
                lines.append(
                    f"{self._full_name()}_sum{base_labels} {self._sums[key]}"
                )
                lines.append(
                    f"{self._full_name()}_count{base_labels} {self._counts[key]}"
                )
        
        return "\n".join(lines)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        histogram: Histogram,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self.histogram = histogram
        self.labels = labels
        self._start: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            duration = time.perf_counter() - self._start
            self.histogram.observe(duration, self.labels)
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time without stopping."""
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start


@dataclass
class OperationMetrics:
    """Metrics for memory operations (add, get, delete, update)."""
    
    operations_total: Counter
    operation_duration: Histogram
    operation_errors: Counter
    operation_size_bytes: Histogram
    
    @classmethod
    def create(cls, config: Optional[MetricsConfig] = None) -> "OperationMetrics":
        """Create a new set of operation metrics."""
        cfg = config or MetricsConfig()
        labels = ["operation", "status"]
        
        return cls(
            operations_total=Counter(
                "operations_total",
                "Total number of memory operations",
                labels=["operation"],
                config=cfg,
            ),
            operation_duration=Histogram(
                "operation_duration_seconds",
                "Duration of memory operations in seconds",
                labels=labels,
                config=cfg,
            ),
            operation_errors=Counter(
                "operation_errors_total",
                "Total number of operation errors",
                labels=["operation", "error_type"],
                config=cfg,
            ),
            operation_size_bytes=Histogram(
                "operation_size_bytes",
                "Size of memory operations in bytes",
                labels=["operation"],
                config=cfg,
                buckets=[64, 256, 1024, 4096, 16384, 65536, 262144, 1048576],
            ),
        )


@dataclass
class SearchMetrics:
    """Metrics for search operations."""
    
    searches_total: Counter
    search_duration: Histogram
    search_results_count: Histogram
    search_method_usage: Counter
    rerank_duration: Histogram
    embedding_duration: Histogram
    fts_hits: Counter
    vector_hits: Counter
    
    @classmethod
    def create(cls, config: Optional[MetricsConfig] = None) -> "SearchMetrics":
        """Create a new set of search metrics."""
        cfg = config or MetricsConfig()
        
        return cls(
            searches_total=Counter(
                "searches_total",
                "Total number of search operations",
                labels=["method"],  # fts, vector, hybrid
                config=cfg,
            ),
            search_duration=Histogram(
                "search_duration_seconds",
                "Duration of search operations in seconds",
                labels=["method"],
                config=cfg,
            ),
            search_results_count=Histogram(
                "search_results_count",
                "Number of results returned per search",
                labels=["method"],
                config=cfg,
                buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500],
            ),
            search_method_usage=Counter(
                "search_method_usage_total",
                "Usage count per search method",
                labels=["method"],
                config=cfg,
            ),
            rerank_duration=Histogram(
                "rerank_duration_seconds",
                "Duration of reranking operations",
                labels=[],
                config=cfg,
            ),
            embedding_duration=Histogram(
                "embedding_duration_seconds",
                "Duration of embedding generation",
                labels=["model"],
                config=cfg,
            ),
            fts_hits=Counter(
                "fts_hits_total",
                "Total FTS search hits",
                labels=[],
                config=cfg,
            ),
            vector_hits=Counter(
                "vector_hits_total",
                "Total vector search hits",
                labels=[],
                config=cfg,
            ),
        )


@dataclass
class StorageMetrics:
    """Metrics for storage usage and health."""
    
    memories_total: Gauge
    memories_by_domain: Gauge
    storage_bytes: Gauge
    branch_count: Gauge
    commit_count: Gauge
    index_size_bytes: Gauge
    fragmentation_ratio: Gauge
    
    @classmethod
    def create(cls, config: Optional[MetricsConfig] = None) -> "StorageMetrics":
        """Create a new set of storage metrics."""
        cfg = config or MetricsConfig()
        
        return cls(
            memories_total=Gauge(
                "memories_total",
                "Total number of memories stored",
                labels=[],
                config=cfg,
            ),
            memories_by_domain=Gauge(
                "memories_by_domain",
                "Number of memories per cognitive domain",
                labels=["domain"],
                config=cfg,
            ),
            storage_bytes=Gauge(
                "storage_bytes",
                "Total storage size in bytes",
                labels=["component"],  # db, fts_index, vector_index
                config=cfg,
            ),
            branch_count=Gauge(
                "branch_count",
                "Number of branches in the memory store",
                labels=[],
                config=cfg,
            ),
            commit_count=Gauge(
                "commit_count",
                "Total number of commits",
                labels=["branch"],
                config=cfg,
            ),
            index_size_bytes=Gauge(
                "index_size_bytes",
                "Size of search indexes in bytes",
                labels=["index_type"],
                config=cfg,
            ),
            fragmentation_ratio=Gauge(
                "fragmentation_ratio",
                "Storage fragmentation ratio (0-1)",
                labels=[],
                config=cfg,
            ),
        )


@dataclass
class CacheMetrics:
    """Metrics for cache performance."""
    
    cache_hits: Counter
    cache_misses: Counter
    cache_size: Gauge
    cache_evictions: Counter
    cache_hit_ratio: Gauge
    
    @classmethod
    def create(cls, config: Optional[MetricsConfig] = None) -> "CacheMetrics":
        """Create a new set of cache metrics."""
        cfg = config or MetricsConfig()
        
        return cls(
            cache_hits=Counter(
                "cache_hits_total",
                "Total number of cache hits",
                labels=["cache_name"],
                config=cfg,
            ),
            cache_misses=Counter(
                "cache_misses_total",
                "Total number of cache misses",
                labels=["cache_name"],
                config=cfg,
            ),
            cache_size=Gauge(
                "cache_size",
                "Current number of items in cache",
                labels=["cache_name"],
                config=cfg,
            ),
            cache_evictions=Counter(
                "cache_evictions_total",
                "Total number of cache evictions",
                labels=["cache_name"],
                config=cfg,
            ),
            cache_hit_ratio=Gauge(
                "cache_hit_ratio",
                "Cache hit ratio (0-1)",
                labels=["cache_name"],
                config=cfg,
            ),
        )


class MemoryMetrics:
    """
    Main metrics collector for agent memory systems.
    
    Provides comprehensive metrics covering operations, search,
    storage, and cache performance.
    
    Example:
        >>> metrics = MemoryMetrics(namespace="my_agent")
        >>> 
        >>> # Track an operation
        >>> with metrics.track_operation("add"):
        ...     store.add("memory content")
        >>> 
        >>> # Record search metrics
        >>> metrics.record_search("hybrid", duration=0.05, results_count=10)
        >>> 
        >>> # Export Prometheus metrics
        >>> print(metrics.export_prometheus())
    """
    
    def __init__(
        self,
        namespace: str = "agent_memory",
        subsystem: str = "",
        config: Optional[MetricsConfig] = None,
    ) -> None:
        self.config = config or MetricsConfig(
            namespace=namespace, subsystem=subsystem
        )
        
        # Initialize metric groups
        self.operations = OperationMetrics.create(self.config)
        self.search = SearchMetrics.create(self.config)
        self.storage = StorageMetrics.create(self.config)
        self.cache = CacheMetrics.create(self.config)
        
        # Custom metrics registry
        self._custom_metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()
    
    def register_metric(self, metric: Metric) -> None:
        """Register a custom metric."""
        with self._lock:
            self._custom_metrics[metric.name] = metric
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a registered metric by name."""
        with self._lock:
            return self._custom_metrics.get(name)
    
    @contextmanager
    def track_operation(
        self,
        operation: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Generator[Timer, None, None]:
        """
        Track an operation with timing and status.
        
        Example:
            >>> with metrics.track_operation("add") as timer:
            ...     store.add("content")
            >>> print(f"Operation took {timer.elapsed:.3f}s")
        """
        op_labels = {"operation": operation}
        status_labels = {"operation": operation, "status": "success"}
        
        self.operations.operations_total.inc(labels=op_labels)
        
        timer = Timer(self.operations.operation_duration, status_labels)
        try:
            with timer:
                yield timer
        except Exception as e:
            status_labels["status"] = "error"
            error_labels = {"operation": operation, "error_type": type(e).__name__}
            self.operations.operation_errors.inc(labels=error_labels)
            raise
    
    def record_search(
        self,
        method: str,
        duration: float,
        results_count: int,
    ) -> None:
        """Record search operation metrics."""
        labels = {"method": method}
        
        self.search.searches_total.inc(labels=labels)
        self.search.search_duration.observe(duration, labels=labels)
        self.search.search_results_count.observe(results_count, labels=labels)
        self.search.search_method_usage.inc(labels=labels)
    
    def record_embedding(self, model: str, duration: float) -> None:
        """Record embedding generation metrics."""
        self.search.embedding_duration.observe(
            duration, labels={"model": model}
        )
    
    def record_rerank(self, duration: float) -> None:
        """Record reranking operation metrics."""
        self.search.rerank_duration.observe(duration)
    
    def update_storage_stats(
        self,
        total_memories: int,
        storage_bytes: int,
        branch_count: int = 1,
        memories_by_domain: Optional[Dict[str, int]] = None,
    ) -> None:
        """Update storage metrics."""
        self.storage.memories_total.set(total_memories)
        self.storage.storage_bytes.set(
            storage_bytes, labels={"component": "total"}
        )
        self.storage.branch_count.set(branch_count)
        
        if memories_by_domain:
            for domain, count in memories_by_domain.items():
                self.storage.memories_by_domain.set(
                    count, labels={"domain": domain}
                )
    
    def record_cache_access(
        self,
        cache_name: str,
        hit: bool,
        current_size: Optional[int] = None,
    ) -> None:
        """Record cache access metrics."""
        labels = {"cache_name": cache_name}
        
        if hit:
            self.cache.cache_hits.inc(labels=labels)
        else:
            self.cache.cache_misses.inc(labels=labels)
        
        if current_size is not None:
            self.cache.cache_size.set(current_size, labels=labels)
        
        # Update hit ratio
        hits = self.cache.cache_hits.get(labels=labels)
        misses = self.cache.cache_misses.get(labels=labels)
        total = hits + misses
        if total > 0:
            self.cache.cache_hit_ratio.set(hits / total, labels=labels)
    
    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []
        
        # Export operation metrics
        for attr in ["operations_total", "operation_duration", 
                     "operation_errors", "operation_size_bytes"]:
            metric = getattr(self.operations, attr)
            lines.append(metric.export_prometheus())
        
        # Export search metrics
        for attr in ["searches_total", "search_duration", "search_results_count",
                     "search_method_usage", "rerank_duration", "embedding_duration",
                     "fts_hits", "vector_hits"]:
            metric = getattr(self.search, attr)
            lines.append(metric.export_prometheus())
        
        # Export storage metrics
        for attr in ["memories_total", "memories_by_domain", "storage_bytes",
                     "branch_count", "commit_count", "index_size_bytes",
                     "fragmentation_ratio"]:
            metric = getattr(self.storage, attr)
            lines.append(metric.export_prometheus())
        
        # Export cache metrics
        for attr in ["cache_hits", "cache_misses", "cache_size",
                     "cache_evictions", "cache_hit_ratio"]:
            metric = getattr(self.cache, attr)
            lines.append(metric.export_prometheus())
        
        # Export custom metrics
        with self._lock:
            for metric in self._custom_metrics.values():
                lines.append(metric.export_prometheus())
        
        return "\n\n".join(lines)
    
    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON for custom dashboards."""
        result: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "namespace": self.config.namespace,
            "metrics": {},
        }
        
        # Collect all metric values
        metrics_dict = result["metrics"]
        
        # Operations
        metrics_dict["operations"] = {
            "total": dict(self.operations.operations_total._values),
            "errors": dict(self.operations.operation_errors._values),
        }
        
        # Search
        metrics_dict["search"] = {
            "total": dict(self.search.searches_total._values),
            "method_usage": dict(self.search.search_method_usage._values),
        }
        
        # Storage
        metrics_dict["storage"] = {
            "memories_total": self.storage.memories_total._values.get((), 0),
            "storage_bytes": dict(self.storage.storage_bytes._values),
            "branch_count": self.storage.branch_count._values.get((), 0),
        }
        
        # Cache
        metrics_dict["cache"] = {
            "hits": dict(self.cache.cache_hits._values),
            "misses": dict(self.cache.cache_misses._values),
            "hit_ratio": dict(self.cache.cache_hit_ratio._values),
        }
        
        return result
    
    def reset(self) -> None:
        """Reset all metrics to zero. Useful for testing."""
        # Reset all metric groups by recreating them
        self.operations = OperationMetrics.create(self.config)
        self.search = SearchMetrics.create(self.config)
        self.storage = StorageMetrics.create(self.config)
        self.cache = CacheMetrics.create(self.config)
        
        with self._lock:
            self._custom_metrics.clear()
