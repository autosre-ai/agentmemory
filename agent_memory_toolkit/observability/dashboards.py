"""
Dashboard Generation Module

Generate Grafana dashboards and alert rules for monitoring agent memory
systems with Prometheus metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class PanelType(Enum):
    """Types of Grafana panels."""
    
    GRAPH = "graph"
    STAT = "stat"
    GAUGE = "gauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    TEXT = "text"
    ROW = "row"
    TIMESERIES = "timeseries"
    BAR_GAUGE = "bargauge"
    PIE_CHART = "piechart"


@dataclass
class DataSource:
    """Grafana data source configuration."""
    
    name: str = "Prometheus"
    type: str = "prometheus"
    uid: str = "prometheus"
    url: str = "http://localhost:9090"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to Grafana datasource reference."""
        return {
            "type": self.type,
            "uid": self.uid,
        }


@dataclass
class PrometheusQuery:
    """A Prometheus query for a panel."""
    
    expr: str
    legend_format: str = ""
    ref_id: str = "A"
    instant: bool = False
    range_query: bool = True
    interval: str = ""
    
    def to_dict(self, datasource: DataSource) -> Dict[str, Any]:
        """Convert to Grafana target format."""
        return {
            "datasource": datasource.to_dict(),
            "expr": self.expr,
            "legendFormat": self.legend_format,
            "refId": self.ref_id,
            "instant": self.instant,
            "range": self.range_query,
            "interval": self.interval,
        }


@dataclass
class AlertCondition:
    """Condition for an alert rule."""
    
    evaluator_type: str = "gt"  # gt, lt, within_range, outside_range
    evaluator_params: List[float] = field(default_factory=lambda: [0])
    operator_type: str = "and"  # and, or
    reducer_type: str = "last"  # avg, min, max, sum, count, last, median
    query_ref_id: str = "A"


@dataclass
class AlertRule:
    """Grafana alert rule definition."""
    
    name: str
    expr: str
    for_duration: str = "5m"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    severity: str = "warning"  # info, warning, critical
    
    def __post_init__(self) -> None:
        if "severity" not in self.labels:
            self.labels["severity"] = self.severity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Grafana alert rule format."""
        return {
            "alert": self.name,
            "expr": self.expr,
            "for": self.for_duration,
            "labels": self.labels,
            "annotations": self.annotations,
        }
    
    def to_prometheus_rule(self) -> Dict[str, Any]:
        """Convert to Prometheus alerting rule format."""
        return {
            "alert": self.name,
            "expr": self.expr,
            "for": self.for_duration,
            "labels": self.labels,
            "annotations": {
                "summary": self.annotations.get(
                    "summary", f"Alert: {self.name}"
                ),
                "description": self.annotations.get(
                    "description", self.expr
                ),
            },
        }


@dataclass
class DashboardPanel:
    """A panel in a Grafana dashboard."""
    
    title: str
    panel_type: PanelType
    queries: List[PrometheusQuery]
    grid_pos: Dict[str, int] = field(
        default_factory=lambda: {"x": 0, "y": 0, "w": 12, "h": 8}
    )
    description: str = ""
    unit: str = ""
    thresholds: Optional[List[Dict[str, Any]]] = None
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)
    
    _id_counter: int = field(default=0, repr=False)
    
    def to_dict(
        self,
        panel_id: int,
        datasource: DataSource,
    ) -> Dict[str, Any]:
        """Convert to Grafana panel format."""
        panel: Dict[str, Any] = {
            "id": panel_id,
            "title": self.title,
            "type": self.panel_type.value,
            "gridPos": self.grid_pos,
            "targets": [
                q.to_dict(datasource) for q in self.queries
            ],
        }
        
        if self.description:
            panel["description"] = self.description
        
        # Panel options
        panel["options"] = self.options.copy()
        
        # Field configuration
        field_config = {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {},
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
            },
            "overrides": [],
        }
        
        if self.unit:
            field_config["defaults"]["unit"] = self.unit
        
        if self.thresholds:
            field_config["defaults"]["thresholds"]["steps"] = self.thresholds
        
        field_config.update(self.field_config)
        panel["fieldConfig"] = field_config
        
        return panel


@dataclass
class DashboardRow:
    """A row in a Grafana dashboard."""
    
    title: str
    panels: List[DashboardPanel] = field(default_factory=list)
    collapsed: bool = False
    
    def to_dict(
        self,
        start_id: int,
        y_offset: int,
        datasource: DataSource,
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Convert to Grafana row panel format.
        
        Returns list of panels and the next Y offset.
        """
        panels: List[Dict[str, Any]] = []
        current_id = start_id
        current_y = y_offset
        
        # Row panel
        row_panel = {
            "id": current_id,
            "type": "row",
            "title": self.title,
            "gridPos": {"x": 0, "y": current_y, "w": 24, "h": 1},
            "collapsed": self.collapsed,
        }
        panels.append(row_panel)
        current_id += 1
        current_y += 1
        
        # Panels in row
        max_height = 0
        x_offset = 0
        
        for panel in self.panels:
            # Update grid position
            panel.grid_pos["y"] = current_y
            panel.grid_pos["x"] = x_offset
            
            panel_dict = panel.to_dict(current_id, datasource)
            
            if self.collapsed:
                # Include panels in row panel for collapsed rows
                if "panels" not in row_panel:
                    row_panel["panels"] = []
                row_panel["panels"].append(panel_dict)
            else:
                panels.append(panel_dict)
            
            current_id += 1
            x_offset += panel.grid_pos["w"]
            max_height = max(max_height, panel.grid_pos["h"])
            
            # Wrap to next line if needed
            if x_offset >= 24:
                x_offset = 0
                current_y += max_height
                max_height = 0
        
        if not self.collapsed and max_height > 0:
            current_y += max_height
        
        return panels, current_y


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    
    title: str = "Agent Memory Toolkit"
    uid: str = "agent-memory-toolkit"
    description: str = "Monitoring dashboard for Agent Memory Toolkit"
    tags: List[str] = field(
        default_factory=lambda: ["agent-memory", "observability"]
    )
    refresh: str = "10s"
    time_from: str = "now-1h"
    time_to: str = "now"
    timezone: str = "browser"
    editable: bool = True
    datasource: DataSource = field(default_factory=DataSource)
    namespace: str = "agent_memory"


@dataclass
class GrafanaDashboard:
    """
    Complete Grafana dashboard definition.
    
    Contains all rows, panels, and configuration needed to
    create a dashboard JSON file.
    """
    
    config: DashboardConfig
    rows: List[DashboardRow] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    templating_vars: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_row(self, row: DashboardRow) -> "GrafanaDashboard":
        """Add a row to the dashboard."""
        self.rows.append(row)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Grafana dashboard JSON format."""
        panels: List[Dict[str, Any]] = []
        panel_id = 1
        y_offset = 0
        
        for row in self.rows:
            row_panels, y_offset = row.to_dict(
                panel_id, y_offset, self.config.datasource
            )
            panels.extend(row_panels)
            panel_id += len(row_panels)
        
        return {
            "id": None,
            "uid": self.config.uid,
            "title": self.config.title,
            "description": self.config.description,
            "tags": self.config.tags,
            "style": "dark",
            "timezone": self.config.timezone,
            "editable": self.config.editable,
            "graphTooltip": 1,  # Shared crosshair
            "refresh": self.config.refresh,
            "time": {
                "from": self.config.time_from,
                "to": self.config.time_to,
            },
            "panels": panels,
            "annotations": {
                "list": self.annotations,
            },
            "templating": {
                "list": self.templating_vars,
            },
            "schemaVersion": 38,
            "version": 1,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export dashboard as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class DashboardGenerator:
    """
    Generate Grafana dashboards for agent memory metrics.
    
    Creates comprehensive dashboards with panels for operations,
    search, storage, and cache metrics.
    
    Example:
        >>> generator = DashboardGenerator()
        >>> dashboard = generator.generate()
        >>> 
        >>> # Export to JSON
        >>> with open("dashboard.json", "w") as f:
        ...     f.write(dashboard.to_json())
        >>> 
        >>> # Get alert rules
        >>> rules = generator.generate_alert_rules()
    """
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
    ) -> None:
        self.config = config or DashboardConfig()
        self.namespace = self.config.namespace
    
    def generate(self) -> GrafanaDashboard:
        """Generate a complete Grafana dashboard."""
        dashboard = GrafanaDashboard(config=self.config)
        
        # Add standard template variables
        dashboard.templating_vars = self._generate_template_vars()
        
        # Add rows
        dashboard.add_row(self._generate_overview_row())
        dashboard.add_row(self._generate_operations_row())
        dashboard.add_row(self._generate_search_row())
        dashboard.add_row(self._generate_storage_row())
        dashboard.add_row(self._generate_cache_row())
        
        return dashboard
    
    def _generate_template_vars(self) -> List[Dict[str, Any]]:
        """Generate Grafana template variables."""
        return [
            {
                "name": "datasource",
                "type": "datasource",
                "query": "prometheus",
                "current": {
                    "selected": True,
                    "text": "Prometheus",
                    "value": "Prometheus",
                },
            },
            {
                "name": "instance",
                "type": "query",
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
                "query": f'label_values({self.namespace}_operations_total, instance)',
                "refresh": 2,
                "includeAll": True,
                "multi": True,
            },
        ]
    
    def _generate_overview_row(self) -> DashboardRow:
        """Generate overview metrics row."""
        return DashboardRow(
            title="Overview",
            panels=[
                # Total memories stat
                DashboardPanel(
                    title="Total Memories",
                    panel_type=PanelType.STAT,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_memories_total",
                            legend_format="Memories",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 0, "w": 4, "h": 4},
                    unit="short",
                    options={"colorMode": "value", "graphMode": "area"},
                ),
                # Operations rate
                DashboardPanel(
                    title="Operations/sec",
                    panel_type=PanelType.STAT,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum(rate({self.namespace}_operations_total[5m]))",
                            legend_format="ops/s",
                        ),
                    ],
                    grid_pos={"x": 4, "y": 0, "w": 4, "h": 4},
                    unit="ops",
                    options={"colorMode": "value", "graphMode": "area"},
                ),
                # Error rate
                DashboardPanel(
                    title="Error Rate",
                    panel_type=PanelType.GAUGE,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum(rate({self.namespace}_operation_errors_total[5m])) / sum(rate({self.namespace}_operations_total[5m])) * 100",
                            legend_format="Error %",
                        ),
                    ],
                    grid_pos={"x": 8, "y": 0, "w": 4, "h": 4},
                    unit="percent",
                    thresholds=[
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 1},
                        {"color": "red", "value": 5},
                    ],
                ),
                # Cache hit ratio
                DashboardPanel(
                    title="Cache Hit Ratio",
                    panel_type=PanelType.GAUGE,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_cache_hit_ratio * 100",
                            legend_format="Hit %",
                        ),
                    ],
                    grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
                    unit="percent",
                    thresholds=[
                        {"color": "red", "value": None},
                        {"color": "yellow", "value": 50},
                        {"color": "green", "value": 80},
                    ],
                ),
                # Storage size
                DashboardPanel(
                    title="Storage Size",
                    panel_type=PanelType.STAT,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum({self.namespace}_storage_bytes)",
                            legend_format="Size",
                        ),
                    ],
                    grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
                    unit="bytes",
                    options={"colorMode": "value", "graphMode": "area"},
                ),
                # Average latency
                DashboardPanel(
                    title="Avg Operation Latency",
                    panel_type=PanelType.STAT,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum(rate({self.namespace}_operation_duration_seconds_sum[5m])) / sum(rate({self.namespace}_operation_duration_seconds_count[5m]))",
                            legend_format="Latency",
                        ),
                    ],
                    grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
                    unit="s",
                    options={"colorMode": "value", "graphMode": "area"},
                ),
            ],
        )
    
    def _generate_operations_row(self) -> DashboardRow:
        """Generate operations metrics row."""
        return DashboardRow(
            title="Operations",
            panels=[
                # Operations over time
                DashboardPanel(
                    title="Operations Over Time",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum by (operation) (rate({self.namespace}_operations_total[5m]))",
                            legend_format="{{operation}}",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
                    unit="ops",
                ),
                # Operation latency histogram
                DashboardPanel(
                    title="Operation Latency Distribution",
                    panel_type=PanelType.HEATMAP,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum by (le) (rate({self.namespace}_operation_duration_seconds_bucket[5m]))",
                            legend_format="{{le}}",
                        ),
                    ],
                    grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
                    unit="s",
                ),
                # Errors over time
                DashboardPanel(
                    title="Errors Over Time",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum by (operation, error_type) (rate({self.namespace}_operation_errors_total[5m]))",
                            legend_format="{{operation}}: {{error_type}}",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
                    unit="ops",
                ),
                # P95 latency by operation
                DashboardPanel(
                    title="P95 Latency by Operation",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"histogram_quantile(0.95, sum by (operation, le) (rate({self.namespace}_operation_duration_seconds_bucket[5m])))",
                            legend_format="{{operation}}",
                        ),
                    ],
                    grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
                    unit="s",
                ),
            ],
        )
    
    def _generate_search_row(self) -> DashboardRow:
        """Generate search metrics row."""
        return DashboardRow(
            title="Search",
            panels=[
                # Searches by method
                DashboardPanel(
                    title="Searches by Method",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum by (method) (rate({self.namespace}_searches_total[5m]))",
                            legend_format="{{method}}",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 0, "w": 8, "h": 8},
                    unit="ops",
                ),
                # Search latency
                DashboardPanel(
                    title="Search Latency (P50/P95/P99)",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"histogram_quantile(0.50, sum by (le) (rate({self.namespace}_search_duration_seconds_bucket[5m])))",
                            legend_format="P50",
                            ref_id="A",
                        ),
                        PrometheusQuery(
                            expr=f"histogram_quantile(0.95, sum by (le) (rate({self.namespace}_search_duration_seconds_bucket[5m])))",
                            legend_format="P95",
                            ref_id="B",
                        ),
                        PrometheusQuery(
                            expr=f"histogram_quantile(0.99, sum by (le) (rate({self.namespace}_search_duration_seconds_bucket[5m])))",
                            legend_format="P99",
                            ref_id="C",
                        ),
                    ],
                    grid_pos={"x": 8, "y": 0, "w": 8, "h": 8},
                    unit="s",
                ),
                # Results count distribution
                DashboardPanel(
                    title="Results per Search",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum(rate({self.namespace}_search_results_count_sum[5m])) / sum(rate({self.namespace}_search_results_count_count[5m]))",
                            legend_format="Avg results",
                        ),
                    ],
                    grid_pos={"x": 16, "y": 0, "w": 8, "h": 8},
                    unit="short",
                ),
                # Embedding duration
                DashboardPanel(
                    title="Embedding Generation Time",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"histogram_quantile(0.95, sum by (model, le) (rate({self.namespace}_embedding_duration_seconds_bucket[5m])))",
                            legend_format="{{model}}",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
                    unit="s",
                ),
                # Reranking duration
                DashboardPanel(
                    title="Reranking Time",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"histogram_quantile(0.95, sum by (le) (rate({self.namespace}_rerank_duration_seconds_bucket[5m])))",
                            legend_format="P95",
                        ),
                    ],
                    grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
                    unit="s",
                ),
            ],
        )
    
    def _generate_storage_row(self) -> DashboardRow:
        """Generate storage metrics row."""
        return DashboardRow(
            title="Storage",
            panels=[
                # Memory count over time
                DashboardPanel(
                    title="Memory Count",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_memories_total",
                            legend_format="Total",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 0, "w": 8, "h": 8},
                    unit="short",
                ),
                # Memories by domain
                DashboardPanel(
                    title="Memories by Domain",
                    panel_type=PanelType.PIE_CHART,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_memories_by_domain",
                            legend_format="{{domain}}",
                        ),
                    ],
                    grid_pos={"x": 8, "y": 0, "w": 8, "h": 8},
                    options={"pieType": "pie", "displayLabels": ["name", "percent"]},
                ),
                # Storage size by component
                DashboardPanel(
                    title="Storage by Component",
                    panel_type=PanelType.BAR_GAUGE,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_storage_bytes",
                            legend_format="{{component}}",
                        ),
                    ],
                    grid_pos={"x": 16, "y": 0, "w": 8, "h": 8},
                    unit="bytes",
                ),
                # Branch count
                DashboardPanel(
                    title="Branches",
                    panel_type=PanelType.STAT,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_branch_count",
                            legend_format="Branches",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 8, "w": 6, "h": 4},
                    unit="short",
                ),
                # Commits by branch
                DashboardPanel(
                    title="Commits by Branch",
                    panel_type=PanelType.TABLE,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_commit_count",
                            legend_format="{{branch}}",
                            instant=True,
                        ),
                    ],
                    grid_pos={"x": 6, "y": 8, "w": 9, "h": 4},
                    unit="short",
                ),
                # Fragmentation ratio
                DashboardPanel(
                    title="Fragmentation",
                    panel_type=PanelType.GAUGE,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_fragmentation_ratio * 100",
                            legend_format="Fragmentation",
                        ),
                    ],
                    grid_pos={"x": 15, "y": 8, "w": 9, "h": 4},
                    unit="percent",
                    thresholds=[
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 20},
                        {"color": "red", "value": 50},
                    ],
                ),
            ],
        )
    
    def _generate_cache_row(self) -> DashboardRow:
        """Generate cache metrics row."""
        return DashboardRow(
            title="Cache",
            panels=[
                # Cache hit/miss rate
                DashboardPanel(
                    title="Cache Hit/Miss Rate",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"sum by (cache_name) (rate({self.namespace}_cache_hits_total[5m]))",
                            legend_format="{{cache_name}} hits",
                            ref_id="A",
                        ),
                        PrometheusQuery(
                            expr=f"sum by (cache_name) (rate({self.namespace}_cache_misses_total[5m]))",
                            legend_format="{{cache_name}} misses",
                            ref_id="B",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
                    unit="ops",
                ),
                # Cache hit ratio over time
                DashboardPanel(
                    title="Cache Hit Ratio",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_cache_hit_ratio * 100",
                            legend_format="{{cache_name}}",
                        ),
                    ],
                    grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
                    unit="percent",
                ),
                # Cache size
                DashboardPanel(
                    title="Cache Size",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"{self.namespace}_cache_size",
                            legend_format="{{cache_name}}",
                        ),
                    ],
                    grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
                    unit="short",
                ),
                # Cache evictions
                DashboardPanel(
                    title="Cache Evictions",
                    panel_type=PanelType.TIMESERIES,
                    queries=[
                        PrometheusQuery(
                            expr=f"rate({self.namespace}_cache_evictions_total[5m])",
                            legend_format="{{cache_name}}",
                        ),
                    ],
                    grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
                    unit="ops",
                ),
            ],
        )
    
    def generate_alert_rules(self) -> List[AlertRule]:
        """Generate standard alert rules for agent memory monitoring."""
        return [
            AlertRule(
                name="HighErrorRate",
                expr=f"sum(rate({self.namespace}_operation_errors_total[5m])) / sum(rate({self.namespace}_operations_total[5m])) > 0.05",
                for_duration="5m",
                severity="warning",
                annotations={
                    "summary": "High error rate detected",
                    "description": "Error rate is above 5% for the last 5 minutes",
                },
            ),
            AlertRule(
                name="HighOperationLatency",
                expr=f"histogram_quantile(0.95, sum by (le) (rate({self.namespace}_operation_duration_seconds_bucket[5m]))) > 1",
                for_duration="5m",
                severity="warning",
                annotations={
                    "summary": "High operation latency",
                    "description": "P95 operation latency is above 1 second",
                },
            ),
            AlertRule(
                name="LowCacheHitRatio",
                expr=f"avg({self.namespace}_cache_hit_ratio) < 0.5",
                for_duration="10m",
                severity="warning",
                annotations={
                    "summary": "Low cache hit ratio",
                    "description": "Cache hit ratio is below 50%",
                },
            ),
            AlertRule(
                name="HighStorageFragmentation",
                expr=f"{self.namespace}_fragmentation_ratio > 0.5",
                for_duration="30m",
                severity="warning",
                annotations={
                    "summary": "High storage fragmentation",
                    "description": "Storage fragmentation is above 50%",
                },
            ),
            AlertRule(
                name="NoRecentOperations",
                expr=f"sum(increase({self.namespace}_operations_total[5m])) == 0",
                for_duration="15m",
                severity="info",
                annotations={
                    "summary": "No operations detected",
                    "description": "No memory operations in the last 15 minutes",
                },
            ),
            AlertRule(
                name="SearchLatencySpike",
                expr=f"histogram_quantile(0.99, sum by (le) (rate({self.namespace}_search_duration_seconds_bucket[5m]))) > 5",
                for_duration="5m",
                severity="critical",
                annotations={
                    "summary": "Search latency spike",
                    "description": "P99 search latency is above 5 seconds",
                },
            ),
        ]
    
    def export_prometheus_rules(self, output_path: str) -> None:
        """Export alert rules as Prometheus rules YAML."""
        rules = self.generate_alert_rules()
        
        rules_yaml = {
            "groups": [
                {
                    "name": "agent_memory_toolkit",
                    "rules": [rule.to_prometheus_rule() for rule in rules],
                }
            ]
        }
        
        # Convert to YAML format manually (to avoid dependency)
        lines = ["groups:"]
        for group in rules_yaml["groups"]:
            lines.append(f"  - name: {group['name']}")
            lines.append("    rules:")
            for rule in group["rules"]:
                lines.append(f"      - alert: {rule['alert']}")
                lines.append(f"        expr: {rule['expr']}")
                lines.append(f"        for: {rule['for']}")
                lines.append("        labels:")
                for k, v in rule["labels"].items():
                    lines.append(f"          {k}: {v}")
                lines.append("        annotations:")
                for k, v in rule["annotations"].items():
                    lines.append(f"          {k}: {v}")
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
    
    def export_dashboard(self, output_path: str) -> None:
        """Export dashboard to a JSON file."""
        dashboard = self.generate()
        with open(output_path, "w") as f:
            f.write(dashboard.to_json())
