"""
Analytics engine for the memory dashboard.

Provides statistics and metrics about memory usage, search patterns,
storage growth, and branch comparisons.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum


@dataclass
class TimeSeriesData:
    """Time series data point."""
    timestamp: datetime
    value: float
    label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "label": self.label,
        }


@dataclass
class MemoryStats:
    """Overall memory statistics."""
    total_memories: int
    total_branches: int
    total_commits: int
    active_memories: int
    deleted_memories: int
    avg_memory_size: float
    total_storage_bytes: int
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]
    memories_by_day: List[TimeSeriesData] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_memories": self.total_memories,
            "total_branches": self.total_branches,
            "total_commits": self.total_commits,
            "active_memories": self.active_memories,
            "deleted_memories": self.deleted_memories,
            "avg_memory_size": self.avg_memory_size,
            "total_storage_bytes": self.total_storage_bytes,
            "oldest_memory": self.oldest_memory.isoformat() if self.oldest_memory else None,
            "newest_memory": self.newest_memory.isoformat() if self.newest_memory else None,
            "memories_by_day": [d.to_dict() for d in self.memories_by_day],
        }


@dataclass
class DomainDistribution:
    """Distribution of memories across cognitive domains."""
    domain_counts: Dict[str, int]
    total: int
    percentages: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.total > 0:
            self.percentages = {
                domain: (count / self.total) * 100
                for domain, count in self.domain_counts.items()
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_counts": self.domain_counts,
            "total": self.total,
            "percentages": self.percentages,
        }


@dataclass
class SearchQuery:
    """A logged search query."""
    query: str
    timestamp: datetime
    result_count: int
    duration_ms: float


@dataclass
class SearchTrends:
    """Search query trends and statistics."""
    total_searches: int
    searches_today: int
    searches_this_week: int
    top_queries: List[Tuple[str, int]]  # (query, count)
    searches_by_day: List[TimeSeriesData]
    avg_results_per_search: float
    avg_search_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_searches": self.total_searches,
            "searches_today": self.searches_today,
            "searches_this_week": self.searches_this_week,
            "top_queries": [{"query": q, "count": c} for q, c in self.top_queries],
            "searches_by_day": [d.to_dict() for d in self.searches_by_day],
            "avg_results_per_search": self.avg_results_per_search,
            "avg_search_time_ms": self.avg_search_time_ms,
        }


@dataclass
class StorageMetrics:
    """Storage usage metrics."""
    database_size_bytes: int
    fts_index_size_bytes: int
    embeddings_size_bytes: int
    metadata_size_bytes: int
    total_size_bytes: int
    size_by_day: List[TimeSeriesData]
    compression_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_size_bytes": self.database_size_bytes,
            "fts_index_size_bytes": self.fts_index_size_bytes,
            "embeddings_size_bytes": self.embeddings_size_bytes,
            "metadata_size_bytes": self.metadata_size_bytes,
            "total_size_bytes": self.total_size_bytes,
            "size_by_day": [d.to_dict() for d in self.size_by_day],
            "compression_ratio": self.compression_ratio,
        }


@dataclass
class BranchStats:
    """Statistics for a single branch."""
    name: str
    memory_count: int
    commit_count: int
    created_at: datetime
    last_commit: Optional[datetime]
    is_current: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "memory_count": self.memory_count,
            "commit_count": self.commit_count,
            "created_at": self.created_at.isoformat(),
            "last_commit": self.last_commit.isoformat() if self.last_commit else None,
            "is_current": self.is_current,
        }


@dataclass
class BranchComparison:
    """Comparison data across branches."""
    branches: List[BranchStats]
    total_unique_memories: int
    shared_memories: int
    divergence_points: Dict[str, Dict[str, int]]  # branch -> branch -> divergent count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "branches": [b.to_dict() for b in self.branches],
            "total_unique_memories": self.total_unique_memories,
            "shared_memories": self.shared_memories,
            "divergence_points": self.divergence_points,
        }


class AnalyticsEngine:
    """
    Analytics engine for computing memory dashboard statistics.
    
    Collects and computes metrics from memory stores including:
    - Memory counts and trends over time
    - Domain distribution
    - Search query trends
    - Storage usage
    - Branch comparisons
    """
    
    def __init__(self, db_path: str, search_log_path: Optional[str] = None):
        """
        Initialize the analytics engine.
        
        Args:
            db_path: Path to the memory SQLite database
            search_log_path: Optional path to search log file (JSONL format)
        """
        self.db_path = db_path
        self.search_log_path = search_log_path or self._default_search_log_path()
        self._ensure_search_log()
    
    def _default_search_log_path(self) -> str:
        """Get default search log path based on database path."""
        db_dir = Path(self.db_path).parent
        return str(db_dir / "search_log.jsonl")
    
    def _ensure_search_log(self):
        """Ensure search log file exists."""
        log_path = Path(self.search_log_path)
        if not log_path.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def log_search(self, query: str, result_count: int, duration_ms: float):
        """
        Log a search query for trend analysis.
        
        Args:
            query: The search query string
            result_count: Number of results returned
            duration_ms: Search duration in milliseconds
        """
        log_entry = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "result_count": result_count,
            "duration_ms": duration_ms,
        }
        with open(self.search_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _get_schema_info(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Get schema information for the memories table."""
        cursor = conn.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]
        
        return {
            'has_deleted_at': 'deleted_at' in columns,
            'has_is_deleted': 'is_deleted' in columns,
            'has_metadata': 'metadata' in columns,
            'has_metadata_json': 'metadata_json' in columns,
            'metadata_col': 'metadata' if 'metadata' in columns else ('metadata_json' if 'metadata_json' in columns else None),
        }
    
    def _get_deleted_filter(self, schema: Dict[str, Any], negate: bool = False) -> str:
        """Get the SQL fragment for filtering deleted memories."""
        if schema['has_deleted_at']:
            return "deleted_at IS NOT NULL" if negate else "deleted_at IS NULL"
        elif schema['has_is_deleted']:
            return "is_deleted = 1" if negate else "is_deleted = 0"
        else:
            return "1=0" if negate else "1=1"  # No filter if no deleted column

    def get_memory_stats(self, days: int = 30) -> MemoryStats:
        """
        Get overall memory statistics.
        
        Args:
            days: Number of days for time series data
            
        Returns:
            MemoryStats with current statistics
        """
        conn = self._get_connection()
        try:
            schema = self._get_schema_info(conn)
            active_filter = self._get_deleted_filter(schema, negate=False)
            deleted_filter = self._get_deleted_filter(schema, negate=True)
            
            # Total counts
            total = conn.execute(
                f"SELECT COUNT(*) FROM memories WHERE {active_filter}"
            ).fetchone()[0]
            
            if schema['has_deleted_at'] or schema['has_is_deleted']:
                deleted = conn.execute(
                    f"SELECT COUNT(*) FROM memories WHERE {deleted_filter}"
                ).fetchone()[0]
            else:
                deleted = 0
            
            # Check if branches/commits tables exist
            try:
                branches = conn.execute(
                    "SELECT COUNT(*) FROM branches"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                branches = 0
            
            try:
                commits = conn.execute(
                    "SELECT COUNT(*) FROM commits"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                commits = 0
            
            # Average memory size
            avg_size_result = conn.execute(
                f"SELECT AVG(LENGTH(content)) FROM memories WHERE {active_filter}"
            ).fetchone()[0]
            avg_size = avg_size_result or 0.0
            
            # Date range
            date_range = conn.execute(
                f"""SELECT MIN(created_at), MAX(created_at) 
                   FROM memories WHERE {active_filter}"""
            ).fetchone()
            
            oldest = None
            newest = None
            if date_range[0]:
                try:
                    oldest = datetime.fromisoformat(date_range[0].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    pass
            if date_range[1]:
                try:
                    newest = datetime.fromisoformat(date_range[1].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    pass
            
            # Memories by day
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            daily_counts = conn.execute(
                f"""SELECT DATE(created_at) as day, COUNT(*) as count
                   FROM memories 
                   WHERE {active_filter} AND created_at >= ?
                   GROUP BY DATE(created_at)
                   ORDER BY day""",
                (cutoff,)
            ).fetchall()
            
            memories_by_day = [
                TimeSeriesData(
                    timestamp=datetime.fromisoformat(row["day"]) if row["day"] else datetime.now(),
                    value=row["count"],
                    label="memories"
                )
                for row in daily_counts
            ]
            
            # Storage size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return MemoryStats(
                total_memories=total + deleted,
                total_branches=branches,
                total_commits=commits,
                active_memories=total,
                deleted_memories=deleted,
                avg_memory_size=avg_size,
                total_storage_bytes=db_size,
                oldest_memory=oldest,
                newest_memory=newest,
                memories_by_day=memories_by_day,
            )
        finally:
            conn.close()
    
    def get_domain_distribution(self) -> DomainDistribution:
        """
        Get distribution of memories across cognitive domains.
        
        Extracts domain from memory metadata if available.
        
        Returns:
            DomainDistribution with domain counts
        """
        conn = self._get_connection()
        try:
            schema = self._get_schema_info(conn)
            active_filter = self._get_deleted_filter(schema, negate=False)
            metadata_col = schema['metadata_col']
            
            # Try to extract domain from metadata JSON
            domain_counts: Dict[str, int] = defaultdict(int)
            
            if metadata_col:
                rows = conn.execute(
                    f"SELECT {metadata_col} as metadata FROM memories WHERE {active_filter}"
                ).fetchall()
            else:
                # No metadata column, count all as unknown
                total_count = conn.execute(
                    f"SELECT COUNT(*) FROM memories WHERE {active_filter}"
                ).fetchone()[0]
                return DomainDistribution(
                    domain_counts={"unknown": total_count},
                    total=total_count,
                )
            
            for row in rows:
                metadata = row["metadata"]
                if metadata:
                    try:
                        meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                        domain = meta_dict.get("domain", "unknown")
                        domain_counts[domain] += 1
                    except (json.JSONDecodeError, TypeError):
                        domain_counts["unknown"] += 1
                else:
                    domain_counts["unknown"] += 1
            
            total = sum(domain_counts.values())
            
            # Sort by count descending
            sorted_domains = dict(sorted(
                domain_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            return DomainDistribution(
                domain_counts=sorted_domains,
                total=total,
            )
        finally:
            conn.close()
    
    def get_search_trends(self, days: int = 30) -> SearchTrends:
        """
        Get search query trends.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            SearchTrends with query statistics
        """
        queries: List[SearchQuery] = []
        
        # Read search log
        if os.path.exists(self.search_log_path):
            with open(self.search_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            queries.append(SearchQuery(
                                query=entry["query"],
                                timestamp=datetime.fromisoformat(entry["timestamp"]),
                                result_count=entry.get("result_count", 0),
                                duration_ms=entry.get("duration_ms", 0.0),
                            ))
                        except (json.JSONDecodeError, KeyError):
                            continue
        
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = today - timedelta(days=7)
        cutoff = now - timedelta(days=days)
        
        # Filter recent queries
        recent_queries = [q for q in queries if q.timestamp >= cutoff]
        
        # Today's searches
        searches_today = len([q for q in queries if q.timestamp >= today])
        
        # This week's searches
        searches_this_week = len([q for q in queries if q.timestamp >= week_ago])
        
        # Top queries
        query_counts: Dict[str, int] = defaultdict(int)
        for q in recent_queries:
            # Normalize query
            normalized = q.query.lower().strip()
            query_counts[normalized] += 1
        
        top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Searches by day
        daily_counts: Dict[str, int] = defaultdict(int)
        for q in recent_queries:
            day = q.timestamp.strftime("%Y-%m-%d")
            daily_counts[day] += 1
        
        searches_by_day = [
            TimeSeriesData(
                timestamp=datetime.fromisoformat(day),
                value=count,
                label="searches"
            )
            for day, count in sorted(daily_counts.items())
        ]
        
        # Averages
        avg_results = 0.0
        avg_time = 0.0
        if recent_queries:
            avg_results = sum(q.result_count for q in recent_queries) / len(recent_queries)
            avg_time = sum(q.duration_ms for q in recent_queries) / len(recent_queries)
        
        return SearchTrends(
            total_searches=len(queries),
            searches_today=searches_today,
            searches_this_week=searches_this_week,
            top_queries=top_queries,
            searches_by_day=searches_by_day,
            avg_results_per_search=avg_results,
            avg_search_time_ms=avg_time,
        )
    
    def get_storage_metrics(self, days: int = 30) -> StorageMetrics:
        """
        Get storage usage metrics.
        
        Args:
            days: Number of days for trend data
            
        Returns:
            StorageMetrics with storage statistics
        """
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        conn = self._get_connection()
        try:
            # Estimate component sizes
            # FTS index size (approximate from page count)
            try:
                fts_pages = conn.execute(
                    "SELECT COUNT(*) FROM memories_fts_data"
                ).fetchone()[0]
                fts_size = fts_pages * 4096  # Approximate page size
            except sqlite3.OperationalError:
                fts_size = 0
            
            # Content size
            content_size_result = conn.execute(
                "SELECT SUM(LENGTH(content)) FROM memories"
            ).fetchone()[0]
            content_size = content_size_result or 0
            
            # Embeddings size (approximate)
            embedding_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            # Typical embedding is ~3KB (768 floats * 4 bytes)
            embeddings_size = embedding_count * 3072
            
            # Metadata size
            metadata_size_result = conn.execute(
                "SELECT SUM(LENGTH(metadata)) FROM memories"
            ).fetchone()[0]
            metadata_size = metadata_size_result or 0
            
            # Compression ratio (content vs total)
            compression_ratio = 1.0
            if content_size > 0:
                compression_ratio = db_size / content_size
            
            # Size tracking over time (mock - would need actual tracking)
            size_by_day: List[TimeSeriesData] = []
            
            return StorageMetrics(
                database_size_bytes=db_size,
                fts_index_size_bytes=fts_size,
                embeddings_size_bytes=embeddings_size,
                metadata_size_bytes=metadata_size,
                total_size_bytes=db_size,
                size_by_day=size_by_day,
                compression_ratio=compression_ratio,
            )
        finally:
            conn.close()
    
    def get_branch_comparison(self) -> BranchComparison:
        """
        Get comparison data across branches.
        
        Returns:
            BranchComparison with branch statistics
        """
        conn = self._get_connection()
        try:
            schema = self._get_schema_info(conn)
            
            # Get current branch - check if is_current column exists
            schema_cols = schema.get('columns', {}).get('branches', [])
            if 'is_current' in schema_cols:
                current_branch_row = conn.execute(
                    "SELECT name FROM branches WHERE is_current = 1"
                ).fetchone()
            else:
                # Fall back to first branch or 'main'
                current_branch_row = conn.execute(
                    "SELECT name FROM branches WHERE name = 'main'"
                ).fetchone()
            current_branch_name = current_branch_row[0] if current_branch_row else "main"
            
            branches_rows = conn.execute(
                "SELECT name, created_at FROM branches"
            ).fetchall()
            
            branch_stats: List[BranchStats] = []
            has_deleted_at = 'deleted_at' in schema.get('columns', {}).get('memories', [])
            
            for row in branches_rows:
                branch_name = row["name"]
                created_at_str = row["created_at"]
                
                try:
                    created_at = datetime.fromisoformat(
                        created_at_str.replace('Z', '+00:00') if created_at_str else datetime.now().isoformat()
                    )
                except (ValueError, AttributeError):
                    created_at = datetime.now()
                
                # Count memories for this branch
                if has_deleted_at:
                    memory_count = conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE branch = ? AND deleted_at IS NULL",
                        (branch_name,)
                    ).fetchone()[0]
                else:
                    memory_count = conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE branch = ?",
                        (branch_name,)
                    ).fetchone()[0]
                
                # Count commits
                commit_count = conn.execute(
                    "SELECT COUNT(*) FROM commits WHERE branch = ?",
                    (branch_name,)
                ).fetchone()[0]
                
                # Last commit
                last_commit_row = conn.execute(
                    """SELECT created_at FROM commits 
                       WHERE branch = ? 
                       ORDER BY created_at DESC LIMIT 1""",
                    (branch_name,)
                ).fetchone()
                
                last_commit = None
                if last_commit_row:
                    try:
                        last_commit = datetime.fromisoformat(
                            last_commit_row[0].replace('Z', '+00:00')
                        )
                    except (ValueError, AttributeError):
                        pass
                
                branch_stats.append(BranchStats(
                    name=branch_name,
                    memory_count=memory_count,
                    commit_count=commit_count,
                    created_at=created_at,
                    last_commit=last_commit,
                    is_current=branch_name == current_branch_name,
                ))
            
            # Total unique memories across all branches
            if has_deleted_at:
                total_unique = conn.execute(
                    "SELECT COUNT(DISTINCT id) FROM memories WHERE deleted_at IS NULL"
                ).fetchone()[0]
            else:
                total_unique = conn.execute(
                    "SELECT COUNT(DISTINCT id) FROM memories"
                ).fetchone()[0]
            
            # Shared memories (appear in multiple branches) - simplified
            # In a real implementation, this would compare memory content hashes
            shared = 0
            
            # Divergence points - simplified
            divergence: Dict[str, Dict[str, int]] = {}
            
            return BranchComparison(
                branches=branch_stats,
                total_unique_memories=total_unique,
                shared_memories=shared,
                divergence_points=divergence,
            )
        finally:
            conn.close()
    
    def get_all_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get all analytics data for the dashboard.
        
        Args:
            days: Number of days for time series data
            
        Returns:
            Dictionary with all analytics data
        """
        return {
            "memory_stats": self.get_memory_stats(days).to_dict(),
            "domain_distribution": self.get_domain_distribution().to_dict(),
            "search_trends": self.get_search_trends(days).to_dict(),
            "storage_metrics": self.get_storage_metrics(days).to_dict(),
            "branch_comparison": self.get_branch_comparison().to_dict(),
            "generated_at": datetime.now().isoformat(),
        }
