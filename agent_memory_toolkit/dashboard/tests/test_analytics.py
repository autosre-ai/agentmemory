"""
Tests for the analytics engine.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_memory_toolkit.dashboard.analytics import (
    AnalyticsEngine,
    MemoryStats,
    DomainDistribution,
    SearchTrends,
    StorageMetrics,
    BranchComparison,
    BranchStats,
    TimeSeriesData,
)


@pytest.fixture
def temp_db():
    """Create a temporary database with test data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    
    # Create tables
    conn.executescript("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT,
            embedding BLOB,
            branch TEXT DEFAULT 'main',
            version INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT,
            deleted_at TEXT
        );
        
        CREATE TABLE branches (
            name TEXT PRIMARY KEY,
            created_at TEXT,
            parent_branch TEXT
        );
        
        CREATE TABLE commits (
            id TEXT PRIMARY KEY,
            branch TEXT,
            message TEXT,
            created_at TEXT
        );
        
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        
        -- Add test data
        INSERT INTO branches (name, created_at) VALUES 
            ('main', '2024-01-01T00:00:00'),
            ('feature', '2024-01-15T00:00:00');
        
        INSERT INTO settings (key, value) VALUES ('current_branch', 'main');
        
        INSERT INTO memories (id, content, metadata, branch, created_at, updated_at) VALUES
            ('mem1', 'The capital of France is Paris', '{"domain": "factual"}', 'main', 
             datetime('now', '-5 days'), datetime('now', '-5 days')),
            ('mem2', 'Python is a programming language', '{"domain": "technical"}', 'main',
             datetime('now', '-3 days'), datetime('now', '-3 days')),
            ('mem3', 'The user prefers dark mode', '{"domain": "preferences"}', 'main',
             datetime('now', '-1 day'), datetime('now', '-1 day')),
            ('mem4', 'Meeting scheduled for Monday', '{"domain": "calendar"}', 'feature',
             datetime('now'), datetime('now'));
        
        INSERT INTO commits (id, branch, message, created_at) VALUES
            ('c1', 'main', 'Initial commit', datetime('now', '-7 days')),
            ('c2', 'main', 'Add memory', datetime('now', '-5 days')),
            ('c3', 'feature', 'Feature commit', datetime('now'));
    """)
    
    conn.close()
    
    yield db_path
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def search_log():
    """Create a temporary search log file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=".jsonl", delete=False) as f:
        # Write some search logs
        logs = [
            {"query": "paris", "timestamp": (datetime.now() - timedelta(days=1)).isoformat(), 
             "result_count": 1, "duration_ms": 5.0},
            {"query": "python", "timestamp": datetime.now().isoformat(), 
             "result_count": 2, "duration_ms": 3.0},
            {"query": "paris", "timestamp": datetime.now().isoformat(), 
             "result_count": 1, "duration_ms": 4.0},
        ]
        for log in logs:
            f.write(json.dumps(log) + "\n")
        log_path = f.name
    
    yield log_path
    
    os.unlink(log_path)


class TestTimeSeriesData:
    """Tests for TimeSeriesData."""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        ts = TimeSeriesData(
            timestamp=datetime(2024, 1, 15, 12, 0),
            value=42.5,
            label="test"
        )
        
        result = ts.to_dict()
        
        assert result["timestamp"] == "2024-01-15T12:00:00"
        assert result["value"] == 42.5
        assert result["label"] == "test"
    
    def test_to_dict_no_label(self):
        """Test converting to dictionary without label."""
        ts = TimeSeriesData(
            timestamp=datetime(2024, 1, 15),
            value=10.0
        )
        
        result = ts.to_dict()
        
        assert result["label"] is None


class TestMemoryStats:
    """Tests for MemoryStats."""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = MemoryStats(
            total_memories=100,
            total_branches=3,
            total_commits=50,
            active_memories=95,
            deleted_memories=5,
            avg_memory_size=256.5,
            total_storage_bytes=1024000,
            oldest_memory=datetime(2024, 1, 1),
            newest_memory=datetime(2024, 1, 15),
            memories_by_day=[
                TimeSeriesData(datetime(2024, 1, 14), 10),
                TimeSeriesData(datetime(2024, 1, 15), 5),
            ]
        )
        
        result = stats.to_dict()
        
        assert result["total_memories"] == 100
        assert result["active_memories"] == 95
        assert result["oldest_memory"] == "2024-01-01T00:00:00"
        assert len(result["memories_by_day"]) == 2


class TestDomainDistribution:
    """Tests for DomainDistribution."""
    
    def test_percentages_calculated(self):
        """Test that percentages are calculated correctly."""
        dist = DomainDistribution(
            domain_counts={"factual": 50, "technical": 30, "preferences": 20},
            total=100
        )
        
        assert dist.percentages["factual"] == 50.0
        assert dist.percentages["technical"] == 30.0
        assert dist.percentages["preferences"] == 20.0
    
    def test_percentages_zero_total(self):
        """Test percentages with zero total."""
        dist = DomainDistribution(
            domain_counts={},
            total=0
        )
        
        assert dist.percentages == {}
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        dist = DomainDistribution(
            domain_counts={"factual": 10},
            total=10
        )
        
        result = dist.to_dict()
        
        assert result["domain_counts"]["factual"] == 10
        assert result["total"] == 10
        assert result["percentages"]["factual"] == 100.0


class TestAnalyticsEngine:
    """Tests for AnalyticsEngine."""
    
    def test_init(self, temp_db):
        """Test engine initialization."""
        engine = AnalyticsEngine(db_path=temp_db)
        
        assert engine.db_path == temp_db
        assert engine.search_log_path is not None
    
    def test_init_with_search_log(self, temp_db, search_log):
        """Test engine initialization with custom search log."""
        engine = AnalyticsEngine(db_path=temp_db, search_log_path=search_log)
        
        assert engine.search_log_path == search_log
    
    def test_get_memory_stats(self, temp_db):
        """Test getting memory statistics."""
        engine = AnalyticsEngine(db_path=temp_db)
        
        stats = engine.get_memory_stats()
        
        assert stats.total_memories == 4  # 4 total memories
        assert stats.active_memories == 4
        assert stats.deleted_memories == 0
        assert stats.total_branches == 2
        assert stats.total_commits == 3
    
    def test_get_memory_stats_with_days(self, temp_db):
        """Test getting memory statistics with custom days."""
        engine = AnalyticsEngine(db_path=temp_db)
        
        stats = engine.get_memory_stats(days=7)
        
        assert isinstance(stats.memories_by_day, list)
    
    def test_get_domain_distribution(self, temp_db):
        """Test getting domain distribution."""
        engine = AnalyticsEngine(db_path=temp_db)
        
        dist = engine.get_domain_distribution()
        
        assert dist.total == 4
        assert "factual" in dist.domain_counts
        assert "technical" in dist.domain_counts
    
    def test_get_search_trends(self, temp_db, search_log):
        """Test getting search trends."""
        engine = AnalyticsEngine(db_path=temp_db, search_log_path=search_log)
        
        trends = engine.get_search_trends()
        
        assert trends.total_searches == 3
        assert trends.searches_today >= 2  # At least 2 searches today
        assert len(trends.top_queries) > 0
        assert trends.top_queries[0][0] == "paris"  # Most common query
    
    def test_get_search_trends_empty(self, temp_db):
        """Test getting search trends with empty log."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            empty_log = f.name
        
        try:
            engine = AnalyticsEngine(db_path=temp_db, search_log_path=empty_log)
            trends = engine.get_search_trends()
            
            assert trends.total_searches == 0
            assert trends.top_queries == []
        finally:
            os.unlink(empty_log)
    
    def test_get_storage_metrics(self, temp_db):
        """Test getting storage metrics."""
        engine = AnalyticsEngine(db_path=temp_db)
        
        metrics = engine.get_storage_metrics()
        
        assert metrics.database_size_bytes > 0
        assert isinstance(metrics.size_by_day, list)
    
    def test_get_branch_comparison(self, temp_db):
        """Test getting branch comparison."""
        engine = AnalyticsEngine(db_path=temp_db)
        
        comparison = engine.get_branch_comparison()
        
        assert len(comparison.branches) == 2
        
        # Find main branch
        main_branch = next(b for b in comparison.branches if b.name == "main")
        assert main_branch.is_current is True
        assert main_branch.memory_count == 3  # 3 memories on main
        
        # Find feature branch
        feature_branch = next(b for b in comparison.branches if b.name == "feature")
        assert feature_branch.is_current is False
        assert feature_branch.memory_count == 1
    
    def test_get_all_analytics(self, temp_db, search_log):
        """Test getting all analytics data."""
        engine = AnalyticsEngine(db_path=temp_db, search_log_path=search_log)
        
        data = engine.get_all_analytics()
        
        assert "memory_stats" in data
        assert "domain_distribution" in data
        assert "search_trends" in data
        assert "storage_metrics" in data
        assert "branch_comparison" in data
        assert "generated_at" in data
    
    def test_log_search(self, temp_db):
        """Test logging a search query."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name
        
        try:
            engine = AnalyticsEngine(db_path=temp_db, search_log_path=log_path)
            
            engine.log_search("test query", result_count=5, duration_ms=10.0)
            
            # Read the log file
            with open(log_path) as f:
                line = f.readline()
                entry = json.loads(line)
            
            assert entry["query"] == "test query"
            assert entry["result_count"] == 5
            assert entry["duration_ms"] == 10.0
        finally:
            os.unlink(log_path)


class TestBranchStats:
    """Tests for BranchStats."""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = BranchStats(
            name="main",
            memory_count=100,
            commit_count=50,
            created_at=datetime(2024, 1, 1),
            last_commit=datetime(2024, 1, 15),
            is_current=True
        )
        
        result = stats.to_dict()
        
        assert result["name"] == "main"
        assert result["memory_count"] == 100
        assert result["is_current"] is True
        assert result["created_at"] == "2024-01-01T00:00:00"
    
    def test_to_dict_no_last_commit(self):
        """Test converting to dictionary without last commit."""
        stats = BranchStats(
            name="new-branch",
            memory_count=0,
            commit_count=0,
            created_at=datetime(2024, 1, 15),
            last_commit=None,
            is_current=False
        )
        
        result = stats.to_dict()
        
        assert result["last_commit"] is None


class TestSearchTrends:
    """Tests for SearchTrends."""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        trends = SearchTrends(
            total_searches=100,
            searches_today=10,
            searches_this_week=50,
            top_queries=[("python", 20), ("memory", 15)],
            searches_by_day=[TimeSeriesData(datetime(2024, 1, 15), 10)],
            avg_results_per_search=5.5,
            avg_search_time_ms=3.2
        )
        
        result = trends.to_dict()
        
        assert result["total_searches"] == 100
        assert result["top_queries"][0]["query"] == "python"
        assert result["top_queries"][0]["count"] == 20


class TestStorageMetrics:
    """Tests for StorageMetrics."""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        metrics = StorageMetrics(
            database_size_bytes=1024000,
            fts_index_size_bytes=51200,
            embeddings_size_bytes=204800,
            metadata_size_bytes=10240,
            total_size_bytes=1024000,
            size_by_day=[],
            compression_ratio=1.5
        )
        
        result = metrics.to_dict()
        
        assert result["database_size_bytes"] == 1024000
        assert result["compression_ratio"] == 1.5


class TestBranchComparison:
    """Tests for BranchComparison."""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        comparison = BranchComparison(
            branches=[
                BranchStats("main", 100, 50, datetime(2024, 1, 1), datetime(2024, 1, 15), True)
            ],
            total_unique_memories=100,
            shared_memories=0,
            divergence_points={}
        )
        
        result = comparison.to_dict()
        
        assert len(result["branches"]) == 1
        assert result["total_unique_memories"] == 100
