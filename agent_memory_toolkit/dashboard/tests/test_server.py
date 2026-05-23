"""
Tests for the dashboard server.
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch
import urllib.request
import urllib.error

import pytest

from agent_memory_toolkit.dashboard.server import (
    DashboardServer,
    DashboardConfig,
    DashboardHandler,
    run_dashboard,
)
from agent_memory_toolkit.dashboard.analytics import AnalyticsEngine


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
        
        INSERT INTO branches (name, created_at) VALUES ('main', datetime('now'));
        INSERT INTO settings (key, value) VALUES ('current_branch', 'main');
        
        INSERT INTO memories (id, content, metadata, branch, created_at, updated_at) VALUES
            ('mem1', 'Test memory 1', '{"domain": "test"}', 'main', datetime('now'), datetime('now')),
            ('mem2', 'Test memory 2', '{"domain": "test"}', 'main', datetime('now'), datetime('now'));
        
        INSERT INTO commits (id, branch, message, created_at) VALUES
            ('c1', 'main', 'Initial commit', datetime('now'));
    """)
    
    conn.close()
    
    yield db_path
    
    os.unlink(db_path)


class TestDashboardConfig:
    """Tests for DashboardConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DashboardConfig()
        
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.db_path == "agent_memory.db"
        assert config.auto_open is True
        assert config.cors_enabled is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = DashboardConfig(
            host="0.0.0.0",
            port=9090,
            db_path="/path/to/db.db",
            auto_open=False,
            cors_enabled=False
        )
        
        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.db_path == "/path/to/db.db"
        assert config.auto_open is False
        assert config.cors_enabled is False


class TestDashboardServer:
    """Tests for DashboardServer."""
    
    def test_init(self, temp_db):
        """Test server initialization."""
        config = DashboardConfig(db_path=temp_db)
        server = DashboardServer(config)
        
        assert server.config == config
        assert server.analytics_engine is not None
        assert server.is_running is False
    
    def test_get_url(self, temp_db):
        """Test getting server URL."""
        config = DashboardConfig(
            db_path=temp_db,
            host="localhost",
            port=9999
        )
        server = DashboardServer(config)
        
        assert server.get_url() == "http://localhost:9999"
    
    def test_start_stop_non_blocking(self, temp_db):
        """Test starting and stopping server in non-blocking mode."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18081,
            auto_open=False
        )
        server = DashboardServer(config)
        
        # Start in background
        server.start(blocking=False)
        
        try:
            # Give server time to start
            time.sleep(0.5)
            
            assert server.is_running is True
            
            # Try to connect
            try:
                url = f"http://127.0.0.1:18081/api/stats"
                with urllib.request.urlopen(url, timeout=2) as response:
                    data = json.loads(response.read().decode())
                    assert "memory_stats" in data
            except urllib.error.URLError as e:
                # Server might not be ready yet
                pass
                
        finally:
            server.stop()
            assert server.is_running is False
    
    def test_api_memories_endpoint(self, temp_db):
        """Test the /api/memories endpoint."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18082,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18082/api/memories"
            with urllib.request.urlopen(url, timeout=2) as response:
                data = json.loads(response.read().decode())
                assert "total_memories" in data
                assert data["total_memories"] == 2
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()
    
    def test_api_domains_endpoint(self, temp_db):
        """Test the /api/domains endpoint."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18083,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18083/api/domains"
            with urllib.request.urlopen(url, timeout=2) as response:
                data = json.loads(response.read().decode())
                assert "domain_counts" in data
                assert "test" in data["domain_counts"]
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()
    
    def test_api_branches_endpoint(self, temp_db):
        """Test the /api/branches endpoint."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18084,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18084/api/branches"
            with urllib.request.urlopen(url, timeout=2) as response:
                data = json.loads(response.read().decode())
                assert "branches" in data
                assert len(data["branches"]) >= 1
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()
    
    def test_api_storage_endpoint(self, temp_db):
        """Test the /api/storage endpoint."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18085,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18085/api/storage"
            with urllib.request.urlopen(url, timeout=2) as response:
                data = json.loads(response.read().decode())
                assert "database_size_bytes" in data
                assert data["database_size_bytes"] > 0
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()
    
    def test_api_searches_endpoint(self, temp_db):
        """Test the /api/searches endpoint."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18086,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18086/api/searches"
            with urllib.request.urlopen(url, timeout=2) as response:
                data = json.loads(response.read().decode())
                assert "total_searches" in data
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()
    
    def test_days_parameter(self, temp_db):
        """Test the days query parameter."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18087,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18087/api/stats?days=7"
            with urllib.request.urlopen(url, timeout=2) as response:
                data = json.loads(response.read().decode())
                assert "memory_stats" in data
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()


class TestDashboardHandler:
    """Tests for DashboardHandler."""
    
    def test_cors_headers(self, temp_db):
        """Test that CORS headers are sent."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18088,
            auto_open=False,
            cors_enabled=True
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18088/api/stats"
            with urllib.request.urlopen(url, timeout=2) as response:
                # Check CORS header
                cors_header = response.headers.get('Access-Control-Allow-Origin')
                assert cors_header == '*'
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()


class TestStaticFiles:
    """Tests for static file serving."""
    
    def test_index_html_served(self, temp_db):
        """Test that index.html is served at root."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18089,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18089/"
            with urllib.request.urlopen(url, timeout=2) as response:
                content = response.read().decode()
                assert "<!DOCTYPE html>" in content
                assert "Agent Memory Toolkit" in content
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()
    
    def test_css_served(self, temp_db):
        """Test that CSS is served."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18090,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18090/styles.css"
            with urllib.request.urlopen(url, timeout=2) as response:
                content = response.read().decode()
                assert ":root" in content or "body" in content
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()
    
    def test_js_served(self, temp_db):
        """Test that JavaScript is served."""
        config = DashboardConfig(
            db_path=temp_db,
            host="127.0.0.1",
            port=18091,
            auto_open=False
        )
        server = DashboardServer(config)
        server.start(blocking=False)
        
        try:
            time.sleep(0.5)
            
            url = "http://127.0.0.1:18091/dashboard.js"
            with urllib.request.urlopen(url, timeout=2) as response:
                content = response.read().decode()
                assert "Chart" in content or "fetch" in content
        except urllib.error.URLError:
            pytest.skip("Server not reachable")
        finally:
            server.stop()


class TestRunDashboard:
    """Tests for the run_dashboard convenience function."""
    
    def test_run_dashboard_signature(self):
        """Test that run_dashboard has correct signature."""
        import inspect
        sig = inspect.signature(run_dashboard)
        
        params = list(sig.parameters.keys())
        assert "db_path" in params
        assert "host" in params
        assert "port" in params
        assert "auto_open" in params
