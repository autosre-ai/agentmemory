"""
Tests for the dashboard CLI commands.
"""

import json
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agentmemory.cli import cli


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
            ('mem2', 'Test memory 2', '{"domain": "technical"}', 'main', datetime('now'), datetime('now'));
        
        INSERT INTO commits (id, branch, message, created_at) VALUES
            ('c1', 'main', 'Initial commit', datetime('now'));
    """)
    
    conn.close()
    
    yield db_path
    
    os.unlink(db_path)


class TestDashboardCLI:
    """Tests for dashboard CLI commands."""
    
    def test_dashboard_group_exists(self):
        """Test that dashboard command group exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', '--help'])
        
        assert result.exit_code == 0
        assert 'dashboard' in result.output.lower()
    
    def test_dashboard_serve_help(self):
        """Test dashboard serve help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'serve', '--help'])
        
        assert result.exit_code == 0
        assert '--host' in result.output
        assert '--port' in result.output
        assert '--db' in result.output
        assert '--no-open' in result.output
    
    def test_dashboard_stats_help(self):
        """Test dashboard stats help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'stats', '--help'])
        
        assert result.exit_code == 0
        assert '--db' in result.output
        assert '--days' in result.output
        assert '--json-output' in result.output
    
    def test_dashboard_stats_text_output(self, temp_db):
        """Test dashboard stats with text output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'stats', '--db', temp_db])
        
        assert result.exit_code == 0
        assert 'Memory Stats' in result.output or 'memories' in result.output.lower()
    
    def test_dashboard_stats_json_output(self, temp_db):
        """Test dashboard stats with JSON output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'stats', '--db', temp_db, '-j'])
        
        assert result.exit_code == 0
        
        # Parse the JSON output
        data = json.loads(result.output)
        assert 'memory_stats' in data
        assert 'domain_distribution' in data
        assert 'storage_metrics' in data
    
    def test_dashboard_stats_custom_days(self, temp_db):
        """Test dashboard stats with custom days parameter."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'stats', '--db', temp_db, '--days', '7', '-j'])
        
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'memory_stats' in data
    
    def test_dashboard_stats_missing_db(self):
        """Test dashboard stats with missing database."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'stats', '--db', '/nonexistent/path.db'])
        
        assert result.exit_code == 1
        assert 'error' in result.output.lower()
    
    @patch('agent_memory.dashboard.server.webbrowser.open')
    def test_dashboard_serve_no_open(self, mock_browser, temp_db):
        """Test that --no-open prevents browser from opening."""
        runner = CliRunner()
        
        # Use keyboard interrupt to stop server immediately
        with patch.object(
            __import__('agent_memory.dashboard.server', fromlist=['DashboardServer']).DashboardServer,
            'start'
        ) as mock_start:
            result = runner.invoke(cli, ['dashboard', 'serve', '--db', temp_db, '--no-open'])
        
        # Even if it fails, it should have tried to start with auto_open=False
        mock_browser.assert_not_called()


class TestDashboardIntegration:
    """Integration tests for the dashboard."""
    
    def test_stats_shows_correct_memory_count(self, temp_db):
        """Test that stats shows correct memory count."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'stats', '--db', temp_db, '-j'])
        
        assert result.exit_code == 0
        data = json.loads(result.output)
        
        # Should show 2 memories
        assert data['memory_stats']['active_memories'] == 2
    
    def test_stats_shows_domain_distribution(self, temp_db):
        """Test that stats shows domain distribution."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'stats', '--db', temp_db, '-j'])
        
        assert result.exit_code == 0
        data = json.loads(result.output)
        
        # Should have domain counts
        domains = data['domain_distribution']['domain_counts']
        assert 'test' in domains or 'technical' in domains
    
    def test_stats_shows_branch_info(self, temp_db):
        """Test that stats shows branch information."""
        runner = CliRunner()
        result = runner.invoke(cli, ['dashboard', 'stats', '--db', temp_db, '-j'])
        
        assert result.exit_code == 0
        data = json.loads(result.output)
        
        # Should have at least the main branch
        branches = data['branch_comparison']['branches']
        assert len(branches) >= 1
        assert any(b['name'] == 'main' for b in branches)
