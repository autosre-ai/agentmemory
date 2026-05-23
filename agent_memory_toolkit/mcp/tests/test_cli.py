"""Tests for MCP CLI commands."""

import pytest
from click.testing import CliRunner

from agent_memory_toolkit.cli import cli


class TestMCPCommands:
    """Tests for MCP CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_mcp_help(self, runner):
        """Test 'amt mcp --help' command."""
        result = runner.invoke(cli, ["mcp", "--help"])
        
        assert result.exit_code == 0
        assert "MCP (Model Context Protocol)" in result.output
    
    def test_mcp_serve_help(self, runner):
        """Test 'amt mcp serve --help' command."""
        result = runner.invoke(cli, ["mcp", "serve", "--help"])
        
        assert result.exit_code == 0
        assert "--transport" in result.output
        assert "--db" in result.output
        assert "--host" in result.output
        assert "--port" in result.output
        assert "stdio" in result.output
        assert "sse" in result.output
    
    def test_mcp_config_claude(self, runner):
        """Test 'amt mcp config claude' command."""
        result = runner.invoke(cli, ["mcp", "config", "claude"])
        
        assert result.exit_code == 0
        assert "mcpServers" in result.output
        assert "agent-memory-toolkit" in result.output
        assert "amt" in result.output
        assert "mcp" in result.output
        assert "serve" in result.output
    
    def test_mcp_config_cursor(self, runner):
        """Test 'amt mcp config cursor' command."""
        result = runner.invoke(cli, ["mcp", "config", "cursor"])
        
        assert result.exit_code == 0
        assert "mcpServers" in result.output
        assert "agent-memory-toolkit" in result.output
        assert "env" in result.output
    
    def test_mcp_config_with_custom_db(self, runner):
        """Test 'amt mcp config' with custom database path."""
        result = runner.invoke(cli, ["mcp", "config", "claude", "--db", "/custom/path/memory.db"])
        
        assert result.exit_code == 0
        assert "memory.db" in result.output
    
    def test_mcp_tools(self, runner):
        """Test 'amt mcp tools' command."""
        result = runner.invoke(cli, ["mcp", "tools"])
        
        assert result.exit_code == 0
        assert "Memory Operations:" in result.output
        assert "memory_add" in result.output
        assert "memory_query" in result.output
        assert "extract_memories" in result.output
        assert "guard_check" in result.output
        assert "compress_context" in result.output
        assert "count_tokens" in result.output
    
    def test_info_includes_mcp(self, runner):
        """Test 'amt info' includes MCP module."""
        result = runner.invoke(cli, ["info"])
        
        assert result.exit_code == 0
        assert "mcp" in result.output.lower() or "Model Context Protocol" in result.output
