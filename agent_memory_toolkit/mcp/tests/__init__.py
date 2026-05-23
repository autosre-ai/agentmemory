"""Tests for MCP module initialization."""

import pytest


def test_module_imports():
    """Test that the mcp module can be imported."""
    from agent_memory_toolkit import mcp
    
    assert hasattr(mcp, 'create_mcp_server')
    assert hasattr(mcp, 'MCPConfig')
    assert hasattr(mcp, 'MemoryToolkit')


def test_mcp_config_defaults():
    """Test MCPConfig default values."""
    from agent_memory_toolkit.mcp import MCPConfig
    
    config = MCPConfig()
    
    assert config.name == "agent-memory-toolkit"
    assert config.host == "127.0.0.1"
    assert config.port == 8765
    assert config.memory_db == "agent_memory.db"
    assert config.security_level == "medium"
    assert config.extraction_mode == "rule"
    assert config.enable_extraction is True
    assert config.enable_security is True
    assert config.enable_compression is True


def test_mcp_config_custom():
    """Test MCPConfig with custom values."""
    from agent_memory_toolkit.mcp import MCPConfig
    
    config = MCPConfig(
        name="custom-server",
        host="0.0.0.0",
        port=9000,
        memory_db="/path/to/db.sqlite",
        security_level="high",
    )
    
    assert config.name == "custom-server"
    assert config.host == "0.0.0.0"
    assert config.port == 9000
    assert config.memory_db == "/path/to/db.sqlite"
    assert config.security_level == "high"
