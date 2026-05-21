"""
MCP (Model Context Protocol) Server for Agent Memory Toolkit.

This module provides an MCP server that exposes the memory toolkit's
functionality as tools that any MCP-compatible LLM client can use.

Supported tools:
- memory_add: Add a new memory
- memory_query: Search memories
- memory_get: Get a specific memory by ID
- memory_update: Update an existing memory
- memory_delete: Delete a memory
- memory_history: Get version history
- extract_memories: Extract structured memories from text
- guard_check: Validate content for security issues
- compress_context: Compress conversation context

Usage:
    # Run as standalone CLI
    amt-mcp serve
    
    # Or via main CLI
    amt mcp serve

    # Or programmatically
    from agentmemory.mcp import create_mcp_server
    server = create_mcp_server()
    server.run()
"""

from .server import (
    create_mcp_server,
    MemoryToolkit,
    MCPConfig,
)

__all__ = [
    "create_mcp_server",
    "MemoryToolkit",
    "MCPConfig",
]
