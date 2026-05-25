"""
MCP (Model Context Protocol) Server for Agent Memory Toolkit.

This module provides an MCP server that exposes the memory toolkit's
functionality as tools that any MCP-compatible LLM client can use.

Supported tools:
- memory_store: Store a new memory (primary)
- memory_retrieve: Retrieve memories by query (primary)
- memory_forget: Remove memories (primary)
- memory_search: Search with filters (primary)
- memory_add: Add a new memory (alias for memory_store)
- memory_query: Search memories (alias for memory_retrieve)
- memory_delete: Delete a memory (alias for memory_forget)
- memory_get: Get a specific memory by ID
- memory_update: Update an existing memory
- memory_list: List memories with pagination
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
    from agent_memory_toolkit.mcp import create_mcp_server
    server = create_mcp_server()
    server.run()
    
    # Access tool schemas
    from agent_memory_toolkit.mcp.tools import get_mcp_tools_list
    tools = get_mcp_tools_list()
"""

from .server import (
    create_mcp_server,
    MemoryToolkit,
    MCPConfig,
    run_server,
)

from .tools import (
    TOOL_SCHEMAS,
    ALL_TOOL_SCHEMAS,
    get_tool_schema,
    get_tools_by_category,
    list_tool_names,
    get_mcp_tools_list,
    # Individual schemas
    MEMORY_STORE_SCHEMA,
    MEMORY_RETRIEVE_SCHEMA,
    MEMORY_FORGET_SCHEMA,
    MEMORY_SEARCH_SCHEMA,
    MEMORY_GET_SCHEMA,
    MEMORY_UPDATE_SCHEMA,
    MEMORY_LIST_SCHEMA,
    MEMORY_HISTORY_SCHEMA,
    EXTRACT_MEMORIES_SCHEMA,
    GUARD_CHECK_SCHEMA,
    COMPRESS_CONTEXT_SCHEMA,
    COUNT_TOKENS_SCHEMA,
)

__all__ = [
    # Server
    "create_mcp_server",
    "run_server",
    "MemoryToolkit",
    "MCPConfig",
    # Tool utilities
    "TOOL_SCHEMAS",
    "ALL_TOOL_SCHEMAS",
    "get_tool_schema",
    "get_tools_by_category",
    "list_tool_names",
    "get_mcp_tools_list",
    # Individual schemas
    "MEMORY_STORE_SCHEMA",
    "MEMORY_RETRIEVE_SCHEMA",
    "MEMORY_FORGET_SCHEMA",
    "MEMORY_SEARCH_SCHEMA",
    "MEMORY_GET_SCHEMA",
    "MEMORY_UPDATE_SCHEMA",
    "MEMORY_LIST_SCHEMA",
    "MEMORY_HISTORY_SCHEMA",
    "EXTRACT_MEMORIES_SCHEMA",
    "GUARD_CHECK_SCHEMA",
    "COMPRESS_CONTEXT_SCHEMA",
    "COUNT_TOKENS_SCHEMA",
]
