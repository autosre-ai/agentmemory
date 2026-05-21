"""
MCP Server Implementation for Agent Memory Toolkit.

Provides a FastMCP-based server that exposes memory toolkit functionality
as MCP tools for LLM clients like Claude Desktop and Cursor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Literal

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    """Configuration for the MCP server."""
    
    # Server settings
    name: str = "agent-memory-toolkit"
    host: str = "127.0.0.1"
    port: int = 8765
    
    # Database paths
    memory_db: str = "agent_memory.db"
    audit_db: str = "audit.db"
    
    # Feature toggles
    enable_extraction: bool = True
    enable_security: bool = True
    enable_compression: bool = True
    enable_team: bool = True
    
    # Security settings
    security_level: str = "medium"
    
    # Extraction settings
    extraction_mode: str = "rule"
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class MemoryToolkit:
    """
    Wrapper around memory toolkit components for MCP tools.
    
    Provides lazy initialization of toolkit components and
    handles the interface between MCP tools and the toolkit.
    """
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self._memory_store = None
        self._extractor = None
        self._guard = None
        self._compressor = None
        self._team_store = None
    
    @property
    def memory_store(self):
        """Lazy-load memory store."""
        if self._memory_store is None:
            from agentmemory.store import MemoryStore
            self._memory_store = MemoryStore(self.config.memory_db)
        return self._memory_store
    
    @property
    def extractor(self):
        """Lazy-load memory extractor."""
        if self._extractor is None:
            from agentmemory.extraction import MemoryExtractor
            self._extractor = MemoryExtractor(mode=self.config.extraction_mode)
        return self._extractor
    
    @property
    def guard(self):
        """Lazy-load memory guard."""
        if self._guard is None:
            from agentmemory.security import MemoryGuard, SecurityLevel
            level_map = {
                "minimal": SecurityLevel.MINIMAL,
                "low": SecurityLevel.LOW,
                "medium": SecurityLevel.MEDIUM,
                "high": SecurityLevel.HIGH,
                "paranoid": SecurityLevel.PARANOID,
            }
            level = level_map.get(self.config.security_level, SecurityLevel.MEDIUM)
            self._guard = MemoryGuard(level=level)
        return self._guard
    
    @property
    def compressor(self):
        """Lazy-load context compressor."""
        if self._compressor is None:
            from agentmemory.compression import ContextCompressor, CompressionConfig
            self._compressor = ContextCompressor(config=CompressionConfig())
        return self._compressor
    
    def close(self):
        """Close all connections."""
        if self._memory_store is not None:
            self._memory_store.close()
            self._memory_store = None
        if self._team_store is not None:
            self._team_store.close()
            self._team_store = None


def create_mcp_server(config: Optional[MCPConfig] = None) -> FastMCP:
    """
    Create and configure an MCP server with memory toolkit tools.
    
    Args:
        config: Optional server configuration
        
    Returns:
        Configured FastMCP server instance
    """
    if config is None:
        config = MCPConfig()
    
    # Create toolkit wrapper
    toolkit = MemoryToolkit(config)
    
    # Create MCP server
    mcp = FastMCP(
        name=config.name,
        instructions="""
Agent Memory Toolkit MCP Server.

This server provides tools for managing AI agent memory:
- Store and retrieve memories with full-text search
- Extract structured facts from text
- Validate content for security issues
- Compress conversation context to fit token budgets

Use these tools to give your agent persistent memory capabilities.
        """,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
    
    # Register memory CRUD tools
    _register_memory_tools(mcp, toolkit)
    
    # Register extraction tools
    if config.enable_extraction:
        _register_extraction_tools(mcp, toolkit)
    
    # Register security tools
    if config.enable_security:
        _register_security_tools(mcp, toolkit)
    
    # Register compression tools
    if config.enable_compression:
        _register_compression_tools(mcp, toolkit)
    
    return mcp


def _register_memory_tools(mcp: FastMCP, toolkit: MemoryToolkit):
    """Register memory CRUD tools."""
    
    @mcp.tool(
        name="memory_add",
        description="""
Add a new memory to the store.

This stores a piece of information that can be retrieved later via search.
Include relevant tags and source information for better organization.

Arguments:
- content: The memory content to store
- source: Optional source of the memory (e.g., "user", "conversation", "document")
- tags: Optional list of tags for categorization
- confidence: Confidence score 0-1 (default: 1.0)

Returns the memory ID and confirmation.
        """
    )
    def memory_add(
        content: str,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 1.0
    ) -> dict:
        """Add a new memory to the store."""
        try:
            metadata = {
                "source": source or "mcp",
                "tags": tags or [],
                "confidence": confidence,
            }
            memory = toolkit.memory_store.add(content, metadata=metadata)
            return {
                "success": True,
                "memory_id": memory.id,
                "content": content[:100] + ("..." if len(content) > 100 else ""),
                "message": f"Memory stored successfully with ID: {memory.id}"
            }
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp.tool(
        name="memory_query",
        description="""
Search memories using full-text search.

Find relevant memories based on a search query. Uses BM25 ranking
for relevance scoring. Returns matching memories sorted by relevance.

Arguments:
- query: Search query string
- limit: Maximum results to return (default: 10)

Returns a list of matching memories with relevance scores.
        """
    )
    def memory_query(query: str, limit: int = 10) -> dict:
        """Search memories using full-text search."""
        try:
            results = toolkit.memory_store.search_fts(query, limit=limit)
            memories = [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "score": r.score,
                    "created_at": r.memory.created_at.isoformat() if hasattr(r.memory, 'created_at') else None,
                }
                for r in results
            ]
            return {
                "success": True,
                "query": query,
                "count": len(memories),
                "memories": memories
            }
        except Exception as e:
            logger.error(f"Error querying memories: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp.tool(
        name="memory_get",
        description="""
Get a specific memory by its ID.

Retrieve the full content and metadata of a memory.

Arguments:
- memory_id: The unique identifier of the memory

Returns the memory content and metadata.
        """
    )
    def memory_get(memory_id: str) -> dict:
        """Get a memory by ID."""
        try:
            memory = toolkit.memory_store.get(memory_id)
            if memory is None:
                return {"success": False, "error": f"Memory not found: {memory_id}"}
            return {
                "success": True,
                "memory": {
                    "id": memory.id,
                    "content": memory.content,
                    "version": memory.version,
                    "created_at": memory.created_at.isoformat() if hasattr(memory, 'created_at') else None,
                    "updated_at": memory.updated_at.isoformat() if hasattr(memory, 'updated_at') else None,
                }
            }
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp.tool(
        name="memory_update",
        description="""
Update an existing memory.

Replace the content of a memory while maintaining version history.

Arguments:
- memory_id: The ID of the memory to update
- content: New content for the memory

Returns confirmation of the update.
        """
    )
    def memory_update(
        memory_id: str,
        content: str,
    ) -> dict:
        """Update an existing memory."""
        try:
            toolkit.memory_store.update(memory_id, content=content)
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Memory {memory_id} updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp.tool(
        name="memory_delete",
        description="""
Delete a memory from the store.

Permanently remove a memory. This action cannot be undone.

Arguments:
- memory_id: The ID of the memory to delete

Returns confirmation of deletion.
        """
    )
    def memory_delete(memory_id: str) -> dict:
        """Delete a memory."""
        try:
            toolkit.memory_store.delete(memory_id)
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Memory {memory_id} deleted successfully"
            }
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp.tool(
        name="memory_list",
        description="""
List memories with optional filtering.

Get a paginated list of stored memories.

Arguments:
- limit: Maximum memories to return (default: 20)
- offset: Offset for pagination (default: 0)
- tag: Optional tag to filter by

Returns a list of memories.
        """
    )
    def memory_list(
        limit: int = 20,
        offset: int = 0,
        tag: Optional[str] = None
    ) -> dict:
        """List memories with pagination."""
        try:
            memories = toolkit.memory_store.list(limit=limit, offset=offset, tag=tag)
            total = toolkit.memory_store.count()
            return {
                "success": True,
                "total": total,
                "offset": offset,
                "limit": limit,
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content[:200] + ("..." if len(m.content) > 200 else ""),
                        "version": m.version,
                        "created_at": m.created_at.isoformat() if hasattr(m, 'created_at') else None,
                    }
                    for m in memories
                ]
            }
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp.tool(
        name="memory_history",
        description="""
Get version history for a memory or the store.

View the change history showing when memories were added, updated, or deleted.

Arguments:
- memory_id: Optional memory ID for specific memory history
- limit: Maximum commits to show (default: 20)

Returns commit history.
        """
    )
    def memory_history(
        memory_id: Optional[str] = None,
        limit: int = 20
    ) -> dict:
        """Get version history."""
        try:
            if memory_id:
                history = toolkit.memory_store.get_memory_history(memory_id)
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "history": history
                }
            else:
                commits = toolkit.memory_store.get_history(limit=limit)
                return {
                    "success": True,
                    "branch": toolkit.memory_store.current_branch,
                    "commits": [
                        {
                            "id": c.id[:8],
                            "message": c.message,
                            "created_at": c.created_at.isoformat(),
                        }
                        for c in commits
                    ]
                }
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return {"success": False, "error": str(e)}


def _register_extraction_tools(mcp: FastMCP, toolkit: MemoryToolkit):
    """Register memory extraction tools."""
    
    @mcp.tool(
        name="extract_memories",
        description="""
Extract structured memories from text.

Analyzes text and extracts factual information into structured memories.
Detects entities, facts, preferences, and other information types.

Arguments:
- text: The text to extract memories from
- source: Optional source identifier for the extracted memories

Returns extracted memories organized by domain (biography, preferences, work, etc.).

Example:
    Input: "My name is John and I work at Google. I prefer Python."
    Output: [
        {domain: "biography", key: "name", value: "John"},
        {domain: "work", key: "company", value: "Google"},
        {domain: "preferences", key: "language", value: "Python"}
    ]
        """
    )
    def extract_memories(text: str, source: Optional[str] = None) -> dict:
        """Extract structured memories from text."""
        try:
            result = toolkit.extractor.extract(text, source=source)
            memories = [
                {
                    "domain": m.domain.value if hasattr(m.domain, 'value') else str(m.domain),
                    "key": m.key,
                    "value": m.value,
                    "confidence": m.confidence,
                }
                for m in result.memories
            ]
            return {
                "success": True,
                "count": len(memories),
                "memories": memories,
                "extraction_method": result.method,
                "processing_time_ms": result.processing_time_ms,
            }
        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return {"success": False, "error": str(e)}


def _register_security_tools(mcp: FastMCP, toolkit: MemoryToolkit):
    """Register security validation tools."""
    
    @mcp.tool(
        name="guard_check",
        description="""
Validate content for security issues.

Checks content for potential injection attacks, suspicious patterns,
and assesses confidence/trustworthiness.

Use this before storing user-provided content to prevent memory poisoning.

Arguments:
- content: The content to validate

Returns validation result with safety assessment and any detected issues.
        """
    )
    def guard_check(content: str) -> dict:
        """Check content for security issues."""
        try:
            result = toolkit.guard.validate_content(content)
            response = {
                "success": True,
                "is_safe": result.is_safe,
                "adjusted_confidence": result.adjusted_confidence,
                "validation_time_ms": result.validation_time_ms,
            }
            
            if result.rejection_reason:
                response["rejection_reason"] = result.rejection_reason
            
            if result.poison_result:
                response["poison_detection"] = {
                    "is_safe": result.poison_result.is_safe,
                    "risk_score": result.poison_result.risk_score,
                    "detected_patterns": [
                        p.value if hasattr(p, 'value') else str(p)
                        for p in result.poison_result.detected_patterns
                    ] if result.poison_result.detected_patterns else [],
                }
            
            return response
        except Exception as e:
            logger.error(f"Error checking content: {e}")
            return {"success": False, "error": str(e)}


def _register_compression_tools(mcp: FastMCP, toolkit: MemoryToolkit):
    """Register context compression tools."""
    
    @mcp.tool(
        name="compress_context",
        description="""
Compress conversation context to fit token budget.

Takes a list of messages and compresses them to fit within a specified
token limit while preserving the most important information.

Uses intelligent strategies like summarization and importance ranking
to maintain context quality while reducing size.

Arguments:
- messages: List of message dicts with 'role' and 'content' fields
- max_tokens: Maximum token budget (default: 4000)
- reserve_tokens: Tokens to reserve for response (default: 500)
- mode: Compression mode - "aggressive", "balanced", "conservative", "lossless"

Returns compressed messages with compression statistics.
        """
    )
    def compress_context(
        messages: List[dict],
        max_tokens: int = 4000,
        reserve_tokens: int = 500,
        mode: str = "balanced"
    ) -> dict:
        """Compress conversation context."""
        try:
            from agentmemory.compression import CompressionConfig, CompressionMode
            
            mode_map = {
                "aggressive": CompressionMode.AGGRESSIVE,
                "balanced": CompressionMode.BALANCED,
                "conservative": CompressionMode.CONSERVATIVE,
                "lossless": CompressionMode.LOSSLESS,
            }
            
            config = CompressionConfig(
                max_tokens=max_tokens,
                reserve_tokens=reserve_tokens,
                mode=mode_map.get(mode, CompressionMode.BALANCED),
            )
            
            from agentmemory.compression import ContextCompressor
            compressor = ContextCompressor(config=config)
            result = compressor.compress(messages)
            
            return {
                "success": True,
                "messages": result.messages,
                "stats": {
                    "original_tokens": result.original_tokens,
                    "compressed_tokens": result.compressed_tokens,
                    "compression_ratio": result.compression_ratio,
                    "tokens_saved": result.tokens_saved,
                    "strategy_used": result.strategy_used,
                }
            }
        except Exception as e:
            logger.error(f"Error compressing context: {e}")
            return {"success": False, "error": str(e)}
    
    @mcp.tool(
        name="count_tokens",
        description="""
Count tokens in text or messages.

Estimate the number of tokens in text content for context management.

Arguments:
- text: Text to count tokens for
- model: Token model to use (default: "gpt-4")

Returns the token count.
        """
    )
    def count_tokens(text: str, model: str = "gpt-4") -> dict:
        """Count tokens in text."""
        try:
            from agentmemory.compression import TokenCounter
            counter = TokenCounter(model=model)
            count = counter.count(text)
            return {
                "success": True,
                "text_length": len(text),
                "token_count": count,
                "model": model,
            }
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return {"success": False, "error": str(e)}


def run_server(config: Optional[MCPConfig] = None, transport: str = "stdio"):
    """
    Run the MCP server.
    
    Args:
        config: Server configuration
        transport: Transport type - "stdio" or "sse"
    """
    if config is None:
        config = MCPConfig()
    
    mcp = create_mcp_server(config)
    
    logger.info(f"Starting MCP server: {config.name}")
    logger.info(f"Transport: {transport}")
    
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Memory Toolkit MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                       help="Transport type (default: stdio)")
    parser.add_argument("--db", default="agent_memory.db",
                       help="Path to memory database")
    parser.add_argument("--host", default="127.0.0.1",
                       help="Host for SSE server")
    parser.add_argument("--port", type=int, default=8765,
                       help="Port for SSE server")
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    config = MCPConfig(
        memory_db=args.db,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    
    run_server(config, transport=args.transport)
