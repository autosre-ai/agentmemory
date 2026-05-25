"""
MCP Tool Definitions with JSON Schemas for Agent Memory Toolkit.

This module provides MCP-compliant tool definitions with proper JSON schemas
as per the Model Context Protocol specification.

Tool naming conventions:
- memory_store: Store a new memory (alias for memory_add)
- memory_retrieve: Retrieve memories by query (alias for memory_query)
- memory_forget: Remove memories (alias for memory_delete)
- memory_search: Search with filters (advanced search)
"""

from __future__ import annotations

from typing import Any, Dict, List

# JSON Schema definitions for MCP tools following the MCP specification
# https://spec.modelcontextprotocol.io/specification/server/tools/

MEMORY_STORE_SCHEMA: Dict[str, Any] = {
    "name": "memory_store",
    "description": """Store a new memory in the agent's memory store.

Use this tool to save important information, facts, preferences, or any data
that should be retrievable later. The memory will be indexed for full-text search.

Best practices:
- Store atomic pieces of information (one concept per memory)
- Include source attribution when available
- Use tags for categorization and filtering
- Set confidence < 1.0 for uncertain information
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory content to store. Should be a clear, self-contained piece of information.",
                "minLength": 1,
                "maxLength": 65536
            },
            "source": {
                "type": "string",
                "description": "Source of the memory (e.g., 'user', 'conversation', 'document', 'inference')",
                "default": "mcp"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization (e.g., ['preference', 'work', 'personal'])",
                "default": []
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0 (1.0 = certain, <1.0 = uncertain)",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 1.0
            },
            "metadata": {
                "type": "object",
                "description": "Additional key-value metadata to store with the memory",
                "additionalProperties": True
            }
        },
        "required": ["content"]
    }
}

MEMORY_RETRIEVE_SCHEMA: Dict[str, Any] = {
    "name": "memory_retrieve",
    "description": """Retrieve memories using natural language query.

Use this to search and retrieve relevant memories from the store.
The search uses BM25 full-text ranking to find the most relevant results.

Use cases:
- Find information stored earlier in the conversation
- Recall user preferences or personal details
- Look up facts or knowledge
- Get context for decision making
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query to find relevant memories",
                "minLength": 1
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "minimum": 1,
                "maximum": 100,
                "default": 10
            },
            "min_score": {
                "type": "number",
                "description": "Minimum relevance score threshold (0.0 to 1.0)",
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["query"]
    }
}

MEMORY_FORGET_SCHEMA: Dict[str, Any] = {
    "name": "memory_forget",
    "description": """Remove a memory from the store.

Use this to delete outdated, incorrect, or no longer relevant memories.
By default, performs a soft delete (memory can be recovered).
Use hard_delete=true for permanent removal.

Caution:
- Deleted memories cannot be searched
- Hard deletions are irreversible
- Consider updating instead of deleting when possible
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The unique identifier of the memory to delete",
                "minLength": 1
            },
            "hard_delete": {
                "type": "boolean",
                "description": "If true, permanently delete the memory; if false, soft delete (recoverable)",
                "default": False
            }
        },
        "required": ["memory_id"]
    }
}

MEMORY_SEARCH_SCHEMA: Dict[str, Any] = {
    "name": "memory_search",
    "description": """Advanced memory search with filters and options.

More powerful than memory_retrieve, supporting:
- Tag filtering
- Date range filtering
- Confidence thresholds
- Multiple search methods (FTS, vector, hybrid)
- Recency boosting

Use this when you need more control over search results.
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string (supports FTS5 syntax for advanced queries)",
                "minLength": 1
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return",
                "minimum": 1,
                "maximum": 100,
                "default": 10
            },
            "method": {
                "type": "string",
                "description": "Search method to use",
                "enum": ["auto", "fts", "vector", "hybrid"],
                "default": "auto"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter results to memories with these tags"
            },
            "min_confidence": {
                "type": "number",
                "description": "Minimum confidence score for results",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "boost_recent": {
                "type": "boolean",
                "description": "Boost scores for more recently created memories",
                "default": False
            },
            "boost_confidence": {
                "type": "boolean",
                "description": "Boost scores based on memory confidence levels",
                "default": False
            },
            "include_deleted": {
                "type": "boolean",
                "description": "Include soft-deleted memories in results",
                "default": False
            }
        },
        "required": ["query"]
    }
}

MEMORY_GET_SCHEMA: Dict[str, Any] = {
    "name": "memory_get",
    "description": """Get a specific memory by its ID.

Retrieve the full content and metadata of a memory when you know its ID.
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The unique identifier of the memory",
                "minLength": 1
            }
        },
        "required": ["memory_id"]
    }
}

MEMORY_UPDATE_SCHEMA: Dict[str, Any] = {
    "name": "memory_update",
    "description": """Update an existing memory.

Modify the content or metadata of a stored memory.
Version history is maintained for rollback capability.
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The ID of the memory to update",
                "minLength": 1
            },
            "content": {
                "type": "string",
                "description": "New content (omit to keep current content)"
            },
            "source": {
                "type": "string",
                "description": "Update source attribution"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Update tags (replaces existing tags)"
            },
            "confidence": {
                "type": "number",
                "description": "Update confidence score",
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["memory_id"]
    }
}

MEMORY_LIST_SCHEMA: Dict[str, Any] = {
    "name": "memory_list",
    "description": """List stored memories with optional filtering.

Get a paginated list of memories for browsing or review.
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum memories to return",
                "minimum": 1,
                "maximum": 100,
                "default": 20
            },
            "offset": {
                "type": "integer",
                "description": "Number of memories to skip (for pagination)",
                "minimum": 0,
                "default": 0
            },
            "tag": {
                "type": "string",
                "description": "Filter by tag"
            },
            "include_deleted": {
                "type": "boolean",
                "description": "Include soft-deleted memories",
                "default": False
            }
        },
        "required": []
    }
}

MEMORY_HISTORY_SCHEMA: Dict[str, Any] = {
    "name": "memory_history",
    "description": """Get version history for a memory or the entire store.

View the change history showing when memories were added, updated, or deleted.
Useful for auditing and understanding memory evolution.
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Memory ID for specific memory history (omit for store history)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum history entries to return",
                "minimum": 1,
                "maximum": 100,
                "default": 20
            }
        },
        "required": []
    }
}

EXTRACT_MEMORIES_SCHEMA: Dict[str, Any] = {
    "name": "extract_memories",
    "description": """Extract structured memories from text.

Analyzes text and extracts factual information into structured memories.
Detects entities, facts, preferences, and other information types.

Example:
    Input: "My name is John and I work at Google. I prefer Python."
    Output: [
        {domain: "biography", key: "name", value: "John"},
        {domain: "work", key: "company", value: "Google"},
        {domain: "preferences", key: "language", value: "Python"}
    ]
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to extract memories from",
                "minLength": 1
            },
            "source": {
                "type": "string",
                "description": "Source identifier for extracted memories"
            },
            "auto_store": {
                "type": "boolean",
                "description": "Automatically store extracted memories",
                "default": False
            }
        },
        "required": ["text"]
    }
}

GUARD_CHECK_SCHEMA: Dict[str, Any] = {
    "name": "guard_check",
    "description": """Validate content for security issues.

Checks content for potential injection attacks, suspicious patterns,
and assesses confidence/trustworthiness.

Use before storing user-provided content to prevent memory poisoning.
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to validate for security issues",
                "minLength": 1
            }
        },
        "required": ["content"]
    }
}

COMPRESS_CONTEXT_SCHEMA: Dict[str, Any] = {
    "name": "compress_context",
    "description": """Compress conversation context to fit token budget.

Takes a list of messages and compresses them to fit within a specified
token limit while preserving the most important information.
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["system", "user", "assistant"]
                        },
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                },
                "description": "List of messages to compress"
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum token budget",
                "minimum": 100,
                "default": 4000
            },
            "reserve_tokens": {
                "type": "integer",
                "description": "Tokens to reserve for response",
                "minimum": 0,
                "default": 500
            },
            "mode": {
                "type": "string",
                "description": "Compression aggressiveness",
                "enum": ["aggressive", "balanced", "conservative", "lossless"],
                "default": "balanced"
            }
        },
        "required": ["messages"]
    }
}

COUNT_TOKENS_SCHEMA: Dict[str, Any] = {
    "name": "count_tokens",
    "description": """Count tokens in text.

Estimate the number of tokens in text content for context management.
""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to count tokens for"
            },
            "model": {
                "type": "string",
                "description": "Tokenizer model to use",
                "default": "gpt-4"
            }
        },
        "required": ["text"]
    }
}

# All tool schemas grouped by category
TOOL_SCHEMAS: Dict[str, List[Dict[str, Any]]] = {
    "memory": [
        MEMORY_STORE_SCHEMA,
        MEMORY_RETRIEVE_SCHEMA,
        MEMORY_FORGET_SCHEMA,
        MEMORY_SEARCH_SCHEMA,
        MEMORY_GET_SCHEMA,
        MEMORY_UPDATE_SCHEMA,
        MEMORY_LIST_SCHEMA,
        MEMORY_HISTORY_SCHEMA,
    ],
    "extraction": [
        EXTRACT_MEMORIES_SCHEMA,
    ],
    "security": [
        GUARD_CHECK_SCHEMA,
    ],
    "compression": [
        COMPRESS_CONTEXT_SCHEMA,
        COUNT_TOKENS_SCHEMA,
    ],
}

# Flat list of all schemas
ALL_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    schema
    for schemas in TOOL_SCHEMAS.values()
    for schema in schemas
]


def get_tool_schema(name: str) -> Dict[str, Any] | None:
    """Get the JSON schema for a specific tool by name."""
    for schema in ALL_TOOL_SCHEMAS:
        if schema["name"] == name:
            return schema
    return None


def get_tools_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all tool schemas for a category."""
    return TOOL_SCHEMAS.get(category, [])


def list_tool_names() -> List[str]:
    """Get list of all available tool names."""
    return [schema["name"] for schema in ALL_TOOL_SCHEMAS]


# MCP tool listing format
def get_mcp_tools_list() -> List[Dict[str, Any]]:
    """
    Get tools in MCP ListTools response format.
    
    Returns a list of tool definitions conforming to the MCP specification.
    """
    return [
        {
            "name": schema["name"],
            "description": schema["description"],
            "inputSchema": schema["inputSchema"],
        }
        for schema in ALL_TOOL_SCHEMAS
    ]
