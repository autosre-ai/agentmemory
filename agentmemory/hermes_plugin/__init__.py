"""Agent Memory Toolkit - Hermes Agent Plugin.

A native Hermes Agent memory provider using the Agent Memory Toolkit
for local-first, persistent memory with structured extraction, 
semantic search, and intelligent compression.

Features:
- SQLite-backed persistent memory with FTS5 full-text search
- Optional vector similarity search with sentence-transformers
- Structured memory extraction across 6 cognitive domains
- Security validation and poison detection
- Intelligent context compression

Config via environment variables:
  AGENT_MEMORY_DB_PATH     — Path to SQLite database (default: ~/.hermes/agent_memory.db)
  AGENT_MEMORY_AUTO_EMBED  — Enable auto-embedding (default: false)
  AGENT_MEMORY_MODEL       — Embedding model (default: all-MiniLM-L6-v2)

Or via $HERMES_HOME/agent_memory.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid import errors if dependencies missing
_TOOLKIT_AVAILABLE = False
_IMPORT_ERROR = None

try:
    from agentmemory import (
        MemoryStore,
        MemoryExtractor,
        ContextCompressor,
        MemoryGuard,
        SecurityLevel,
        CompressionMode,
    )
    _TOOLKIT_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars with $HERMES_HOME/agent_memory.json overrides."""
    try:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
    except ImportError:
        hermes_home = Path.home() / ".hermes"

    config = {
        "db_path": os.environ.get("AGENT_MEMORY_DB_PATH", str(hermes_home / "agent_memory.db")),
        "auto_embed": os.environ.get("AGENT_MEMORY_AUTO_EMBED", "false").lower() == "true",
        "embedding_model": os.environ.get("AGENT_MEMORY_MODEL", "all-MiniLM-L6-v2"),
        "extraction_mode": os.environ.get("AGENT_MEMORY_EXTRACTION_MODE", "rule"),
        "security_level": os.environ.get("AGENT_MEMORY_SECURITY_LEVEL", "medium"),
        "max_results": int(os.environ.get("AGENT_MEMORY_MAX_RESULTS", "10")),
    }

    config_path = hermes_home / "agent_memory.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------

ADD_MEMORY_SCHEMA = {
    "name": "memory_add",
    "description": (
        "Store a new memory/fact about the user or context. "
        "Memories are persistent across sessions. Use for preferences, "
        "decisions, facts, or any information worth remembering. "
        "Optionally specify tags for categorization and confidence level."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory/fact to store. Be specific and factual.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization (e.g., ['preference', 'work'])",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence level 0.0-1.0 (default: 0.9)",
            },
            "source": {
                "type": "string",
                "description": "Source of this memory (e.g., 'user_stated', 'inferred')",
            },
        },
        "required": ["content"],
    },
}

QUERY_MEMORY_SCHEMA = {
    "name": "memory_query",
    "description": (
        "Search stored memories by meaning or keywords. "
        "Returns relevant memories ranked by relevance. "
        "Use mode='fts' for keyword search, 'vector' for semantic similarity, "
        "'hybrid' for combined (default)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memories.",
            },
            "mode": {
                "type": "string",
                "enum": ["fts", "vector", "hybrid", "auto"],
                "description": "Search mode (default: auto - picks best based on query)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default: 10)",
            },
            "tag": {
                "type": "string",
                "description": "Filter by tag",
            },
        },
        "required": ["query"],
    },
}

EXTRACT_MEMORIES_SCHEMA = {
    "name": "memory_extract",
    "description": (
        "Extract structured memories from text or conversation. "
        "Automatically identifies facts across 6 domains: biography, "
        "preferences, work, social, temporal, procedural. "
        "Extracted memories are stored automatically."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to extract memories from.",
            },
            "domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter to specific domains (e.g., ['preferences', 'work'])",
            },
            "auto_store": {
                "type": "boolean",
                "description": "Automatically store extracted memories (default: true)",
            },
        },
        "required": ["text"],
    },
}

COMPRESS_CONTEXT_SCHEMA = {
    "name": "memory_compress",
    "description": (
        "Compress conversation history to fit within token budget. "
        "Preserves critical information while reducing size. "
        "Use when context is getting too long."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Conversation messages to compress.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Target token budget (default: 4000)",
            },
            "mode": {
                "type": "string",
                "enum": ["aggressive", "balanced", "conservative", "lossless"],
                "description": "Compression mode (default: balanced)",
            },
        },
        "required": ["messages"],
    },
}

PROFILE_SCHEMA = {
    "name": "memory_profile",
    "description": (
        "Get a summary of stored memories about the user. "
        "Returns key facts organized by domain/category."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "Filter to specific domain (e.g., 'preferences')",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum memories per domain (default: 5)",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider Implementation
# ---------------------------------------------------------------------------

class AgentMemoryToolkitProvider:
    """Agent Memory Toolkit provider for Hermes Agent.
    
    Provides local-first persistent memory with:
    - SQLite + FTS5 full-text search
    - Optional vector similarity search
    - Structured memory extraction
    - Security validation
    - Context compression
    """

    def __init__(self):
        self._config: Optional[dict] = None
        self._store: Optional["MemoryStore"] = None
        self._extractor: Optional["MemoryExtractor"] = None
        self._compressor: Optional["ContextCompressor"] = None
        self._guard: Optional["MemoryGuard"] = None
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._session_id: str = ""

    @property
    def name(self) -> str:
        return "agent-memory-toolkit"

    def is_available(self) -> bool:
        """Check if the toolkit is installed and configured."""
        if not _TOOLKIT_AVAILABLE:
            return False
        cfg = _load_config()
        return bool(cfg.get("db_path"))

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields for setup wizard."""
        return [
            {
                "key": "db_path",
                "description": "Path to SQLite database",
                "default": "~/.hermes/agent_memory.db",
            },
            {
                "key": "auto_embed",
                "description": "Enable vector embeddings for semantic search",
                "default": "false",
                "choices": ["true", "false"],
            },
            {
                "key": "embedding_model",
                "description": "Embedding model (if auto_embed=true)",
                "default": "all-MiniLM-L6-v2",
            },
            {
                "key": "extraction_mode",
                "description": "Memory extraction mode",
                "default": "rule",
                "choices": ["rule", "llm", "hybrid"],
            },
            {
                "key": "security_level",
                "description": "Security validation level",
                "default": "medium",
                "choices": ["minimal", "low", "medium", "high", "paranoid"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to $HERMES_HOME/agent_memory.json."""
        config_path = Path(hermes_home) / "agent_memory.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize the memory store and components."""
        if not _TOOLKIT_AVAILABLE:
            logger.warning(f"Agent Memory Toolkit not available: {_IMPORT_ERROR}")
            return

        self._config = _load_config()
        self._session_id = session_id

        try:
            # Initialize memory store
            db_path = self._config.get("db_path", ":memory:")
            if db_path.startswith("~"):
                db_path = str(Path(db_path).expanduser())
            
            # Ensure directory exists
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            self._store = MemoryStore(
                db_path=db_path,
                auto_embed=self._config.get("auto_embed", False),
                embedding_model=self._config.get("embedding_model", "all-MiniLM-L6-v2"),
            )

            # Initialize extractor
            extraction_mode = self._config.get("extraction_mode", "rule")
            self._extractor = MemoryExtractor(mode=extraction_mode)

            # Initialize compressor
            self._compressor = ContextCompressor(
                max_tokens=4000,
                mode=CompressionMode.BALANCED,
            )

            # Initialize security guard
            security_level = self._config.get("security_level", "medium")
            level_map = {
                "minimal": SecurityLevel.MINIMAL,
                "low": SecurityLevel.LOW,
                "medium": SecurityLevel.MEDIUM,
                "high": SecurityLevel.HIGH,
                "paranoid": SecurityLevel.PARANOID,
            }
            self._guard = MemoryGuard(level=level_map.get(security_level, SecurityLevel.MEDIUM))

            logger.debug(f"Agent Memory Toolkit initialized: db={db_path}, session={session_id}")

        except Exception as e:
            logger.warning(f"Agent Memory Toolkit init failed: {e}")
            self._store = None

    def system_prompt_block(self) -> str:
        """Return system prompt text describing memory capabilities."""
        if not self._store:
            return ""
        
        memory_count = len(list(self._store.list(limit=1000)))
        return (
            "# Agent Memory Toolkit\n"
            f"Active. {memory_count} memories stored.\n"
            "Use memory_add to store facts, memory_query to search, "
            "memory_extract to extract from text, memory_profile for overview.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched relevant context."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        
        if not result:
            return ""
        return f"## Relevant Memories\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background prefetch for next turn."""
        if not self._store:
            return

        def _run():
            try:
                results = self._store.search(query, limit=5)
                if results:
                    lines = [f"- {r.memory.content}" for r in results[:5]]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug(f"Memory prefetch failed: {e}")

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="memory-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Extract and store memories from the turn (non-blocking)."""
        if not self._store or not self._extractor:
            return

        def _sync():
            try:
                # Extract memories from user message
                if user_content.strip():
                    result = self._extractor.extract(user_content, source="user_message")
                    for memory in result.memories:
                        # Validate before storing
                        if self._guard:
                            validation = self._guard.validate_content(memory.value)
                            if not validation.is_safe:
                                continue
                        
                        self._store.add(
                            f"[{memory.domain.value}] {memory.key}: {memory.value}",
                            metadata={
                                "source": "auto_extract",
                                "domain": memory.domain.value,
                                "confidence": memory.confidence,
                                "session_id": session_id or self._session_id,
                            }
                        )
            except Exception as e:
                logger.debug(f"Memory sync failed: {e}")

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="memory-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for this provider."""
        return [
            ADD_MEMORY_SCHEMA,
            QUERY_MEMORY_SCHEMA,
            EXTRACT_MEMORIES_SCHEMA,
            COMPRESS_CONTEXT_SCHEMA,
            PROFILE_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call."""
        if not self._store:
            return json.dumps({"error": "Memory store not initialized"})

        try:
            if tool_name == "memory_add":
                return self._handle_add(args)
            elif tool_name == "memory_query":
                return self._handle_query(args)
            elif tool_name == "memory_extract":
                return self._handle_extract(args)
            elif tool_name == "memory_compress":
                return self._handle_compress(args)
            elif tool_name == "memory_profile":
                return self._handle_profile(args)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _handle_add(self, args: dict) -> str:
        """Handle memory_add tool call."""
        content = args.get("content", "")
        if not content:
            return json.dumps({"error": "Missing required parameter: content"})

        # Validate content
        if self._guard:
            validation = self._guard.validate_content(content)
            if not validation.is_safe:
                return json.dumps({
                    "error": f"Content rejected: {validation.rejection_reason}",
                    "quarantined": validation.is_quarantined,
                })

        tags = args.get("tags", [])
        confidence = args.get("confidence", 0.9)
        source = args.get("source", "user_stated")

        memory = self._store.add(
            content,
            metadata={
                "tags": tags,
                "confidence": confidence,
                "source": source,
                "session_id": self._session_id,
            }
        )

        return json.dumps({
            "result": "Memory stored successfully",
            "memory_id": memory.id,
        })

    def _handle_query(self, args: dict) -> str:
        """Handle memory_query tool call."""
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "Missing required parameter: query"})

        mode = args.get("mode", "auto")
        limit = min(args.get("limit", 10), 50)
        tag = args.get("tag")

        # Choose search method
        if mode == "fts":
            results = self._store.search_fts(query, limit=limit)
        elif mode == "vector":
            if not self._config.get("auto_embed"):
                return json.dumps({"error": "Vector search requires auto_embed=true"})
            results = self._store.search_vector(query, limit=limit)
        else:
            results = self._store.search(query, limit=limit)

        # Filter by tag if specified
        if tag and results:
            results = [
                r for r in results
                if tag in (r.memory.metadata.tags or [])
            ]

        if not results:
            return json.dumps({"result": "No relevant memories found.", "count": 0})

        items = []
        for r in results:
            item = {
                "content": r.memory.content,
                "score": round(r.score, 3),
            }
            # Safely access metadata fields
            if r.memory.metadata:
                if hasattr(r.memory.metadata, 'tags'):
                    item["tags"] = r.memory.metadata.tags or []
                if hasattr(r.memory.metadata, 'created_at') and r.memory.metadata.created_at:
                    item["created_at"] = r.memory.metadata.created_at.isoformat()
            items.append(item)

        return json.dumps({"results": items, "count": len(items)})

    def _handle_extract(self, args: dict) -> str:
        """Handle memory_extract tool call."""
        text = args.get("text", "")
        if not text:
            return json.dumps({"error": "Missing required parameter: text"})

        if not self._extractor:
            return json.dumps({"error": "Extractor not initialized"})

        domains = args.get("domains")
        auto_store = args.get("auto_store", True)

        result = self._extractor.extract(text, source="manual_extract")

        # Filter by domains if specified
        memories = result.memories
        if domains:
            memories = [m for m in memories if m.domain.value in domains]

        # Store if auto_store
        stored_count = 0
        if auto_store and self._store:
            for memory in memories:
                if self._guard:
                    validation = self._guard.validate_content(memory.value)
                    if not validation.is_safe:
                        continue

                self._store.add(
                    f"[{memory.domain.value}] {memory.key}: {memory.value}",
                    metadata={
                        "source": "extracted",
                        "domain": memory.domain.value,
                        "confidence": memory.confidence,
                    }
                )
                stored_count += 1

        items = [
            {
                "domain": m.domain.value,
                "key": m.key,
                "value": m.value,
                "confidence": m.confidence,
            }
            for m in memories
        ]

        return json.dumps({
            "extracted": items,
            "count": len(items),
            "stored": stored_count if auto_store else 0,
        })

    def _handle_compress(self, args: dict) -> str:
        """Handle memory_compress tool call."""
        messages = args.get("messages", [])
        if not messages:
            return json.dumps({"error": "Missing required parameter: messages"})

        if not self._compressor:
            return json.dumps({"error": "Compressor not initialized"})

        max_tokens = args.get("max_tokens", 4000)
        mode_str = args.get("mode", "balanced")

        mode_map = {
            "aggressive": CompressionMode.AGGRESSIVE,
            "balanced": CompressionMode.BALANCED,
            "conservative": CompressionMode.CONSERVATIVE,
            "lossless": CompressionMode.LOSSLESS,
        }

        # Create compressor with specified settings
        compressor = ContextCompressor(
            max_tokens=max_tokens,
            mode=mode_map.get(mode_str, CompressionMode.BALANCED),
        )

        result = compressor.compress(messages)

        return json.dumps({
            "compressed_messages": result.messages,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "compression_ratio": round(result.compression_ratio, 2),
            "strategy_used": result.strategy_used,
        })

    def _handle_profile(self, args: dict) -> str:
        """Handle memory_profile tool call."""
        domain = args.get("domain")
        limit = args.get("limit", 5)

        # Get all memories
        all_memories = list(self._store.list(limit=1000))

        if not all_memories:
            return json.dumps({"result": "No memories stored yet."})

        # Group by domain/tag
        by_domain: Dict[str, List[str]] = {}
        for mem in all_memories:
            # Try to extract domain from content
            content = mem.content
            mem_domain = "general"
            if content.startswith("["):
                try:
                    mem_domain = content.split("]")[0][1:].lower()
                    content = content.split("] ", 1)[1] if "] " in content else content
                except:
                    pass

            if domain and mem_domain != domain:
                continue

            if mem_domain not in by_domain:
                by_domain[mem_domain] = []
            if len(by_domain[mem_domain]) < limit:
                by_domain[mem_domain].append(content)

        if not by_domain:
            return json.dumps({"result": "No memories found for specified criteria."})

        return json.dumps({
            "profile": by_domain,
            "total_memories": len(all_memories),
            "domains": list(by_domain.keys()),
        })

    def shutdown(self) -> None:
        """Clean shutdown."""
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

        if self._store:
            self._store.close()
            self._store = None


# ---------------------------------------------------------------------------
# Plugin Registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register Agent Memory Toolkit as a memory provider plugin."""
    try:
        ctx.register_memory_provider(AgentMemoryToolkitProvider())
        logger.debug("Agent Memory Toolkit plugin registered")
    except Exception as e:
        logger.warning(f"Failed to register Agent Memory Toolkit: {e}")


# Allow direct import of provider
__all__ = [
    "AgentMemoryToolkitProvider",
    "register",
    "_load_config",
]
