"""LlamaIndex integration for Agent Memory Toolkit.

This module provides LlamaIndex-compatible memory and storage classes that use
the Agent Memory Toolkit's MemoryStore as a backend.

Classes:
    AgentMemoryToolkitStore: A LlamaIndex BaseChatStore implementation for chat history.
    AgentMemoryToolkitChatStore: Alias for AgentMemoryToolkitStore.

Example:
    >>> from llama_index.core.memory import ChatMemoryBuffer
    >>> from agentmemory import MemoryStore
    >>> from agentmemory.integrations.llamaindex import AgentMemoryToolkitStore
    >>> 
    >>> store = MemoryStore("agent_memory.db")
    >>> chat_store = AgentMemoryToolkitStore(store=store)
    >>> 
    >>> memory = ChatMemoryBuffer.from_defaults(
    ...     chat_store=chat_store,
    ...     chat_store_key="user_123",
    ... )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from llama_index.core.storage.chat_store import BaseChatStore
    from llama_index.core.llms import ChatMessage, MessageRole
    
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    BaseChatStore = object  # type: ignore
    ChatMessage = None  # type: ignore
    MessageRole = None  # type: ignore
    
    # Define stubs for type checking
    if TYPE_CHECKING:
        from llama_index.core.storage.chat_store import BaseChatStore
        from llama_index.core.llms import ChatMessage, MessageRole

from agentmemory.store import MemoryStore, Memory, MemoryMetadata

logger = logging.getLogger(__name__)


def _check_llamaindex_available():
    """Raise ImportError if LlamaIndex is not available."""
    if not LLAMAINDEX_AVAILABLE:
        raise ImportError(
            "LlamaIndex is required for this integration. "
            "Install it with: pip install llama-index-core"
        )


class AgentMemoryToolkitStore(BaseChatStore):
    """
    LlamaIndex BaseChatStore implementation using Agent Memory Toolkit as backend.

    This chat store persists chat messages to the MemoryStore, allowing for
    searchable, versioned conversation history.

    Attributes:
        store: The underlying MemoryStore instance.
        tag_prefix: Prefix for tags used to identify chat messages.

    Example:
        >>> from llama_index.core.memory import ChatMemoryBuffer
        >>> from agentmemory import MemoryStore
        >>> from agentmemory.integrations.llamaindex import AgentMemoryToolkitStore
        >>> 
        >>> store = MemoryStore("agent_memory.db")
        >>> chat_store = AgentMemoryToolkitStore(store=store)
        >>> 
        >>> # Use with ChatMemoryBuffer
        >>> memory = ChatMemoryBuffer.from_defaults(
        ...     chat_store=chat_store,
        ...     chat_store_key="user_123",
        ... )
        >>> 
        >>> # Or use directly
        >>> chat_store.add_message("user_123", ChatMessage(role="user", content="Hello"))
        >>> messages = chat_store.get_messages("user_123")
    """

    def __init__(
        self,
        store: MemoryStore,
        tag_prefix: str = "llamaindex_chat",
    ):
        """
        Initialize the chat store.

        Args:
            store: The MemoryStore instance to use as backend.
            tag_prefix: Prefix for tags used to identify chat messages.
        """
        _check_llamaindex_available()
        self._store = store
        self._tag_prefix = tag_prefix
        # Cache for memory IDs per key
        self._key_memory_ids: Dict[str, List[str]] = {}

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for serialization."""
        return "AgentMemoryToolkitStore"

    def _get_key_tag(self, key: str) -> str:
        """Get the tag for a specific chat key."""
        return f"{self._tag_prefix}:{key}"

    def _message_to_dict(self, message: ChatMessage) -> Dict[str, Any]:
        """Convert a ChatMessage to a dictionary."""
        return {
            "role": message.role.value if hasattr(message.role, 'value') else str(message.role),
            "content": message.content,
            "additional_kwargs": getattr(message, "additional_kwargs", {}),
        }

    def _dict_to_message(self, data: Dict[str, Any]) -> ChatMessage:
        """Convert a dictionary to a ChatMessage."""
        role_str = data.get("role", "user")
        
        # Map string role to MessageRole enum
        role_mapping = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "function": MessageRole.FUNCTION,
            "tool": MessageRole.TOOL,
            "chatbot": MessageRole.CHATBOT,
            "model": MessageRole.MODEL,
        }
        
        role = role_mapping.get(role_str, MessageRole.USER)
        
        return ChatMessage(
            role=role,
            content=data.get("content", ""),
            additional_kwargs=data.get("additional_kwargs", {}),
        )

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """
        Set messages for a key, overwriting any existing messages.

        Args:
            key: The conversation key (e.g., user_id or session_id).
            messages: List of ChatMessage objects to store.
        """
        # First, delete existing messages for this key
        self.delete_messages(key)
        
        # Add new messages
        memory_ids = []
        for i, message in enumerate(messages):
            memory = self._store.add(
                content=json.dumps(self._message_to_dict(message)),
                metadata=MemoryMetadata(
                    source="llamaindex_chat",
                    tags=[self._get_key_tag(key), "chat_message"],
                    extra={"sequence": i},
                ),
            )
            memory_ids.append(memory.id)
        
        self._key_memory_ids[key] = memory_ids

    def get_messages(self, key: str) -> List[ChatMessage]:
        """
        Get all messages for a key.

        Args:
            key: The conversation key.

        Returns:
            List of ChatMessage objects, ordered by creation time.
        """
        memories = self._store.list(tag=self._get_key_tag(key), limit=1000)
        
        # Sort by creation time to maintain order
        memories.sort(key=lambda m: m.created_at)
        
        messages = []
        memory_ids = []
        
        for memory in memories:
            try:
                data = json.loads(memory.content)
                messages.append(self._dict_to_message(data))
                memory_ids.append(memory.id)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse message from memory {memory.id}")
                continue
        
        # Update cache
        self._key_memory_ids[key] = memory_ids
        
        return messages

    def add_message(self, key: str, message: ChatMessage) -> None:
        """
        Add a message for a key.

        Args:
            key: The conversation key.
            message: The ChatMessage to add.
        """
        # Get current sequence number
        if key not in self._key_memory_ids:
            self.get_messages(key)  # Load existing messages
        
        sequence = len(self._key_memory_ids.get(key, []))
        
        memory = self._store.add(
            content=json.dumps(self._message_to_dict(message)),
            metadata=MemoryMetadata(
                source="llamaindex_chat",
                tags=[self._get_key_tag(key), "chat_message"],
                extra={"sequence": sequence},
            ),
        )
        
        if key not in self._key_memory_ids:
            self._key_memory_ids[key] = []
        self._key_memory_ids[key].append(memory.id)

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """
        Delete all messages for a key.

        Args:
            key: The conversation key.

        Returns:
            The deleted messages, or None if no messages existed.
        """
        # Get existing messages first
        memories = self._store.list(tag=self._get_key_tag(key), limit=1000)
        
        if not memories:
            return None
        
        # Parse messages before deletion
        messages = []
        for memory in memories:
            try:
                data = json.loads(memory.content)
                messages.append(self._dict_to_message(data))
            except json.JSONDecodeError:
                continue
        
        # Delete memories
        for memory in memories:
            try:
                self._store.delete(memory.id)
            except Exception as e:
                logger.warning(f"Failed to delete memory {memory.id}: {e}")
        
        # Clear cache
        self._key_memory_ids.pop(key, None)
        
        return messages if messages else None

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Delete a specific message by index.

        Args:
            key: The conversation key.
            idx: The index of the message to delete.

        Returns:
            The deleted message, or None if not found.
        """
        # Load messages if not cached
        if key not in self._key_memory_ids:
            self.get_messages(key)
        
        memory_ids = self._key_memory_ids.get(key, [])
        
        if idx < 0 or idx >= len(memory_ids):
            return None
        
        memory_id = memory_ids[idx]
        
        try:
            memory = self._store.get(memory_id)
            data = json.loads(memory.content)
            message = self._dict_to_message(data)
            
            self._store.delete(memory_id)
            self._key_memory_ids[key].pop(idx)
            
            return message
        except Exception as e:
            logger.warning(f"Failed to delete message at index {idx}: {e}")
            return None

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """
        Delete the last message for a key.

        Args:
            key: The conversation key.

        Returns:
            The deleted message, or None if no messages existed.
        """
        if key not in self._key_memory_ids:
            self.get_messages(key)
        
        memory_ids = self._key_memory_ids.get(key, [])
        
        if not memory_ids:
            return None
        
        return self.delete_message(key, len(memory_ids) - 1)

    def get_keys(self) -> List[str]:
        """
        Get all conversation keys.

        Returns:
            List of all conversation keys that have messages.
        """
        # List all memories with our tag prefix
        memories = self._store.list(limit=10000)
        
        keys = set()
        for memory in memories:
            for tag in memory.metadata.tags:
                if tag.startswith(f"{self._tag_prefix}:"):
                    key = tag[len(f"{self._tag_prefix}:"):]
                    keys.add(key)
        
        return list(keys)


# Alias for consistency with naming conventions
AgentMemoryToolkitChatStore = AgentMemoryToolkitStore


class AgentMemoryToolkitVectorStore:
    """
    A vector store adapter for LlamaIndex that uses Agent Memory Toolkit.

    This class provides vector storage capabilities for LlamaIndex nodes
    using the MemoryStore's vector search functionality.

    Note: This is a basic implementation. For full LlamaIndex VectorStore
    compatibility, you may need to extend the BaseVectorStore class.

    Example:
        >>> from agentmemory import MemoryStore
        >>> from agentmemory.integrations.llamaindex import AgentMemoryToolkitVectorStore
        >>> 
        >>> store = MemoryStore("agent_memory.db", auto_embed=True)
        >>> vector_store = AgentMemoryToolkitVectorStore(store=store)
        >>> 
        >>> # Add text
        >>> node_id = vector_store.add_text("The capital of France is Paris")
        >>> 
        >>> # Search
        >>> results = vector_store.query("What is the capital of France?")
    """

    def __init__(
        self,
        store: MemoryStore,
        namespace: str = "llamaindex_nodes",
    ):
        """
        Initialize the vector store.

        Args:
            store: The MemoryStore instance to use as backend.
            namespace: Namespace tag for organizing nodes.
        """
        # Note: VectorStore doesn't require LlamaIndex to be installed
        # since it just wraps our MemoryStore
        self._store = store
        self._namespace = namespace

    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        Add text to the vector store.

        Args:
            text: The text content to add.
            metadata: Optional metadata dictionary.
            node_id: Optional specific ID for the node.

        Returns:
            The ID of the stored memory/node.
        """
        meta = MemoryMetadata(
            source="llamaindex_vectorstore",
            tags=[f"namespace:{self._namespace}", "node"],
            extra=metadata or {},
        )
        
        memory = self._store.add(content=text, metadata=meta)
        return memory.id

    def delete_text(self, node_id: str) -> bool:
        """
        Delete a text/node from the vector store.

        Args:
            node_id: The ID of the node to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            self._store.delete(node_id)
            return True
        except Exception:
            return False

    def query(
        self,
        query_text: str,
        limit: int = 10,
        search_method: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar texts.

        Args:
            query_text: The query text.
            limit: Maximum number of results.
            search_method: Search method ("fts", "vector", "hybrid", "auto").

        Returns:
            List of dictionaries with id, content, score, and metadata.
        """
        results = self._store.search(
            query=query_text,
            limit=limit,
            method=search_method,
        )
        
        return [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "score": r.score,
                "metadata": r.memory.metadata.to_dict(),
            }
            for r in results
        ]

    def get_text(self, node_id: str) -> Optional[str]:
        """
        Get text by node ID.

        Args:
            node_id: The ID of the node.

        Returns:
            The text content, or None if not found.
        """
        try:
            memory = self._store.get(node_id)
            return memory.content
        except Exception:
            return None
