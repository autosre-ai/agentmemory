"""LangChain integration for Agent Memory Toolkit.

This module provides LangChain-compatible memory classes that use the
Agent Memory Toolkit's MemoryStore as a backend.

Classes:
    AgentMemoryToolkitMemory: A LangChain BaseMemory implementation for general use.
    AgentMemoryToolkitChatMemory: A chat-focused memory with message history support.

Example:
    >>> from langchain.chains import ConversationChain
    >>> from langchain_openai import ChatOpenAI
    >>> from agent_memory_toolkit import MemoryStore
    >>> from agent_memory_toolkit.integrations.langchain import AgentMemoryToolkitMemory
    >>> 
    >>> store = MemoryStore("agent_memory.db")
    >>> memory = AgentMemoryToolkitMemory(store=store)
    >>> 
    >>> llm = ChatOpenAI()
    >>> chain = ConversationChain(llm=llm, memory=memory)
    >>> response = chain.invoke({"input": "Hello!"})
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from pydantic import Field, PrivateAttr

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseMemory = object  # type: ignore
    AIMessage = None  # type: ignore
    BaseMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    
    # Define a stub PrivateAttr for when LangChain is not available
    def PrivateAttr(default=None, default_factory=None):  # type: ignore
        return default if default is not None else (default_factory() if default_factory else None)
    
    # Define stubs for type checking
    if TYPE_CHECKING:
        from langchain_core.memory import BaseMemory
        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
        )

from agent_memory_toolkit.store import MemoryStore, Memory, MemoryMetadata

logger = logging.getLogger(__name__)


def _check_langchain_available():
    """Raise ImportError if LangChain is not available."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install it with: pip install langchain-core"
        )


class AgentMemoryToolkitMemory(BaseMemory):
    """
    LangChain BaseMemory implementation using Agent Memory Toolkit as backend.

    This memory class stores conversation context and retrieved facts in the
    MemoryStore, allowing for persistent, searchable memory across sessions.

    Attributes:
        input_key: The key to use for input in conversation history.
        output_key: The key to use for output in conversation history.
        memory_key: The key to return memory variables under.
        return_messages: Whether to return messages as a list of BaseMessage objects.
        max_history_length: Maximum number of conversation turns to keep.
        search_on_load: Whether to search for relevant memories when loading.
        search_limit: Maximum number of search results to include.

    Example:
        >>> from agent_memory_toolkit import MemoryStore
        >>> from agent_memory_toolkit.integrations.langchain import AgentMemoryToolkitMemory
        >>> 
        >>> store = MemoryStore("agent_memory.db")
        >>> memory = AgentMemoryToolkitMemory(store=store)
        >>> 
        >>> # Save context
        >>> memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
        >>> 
        >>> # Load memory variables
        >>> vars = memory.load_memory_variables({"input": "What did I say?"})
    """

    # Pydantic fields
    input_key: str = "input"
    output_key: str = "output"
    memory_key: str = "history"
    return_messages: bool = False
    max_history_length: int = 10
    search_on_load: bool = False
    search_limit: int = 5
    session_id: str = "default"

    # Private attributes (not Pydantic fields)
    _store: MemoryStore = PrivateAttr()
    _history: List[Dict[str, str]] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        store: MemoryStore,
        input_key: str = "input",
        output_key: str = "output",
        memory_key: str = "history",
        return_messages: bool = False,
        max_history_length: int = 10,
        search_on_load: bool = False,
        search_limit: int = 5,
        session_id: str = "default",
        **kwargs,
    ):
        """
        Initialize the memory.

        Args:
            store: The MemoryStore instance to use as backend.
            input_key: The key to use for input in conversation history.
            output_key: The key to use for output in conversation history.
            memory_key: The key to return memory variables under.
            return_messages: Whether to return messages as BaseMessage objects.
            max_history_length: Maximum number of conversation turns to keep.
            search_on_load: Whether to search for relevant memories when loading.
            search_limit: Maximum number of search results to include.
            session_id: Unique identifier for this conversation session.
        """
        _check_langchain_available()
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            memory_key=memory_key,
            return_messages=return_messages,
            max_history_length=max_history_length,
            search_on_load=search_on_load,
            search_limit=search_limit,
            session_id=session_id,
            **kwargs,
        )
        self._store = store
        self._history = []
        self._load_session_history()

    def _load_session_history(self) -> None:
        """Load conversation history from the store for this session."""
        # Search for memories with this session_id in tags
        memories = self._store.list(tag=f"session:{self.session_id}", limit=100)
        
        # Sort by creation time
        memories.sort(key=lambda m: m.created_at)
        
        # Parse conversation turns from memories
        for memory in memories:
            try:
                data = json.loads(memory.content)
                if "role" in data and "content" in data:
                    self._history.append({
                        "role": data["role"],
                        "content": data["content"],
                        "memory_id": memory.id,
                    })
            except json.JSONDecodeError:
                # Not a conversation memory, skip
                continue

        # Trim to max length
        if len(self._history) > self.max_history_length * 2:
            self._history = self._history[-(self.max_history_length * 2):]

    @property
    def memory_variables(self) -> List[str]:
        """Return the list of memory variable keys."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return memory variables to be used in prompts.

        Args:
            inputs: The current inputs to the chain/agent.

        Returns:
            Dictionary containing memory variables.
        """
        # Get conversation history
        history = self._format_history()
        
        # Optionally search for relevant memories
        if self.search_on_load and self.input_key in inputs:
            query = inputs[self.input_key]
            search_results = self._store.search(query, limit=self.search_limit)
            
            # Add relevant facts to context
            if search_results:
                facts = "\n".join([
                    f"- {r.memory.content}" for r in search_results
                    if not r.memory.metadata.tags or 
                    f"session:{self.session_id}" not in r.memory.metadata.tags
                ])
                if facts:
                    history = f"Relevant context:\n{facts}\n\n{history}"

        if self.return_messages:
            return {self.memory_key: self._get_messages()}
        
        return {self.memory_key: history}

    def _format_history(self) -> str:
        """Format conversation history as a string."""
        lines = []
        for turn in self._history:
            role = "Human" if turn["role"] == "human" else "AI"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def _get_messages(self) -> List[BaseMessage]:
        """Get conversation history as LangChain messages."""
        messages = []
        for turn in self._history:
            if turn["role"] == "human":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "ai":
                messages.append(AIMessage(content=turn["content"]))
            elif turn["role"] == "system":
                messages.append(SystemMessage(content=turn["content"]))
        return messages

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save the context from this conversation turn.

        Args:
            inputs: The inputs to the chain/agent.
            outputs: The outputs from the chain/agent.
        """
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")

        # Save human input
        human_memory = self._store.add(
            content=json.dumps({"role": "human", "content": input_str}),
            metadata=MemoryMetadata(
                source="langchain_conversation",
                tags=[f"session:{self.session_id}", "conversation"],
            ),
        )
        self._history.append({
            "role": "human",
            "content": input_str,
            "memory_id": human_memory.id,
        })

        # Save AI output
        ai_memory = self._store.add(
            content=json.dumps({"role": "ai", "content": output_str}),
            metadata=MemoryMetadata(
                source="langchain_conversation",
                tags=[f"session:{self.session_id}", "conversation"],
            ),
        )
        self._history.append({
            "role": "ai",
            "content": output_str,
            "memory_id": ai_memory.id,
        })

        # Trim history if needed
        if len(self._history) > self.max_history_length * 2:
            self._history = self._history[-(self.max_history_length * 2):]

    def clear(self) -> None:
        """Clear the conversation history for this session."""
        # Delete memories for this session from the store
        for turn in self._history:
            if "memory_id" in turn:
                try:
                    self._store.delete(turn["memory_id"])
                except Exception:
                    pass  # Memory might already be deleted
        
        self._history = []

    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """
        Add a standalone memory (not conversation history).

        This is useful for storing facts, learned information, or other
        context that should be searchable but isn't part of the conversation.

        Args:
            content: The memory content.
            metadata: Optional metadata dictionary.

        Returns:
            The created Memory object.
        """
        meta = MemoryMetadata.from_dict(metadata or {})
        if "langchain" not in meta.tags:
            meta.tags.append("langchain")
        return self._store.add(content=content, metadata=meta)

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        method: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories.

        Args:
            query: The search query.
            limit: Maximum number of results.
            method: Search method ("fts", "vector", "hybrid", or "auto").

        Returns:
            List of dictionaries containing memory content and metadata.
        """
        results = self._store.search(query, limit=limit, method=method)
        return [
            {
                "content": r.memory.content,
                "score": r.score,
                "metadata": r.memory.metadata.to_dict(),
                "id": r.memory.id,
            }
            for r in results
        ]


class AgentMemoryToolkitChatMemory(BaseMemory):
    """
    LangChain chat-focused memory with full message history support.

    This memory class maintains a full chat message history and integrates
    with LangChain's chat models and chat-based chains.

    Attributes:
        memory_key: The key to return messages under.
        max_messages: Maximum number of messages to keep.
        human_prefix: Prefix for human messages when formatting.
        ai_prefix: Prefix for AI messages when formatting.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain.chains import ConversationChain
        >>> from agent_memory_toolkit import MemoryStore
        >>> from agent_memory_toolkit.integrations.langchain import AgentMemoryToolkitChatMemory
        >>> 
        >>> store = MemoryStore("agent_memory.db")
        >>> memory = AgentMemoryToolkitChatMemory(store=store, session_id="chat_123")
        >>> 
        >>> llm = ChatOpenAI()
        >>> chain = ConversationChain(llm=llm, memory=memory)
    """

    memory_key: str = "chat_history"
    return_messages: bool = True
    max_messages: int = 50
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    session_id: str = "default"

    _store: MemoryStore = PrivateAttr()
    _messages: List[BaseMessage] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        store: MemoryStore,
        memory_key: str = "chat_history",
        max_messages: int = 50,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        session_id: str = "default",
        **kwargs,
    ):
        """
        Initialize the chat memory.

        Args:
            store: The MemoryStore instance to use as backend.
            memory_key: The key to return messages under.
            max_messages: Maximum number of messages to keep.
            human_prefix: Prefix for human messages when formatting.
            ai_prefix: Prefix for AI messages when formatting.
            session_id: Unique identifier for this chat session.
        """
        _check_langchain_available()
        super().__init__(
            memory_key=memory_key,
            return_messages=True,
            max_messages=max_messages,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            session_id=session_id,
            **kwargs,
        )
        self._store = store
        self._messages = []
        self._load_session_messages()

    def _load_session_messages(self) -> None:
        """Load chat messages from the store for this session."""
        memories = self._store.list(tag=f"chat_session:{self.session_id}", limit=self.max_messages)
        memories.sort(key=lambda m: m.created_at)
        
        for memory in memories:
            try:
                data = json.loads(memory.content)
                msg_type = data.get("type", "human")
                content = data.get("content", "")
                
                if msg_type == "human":
                    self._messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    self._messages.append(AIMessage(content=content))
                elif msg_type == "system":
                    self._messages.append(SystemMessage(content=content))
            except json.JSONDecodeError:
                continue

    @property
    def memory_variables(self) -> List[str]:
        """Return the list of memory variable keys."""
        return [self.memory_key]

    @property
    def messages(self) -> List[BaseMessage]:
        """Return the current list of messages."""
        return self._messages.copy()

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return memory variables to be used in prompts."""
        return {self.memory_key: self._messages.copy()}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context from this conversation turn."""
        input_key = "input"
        output_key = "output"
        
        # Find actual keys
        for key in inputs:
            if key not in ["stop", "callbacks"]:
                input_key = key
                break
        for key in outputs:
            output_key = key
            break

        input_str = inputs.get(input_key, "")
        output_str = outputs.get(output_key, "")

        # Save human message
        self._store.add(
            content=json.dumps({"type": "human", "content": input_str}),
            metadata=MemoryMetadata(
                source="langchain_chat",
                tags=[f"chat_session:{self.session_id}", "chat_message"],
            ),
        )
        self._messages.append(HumanMessage(content=input_str))

        # Save AI message
        self._store.add(
            content=json.dumps({"type": "ai", "content": output_str}),
            metadata=MemoryMetadata(
                source="langchain_chat",
                tags=[f"chat_session:{self.session_id}", "chat_message"],
            ),
        )
        self._messages.append(AIMessage(content=output_str))

        # Trim if needed
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def add_user_message(self, message: str) -> None:
        """Add a user message to history."""
        self._store.add(
            content=json.dumps({"type": "human", "content": message}),
            metadata=MemoryMetadata(
                source="langchain_chat",
                tags=[f"chat_session:{self.session_id}", "chat_message"],
            ),
        )
        self._messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to history."""
        self._store.add(
            content=json.dumps({"type": "ai", "content": message}),
            metadata=MemoryMetadata(
                source="langchain_chat",
                tags=[f"chat_session:{self.session_id}", "chat_message"],
            ),
        )
        self._messages.append(AIMessage(content=message))

    def clear(self) -> None:
        """Clear the chat history for this session."""
        # Note: This clears the in-memory messages but not the persisted ones
        # To fully clear, you would need to delete from the store
        self._messages = []
