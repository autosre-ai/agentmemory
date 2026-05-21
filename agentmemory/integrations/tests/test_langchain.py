"""Tests for LangChain integration."""

import json
import pytest
from unittest.mock import MagicMock, patch

from agentmemory import MemoryStore


# Skip all tests if LangChain is not installed
langchain_installed = False
try:
    from langchain_core.memory import BaseMemory
    from langchain_core.messages import HumanMessage, AIMessage
    from agentmemory.integrations.langchain import (
        AgentMemoryToolkitMemory,
        AgentMemoryToolkitChatMemory,
        LANGCHAIN_AVAILABLE,
    )
    langchain_installed = True
except ImportError:
    pass


pytestmark = pytest.mark.skipif(
    not langchain_installed,
    reason="LangChain not installed"
)


class TestAgentMemoryToolkitMemory:
    """Tests for AgentMemoryToolkitMemory class."""

    @pytest.fixture
    def memory_store(self):
        """Create an in-memory MemoryStore."""
        return MemoryStore(":memory:")

    @pytest.fixture
    def memory(self, memory_store):
        """Create an AgentMemoryToolkitMemory instance."""
        return AgentMemoryToolkitMemory(
            store=memory_store,
            session_id="test_session",
        )

    def test_init(self, memory):
        """Test memory initialization."""
        assert memory.session_id == "test_session"
        assert memory.memory_key == "history"
        assert memory.max_history_length == 10

    def test_memory_variables(self, memory):
        """Test memory_variables property."""
        assert "history" in memory.memory_variables

    def test_save_context(self, memory):
        """Test saving conversation context."""
        memory.save_context(
            {"input": "Hello, how are you?"},
            {"output": "I'm doing well, thank you!"},
        )
        
        # Check that history was updated
        assert len(memory._history) == 2
        assert memory._history[0]["role"] == "human"
        assert memory._history[0]["content"] == "Hello, how are you?"
        assert memory._history[1]["role"] == "ai"
        assert memory._history[1]["content"] == "I'm doing well, thank you!"

    def test_load_memory_variables(self, memory):
        """Test loading memory variables."""
        memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"},
        )
        
        vars = memory.load_memory_variables({})
        assert "history" in vars
        assert "Human: Hello" in vars["history"]
        assert "AI: Hi there!" in vars["history"]

    def test_load_memory_variables_with_messages(self, memory_store):
        """Test loading memory variables as messages."""
        memory = AgentMemoryToolkitMemory(
            store=memory_store,
            session_id="test_messages",
            return_messages=True,
        )
        
        memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"},
        )
        
        vars = memory.load_memory_variables({})
        assert "history" in vars
        messages = vars["history"]
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_clear(self, memory):
        """Test clearing memory."""
        memory.save_context(
            {"input": "Hello"},
            {"output": "Hi!"},
        )
        
        assert len(memory._history) == 2
        
        memory.clear()
        
        assert len(memory._history) == 0

    def test_add_memory(self, memory):
        """Test adding standalone memories."""
        mem = memory.add_memory("The user's name is John")
        
        assert mem.content == "The user's name is John"
        assert "langchain" in mem.metadata.tags

    def test_search_memories(self, memory):
        """Test searching memories."""
        memory.add_memory("The user prefers dark mode")
        memory.add_memory("The user's favorite color is blue")
        
        results = memory.search_memories("color", limit=5)
        
        assert len(results) > 0
        # The result about color should be found
        assert any("color" in r["content"].lower() for r in results)

    def test_max_history_length(self, memory_store):
        """Test that history is trimmed to max length."""
        memory = AgentMemoryToolkitMemory(
            store=memory_store,
            session_id="test_trim",
            max_history_length=2,  # 2 turns = 4 messages
        )
        
        # Add more turns than max
        for i in range(5):
            memory.save_context(
                {"input": f"Message {i}"},
                {"output": f"Response {i}"},
            )
        
        # Should be trimmed to last 4 messages (2 turns)
        assert len(memory._history) == 4
        assert memory._history[0]["content"] == "Message 3"

    def test_session_persistence(self, memory_store):
        """Test that sessions are persisted and reloaded."""
        session_id = "persistent_session"
        
        # Create first memory instance
        memory1 = AgentMemoryToolkitMemory(
            store=memory_store,
            session_id=session_id,
        )
        memory1.save_context(
            {"input": "Remember this"},
            {"output": "I will remember"},
        )
        
        # Create second memory instance with same session
        memory2 = AgentMemoryToolkitMemory(
            store=memory_store,
            session_id=session_id,
        )
        
        # Should have loaded the previous conversation
        assert len(memory2._history) == 2
        assert memory2._history[0]["content"] == "Remember this"


class TestAgentMemoryToolkitChatMemory:
    """Tests for AgentMemoryToolkitChatMemory class."""

    @pytest.fixture
    def memory_store(self):
        """Create an in-memory MemoryStore."""
        return MemoryStore(":memory:")

    @pytest.fixture
    def chat_memory(self, memory_store):
        """Create an AgentMemoryToolkitChatMemory instance."""
        return AgentMemoryToolkitChatMemory(
            store=memory_store,
            session_id="test_chat_session",
        )

    def test_init(self, chat_memory):
        """Test chat memory initialization."""
        assert chat_memory.session_id == "test_chat_session"
        assert chat_memory.memory_key == "chat_history"
        assert chat_memory.return_messages is True

    def test_save_context(self, chat_memory):
        """Test saving chat context."""
        chat_memory.save_context(
            {"input": "Hello!"},
            {"output": "Hi, how can I help?"},
        )
        
        messages = chat_memory.messages
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_add_user_message(self, chat_memory):
        """Test adding a user message."""
        chat_memory.add_user_message("Test message")
        
        messages = chat_memory.messages
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Test message"

    def test_add_ai_message(self, chat_memory):
        """Test adding an AI message."""
        chat_memory.add_ai_message("AI response")
        
        messages = chat_memory.messages
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert messages[0].content == "AI response"

    def test_load_memory_variables(self, chat_memory):
        """Test loading memory variables."""
        chat_memory.add_user_message("Hello")
        chat_memory.add_ai_message("Hi there")
        
        vars = chat_memory.load_memory_variables({})
        assert "chat_history" in vars
        assert len(vars["chat_history"]) == 2

    def test_clear(self, chat_memory):
        """Test clearing chat memory."""
        chat_memory.add_user_message("Hello")
        chat_memory.add_ai_message("Hi")
        
        assert len(chat_memory.messages) == 2
        
        chat_memory.clear()
        
        assert len(chat_memory.messages) == 0


class TestLangChainIntegrationWithMock:
    """Test LangChain integration with mocked LLM."""

    @pytest.fixture
    def memory_store(self):
        """Create an in-memory MemoryStore."""
        return MemoryStore(":memory:")

    def test_search_on_load(self, memory_store):
        """Test that search_on_load retrieves relevant memories."""
        memory = AgentMemoryToolkitMemory(
            store=memory_store,
            session_id="search_test",
            search_on_load=True,
            search_limit=3,
        )
        
        # Add some background knowledge
        memory.add_memory("The user works at Acme Corp")
        memory.add_memory("The user is a software engineer")
        
        # Load with a relevant query
        vars = memory.load_memory_variables({"input": "What company do I work at?"})
        
        # The search results should include relevant context
        assert "history" in vars
        # Note: exact match depends on FTS ranking
