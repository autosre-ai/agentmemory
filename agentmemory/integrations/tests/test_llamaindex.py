"""Tests for LlamaIndex integration."""

import json
import pytest
from unittest.mock import MagicMock, patch

from agentmemory import MemoryStore


# Skip all tests if LlamaIndex is not installed
llamaindex_installed = False
try:
    from llama_index.core.storage.chat_store import BaseChatStore
    from llama_index.core.llms import ChatMessage, MessageRole
    from agentmemory.integrations.llamaindex import (
        AgentMemoryToolkitStore,
        AgentMemoryToolkitChatStore,
        AgentMemoryToolkitVectorStore,
        LLAMAINDEX_AVAILABLE,
    )
    llamaindex_installed = True
except ImportError:
    pass


pytestmark = pytest.mark.skipif(
    not llamaindex_installed,
    reason="LlamaIndex not installed"
)


class TestAgentMemoryToolkitStore:
    """Tests for AgentMemoryToolkitStore class."""

    @pytest.fixture
    def memory_store(self):
        """Create an in-memory MemoryStore."""
        return MemoryStore(":memory:")

    @pytest.fixture
    def chat_store(self, memory_store):
        """Create an AgentMemoryToolkitStore instance."""
        return AgentMemoryToolkitStore(store=memory_store)

    def test_init(self, chat_store):
        """Test chat store initialization."""
        assert chat_store._tag_prefix == "llamaindex_chat"

    def test_class_name(self, chat_store):
        """Test class_name method."""
        assert chat_store.class_name() == "AgentMemoryToolkitStore"

    def test_add_message(self, chat_store):
        """Test adding a message."""
        msg = ChatMessage(role=MessageRole.USER, content="Hello!")
        chat_store.add_message("user_123", msg)
        
        messages = chat_store.get_messages("user_123")
        assert len(messages) == 1
        assert messages[0].content == "Hello!"
        assert messages[0].role == MessageRole.USER

    def test_set_messages(self, chat_store):
        """Test setting messages (overwrites existing)."""
        # Add initial message
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.USER, content="First message")
        )
        
        # Set new messages (should overwrite)
        new_messages = [
            ChatMessage(role=MessageRole.USER, content="New message 1"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response 1"),
        ]
        chat_store.set_messages("user_123", new_messages)
        
        messages = chat_store.get_messages("user_123")
        assert len(messages) == 2
        assert messages[0].content == "New message 1"
        assert messages[1].content == "Response 1"

    def test_get_messages_empty(self, chat_store):
        """Test getting messages for a key with no messages."""
        messages = chat_store.get_messages("nonexistent_key")
        assert len(messages) == 0

    def test_delete_messages(self, chat_store):
        """Test deleting all messages for a key."""
        # Add messages
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.USER, content="Message 1")
        )
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.ASSISTANT, content="Response 1")
        )
        
        # Delete and verify return value
        deleted = chat_store.delete_messages("user_123")
        assert len(deleted) == 2
        
        # Verify they're gone
        messages = chat_store.get_messages("user_123")
        assert len(messages) == 0

    def test_delete_messages_nonexistent(self, chat_store):
        """Test deleting messages for a nonexistent key."""
        result = chat_store.delete_messages("nonexistent_key")
        assert result is None

    def test_delete_message_by_index(self, chat_store):
        """Test deleting a specific message by index."""
        # Add messages
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.USER, content="Message 0")
        )
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.ASSISTANT, content="Message 1")
        )
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.USER, content="Message 2")
        )
        
        # Delete middle message
        deleted = chat_store.delete_message("user_123", 1)
        assert deleted.content == "Message 1"
        
        # Verify remaining messages
        messages = chat_store.get_messages("user_123")
        assert len(messages) == 2
        assert messages[0].content == "Message 0"
        assert messages[1].content == "Message 2"

    def test_delete_message_invalid_index(self, chat_store):
        """Test deleting a message with invalid index."""
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.USER, content="Only message")
        )
        
        result = chat_store.delete_message("user_123", 5)
        assert result is None

    def test_delete_last_message(self, chat_store):
        """Test deleting the last message."""
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.USER, content="First")
        )
        chat_store.add_message(
            "user_123",
            ChatMessage(role=MessageRole.ASSISTANT, content="Last")
        )
        
        deleted = chat_store.delete_last_message("user_123")
        assert deleted.content == "Last"
        
        messages = chat_store.get_messages("user_123")
        assert len(messages) == 1
        assert messages[0].content == "First"

    def test_get_keys(self, chat_store):
        """Test getting all conversation keys."""
        # Add messages for multiple keys
        chat_store.add_message(
            "user_1",
            ChatMessage(role=MessageRole.USER, content="Hello from user 1")
        )
        chat_store.add_message(
            "user_2",
            ChatMessage(role=MessageRole.USER, content="Hello from user 2")
        )
        
        keys = chat_store.get_keys()
        assert "user_1" in keys
        assert "user_2" in keys

    def test_message_ordering(self, chat_store):
        """Test that messages maintain their order."""
        messages_to_add = [
            ChatMessage(role=MessageRole.USER, content=f"Message {i}")
            for i in range(5)
        ]
        
        for msg in messages_to_add:
            chat_store.add_message("user_123", msg)
        
        retrieved = chat_store.get_messages("user_123")
        for i, msg in enumerate(retrieved):
            assert msg.content == f"Message {i}"

    def test_multiple_users_isolation(self, chat_store):
        """Test that different users' messages are isolated."""
        chat_store.add_message(
            "user_1",
            ChatMessage(role=MessageRole.USER, content="User 1 message")
        )
        chat_store.add_message(
            "user_2",
            ChatMessage(role=MessageRole.USER, content="User 2 message")
        )
        
        user1_msgs = chat_store.get_messages("user_1")
        user2_msgs = chat_store.get_messages("user_2")
        
        assert len(user1_msgs) == 1
        assert len(user2_msgs) == 1
        assert user1_msgs[0].content == "User 1 message"
        assert user2_msgs[0].content == "User 2 message"


class TestAgentMemoryToolkitChatStoreAlias:
    """Test that AgentMemoryToolkitChatStore is an alias for AgentMemoryToolkitStore."""

    def test_alias(self):
        """Test that alias points to the same class."""
        assert AgentMemoryToolkitChatStore is AgentMemoryToolkitStore


class TestAgentMemoryToolkitVectorStore:
    """Tests for AgentMemoryToolkitVectorStore class."""

    @pytest.fixture
    def memory_store(self):
        """Create an in-memory MemoryStore."""
        return MemoryStore(":memory:")

    @pytest.fixture
    def vector_store(self, memory_store):
        """Create an AgentMemoryToolkitVectorStore instance."""
        return AgentMemoryToolkitVectorStore(store=memory_store)

    def test_add_text(self, vector_store):
        """Test adding text to the vector store."""
        node_id = vector_store.add_text("The capital of France is Paris")
        assert node_id is not None

    def test_get_text(self, vector_store):
        """Test getting text by node ID."""
        node_id = vector_store.add_text("Test content")
        
        content = vector_store.get_text(node_id)
        assert content == "Test content"

    def test_get_text_nonexistent(self, vector_store):
        """Test getting text with nonexistent ID."""
        result = vector_store.get_text("nonexistent_id")
        assert result is None

    def test_delete_text(self, vector_store):
        """Test deleting text from the vector store."""
        node_id = vector_store.add_text("To be deleted")
        
        result = vector_store.delete_text(node_id)
        assert result is True
        
        # Verify it's gone
        content = vector_store.get_text(node_id)
        assert content is None

    def test_delete_text_nonexistent(self, vector_store):
        """Test deleting nonexistent text."""
        result = vector_store.delete_text("nonexistent_id")
        assert result is False

    def test_query(self, vector_store):
        """Test querying the vector store."""
        vector_store.add_text("The capital of France is Paris")
        vector_store.add_text("Python is a programming language")
        vector_store.add_text("The Eiffel Tower is in Paris")
        
        results = vector_store.query("French capital city", limit=3)
        
        assert len(results) > 0
        # Results should contain id, content, score, metadata
        assert "id" in results[0]
        assert "content" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]

    def test_add_text_with_metadata(self, vector_store):
        """Test adding text with custom metadata."""
        node_id = vector_store.add_text(
            "Test content",
            metadata={"source": "test", "page": 1}
        )
        
        content = vector_store.get_text(node_id)
        assert content == "Test content"
