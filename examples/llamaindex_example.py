"""Example: Using Agent Memory Toolkit with LlamaIndex.

This example demonstrates how to use the AgentMemoryToolkitStore class
as a chat storage backend for LlamaIndex memory and agents.

Requirements:
    pip install agent-memory-toolkit llama-index-core llama-index-llms-openai
"""

from agent_memory_toolkit import MemoryStore
from agent_memory_toolkit.integrations.llamaindex import (
    AgentMemoryToolkitStore,
    AgentMemoryToolkitVectorStore,
)


def basic_chat_store_example():
    """
    Basic example showing chat message storage.
    
    The chat store persists messages across sessions, allowing
    conversation continuity.
    """
    print("=" * 60)
    print("Basic Chat Store Example")
    print("=" * 60)
    
    # Create a persistent memory store
    store = MemoryStore("llamaindex_memory.db")
    
    # Create chat store
    chat_store = AgentMemoryToolkitStore(store=store)
    
    # We need to import LlamaIndex types
    try:
        from llama_index.core.llms import ChatMessage, MessageRole
    except ImportError:
        print("LlamaIndex not installed. Showing API only.")
        store.close()
        return
    
    # Add messages for a user
    user_id = "user_123"
    
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello! I need help with Python."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi! I'd be happy to help with Python."),
        ChatMessage(role=MessageRole.USER, content="How do I read a file?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="You can use open() to read files..."),
    ]
    
    # Set all messages at once
    chat_store.set_messages(user_id, messages)
    print(f"Added {len(messages)} messages for {user_id}")
    
    # Retrieve messages
    retrieved = chat_store.get_messages(user_id)
    print(f"\nRetrieved messages for {user_id}:")
    for msg in retrieved:
        print(f"  [{msg.role}]: {msg.content[:40]}...")
    
    # Add another message
    chat_store.add_message(
        user_id,
        ChatMessage(role=MessageRole.USER, content="What about writing to a file?")
    )
    print(f"\nAdded another message. Total: {len(chat_store.get_messages(user_id))}")
    
    # Delete the last message
    deleted = chat_store.delete_last_message(user_id)
    print(f"Deleted last message: {deleted.content[:40]}...")
    
    store.close()


def multi_user_example():
    """
    Example showing multi-user conversation management.
    
    Each user has their own isolated conversation history.
    """
    print("\n" + "=" * 60)
    print("Multi-User Chat Store Example")
    print("=" * 60)
    
    store = MemoryStore(":memory:")
    chat_store = AgentMemoryToolkitStore(store=store)
    
    try:
        from llama_index.core.llms import ChatMessage, MessageRole
    except ImportError:
        print("LlamaIndex not installed. Showing API only.")
        store.close()
        return
    
    # Simulate multiple users
    users = {
        "alice": [
            ChatMessage(role=MessageRole.USER, content="I'm working on a web app"),
            ChatMessage(role=MessageRole.ASSISTANT, content="What framework?"),
        ],
        "bob": [
            ChatMessage(role=MessageRole.USER, content="Help me with databases"),
            ChatMessage(role=MessageRole.ASSISTANT, content="What database?"),
        ],
        "charlie": [
            ChatMessage(role=MessageRole.USER, content="I need help with ML"),
            ChatMessage(role=MessageRole.ASSISTANT, content="What ML library?"),
        ],
    }
    
    # Add messages for each user
    for user_id, messages in users.items():
        chat_store.set_messages(user_id, messages)
    
    # Get all keys (user IDs)
    keys = chat_store.get_keys()
    print(f"Active users: {keys}")
    
    # Check each user's messages
    for user_id in keys:
        messages = chat_store.get_messages(user_id)
        print(f"\n{user_id}'s conversation ({len(messages)} messages):")
        for msg in messages:
            print(f"  [{msg.role}]: {msg.content}")
    
    # Delete one user's history
    print(f"\nDeleting bob's conversation...")
    chat_store.delete_messages("bob")
    
    print(f"Remaining users: {chat_store.get_keys()}")
    
    store.close()


def vector_store_example():
    """
    Example showing vector store functionality.
    
    This allows semantic search over stored documents/text.
    """
    print("\n" + "=" * 60)
    print("Vector Store Example")
    print("=" * 60)
    
    # Note: For semantic search, we need auto_embed=True
    # This requires sentence-transformers to be installed
    try:
        store = MemoryStore(":memory:", auto_embed=True)
        use_vector = True
    except Exception:
        print("sentence-transformers not available, using FTS only")
        store = MemoryStore(":memory:")
        use_vector = False
    
    vector_store = AgentMemoryToolkitVectorStore(store=store)
    
    # Add some documents
    documents = [
        "Python is a high-level programming language known for its readability.",
        "JavaScript is primarily used for web development and runs in browsers.",
        "Rust provides memory safety without garbage collection.",
        "Machine learning models learn patterns from training data.",
        "Docker containers package applications with their dependencies.",
    ]
    
    print("Adding documents to vector store...")
    for doc in documents:
        node_id = vector_store.add_text(doc)
        print(f"  Added: {doc[:50]}... (ID: {node_id[:8]}...)")
    
    # Search
    queries = [
        "web programming",
        "memory management",
        "AI and data science",
    ]
    
    for query in queries:
        print(f"\nSearch: '{query}'")
        results = vector_store.query(query, limit=2)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['content'][:50]}... (score: {r['score']:.3f})")
    
    store.close()


def llamaindex_memory_buffer_example():
    """
    Example showing integration with LlamaIndex ChatMemoryBuffer.
    
    This demonstrates how to use our chat store with LlamaIndex's
    built-in memory system.
    """
    print("\n" + "=" * 60)
    print("LlamaIndex ChatMemoryBuffer Example")
    print("=" * 60)
    
    try:
        from llama_index.core.memory import ChatMemoryBuffer
        from llama_index.core.llms import ChatMessage, MessageRole
    except ImportError:
        print("LlamaIndex not installed. Showing API only.")
        return
    
    store = MemoryStore(":memory:")
    chat_store = AgentMemoryToolkitStore(store=store)
    
    # Create a ChatMemoryBuffer with our store
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key="user_session_123",
        token_limit=3000,  # Limit context to 3000 tokens
    )
    
    # Simulate conversation turns
    print("Simulating conversation with memory buffer...")
    
    # Add some messages
    memory.put(ChatMessage(role=MessageRole.USER, content="What is Python?"))
    memory.put(ChatMessage(role=MessageRole.ASSISTANT, 
        content="Python is a versatile programming language."))
    
    memory.put(ChatMessage(role=MessageRole.USER, content="What can I do with it?"))
    memory.put(ChatMessage(role=MessageRole.ASSISTANT,
        content="You can do web development, data science, automation, and more."))
    
    # Get the memory buffer
    messages = memory.get()
    print(f"\nCurrent memory buffer ({len(messages)} messages):")
    for msg in messages:
        print(f"  [{msg.role}]: {msg.content[:50]}...")
    
    # The messages are persisted in our store
    print(f"\nPersisted messages in chat store:")
    stored = chat_store.get_messages("user_session_123")
    print(f"  Count: {len(stored)}")
    
    store.close()


def chat_engine_example():
    """
    Example showing integration with a LlamaIndex chat engine.
    
    Note: This example requires an OpenAI API key to run.
    """
    print("\n" + "=" * 60)
    print("LlamaIndex Chat Engine Example")
    print("=" * 60)
    
    try:
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.core.memory import ChatMemoryBuffer
        from llama_index.llms.openai import OpenAI
        import os
        
        if not os.environ.get("OPENAI_API_KEY"):
            print("Skipping: OPENAI_API_KEY not set")
            return
        
        store = MemoryStore(":memory:")
        chat_store = AgentMemoryToolkitStore(store=store)
        
        # Create memory
        memory = ChatMemoryBuffer.from_defaults(
            chat_store=chat_store,
            chat_store_key="chat_engine_session",
        )
        
        # Create chat engine
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
        chat_engine = SimpleChatEngine.from_defaults(
            llm=llm,
            memory=memory,
        )
        
        # Have a conversation
        response = chat_engine.chat("Hi! My name is Alice.")
        print(f"AI: {response.response}")
        
        response = chat_engine.chat("What's my name?")
        print(f"AI: {response.response}")
        
        store.close()
        
    except ImportError as e:
        print(f"Skipping: Required packages not installed ({e})")


if __name__ == "__main__":
    basic_chat_store_example()
    multi_user_example()
    vector_store_example()
    llamaindex_memory_buffer_example()
    chat_engine_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
