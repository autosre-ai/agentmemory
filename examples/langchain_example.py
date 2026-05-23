"""Example: Using Agent Memory Toolkit with LangChain.

This example demonstrates how to use the AgentMemoryToolkitMemory class
as a drop-in memory backend for LangChain chains and agents.

Requirements:
    pip install agent-memory-toolkit langchain langchain-openai
"""

from agent_memory_toolkit import MemoryStore
from agent_memory_toolkit.integrations.langchain import (
    AgentMemoryToolkitMemory,
    AgentMemoryToolkitChatMemory,
)


def basic_conversation_example():
    """
    Basic example showing conversation memory with persistence.
    
    The memory persists across sessions, so you can reload the same
    conversation later by using the same session_id.
    """
    print("=" * 60)
    print("Basic Conversation Memory Example")
    print("=" * 60)
    
    # Create a persistent memory store
    store = MemoryStore("langchain_memory.db")
    
    # Create memory with a specific session ID
    memory = AgentMemoryToolkitMemory(
        store=store,
        session_id="example_session_1",
        max_history_length=20,
    )
    
    # Simulate a conversation
    conversations = [
        ("What's the weather like?", "I don't have access to real-time weather data."),
        ("Can you remember my name is Alice?", "Of course! I'll remember that your name is Alice."),
        ("What's 2 + 2?", "2 + 2 equals 4."),
    ]
    
    for user_input, ai_response in conversations:
        memory.save_context(
            {"input": user_input},
            {"output": ai_response},
        )
        print(f"User: {user_input}")
        print(f"AI: {ai_response}")
        print()
    
    # Load memory variables
    print("Memory contents:")
    print(memory.load_memory_variables({})["history"])
    print()
    
    # Demonstrate persistence - create a new memory instance
    # with the same session_id
    print("Creating new memory instance with same session_id...")
    memory2 = AgentMemoryToolkitMemory(
        store=store,
        session_id="example_session_1",
    )
    
    print("Loaded conversation history:")
    print(memory2.load_memory_variables({})["history"])
    
    store.close()


def search_with_memory_example():
    """
    Example showing how to search for relevant memories.
    
    This is useful for building agents that can recall specific
    information from past conversations.
    """
    print("\n" + "=" * 60)
    print("Search with Memory Example")
    print("=" * 60)
    
    store = MemoryStore(":memory:")  # In-memory for demo
    
    memory = AgentMemoryToolkitMemory(
        store=store,
        session_id="search_example",
        search_on_load=True,  # Enable automatic search
        search_limit=3,
    )
    
    # Add some background knowledge
    memory.add_memory("Alice works at TechCorp as a software engineer.")
    memory.add_memory("Bob is Alice's manager and has been at the company for 5 years.")
    memory.add_memory("The company uses Python and TypeScript for development.")
    memory.add_memory("Team meetings are held every Monday at 10 AM.")
    
    # Search for relevant information
    print("\nSearching for 'programming languages':")
    results = memory.search_memories("programming languages", limit=2)
    for r in results:
        print(f"  - {r['content']} (score: {r['score']:.3f})")
    
    print("\nSearching for 'team schedule':")
    results = memory.search_memories("team schedule", limit=2)
    for r in results:
        print(f"  - {r['content']} (score: {r['score']:.3f})")
    
    store.close()


def chat_memory_example():
    """
    Example using AgentMemoryToolkitChatMemory for chat-based applications.
    
    This memory class returns LangChain message objects, which work
    well with chat-based LLMs and chains.
    """
    print("\n" + "=" * 60)
    print("Chat Memory Example")
    print("=" * 60)
    
    store = MemoryStore(":memory:")
    
    memory = AgentMemoryToolkitChatMemory(
        store=store,
        session_id="chat_example",
        max_messages=50,
    )
    
    # Add messages directly
    memory.add_user_message("Hello! I'm learning Python.")
    memory.add_ai_message("That's great! Python is a wonderful language. What would you like to learn?")
    memory.add_user_message("Can you explain list comprehensions?")
    memory.add_ai_message("List comprehensions are a concise way to create lists...")
    
    # Get messages
    print("\nChat history (as message objects):")
    for msg in memory.messages:
        role = type(msg).__name__
        print(f"  [{role}]: {msg.content[:50]}...")
    
    # Load memory variables (returns list of BaseMessage objects)
    print("\nMemory variables:")
    vars = memory.load_memory_variables({})
    print(f"  Key: {memory.memory_key}")
    print(f"  Messages: {len(vars[memory.memory_key])} messages")
    
    store.close()


def langchain_chain_integration_example():
    """
    Example showing integration with LangChain ConversationChain.
    
    Note: This example requires an OpenAI API key to run.
    """
    print("\n" + "=" * 60)
    print("LangChain Chain Integration Example")
    print("=" * 60)
    
    try:
        from langchain.chains import ConversationChain
        from langchain_openai import ChatOpenAI
        import os
        
        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("Skipping: OPENAI_API_KEY not set")
            return
        
        store = MemoryStore(":memory:")
        
        # Create memory
        memory = AgentMemoryToolkitMemory(
            store=store,
            session_id="langchain_chain_example",
            return_messages=False,  # ConversationChain expects string history
        )
        
        # Create LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create chain with our memory
        chain = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True,
        )
        
        # Run the chain
        response = chain.invoke({"input": "Hi! My name is Alice."})
        print(f"Response: {response['response']}")
        
        response = chain.invoke({"input": "What's my name?"})
        print(f"Response: {response['response']}")
        
        store.close()
        
    except ImportError as e:
        print(f"Skipping: Required packages not installed ({e})")


if __name__ == "__main__":
    basic_conversation_example()
    search_with_memory_example()
    chat_memory_example()
    langchain_chain_integration_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
