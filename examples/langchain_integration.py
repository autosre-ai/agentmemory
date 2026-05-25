#!/usr/bin/env python3
"""
LangChain Integration Example - Agent Memory Toolkit

Demonstrates practical LangChain integration patterns:
1. Persistent Conversation Memory
2. Memory-Augmented Agents
3. RAG with Memory Context
4. Custom Memory Tools

Requirements:
    pip install agent-memory-toolkit langchain langchain-openai
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_memory_toolkit import MemoryStore
from agent_memory_toolkit.integrations.langchain import (
    AgentMemoryToolkitMemory,
    AgentMemoryToolkitChatMemory,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Example 1: Persistent Conversation Chain
# =============================================================================

def example_conversation_chain():
    """
    Create a LangChain conversation chain with persistent memory.
    
    The memory persists across Python sessions, so you can pick up
    conversations where you left off.
    """
    print_header("1. PERSISTENT CONVERSATION CHAIN")
    
    # Create a persistent store (use file path for persistence)
    store = MemoryStore(":memory:")  # Use "conversations.db" for persistence
    
    # Create memory with a specific session ID
    memory = AgentMemoryToolkitMemory(
        store=store,
        session_id="langchain_demo_session",
        max_history_length=50,  # Keep last 50 exchanges
    )
    
    # Simulate a conversation (no API key needed for demo)
    print("\n  Simulating a conversation...")
    
    exchanges = [
        ("Hi! I'm a Python developer working on a web app.", 
         "Nice to meet you! I'd be happy to help with your Python web app."),
        ("I'm using FastAPI with PostgreSQL.",
         "Great choices! FastAPI is excellent for building APIs."),
        ("Can you remember my stack for future reference?",
         "Absolutely! I'll remember you're using FastAPI with PostgreSQL."),
    ]
    
    for user_input, ai_response in exchanges:
        # Save context (this is what LangChain does internally)
        memory.save_context(
            {"input": user_input},
            {"output": ai_response},
        )
        print(f"\n  User: {user_input}")
        print(f"  AI: {ai_response}")
    
    # Load and display memory
    print("\n  Loading memory variables...")
    history = memory.load_memory_variables({})["history"]
    print(f"\n  Stored history:\n  {history[:200]}...")
    
    # Demonstrate persistence - create new memory with same session
    print("\n  Creating new memory instance (simulating new session)...")
    memory2 = AgentMemoryToolkitMemory(
        store=store,
        session_id="langchain_demo_session",
    )
    
    loaded = memory2.load_memory_variables({})["history"]
    print(f"  Loaded from previous session: {len(loaded)} characters")
    
    store.close()
    print("\n  ✓ Conversation persisted successfully")


# =============================================================================
# Example 2: Memory-Augmented Agent
# =============================================================================

def example_memory_agent():
    """
    Create a LangChain agent that can store and recall memories.
    
    This pattern is useful for agents that need to remember user
    preferences, past interactions, or learned information.
    """
    print_header("2. MEMORY-AUGMENTED AGENT")
    
    store = MemoryStore(":memory:")
    
    # Add some background knowledge
    print("\n  Adding background knowledge to memory...")
    
    knowledge = [
        "User prefers dark mode in all applications",
        "User's timezone is PST (Pacific Standard Time)",
        "User's preferred programming language is Python",
        "User works at TechCorp as a senior engineer",
        "User has meetings on Monday and Wednesday mornings",
        "User prefers async communication over meetings",
        "The project uses PostgreSQL 15 with pgvector extension",
        "Deployment happens via GitHub Actions to AWS ECS",
    ]
    
    for item in knowledge:
        store.add(item, metadata={"tags": ["user_profile", "preferences"]})
        print(f"    ✓ Added: {item[:50]}...")
    
    # Create memory with search capability
    memory = AgentMemoryToolkitMemory(
        store=store,
        session_id="agent_session",
        search_on_load=True,  # Search for relevant memories
        search_limit=3,
    )
    
    # Simulate agent queries
    queries = [
        "What are the user's communication preferences?",
        "What technology stack is being used?",
        "When does the user have meetings?",
    ]
    
    print("\n  Searching memories for relevant context...")
    
    for query in queries:
        print(f"\n  Query: {query}")
        results = memory.search_memories(query, limit=2)
        for r in results:
            print(f"    → {r['content'][:60]}... (score: {r['score']:.3f})")
    
    store.close()
    print("\n  ✓ Memory search completed")


# =============================================================================
# Example 3: RAG with Memory Context
# =============================================================================

def example_rag_with_memory():
    """
    Combine retrieval-augmented generation with persistent memory.
    
    This shows how to inject relevant memories into your prompts
    for more contextual responses.
    """
    print_header("3. RAG WITH MEMORY CONTEXT")
    
    store = MemoryStore(":memory:")
    
    # Store some project documentation
    docs = [
        "The API uses JWT tokens for authentication. Tokens expire after 24 hours.",
        "Rate limiting is set to 100 requests per minute per user.",
        "Error responses follow RFC 7807 Problem Details format.",
        "All endpoints require the X-API-Version header.",
        "Database migrations run automatically on deployment.",
        "Feature flags are managed through LaunchDarkly.",
    ]
    
    print("\n  Storing project documentation...")
    for doc in docs:
        store.add(doc, metadata={"source": "project_docs", "tags": ["api", "docs"]})
    
    def build_prompt_with_memory(query: str, limit: int = 3) -> str:
        """Build a prompt that includes relevant memories."""
        results = store.search_fts(query, limit=limit)
        
        if not results:
            context = "No relevant documentation found."
        else:
            context = "\n".join(f"- {r.memory.content}" for r in results)
        
        prompt = f"""Based on the following project documentation:

{context}

Answer this question: {query}"""
        
        return prompt
    
    # Test with some queries
    queries = [
        "How does authentication work?",
        "What is the rate limit?",
        "How are errors formatted?",
    ]
    
    print("\n  Building prompts with memory context...")
    
    for query in queries:
        print(f"\n  Query: {query}")
        prompt = build_prompt_with_memory(query)
        print(f"  Prompt preview:\n  {prompt[:150]}...")
    
    store.close()
    print("\n  ✓ RAG prompts built with memory context")


# =============================================================================
# Example 4: Multi-Session Chat Memory
# =============================================================================

def example_multi_session():
    """
    Manage multiple chat sessions with isolated memories.
    
    Useful for applications serving multiple users or
    maintaining separate conversation contexts.
    """
    print_header("4. MULTI-SESSION CHAT MEMORY")
    
    store = MemoryStore(":memory:")
    
    # Simulate multiple users
    users = {
        "alice_123": [
            ("I need help with React", "Sure! I can help with React development."),
            ("I'm building a dashboard", "What kind of data will the dashboard show?"),
        ],
        "bob_456": [
            ("Python question here", "Happy to help with Python!"),
            ("How do I use asyncio?", "asyncio is Python's async programming library."),
        ],
        "charlie_789": [
            ("DevOps help needed", "I can assist with DevOps topics."),
            ("Setting up CI/CD", "Let's talk about your CI/CD pipeline."),
        ],
    }
    
    print("\n  Creating sessions for multiple users...")
    
    for user_id, exchanges in users.items():
        memory = AgentMemoryToolkitChatMemory(
            store=store,
            session_id=user_id,
        )
        
        for user_msg, ai_msg in exchanges:
            memory.add_user_message(user_msg)
            memory.add_ai_message(ai_msg)
        
        print(f"    ✓ {user_id}: {len(exchanges) * 2} messages")
    
    # Load each user's history
    print("\n  Loading user histories...")
    
    for user_id in users.keys():
        memory = AgentMemoryToolkitChatMemory(
            store=store,
            session_id=user_id,
        )
        
        messages = memory.messages
        first_msg = messages[0].content if messages else "No messages"
        print(f"    {user_id}: {len(messages)} messages - \"{first_msg[:30]}...\"")
    
    # Clear one user's history
    print("\n  Clearing bob_456's history...")
    bob_memory = AgentMemoryToolkitChatMemory(store=store, session_id="bob_456")
    bob_memory.clear()
    print(f"    bob_456 now has {len(bob_memory.messages)} messages")
    
    store.close()
    print("\n  ✓ Multi-session management demonstrated")


# =============================================================================
# Example 5: Full LangChain Chain (requires API key)
# =============================================================================

def example_full_chain():
    """
    Create a complete LangChain conversation chain.
    
    This example requires OPENAI_API_KEY to be set.
    """
    print_header("5. FULL LANGCHAIN CHAIN (requires OPENAI_API_KEY)")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n  ⚠ OPENAI_API_KEY not set - showing code example only")
        print("""
  To run this example:
  
    export OPENAI_API_KEY="sk-your-key-here"
    python langchain_integration.py
    
  Code example:
  
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI
    from agent_memory_toolkit import MemoryStore
    from agent_memory_toolkit.integrations.langchain import AgentMemoryToolkitMemory
    
    store = MemoryStore("conversations.db")
    memory = AgentMemoryToolkitMemory(
        store=store,
        session_id="user_session",
        return_messages=False,
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = ConversationChain(llm=llm, memory=memory, verbose=True)
    
    # Use the chain
    response = chain.invoke({"input": "Remember my name is Alice"})
    print(response["response"])
    
    response = chain.invoke({"input": "What's my name?"})
    print(response["response"])  # Should remember Alice
""")
        return
    
    try:
        from langchain.chains import ConversationChain
        from langchain_openai import ChatOpenAI
        
        store = MemoryStore(":memory:")
        memory = AgentMemoryToolkitMemory(
            store=store,
            session_id="full_chain_demo",
            return_messages=False,
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = ConversationChain(llm=llm, memory=memory)
        
        print("\n  Running conversation chain...")
        
        # First exchange
        response = chain.invoke({"input": "Hi! My name is Alice and I work on ML projects."})
        print(f"\n  User: Hi! My name is Alice and I work on ML projects.")
        print(f"  AI: {response['response'][:100]}...")
        
        # Test memory recall
        response = chain.invoke({"input": "What's my name and what do I work on?"})
        print(f"\n  User: What's my name and what do I work on?")
        print(f"  AI: {response['response'][:100]}...")
        
        store.close()
        print("\n  ✓ Full chain completed successfully")
        
    except ImportError as e:
        print(f"\n  ⚠ Required packages not installed: {e}")
        print("  Run: pip install langchain langchain-openai")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  AGENT MEMORY TOOLKIT - LANGCHAIN INTEGRATION")
    print("=" * 60)
    
    example_conversation_chain()
    example_memory_agent()
    example_rag_with_memory()
    example_multi_session()
    example_full_chain()
    
    print("\n" + "=" * 60)
    print("  ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print("""
  Next steps:
  
  1. Try with a real database: MemoryStore("memory.db")
  2. Set OPENAI_API_KEY to test the full chain
  3. See langchain_example.py for more detailed examples
  4. Check the docs: https://agent-memory-toolkit.readthedocs.io
""")


if __name__ == "__main__":
    main()
