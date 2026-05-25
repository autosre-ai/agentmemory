#!/usr/bin/env python3
"""
OpenAI Integration Example - Agent Memory Toolkit

Demonstrates how to use Agent Memory Toolkit with the OpenAI Python SDK:
1. Basic Memory-Augmented Chat
2. Function Calling with Memory Tools
3. Assistants API with Persistent Context
4. Structured Memory Extraction

Requirements:
    pip install agent-memory-toolkit openai
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_memory_toolkit import MemoryStore, MemoryExtractor


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Example 1: Memory-Augmented Chat Completions
# =============================================================================

def example_memory_chat():
    """
    Augment OpenAI chat completions with relevant memories.
    
    This pattern retrieves relevant context from the memory store
    and includes it in the system prompt.
    """
    print_header("1. MEMORY-AUGMENTED CHAT")
    
    store = MemoryStore(":memory:")
    
    # Store user profile and preferences
    user_info = [
        "User name: Sarah Chen",
        "User role: Senior Software Engineer at TechCorp",
        "User expertise: Python, FastAPI, PostgreSQL",
        "User timezone: PST (UTC-8)",
        "User preference: Prefers concise, technical responses",
        "User preference: Uses vim keybindings",
        "Current project: Building a real-time analytics dashboard",
        "Tech stack: Python 3.11, FastAPI, PostgreSQL 15, Redis",
    ]
    
    print("\n  Storing user profile...")
    for info in user_info:
        store.add(info, metadata={"tags": ["user_profile"]})
        print(f"    ✓ {info[:50]}...")
    
    def get_relevant_context(query: str, limit: int = 5) -> str:
        """Retrieve relevant memories for the query."""
        results = store.search_fts(query, limit=limit)
        if not results:
            return ""
        return "\n".join(f"- {r.memory.content}" for r in results)
    
    def create_messages(user_query: str) -> list[dict]:
        """Build messages with memory context."""
        context = get_relevant_context(user_query)
        
        system_prompt = f"""You are a helpful assistant with memory of past interactions.

User Context:
{context if context else "No relevant context found."}

Use this context to personalize your responses."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
    
    # Test queries
    queries = [
        "What tech stack am I using?",
        "Give me a FastAPI code example",
        "What's my preferred coding style?",
    ]
    
    print("\n  Building memory-augmented prompts...")
    
    for query in queries:
        messages = create_messages(query)
        context = get_relevant_context(query)
        
        print(f"\n  Query: {query}")
        print(f"  Context found: {len(context.split(chr(10)))} relevant items")
        print(f"  System prompt preview: {messages[0]['content'][:100]}...")
    
    store.close()
    print("\n  ✓ Memory context integration demonstrated")


# =============================================================================
# Example 2: Function Calling with Memory Tools
# =============================================================================

def example_function_calling():
    """
    Define OpenAI function tools for memory operations.
    
    This allows the AI to store and retrieve memories as part
    of the conversation flow.
    """
    print_header("2. FUNCTION CALLING WITH MEMORY TOOLS")
    
    store = MemoryStore(":memory:")
    
    # Define tools for OpenAI function calling
    memory_tools = [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Store important information for future reference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to remember",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to categorize the memory",
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search for previously stored information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget",
                "description": "Delete a specific memory by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to delete",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
        },
    ]
    
    print("\n  Defined memory tools:")
    for tool in memory_tools:
        print(f"    - {tool['function']['name']}: {tool['function']['description'][:50]}...")
    
    # Simulate tool execution
    def execute_tool(name: str, args: dict) -> str:
        """Execute a memory tool."""
        if name == "remember":
            memory = store.add(
                args["content"],
                metadata={"tags": args.get("tags", [])},
            )
            return json.dumps({"success": True, "id": memory.id})
        
        elif name == "recall":
            results = store.search_fts(args["query"], limit=args.get("limit", 5))
            memories = [
                {"id": r.memory.id, "content": r.memory.content, "score": r.score}
                for r in results
            ]
            return json.dumps({"memories": memories, "count": len(memories)})
        
        elif name == "forget":
            store.delete(args["memory_id"])
            return json.dumps({"success": True})
        
        return json.dumps({"error": "Unknown tool"})
    
    # Simulate a conversation with tool calls
    print("\n  Simulating tool calls...")
    
    # Store some memories
    result = execute_tool("remember", {
        "content": "User's favorite color is blue",
        "tags": ["preferences"],
    })
    print(f"\n  Tool: remember('User's favorite color is blue')")
    print(f"  Result: {result}")
    
    result = execute_tool("remember", {
        "content": "User's birthday is March 15",
        "tags": ["personal"],
    })
    print(f"\n  Tool: remember('User's birthday is March 15')")
    print(f"  Result: {result}")
    
    # Recall memories
    result = execute_tool("recall", {"query": "color preferences"})
    print(f"\n  Tool: recall('color preferences')")
    print(f"  Result: {result}")
    
    store.close()
    print("\n  ✓ Function calling tools demonstrated")


# =============================================================================
# Example 3: Conversation with Automatic Memory Extraction
# =============================================================================

def example_auto_extraction():
    """
    Automatically extract and store memories from conversations.
    
    Uses the rule-based extractor to identify important information
    from user messages.
    """
    print_header("3. AUTOMATIC MEMORY EXTRACTION")
    
    store = MemoryStore(":memory:")
    extractor = MemoryExtractor(mode="rule")
    
    # Simulate conversation messages
    messages = [
        "Hi, I'm Alex and I work as a data scientist at DataCorp.",
        "I usually start work around 9am PST.",
        "For projects, I prefer using Python with pandas and scikit-learn.",
        "My email is alex@datacorp.com if you need to reach me.",
        "I have a weekly sync with my team every Monday at 2pm.",
    ]
    
    print("\n  Processing conversation for memories...")
    
    for msg in messages:
        print(f"\n  Message: {msg}")
        
        # Extract memories from the message
        result = extractor.extract(msg, source="conversation")
        
        if result.memories:
            print(f"  Extracted {len(result.memories)} memories:")
            for memory in result.memories:
                # Store the extracted memory
                stored = store.add(
                    f"{memory.key}: {memory.value}",
                    metadata={
                        "domain": memory.domain.value,
                        "confidence": memory.confidence,
                        "tags": [memory.domain.value],
                    },
                )
                print(f"    ✓ [{memory.domain.value}] {memory.key}: {memory.value}")
        else:
            print("  No structured memories extracted")
    
    # Show all stored memories
    print("\n  All stored memories:")
    all_memories = store.list(limit=20)
    for m in all_memories:
        print(f"    - {m.content}")
    
    store.close()
    print(f"\n  ✓ Extracted and stored {len(all_memories)} memories")


# =============================================================================
# Example 4: Multi-Turn Conversation with Context
# =============================================================================

def example_multi_turn():
    """
    Maintain context across multiple conversation turns.
    
    This shows how to build a conversation loop that stores
    each exchange and uses past context.
    """
    print_header("4. MULTI-TURN CONVERSATION")
    
    store = MemoryStore(":memory:")
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    class ConversationManager:
        """Manage a conversation with memory."""
        
        def __init__(self, store: MemoryStore, session_id: str):
            self.store = store
            self.session_id = session_id
            self.turns = []
        
        def add_turn(self, user_msg: str, assistant_msg: str):
            """Add a conversation turn."""
            self.turns.append({
                "user": user_msg,
                "assistant": assistant_msg,
                "timestamp": datetime.now().isoformat(),
            })
            
            # Store in memory
            self.store.add(
                f"[{self.session_id}] User: {user_msg}",
                metadata={"role": "user", "session": self.session_id},
            )
            self.store.add(
                f"[{self.session_id}] Assistant: {assistant_msg}",
                metadata={"role": "assistant", "session": self.session_id},
            )
        
        def get_context(self, max_turns: int = 5) -> str:
            """Get recent conversation context."""
            recent = self.turns[-max_turns:]
            lines = []
            for turn in recent:
                lines.append(f"User: {turn['user']}")
                lines.append(f"Assistant: {turn['assistant']}")
            return "\n".join(lines)
        
        def build_messages(self, user_input: str) -> list[dict]:
            """Build messages for API call."""
            context = self.get_context()
            
            system = """You are a helpful assistant. Use the conversation context 
to provide consistent and relevant responses."""
            
            messages = [{"role": "system", "content": system}]
            
            # Add conversation history
            for turn in self.turns[-5:]:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
            
            # Add current message
            messages.append({"role": "user", "content": user_input})
            
            return messages
    
    # Simulate a conversation
    conversation = ConversationManager(store, session_id)
    
    exchanges = [
        ("Hi! I'm working on a Python project.", 
         "Great! What kind of Python project are you working on?"),
        ("It's a REST API for managing tasks.",
         "Nice! Are you using a framework like FastAPI or Flask?"),
        ("FastAPI. I'm trying to add authentication.",
         "FastAPI works great with OAuth2. Would you like an example?"),
        ("Yes please, show me JWT authentication.",
         "Here's a basic JWT setup for FastAPI..."),
    ]
    
    print(f"\n  Session: {session_id}")
    print("\n  Building conversation...")
    
    for user_msg, assistant_msg in exchanges:
        conversation.add_turn(user_msg, assistant_msg)
        print(f"\n  User: {user_msg}")
        print(f"  Assistant: {assistant_msg}")
    
    # Show how context is built for next message
    print("\n  Context for next message:")
    context = conversation.get_context(max_turns=3)
    print(f"  {context[:200]}...")
    
    # Show messages that would be sent to API
    messages = conversation.build_messages("How do I verify the JWT token?")
    print(f"\n  Messages prepared for API: {len(messages)} messages")
    print(f"  Last message: {messages[-1]['content']}")
    
    store.close()
    print(f"\n  ✓ {len(exchanges)} turns stored with context")


# =============================================================================
# Example 5: Full OpenAI Integration (requires API key)
# =============================================================================

def example_full_integration():
    """
    Complete integration with OpenAI API.
    
    This example requires OPENAI_API_KEY to be set.
    """
    print_header("5. FULL OPENAI INTEGRATION (requires OPENAI_API_KEY)")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n  ⚠ OPENAI_API_KEY not set - showing code example only")
        print("""
  To run this example:
  
    export OPENAI_API_KEY="sk-your-key-here"
    python openai_integration.py
    
  Code example:
  
    from openai import OpenAI
    from agent_memory_toolkit import MemoryStore
    
    client = OpenAI()
    store = MemoryStore("memory.db")
    
    # Store user context
    store.add("User prefers Python with type hints")
    store.add("User's project uses FastAPI")
    
    # Get relevant context
    results = store.search_fts("coding style", limit=3)
    context = "\\n".join(r.memory.content for r in results)
    
    # Create completion with context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"User context:\\n{context}"},
            {"role": "user", "content": "Write a function to fetch data from an API"},
        ],
    )
    
    print(response.choices[0].message.content)
    
    # Store the interaction
    store.add(f"User asked: Write a function to fetch data from an API")
    store.add(f"Assistant provided: {response.choices[0].message.content[:100]}...")
""")
        return
    
    try:
        from openai import OpenAI
        
        client = OpenAI()
        store = MemoryStore(":memory:")
        
        # Store some context
        store.add("User is building a Python web application")
        store.add("User prefers async programming patterns")
        store.add("User uses FastAPI for their backend")
        
        print("\n  Calling OpenAI API with memory context...")
        
        # Get relevant context
        results = store.search_fts("Python web programming", limit=3)
        context = "\n".join(f"- {r.memory.content}" for r in results)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful coding assistant.\n\nUser context:\n{context}",
                },
                {
                    "role": "user",
                    "content": "Give me a quick tip for my project",
                },
            ],
            max_tokens=150,
        )
        
        answer = response.choices[0].message.content
        print(f"\n  Response: {answer[:200]}...")
        
        # Store the interaction
        store.add(f"Tip given: {answer[:100]}...", metadata={"type": "interaction"})
        
        store.close()
        print("\n  ✓ OpenAI integration completed")
        
    except ImportError:
        print("\n  ⚠ openai package not installed")
        print("  Run: pip install openai")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  AGENT MEMORY TOOLKIT - OPENAI INTEGRATION")
    print("=" * 60)
    
    example_memory_chat()
    example_function_calling()
    example_auto_extraction()
    example_multi_turn()
    example_full_integration()
    
    print("\n" + "=" * 60)
    print("  ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print("""
  Key patterns demonstrated:
  
  1. Context injection: Add relevant memories to system prompts
  2. Function tools: Let the AI store and retrieve memories
  3. Auto-extraction: Extract structured data from conversations
  4. Multi-turn: Maintain context across conversation turns
  5. Full integration: Complete OpenAI API workflow
  
  Next steps:
  
  - Use a persistent database: MemoryStore("memory.db")
  - Try semantic search with embeddings: store.search_semantic()
  - Add the MCP server for Claude Desktop integration
  - Check the docs: https://agent-memory-toolkit.readthedocs.io
""")


if __name__ == "__main__":
    main()
