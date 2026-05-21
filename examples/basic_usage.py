#!/usr/bin/env python3
"""
Basic Usage Example - Agent Memory Toolkit

Demonstrates core functionality:
- Memory extraction from text
- Persistent storage
- Search operations
- Version control
"""

import sys
sys.path.insert(0, "..")

from agentmemory import (
    MemoryExtractor,
    CognitiveDomain,
    Memory,
    ExtractionResult,
)
from agentmemory.store import MemoryStore


def demo_extraction():
    """Demonstrate memory extraction from text."""
    print("=" * 60)
    print("1. MEMORY EXTRACTION")
    print("=" * 60)
    
    # Initialize extractor (rule-based for offline use)
    extractor = MemoryExtractor(mode="rule")
    
    # Sample conversation text
    text = """
    Hi, I'm Sarah Chen. I work as a Senior Software Engineer at TechCorp 
    in San Francisco. I've been coding for about 10 years now.
    
    My favorite programming language is Python, though I also use TypeScript 
    for frontend work. I prefer dark mode in all my editors and use VS Code
    with vim keybindings.
    
    I usually work from 9am to 5pm PST, and I have a standing meeting with 
    my team lead John Smith every Monday at 10am. 
    
    For code reviews, I always run the linter first, then check test coverage,
    and finally review the actual logic.
    """
    
    # Extract memories
    result = extractor.extract(text, source="onboarding_chat")
    
    print(f"\nExtracted {len(result.memories)} memories in {result.processing_time_ms:.1f}ms:\n")
    
    # Group by domain for display
    by_domain = {}
    for memory in result.memories:
        domain = memory.domain.value
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(memory)
    
    for domain, memories in sorted(by_domain.items()):
        print(f"  {domain.upper()}:")
        for m in memories:
            print(f"    • {m.key}: {m.value} (confidence: {m.confidence:.2f})")
        print()
    
    return result.memories


def demo_storage(memories: list):
    """Demonstrate persistent storage operations."""
    print("=" * 60)
    print("2. PERSISTENT STORAGE")
    print("=" * 60)
    
    # Create in-memory store (use file path for persistence)
    # For real use: store = MemoryStore("agent_memory.db")
    store = MemoryStore(":memory:")
    
    print("\n  Adding memories to store...")
    
    # Add extracted memories as persistent entries
    stored = []
    for memory in memories[:5]:  # Store first 5
        stored_memory = store.add(
            content=f"{memory.key}: {memory.value}",
            metadata={
                "domain": memory.domain.value,
                "source": memory.source,
                "confidence": memory.confidence,
                "tags": [memory.domain.value],
            }
        )
        stored.append(stored_memory)
        print(f"    ✓ Added: {stored_memory.id[:8]}... - {memory.key}")
    
    # Add some additional memories
    additional_memories = [
        "Sarah prefers async communication over meetings",
        "The project uses PostgreSQL for the database",
        "Deployment happens every Tuesday and Thursday",
        "The team follows trunk-based development",
    ]
    
    for content in additional_memories:
        m = store.add(content, metadata={"tags": ["general"]})
        stored.append(m)
        print(f"    ✓ Added: {m.id[:8]}... - {content[:40]}...")
    
    print(f"\n  Total memories in store: {store.count()}")
    
    return store, stored


def demo_search(store: MemoryStore):
    """Demonstrate search capabilities."""
    print("\n" + "=" * 60)
    print("3. SEARCH OPERATIONS")
    print("=" * 60)
    
    # Full-text search
    print("\n  Full-text search for 'Python':")
    results = store.search_fts("Python")
    for r in results:
        print(f"    [{r.score:.2f}] {r.memory.content[:50]}...")
    
    # Search with filters
    print("\n  Listing memories with 'work' tag:")
    work_memories = store.list(tag="work", limit=5)
    for m in work_memories:
        print(f"    • {m.content[:50]}...")
    
    # Search for specific information
    print("\n  Full-text search for 'meeting':")
    results = store.search_fts("meeting")
    for r in results:
        print(f"    [{r.score:.2f}] {r.memory.content[:50]}...")


def demo_versioning(store: MemoryStore, stored: list):
    """Demonstrate version control features."""
    print("\n" + "=" * 60)
    print("4. VERSION CONTROL")
    print("=" * 60)
    
    if not stored:
        print("\n  No stored memories to update")
        return
    
    # Get a memory to update
    memory = stored[0]
    print(f"\n  Original memory (v{memory.version}):")
    print(f"    {memory.content}")
    
    # Update the memory
    updated = store.update(
        memory.id,
        content="name: Sarah Chen (verified)",
        metadata={"confidence": 1.0, "verified": True}
    )
    print(f"\n  Updated memory (v{updated.version}):")
    print(f"    {updated.content}")
    
    # Create a commit
    print("\n  Creating commit...")
    commit = store.commit("Initial profile setup")
    print(f"    Commit: {commit.id[:8]}...")
    print(f"    Message: {commit.message}")
    print(f"    Memories: {len(commit.memory_snapshot)}")
    
    # View commit log
    print("\n  Commit history:")
    for c in store.get_history(limit=5):
        print(f"    • {c.id[:8]}... - {c.message}")


def demo_branching(store: MemoryStore):
    """Demonstrate branching capabilities."""
    print("\n" + "=" * 60)
    print("5. BRANCHING")
    print("=" * 60)
    
    print(f"\n  Current branch: {store.current_branch}")
    
    # Create a new branch
    print("\n  Creating 'experiment' branch...")
    store.create_branch("experiment")
    
    # List branches
    print("\n  Available branches:")
    for branch in store.list_branches():
        marker = " *" if branch.name == store.current_branch else ""
        print(f"    • {branch.name}{marker}")
    
    # Switch to experiment branch
    store.checkout("experiment")
    print(f"\n  Switched to: {store.current_branch}")
    
    # Add experimental memory
    exp_memory = store.add(
        "Experimenting with new workflow",
        metadata={"tags": ["experiment"]}
    )
    print(f"  Added experimental memory: {exp_memory.id[:8]}...")
    
    # Switch back
    store.checkout("main")
    print(f"\n  Switched back to: {store.current_branch}")


def demo_export_import(store: MemoryStore):
    """Demonstrate export/import capabilities."""
    print("\n" + "=" * 60)
    print("6. EXPORT/IMPORT")
    print("=" * 60)
    
    import tempfile
    import os
    
    # Export to JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        export_path = f.name
    
    count = store.export_json(export_path)
    print(f"\n  Exported {count} memories to {export_path}")
    
    # Show first few lines
    with open(export_path) as f:
        content = f.read()
        preview = content[:500] + "..." if len(content) > 500 else content
        print(f"\n  Export preview:\n  {preview[:200]}...")
    
    # Clean up
    os.unlink(export_path)
    print(f"\n  Cleaned up temporary file")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("  AGENT MEMORY TOOLKIT - BASIC USAGE DEMO")
    print("=" * 60)
    
    # Run demonstrations
    memories = demo_extraction()
    store, stored = demo_storage(memories)
    demo_search(store)
    demo_versioning(store, stored)
    demo_branching(store)
    demo_export_import(store)
    
    # Cleanup
    store.close()
    
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
