#!/usr/bin/env python3
"""
Team Collaboration Example - Agent Memory Toolkit

Demonstrates multi-agent memory sharing:
- Creating agent-specific memory stores
- Git-like branching and merging
- Conflict resolution strategies
- Filesystem sync protocol
- Access control and namespaces
- Event hooks for collaboration
"""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, "..")

from agent_memory_toolkit.team import TeamMemoryStore
from agent_memory_toolkit.team.models import ConflictResolution, EventType


def demo_basic_team():
    """Demonstrate basic team memory setup."""
    print("=" * 60)
    print("1. BASIC TEAM MEMORY SETUP")
    print("=" * 60)
    
    # Create memory stores for two agents
    print("\n  Setting up agents Alice and Bob...")
    
    alice = TeamMemoryStore(":memory:", agent_id="alice", default_namespace="shared")
    bob = TeamMemoryStore(":memory:", agent_id="bob", default_namespace="shared")
    
    print(f"    ✓ Alice's store initialized (agent_id: {alice.agent_id})")
    print(f"    ✓ Bob's store initialized (agent_id: {bob.agent_id})")
    
    # Alice adds some memories
    print("\n  Alice adding project memories...")
    
    memories = [
        ("Project deadline is Friday 5pm", {"tags": ["deadline"]}),
        ("API uses OAuth 2.0 for authentication", {"tags": ["technical"]}),
        ("Daily standup at 10am PST", {"tags": ["schedule"]}),
    ]
    
    for content, metadata in memories:
        m = alice.add(content, metadata=metadata)
        print(f"    ✓ {content[:40]}...")
    
    print(f"\n  Alice's memory count: {len(alice.list())}")
    
    alice.close()
    bob.close()
    
    return True


def demo_branching_workflow():
    """Demonstrate branching workflow."""
    print("\n" + "=" * 60)
    print("2. BRANCHING WORKFLOW")
    print("=" * 60)
    
    store = TeamMemoryStore(":memory:", agent_id="alice")
    
    # Add initial memories
    print("\n  Setting up main branch with memories...")
    store.add("Base configuration uses PostgreSQL")
    store.add("Frontend uses React 18")
    store.commit("Initial project setup")
    
    print(f"    Current branch: {store.current_branch}")
    print(f"    Memory count: {len(store.list())}")
    
    # Create experimental branch
    print("\n  Creating 'feature/new-db' branch...")
    store.create_branch("feature/new-db")
    store.checkout("feature/new-db")
    
    print(f"    Switched to: {store.current_branch}")
    
    # Make changes on branch
    print("\n  Making changes on feature branch...")
    store.add("Experimenting with MongoDB instead")
    store.add("Added new caching layer with Redis")
    
    print(f"    Feature branch memories: {len(store.list())}")
    
    # List all branches
    print("\n  Available branches:")
    for branch in store.list_branches():
        marker = " *" if branch.name == store.current_branch else ""
        print(f"    • {branch.name}{marker} (by {branch.created_by})")
    
    # Switch back to main
    store.checkout("main")
    print(f"\n  Switched back to: {store.current_branch}")
    print(f"    Main branch memories: {len(store.list())}")
    
    store.close()
    return True


def demo_conflict_resolution():
    """Demonstrate conflict resolution strategies."""
    print("\n" + "=" * 60)
    print("3. CONFLICT RESOLUTION")
    print("=" * 60)
    
    # Create two stores that will conflict
    store = TeamMemoryStore(
        ":memory:", 
        agent_id="alice",
        conflict_strategy=ConflictResolution.LATEST_WINS
    )
    
    print("\n  Setting up scenario with potential conflicts...")
    
    # Add memory on main
    mem = store.add("Project uses Python 3.11", metadata={"tags": ["tech"]})
    store.commit("Set Python version")
    
    # Create branch and modify
    store.create_branch("update-python")
    store.checkout("update-python")
    
    # Simulate time passing
    import time
    time.sleep(0.1)
    
    # Update on branch
    store.add("Project uses Python 3.12", metadata={"tags": ["tech"]})
    store.commit("Updated Python version")
    
    print(f"    Main branch: Python 3.11")
    print(f"    update-python branch: Python 3.12")
    
    # Merge with different strategies
    print("\n  Merging with LATEST_WINS strategy...")
    store.checkout("main")
    conflicts = store.merge("update-python", conflict_strategy=ConflictResolution.LATEST_WINS)
    
    print(f"    Conflicts resolved: {len(conflicts)} remaining")
    
    # Show available strategies
    print("\n  Available conflict resolution strategies:")
    for strategy in ConflictResolution:
        print(f"    • {strategy.name}: {strategy.value}")
    
    store.close()
    return True


def demo_sync_protocol():
    """Demonstrate filesystem sync protocol."""
    print("\n" + "=" * 60)
    print("4. FILESYSTEM SYNC PROTOCOL")
    print("=" * 60)
    
    # Create temp directory for sync
    sync_dir = tempfile.mkdtemp(prefix="agent_memory_sync_")
    print(f"\n  Created sync directory: {sync_dir}")
    
    try:
        # Alice's store
        print("\n  Setting up Alice's local store...")
        alice = TeamMemoryStore(":memory:", agent_id="alice")
        
        alice.add("Alice's finding: API rate limit is 100/min")
        alice.add("Alice's note: Customer prefers email contact")
        alice.commit("Alice's initial data")
        
        print(f"    Alice has {len(alice.list())} memories")
        
        # Push to shared location
        print("\n  Alice pushing to shared location...")
        push_result = alice.push(sync_dir)
        print(f"    Pushed: {push_result.memories_pushed} memories")
        print(f"    Conflicts: {push_result.conflicts}")
        
        # Bob's store
        print("\n  Setting up Bob's local store...")
        bob = TeamMemoryStore(":memory:", agent_id="bob")
        
        # Pull from shared location
        print("\n  Bob pulling from shared location...")
        pull_result = bob.pull(sync_dir)
        print(f"    Pulled: {pull_result.memories_pulled} memories")
        
        # Verify sync
        print("\n  Verifying sync:")
        print(f"    Alice's memories: {len(alice.list())}")
        print(f"    Bob's memories: {len(bob.list())}")
        
        # Bob adds his own findings
        print("\n  Bob adding his own data...")
        bob.add("Bob's finding: Billing cycle is monthly")
        bob.commit("Bob's additions")
        
        # Full bidirectional sync
        print("\n  Bob doing full sync...")
        sync_result = bob.sync(sync_dir)
        print(f"    Pushed: {sync_result.memories_pushed}")
        print(f"    Pulled: {sync_result.memories_pulled}")
        
        alice.close()
        bob.close()
        
    finally:
        # Cleanup
        shutil.rmtree(sync_dir)
        print(f"\n  Cleaned up sync directory")
    
    return True


def demo_namespaces():
    """Demonstrate namespace-based organization."""
    print("\n" + "=" * 60)
    print("5. NAMESPACES")
    print("=" * 60)
    
    store = TeamMemoryStore(":memory:", agent_id="alice")
    
    # Create namespaces
    print("\n  Creating namespaces...")
    store.create_namespace("project-alpha", "Alpha project workspace")
    store.create_namespace("project-beta", "Beta project workspace")
    store.create_namespace("shared", "Shared company knowledge")
    
    # List namespaces
    print("\n  Available namespaces:")
    for ns in store.list_namespaces():
        print(f"    • {ns}")
    
    # Add memories to different namespaces
    print("\n  Adding memories to namespaces...")
    
    store.add("Alpha deadline: Q1 2025", namespace="project-alpha")
    store.add("Alpha tech: Rust backend", namespace="project-alpha")
    
    store.add("Beta deadline: Q2 2025", namespace="project-beta")
    store.add("Beta tech: Go backend", namespace="project-beta")
    
    store.add("Company holiday: Dec 25-Jan 2", namespace="shared")
    
    # Query by namespace
    print("\n  Memories in project-alpha:")
    for m in store.list(namespace="project-alpha"):
        print(f"    • {m.content}")
    
    print("\n  Memories in project-beta:")
    for m in store.list(namespace="project-beta"):
        print(f"    • {m.content}")
    
    store.close()
    return True


def demo_access_control():
    """Demonstrate access control."""
    print("\n" + "=" * 60)
    print("6. ACCESS CONTROL")
    print("=" * 60)
    
    from agent_memory_toolkit.team.models import Permission
    
    store = TeamMemoryStore(":memory:", agent_id="admin")
    
    print("\n  Setting up access control...")
    
    # Create namespace with restricted access
    store.create_namespace("confidential", "Restricted data")
    
    # Grant permissions
    print("\n  Granting permissions:")
    
    store.access.grant("alice", Permission.READ, "confidential")
    print("    ✓ alice: READ on 'confidential'")
    
    store.access.grant("alice", Permission.WRITE, "confidential")
    print("    ✓ alice: WRITE on 'confidential'")
    
    store.access.grant("bob", Permission.READ, "confidential")
    print("    ✓ bob: READ on 'confidential'")
    
    # Check permissions
    print("\n  Checking permissions:")
    
    can_read = store.access.has_permission("alice", Permission.READ, "confidential")
    print(f"    alice can READ confidential: {can_read}")
    
    can_write = store.access.has_permission("bob", Permission.WRITE, "confidential")
    print(f"    bob can WRITE confidential: {can_write}")
    
    # List permissions
    print("\n  All permissions for 'confidential' namespace:")
    perms = store.access.list_rules(namespace="confidential")
    for p in perms:
        print(f"    • {p.agent_id}: {p.permission.name}")
    
    store.close()
    return True


def demo_event_hooks():
    """Demonstrate event hooks for collaboration."""
    print("\n" + "=" * 60)
    print("7. EVENT HOOKS")
    print("=" * 60)
    
    store = TeamMemoryStore(":memory:", agent_id="alice")
    
    # Track events
    events_received = []
    
    def on_memory_created(event):
        events_received.append(("CREATED", event.data.get("memory_id", "")[:8]))
        print(f"    → Event: Memory created ({event.data.get('memory_id', '')[:8]}...)")
    
    def on_memory_updated(event):
        events_received.append(("UPDATED", event.data.get("memory_id", "")[:8]))
        print(f"    → Event: Memory updated ({event.data.get('memory_id', '')[:8]}...)")
    
    def on_branch_created(event):
        events_received.append(("BRANCH", event.data.get("branch", "")))
        print(f"    → Event: Branch created ({event.data.get('branch', '')})")
    
    # Register hooks
    print("\n  Registering event hooks...")
    store.on(EventType.MEMORY_CREATED, on_memory_created)
    store.on(EventType.MEMORY_UPDATED, on_memory_updated)
    store.on(EventType.BRANCH_CREATED, on_branch_created)
    
    # Perform operations that trigger events
    print("\n  Performing operations:")
    
    print("    Adding memory...")
    mem = store.add("Test memory for events")
    
    print("    Updating memory...")
    store.update(mem.id, content="Updated test memory")
    
    print("    Creating branch...")
    store.create_branch("event-test")
    
    # Summary
    print(f"\n  Total events received: {len(events_received)}")
    for event_type, event_id in events_received:
        print(f"    • {event_type}: {event_id}")
    
    # Unregister hooks
    store.off(EventType.MEMORY_CREATED, on_memory_created)
    print("\n  Unregistered MEMORY_CREATED hook")
    
    store.close()
    return True


def demo_export_import():
    """Demonstrate export/import between agents."""
    print("\n" + "=" * 60)
    print("8. EXPORT/IMPORT")
    print("=" * 60)
    
    # Create temp file
    export_file = Path(tempfile.mktemp(suffix=".json"))
    
    try:
        # Alice exports her data
        print("\n  Alice exporting data...")
        alice = TeamMemoryStore(":memory:", agent_id="alice")
        
        alice.add("Shared insight: Customer churn is 5%")
        alice.add("Shared insight: NPS score is 72")
        alice.add("Shared insight: DAU is 10,000")
        
        count = alice.export_json(export_file)
        print(f"    Exported {count} memories to {export_file.name}")
        
        # Show file contents
        print(f"\n  Export file preview:")
        with open(export_file) as f:
            content = f.read()
            print(f"    {content[:200]}...")
        
        # Bob imports the data
        print("\n  Bob importing data...")
        bob = TeamMemoryStore(":memory:", agent_id="bob")
        
        imported = bob.import_json(export_file)
        print(f"    Imported {imported} memories")
        
        print("\n  Bob's memories after import:")
        for m in bob.list():
            print(f"    • {m.content}")
        
        alice.close()
        bob.close()
        
    finally:
        if export_file.exists():
            export_file.unlink()
    
    return True


def main():
    """Run all team collaboration demos."""
    print("\n" + "=" * 60)
    print("  AGENT MEMORY TOOLKIT - TEAM COLLABORATION DEMO")
    print("=" * 60)
    
    demos = [
        ("Basic Team Setup", demo_basic_team),
        ("Branching Workflow", demo_branching_workflow),
        ("Conflict Resolution", demo_conflict_resolution),
        ("Sync Protocol", demo_sync_protocol),
        ("Namespaces", demo_namespaces),
        ("Access Control", demo_access_control),
        ("Event Hooks", demo_event_hooks),
        ("Export/Import", demo_export_import),
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            success = demo_func()
            results.append((name, "✓" if success else "✗"))
        except Exception as e:
            print(f"\n  Error in {name}: {e}")
            results.append((name, "✗"))
    
    # Summary
    print("\n" + "=" * 60)
    print("  DEMO SUMMARY")
    print("=" * 60)
    
    for name, status in results:
        print(f"    {status} {name}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
