"""
Team Memory Protocol - Git-like memory sharing for AI agents.

This module provides a production-ready system for sharing memories across
multiple AI agents with:

- **Git-like branching**: Create, checkout, merge branches
- **Conflict resolution**: Latest-wins, manual, merge, ours, theirs strategies
- **Agent namespaces**: Organize memories by context/domain
- **Sync protocol**: Push/pull via filesystem (extensible to HTTP)
- **Access control**: Fine-grained read/write permissions
- **Event hooks**: React to memory changes

Example:
    >>> from agent_memory_toolkit.team import TeamMemoryStore, ConflictResolution
    >>> 
    >>> # Create a store for agent "alice"
    >>> store = TeamMemoryStore("team.db", agent_id="alice")
    >>> 
    >>> # Add memories
    >>> store.add("Project deadline is next Friday")
    >>> store.add("Use Python 3.11+", namespace="technical")
    >>> 
    >>> # Branch for experiments
    >>> store.create_branch("experiment")
    >>> store.checkout("experiment")
    >>> 
    >>> # Sync with team
    >>> result = store.sync("/shared/memories")
    >>> print(f"Pushed: {result.memories_pushed}, Pulled: {result.memories_pulled}")
"""

from .models import (
    TeamMemory,
    TeamMemoryMetadata,
    TeamBranch,
    TeamCommit,
    ConflictResolution,
    Permission,
    MergeConflict,
    SyncResult,
    Event,
    EventType,
    EventHook,
    AccessRule,
)

from .team_memory import (
    TeamMemoryStore,
    Memory,  # Alias
    Branch,  # Alias
)

from .access_control import AccessControl

from .sync_protocol import SyncProtocol

from .exceptions import (
    TeamMemoryError,
    MemoryNotFoundError,
    BranchNotFoundError,
    BranchExistsError,
    CommitNotFoundError,
    MergeConflictError,
    PermissionDeniedError,
    SyncError,
    LockError,
    NamespaceNotFoundError,
)

__all__ = [
    # Main class
    "TeamMemoryStore",
    # Models
    "TeamMemory",
    "TeamMemoryMetadata",
    "TeamBranch",
    "TeamCommit",
    "MergeConflict",
    "SyncResult",
    "Event",
    "AccessRule",
    # Aliases
    "Memory",
    "Branch",
    # Enums
    "ConflictResolution",
    "Permission",
    "EventType",
    # Types
    "EventHook",
    # Components
    "AccessControl",
    "SyncProtocol",
    # Exceptions
    "TeamMemoryError",
    "MemoryNotFoundError",
    "BranchNotFoundError",
    "BranchExistsError",
    "CommitNotFoundError",
    "MergeConflictError",
    "PermissionDeniedError",
    "SyncError",
    "LockError",
    "NamespaceNotFoundError",
]
