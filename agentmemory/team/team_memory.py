"""
Team Memory Protocol - Git-like memory sharing for AI agents.

This module provides a production-ready system for sharing memories across
multiple AI agents with:
- Git-like branching (create, checkout, merge)
- Conflict resolution strategies
- Agent namespaces
- Filesystem-based sync protocol
- Access control
- Event hooks

Example:
    >>> from agentmemory.team import TeamMemoryStore
    >>> 
    >>> # Create a store for agent "alice"
    >>> store = TeamMemoryStore("team.db", agent_id="alice")
    >>> 
    >>> # Add a memory
    >>> memory = store.add("Important fact about the project")
    >>> 
    >>> # Create a branch
    >>> store.create_branch("experiment")
    >>> store.checkout("experiment")
    >>> 
    >>> # Sync with shared location
    >>> store.sync("/shared/memories")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Callable

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
)
from .exceptions import (
    TeamMemoryError,
    MemoryNotFoundError,
    BranchNotFoundError,
    BranchExistsError,
    CommitNotFoundError,
    MergeConflictError,
    PermissionDeniedError,
)
from .schema import apply_schema, run_migrations
from .access_control import AccessControl
from .sync_protocol import SyncProtocol

logger = logging.getLogger(__name__)


class TeamMemoryStore:
    """
    A memory store for team collaboration with Git-like versioning.
    
    Features:
    - Git-like branching (create, checkout, merge)
    - Multiple conflict resolution strategies
    - Agent namespaces for organization
    - Filesystem-based sync protocol
    - Fine-grained access control
    - Event hooks for extensibility
    
    Thread-safe with per-connection locking.
    """
    
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        agent_id: str | None = None,
        default_namespace: str = "default",
        conflict_strategy: ConflictResolution = ConflictResolution.LATEST_WINS,
    ):
        """
        Initialize the team memory store.
        
        Args:
            db_path: Path to SQLite database, or ":memory:" for in-memory
            agent_id: Unique identifier for this agent
            default_namespace: Default namespace for memories
            conflict_strategy: Default strategy for conflict resolution
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self._agent_id = agent_id or f"agent-{id(self)}"
        self._default_namespace = default_namespace
        self._conflict_strategy = conflict_strategy
        
        self._conn: sqlite3.Connection | None = None
        self._current_branch = "main"
        self._lock = threading.RLock()
        
        # Event hooks
        self._hooks: dict[EventType, list[EventHook]] = {t: [] for t in EventType}
        
        # Initialize database
        self._init_db()
        
        # Initialize access control
        self._access = AccessControl(self._conn, self._agent_id)
        
        # Initialize sync protocol
        self._sync = SyncProtocol(
            self._conn,
            self._agent_id,
            self._conflict_strategy,
        )
    
    @property
    def agent_id(self) -> str:
        """Current agent ID."""
        return self._agent_id
    
    @property
    def current_branch(self) -> str:
        """Current checked out branch."""
        return self._current_branch
    
    @property
    def access(self) -> AccessControl:
        """Access control manager."""
        return self._access
    
    def _init_db(self) -> None:
        """Initialize the database connection and schema."""
        self._conn = sqlite3.connect(
            self.db_path if isinstance(self.db_path, str) else str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        
        apply_schema(self._conn)
        run_migrations(self._conn)
        
        self._ensure_main_branch()
        self._ensure_default_namespace()
    
    def _ensure_main_branch(self) -> None:
        """Ensure the main branch exists."""
        cursor = self._conn.execute(
            "SELECT name FROM team_branches WHERE name = 'main'"
        )
        if cursor.fetchone() is None:
            branch = TeamBranch.create("main", created_by=self._agent_id)
            self._conn.execute(
                """
                INSERT INTO team_branches 
                (name, head_commit_id, created_at, created_by, parent_branch, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    branch.name,
                    branch.head_commit_id,
                    branch.created_at.isoformat(),
                    branch.created_by,
                    branch.parent_branch,
                    int(branch.is_active),
                ),
            )
            self._conn.commit()
    
    def _ensure_default_namespace(self) -> None:
        """Ensure the default namespace exists."""
        cursor = self._conn.execute(
            "SELECT name FROM team_namespaces WHERE name = ?",
            (self._default_namespace,),
        )
        if cursor.fetchone() is None:
            self._conn.execute(
                """
                INSERT INTO team_namespaces (name, description, created_at, created_by, is_active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (
                    self._default_namespace,
                    "Default namespace",
                    datetime.utcnow().isoformat(),
                    self._agent_id,
                ),
            )
            self._conn.commit()
    
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for database transactions."""
        with self._lock:
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "TeamMemoryStore":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    # ==================== Event Hooks ====================
    
    def on(self, event_type: EventType, hook: EventHook) -> None:
        """
        Register an event hook.
        
        Args:
            event_type: Type of event to listen for
            hook: Callback function that receives Event objects
        """
        self._hooks[event_type].append(hook)
        self._sync.set_event_hooks(self._get_all_hooks())
    
    def off(self, event_type: EventType, hook: EventHook) -> bool:
        """
        Unregister an event hook.
        
        Args:
            event_type: Type of event
            hook: Hook to remove
            
        Returns:
            True if hook was removed
        """
        try:
            self._hooks[event_type].remove(hook)
            self._sync.set_event_hooks(self._get_all_hooks())
            return True
        except ValueError:
            return False
    
    def _get_all_hooks(self) -> list[EventHook]:
        """Get flat list of all hooks for sync protocol."""
        all_hooks = []
        for hooks in self._hooks.values():
            all_hooks.extend(hooks)
        return all_hooks
    
    def _emit_event(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """Emit an event to registered hooks."""
        event = Event.create(event_type, self._agent_id, data)
        
        # Log to database
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO team_events (id, event_type, timestamp, agent_id, data_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.type.name,
                    event.timestamp.isoformat(),
                    event.agent_id,
                    json.dumps(event.data),
                ),
            )
            self._conn.commit()
        
        # Notify hooks
        for hook in self._hooks[event_type]:
            try:
                hook(event)
            except Exception as e:
                logger.warning(f"Event hook error for {event_type.name}: {e}")
    
    # ==================== CRUD Operations ====================
    
    def add(
        self,
        content: str,
        metadata: TeamMemoryMetadata | dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> TeamMemory:
        """
        Add a new memory.
        
        Args:
            content: Memory content text
            metadata: Optional metadata
            namespace: Namespace (defaults to default_namespace)
            
        Returns:
            Created TeamMemory object
        """
        if isinstance(metadata, dict):
            metadata = TeamMemoryMetadata.from_dict(metadata)
        elif metadata is None:
            metadata = TeamMemoryMetadata()
        
        namespace = namespace or self._default_namespace
        metadata.namespace = namespace
        metadata.agent_id = self._agent_id
        
        # Check write permission
        self._access.check_permission(
            self._agent_id,
            Permission.WRITE,
            namespace,
            "add memory",
        )
        
        memory = TeamMemory.create(
            content=content,
            metadata=metadata,
            branch=self._current_branch,
            agent_id=self._agent_id,
        )
        
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO team_memories
                (id, content, metadata_json, created_at, updated_at, version,
                 is_deleted, branch, namespace, agent_id, vector_clock_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.metadata.to_json(),
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.version,
                    int(memory.is_deleted),
                    memory.branch,
                    namespace,
                    self._agent_id,
                    json.dumps(memory.vector_clock),
                ),
            )
            
            # Record version
            self._record_version(memory, "create")
        
        self._emit_event(EventType.MEMORY_CREATED, {"memory_id": memory.id})
        
        logger.debug(f"Added memory: {memory.id}")
        return memory
    
    def get(self, memory_id: str) -> TeamMemory:
        """
        Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            TeamMemory object
            
        Raises:
            MemoryNotFoundError: If not found
        """
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT id, content, metadata_json, created_at, updated_at,
                       version, is_deleted, branch, namespace, agent_id, vector_clock_json
                FROM team_memories
                WHERE id = ? AND branch = ? AND is_deleted = 0
                """,
                (memory_id, self._current_branch),
            )
            row = cursor.fetchone()
        
        if row is None:
            raise MemoryNotFoundError(memory_id)
        
        memory = self._row_to_memory(row)
        
        # Check read permission
        self._access.check_permission(
            self._agent_id,
            Permission.READ,
            memory.metadata.namespace,
            "read memory",
        )
        
        return memory
    
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: TeamMemoryMetadata | dict[str, Any] | None = None,
    ) -> TeamMemory:
        """
        Update an existing memory.
        
        Args:
            memory_id: Memory ID to update
            content: New content (optional)
            metadata: New metadata (optional)
            
        Returns:
            Updated TeamMemory object
        """
        memory = self.get(memory_id)
        
        # Check write permission
        self._access.check_permission(
            self._agent_id,
            Permission.WRITE,
            memory.metadata.namespace,
            "update memory",
        )
        
        if content is not None:
            memory.content = content
        
        if metadata is not None:
            if isinstance(metadata, dict):
                memory.metadata = TeamMemoryMetadata.from_dict(metadata)
            else:
                memory.metadata = metadata
        
        memory.updated_at = datetime.utcnow()
        memory.version += 1
        memory.vector_clock[self._agent_id] = memory.vector_clock.get(self._agent_id, 0) + 1
        
        with self.transaction():
            self._conn.execute(
                """
                UPDATE team_memories SET
                    content = ?,
                    metadata_json = ?,
                    updated_at = ?,
                    version = ?,
                    vector_clock_json = ?
                WHERE id = ? AND branch = ?
                """,
                (
                    memory.content,
                    memory.metadata.to_json(),
                    memory.updated_at.isoformat(),
                    memory.version,
                    json.dumps(memory.vector_clock),
                    memory_id,
                    self._current_branch,
                ),
            )
            
            self._record_version(memory, "update")
        
        self._emit_event(EventType.MEMORY_UPDATED, {"memory_id": memory_id})
        
        return memory
    
    def delete(self, memory_id: str, hard: bool = False) -> None:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID to delete
            hard: Permanently delete if True, soft delete if False
        """
        memory = self.get(memory_id)
        
        # Check write permission
        self._access.check_permission(
            self._agent_id,
            Permission.WRITE,
            memory.metadata.namespace,
            "delete memory",
        )
        
        with self.transaction():
            if hard:
                self._conn.execute(
                    "DELETE FROM team_memory_versions WHERE memory_id = ?",
                    (memory_id,),
                )
                self._conn.execute(
                    "DELETE FROM team_memories WHERE id = ? AND branch = ?",
                    (memory_id, self._current_branch),
                )
            else:
                memory.is_deleted = True
                memory.updated_at = datetime.utcnow()
                memory.version += 1
                
                self._conn.execute(
                    """
                    UPDATE team_memories SET
                        is_deleted = 1,
                        updated_at = ?,
                        version = ?
                    WHERE id = ? AND branch = ?
                    """,
                    (
                        memory.updated_at.isoformat(),
                        memory.version,
                        memory_id,
                        self._current_branch,
                    ),
                )
                
                self._record_version(memory, "delete")
        
        self._emit_event(EventType.MEMORY_DELETED, {"memory_id": memory_id, "hard": hard})
    
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        namespace: str | None = None,
        tag: str | None = None,
        include_deleted: bool = False,
    ) -> list[TeamMemory]:
        """
        List memories with filtering and pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number to skip
            namespace: Filter by namespace
            tag: Filter by tag
            include_deleted: Include soft-deleted memories
            
        Returns:
            List of TeamMemory objects
        """
        query = """
            SELECT id, content, metadata_json, created_at, updated_at,
                   version, is_deleted, branch, namespace, agent_id, vector_clock_json
            FROM team_memories
            WHERE branch = ?
        """
        params: list[Any] = [self._current_branch]
        
        if not include_deleted:
            query += " AND is_deleted = 0"
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        if tag:
            query += " AND json_extract(metadata_json, '$.tags') LIKE ?"
            params.append(f'%"{tag}"%')
        
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
        
        return [self._row_to_memory(row) for row in rows]
    
    def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
    ) -> list[TeamMemory]:
        """
        Full-text search memories.
        
        Args:
            query: Search query (FTS5 syntax)
            limit: Maximum results
            namespace: Filter by namespace
            
        Returns:
            List of matching TeamMemory objects
        """
        sql = """
            SELECT m.id, m.content, m.metadata_json, m.created_at, m.updated_at,
                   m.version, m.is_deleted, m.branch, m.namespace, m.agent_id, m.vector_clock_json
            FROM team_memories_fts f
            JOIN team_memories m ON f.rowid = m.rowid
            WHERE team_memories_fts MATCH ?
              AND m.branch = ?
              AND m.is_deleted = 0
        """
        params: list[Any] = [query, self._current_branch]
        
        if namespace:
            sql += " AND m.namespace = ?"
            params.append(namespace)
        
        sql += " LIMIT ?"
        params.append(limit)
        
        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        
        return [self._row_to_memory(row) for row in rows]
    
    def _record_version(self, memory: TeamMemory, operation: str) -> None:
        """Record a memory version for history."""
        self._conn.execute(
            """
            INSERT INTO team_memory_versions
            (memory_id, content, metadata_json, version, created_at, operation, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.metadata.to_json(),
                memory.version,
                datetime.utcnow().isoformat(),
                operation,
                self._agent_id,
            ),
        )
    
    def _row_to_memory(self, row: sqlite3.Row) -> TeamMemory:
        """Convert database row to TeamMemory."""
        return TeamMemory(
            id=row["id"],
            content=row["content"],
            metadata=TeamMemoryMetadata.from_json(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            version=row["version"],
            is_deleted=bool(row["is_deleted"]),
            branch=row["branch"],
            vector_clock=json.loads(row["vector_clock_json"]) if row["vector_clock_json"] else {},
        )
    
    # ==================== Branch Operations ====================
    
    def create_branch(
        self,
        name: str,
        from_branch: str | None = None,
    ) -> TeamBranch:
        """
        Create a new branch.
        
        Args:
            name: Branch name
            from_branch: Parent branch (defaults to current)
            
        Returns:
            Created TeamBranch
            
        Raises:
            BranchExistsError: If branch already exists
        """
        from_branch = from_branch or self._current_branch
        
        with self._lock:
            # Check if branch exists
            cursor = self._conn.execute(
                "SELECT name FROM team_branches WHERE name = ?",
                (name,),
            )
            if cursor.fetchone():
                raise BranchExistsError(name)
            
            # Get parent branch head
            cursor = self._conn.execute(
                "SELECT head_commit_id FROM team_branches WHERE name = ?",
                (from_branch,),
            )
            parent_row = cursor.fetchone()
            if parent_row is None:
                raise BranchNotFoundError(from_branch)
            
            parent_head = parent_row[0]
            
            branch = TeamBranch.create(
                name=name,
                head_commit_id=parent_head,
                created_by=self._agent_id,
                parent_branch=from_branch,
            )
            
            self._conn.execute(
                """
                INSERT INTO team_branches
                (name, head_commit_id, created_at, created_by, parent_branch, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    branch.name,
                    branch.head_commit_id,
                    branch.created_at.isoformat(),
                    branch.created_by,
                    branch.parent_branch,
                    int(branch.is_active),
                ),
            )
            
            # Copy memories from parent branch
            self._conn.execute(
                """
                INSERT INTO team_memories
                (id, content, metadata_json, created_at, updated_at, version,
                 is_deleted, branch, namespace, agent_id, vector_clock_json)
                SELECT 
                    id || '-' || ?, content, metadata_json, created_at, ?, version,
                    is_deleted, ?, namespace, agent_id, vector_clock_json
                FROM team_memories
                WHERE branch = ? AND is_deleted = 0
                """,
                (
                    name[:8],  # Add suffix for new IDs
                    datetime.utcnow().isoformat(),
                    name,
                    from_branch,
                ),
            )
            
            self._conn.commit()
        
        self._emit_event(EventType.BRANCH_CREATED, {
            "branch": name,
            "from_branch": from_branch,
        })
        
        return branch
    
    def checkout(self, branch_name: str) -> None:
        """
        Switch to a different branch.
        
        Args:
            branch_name: Branch to switch to
            
        Raises:
            BranchNotFoundError: If branch doesn't exist
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT name FROM team_branches WHERE name = ? AND is_active = 1",
                (branch_name,),
            )
            if cursor.fetchone() is None:
                raise BranchNotFoundError(branch_name)
            
            self._current_branch = branch_name
    
    def merge(
        self,
        source_branch: str,
        conflict_strategy: ConflictResolution | None = None,
    ) -> list[MergeConflict]:
        """
        Merge another branch into the current branch.
        
        Args:
            source_branch: Branch to merge from
            conflict_strategy: Override default conflict strategy
            
        Returns:
            List of unresolved conflicts (empty if all resolved)
            
        Raises:
            BranchNotFoundError: If source branch doesn't exist
            MergeConflictError: If MANUAL strategy and conflicts exist
        """
        strategy = conflict_strategy or self._conflict_strategy
        
        with self._lock:
            # Verify source branch exists
            cursor = self._conn.execute(
                "SELECT name FROM team_branches WHERE name = ? AND is_active = 1",
                (source_branch,),
            )
            if cursor.fetchone() is None:
                raise BranchNotFoundError(source_branch)
            
            # Get source memories
            cursor = self._conn.execute(
                """
                SELECT id, content, metadata_json, created_at, updated_at,
                       version, is_deleted, branch, namespace, agent_id, vector_clock_json
                FROM team_memories
                WHERE branch = ?
                """,
                (source_branch,),
            )
            source_memories = {row["id"]: self._row_to_memory(row) for row in cursor.fetchall()}
            
            # Get target memories
            cursor = self._conn.execute(
                """
                SELECT id, content, metadata_json, created_at, updated_at,
                       version, is_deleted, branch, namespace, agent_id, vector_clock_json
                FROM team_memories
                WHERE branch = ?
                """,
                (self._current_branch,),
            )
            target_memories = {row["id"]: self._row_to_memory(row) for row in cursor.fetchall()}
        
        conflicts: list[MergeConflict] = []
        
        for memory_id, source_mem in source_memories.items():
            # Strip branch suffix from ID if present
            base_id = memory_id.rsplit("-", 1)[0] if "-" in memory_id else memory_id
            
            # Find matching memory in target
            target_mem = None
            for tid, tm in target_memories.items():
                target_base = tid.rsplit("-", 1)[0] if "-" in tid else tid
                if target_base == base_id:
                    target_mem = tm
                    break
            
            if target_mem is None:
                # New memory - copy to target
                self._copy_memory_to_branch(source_mem, self._current_branch)
            elif source_mem.version != target_mem.version:
                # Potential conflict
                conflict = self._handle_merge_conflict(
                    target_mem, source_mem, strategy
                )
                if conflict:
                    conflicts.append(conflict)
        
        if conflicts and strategy == ConflictResolution.MANUAL:
            raise MergeConflictError(conflicts)
        
        self._emit_event(EventType.BRANCH_MERGED, {
            "source": source_branch,
            "target": self._current_branch,
            "conflicts": len(conflicts),
        })
        
        return conflicts
    
    def _copy_memory_to_branch(self, memory: TeamMemory, branch: str) -> None:
        """Copy a memory to a different branch."""
        new_id = f"{memory.id}-{branch[:8]}"
        
        self._conn.execute(
            """
            INSERT OR REPLACE INTO team_memories
            (id, content, metadata_json, created_at, updated_at, version,
             is_deleted, branch, namespace, agent_id, vector_clock_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                memory.content,
                memory.metadata.to_json(),
                memory.created_at.isoformat(),
                datetime.utcnow().isoformat(),
                memory.version,
                int(memory.is_deleted),
                branch,
                memory.metadata.namespace,
                self._agent_id,
                json.dumps(memory.vector_clock),
            ),
        )
        self._conn.commit()
    
    def _handle_merge_conflict(
        self,
        local: TeamMemory,
        remote: TeamMemory,
        strategy: ConflictResolution,
    ) -> MergeConflict | None:
        """Handle a merge conflict based on strategy."""
        conflict = MergeConflict(
            memory_id=local.id,
            local_version=local,
            remote_version=remote,
            conflict_type="both_modified",
        )
        
        self._emit_event(EventType.CONFLICT_DETECTED, {
            "memory_id": local.id,
            "strategy": strategy.name,
        })
        
        if strategy == ConflictResolution.LATEST_WINS:
            if remote.updated_at > local.updated_at:
                self._apply_remote_changes(local, remote)
            self._emit_event(EventType.CONFLICT_RESOLVED, {
                "memory_id": local.id,
                "winner": "remote" if remote.updated_at > local.updated_at else "local",
            })
            return None
            
        elif strategy == ConflictResolution.OURS:
            self._emit_event(EventType.CONFLICT_RESOLVED, {
                "memory_id": local.id,
                "winner": "local",
            })
            return None
            
        elif strategy == ConflictResolution.THEIRS:
            self._apply_remote_changes(local, remote)
            self._emit_event(EventType.CONFLICT_RESOLVED, {
                "memory_id": local.id,
                "winner": "remote",
            })
            return None
            
        elif strategy == ConflictResolution.MERGE:
            if local.content == remote.content:
                # Same content - merge metadata
                merged_metadata = TeamMemoryMetadata(
                    source=local.metadata.source or remote.metadata.source,
                    confidence=max(local.metadata.confidence, remote.metadata.confidence),
                    tags=list(set(local.metadata.tags + remote.metadata.tags)),
                    agent_id=self._agent_id,
                    namespace=local.metadata.namespace,
                    extra={**remote.metadata.extra, **local.metadata.extra},
                )
                self.update(local.id, metadata=merged_metadata)
                self._emit_event(EventType.CONFLICT_RESOLVED, {
                    "memory_id": local.id,
                    "winner": "merged",
                })
                return None
            # Content differs - fall through to return conflict
        
        # MANUAL or unresolved MERGE
        return conflict
    
    def _apply_remote_changes(self, local: TeamMemory, remote: TeamMemory) -> None:
        """Apply remote memory changes to local."""
        self._conn.execute(
            """
            UPDATE team_memories SET
                content = ?,
                metadata_json = ?,
                updated_at = ?,
                version = ?,
                vector_clock_json = ?
            WHERE id = ? AND branch = ?
            """,
            (
                remote.content,
                remote.metadata.to_json(),
                datetime.utcnow().isoformat(),
                remote.version,
                json.dumps(remote.vector_clock),
                local.id,
                self._current_branch,
            ),
        )
        self._conn.commit()
    
    def list_branches(self, include_inactive: bool = False) -> list[TeamBranch]:
        """List all branches."""
        query = "SELECT * FROM team_branches"
        if not include_inactive:
            query += " WHERE is_active = 1"
        
        with self._lock:
            cursor = self._conn.execute(query)
            return [
                TeamBranch(
                    name=row["name"],
                    head_commit_id=row["head_commit_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    created_by=row["created_by"],
                    parent_branch=row["parent_branch"],
                    is_active=bool(row["is_active"]),
                )
                for row in cursor.fetchall()
            ]
    
    def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """
        Delete a branch.
        
        Args:
            branch_name: Branch to delete
            force: Delete even if not merged
        """
        if branch_name == "main":
            raise TeamMemoryError("Cannot delete main branch")
        
        if branch_name == self._current_branch:
            raise TeamMemoryError("Cannot delete current branch")
        
        with self.transaction():
            # Soft delete - mark as inactive
            self._conn.execute(
                "UPDATE team_branches SET is_active = 0 WHERE name = ?",
                (branch_name,),
            )
            
            if force:
                # Hard delete memories
                self._conn.execute(
                    "DELETE FROM team_memories WHERE branch = ?",
                    (branch_name,),
                )
        
        self._emit_event(EventType.BRANCH_DELETED, {"branch": branch_name, "force": force})
    
    # ==================== Namespace Operations ====================
    
    def create_namespace(self, name: str, description: str = "") -> None:
        """Create a new namespace."""
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO team_namespaces (name, description, created_at, created_by, is_active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (name, description, datetime.utcnow().isoformat(), self._agent_id),
            )
    
    def list_namespaces(self) -> list[str]:
        """List all active namespaces."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT name FROM team_namespaces WHERE is_active = 1"
            )
            return [row[0] for row in cursor.fetchall()]
    
    # ==================== Commit Operations ====================
    
    def commit(self, message: str) -> TeamCommit:
        """
        Create a commit of current state.
        
        Args:
            message: Commit message
            
        Returns:
            Created TeamCommit
        """
        with self._lock:
            # Get current head
            cursor = self._conn.execute(
                "SELECT head_commit_id FROM team_branches WHERE name = ?",
                (self._current_branch,),
            )
            row = cursor.fetchone()
            parent_id = row[0] if row else None
            
            # Build memory snapshot
            cursor = self._conn.execute(
                "SELECT id, version FROM team_memories WHERE branch = ? AND is_deleted = 0",
                (self._current_branch,),
            )
            snapshot = {row[0]: row[1] for row in cursor.fetchall()}
            
            commit = TeamCommit.create(
                branch=self._current_branch,
                parent_id=parent_id,
                message=message,
                memory_snapshot=snapshot,
                created_by=self._agent_id,
            )
            
            self._conn.execute(
                """
                INSERT INTO team_commits
                (id, branch, parent_id, message, created_at, created_by, memory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    commit.id,
                    commit.branch,
                    commit.parent_id,
                    commit.message,
                    commit.created_at.isoformat(),
                    commit.created_by,
                    json.dumps(commit.memory_snapshot),
                ),
            )
            
            # Update branch head
            self._conn.execute(
                "UPDATE team_branches SET head_commit_id = ? WHERE name = ?",
                (commit.id, self._current_branch),
            )
            
            self._conn.commit()
        
        return commit
    
    def log(self, limit: int = 10) -> list[TeamCommit]:
        """Get commit history for current branch."""
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT id, branch, parent_id, message, created_at, created_by, memory_snapshot_json
                FROM team_commits
                WHERE branch = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (self._current_branch, limit),
            )
            return [
                TeamCommit(
                    id=row[0],
                    branch=row[1],
                    parent_id=row[2],
                    message=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    created_by=row[5],
                    memory_snapshot=json.loads(row[6]) if row[6] else {},
                )
                for row in cursor.fetchall()
            ]
    
    # ==================== Sync Operations ====================
    
    def push(
        self,
        remote_path: str | Path,
        branch: str | None = None,
        namespace: str | None = None,
    ) -> SyncResult:
        """
        Push local changes to remote.
        
        Args:
            remote_path: Path to remote sync directory
            branch: Branch to push (defaults to current)
            namespace: Only push this namespace
            
        Returns:
            SyncResult with statistics
        """
        branch = branch or self._current_branch
        return self._sync.push(remote_path, branch, namespace)
    
    def pull(
        self,
        remote_path: str | Path,
        branch: str | None = None,
        namespace: str | None = None,
    ) -> SyncResult:
        """
        Pull remote changes to local.
        
        Args:
            remote_path: Path to remote sync directory
            branch: Branch to pull (defaults to current)
            namespace: Only pull this namespace
            
        Returns:
            SyncResult with statistics and conflicts
        """
        branch = branch or self._current_branch
        return self._sync.pull(remote_path, branch, namespace)
    
    def sync(
        self,
        remote_path: str | Path,
        branch: str | None = None,
        namespace: str | None = None,
    ) -> SyncResult:
        """
        Full bidirectional sync.
        
        Args:
            remote_path: Path to remote sync directory
            branch: Branch to sync (defaults to current)
            namespace: Only sync this namespace
            
        Returns:
            SyncResult with combined statistics
        """
        branch = branch or self._current_branch
        return self._sync.sync(remote_path, branch, namespace)
    
    # ==================== Export/Import ====================
    
    def export_json(self, path: str | Path, namespace: str | None = None) -> int:
        """
        Export memories to JSON file.
        
        Args:
            path: Output file path
            namespace: Only export this namespace
            
        Returns:
            Number of memories exported
        """
        memories = self.list(limit=10000, namespace=namespace)
        
        data = {
            "version": "1.0",
            "agent_id": self._agent_id,
            "branch": self._current_branch,
            "exported_at": datetime.utcnow().isoformat(),
            "memories": [m.to_dict() for m in memories],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        return len(memories)
    
    def import_json(self, path: str | Path, namespace: str | None = None) -> int:
        """
        Import memories from JSON file.
        
        Args:
            path: Input file path
            namespace: Override namespace for all imports
            
        Returns:
            Number of memories imported
        """
        with open(path) as f:
            data = json.load(f)
        
        count = 0
        for mem_data in data.get("memories", []):
            memory = TeamMemory.from_dict(mem_data)
            
            if namespace:
                memory.metadata.namespace = namespace
            
            # Check if exists
            try:
                existing = self.get(memory.id)
                if memory.version > existing.version:
                    self.update(memory.id, memory.content, memory.metadata)
                    count += 1
            except MemoryNotFoundError:
                self.add(memory.content, memory.metadata, memory.metadata.namespace)
                count += 1
        
        return count


# Alias for backwards compatibility
Memory = TeamMemory
Branch = TeamBranch
