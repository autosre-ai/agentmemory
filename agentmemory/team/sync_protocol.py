"""Sync protocol for Team Memory - filesystem-based sync."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import filelock

from .models import (
    TeamMemory,
    TeamMemoryMetadata,
    TeamBranch,
    TeamCommit,
    ConflictResolution,
    MergeConflict,
    SyncResult,
    Event,
    EventType,
)
from .exceptions import SyncError, LockError

logger = logging.getLogger(__name__)


class SyncProtocol:
    """
    Filesystem-based sync protocol for team memories.
    
    This protocol uses a shared filesystem location (could be a network drive,
    cloud-synced folder, or local directory) to synchronize memories between
    agents.
    
    Directory structure:
    remote_path/
    ├── manifests/
    │   └── {branch}.json         # Branch state manifest
    ├── memories/
    │   └── {memory_id}.json      # Individual memory files
    ├── commits/
    │   └── {commit_id}.json      # Commit files
    └── locks/
        └── {resource}.lock       # Lock files
    """
    
    def __init__(
        self,
        conn: sqlite3.Connection,
        agent_id: str,
        conflict_strategy: ConflictResolution = ConflictResolution.LATEST_WINS,
        lock_timeout: float = 30.0,
    ):
        """
        Initialize sync protocol.
        
        Args:
            conn: SQLite database connection
            agent_id: Current agent's ID
            conflict_strategy: Strategy for resolving conflicts
            lock_timeout: Timeout for acquiring locks (seconds)
        """
        self._conn = conn
        self._agent_id = agent_id
        self._conflict_strategy = conflict_strategy
        self._lock_timeout = lock_timeout
        self._event_hooks: list = []
    
    def set_event_hooks(self, hooks: list) -> None:
        """Set event hooks for sync events."""
        self._event_hooks = hooks
    
    def _emit_event(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """Emit an event to all registered hooks."""
        event = Event.create(event_type, self._agent_id, data)
        for hook in self._event_hooks:
            try:
                hook(event)
            except Exception as e:
                logger.warning(f"Event hook error: {e}")
    
    def push(
        self,
        remote_path: str | Path,
        branch: str = "main",
        namespace: str | None = None,
    ) -> SyncResult:
        """
        Push local memories to remote.
        
        Args:
            remote_path: Path to remote sync directory
            branch: Branch to push
            namespace: Only push memories in this namespace
            
        Returns:
            SyncResult with push statistics
        """
        remote = Path(remote_path)
        self._ensure_remote_structure(remote)
        
        lock_path = remote / "locks" / f"{branch}.lock"
        lock = filelock.FileLock(str(lock_path), timeout=self._lock_timeout)
        
        try:
            with lock:
                return self._do_push(remote, branch, namespace)
        except filelock.Timeout:
            raise LockError(f"branch:{branch}", self._lock_timeout)
    
    def _do_push(
        self,
        remote: Path,
        branch: str,
        namespace: str | None,
    ) -> SyncResult:
        """Execute push operation while holding lock."""
        result = SyncResult(success=True)
        
        # Get local memories
        query = "SELECT * FROM team_memories WHERE branch = ?"
        params: list[Any] = [branch]
        
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        
        cursor = self._conn.execute(query, params)
        local_memories = list(cursor.fetchall())
        
        # Load remote manifest
        manifest_path = remote / "manifests" / f"{branch}.json"
        remote_manifest = self._load_manifest(manifest_path)
        
        # Compare and push changed memories
        pushed_count = 0
        
        for row in local_memories:
            memory_id = row[0]
            local_version = row[5]  # version column
            local_updated = row[4]  # updated_at column
            
            remote_info = remote_manifest.get("memories", {}).get(memory_id, {})
            remote_version = remote_info.get("version", 0)
            
            if local_version > remote_version:
                # Push this memory
                memory_data = self._row_to_memory_dict(row)
                memory_path = remote / "memories" / f"{memory_id}.json"
                
                with open(memory_path, "w") as f:
                    json.dump(memory_data, f, indent=2)
                
                # Update manifest
                if "memories" not in remote_manifest:
                    remote_manifest["memories"] = {}
                
                remote_manifest["memories"][memory_id] = {
                    "version": local_version,
                    "updated_at": local_updated,
                    "agent_id": self._agent_id,
                }
                
                pushed_count += 1
        
        # Update manifest
        remote_manifest["last_push"] = datetime.utcnow().isoformat()
        remote_manifest["last_push_by"] = self._agent_id
        
        with open(manifest_path, "w") as f:
            json.dump(remote_manifest, f, indent=2)
        
        result.memories_pushed = pushed_count
        
        self._emit_event(EventType.SYNC_PUSH, {
            "remote": str(remote),
            "branch": branch,
            "memories_pushed": pushed_count,
        })
        
        return result
    
    def pull(
        self,
        remote_path: str | Path,
        branch: str = "main",
        namespace: str | None = None,
    ) -> SyncResult:
        """
        Pull remote memories to local.
        
        Args:
            remote_path: Path to remote sync directory
            branch: Branch to pull
            namespace: Only pull memories in this namespace
            
        Returns:
            SyncResult with pull statistics and any conflicts
        """
        remote = Path(remote_path)
        
        if not remote.exists():
            raise SyncError(f"Remote path does not exist: {remote}")
        
        lock_path = remote / "locks" / f"{branch}.lock"
        lock = filelock.FileLock(str(lock_path), timeout=self._lock_timeout)
        
        try:
            with lock:
                return self._do_pull(remote, branch, namespace)
        except filelock.Timeout:
            raise LockError(f"branch:{branch}", self._lock_timeout)
    
    def _do_pull(
        self,
        remote: Path,
        branch: str,
        namespace: str | None,
    ) -> SyncResult:
        """Execute pull operation while holding lock."""
        result = SyncResult(success=True)
        
        # Load remote manifest
        manifest_path = remote / "manifests" / f"{branch}.json"
        remote_manifest = self._load_manifest(manifest_path)
        
        pulled_count = 0
        conflicts: list[MergeConflict] = []
        
        # Process each memory in remote manifest
        for memory_id, remote_info in remote_manifest.get("memories", {}).items():
            memory_path = remote / "memories" / f"{memory_id}.json"
            
            if not memory_path.exists():
                continue
            
            with open(memory_path) as f:
                remote_data = json.load(f)
            
            # Check namespace filter
            if namespace and remote_data.get("metadata", {}).get("namespace") != namespace:
                continue
            
            # Get local version
            cursor = self._conn.execute(
                "SELECT * FROM team_memories WHERE id = ? AND branch = ?",
                (memory_id, branch),
            )
            local_row = cursor.fetchone()
            
            if local_row is None:
                # New memory from remote - insert
                self._insert_memory_from_dict(remote_data, branch)
                pulled_count += 1
            else:
                # Existing memory - check for conflict
                local_version = local_row[5]
                remote_version = remote_data.get("version", 1)
                
                local_updated = datetime.fromisoformat(local_row[4])
                remote_updated = datetime.fromisoformat(remote_data.get("updated_at"))
                
                # Check if local has changes remote doesn't know about
                local_agent = local_row[9]  # agent_id column
                remote_agent = remote_data.get("metadata", {}).get("agent_id")
                
                if local_version != remote_version and local_agent != remote_agent:
                    # Potential conflict
                    conflict = self._resolve_conflict(
                        local_row, remote_data, local_updated, remote_updated
                    )
                    
                    if conflict:
                        conflicts.append(conflict)
                    else:
                        pulled_count += 1
                elif remote_version > local_version:
                    # Remote is newer - update
                    self._update_memory_from_dict(remote_data, branch)
                    pulled_count += 1
        
        result.memories_pulled = pulled_count
        result.conflicts = conflicts
        result.success = len(conflicts) == 0 or self._conflict_strategy != ConflictResolution.MANUAL
        
        self._emit_event(EventType.SYNC_PULL, {
            "remote": str(remote),
            "branch": branch,
            "memories_pulled": pulled_count,
            "conflicts": len(conflicts),
        })
        
        return result
    
    def _resolve_conflict(
        self,
        local_row: sqlite3.Row,
        remote_data: dict[str, Any],
        local_updated: datetime,
        remote_updated: datetime,
    ) -> MergeConflict | None:
        """
        Resolve a conflict between local and remote versions.
        
        Returns:
            MergeConflict if manual resolution needed, None if resolved
        """
        local_memory = self._row_to_memory(local_row)
        remote_memory = TeamMemory.from_dict(remote_data)
        
        conflict = MergeConflict(
            memory_id=local_memory.id,
            local_version=local_memory,
            remote_version=remote_memory,
            conflict_type="both_modified",
        )
        
        self._emit_event(EventType.CONFLICT_DETECTED, {
            "memory_id": local_memory.id,
            "strategy": self._conflict_strategy.name,
        })
        
        if self._conflict_strategy == ConflictResolution.LATEST_WINS:
            if remote_updated > local_updated:
                self._update_memory_from_dict(remote_data, local_memory.branch)
            # Otherwise keep local
            self._emit_event(EventType.CONFLICT_RESOLVED, {
                "memory_id": local_memory.id,
                "winner": "remote" if remote_updated > local_updated else "local",
            })
            return None
            
        elif self._conflict_strategy == ConflictResolution.OURS:
            # Keep local, do nothing
            self._emit_event(EventType.CONFLICT_RESOLVED, {
                "memory_id": local_memory.id,
                "winner": "local",
            })
            return None
            
        elif self._conflict_strategy == ConflictResolution.THEIRS:
            self._update_memory_from_dict(remote_data, local_memory.branch)
            self._emit_event(EventType.CONFLICT_RESOLVED, {
                "memory_id": local_memory.id,
                "winner": "remote",
            })
            return None
            
        elif self._conflict_strategy == ConflictResolution.MERGE:
            # Attempt automatic merge
            merged = self._auto_merge(local_memory, remote_memory)
            if merged:
                self._update_memory_from_obj(merged)
                self._emit_event(EventType.CONFLICT_RESOLVED, {
                    "memory_id": local_memory.id,
                    "winner": "merged",
                })
                return None
            # Fall through to manual
            
        # MANUAL or failed merge
        return conflict
    
    def _auto_merge(
        self,
        local: TeamMemory,
        remote: TeamMemory,
    ) -> TeamMemory | None:
        """
        Attempt to automatically merge two memory versions.
        
        Returns:
            Merged memory if successful, None if manual intervention needed
        """
        # Simple merge: if content is the same, merge metadata
        if local.content == remote.content:
            merged = TeamMemory(
                id=local.id,
                content=local.content,
                metadata=TeamMemoryMetadata(
                    source=local.metadata.source or remote.metadata.source,
                    confidence=max(local.metadata.confidence, remote.metadata.confidence),
                    tags=list(set(local.metadata.tags + remote.metadata.tags)),
                    agent_id=self._agent_id,
                    namespace=local.metadata.namespace,
                    extra={**remote.metadata.extra, **local.metadata.extra},
                ),
                created_at=min(local.created_at, remote.created_at),
                updated_at=datetime.utcnow(),
                version=max(local.version, remote.version) + 1,
                is_deleted=local.is_deleted and remote.is_deleted,
                branch=local.branch,
                vector_clock=self._merge_vector_clocks(local.vector_clock, remote.vector_clock),
            )
            return merged
        
        return None
    
    def _merge_vector_clocks(
        self,
        clock1: dict[str, int],
        clock2: dict[str, int],
    ) -> dict[str, int]:
        """Merge two vector clocks by taking max of each entry."""
        merged = dict(clock1)
        for agent, version in clock2.items():
            merged[agent] = max(merged.get(agent, 0), version)
        merged[self._agent_id] = merged.get(self._agent_id, 0) + 1
        return merged
    
    def sync(
        self,
        remote_path: str | Path,
        branch: str = "main",
        namespace: str | None = None,
    ) -> SyncResult:
        """
        Full bidirectional sync (pull then push).
        
        Args:
            remote_path: Path to remote sync directory
            branch: Branch to sync
            namespace: Only sync memories in this namespace
            
        Returns:
            Combined SyncResult
        """
        # Pull first to get latest
        pull_result = self.pull(remote_path, branch, namespace)
        
        if not pull_result.success and self._conflict_strategy == ConflictResolution.MANUAL:
            return pull_result
        
        # Then push local changes
        push_result = self.push(remote_path, branch, namespace)
        
        # Combine results
        return SyncResult(
            success=pull_result.success and push_result.success,
            memories_pushed=push_result.memories_pushed,
            memories_pulled=pull_result.memories_pulled,
            conflicts=pull_result.conflicts,
            errors=pull_result.errors + push_result.errors,
        )
    
    def _ensure_remote_structure(self, remote: Path) -> None:
        """Create remote directory structure if needed."""
        (remote / "manifests").mkdir(parents=True, exist_ok=True)
        (remote / "memories").mkdir(parents=True, exist_ok=True)
        (remote / "commits").mkdir(parents=True, exist_ok=True)
        (remote / "locks").mkdir(parents=True, exist_ok=True)
    
    def _load_manifest(self, path: Path) -> dict[str, Any]:
        """Load manifest file or return empty dict."""
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}
    
    def _row_to_memory_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a memory dict."""
        metadata = json.loads(row[2])  # metadata_json
        vector_clock = json.loads(row[10]) if len(row) > 10 and row[10] else {}
        
        return {
            "id": row[0],
            "content": row[1],
            "metadata": metadata,
            "created_at": row[3],
            "updated_at": row[4],
            "version": row[5],
            "is_deleted": bool(row[6]),
            "branch": row[7],
            "vector_clock": vector_clock,
        }
    
    def _row_to_memory(self, row: sqlite3.Row) -> TeamMemory:
        """Convert a database row to a TeamMemory object."""
        return TeamMemory.from_dict(self._row_to_memory_dict(row))
    
    def _insert_memory_from_dict(self, data: dict[str, Any], branch: str) -> None:
        """Insert a memory from a dict."""
        metadata = data.get("metadata", {})
        self._conn.execute(
            """
            INSERT INTO team_memories 
            (id, content, metadata_json, created_at, updated_at, version, 
             is_deleted, branch, namespace, agent_id, vector_clock_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["content"],
                json.dumps(metadata),
                data["created_at"],
                data["updated_at"],
                data.get("version", 1),
                int(data.get("is_deleted", False)),
                branch,
                metadata.get("namespace", "default"),
                metadata.get("agent_id"),
                json.dumps(data.get("vector_clock", {})),
            ),
        )
        self._conn.commit()
    
    def _update_memory_from_dict(self, data: dict[str, Any], branch: str) -> None:
        """Update a memory from a dict."""
        metadata = data.get("metadata", {})
        self._conn.execute(
            """
            UPDATE team_memories SET
                content = ?,
                metadata_json = ?,
                updated_at = ?,
                version = ?,
                is_deleted = ?,
                vector_clock_json = ?
            WHERE id = ? AND branch = ?
            """,
            (
                data["content"],
                json.dumps(metadata),
                data["updated_at"],
                data.get("version", 1),
                int(data.get("is_deleted", False)),
                json.dumps(data.get("vector_clock", {})),
                data["id"],
                branch,
            ),
        )
        self._conn.commit()
    
    def _update_memory_from_obj(self, memory: TeamMemory) -> None:
        """Update a memory from a TeamMemory object."""
        self._update_memory_from_dict(memory.to_dict(), memory.branch)
