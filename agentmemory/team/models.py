"""Data models for Team Memory Protocol."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable
import json
import uuid


class ConflictResolution(Enum):
    """Strategies for resolving memory conflicts during merge."""
    
    LATEST_WINS = auto()  # Most recently updated memory wins
    MANUAL = auto()  # Raise exception for manual resolution
    MERGE = auto()  # Attempt to merge content (for compatible types)
    OURS = auto()  # Always keep local version
    THEIRS = auto()  # Always take remote version


class Permission(Enum):
    """Access permission levels."""
    
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3  # Can manage access control


class EventType(Enum):
    """Types of events that can trigger hooks."""
    
    MEMORY_CREATED = auto()
    MEMORY_UPDATED = auto()
    MEMORY_DELETED = auto()
    BRANCH_CREATED = auto()
    BRANCH_MERGED = auto()
    BRANCH_DELETED = auto()
    SYNC_PUSH = auto()
    SYNC_PULL = auto()
    CONFLICT_DETECTED = auto()
    CONFLICT_RESOLVED = auto()


@dataclass
class TeamMemoryMetadata:
    """Extended metadata for team memories."""
    
    source: str | None = None
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    agent_id: str | None = None
    namespace: str = "default"
    extra: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "tags": self.tags,
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "extra": self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamMemoryMetadata":
        return cls(
            source=data.get("source"),
            confidence=data.get("confidence", 1.0),
            tags=data.get("tags", []),
            agent_id=data.get("agent_id"),
            namespace=data.get("namespace", "default"),
            extra=data.get("extra", {}),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "TeamMemoryMetadata":
        return cls.from_dict(json.loads(json_str))


@dataclass
class TeamMemory:
    """A memory entry with team collaboration features."""
    
    id: str
    content: str
    metadata: TeamMemoryMetadata
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_deleted: bool = False
    branch: str = "main"
    vector_clock: dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        content: str,
        metadata: TeamMemoryMetadata | None = None,
        branch: str = "main",
        agent_id: str | None = None,
    ) -> "TeamMemory":
        now = datetime.utcnow()
        if metadata is None:
            metadata = TeamMemoryMetadata()
        if agent_id:
            metadata.agent_id = agent_id
        
        memory_id = str(uuid.uuid4())
        vector_clock = {agent_id: 1} if agent_id else {}
        
        return cls(
            id=memory_id,
            content=content,
            metadata=metadata,
            created_at=now,
            updated_at=now,
            version=1,
            is_deleted=False,
            branch=branch,
            vector_clock=vector_clock,
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "is_deleted": self.is_deleted,
            "branch": self.branch,
            "vector_clock": self.vector_clock,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamMemory":
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=TeamMemoryMetadata.from_dict(data.get("metadata", {})),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", 1),
            is_deleted=data.get("is_deleted", False),
            branch=data.get("branch", "main"),
            vector_clock=data.get("vector_clock", {}),
        )


@dataclass
class TeamBranch:
    """A branch for team memory versioning."""
    
    name: str
    head_commit_id: str | None
    created_at: datetime
    created_by: str | None = None
    parent_branch: str | None = None
    is_active: bool = True
    
    @classmethod
    def create(
        cls,
        name: str,
        head_commit_id: str | None = None,
        created_by: str | None = None,
        parent_branch: str | None = None,
    ) -> "TeamBranch":
        return cls(
            name=name,
            head_commit_id=head_commit_id,
            created_at=datetime.utcnow(),
            created_by=created_by,
            parent_branch=parent_branch,
            is_active=True,
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "head_commit_id": self.head_commit_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "parent_branch": self.parent_branch,
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamBranch":
        return cls(
            name=data["name"],
            head_commit_id=data.get("head_commit_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by"),
            parent_branch=data.get("parent_branch"),
            is_active=data.get("is_active", True),
        )


@dataclass
class TeamCommit:
    """A commit representing a snapshot of memory state."""
    
    id: str
    branch: str
    parent_id: str | None
    message: str
    created_at: datetime
    created_by: str | None = None
    memory_snapshot: dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        branch: str,
        parent_id: str | None,
        message: str,
        memory_snapshot: dict[str, int],
        created_by: str | None = None,
    ) -> "TeamCommit":
        return cls(
            id=str(uuid.uuid4()),
            branch=branch,
            parent_id=parent_id,
            message=message,
            created_at=datetime.utcnow(),
            created_by=created_by,
            memory_snapshot=memory_snapshot,
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "branch": self.branch,
            "parent_id": self.parent_id,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "memory_snapshot": self.memory_snapshot,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeamCommit":
        return cls(
            id=data["id"],
            branch=data["branch"],
            parent_id=data.get("parent_id"),
            message=data["message"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by"),
            memory_snapshot=data.get("memory_snapshot", {}),
        )


@dataclass
class AccessRule:
    """Access control rule for an agent or namespace."""
    
    agent_id: str | None  # None means applies to all agents
    namespace: str | None  # None means applies to all namespaces
    permission: Permission
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "permission": self.permission.value,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccessRule":
        return cls(
            agent_id=data.get("agent_id"),
            namespace=data.get("namespace"),
            permission=Permission(data["permission"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class MergeConflict:
    """Represents a conflict during merge."""
    
    memory_id: str
    local_version: TeamMemory
    remote_version: TeamMemory
    conflict_type: str  # "content", "deletion", "both_modified"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "local_version": self.local_version.to_dict(),
            "remote_version": self.remote_version.to_dict(),
            "conflict_type": self.conflict_type,
        }


@dataclass 
class SyncResult:
    """Result of a sync operation."""
    
    success: bool
    memories_pushed: int = 0
    memories_pulled: int = 0
    conflicts: list[MergeConflict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "memories_pushed": self.memories_pushed,
            "memories_pulled": self.memories_pulled,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "errors": self.errors,
        }


@dataclass
class Event:
    """An event in the team memory system."""
    
    id: str
    type: EventType
    timestamp: datetime
    agent_id: str | None
    data: dict[str, Any]
    
    @classmethod
    def create(
        cls,
        event_type: EventType,
        agent_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> "Event":
        return cls(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            data=data or {},
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "data": self.data,
        }


# Type alias for event hooks
EventHook = Callable[[Event], None]
