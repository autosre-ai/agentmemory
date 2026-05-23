"""Data models for the memory store."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json
import uuid


@dataclass
class MemoryMetadata:
    """Metadata associated with a memory."""

    source: str | None = None
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "confidence": self.confidence,
            "tags": self.tags,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryMetadata":
        """Create from dictionary."""
        return cls(
            source=data.get("source"),
            confidence=data.get("confidence", 1.0),
            tags=data.get("tags", []),
            extra=data.get("extra", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryMetadata":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    content: str
    metadata: MemoryMetadata
    created_at: datetime
    updated_at: datetime
    embedding: list[float] | None = None
    version: int = 1
    is_deleted: bool = False

    @classmethod
    def create(
        cls,
        content: str,
        metadata: MemoryMetadata | None = None,
        embedding: list[float] | None = None,
    ) -> "Memory":
        """Factory method to create a new memory with generated ID and timestamps."""
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata or MemoryMetadata(),
            created_at=now,
            updated_at=now,
            embedding=embedding,
            version=1,
            is_deleted=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "embedding": self.embedding,
            "version": self.version,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=MemoryMetadata.from_dict(data.get("metadata", {})),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            embedding=data.get("embedding"),
            version=data.get("version", 1),
            is_deleted=data.get("is_deleted", False),
        )


@dataclass
class SearchResult:
    """A search result with relevance score."""

    memory: Memory
    score: float
    match_type: str  # "fts", "vector", or "hybrid"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory": self.memory.to_dict(),
            "score": self.score,
            "match_type": self.match_type,
        }


@dataclass
class Commit:
    """A git-like commit representing a snapshot of memory state."""

    id: str
    branch: str
    parent_id: str | None
    message: str
    created_at: datetime
    memory_snapshot: dict[str, int]  # memory_id -> version at commit time

    @classmethod
    def create(
        cls,
        branch: str,
        parent_id: str | None,
        message: str,
        memory_snapshot: dict[str, int],
    ) -> "Commit":
        """Factory method to create a new commit."""
        return cls(
            id=str(uuid.uuid4()),
            branch=branch,
            parent_id=parent_id,
            message=message,
            created_at=datetime.utcnow(),
            memory_snapshot=memory_snapshot,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "branch": self.branch,
            "parent_id": self.parent_id,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "memory_snapshot": self.memory_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Commit":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            branch=data["branch"],
            parent_id=data.get("parent_id"),
            message=data["message"],
            created_at=datetime.fromisoformat(data["created_at"]),
            memory_snapshot=data.get("memory_snapshot", {}),
        )


@dataclass
class Branch:
    """A git-like branch for versioning."""

    name: str
    head_commit_id: str | None
    created_at: datetime
    is_active: bool = True

    @classmethod
    def create(cls, name: str, head_commit_id: str | None = None) -> "Branch":
        """Factory method to create a new branch."""
        return cls(
            name=name,
            head_commit_id=head_commit_id,
            created_at=datetime.utcnow(),
            is_active=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "head_commit_id": self.head_commit_id,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Branch":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            head_commit_id=data.get("head_commit_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            is_active=data.get("is_active", True),
        )
