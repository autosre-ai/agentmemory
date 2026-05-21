"""Data models for the knowledge graph module."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum
import json
import uuid


class EntityType(Enum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    DATE = "date"
    EVENT = "event"
    PRODUCT = "product"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """An entity extracted from text."""
    
    id: str
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    mention_count: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create(
        cls,
        name: str,
        entity_type: EntityType,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Entity":
        """Factory method to create a new entity."""
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            entity_type=entity_type,
            aliases=aliases or [],
            metadata=metadata or {},
            mention_count=1,
            created_at=now,
            updated_at=now,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "mention_count": self.mention_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            aliases=data.get("aliases", []),
            metadata=data.get("metadata", {}),
            mention_count=data.get("mention_count", 1),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class Relation:
    """A relation between two entities (a triple)."""
    
    id: str
    subject_id: str  # Entity ID
    predicate: str   # Relation type (e.g., "works_at", "located_in")
    object_id: str   # Entity ID
    confidence: float = 1.0
    source_memory_id: str | None = None  # Link back to source memory
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create(
        cls,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float = 1.0,
        source_memory_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Relation":
        """Factory method to create a new relation."""
        return cls(
            id=str(uuid.uuid4()),
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            source_memory_id=source_memory_id,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "source_memory_id": self.source_memory_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            subject_id=data["subject_id"],
            predicate=data["predicate"],
            object_id=data["object_id"],
            confidence=data.get("confidence", 1.0),
            source_memory_id=data.get("source_memory_id"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class EntityMention:
    """A mention of an entity in text (before resolution to Entity)."""
    
    text: str
    entity_type: EntityType
    start: int  # Character offset
    end: int    # Character offset
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.text.lower(), self.entity_type))
    
    def __eq__(self, other):
        if not isinstance(other, EntityMention):
            return False
        return self.text.lower() == other.text.lower() and self.entity_type == other.entity_type


@dataclass
class RelationMention:
    """A potential relation extracted from text (before entity resolution)."""
    
    subject_text: str
    predicate: str
    object_text: str
    confidence: float = 1.0


@dataclass
class GraphSearchResult:
    """Result from a graph search operation."""
    
    entity: Entity
    relations: list[tuple[Relation, Entity]]  # Related (relation, connected_entity)
    distance: int = 0  # Hops from query entity
    score: float = 1.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity": self.entity.to_dict(),
            "relations": [
                {"relation": r.to_dict(), "connected_entity": e.to_dict()}
                for r, e in self.relations
            ],
            "distance": self.distance,
            "score": self.score,
        }
