"""Relationship types and edge definitions for the knowledge graph.

This module defines the relationship types, edge properties, and 
relationship patterns used to connect entities in the knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RelationshipCategory(Enum):
    """Categories of relationships in the knowledge graph."""
    SEMANTIC = "semantic"           # Meaning-based relationships (is_a, similar_to)
    TEMPORAL = "temporal"           # Time-based relationships (before, after, during)
    CAUSAL = "causal"               # Cause-effect relationships (causes, enables)
    SPATIAL = "spatial"             # Location-based relationships (located_in, near)
    ASSOCIATIVE = "associative"     # Loose associations (related_to, mentioned_with)
    HIERARCHICAL = "hierarchical"   # Parent-child relationships (contains, part_of)
    PROCEDURAL = "procedural"       # Process relationships (follows, precedes)
    SOCIAL = "social"               # Social relationships (knows, works_with)


class RelationshipType(Enum):
    """Standard relationship types in the knowledge graph."""
    # Semantic relationships
    IS_A = "is_a"
    INSTANCE_OF = "instance_of"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    SYNONYM_OF = "synonym_of"
    DEFINED_AS = "defined_as"
    
    # Temporal relationships
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    OVERLAPS = "overlaps"
    
    # Causal relationships
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    ENABLES = "enables"
    PREVENTS = "prevents"
    REQUIRES = "requires"
    LEADS_TO = "leads_to"
    
    # Spatial relationships
    LOCATED_IN = "located_in"
    CONTAINS = "contains"
    NEAR = "near"
    ADJACENT_TO = "adjacent_to"
    
    # Associative relationships
    RELATED_TO = "related_to"
    MENTIONED_WITH = "mentioned_with"
    ASSOCIATED_WITH = "associated_with"
    REFERS_TO = "refers_to"
    
    # Hierarchical relationships
    PART_OF = "part_of"
    HAS_PART = "has_part"
    BELONGS_TO = "belongs_to"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    
    # Procedural relationships
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    STEP_OF = "step_of"
    DEPENDS_ON = "depends_on"
    
    # Social relationships
    KNOWS = "knows"
    WORKS_WITH = "works_with"
    CREATED_BY = "created_by"
    OWNED_BY = "owned_by"
    
    # Memory-specific relationships
    DERIVED_FROM = "derived_from"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    ELABORATES = "elaborates"
    SUMMARIZES = "summarizes"
    UPDATES = "updates"


@dataclass
class RelationshipProperties:
    """Properties of a relationship between entities."""
    weight: float = 1.0                   # Strength of the relationship [0, 1]
    confidence: float = 1.0               # Confidence in the relationship [0, 1]
    bidirectional: bool = False           # Whether the relationship goes both ways
    temporal: bool = False                # Whether the relationship changes over time
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: str | None = None             # Where this relationship was derived from
    evidence: list[str] = field(default_factory=list)  # Supporting evidence
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def update_weight(self, delta: float, min_weight: float = 0.0, max_weight: float = 1.0) -> None:
        """Update the relationship weight within bounds."""
        self.weight = max(min_weight, min(max_weight, self.weight + delta))
        self.updated_at = datetime.now()
    
    def add_evidence(self, evidence: str) -> None:
        """Add supporting evidence for this relationship."""
        if evidence not in self.evidence:
            self.evidence.append(evidence)
            self.updated_at = datetime.now()
    
    def decay(self, decay_rate: float = 0.1) -> float:
        """Apply temporal decay to the relationship weight."""
        self.weight = max(0.0, self.weight - decay_rate)
        self.updated_at = datetime.now()
        return self.weight


@dataclass
class Relationship:
    """A relationship between two entities in the knowledge graph."""
    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: RelationshipProperties = field(default_factory=RelationshipProperties)
    
    def __hash__(self) -> int:
        return hash(self.relationship_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relationship):
            return False
        return self.relationship_id == other.relationship_id
    
    def to_dict(self) -> dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "weight": self.properties.weight,
            "confidence": self.properties.confidence,
            "bidirectional": self.properties.bidirectional,
            "created_at": self.properties.created_at.isoformat(),
            "updated_at": self.properties.updated_at.isoformat(),
            "source": self.properties.source,
            "evidence": self.properties.evidence,
            "metadata": self.properties.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relationship":
        """Create a relationship from dictionary representation."""
        properties = RelationshipProperties(
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            bidirectional=data.get("bidirectional", False),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            source=data.get("source"),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
        )
        return cls(
            relationship_id=data["relationship_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=RelationshipType(data["relationship_type"]),
            properties=properties,
        )


@dataclass
class RelationshipPattern:
    """A pattern for extracting or matching relationships."""
    pattern_id: str
    name: str
    source_type: str | None = None        # Expected entity type for source
    target_type: str | None = None        # Expected entity type for target
    relationship_types: list[RelationshipType] = field(default_factory=list)
    category: RelationshipCategory | None = None
    regex_patterns: list[str] = field(default_factory=list)  # Text patterns
    keywords: list[str] = field(default_factory=list)        # Trigger keywords
    
    def matches_text(self, text: str) -> bool:
        """Check if text matches any of the pattern's keywords."""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.keywords)


# Standard relationship patterns for common use cases
STANDARD_PATTERNS: list[RelationshipPattern] = [
    RelationshipPattern(
        pattern_id="causation",
        name="Causation Pattern",
        relationship_types=[RelationshipType.CAUSES, RelationshipType.LEADS_TO, RelationshipType.ENABLES],
        category=RelationshipCategory.CAUSAL,
        keywords=["because", "therefore", "thus", "causes", "leads to", "results in", "enables"],
    ),
    RelationshipPattern(
        pattern_id="temporal_sequence",
        name="Temporal Sequence Pattern",
        relationship_types=[RelationshipType.BEFORE, RelationshipType.AFTER, RelationshipType.FOLLOWS],
        category=RelationshipCategory.TEMPORAL,
        keywords=["before", "after", "then", "next", "subsequently", "previously", "followed by"],
    ),
    RelationshipPattern(
        pattern_id="hierarchy",
        name="Hierarchy Pattern",
        relationship_types=[RelationshipType.IS_A, RelationshipType.PART_OF, RelationshipType.BELONGS_TO],
        category=RelationshipCategory.HIERARCHICAL,
        keywords=["is a", "type of", "part of", "belongs to", "includes", "contains"],
    ),
    RelationshipPattern(
        pattern_id="similarity",
        name="Similarity Pattern",
        relationship_types=[RelationshipType.SIMILAR_TO, RelationshipType.RELATED_TO],
        category=RelationshipCategory.SEMANTIC,
        keywords=["similar to", "like", "resembles", "comparable to", "related to"],
    ),
    RelationshipPattern(
        pattern_id="contradiction",
        name="Contradiction Pattern",
        relationship_types=[RelationshipType.CONTRADICTS, RelationshipType.OPPOSITE_OF],
        category=RelationshipCategory.SEMANTIC,
        keywords=["contradicts", "opposes", "conflicts with", "unlike", "contrary to", "however", "but"],
    ),
]


def get_inverse_relationship(rel_type: RelationshipType) -> RelationshipType | None:
    """Get the inverse of a relationship type, if one exists."""
    inverses = {
        RelationshipType.IS_A: None,  # No direct inverse
        RelationshipType.CAUSES: RelationshipType.CAUSED_BY,
        RelationshipType.CAUSED_BY: RelationshipType.CAUSES,
        RelationshipType.BEFORE: RelationshipType.AFTER,
        RelationshipType.AFTER: RelationshipType.BEFORE,
        RelationshipType.PART_OF: RelationshipType.HAS_PART,
        RelationshipType.HAS_PART: RelationshipType.PART_OF,
        RelationshipType.CONTAINS: RelationshipType.LOCATED_IN,
        RelationshipType.LOCATED_IN: RelationshipType.CONTAINS,
        RelationshipType.PARENT_OF: RelationshipType.CHILD_OF,
        RelationshipType.CHILD_OF: RelationshipType.PARENT_OF,
        RelationshipType.FOLLOWS: RelationshipType.PRECEDES,
        RelationshipType.PRECEDES: RelationshipType.FOLLOWS,
        RelationshipType.CREATED_BY: None,
        RelationshipType.DERIVED_FROM: None,
    }
    return inverses.get(rel_type)


def get_relationship_category(rel_type: RelationshipType) -> RelationshipCategory:
    """Get the category of a relationship type."""
    category_mapping = {
        # Semantic
        RelationshipType.IS_A: RelationshipCategory.SEMANTIC,
        RelationshipType.INSTANCE_OF: RelationshipCategory.SEMANTIC,
        RelationshipType.SIMILAR_TO: RelationshipCategory.SEMANTIC,
        RelationshipType.OPPOSITE_OF: RelationshipCategory.SEMANTIC,
        RelationshipType.SYNONYM_OF: RelationshipCategory.SEMANTIC,
        RelationshipType.DEFINED_AS: RelationshipCategory.SEMANTIC,
        # Temporal
        RelationshipType.BEFORE: RelationshipCategory.TEMPORAL,
        RelationshipType.AFTER: RelationshipCategory.TEMPORAL,
        RelationshipType.DURING: RelationshipCategory.TEMPORAL,
        RelationshipType.STARTS_WITH: RelationshipCategory.TEMPORAL,
        RelationshipType.ENDS_WITH: RelationshipCategory.TEMPORAL,
        RelationshipType.OVERLAPS: RelationshipCategory.TEMPORAL,
        # Causal
        RelationshipType.CAUSES: RelationshipCategory.CAUSAL,
        RelationshipType.CAUSED_BY: RelationshipCategory.CAUSAL,
        RelationshipType.ENABLES: RelationshipCategory.CAUSAL,
        RelationshipType.PREVENTS: RelationshipCategory.CAUSAL,
        RelationshipType.REQUIRES: RelationshipCategory.CAUSAL,
        RelationshipType.LEADS_TO: RelationshipCategory.CAUSAL,
        # Spatial
        RelationshipType.LOCATED_IN: RelationshipCategory.SPATIAL,
        RelationshipType.CONTAINS: RelationshipCategory.SPATIAL,
        RelationshipType.NEAR: RelationshipCategory.SPATIAL,
        RelationshipType.ADJACENT_TO: RelationshipCategory.SPATIAL,
        # Associative
        RelationshipType.RELATED_TO: RelationshipCategory.ASSOCIATIVE,
        RelationshipType.MENTIONED_WITH: RelationshipCategory.ASSOCIATIVE,
        RelationshipType.ASSOCIATED_WITH: RelationshipCategory.ASSOCIATIVE,
        RelationshipType.REFERS_TO: RelationshipCategory.ASSOCIATIVE,
        # Hierarchical
        RelationshipType.PART_OF: RelationshipCategory.HIERARCHICAL,
        RelationshipType.HAS_PART: RelationshipCategory.HIERARCHICAL,
        RelationshipType.BELONGS_TO: RelationshipCategory.HIERARCHICAL,
        RelationshipType.PARENT_OF: RelationshipCategory.HIERARCHICAL,
        RelationshipType.CHILD_OF: RelationshipCategory.HIERARCHICAL,
        # Procedural
        RelationshipType.FOLLOWS: RelationshipCategory.PROCEDURAL,
        RelationshipType.PRECEDES: RelationshipCategory.PROCEDURAL,
        RelationshipType.STEP_OF: RelationshipCategory.PROCEDURAL,
        RelationshipType.DEPENDS_ON: RelationshipCategory.PROCEDURAL,
        # Social
        RelationshipType.KNOWS: RelationshipCategory.SOCIAL,
        RelationshipType.WORKS_WITH: RelationshipCategory.SOCIAL,
        RelationshipType.CREATED_BY: RelationshipCategory.SOCIAL,
        RelationshipType.OWNED_BY: RelationshipCategory.SOCIAL,
    }
    return category_mapping.get(rel_type, RelationshipCategory.ASSOCIATIVE)
