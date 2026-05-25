"""Knowledge graph store for agent memories.

This module implements a knowledge graph that stores entities and their 
relationships, enabling structured reasoning about agent memories. The 
graph supports entity extraction, relationship inference, and graph-based
queries for enhanced memory retrieval.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator

from .relationships import (
    Relationship,
    RelationshipType,
    RelationshipProperties,
    RelationshipCategory,
    get_inverse_relationship,
    get_relationship_category,
)

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    CONCEPT = "concept"           # Abstract concepts or ideas
    FACT = "fact"                 # Factual statements
    ENTITY = "entity"             # Named entities (people, places, things)
    EVENT = "event"               # Events or occurrences
    PROCEDURE = "procedure"       # Procedures or processes
    PREFERENCE = "preference"     # User preferences
    GOAL = "goal"                 # Goals or objectives
    MEMORY = "memory"             # Direct memory references
    TOPIC = "topic"               # Topic or subject clusters


@dataclass
class Entity:
    """An entity (node) in the knowledge graph."""
    entity_id: str
    name: str
    entity_type: EntityType
    description: str = ""
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source_memory_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5       # [0, 1] importance score
    access_count: int = 0         # How often this entity is accessed
    
    def __hash__(self) -> int:
        return hash(self.entity_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.entity_id == other.entity_id
    
    def add_alias(self, alias: str) -> None:
        """Add an alternative name for this entity."""
        if alias.lower() not in [a.lower() for a in self.aliases]:
            self.aliases.append(alias)
            self.updated_at = datetime.now()
    
    def link_memory(self, memory_id: str) -> None:
        """Link a memory to this entity."""
        if memory_id not in self.source_memory_ids:
            self.source_memory_ids.append(memory_id)
            self.updated_at = datetime.now()
    
    def record_access(self) -> None:
        """Record an access to this entity."""
        self.access_count += 1
        self.updated_at = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_memory_ids": self.source_memory_ids,
            "aliases": self.aliases,
            "properties": self.properties,
            "importance": self.importance,
            "access_count": self.access_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        """Create an entity from dictionary representation."""
        return cls(
            entity_id=data["entity_id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            description=data.get("description", ""),
            embedding=data.get("embedding"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            source_memory_ids=data.get("source_memory_ids", []),
            aliases=data.get("aliases", []),
            properties=data.get("properties", {}),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
        )


@dataclass
class KnowledgeGraphConfig:
    """Configuration for the knowledge graph store."""
    # Entity settings
    max_entities: int = 100000
    entity_merge_threshold: float = 0.9   # Similarity threshold for merging
    
    # Relationship settings
    max_relationships_per_entity: int = 1000
    default_relationship_weight: float = 1.0
    enable_inverse_relationships: bool = True
    
    # Decay settings
    enable_decay: bool = True
    decay_rate: float = 0.01
    min_weight_threshold: float = 0.1
    
    # Embedding settings
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 10000


@dataclass
class GraphStats:
    """Statistics about the knowledge graph."""
    total_entities: int = 0
    total_relationships: int = 0
    entities_by_type: dict[str, int] = field(default_factory=dict)
    relationships_by_type: dict[str, int] = field(default_factory=dict)
    avg_relationships_per_entity: float = 0.0
    most_connected_entities: list[tuple[str, int]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class KnowledgeGraphStore:
    """A knowledge graph store for managing entities and relationships.
    
    The knowledge graph provides a structured way to store and query
    relationships between concepts, facts, and other entities extracted
    from agent memories.
    
    Example:
        graph = KnowledgeGraphStore()
        
        # Add entities
        entity1 = graph.add_entity("Python", EntityType.CONCEPT, "A programming language")
        entity2 = graph.add_entity("Programming", EntityType.CONCEPT, "Writing code")
        
        # Add relationship
        graph.add_relationship(entity1.entity_id, entity2.entity_id, RelationshipType.IS_A)
        
        # Query the graph
        related = graph.get_related_entities(entity1.entity_id, max_depth=2)
    """
    
    def __init__(self, config: KnowledgeGraphConfig | None = None):
        """Initialize the knowledge graph store.
        
        Args:
            config: Configuration for the graph store.
        """
        self.config = config or KnowledgeGraphConfig()
        
        # Entity storage
        self._entities: dict[str, Entity] = {}
        self._entity_name_index: dict[str, set[str]] = defaultdict(set)  # name -> entity_ids
        self._entity_type_index: dict[EntityType, set[str]] = defaultdict(set)
        
        # Relationship storage (adjacency lists)
        self._outgoing: dict[str, dict[str, Relationship]] = defaultdict(dict)  # source_id -> {target_id -> rel}
        self._incoming: dict[str, dict[str, Relationship]] = defaultdict(dict)  # target_id -> {source_id -> rel}
        self._relationships: dict[str, Relationship] = {}  # relationship_id -> relationship
        
        # Caching
        self._path_cache: dict[tuple[str, str], list[list[str]]] | None = {} if self.config.enable_caching else None
        
        logger.info("Initialized KnowledgeGraphStore with config: %s", self.config)
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        description: str = "",
        embedding: list[float] | None = None,
        properties: dict[str, Any] | None = None,
        importance: float = 0.5,
        source_memory_id: str | None = None,
    ) -> Entity:
        """Add a new entity to the knowledge graph.
        
        Args:
            name: The name of the entity.
            entity_type: The type of entity.
            description: Optional description.
            embedding: Optional embedding vector.
            properties: Optional additional properties.
            importance: Importance score [0, 1].
            source_memory_id: Optional ID of the source memory.
            
        Returns:
            The created entity.
        """
        if len(self._entities) >= self.config.max_entities:
            raise ValueError(f"Maximum entities ({self.config.max_entities}) reached")
        
        entity_id = str(uuid.uuid4())
        entity = Entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            description=description,
            embedding=embedding,
            properties=properties or {},
            importance=importance,
            source_memory_ids=[source_memory_id] if source_memory_id else [],
        )
        
        self._entities[entity_id] = entity
        self._entity_name_index[name.lower()].add(entity_id)
        self._entity_type_index[entity_type].add(entity_id)
        
        logger.debug("Added entity: %s (%s)", name, entity_type.value)
        return entity
    
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID.
        
        Args:
            entity_id: The entity ID.
            
        Returns:
            The entity or None if not found.
        """
        entity = self._entities.get(entity_id)
        if entity:
            entity.record_access()
        return entity
    
    def find_entities_by_name(self, name: str, fuzzy: bool = False) -> list[Entity]:
        """Find entities by name.
        
        Args:
            name: The name to search for.
            fuzzy: If True, performs fuzzy matching.
            
        Returns:
            List of matching entities.
        """
        if fuzzy:
            # Simple fuzzy matching - check if name is contained
            matches = []
            name_lower = name.lower()
            for indexed_name, entity_ids in self._entity_name_index.items():
                if name_lower in indexed_name or indexed_name in name_lower:
                    for eid in entity_ids:
                        entity = self._entities.get(eid)
                        if entity:
                            matches.append(entity)
            return matches
        else:
            entity_ids = self._entity_name_index.get(name.lower(), set())
            return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def find_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Find all entities of a given type.
        
        Args:
            entity_type: The entity type to filter by.
            
        Returns:
            List of entities of that type.
        """
        entity_ids = self._entity_type_index.get(entity_type, set())
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def update_entity(
        self,
        entity_id: str,
        name: str | None = None,
        description: str | None = None,
        embedding: list[float] | None = None,
        properties: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> Entity | None:
        """Update an existing entity.
        
        Args:
            entity_id: The entity ID to update.
            name: New name (optional).
            description: New description (optional).
            embedding: New embedding (optional).
            properties: Properties to merge (optional).
            importance: New importance score (optional).
            
        Returns:
            The updated entity or None if not found.
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return None
        
        if name is not None and name != entity.name:
            # Update name index
            self._entity_name_index[entity.name.lower()].discard(entity_id)
            entity.name = name
            self._entity_name_index[name.lower()].add(entity_id)
        
        if description is not None:
            entity.description = description
        if embedding is not None:
            entity.embedding = embedding
        if properties is not None:
            entity.properties.update(properties)
        if importance is not None:
            entity.importance = importance
        
        entity.updated_at = datetime.now()
        self._invalidate_cache()
        
        return entity
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships.
        
        Args:
            entity_id: The entity ID to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return False
        
        # Remove all relationships involving this entity
        for target_id, rel in list(self._outgoing.get(entity_id, {}).items()):
            del self._relationships[rel.relationship_id]
            if entity_id in self._incoming.get(target_id, {}):
                del self._incoming[target_id][entity_id]
        
        for source_id, rel in list(self._incoming.get(entity_id, {}).items()):
            del self._relationships[rel.relationship_id]
            if entity_id in self._outgoing.get(source_id, {}):
                del self._outgoing[source_id][entity_id]
        
        # Remove from adjacency lists
        self._outgoing.pop(entity_id, None)
        self._incoming.pop(entity_id, None)
        
        # Remove from indices
        self._entity_name_index[entity.name.lower()].discard(entity_id)
        self._entity_type_index[entity.entity_type].discard(entity_id)
        
        # Remove entity
        del self._entities[entity_id]
        self._invalidate_cache()
        
        logger.debug("Deleted entity: %s", entity_id)
        return True
    
    def merge_entities(self, primary_id: str, secondary_id: str) -> Entity | None:
        """Merge two entities, keeping the primary and transferring relationships.
        
        Args:
            primary_id: The entity to keep.
            secondary_id: The entity to merge into primary.
            
        Returns:
            The merged entity or None if either not found.
        """
        primary = self._entities.get(primary_id)
        secondary = self._entities.get(secondary_id)
        
        if not primary or not secondary:
            return None
        
        # Merge aliases
        primary.add_alias(secondary.name)
        for alias in secondary.aliases:
            primary.add_alias(alias)
        
        # Merge source memories
        for mem_id in secondary.source_memory_ids:
            primary.link_memory(mem_id)
        
        # Merge properties
        for key, value in secondary.properties.items():
            if key not in primary.properties:
                primary.properties[key] = value
        
        # Transfer outgoing relationships
        for target_id, rel in list(self._outgoing.get(secondary_id, {}).items()):
            if target_id != primary_id:  # Avoid self-loops
                self.add_relationship(
                    primary_id, target_id, rel.relationship_type,
                    weight=rel.properties.weight,
                    confidence=rel.properties.confidence,
                )
        
        # Transfer incoming relationships
        for source_id, rel in list(self._incoming.get(secondary_id, {}).items()):
            if source_id != primary_id:  # Avoid self-loops
                self.add_relationship(
                    source_id, primary_id, rel.relationship_type,
                    weight=rel.properties.weight,
                    confidence=rel.properties.confidence,
                )
        
        # Delete secondary entity
        self.delete_entity(secondary_id)
        
        primary.importance = max(primary.importance, secondary.importance)
        primary.updated_at = datetime.now()
        
        logger.info("Merged entity %s into %s", secondary_id, primary_id)
        return primary
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        weight: float = 1.0,
        confidence: float = 1.0,
        bidirectional: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Relationship | None:
        """Add a relationship between two entities.
        
        Args:
            source_id: The source entity ID.
            target_id: The target entity ID.
            relationship_type: The type of relationship.
            weight: Relationship weight [0, 1].
            confidence: Confidence score [0, 1].
            bidirectional: Whether the relationship goes both ways.
            metadata: Optional additional metadata.
            
        Returns:
            The created relationship or None if entities not found.
        """
        if source_id not in self._entities or target_id not in self._entities:
            logger.warning("Cannot add relationship: entity not found")
            return None
        
        # Check if relationship already exists
        existing = self._outgoing.get(source_id, {}).get(target_id)
        if existing and existing.relationship_type == relationship_type:
            # Update existing relationship
            existing.properties.weight = max(existing.properties.weight, weight)
            existing.properties.confidence = max(existing.properties.confidence, confidence)
            existing.properties.updated_at = datetime.now()
            return existing
        
        relationship_id = str(uuid.uuid4())
        properties = RelationshipProperties(
            weight=weight,
            confidence=confidence,
            bidirectional=bidirectional,
            metadata=metadata or {},
        )
        
        relationship = Relationship(
            relationship_id=relationship_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=properties,
        )
        
        self._relationships[relationship_id] = relationship
        self._outgoing[source_id][target_id] = relationship
        self._incoming[target_id][source_id] = relationship
        
        # Add inverse relationship if enabled
        if self.config.enable_inverse_relationships:
            inverse_type = get_inverse_relationship(relationship_type)
            if inverse_type:
                self.add_relationship(
                    target_id, source_id, inverse_type,
                    weight=weight, confidence=confidence,
                )
        
        self._invalidate_cache()
        logger.debug(
            "Added relationship: %s -[%s]-> %s",
            source_id, relationship_type.value, target_id
        )
        return relationship
    
    def get_relationship(self, relationship_id: str) -> Relationship | None:
        """Get a relationship by ID.
        
        Args:
            relationship_id: The relationship ID.
            
        Returns:
            The relationship or None if not found.
        """
        return self._relationships.get(relationship_id)
    
    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_types: list[RelationshipType] | None = None,
    ) -> list[Relationship]:
        """Get relationships for an entity.
        
        Args:
            entity_id: The entity ID.
            direction: "outgoing", "incoming", or "both".
            relationship_types: Optional filter by relationship types.
            
        Returns:
            List of relationships.
        """
        relationships = []
        
        if direction in ("outgoing", "both"):
            for rel in self._outgoing.get(entity_id, {}).values():
                if relationship_types is None or rel.relationship_type in relationship_types:
                    relationships.append(rel)
        
        if direction in ("incoming", "both"):
            for rel in self._incoming.get(entity_id, {}).values():
                if relationship_types is None or rel.relationship_type in relationship_types:
                    relationships.append(rel)
        
        return relationships
    
    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship by ID.
        
        Args:
            relationship_id: The relationship ID to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        relationship = self._relationships.get(relationship_id)
        if not relationship:
            return False
        
        source_id = relationship.source_id
        target_id = relationship.target_id
        
        # Remove from adjacency lists
        if target_id in self._outgoing.get(source_id, {}):
            del self._outgoing[source_id][target_id]
        if source_id in self._incoming.get(target_id, {}):
            del self._incoming[target_id][source_id]
        
        # Remove relationship
        del self._relationships[relationship_id]
        self._invalidate_cache()
        
        return True
    
    def get_related_entities(
        self,
        entity_id: str,
        max_depth: int = 1,
        relationship_types: list[RelationshipType] | None = None,
        entity_types: list[EntityType] | None = None,
        min_weight: float = 0.0,
    ) -> list[tuple[Entity, int, Relationship]]:
        """Get entities related to the given entity.
        
        Args:
            entity_id: The starting entity ID.
            max_depth: Maximum depth to traverse (1 = direct neighbors).
            relationship_types: Optional filter by relationship types.
            entity_types: Optional filter by entity types.
            min_weight: Minimum relationship weight to consider.
            
        Returns:
            List of (entity, depth, relationship) tuples.
        """
        if entity_id not in self._entities:
            return []
        
        results = []
        visited = {entity_id}
        current_level = [(entity_id, None)]  # (id, relationship)
        
        for depth in range(1, max_depth + 1):
            next_level = []
            
            for current_id, _ in current_level:
                for rel in self._outgoing.get(current_id, {}).values():
                    if rel.properties.weight < min_weight:
                        continue
                    if relationship_types and rel.relationship_type not in relationship_types:
                        continue
                    
                    target = self._entities.get(rel.target_id)
                    if target and target.entity_id not in visited:
                        if entity_types is None or target.entity_type in entity_types:
                            results.append((target, depth, rel))
                        visited.add(target.entity_id)
                        next_level.append((target.entity_id, rel))
            
            current_level = next_level
            if not current_level:
                break
        
        return results
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        relationship_types: list[RelationshipType] | None = None,
    ) -> list[tuple[Entity, Relationship | None]] | None:
        """Find a path between two entities.
        
        Args:
            source_id: The starting entity ID.
            target_id: The target entity ID.
            max_depth: Maximum path length.
            relationship_types: Optional filter by relationship types.
            
        Returns:
            List of (entity, relationship) pairs forming the path, or None.
        """
        if source_id not in self._entities or target_id not in self._entities:
            return None
        
        if source_id == target_id:
            return []
        
        # Check cache
        cache_key = (source_id, target_id)
        if self._path_cache is not None and cache_key in self._path_cache:
            cached = self._path_cache[cache_key]
            if cached:
                return self._reconstruct_path(cached[0])
        
        # BFS to find shortest path
        queue = [(source_id, [])]  # (current_id, path)
        visited = {source_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            for rel in self._outgoing.get(current_id, {}).values():
                if relationship_types and rel.relationship_type not in relationship_types:
                    continue
                
                next_id = rel.target_id
                new_path = path + [(current_id, rel.relationship_id)]
                
                if next_id == target_id:
                    new_path.append((target_id, None))
                    # Cache result
                    if self._path_cache is not None:
                        path_ids = [p[0] for p in new_path]
                        self._path_cache[cache_key] = [path_ids]
                    return self._reconstruct_path([p[0] for p in new_path])
                
                if next_id not in visited and len(path) < max_depth:
                    visited.add(next_id)
                    queue.append((next_id, new_path))
        
        return None
    
    def _reconstruct_path(self, entity_ids: list[str]) -> list[tuple[Entity, Relationship | None]]:
        """Reconstruct a path from entity IDs."""
        path = []
        for i, entity_id in enumerate(entity_ids):
            entity = self._entities.get(entity_id)
            if entity:
                rel = None
                if i > 0:
                    prev_id = entity_ids[i - 1]
                    rel = self._outgoing.get(prev_id, {}).get(entity_id)
                path.append((entity, rel))
        return path
    
    def get_stats(self) -> GraphStats:
        """Get statistics about the knowledge graph.
        
        Returns:
            GraphStats object with graph statistics.
        """
        entities_by_type = defaultdict(int)
        for entity in self._entities.values():
            entities_by_type[entity.entity_type.value] += 1
        
        relationships_by_type = defaultdict(int)
        for rel in self._relationships.values():
            relationships_by_type[rel.relationship_type.value] += 1
        
        # Find most connected entities
        connection_counts = []
        for entity_id in self._entities:
            count = len(self._outgoing.get(entity_id, {})) + len(self._incoming.get(entity_id, {}))
            connection_counts.append((entity_id, count))
        connection_counts.sort(key=lambda x: x[1], reverse=True)
        
        total_entities = len(self._entities)
        total_relationships = len(self._relationships)
        
        return GraphStats(
            total_entities=total_entities,
            total_relationships=total_relationships,
            entities_by_type=dict(entities_by_type),
            relationships_by_type=dict(relationships_by_type),
            avg_relationships_per_entity=total_relationships / total_entities if total_entities > 0 else 0.0,
            most_connected_entities=connection_counts[:10],
            last_updated=datetime.now(),
        )
    
    def apply_decay(self) -> int:
        """Apply temporal decay to all relationships.
        
        Returns:
            Number of relationships removed due to decay.
        """
        if not self.config.enable_decay:
            return 0
        
        removed = 0
        to_remove = []
        
        for rel_id, rel in self._relationships.items():
            new_weight = rel.properties.decay(self.config.decay_rate)
            if new_weight < self.config.min_weight_threshold:
                to_remove.append(rel_id)
        
        for rel_id in to_remove:
            self.delete_relationship(rel_id)
            removed += 1
        
        if removed > 0:
            logger.info("Removed %d relationships due to decay", removed)
        
        return removed
    
    def _invalidate_cache(self) -> None:
        """Invalidate the path cache."""
        if self._path_cache is not None:
            self._path_cache.clear()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a dictionary.
        
        Returns:
            Dictionary representation of the graph.
        """
        return {
            "entities": {eid: e.to_dict() for eid, e in self._entities.items()},
            "relationships": {rid: r.to_dict() for rid, r in self._relationships.items()},
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any], config: KnowledgeGraphConfig | None = None) -> "KnowledgeGraphStore":
        """Deserialize a graph from a dictionary.
        
        Args:
            data: Dictionary representation.
            config: Optional configuration.
            
        Returns:
            KnowledgeGraphStore instance.
        """
        store = cls(config)
        
        # Load entities
        for entity_data in data.get("entities", {}).values():
            entity = Entity.from_dict(entity_data)
            store._entities[entity.entity_id] = entity
            store._entity_name_index[entity.name.lower()].add(entity.entity_id)
            store._entity_type_index[entity.entity_type].add(entity.entity_id)
        
        # Load relationships
        for rel_data in data.get("relationships", {}).values():
            rel = Relationship.from_dict(rel_data)
            store._relationships[rel.relationship_id] = rel
            store._outgoing[rel.source_id][rel.target_id] = rel
            store._incoming[rel.target_id][rel.source_id] = rel
        
        return store
    
    def __len__(self) -> int:
        return len(self._entities)
    
    def __iter__(self) -> Iterator[Entity]:
        return iter(self._entities.values())
