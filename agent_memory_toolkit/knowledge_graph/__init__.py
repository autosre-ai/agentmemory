"""
Knowledge Graph module for agent-memory-toolkit.

Provides entity extraction, relation extraction, and graph-based retrieval
to augment the memory store with structured knowledge.
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .models import (
    Entity,
    EntityType,
    Relation,
    EntityMention,
    RelationMention,
    GraphSearchResult,
)
from .extractor import (
    EntityExtractor,
    RelationExtractor,
    CombinedExtractor,
)
from .graph_store import GraphStore

if TYPE_CHECKING:
    from ..store.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Knowledge Graph for entity and relation management.
    
    Provides:
    - Entity extraction from text (people, organizations, locations, etc.)
    - Relation extraction (subject-predicate-object triples)
    - Graph storage and querying
    - Integration with MemoryStore for graph-augmented retrieval
    
    Example:
        >>> kg = KnowledgeGraph()
        >>> # Extract and store from text
        >>> kg.process_text("John Smith works at Acme Corp in New York.")
        >>> # Search entities
        >>> results = kg.search_entities("John")
        >>> # Find related entities
        >>> neighbors = kg.get_related("John Smith")
        
        # With MemoryStore integration
        >>> from agent_memory_toolkit.store import MemoryStore
        >>> store = MemoryStore("memories.db")
        >>> kg = KnowledgeGraph.from_memory_store(store)
        >>> kg.index_memory(memory)  # Extract entities from memory
        >>> results = kg.graph_augmented_search("John", store)
    """
    
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        connection: sqlite3.Connection | None = None,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
    ):
        """
        Initialize the knowledge graph.
        
        Args:
            db_path: Path to SQLite database, or ":memory:" for in-memory
            connection: Optional existing SQLite connection (for sharing with MemoryStore)
            entity_extractor: Custom entity extractor
            relation_extractor: Custom relation extractor
        """
        self.graph_store = GraphStore(db_path=db_path, connection=connection)
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.relation_extractor = relation_extractor or RelationExtractor()
        self.combined_extractor = CombinedExtractor(
            entity_extractor=self.entity_extractor,
            relation_extractor=self.relation_extractor,
        )
    
    @classmethod
    def from_memory_store(
        cls,
        memory_store: "MemoryStore",
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
    ) -> "KnowledgeGraph":
        """
        Create a KnowledgeGraph that shares the database with a MemoryStore.
        
        This allows the knowledge graph to be stored alongside memories
        in the same SQLite database.
        
        Args:
            memory_store: The MemoryStore to integrate with
            entity_extractor: Custom entity extractor
            relation_extractor: Custom relation extractor
            
        Returns:
            KnowledgeGraph instance sharing the MemoryStore's database
        """
        return cls(
            connection=memory_store._conn,
            entity_extractor=entity_extractor,
            relation_extractor=relation_extractor,
        )
    
    def close(self) -> None:
        """Close the graph store."""
        self.graph_store.close()
    
    def __enter__(self) -> "KnowledgeGraph":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    # ==================== Text Processing ====================
    
    def extract_entities(self, text: str) -> list[EntityMention]:
        """
        Extract entity mentions from text.
        
        Args:
            text: Input text
            
        Returns:
            List of EntityMention objects
        """
        return self.entity_extractor.extract(text)
    
    def extract_relations(self, text: str) -> list[RelationMention]:
        """
        Extract relation mentions from text.
        
        Args:
            text: Input text
            
        Returns:
            List of RelationMention objects
        """
        return self.relation_extractor.extract(text)
    
    def extract_all(
        self, 
        text: str
    ) -> tuple[list[EntityMention], list[RelationMention]]:
        """
        Extract both entities and relations from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (entity_mentions, relation_mentions)
        """
        return self.combined_extractor.extract_all(text)
    
    def process_text(
        self,
        text: str,
        source_memory_id: str | None = None,
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Extract entities and relations from text and store them in the graph.
        
        Args:
            text: Input text to process
            source_memory_id: Optional memory ID to link extracted data to
            
        Returns:
            Tuple of (entities, relations) that were added to the graph
        """
        entity_mentions, relation_mentions = self.extract_all(text)
        
        # Convert mentions to entities and store
        entities: list[Entity] = []
        entity_map: dict[str, Entity] = {}  # lowercase name -> entity
        
        for mention in entity_mentions:
            entity = Entity.create(
                name=mention.text,
                entity_type=mention.entity_type,
            )
            stored_entity = self.graph_store.add_entity(entity)
            entities.append(stored_entity)
            entity_map[mention.text.lower()] = stored_entity
            
            # Link to source memory if provided
            if source_memory_id:
                self.graph_store.link_entity_to_memory(
                    stored_entity.id, 
                    source_memory_id
                )
        
        # Convert relation mentions to relations and store
        relations: list[Relation] = []
        
        for rel_mention in relation_mentions:
            subject_key = rel_mention.subject_text.lower()
            object_key = rel_mention.object_text.lower()
            
            # Get or create subject entity
            if subject_key in entity_map:
                subject = entity_map[subject_key]
            else:
                subject = self.graph_store.find_entity_by_name(rel_mention.subject_text)
                if not subject:
                    subject = Entity.create(
                        name=rel_mention.subject_text,
                        entity_type=EntityType.UNKNOWN,
                    )
                    subject = self.graph_store.add_entity(subject)
                entity_map[subject_key] = subject
            
            # Get or create object entity
            if object_key in entity_map:
                obj = entity_map[object_key]
            else:
                obj = self.graph_store.find_entity_by_name(rel_mention.object_text)
                if not obj:
                    obj = Entity.create(
                        name=rel_mention.object_text,
                        entity_type=EntityType.UNKNOWN,
                    )
                    obj = self.graph_store.add_entity(obj)
                entity_map[object_key] = obj
            
            # Create and store relation
            relation = Relation.create(
                subject_id=subject.id,
                predicate=rel_mention.predicate,
                object_id=obj.id,
                confidence=rel_mention.confidence,
                source_memory_id=source_memory_id,
            )
            stored_relation = self.graph_store.add_relation(relation)
            relations.append(stored_relation)
        
        logger.info(
            f"Processed text: extracted {len(entities)} entities, "
            f"{len(relations)} relations"
        )
        
        return entities, relations
    
    # ==================== Entity Operations ====================
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Entity:
        """
        Manually add an entity to the graph.
        
        Args:
            name: Entity name
            entity_type: Type of entity
            aliases: Alternative names
            metadata: Additional metadata
            
        Returns:
            The created Entity
        """
        entity = Entity.create(
            name=name,
            entity_type=entity_type,
            aliases=aliases,
            metadata=metadata,
        )
        return self.graph_store.add_entity(entity)
    
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self.graph_store.get_entity(entity_id)
    
    def find_entity(
        self, 
        name: str, 
        entity_type: EntityType | None = None
    ) -> Entity | None:
        """Find an entity by name."""
        return self.graph_store.find_entity_by_name(name, entity_type)
    
    def search_entities(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """
        Search for entities by name.
        
        Args:
            query: Search query
            entity_type: Optional type filter
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        return self.graph_store.search_entities(query, entity_type, limit)
    
    def list_entities(
        self,
        entity_type: EntityType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities with optional filtering."""
        return self.graph_store.list_entities(entity_type, limit, offset)
    
    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity and its relations."""
        self.graph_store.delete_entity(entity_id)
    
    # ==================== Relation Operations ====================
    
    def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float = 1.0,
        source_memory_id: str | None = None,
    ) -> Relation:
        """
        Manually add a relation to the graph.
        
        Args:
            subject_id: Subject entity ID
            predicate: Relation type
            object_id: Object entity ID
            confidence: Confidence score
            source_memory_id: Optional source memory ID
            
        Returns:
            The created Relation
        """
        relation = Relation.create(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            confidence=confidence,
            source_memory_id=source_memory_id,
        )
        return self.graph_store.add_relation(relation)
    
    def get_relations(
        self,
        entity_id: str,
        predicate: str | None = None,
        direction: str = "both",
    ) -> list[Relation]:
        """
        Get relations for an entity.
        
        Args:
            entity_id: Entity ID
            predicate: Optional predicate filter
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of relations
        """
        return self.graph_store.get_relations_for_entity(entity_id, predicate, direction)
    
    def delete_relation(self, relation_id: str) -> None:
        """Delete a relation."""
        self.graph_store.delete_relation(relation_id)
    
    # ==================== Graph Search ====================
    
    def get_related(
        self,
        entity_name_or_id: str,
        max_depth: int = 1,
        predicate_filter: list[str] | None = None,
    ) -> list[GraphSearchResult]:
        """
        Find entities related to a given entity.
        
        Args:
            entity_name_or_id: Entity name or ID
            max_depth: Maximum hops to traverse
            predicate_filter: Optional list of predicates to follow
            
        Returns:
            List of GraphSearchResult objects
        """
        # Try to find entity by ID first, then by name
        entity = self.graph_store.get_entity(entity_name_or_id)
        if not entity:
            entity = self.graph_store.find_entity_by_name(entity_name_or_id)
        
        if not entity:
            return []
        
        return self.graph_store.get_neighbors(
            entity.id, 
            max_depth, 
            predicate_filter
        )
    
    def find_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
    ) -> list[tuple[Entity, Relation | None]] | None:
        """
        Find a path between two entities.
        
        Args:
            start: Start entity name or ID
            end: End entity name or ID
            max_depth: Maximum path length
            
        Returns:
            List of (entity, relation) tuples, or None if no path exists
        """
        # Resolve entity names to IDs
        start_entity = self.graph_store.get_entity(start)
        if not start_entity:
            start_entity = self.graph_store.find_entity_by_name(start)
        
        end_entity = self.graph_store.get_entity(end)
        if not end_entity:
            end_entity = self.graph_store.find_entity_by_name(end)
        
        if not start_entity or not end_entity:
            return None
        
        return self.graph_store.find_path(start_entity.id, end_entity.id, max_depth)
    
    # ==================== Memory Integration ====================
    
    def index_memory(
        self,
        memory: Any,  # Memory from MemoryStore
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Extract and index entities and relations from a memory.
        
        Args:
            memory: Memory object from MemoryStore
            
        Returns:
            Tuple of (entities, relations) extracted and stored
        """
        return self.process_text(memory.content, source_memory_id=memory.id)
    
    def index_memories(
        self,
        memories: list[Any],
    ) -> tuple[int, int]:
        """
        Batch index multiple memories.
        
        Args:
            memories: List of Memory objects
            
        Returns:
            Tuple of (total_entities, total_relations) extracted
        """
        total_entities = 0
        total_relations = 0
        
        for memory in memories:
            entities, relations = self.index_memory(memory)
            total_entities += len(entities)
            total_relations += len(relations)
        
        return total_entities, total_relations
    
    def get_entities_for_memory(self, memory_id: str) -> list[Entity]:
        """Get all entities linked to a memory."""
        return self.graph_store.get_entities_for_memory(memory_id)
    
    def get_memories_for_entity(self, entity_id: str) -> list[str]:
        """Get all memory IDs linked to an entity."""
        return self.graph_store.get_memories_for_entity(entity_id)
    
    def graph_augmented_search(
        self,
        query: str,
        memory_store: "MemoryStore",
        entity_boost: float = 0.3,
        max_graph_depth: int = 1,
        limit: int = 10,
    ) -> list[tuple[Any, float]]:
        """
        Search memories with graph augmentation.
        
        Finds relevant memories by:
        1. Extracting entities from the query
        2. Finding related entities in the graph
        3. Boosting memories that contain these entities
        
        Args:
            query: Search query
            memory_store: MemoryStore to search
            entity_boost: Score boost for entity matches (0-1)
            max_graph_depth: How far to traverse in the graph
            limit: Maximum results
            
        Returns:
            List of (memory, augmented_score) tuples
        """
        # Extract entities from query
        query_entities = self.entity_extractor.extract(query)
        
        # Build set of relevant entity IDs (query entities + neighbors)
        relevant_entity_ids: set[str] = set()
        
        for mention in query_entities:
            entity = self.graph_store.find_entity_by_name(mention.text)
            if entity:
                relevant_entity_ids.add(entity.id)
                
                # Add neighbors
                neighbors = self.graph_store.get_neighbors(
                    entity.id, 
                    max_depth=max_graph_depth
                )
                for result in neighbors:
                    relevant_entity_ids.add(result.entity.id)
        
        # Get memory IDs that contain these entities
        memory_entity_scores: dict[str, float] = {}
        for entity_id in relevant_entity_ids:
            memory_ids = self.graph_store.get_memories_for_entity(entity_id)
            for mid in memory_ids:
                memory_entity_scores[mid] = memory_entity_scores.get(mid, 0) + 1
        
        # Normalize entity scores
        if memory_entity_scores:
            max_score = max(memory_entity_scores.values())
            memory_entity_scores = {
                k: v / max_score * entity_boost 
                for k, v in memory_entity_scores.items()
            }
        
        # Search memories using the memory store
        # This returns SearchResult objects with .memory and .score
        search_results = memory_store.search(query, limit=limit * 2)
        
        # Combine scores
        augmented_results: list[tuple[Any, float]] = []
        for result in search_results:
            base_score = result.score
            entity_score = memory_entity_scores.get(result.memory.id, 0)
            augmented_score = base_score + entity_score
            augmented_results.append((result.memory, augmented_score))
        
        # Sort by augmented score and limit
        augmented_results.sort(key=lambda x: x[1], reverse=True)
        return augmented_results[:limit]
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return self.graph_store.get_stats()


# Export all public classes and types
__all__ = [
    # Main class
    "KnowledgeGraph",
    # Models
    "Entity",
    "EntityType",
    "Relation",
    "EntityMention",
    "RelationMention",
    "GraphSearchResult",
    # Extractors
    "EntityExtractor",
    "RelationExtractor",
    "CombinedExtractor",
    # Storage
    "GraphStore",
]
