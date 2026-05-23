"""SQLite-based graph storage for entities and relations."""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager

from .models import Entity, EntityType, Relation, GraphSearchResult

logger = logging.getLogger(__name__)


# Schema for the knowledge graph tables
GRAPH_SCHEMA = """
-- Entities table
CREATE TABLE IF NOT EXISTS kg_entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases_json TEXT DEFAULT '[]',
    metadata_json TEXT DEFAULT '{}',
    mention_count INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Index for entity lookups
CREATE INDEX IF NOT EXISTS idx_kg_entities_name_normalized ON kg_entities(name_normalized);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type);

-- Full-text search for entities
CREATE VIRTUAL TABLE IF NOT EXISTS kg_entities_fts USING fts5(
    name,
    aliases,
    content=kg_entities,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS kg_entities_ai AFTER INSERT ON kg_entities BEGIN
    INSERT INTO kg_entities_fts(rowid, name, aliases) 
    VALUES (NEW.rowid, NEW.name, NEW.aliases_json);
END;

CREATE TRIGGER IF NOT EXISTS kg_entities_ad AFTER DELETE ON kg_entities BEGIN
    INSERT INTO kg_entities_fts(kg_entities_fts, rowid, name, aliases) 
    VALUES ('delete', OLD.rowid, OLD.name, OLD.aliases_json);
END;

CREATE TRIGGER IF NOT EXISTS kg_entities_au AFTER UPDATE ON kg_entities BEGIN
    INSERT INTO kg_entities_fts(kg_entities_fts, rowid, name, aliases) 
    VALUES ('delete', OLD.rowid, OLD.name, OLD.aliases_json);
    INSERT INTO kg_entities_fts(rowid, name, aliases) 
    VALUES (NEW.rowid, NEW.name, NEW.aliases_json);
END;

-- Relations table (triples)
CREATE TABLE IF NOT EXISTS kg_relations (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    source_memory_id TEXT,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES kg_entities(id) ON DELETE CASCADE,
    FOREIGN KEY (object_id) REFERENCES kg_entities(id) ON DELETE CASCADE
);

-- Indexes for relation traversal
CREATE INDEX IF NOT EXISTS idx_kg_relations_subject ON kg_relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_object ON kg_relations(object_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_predicate ON kg_relations(predicate);
CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_memory_id);

-- Entity-Memory linking table
CREATE TABLE IF NOT EXISTS kg_entity_memories (
    entity_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (entity_id, memory_id),
    FOREIGN KEY (entity_id) REFERENCES kg_entities(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kg_entity_memories_memory ON kg_entity_memories(memory_id);
"""


class GraphStore:
    """
    SQLite-based storage for the knowledge graph.
    
    Provides CRUD operations for entities and relations,
    as well as graph traversal and search capabilities.
    """
    
    def __init__(
        self,
        db_path: str | Path = ":memory:",
        connection: sqlite3.Connection | None = None,
    ):
        """
        Initialize the graph store.
        
        Args:
            db_path: Path to SQLite database, or ":memory:" for in-memory
            connection: Optional existing SQLite connection to use
        """
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = connection
        self._owns_connection = connection is None
        
        if self._owns_connection:
            self._init_db()
        else:
            self._apply_schema()
    
    def _init_db(self) -> None:
        """Initialize the database connection and schema."""
        self._conn = sqlite3.connect(
            self.db_path if isinstance(self.db_path, str) else str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._apply_schema()
    
    def _apply_schema(self) -> None:
        """Apply the graph schema to the database."""
        self._conn.executescript(GRAPH_SCHEMA)
        self._conn.commit()
    
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for database transactions."""
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
    
    def close(self) -> None:
        """Close the database connection if we own it."""
        if self._owns_connection and self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self) -> "GraphStore":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    # ==================== Entity Operations ====================
    
    def add_entity(self, entity: Entity) -> Entity:
        """
        Add an entity to the graph.
        
        If an entity with the same normalized name and type exists,
        updates the existing entity instead.
        
        Args:
            entity: Entity to add
            
        Returns:
            The added or existing entity
        """
        normalized = self._normalize_name(entity.name)
        
        # Check for existing entity
        existing = self.find_entity_by_name(entity.name, entity.entity_type)
        if existing:
            # Update mention count and merge aliases
            existing.mention_count += 1
            existing.aliases = list(set(existing.aliases + entity.aliases))
            existing.updated_at = datetime.utcnow()
            return self.update_entity(existing)
        
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO kg_entities 
                (id, name, name_normalized, entity_type, aliases_json, metadata_json, 
                 mention_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity.id,
                    entity.name,
                    normalized,
                    entity.entity_type.value,
                    json.dumps(entity.aliases),
                    json.dumps(entity.metadata),
                    entity.mention_count,
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                ),
            )
        
        logger.debug(f"Added entity: {entity.name} ({entity.entity_type.value})")
        return entity
    
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        cursor = self._conn.execute(
            """
            SELECT id, name, entity_type, aliases_json, metadata_json, 
                   mention_count, created_at, updated_at
            FROM kg_entities
            WHERE id = ?
            """,
            (entity_id,),
        )
        row = cursor.fetchone()
        return self._row_to_entity(row) if row else None
    
    def find_entity_by_name(
        self, 
        name: str, 
        entity_type: EntityType | None = None
    ) -> Entity | None:
        """
        Find an entity by name (case-insensitive).
        
        Args:
            name: Entity name to search for
            entity_type: Optional type filter
            
        Returns:
            Entity if found, None otherwise
        """
        normalized = self._normalize_name(name)
        
        if entity_type:
            cursor = self._conn.execute(
                """
                SELECT id, name, entity_type, aliases_json, metadata_json,
                       mention_count, created_at, updated_at
                FROM kg_entities
                WHERE name_normalized = ? AND entity_type = ?
                """,
                (normalized, entity_type.value),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT id, name, entity_type, aliases_json, metadata_json,
                       mention_count, created_at, updated_at
                FROM kg_entities
                WHERE name_normalized = ?
                """,
                (normalized,),
            )
        
        row = cursor.fetchone()
        return self._row_to_entity(row) if row else None
    
    def search_entities(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """
        Search entities using FTS.
        
        Args:
            query: Search query
            entity_type: Optional type filter
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        # Build FTS query
        fts_query = f'"{query}"*'  # Prefix match
        
        if entity_type:
            cursor = self._conn.execute(
                """
                SELECT e.id, e.name, e.entity_type, e.aliases_json, e.metadata_json,
                       e.mention_count, e.created_at, e.updated_at,
                       bm25(kg_entities_fts) as rank
                FROM kg_entities_fts fts
                JOIN kg_entities e ON e.rowid = fts.rowid
                WHERE kg_entities_fts MATCH ? AND e.entity_type = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, entity_type.value, limit),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT e.id, e.name, e.entity_type, e.aliases_json, e.metadata_json,
                       e.mention_count, e.created_at, e.updated_at,
                       bm25(kg_entities_fts) as rank
                FROM kg_entities_fts fts
                JOIN kg_entities e ON e.rowid = fts.rowid
                WHERE kg_entities_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )
        
        return [self._row_to_entity(row) for row in cursor.fetchall()]
    
    def update_entity(self, entity: Entity) -> Entity:
        """Update an existing entity."""
        with self.transaction():
            self._conn.execute(
                """
                UPDATE kg_entities
                SET name = ?, name_normalized = ?, entity_type = ?, 
                    aliases_json = ?, metadata_json = ?, mention_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    entity.name,
                    self._normalize_name(entity.name),
                    entity.entity_type.value,
                    json.dumps(entity.aliases),
                    json.dumps(entity.metadata),
                    entity.mention_count,
                    entity.updated_at.isoformat(),
                    entity.id,
                ),
            )
        return entity
    
    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity and its relations."""
        with self.transaction():
            # Relations are deleted by CASCADE
            self._conn.execute("DELETE FROM kg_entities WHERE id = ?", (entity_id,))
    
    def list_entities(
        self,
        entity_type: EntityType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities with optional filtering."""
        if entity_type:
            cursor = self._conn.execute(
                """
                SELECT id, name, entity_type, aliases_json, metadata_json,
                       mention_count, created_at, updated_at
                FROM kg_entities
                WHERE entity_type = ?
                ORDER BY mention_count DESC, name
                LIMIT ? OFFSET ?
                """,
                (entity_type.value, limit, offset),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT id, name, entity_type, aliases_json, metadata_json,
                       mention_count, created_at, updated_at
                FROM kg_entities
                ORDER BY mention_count DESC, name
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        
        return [self._row_to_entity(row) for row in cursor.fetchall()]
    
    # ==================== Relation Operations ====================
    
    def add_relation(self, relation: Relation) -> Relation:
        """
        Add a relation to the graph.
        
        Args:
            relation: Relation to add
            
        Returns:
            The added relation
        """
        # Check if relation already exists
        existing = self.find_relation(
            relation.subject_id, 
            relation.predicate, 
            relation.object_id
        )
        if existing:
            return existing
        
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO kg_relations
                (id, subject_id, predicate, object_id, confidence, 
                 source_memory_id, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation.id,
                    relation.subject_id,
                    relation.predicate,
                    relation.object_id,
                    relation.confidence,
                    relation.source_memory_id,
                    json.dumps(relation.metadata),
                    relation.created_at.isoformat(),
                ),
            )
        
        logger.debug(f"Added relation: {relation.subject_id} -{relation.predicate}-> {relation.object_id}")
        return relation
    
    def get_relation(self, relation_id: str) -> Relation | None:
        """Get a relation by ID."""
        cursor = self._conn.execute(
            """
            SELECT id, subject_id, predicate, object_id, confidence,
                   source_memory_id, metadata_json, created_at
            FROM kg_relations
            WHERE id = ?
            """,
            (relation_id,),
        )
        row = cursor.fetchone()
        return self._row_to_relation(row) if row else None
    
    def find_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> Relation | None:
        """Find a specific relation by its components."""
        cursor = self._conn.execute(
            """
            SELECT id, subject_id, predicate, object_id, confidence,
                   source_memory_id, metadata_json, created_at
            FROM kg_relations
            WHERE subject_id = ? AND predicate = ? AND object_id = ?
            """,
            (subject_id, predicate, object_id),
        )
        row = cursor.fetchone()
        return self._row_to_relation(row) if row else None
    
    def get_relations_for_entity(
        self,
        entity_id: str,
        predicate: str | None = None,
        direction: str = "both",  # "outgoing", "incoming", or "both"
    ) -> list[Relation]:
        """
        Get all relations for an entity.
        
        Args:
            entity_id: Entity ID
            predicate: Optional predicate filter
            direction: "outgoing" (entity as subject), "incoming" (entity as object), or "both"
            
        Returns:
            List of relations
        """
        relations = []
        
        if direction in ("outgoing", "both"):
            if predicate:
                cursor = self._conn.execute(
                    """
                    SELECT id, subject_id, predicate, object_id, confidence,
                           source_memory_id, metadata_json, created_at
                    FROM kg_relations
                    WHERE subject_id = ? AND predicate = ?
                    """,
                    (entity_id, predicate),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT id, subject_id, predicate, object_id, confidence,
                           source_memory_id, metadata_json, created_at
                    FROM kg_relations
                    WHERE subject_id = ?
                    """,
                    (entity_id,),
                )
            relations.extend([self._row_to_relation(row) for row in cursor.fetchall()])
        
        if direction in ("incoming", "both"):
            if predicate:
                cursor = self._conn.execute(
                    """
                    SELECT id, subject_id, predicate, object_id, confidence,
                           source_memory_id, metadata_json, created_at
                    FROM kg_relations
                    WHERE object_id = ? AND predicate = ?
                    """,
                    (entity_id, predicate),
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT id, subject_id, predicate, object_id, confidence,
                           source_memory_id, metadata_json, created_at
                    FROM kg_relations
                    WHERE object_id = ?
                    """,
                    (entity_id,),
                )
            relations.extend([self._row_to_relation(row) for row in cursor.fetchall()])
        
        return relations
    
    def delete_relation(self, relation_id: str) -> None:
        """Delete a relation."""
        with self.transaction():
            self._conn.execute("DELETE FROM kg_relations WHERE id = ?", (relation_id,))
    
    def delete_relations_for_memory(self, memory_id: str) -> int:
        """Delete all relations extracted from a specific memory."""
        with self.transaction():
            cursor = self._conn.execute(
                "DELETE FROM kg_relations WHERE source_memory_id = ?",
                (memory_id,),
            )
            return cursor.rowcount
    
    # ==================== Entity-Memory Linking ====================
    
    def link_entity_to_memory(self, entity_id: str, memory_id: str) -> None:
        """Link an entity to a memory."""
        with self.transaction():
            self._conn.execute(
                """
                INSERT OR IGNORE INTO kg_entity_memories (entity_id, memory_id, created_at)
                VALUES (?, ?, ?)
                """,
                (entity_id, memory_id, datetime.utcnow().isoformat()),
            )
    
    def get_entities_for_memory(self, memory_id: str) -> list[Entity]:
        """Get all entities linked to a memory."""
        cursor = self._conn.execute(
            """
            SELECT e.id, e.name, e.entity_type, e.aliases_json, e.metadata_json,
                   e.mention_count, e.created_at, e.updated_at
            FROM kg_entities e
            JOIN kg_entity_memories em ON e.id = em.entity_id
            WHERE em.memory_id = ?
            """,
            (memory_id,),
        )
        return [self._row_to_entity(row) for row in cursor.fetchall()]
    
    def get_memories_for_entity(self, entity_id: str) -> list[str]:
        """Get all memory IDs linked to an entity."""
        cursor = self._conn.execute(
            "SELECT memory_id FROM kg_entity_memories WHERE entity_id = ?",
            (entity_id,),
        )
        return [row["memory_id"] for row in cursor.fetchall()]
    
    # ==================== Graph Traversal ====================
    
    def get_neighbors(
        self,
        entity_id: str,
        max_depth: int = 1,
        predicate_filter: list[str] | None = None,
    ) -> list[GraphSearchResult]:
        """
        Get neighboring entities up to a certain depth.
        
        Args:
            entity_id: Starting entity ID
            max_depth: Maximum hops from starting entity
            predicate_filter: Optional list of predicates to follow
            
        Returns:
            List of GraphSearchResult objects
        """
        results: list[GraphSearchResult] = []
        visited: set[str] = {entity_id}
        current_level: set[str] = {entity_id}
        
        for depth in range(1, max_depth + 1):
            next_level: set[str] = set()
            
            for current_id in current_level:
                relations = self.get_relations_for_entity(current_id)
                
                for relation in relations:
                    if predicate_filter and relation.predicate not in predicate_filter:
                        continue
                    
                    # Get the connected entity
                    connected_id = (
                        relation.object_id 
                        if relation.subject_id == current_id 
                        else relation.subject_id
                    )
                    
                    if connected_id in visited:
                        continue
                    
                    connected_entity = self.get_entity(connected_id)
                    if not connected_entity:
                        continue
                    
                    visited.add(connected_id)
                    next_level.add(connected_id)
                    
                    # Get all relations for this connected entity
                    connected_relations = self.get_relations_for_entity(connected_id)
                    relation_tuples = []
                    for rel in connected_relations:
                        other_id = (
                            rel.object_id 
                            if rel.subject_id == connected_id 
                            else rel.subject_id
                        )
                        other_entity = self.get_entity(other_id)
                        if other_entity:
                            relation_tuples.append((rel, other_entity))
                    
                    results.append(GraphSearchResult(
                        entity=connected_entity,
                        relations=relation_tuples,
                        distance=depth,
                        score=1.0 / depth,  # Closer entities get higher scores
                    ))
            
            current_level = next_level
            if not current_level:
                break
        
        return results
    
    def find_path(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_depth: int = 5,
    ) -> list[tuple[Entity, Relation]] | None:
        """
        Find a path between two entities.
        
        Args:
            start_entity_id: Starting entity ID
            end_entity_id: Target entity ID
            max_depth: Maximum path length
            
        Returns:
            List of (entity, relation) tuples representing the path, or None if no path exists
        """
        if start_entity_id == end_entity_id:
            entity = self.get_entity(start_entity_id)
            return [(entity, None)] if entity else None
        
        # BFS to find shortest path
        queue = [(start_entity_id, [(start_entity_id, None)])]
        visited = {start_entity_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            relations = self.get_relations_for_entity(current_id)
            
            for relation in relations:
                next_id = (
                    relation.object_id 
                    if relation.subject_id == current_id 
                    else relation.subject_id
                )
                
                if next_id in visited:
                    continue
                
                new_path = path + [(next_id, relation)]
                
                if next_id == end_entity_id:
                    # Convert IDs to entities
                    result = []
                    for entity_id, rel in new_path:
                        entity = self.get_entity(entity_id)
                        if entity:
                            result.append((entity, rel))
                    return result
                
                visited.add(next_id)
                queue.append((next_id, new_path))
        
        return None
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the graph."""
        entity_count = self._conn.execute(
            "SELECT COUNT(*) FROM kg_entities"
        ).fetchone()[0]
        
        relation_count = self._conn.execute(
            "SELECT COUNT(*) FROM kg_relations"
        ).fetchone()[0]
        
        type_counts = {}
        cursor = self._conn.execute(
            "SELECT entity_type, COUNT(*) FROM kg_entities GROUP BY entity_type"
        )
        for row in cursor.fetchall():
            type_counts[row[0]] = row[1]
        
        predicate_counts = {}
        cursor = self._conn.execute(
            "SELECT predicate, COUNT(*) FROM kg_relations GROUP BY predicate ORDER BY COUNT(*) DESC LIMIT 10"
        )
        for row in cursor.fetchall():
            predicate_counts[row[0]] = row[1]
        
        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "entity_type_counts": type_counts,
            "top_predicates": predicate_counts,
        }
    
    # ==================== Helper Methods ====================
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        return name.lower().strip()
    
    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert a database row to an Entity object."""
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            aliases=json.loads(row["aliases_json"]),
            metadata=json.loads(row["metadata_json"]),
            mention_count=row["mention_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
    
    def _row_to_relation(self, row: sqlite3.Row) -> Relation:
        """Convert a database row to a Relation object."""
        return Relation(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            confidence=row["confidence"],
            source_memory_id=row["source_memory_id"],
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
