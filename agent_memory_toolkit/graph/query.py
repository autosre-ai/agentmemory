"""Graph query language for the knowledge graph.

This module provides a fluent query interface for traversing and 
querying the knowledge graph. It supports pattern matching, filtering,
aggregation, and complex traversal operations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Iterator, TypeVar

from .knowledge import Entity, EntityType, KnowledgeGraphStore
from .relationships import (
    Relationship,
    RelationshipType,
    RelationshipCategory,
    get_relationship_category,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AggregateFunction(Enum):
    """Aggregate functions for query results."""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COLLECT = "collect"


class SortOrder(Enum):
    """Sort order for query results."""
    ASC = "asc"
    DESC = "desc"


@dataclass
class QueryMatch:
    """A single match from a graph query."""
    entity: Entity
    path: list[tuple[Entity, Relationship | None]] = field(default_factory=list)
    depth: int = 0
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.entity.entity_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueryMatch):
            return False
        return self.entity.entity_id == other.entity.entity_id


@dataclass
class QueryResult:
    """Result of a graph query."""
    matches: list[QueryMatch] = field(default_factory=list)
    total_count: int = 0
    execution_time_ms: float = 0.0
    truncated: bool = False
    aggregations: dict[str, Any] = field(default_factory=dict)
    
    def entities(self) -> list[Entity]:
        """Get all matched entities."""
        return [m.entity for m in self.matches]
    
    def paths(self) -> list[list[tuple[Entity, Relationship | None]]]:
        """Get all matched paths."""
        return [m.path for m in self.matches if m.path]
    
    def top(self, n: int) -> list[QueryMatch]:
        """Get top N matches by score."""
        return sorted(self.matches, key=lambda m: m.score, reverse=True)[:n]
    
    def __len__(self) -> int:
        return len(self.matches)
    
    def __iter__(self) -> Iterator[QueryMatch]:
        return iter(self.matches)
    
    def __bool__(self) -> bool:
        return len(self.matches) > 0


class EntityFilter:
    """Filter for matching entities in queries."""
    
    def __init__(self):
        self._filters: list[Callable[[Entity], bool]] = []
    
    def by_type(self, *entity_types: EntityType) -> "EntityFilter":
        """Filter by entity types."""
        types_set = set(entity_types)
        self._filters.append(lambda e: e.entity_type in types_set)
        return self
    
    def by_name(self, pattern: str, regex: bool = False) -> "EntityFilter":
        """Filter by name pattern."""
        if regex:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._filters.append(lambda e: compiled.search(e.name) is not None)
        else:
            pattern_lower = pattern.lower()
            self._filters.append(lambda e: pattern_lower in e.name.lower())
        return self
    
    def by_importance(self, min_importance: float = 0.0, max_importance: float = 1.0) -> "EntityFilter":
        """Filter by importance range."""
        self._filters.append(lambda e: min_importance <= e.importance <= max_importance)
        return self
    
    def by_property(self, key: str, value: Any = None, exists: bool = True) -> "EntityFilter":
        """Filter by property existence or value."""
        if value is not None:
            self._filters.append(lambda e: e.properties.get(key) == value)
        elif exists:
            self._filters.append(lambda e: key in e.properties)
        else:
            self._filters.append(lambda e: key not in e.properties)
        return self
    
    def by_created_after(self, after: datetime) -> "EntityFilter":
        """Filter by creation time."""
        self._filters.append(lambda e: e.created_at > after)
        return self
    
    def by_created_before(self, before: datetime) -> "EntityFilter":
        """Filter by creation time."""
        self._filters.append(lambda e: e.created_at < before)
        return self
    
    def by_access_count(self, min_count: int = 0, max_count: int | None = None) -> "EntityFilter":
        """Filter by access count."""
        if max_count is not None:
            self._filters.append(lambda e: min_count <= e.access_count <= max_count)
        else:
            self._filters.append(lambda e: e.access_count >= min_count)
        return self
    
    def custom(self, predicate: Callable[[Entity], bool]) -> "EntityFilter":
        """Add a custom filter predicate."""
        self._filters.append(predicate)
        return self
    
    def matches(self, entity: Entity) -> bool:
        """Check if an entity matches all filters."""
        return all(f(entity) for f in self._filters)


class RelationshipFilter:
    """Filter for matching relationships in queries."""
    
    def __init__(self):
        self._filters: list[Callable[[Relationship], bool]] = []
    
    def by_type(self, *rel_types: RelationshipType) -> "RelationshipFilter":
        """Filter by relationship types."""
        types_set = set(rel_types)
        self._filters.append(lambda r: r.relationship_type in types_set)
        return self
    
    def by_category(self, *categories: RelationshipCategory) -> "RelationshipFilter":
        """Filter by relationship categories."""
        cats_set = set(categories)
        self._filters.append(lambda r: get_relationship_category(r.relationship_type) in cats_set)
        return self
    
    def by_weight(self, min_weight: float = 0.0, max_weight: float = 1.0) -> "RelationshipFilter":
        """Filter by weight range."""
        self._filters.append(lambda r: min_weight <= r.properties.weight <= max_weight)
        return self
    
    def by_confidence(self, min_confidence: float = 0.0) -> "RelationshipFilter":
        """Filter by minimum confidence."""
        self._filters.append(lambda r: r.properties.confidence >= min_confidence)
        return self
    
    def bidirectional_only(self) -> "RelationshipFilter":
        """Filter to only bidirectional relationships."""
        self._filters.append(lambda r: r.properties.bidirectional)
        return self
    
    def custom(self, predicate: Callable[[Relationship], bool]) -> "RelationshipFilter":
        """Add a custom filter predicate."""
        self._filters.append(predicate)
        return self
    
    def matches(self, relationship: Relationship) -> bool:
        """Check if a relationship matches all filters."""
        return all(f(relationship) for f in self._filters)


class GraphQuery:
    """Fluent query builder for the knowledge graph.
    
    Example:
        query = GraphQuery(graph)
        results = (query
            .start_from_type(EntityType.CONCEPT)
            .filter_entities(EntityFilter().by_importance(0.5))
            .traverse(RelationshipType.IS_A, max_depth=2)
            .filter_relationships(RelationshipFilter().by_weight(0.7))
            .sort_by("importance", SortOrder.DESC)
            .limit(10)
            .execute())
    """
    
    def __init__(self, graph: KnowledgeGraphStore):
        """Initialize a query on a knowledge graph.
        
        Args:
            graph: The knowledge graph to query.
        """
        self._graph = graph
        self._start_entities: list[str] | None = None
        self._entity_filter: EntityFilter | None = None
        self._relationship_filter: RelationshipFilter | None = None
        self._traversal_types: list[RelationshipType] | None = None
        self._max_depth: int = 1
        self._direction: str = "outgoing"
        self._sort_key: str | None = None
        self._sort_order: SortOrder = SortOrder.ASC
        self._limit: int | None = None
        self._offset: int = 0
        self._aggregations: list[tuple[str, AggregateFunction]] = []
        self._distinct: bool = False
        self._include_paths: bool = False
    
    def start_from(self, *entity_ids: str) -> "GraphQuery":
        """Start the query from specific entities.
        
        Args:
            entity_ids: Entity IDs to start from.
            
        Returns:
            Self for method chaining.
        """
        self._start_entities = list(entity_ids)
        return self
    
    def start_from_type(self, *entity_types: EntityType) -> "GraphQuery":
        """Start the query from all entities of given types.
        
        Args:
            entity_types: Entity types to match.
            
        Returns:
            Self for method chaining.
        """
        entity_ids = []
        for et in entity_types:
            entities = self._graph.find_entities_by_type(et)
            entity_ids.extend(e.entity_id for e in entities)
        self._start_entities = entity_ids
        return self
    
    def start_from_name(self, name: str, fuzzy: bool = False) -> "GraphQuery":
        """Start the query from entities matching a name.
        
        Args:
            name: Name to search for.
            fuzzy: Whether to use fuzzy matching.
            
        Returns:
            Self for method chaining.
        """
        entities = self._graph.find_entities_by_name(name, fuzzy=fuzzy)
        self._start_entities = [e.entity_id for e in entities]
        return self
    
    def filter_entities(self, entity_filter: EntityFilter) -> "GraphQuery":
        """Apply an entity filter.
        
        Args:
            entity_filter: The filter to apply.
            
        Returns:
            Self for method chaining.
        """
        self._entity_filter = entity_filter
        return self
    
    def filter_relationships(self, rel_filter: RelationshipFilter) -> "GraphQuery":
        """Apply a relationship filter.
        
        Args:
            rel_filter: The filter to apply.
            
        Returns:
            Self for method chaining.
        """
        self._relationship_filter = rel_filter
        return self
    
    def traverse(
        self,
        *relationship_types: RelationshipType,
        max_depth: int = 1,
        direction: str = "outgoing",
    ) -> "GraphQuery":
        """Traverse relationships from the starting entities.
        
        Args:
            relationship_types: Types of relationships to traverse.
            max_depth: Maximum traversal depth.
            direction: "outgoing", "incoming", or "both".
            
        Returns:
            Self for method chaining.
        """
        self._traversal_types = list(relationship_types) if relationship_types else None
        self._max_depth = max_depth
        self._direction = direction
        return self
    
    def with_paths(self) -> "GraphQuery":
        """Include paths in the results.
        
        Returns:
            Self for method chaining.
        """
        self._include_paths = True
        return self
    
    def sort_by(self, key: str, order: SortOrder = SortOrder.ASC) -> "GraphQuery":
        """Sort results by a key.
        
        Args:
            key: The key to sort by (e.g., "importance", "access_count", "created_at").
            order: Sort order.
            
        Returns:
            Self for method chaining.
        """
        self._sort_key = key
        self._sort_order = order
        return self
    
    def limit(self, limit: int) -> "GraphQuery":
        """Limit the number of results.
        
        Args:
            limit: Maximum number of results.
            
        Returns:
            Self for method chaining.
        """
        self._limit = limit
        return self
    
    def offset(self, offset: int) -> "GraphQuery":
        """Skip a number of results.
        
        Args:
            offset: Number of results to skip.
            
        Returns:
            Self for method chaining.
        """
        self._offset = offset
        return self
    
    def distinct(self) -> "GraphQuery":
        """Return only distinct entities.
        
        Returns:
            Self for method chaining.
        """
        self._distinct = True
        return self
    
    def aggregate(self, field: str, function: AggregateFunction) -> "GraphQuery":
        """Add an aggregation to the query.
        
        Args:
            field: Field to aggregate.
            function: Aggregation function.
            
        Returns:
            Self for method chaining.
        """
        self._aggregations.append((field, function))
        return self
    
    def execute(self) -> QueryResult:
        """Execute the query and return results.
        
        Returns:
            QueryResult with matching entities.
        """
        start_time = datetime.now()
        matches: list[QueryMatch] = []
        seen_ids: set[str] = set()
        
        # Get starting entities
        if self._start_entities:
            start_ids = self._start_entities
        else:
            start_ids = list(self._graph._entities.keys())
        
        # Filter starting entities
        for entity_id in start_ids:
            entity = self._graph.get_entity(entity_id)
            if not entity:
                continue
            
            if self._entity_filter and not self._entity_filter.matches(entity):
                continue
            
            # No traversal - just return filtered starting entities
            if self._max_depth == 0 or (self._traversal_types is None and self._max_depth == 1):
                if self._distinct and entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)
                matches.append(QueryMatch(
                    entity=entity,
                    depth=0,
                    score=entity.importance,
                ))
            else:
                # Traverse from this entity
                traversed = self._traverse_from(entity_id)
                for match in traversed:
                    if self._distinct and match.entity.entity_id in seen_ids:
                        continue
                    seen_ids.add(match.entity.entity_id)
                    matches.append(match)
        
        # Sort results
        if self._sort_key:
            reverse = self._sort_order == SortOrder.DESC
            matches.sort(key=lambda m: self._get_sort_value(m), reverse=reverse)
        
        # Calculate aggregations
        aggregations = self._compute_aggregations(matches)
        
        total_count = len(matches)
        
        # Apply offset and limit
        if self._offset:
            matches = matches[self._offset:]
        if self._limit:
            truncated = len(matches) > self._limit
            matches = matches[:self._limit]
        else:
            truncated = False
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            matches=matches,
            total_count=total_count,
            execution_time_ms=execution_time,
            truncated=truncated,
            aggregations=aggregations,
        )
    
    def _traverse_from(self, entity_id: str) -> list[QueryMatch]:
        """Traverse the graph from an entity."""
        results = []
        visited = {entity_id}
        current_level = [(entity_id, [])]  # (id, path)
        
        for depth in range(1, self._max_depth + 1):
            next_level = []
            
            for current_id, path in current_level:
                relationships = self._graph.get_relationships(
                    current_id,
                    direction=self._direction,
                    relationship_types=self._traversal_types,
                )
                
                for rel in relationships:
                    # Apply relationship filter
                    if self._relationship_filter and not self._relationship_filter.matches(rel):
                        continue
                    
                    # Determine next entity based on direction
                    if self._direction == "incoming":
                        next_id = rel.source_id
                    else:
                        next_id = rel.target_id if rel.target_id != current_id else rel.source_id
                    
                    if next_id in visited:
                        continue
                    
                    next_entity = self._graph.get_entity(next_id)
                    if not next_entity:
                        continue
                    
                    # Apply entity filter to traversed entities
                    if self._entity_filter and not self._entity_filter.matches(next_entity):
                        continue
                    
                    visited.add(next_id)
                    
                    # Build path
                    current_entity = self._graph.get_entity(current_id)
                    new_path = path + [(current_entity, rel)] if self._include_paths and current_entity else []
                    if self._include_paths:
                        new_path.append((next_entity, None))
                    
                    # Calculate score based on depth and weight
                    score = next_entity.importance * rel.properties.weight * (1.0 / depth)
                    
                    results.append(QueryMatch(
                        entity=next_entity,
                        path=new_path,
                        depth=depth,
                        score=score,
                    ))
                    
                    next_level.append((next_id, new_path))
            
            current_level = next_level
            if not current_level:
                break
        
        return results
    
    def _get_sort_value(self, match: QueryMatch) -> Any:
        """Get the sort value for a match."""
        entity = match.entity
        if self._sort_key == "importance":
            return entity.importance
        elif self._sort_key == "access_count":
            return entity.access_count
        elif self._sort_key == "created_at":
            return entity.created_at
        elif self._sort_key == "updated_at":
            return entity.updated_at
        elif self._sort_key == "depth":
            return match.depth
        elif self._sort_key == "score":
            return match.score
        elif self._sort_key == "name":
            return entity.name.lower()
        else:
            return entity.properties.get(self._sort_key, 0)
    
    def _compute_aggregations(self, matches: list[QueryMatch]) -> dict[str, Any]:
        """Compute aggregations over matches."""
        aggregations = {}
        
        for field, func in self._aggregations:
            values = []
            for match in matches:
                if field == "importance":
                    values.append(match.entity.importance)
                elif field == "access_count":
                    values.append(match.entity.access_count)
                elif field == "score":
                    values.append(match.score)
                elif field == "depth":
                    values.append(match.depth)
                else:
                    val = match.entity.properties.get(field)
                    if val is not None:
                        values.append(val)
            
            key = f"{func.value}_{field}"
            if func == AggregateFunction.COUNT:
                aggregations[key] = len(values)
            elif func == AggregateFunction.SUM and values:
                aggregations[key] = sum(values)
            elif func == AggregateFunction.AVG and values:
                aggregations[key] = sum(values) / len(values)
            elif func == AggregateFunction.MIN and values:
                aggregations[key] = min(values)
            elif func == AggregateFunction.MAX and values:
                aggregations[key] = max(values)
            elif func == AggregateFunction.COLLECT:
                aggregations[key] = values
        
        return aggregations


class PatternMatcher:
    """Pattern-based matching for graph structures.
    
    Allows matching complex graph patterns like:
    - (A)-[IS_A]->(B)-[PART_OF]->(C)
    - (A)-[*1..3]->(B)  # Variable length paths
    """
    
    def __init__(self, graph: KnowledgeGraphStore):
        """Initialize the pattern matcher.
        
        Args:
            graph: The knowledge graph to match against.
        """
        self._graph = graph
    
    def match_triple(
        self,
        subject: Entity | str | EntityType | None = None,
        predicate: RelationshipType | None = None,
        obj: Entity | str | EntityType | None = None,
    ) -> list[tuple[Entity, Relationship, Entity]]:
        """Match triples (subject, predicate, object) in the graph.
        
        Args:
            subject: Subject filter (entity, name, type, or None for any).
            predicate: Relationship type filter.
            obj: Object filter (entity, name, type, or None for any).
            
        Returns:
            List of matching (subject, relationship, object) triples.
        """
        results = []
        
        # Determine candidate subjects
        subjects = self._resolve_node_filter(subject)
        
        for subj in subjects:
            rels = self._graph.get_relationships(subj.entity_id, direction="outgoing")
            
            for rel in rels:
                # Check predicate
                if predicate and rel.relationship_type != predicate:
                    continue
                
                # Check object
                obj_entity = self._graph.get_entity(rel.target_id)
                if not obj_entity:
                    continue
                
                if not self._matches_node_filter(obj_entity, obj):
                    continue
                
                results.append((subj, rel, obj_entity))
        
        return results
    
    def match_path_pattern(
        self,
        path_spec: list[tuple[RelationshipType, bool]],  # (type, is_outgoing)
        start: Entity | str | EntityType | None = None,
        end: Entity | str | EntityType | None = None,
    ) -> list[list[tuple[Entity, Relationship | None]]]:
        """Match a specific path pattern.
        
        Args:
            start: Starting node filter.
            path_spec: List of (relationship_type, is_outgoing) tuples.
            end: Ending node filter.
            
        Returns:
            List of matching paths.
        """
        results = []
        starts = self._resolve_node_filter(start)
        
        for start_entity in starts:
            paths = self._match_path_recursive(
                start_entity, path_spec, 0, [(start_entity, None)]
            )
            
            for path in paths:
                if end is None:
                    results.append(path)
                else:
                    final_entity = path[-1][0]
                    if self._matches_node_filter(final_entity, end):
                        results.append(path)
        
        return results
    
    def _match_path_recursive(
        self,
        current: Entity,
        path_spec: list[tuple[RelationshipType, bool]],
        index: int,
        current_path: list[tuple[Entity, Relationship | None]],
    ) -> list[list[tuple[Entity, Relationship | None]]]:
        """Recursively match path patterns."""
        if index >= len(path_spec):
            return [current_path]
        
        rel_type, is_outgoing = path_spec[index]
        direction = "outgoing" if is_outgoing else "incoming"
        
        rels = self._graph.get_relationships(
            current.entity_id,
            direction=direction,
            relationship_types=[rel_type],
        )
        
        results = []
        for rel in rels:
            next_id = rel.target_id if is_outgoing else rel.source_id
            next_entity = self._graph.get_entity(next_id)
            
            if next_entity:
                new_path = current_path + [(next_entity, rel)]
                results.extend(
                    self._match_path_recursive(next_entity, path_spec, index + 1, new_path)
                )
        
        return results
    
    def _resolve_node_filter(
        self, filter_spec: Entity | str | EntityType | None
    ) -> list[Entity]:
        """Resolve a node filter specification to a list of entities."""
        if filter_spec is None:
            return list(self._graph)
        elif isinstance(filter_spec, Entity):
            return [filter_spec]
        elif isinstance(filter_spec, EntityType):
            return self._graph.find_entities_by_type(filter_spec)
        elif isinstance(filter_spec, str):
            # Try as ID first, then as name
            entity = self._graph.get_entity(filter_spec)
            if entity:
                return [entity]
            return self._graph.find_entities_by_name(filter_spec, fuzzy=True)
        return []
    
    def _matches_node_filter(
        self, entity: Entity, filter_spec: Entity | str | EntityType | None
    ) -> bool:
        """Check if an entity matches a filter specification."""
        if filter_spec is None:
            return True
        elif isinstance(filter_spec, Entity):
            return entity.entity_id == filter_spec.entity_id
        elif isinstance(filter_spec, EntityType):
            return entity.entity_type == filter_spec
        elif isinstance(filter_spec, str):
            return (
                entity.entity_id == filter_spec or
                entity.name.lower() == filter_spec.lower() or
                filter_spec.lower() in entity.name.lower()
            )
        return False


def parse_query_string(query_string: str) -> dict[str, Any]:
    """Parse a simple query string into query parameters.
    
    Supports a simple query syntax:
    - "type:concept importance:>0.5 python"
    - "from:entity_id traverse:is_a,part_of depth:2"
    
    Args:
        query_string: The query string to parse.
        
    Returns:
        Dictionary of parsed query parameters.
    """
    params: dict[str, Any] = {
        "text_search": [],
        "filters": {},
        "options": {},
    }
    
    tokens = query_string.split()
    
    for token in tokens:
        if ":" in token:
            key, value = token.split(":", 1)
            key = key.lower()
            
            if key == "type":
                params["filters"]["entity_type"] = value
            elif key == "importance":
                if value.startswith(">"):
                    params["filters"]["min_importance"] = float(value[1:])
                elif value.startswith("<"):
                    params["filters"]["max_importance"] = float(value[1:])
                else:
                    params["filters"]["importance"] = float(value)
            elif key == "from":
                params["options"]["start_from"] = value
            elif key == "traverse":
                params["options"]["traverse"] = value.split(",")
            elif key == "depth":
                params["options"]["max_depth"] = int(value)
            elif key == "limit":
                params["options"]["limit"] = int(value)
            else:
                params["filters"][key] = value
        else:
            params["text_search"].append(token)
    
    return params
