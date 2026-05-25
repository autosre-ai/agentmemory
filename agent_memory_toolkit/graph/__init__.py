"""
Knowledge Graph Module for Agent Memory Toolkit

Graph-based memory storage and reasoning capabilities for AI agents.
Enables structured knowledge representation, relationship inference,
and graph-based queries for enhanced memory retrieval.

Usage:
    from agent_memory_toolkit.graph import (
        KnowledgeGraphStore,
        Entity,
        EntityType,
        Relationship,
        RelationshipType,
        GraphQuery,
        ReasoningEngine,
    )
    
    # Create a knowledge graph
    graph = KnowledgeGraphStore()
    
    # Add entities
    python = graph.add_entity("Python", EntityType.CONCEPT, "A programming language")
    programming = graph.add_entity("Programming", EntityType.CONCEPT, "Writing code")
    
    # Add relationships
    graph.add_relationship(python.entity_id, programming.entity_id, RelationshipType.IS_A)
    
    # Query the graph
    query = GraphQuery(graph)
    results = (query
        .start_from_type(EntityType.CONCEPT)
        .traverse(RelationshipType.IS_A, max_depth=2)
        .limit(10)
        .execute())
    
    # Perform reasoning
    engine = ReasoningEngine(graph)
    inferences = engine.infer_relationships(max_depth=2)
"""

from .relationships import (
    RelationshipCategory,
    RelationshipType,
    RelationshipProperties,
    Relationship,
    RelationshipPattern,
    STANDARD_PATTERNS,
    get_inverse_relationship,
    get_relationship_category,
)
from .knowledge import (
    EntityType,
    Entity,
    KnowledgeGraphConfig,
    GraphStats,
    KnowledgeGraphStore,
)
from .query import (
    AggregateFunction,
    SortOrder,
    QueryMatch,
    QueryResult,
    EntityFilter,
    RelationshipFilter,
    GraphQuery,
    PatternMatcher,
    parse_query_string,
)
from .reasoning import (
    InferenceType,
    InferenceRule,
    InferenceResult,
    Explanation,
    Analogy,
    CausalChain,
    STANDARD_RULES,
    ReasoningEngine,
)

__all__ = [
    # Relationships
    "RelationshipCategory",
    "RelationshipType",
    "RelationshipProperties",
    "Relationship",
    "RelationshipPattern",
    "STANDARD_PATTERNS",
    "get_inverse_relationship",
    "get_relationship_category",
    # Knowledge
    "EntityType",
    "Entity",
    "KnowledgeGraphConfig",
    "GraphStats",
    "KnowledgeGraphStore",
    # Query
    "AggregateFunction",
    "SortOrder",
    "QueryMatch",
    "QueryResult",
    "EntityFilter",
    "RelationshipFilter",
    "GraphQuery",
    "PatternMatcher",
    "parse_query_string",
    # Reasoning
    "InferenceType",
    "InferenceRule",
    "InferenceResult",
    "Explanation",
    "Analogy",
    "CausalChain",
    "STANDARD_RULES",
    "ReasoningEngine",
]
