# Knowledge Graph

Agent Memory Toolkit provides a comprehensive knowledge graph module for structured knowledge representation, relationship management, and graph-based reasoning.

## Overview

The knowledge graph module consists of four main components:

- **Knowledge Store**: Entity and relationship storage with indexing and caching
- **Relationships**: Typed relationships with properties, weights, and confidence scores
- **Query**: Fluent query interface with filtering, traversal, and aggregation
- **Reasoning**: Inference engine with transitive, symmetric, and causal reasoning

## Quick Start

```python
from agent_memory_toolkit.graph import (
    KnowledgeGraphStore,
    Entity,
    EntityType,
    RelationshipType,
    GraphQuery,
    ReasoningEngine,
)

# Create a knowledge graph
graph = KnowledgeGraphStore()

# Add entities
python = graph.add_entity(
    name="Python",
    entity_type=EntityType.CONCEPT,
    description="A programming language",
    importance=0.8,
)

programming = graph.add_entity(
    name="Programming",
    entity_type=EntityType.CONCEPT,
    description="The art of writing code",
)

# Add a relationship
graph.add_relationship(
    python.entity_id,
    programming.entity_id,
    RelationshipType.IS_A,
    weight=0.9,
    confidence=1.0,
)

# Query related entities
related = graph.get_related_entities(python.entity_id, max_depth=2)
for entity, depth, rel in related:
    print(f"  {entity.name} (depth={depth}, via {rel.relationship_type.value})")
```

## Entity Types

Entities represent nodes in the knowledge graph:

```python
from agent_memory_toolkit.graph import EntityType

# Available entity types
EntityType.CONCEPT      # Abstract concepts or ideas
EntityType.FACT         # Factual statements
EntityType.ENTITY       # Named entities (people, places, things)
EntityType.EVENT        # Events or occurrences
EntityType.PROCEDURE    # Procedures or processes
EntityType.PREFERENCE   # User preferences
EntityType.GOAL         # Goals or objectives
EntityType.MEMORY       # Direct memory references
EntityType.TOPIC        # Topic or subject clusters
```

## Relationship Types

Relationships connect entities with typed, weighted edges:

```python
from agent_memory_toolkit.graph import RelationshipType, RelationshipCategory

# Semantic relationships
RelationshipType.IS_A           # "Python is_a Language"
RelationshipType.SIMILAR_TO     # "Python similar_to JavaScript"
RelationshipType.OPPOSITE_OF    # "Hot opposite_of Cold"

# Temporal relationships
RelationshipType.BEFORE         # "Meeting before Lunch"
RelationshipType.AFTER          # "Dessert after Dinner"
RelationshipType.DURING         # "Notes during Meeting"

# Causal relationships
RelationshipType.CAUSES         # "Rain causes Flooding"
RelationshipType.ENABLES        # "Power enables Computer"
RelationshipType.PREVENTS       # "Vaccine prevents Disease"

# Hierarchical relationships
RelationshipType.PART_OF        # "Wheel part_of Car"
RelationshipType.CONTAINS       # "Book contains Chapters"
RelationshipType.BELONGS_TO     # "File belongs_to Folder"

# Memory-specific relationships
RelationshipType.DERIVED_FROM   # Memory derivation
RelationshipType.CONTRADICTS    # Conflicting information
RelationshipType.SUPPORTS       # Supporting evidence
RelationshipType.UPDATES        # Information updates
```

## Knowledge Graph Store

### Configuration

```python
from agent_memory_toolkit.graph import KnowledgeGraphStore, KnowledgeGraphConfig

config = KnowledgeGraphConfig(
    max_entities=100000,
    entity_merge_threshold=0.9,
    max_relationships_per_entity=1000,
    enable_inverse_relationships=True,
    enable_decay=True,
    decay_rate=0.01,
    min_weight_threshold=0.1,
    enable_caching=True,
)

graph = KnowledgeGraphStore(config=config)
```

### Adding Entities

```python
# Basic entity
entity = graph.add_entity(
    name="Machine Learning",
    entity_type=EntityType.CONCEPT,
)

# Entity with full metadata
entity = graph.add_entity(
    name="GPT-4",
    entity_type=EntityType.ENTITY,
    description="Large language model by OpenAI",
    embedding=[0.1, 0.2, ...],  # Optional embedding vector
    properties={"version": "4", "provider": "OpenAI"},
    importance=0.9,
    source_memory_id="mem_123",  # Link to source memory
)
```

### Finding Entities

```python
# Find by ID
entity = graph.get_entity("entity_id")

# Find by name (exact match)
entities = graph.find_entities_by_name("Python")

# Find by name (fuzzy match)
entities = graph.find_entities_by_name("Pyth", fuzzy=True)

# Find by type
concepts = graph.find_entities_by_type(EntityType.CONCEPT)
```

### Adding Relationships

```python
# Basic relationship
graph.add_relationship(
    source_id=entity1.entity_id,
    target_id=entity2.entity_id,
    relationship_type=RelationshipType.IS_A,
)

# Relationship with properties
graph.add_relationship(
    source_id=cause.entity_id,
    target_id=effect.entity_id,
    relationship_type=RelationshipType.CAUSES,
    weight=0.8,           # Strength of relationship
    confidence=0.95,      # Confidence in relationship
    bidirectional=False,  # One-way relationship
    metadata={"source": "user_input"},
)
```

### Traversing the Graph

```python
# Get directly related entities
related = graph.get_related_entities(
    entity_id=python.entity_id,
    max_depth=2,
    relationship_types=[RelationshipType.IS_A, RelationshipType.SIMILAR_TO],
    entity_types=[EntityType.CONCEPT],
    min_weight=0.5,
)

for entity, depth, relationship in related:
    print(f"{entity.name} at depth {depth}")

# Find path between entities
path = graph.find_path(
    source_id=entity1.entity_id,
    target_id=entity2.entity_id,
    max_depth=5,
)

if path:
    for entity, rel in path:
        print(f"-> {entity.name}")
```

### Merging Entities

```python
# Merge duplicate entities
merged = graph.merge_entities(
    primary_id=canonical_entity.entity_id,
    secondary_id=duplicate_entity.entity_id,
)
# All relationships from secondary are transferred to primary
```

### Graph Statistics

```python
stats = graph.get_stats()

print(f"Total entities: {stats.total_entities}")
print(f"Total relationships: {stats.total_relationships}")
print(f"Entities by type: {stats.entities_by_type}")
print(f"Most connected: {stats.most_connected_entities[:5]}")
```

## Graph Queries

The query interface provides a fluent API for complex graph queries:

### Basic Queries

```python
from agent_memory_toolkit.graph import GraphQuery, EntityFilter, SortOrder

query = GraphQuery(graph)

# Find all concepts
results = (query
    .start_from_type(EntityType.CONCEPT)
    .execute())

# Find by name
results = (query
    .start_from_name("Python", fuzzy=True)
    .execute())
```

### Filtering

```python
# Filter by entity properties
entity_filter = (EntityFilter()
    .by_type(EntityType.CONCEPT, EntityType.FACT)
    .by_importance(min_importance=0.5)
    .by_created_after(datetime(2024, 1, 1))
    .by_property("verified", value=True))

results = (GraphQuery(graph)
    .start_from_type(EntityType.CONCEPT)
    .filter_entities(entity_filter)
    .execute())
```

### Traversal

```python
from agent_memory_toolkit.graph import RelationshipFilter, RelationshipCategory

# Traverse specific relationship types
results = (GraphQuery(graph)
    .start_from(entity_id)
    .traverse(
        RelationshipType.IS_A,
        RelationshipType.PART_OF,
        max_depth=3,
        direction="outgoing",  # or "incoming", "both"
    )
    .execute())

# Filter relationships during traversal
rel_filter = (RelationshipFilter()
    .by_type(RelationshipType.CAUSES)
    .by_weight(min_weight=0.7)
    .by_confidence(min_confidence=0.8))

results = (GraphQuery(graph)
    .start_from(entity_id)
    .traverse(max_depth=2)
    .filter_relationships(rel_filter)
    .execute())
```

### Sorting and Pagination

```python
results = (GraphQuery(graph)
    .start_from_type(EntityType.CONCEPT)
    .sort_by("importance", SortOrder.DESC)
    .offset(10)
    .limit(20)
    .execute())

print(f"Total: {results.total_count}")
print(f"Returned: {len(results.matches)}")
print(f"Truncated: {results.truncated}")
```

### Aggregations

```python
from agent_memory_toolkit.graph import AggregateFunction

results = (GraphQuery(graph)
    .start_from_type(EntityType.CONCEPT)
    .aggregate("importance", AggregateFunction.AVG)
    .aggregate("access_count", AggregateFunction.SUM)
    .aggregate("importance", AggregateFunction.MAX)
    .execute())

print(f"Avg importance: {results.aggregations['avg_importance']}")
print(f"Total accesses: {results.aggregations['sum_access_count']}")
```

### Path Inclusion

```python
# Include paths in results
results = (GraphQuery(graph)
    .start_from(start_entity_id)
    .traverse(max_depth=3)
    .with_paths()
    .execute())

for match in results:
    print(f"Entity: {match.entity.name}")
    print(f"Path length: {len(match.path)}")
    for entity, rel in match.path:
        print(f"  -> {entity.name}")
```

## Pattern Matching

Match complex graph patterns:

```python
from agent_memory_toolkit.graph import PatternMatcher

matcher = PatternMatcher(graph)

# Match triples (subject, predicate, object)
triples = matcher.match_triple(
    subject=EntityType.CONCEPT,        # Any concept
    predicate=RelationshipType.IS_A,   # is_a relationship
    obj="Programming",                  # Target name
)

for subject, rel, obj in triples:
    print(f"{subject.name} is_a {obj.name}")

# Match path patterns
paths = matcher.match_path_pattern(
    path_spec=[
        (RelationshipType.IS_A, True),      # Outgoing IS_A
        (RelationshipType.PART_OF, True),   # Outgoing PART_OF
    ],
    start=EntityType.CONCEPT,
    end=None,  # Any ending
)
```

## Query String Parsing

Parse simple query strings:

```python
from agent_memory_toolkit.graph import parse_query_string

params = parse_query_string("type:concept importance:>0.5 machine learning")

# params = {
#     "text_search": ["machine", "learning"],
#     "filters": {
#         "entity_type": "concept",
#         "min_importance": 0.5,
#     },
#     "options": {},
# }
```

## Reasoning Engine

The reasoning engine enables inference over the knowledge graph:

### Inference

```python
from agent_memory_toolkit.graph import ReasoningEngine, InferenceType

engine = ReasoningEngine(graph)

# Run inference
inferences = engine.infer_relationships(
    max_depth=2,
    min_confidence=0.5,
    inference_types=[InferenceType.TRANSITIVE, InferenceType.SYMMETRIC],
)

for inf in inferences:
    print(f"Inferred: {inf.source_entity.name} -> {inf.target_entity.name}")
    print(f"  Type: {inf.inferred_type.value}")
    print(f"  Confidence: {inf.confidence:.2f}")
    print(f"  Explanation: {inf.explanation}")

# Apply inferences to graph
inferences = engine.infer_relationships(
    apply_to_graph=True,  # Adds inferred relationships
)
```

### Custom Inference Rules

```python
from agent_memory_toolkit.graph import InferenceRule

custom_rule = InferenceRule(
    rule_id="custom_inheritance",
    name="Custom Property Inheritance",
    inference_type=InferenceType.INHERITANCE,
    antecedent_types=[RelationshipType.IS_A],
    consequent_type=RelationshipType.RELATED_TO,
    confidence_factor=0.7,
    description="Inherit properties from parent types",
)

engine.add_rule(custom_rule)
```

### Causal Reasoning

```python
# Trace causal chains
chains = engine.trace_causality(
    cause_entity_id=rain.entity_id,
    max_steps=5,
    min_confidence=0.5,
)

for chain in chains:
    print(f"Chain confidence: {chain.total_confidence:.2f}")
    for cause, rel, effect in chain.steps:
        print(f"  {cause.name} -> {effect.name}")
    print(f"  Final effects: {[e.name for e in chain.effects]}")
```

### Analogy Finding

```python
# Find analogies for an entity
analogies = engine.find_analogies(
    source_entity_id=python.entity_id,
    target_type=EntityType.CONCEPT,  # Optional filter
    min_similarity=0.5,
    max_results=10,
)

for analogy in analogies:
    print(f"Similarity: {analogy.similarity_score:.2f}")
    print(f"Mapping: {analogy.mapping}")
    print(f"Explanation: {analogy.explanation}")
```

### Explanation Generation

```python
# Explain relationship between entities
explanation = engine.explain_relationship(
    source_id=entity1.entity_id,
    target_id=entity2.entity_id,
    max_path_length=4,
)

if explanation:
    print(f"Statement: {explanation.statement}")
    print(f"Confidence: {explanation.confidence:.2f}")
    print("Reasoning chain:")
    for step in explanation.reasoning_chain:
        print(f"  - {step}")
```

### Contradiction Detection

```python
# Find contradictions in the graph
contradictions = engine.find_contradictions()

for rel1, rel2, explanation in contradictions:
    print(f"Contradiction: {explanation}")
```

### Relationship Suggestions

```python
# Get suggestions for new relationships
suggestions = engine.suggest_relationships(
    entity_id=entity.entity_id,
    min_confidence=0.5,
    max_suggestions=10,
)

for target, rel_type, confidence, reason in suggestions:
    print(f"Suggest: {entity.name} -{rel_type.value}-> {target.name}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Reason: {reason}")
```

## Persistence

### Serialization

```python
import json

# Export graph to JSON
graph_data = graph.to_dict()
with open("knowledge_graph.json", "w") as f:
    json.dump(graph_data, f)

# Import graph from JSON
with open("knowledge_graph.json", "r") as f:
    graph_data = json.load(f)

graph = KnowledgeGraphStore.from_dict(graph_data)
```

## Integration with Memory Store

```python
from agent_memory_toolkit import MemoryStore
from agent_memory_toolkit.graph import KnowledgeGraphStore, EntityType

# Create both stores
memory_store = MemoryStore("./memories")
graph = KnowledgeGraphStore()

# When adding a memory, also add to graph
memory_id = memory_store.add("Python is a programming language")

# Extract entities and relationships
python_entity = graph.add_entity(
    name="Python",
    entity_type=EntityType.CONCEPT,
    source_memory_id=memory_id,
)

# Link memory to entity
python_entity.link_memory(memory_id)
```

## Decay and Maintenance

```python
# Apply temporal decay to relationships
removed_count = graph.apply_decay()
print(f"Removed {removed_count} weak relationships")

# Manual relationship pruning
stats = graph.get_stats()
if stats.total_relationships > 100000:
    # Remove weakest relationships
    for entity in graph:
        rels = graph.get_relationships(entity.entity_id)
        for rel in rels:
            if rel.properties.weight < 0.1:
                graph.delete_relationship(rel.relationship_id)
```

## Best Practices

1. **Entity Naming**: Use consistent, normalized names for entities
2. **Relationship Weights**: Start with 1.0 and decay over time
3. **Confidence Scores**: Use lower confidence for inferred relationships
4. **Regular Maintenance**: Apply decay periodically to prune weak relationships
5. **Batch Operations**: Add entities and relationships in batches for performance
6. **Memory Linking**: Always link entities to their source memories
7. **Type Consistency**: Use appropriate entity types for better querying
