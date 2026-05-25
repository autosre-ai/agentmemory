"""Graph-based reasoning for knowledge graphs.

This module provides reasoning capabilities over the knowledge graph,
including inference, analogy detection, causal reasoning, and 
explanation generation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from .knowledge import Entity, EntityType, KnowledgeGraphStore
from .relationships import (
    Relationship,
    RelationshipType,
    RelationshipCategory,
    RelationshipProperties,
    get_relationship_category,
    get_inverse_relationship,
)
from .query import GraphQuery, QueryResult, EntityFilter, RelationshipFilter

logger = logging.getLogger(__name__)


class InferenceType(Enum):
    """Types of inference supported by the reasoning engine."""
    TRANSITIVE = "transitive"         # A->B, B->C => A->C
    SYMMETRIC = "symmetric"           # A->B => B->A
    INHERITANCE = "inheritance"       # Subclass/superclass inference
    COMPOSITION = "composition"       # Part-whole relationships
    ANALOGY = "analogy"               # Analogical reasoning
    CAUSAL = "causal"                 # Causal chain inference
    TEMPORAL = "temporal"             # Temporal ordering inference


@dataclass
class InferenceRule:
    """A rule for inferring new relationships."""
    rule_id: str
    name: str
    inference_type: InferenceType
    antecedent_types: list[RelationshipType]  # Required relationship types
    consequent_type: RelationshipType          # Inferred relationship type
    confidence_factor: float = 0.8             # Confidence reduction for inference
    description: str = ""
    
    def __hash__(self) -> int:
        return hash(self.rule_id)


@dataclass
class InferenceResult:
    """Result of an inference operation."""
    source_entity: Entity
    target_entity: Entity
    inferred_type: RelationshipType
    confidence: float
    rule_used: InferenceRule
    supporting_path: list[tuple[Entity, Relationship | None]]
    explanation: str


@dataclass
class Explanation:
    """An explanation for a relationship or inference."""
    statement: str
    confidence: float
    evidence: list[tuple[Entity, Relationship | None]]
    reasoning_chain: list[str]
    sources: list[str] = field(default_factory=list)


@dataclass 
class Analogy:
    """An analogy between two sets of entities."""
    source_entities: list[Entity]
    target_entities: list[Entity]
    mapping: dict[str, str]  # source_id -> target_id
    similarity_score: float
    shared_structure: list[tuple[RelationshipType, str, str]]  # (rel_type, mapped_src, mapped_tgt)
    explanation: str


@dataclass
class CausalChain:
    """A chain of causal relationships."""
    steps: list[tuple[Entity, Relationship, Entity]]
    total_confidence: float
    effects: list[Entity]
    time_ordering: list[datetime] | None = None


# Standard inference rules
STANDARD_RULES: list[InferenceRule] = [
    InferenceRule(
        rule_id="transitive_is_a",
        name="Transitive IS_A",
        inference_type=InferenceType.TRANSITIVE,
        antecedent_types=[RelationshipType.IS_A, RelationshipType.IS_A],
        consequent_type=RelationshipType.IS_A,
        confidence_factor=0.9,
        description="If A is_a B and B is_a C, then A is_a C",
    ),
    InferenceRule(
        rule_id="transitive_part_of",
        name="Transitive PART_OF",
        inference_type=InferenceType.TRANSITIVE,
        antecedent_types=[RelationshipType.PART_OF, RelationshipType.PART_OF],
        consequent_type=RelationshipType.PART_OF,
        confidence_factor=0.85,
        description="If A part_of B and B part_of C, then A part_of C",
    ),
    InferenceRule(
        rule_id="transitive_causes",
        name="Transitive CAUSES",
        inference_type=InferenceType.CAUSAL,
        antecedent_types=[RelationshipType.CAUSES, RelationshipType.CAUSES],
        consequent_type=RelationshipType.LEADS_TO,
        confidence_factor=0.7,
        description="If A causes B and B causes C, then A leads_to C",
    ),
    InferenceRule(
        rule_id="symmetric_similar",
        name="Symmetric SIMILAR_TO",
        inference_type=InferenceType.SYMMETRIC,
        antecedent_types=[RelationshipType.SIMILAR_TO],
        consequent_type=RelationshipType.SIMILAR_TO,
        confidence_factor=0.95,
        description="If A similar_to B, then B similar_to A",
    ),
    InferenceRule(
        rule_id="inheritance_via_is_a",
        name="Property Inheritance",
        inference_type=InferenceType.INHERITANCE,
        antecedent_types=[RelationshipType.IS_A],
        consequent_type=RelationshipType.RELATED_TO,
        confidence_factor=0.75,
        description="Properties of superclass may apply to subclass",
    ),
    InferenceRule(
        rule_id="temporal_ordering",
        name="Temporal Ordering",
        inference_type=InferenceType.TEMPORAL,
        antecedent_types=[RelationshipType.BEFORE, RelationshipType.BEFORE],
        consequent_type=RelationshipType.BEFORE,
        confidence_factor=0.95,
        description="If A before B and B before C, then A before C",
    ),
]


class ReasoningEngine:
    """Engine for performing reasoning over a knowledge graph.
    
    Example:
        engine = ReasoningEngine(graph)
        
        # Run inference
        inferences = engine.infer_relationships(max_depth=2)
        
        # Find analogies
        analogies = engine.find_analogies(source_entity, target_type)
        
        # Trace causality
        causal_chain = engine.trace_causality(cause_entity, max_steps=5)
        
        # Generate explanation
        explanation = engine.explain_relationship(entity1, entity2)
    """
    
    def __init__(
        self,
        graph: KnowledgeGraphStore,
        rules: list[InferenceRule] | None = None,
    ):
        """Initialize the reasoning engine.
        
        Args:
            graph: The knowledge graph to reason over.
            rules: Optional custom inference rules.
        """
        self._graph = graph
        self._rules = rules or STANDARD_RULES.copy()
        self._rule_index: dict[InferenceType, list[InferenceRule]] = defaultdict(list)
        
        for rule in self._rules:
            self._rule_index[rule.inference_type].append(rule)
        
        logger.info("Initialized ReasoningEngine with %d rules", len(self._rules))
    
    def add_rule(self, rule: InferenceRule) -> None:
        """Add a custom inference rule.
        
        Args:
            rule: The rule to add.
        """
        self._rules.append(rule)
        self._rule_index[rule.inference_type].append(rule)
    
    def infer_relationships(
        self,
        max_depth: int = 2,
        min_confidence: float = 0.5,
        inference_types: list[InferenceType] | None = None,
        apply_to_graph: bool = False,
    ) -> list[InferenceResult]:
        """Infer new relationships using the registered rules.
        
        Args:
            max_depth: Maximum inference chain depth.
            min_confidence: Minimum confidence for inferred relationships.
            inference_types: Filter to specific inference types.
            apply_to_graph: Whether to add inferred relationships to the graph.
            
        Returns:
            List of inferred relationships.
        """
        results = []
        
        # Filter rules by type
        active_rules = self._rules
        if inference_types:
            active_rules = [r for r in self._rules if r.inference_type in inference_types]
        
        for rule in active_rules:
            if rule.inference_type == InferenceType.TRANSITIVE:
                results.extend(self._apply_transitive_rule(rule, max_depth, min_confidence))
            elif rule.inference_type == InferenceType.SYMMETRIC:
                results.extend(self._apply_symmetric_rule(rule, min_confidence))
            elif rule.inference_type == InferenceType.CAUSAL:
                results.extend(self._apply_causal_rule(rule, max_depth, min_confidence))
        
        # Apply to graph if requested
        if apply_to_graph:
            for result in results:
                self._graph.add_relationship(
                    result.source_entity.entity_id,
                    result.target_entity.entity_id,
                    result.inferred_type,
                    confidence=result.confidence,
                    metadata={"inferred": True, "rule": result.rule_used.rule_id},
                )
        
        logger.info("Inferred %d new relationships", len(results))
        return results
    
    def _apply_transitive_rule(
        self,
        rule: InferenceRule,
        max_depth: int,
        min_confidence: float,
    ) -> list[InferenceResult]:
        """Apply a transitive inference rule."""
        results = []
        
        if len(rule.antecedent_types) < 2:
            return results
        
        rel_type = rule.antecedent_types[0]
        
        # For each entity, find transitive closures
        for entity in self._graph:
            visited = {entity.entity_id}
            current_level = [(entity, 1.0, [(entity, None)])]  # (entity, confidence, path)
            
            for depth in range(1, max_depth + 1):
                next_level = []
                
                for current, conf, path in current_level:
                    rels = self._graph.get_relationships(
                        current.entity_id,
                        direction="outgoing",
                        relationship_types=[rel_type],
                    )
                    
                    for rel in rels:
                        target = self._graph.get_entity(rel.target_id)
                        if not target or target.entity_id in visited:
                            continue
                        
                        visited.add(target.entity_id)
                        new_conf = conf * rel.properties.confidence * rule.confidence_factor
                        
                        if depth > 1 and new_conf >= min_confidence:
                            # We've gone at least 2 steps, so we can infer
                            new_path = path + [(target, rel)]
                            results.append(InferenceResult(
                                source_entity=entity,
                                target_entity=target,
                                inferred_type=rule.consequent_type,
                                confidence=new_conf,
                                rule_used=rule,
                                supporting_path=new_path,
                                explanation=f"Transitive inference: {entity.name} -> {target.name} via {rule.name}",
                            ))
                        
                        new_path = path + [(target, rel)]
                        next_level.append((target, new_conf, new_path))
                
                current_level = next_level
        
        return results
    
    def _apply_symmetric_rule(
        self,
        rule: InferenceRule,
        min_confidence: float,
    ) -> list[InferenceResult]:
        """Apply a symmetric inference rule."""
        results = []
        rel_type = rule.antecedent_types[0]
        
        for entity in self._graph:
            rels = self._graph.get_relationships(
                entity.entity_id,
                direction="outgoing",
                relationship_types=[rel_type],
            )
            
            for rel in rels:
                target = self._graph.get_entity(rel.target_id)
                if not target:
                    continue
                
                # Check if reverse already exists
                existing = self._graph.get_relationships(
                    target.entity_id,
                    direction="outgoing",
                    relationship_types=[rule.consequent_type],
                )
                
                has_reverse = any(
                    r.target_id == entity.entity_id 
                    for r in existing
                )
                
                if not has_reverse:
                    new_conf = rel.properties.confidence * rule.confidence_factor
                    if new_conf >= min_confidence:
                        results.append(InferenceResult(
                            source_entity=target,
                            target_entity=entity,
                            inferred_type=rule.consequent_type,
                            confidence=new_conf,
                            rule_used=rule,
                            supporting_path=[(entity, rel), (target, None)],
                            explanation=f"Symmetric inference: {target.name} -> {entity.name}",
                        ))
        
        return results
    
    def _apply_causal_rule(
        self,
        rule: InferenceRule,
        max_depth: int,
        min_confidence: float,
    ) -> list[InferenceResult]:
        """Apply a causal inference rule."""
        # Similar to transitive, but specifically for causal relationships
        return self._apply_transitive_rule(rule, max_depth, min_confidence)
    
    def trace_causality(
        self,
        cause_entity_id: str,
        max_steps: int = 5,
        min_confidence: float = 0.5,
    ) -> list[CausalChain]:
        """Trace causal chains from a given entity.
        
        Args:
            cause_entity_id: Starting entity ID.
            max_steps: Maximum chain length.
            min_confidence: Minimum confidence threshold.
            
        Returns:
            List of causal chains.
        """
        cause = self._graph.get_entity(cause_entity_id)
        if not cause:
            return []
        
        causal_types = [
            RelationshipType.CAUSES,
            RelationshipType.LEADS_TO,
            RelationshipType.ENABLES,
        ]
        
        chains: list[CausalChain] = []
        
        # DFS to find all causal chains
        def dfs(
            current: Entity,
            chain: list[tuple[Entity, Relationship, Entity]],
            confidence: float,
        ) -> None:
            if len(chain) >= max_steps:
                if chain:
                    effects = [step[2] for step in chain]
                    chains.append(CausalChain(
                        steps=chain,
                        total_confidence=confidence,
                        effects=effects,
                    ))
                return
            
            rels = self._graph.get_relationships(
                current.entity_id,
                direction="outgoing",
                relationship_types=causal_types,
            )
            
            has_continuation = False
            for rel in rels:
                effect = self._graph.get_entity(rel.target_id)
                if not effect:
                    continue
                
                new_conf = confidence * rel.properties.confidence
                if new_conf < min_confidence:
                    continue
                
                has_continuation = True
                new_chain = chain + [(current, rel, effect)]
                dfs(effect, new_chain, new_conf)
            
            # End of chain
            if not has_continuation and chain:
                effects = [step[2] for step in chain]
                chains.append(CausalChain(
                    steps=chain,
                    total_confidence=confidence,
                    effects=effects,
                ))
        
        dfs(cause, [], 1.0)
        
        # Sort by confidence
        chains.sort(key=lambda c: c.total_confidence, reverse=True)
        
        return chains
    
    def find_analogies(
        self,
        source_entity_id: str,
        target_type: EntityType | None = None,
        min_similarity: float = 0.5,
        max_results: int = 10,
    ) -> list[Analogy]:
        """Find analogical matches for an entity.
        
        Args:
            source_entity_id: The source entity to find analogies for.
            target_type: Optional type filter for target entities.
            min_similarity: Minimum structural similarity.
            max_results: Maximum number of results.
            
        Returns:
            List of analogies found.
        """
        source = self._graph.get_entity(source_entity_id)
        if not source:
            return []
        
        # Get source entity's relationship structure
        source_structure = self._get_relationship_structure(source.entity_id)
        
        if not source_structure:
            return []
        
        analogies = []
        
        # Compare with other entities
        for candidate in self._graph:
            if candidate.entity_id == source.entity_id:
                continue
            
            if target_type and candidate.entity_type != target_type:
                continue
            
            candidate_structure = self._get_relationship_structure(candidate.entity_id)
            
            if not candidate_structure:
                continue
            
            similarity, shared = self._compute_structural_similarity(
                source_structure, candidate_structure
            )
            
            if similarity >= min_similarity:
                analogies.append(Analogy(
                    source_entities=[source],
                    target_entities=[candidate],
                    mapping={source.entity_id: candidate.entity_id},
                    similarity_score=similarity,
                    shared_structure=shared,
                    explanation=self._generate_analogy_explanation(source, candidate, shared),
                ))
        
        # Sort by similarity
        analogies.sort(key=lambda a: a.similarity_score, reverse=True)
        
        return analogies[:max_results]
    
    def _get_relationship_structure(
        self, entity_id: str
    ) -> dict[RelationshipType, list[str]]:
        """Get the relationship structure of an entity."""
        structure: dict[RelationshipType, list[str]] = defaultdict(list)
        
        rels = self._graph.get_relationships(entity_id, direction="both")
        
        for rel in rels:
            other_id = rel.target_id if rel.source_id == entity_id else rel.source_id
            other = self._graph.get_entity(other_id)
            if other:
                structure[rel.relationship_type].append(other.entity_type.value)
        
        return structure
    
    def _compute_structural_similarity(
        self,
        struct1: dict[RelationshipType, list[str]],
        struct2: dict[RelationshipType, list[str]],
    ) -> tuple[float, list[tuple[RelationshipType, str, str]]]:
        """Compute structural similarity between two relationship structures."""
        shared = []
        total_types = set(struct1.keys()) | set(struct2.keys())
        
        if not total_types:
            return 0.0, []
        
        matching_count = 0
        total_count = 0
        
        for rel_type in total_types:
            types1 = set(struct1.get(rel_type, []))
            types2 = set(struct2.get(rel_type, []))
            
            common = types1 & types2
            all_types = types1 | types2
            
            if all_types:
                total_count += len(all_types)
                matching_count += len(common)
                
                for t in common:
                    shared.append((rel_type, t, t))
        
        similarity = matching_count / total_count if total_count > 0 else 0.0
        
        return similarity, shared
    
    def _generate_analogy_explanation(
        self,
        source: Entity,
        target: Entity,
        shared: list[tuple[RelationshipType, str, str]],
    ) -> str:
        """Generate a natural language explanation for an analogy."""
        if not shared:
            return f"{source.name} and {target.name} share no common relationship patterns."
        
        patterns = []
        for rel_type, src_type, tgt_type in shared:
            patterns.append(f"{rel_type.value} relationships with {src_type} entities")
        
        pattern_str = ", ".join(patterns[:3])
        if len(patterns) > 3:
            pattern_str += f" and {len(patterns) - 3} more"
        
        return f"{source.name} is analogous to {target.name} because both have {pattern_str}."
    
    def explain_relationship(
        self,
        source_id: str,
        target_id: str,
        max_path_length: int = 4,
    ) -> Explanation | None:
        """Generate an explanation for the relationship between two entities.
        
        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            max_path_length: Maximum path length to consider.
            
        Returns:
            Explanation object or None if no relationship found.
        """
        source = self._graph.get_entity(source_id)
        target = self._graph.get_entity(target_id)
        
        if not source or not target:
            return None
        
        # Try to find a direct relationship
        direct_rels = self._graph.get_relationships(source_id, direction="outgoing")
        for rel in direct_rels:
            if rel.target_id == target_id:
                return Explanation(
                    statement=f"{source.name} {rel.relationship_type.value} {target.name}",
                    confidence=rel.properties.confidence,
                    evidence=[(source, rel), (target, None)],
                    reasoning_chain=[
                        f"Direct relationship: {rel.relationship_type.value}",
                    ],
                    sources=rel.properties.evidence,
                )
        
        # Try to find an indirect path
        path = self._graph.find_path(source_id, target_id, max_depth=max_path_length)
        
        if path:
            reasoning_chain = []
            total_confidence = 1.0
            
            for i, (entity, rel) in enumerate(path):
                if rel:
                    reasoning_chain.append(
                        f"Step {i}: {path[i-1][0].name if i > 0 else source.name} "
                        f"{rel.relationship_type.value} {entity.name}"
                    )
                    total_confidence *= rel.properties.confidence
            
            return Explanation(
                statement=f"{source.name} is connected to {target.name} through {len(path)-1} relationships",
                confidence=total_confidence,
                evidence=path,
                reasoning_chain=reasoning_chain,
            )
        
        return None
    
    def find_contradictions(
        self,
        entity_id: str | None = None,
    ) -> list[tuple[Relationship, Relationship, str]]:
        """Find contradictory relationships in the graph.
        
        Args:
            entity_id: Optional entity to check for contradictions.
            
        Returns:
            List of (rel1, rel2, explanation) tuples.
        """
        contradictions = []
        
        entities = [self._graph.get_entity(entity_id)] if entity_id else list(self._graph)
        entities = [e for e in entities if e is not None]
        
        # Known contradictory pairs
        contradictory_pairs = [
            (RelationshipType.CAUSES, RelationshipType.PREVENTS),
            (RelationshipType.BEFORE, RelationshipType.AFTER),
            (RelationshipType.SUPPORTS, RelationshipType.CONTRADICTS),
        ]
        
        for entity in entities:
            rels = self._graph.get_relationships(entity.entity_id, direction="outgoing")
            
            # Group by target
            by_target: dict[str, list[Relationship]] = defaultdict(list)
            for rel in rels:
                by_target[rel.target_id].append(rel)
            
            # Check for contradictions to same target
            for target_id, target_rels in by_target.items():
                for i, rel1 in enumerate(target_rels):
                    for rel2 in target_rels[i+1:]:
                        for pair in contradictory_pairs:
                            if (rel1.relationship_type == pair[0] and rel2.relationship_type == pair[1]) or \
                               (rel1.relationship_type == pair[1] and rel2.relationship_type == pair[0]):
                                target = self._graph.get_entity(target_id)
                                target_name = target.name if target else target_id
                                contradictions.append((
                                    rel1,
                                    rel2,
                                    f"{entity.name} both {rel1.relationship_type.value} and "
                                    f"{rel2.relationship_type.value} {target_name}",
                                ))
        
        return contradictions
    
    def suggest_relationships(
        self,
        entity_id: str,
        min_confidence: float = 0.5,
        max_suggestions: int = 10,
    ) -> list[tuple[Entity, RelationshipType, float, str]]:
        """Suggest potential relationships for an entity.
        
        Args:
            entity_id: The entity to suggest relationships for.
            min_confidence: Minimum confidence for suggestions.
            max_suggestions: Maximum number of suggestions.
            
        Returns:
            List of (target_entity, rel_type, confidence, reason) tuples.
        """
        entity = self._graph.get_entity(entity_id)
        if not entity:
            return []
        
        suggestions = []
        
        # Find similar entities and their relationships
        similar = self._graph.get_related_entities(
            entity_id,
            max_depth=2,
            relationship_types=[RelationshipType.SIMILAR_TO, RelationshipType.RELATED_TO],
            min_weight=0.5,
        )
        
        for similar_entity, depth, _ in similar:
            # Get relationships from similar entity
            similar_rels = self._graph.get_relationships(
                similar_entity.entity_id,
                direction="outgoing",
            )
            
            for rel in similar_rels:
                target = self._graph.get_entity(rel.target_id)
                if not target or target.entity_id == entity_id:
                    continue
                
                # Check if relationship already exists
                existing = self._graph.get_relationships(entity_id, direction="outgoing")
                already_exists = any(
                    r.target_id == target.entity_id and r.relationship_type == rel.relationship_type
                    for r in existing
                )
                
                if not already_exists:
                    # Confidence decreases with depth
                    confidence = rel.properties.confidence * (0.8 ** depth)
                    if confidence >= min_confidence:
                        reason = f"Similar entity {similar_entity.name} has this relationship"
                        suggestions.append((target, rel.relationship_type, confidence, reason))
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_suggestions = []
        for target, rel_type, conf, reason in suggestions:
            key = (target.entity_id, rel_type)
            if key not in seen:
                seen.add(key)
                unique_suggestions.append((target, rel_type, conf, reason))
        
        unique_suggestions.sort(key=lambda x: x[2], reverse=True)
        
        return unique_suggestions[:max_suggestions]
