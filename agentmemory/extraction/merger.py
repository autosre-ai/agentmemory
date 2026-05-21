"""
Memory Merger

Combines memories from multiple sources with:
- Deduplication
- Conflict resolution
- Confidence aggregation
- Source tracking
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from .domains import CognitiveDomain, Memory
from .deduplication import MemoryDeduplicator, DeduplicationResult
from .conflict_resolver import ConflictResolver, ConflictStrategy, ResolutionResult


@dataclass
class MergeResult:
    """Result of memory merge operation."""
    
    merged_memories: list[Memory]
    total_input: int
    duplicates_removed: int
    conflicts_resolved: int
    sources_merged: int
    
    def by_domain(self) -> dict[CognitiveDomain, list[Memory]]:
        """Group merged memories by domain."""
        result: dict[CognitiveDomain, list[Memory]] = defaultdict(list)
        for memory in self.merged_memories:
            result[memory.domain].append(memory)
        return dict(result)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "merged_memories": [m.to_dict() for m in self.merged_memories],
            "total_input": self.total_input,
            "duplicates_removed": self.duplicates_removed,
            "conflicts_resolved": self.conflicts_resolved,
            "sources_merged": self.sources_merged,
        }


class MemoryMerger:
    """
    Merge memories from multiple sources.
    
    Handles:
    - Deduplication across sources
    - Conflict resolution
    - Confidence boosting for corroborated facts
    - Source provenance tracking
    """
    
    def __init__(
        self,
        deduplicator: Optional[MemoryDeduplicator] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        boost_corroborated: bool = True,
        corroboration_boost: float = 0.1,
    ):
        """
        Initialize memory merger.
        
        Args:
            deduplicator: Deduplicator instance (created if not provided)
            conflict_resolver: Conflict resolver (created if not provided)
            boost_corroborated: Whether to boost confidence for corroborated memories
            corroboration_boost: Confidence boost per corroboration (up to 1.0)
        """
        self.deduplicator = deduplicator or MemoryDeduplicator(strategy="fuzzy")
        self.conflict_resolver = conflict_resolver or ConflictResolver(
            strategy=ConflictStrategy.CONFIDENCE_WINS
        )
        self.boost_corroborated = boost_corroborated
        self.corroboration_boost = corroboration_boost
    
    def merge(
        self,
        *memory_sets: list[Memory],
        source_labels: Optional[list[str]] = None
    ) -> MergeResult:
        """
        Merge multiple sets of memories.
        
        Args:
            *memory_sets: Variable number of memory lists to merge
            source_labels: Optional labels for each source
            
        Returns:
            MergeResult with merged memories and statistics
        """
        if not memory_sets:
            return MergeResult([], 0, 0, 0, 0)
        
        # Track sources
        sources = set()
        all_memories: list[Memory] = []
        
        for i, memory_set in enumerate(memory_sets):
            source_label = source_labels[i] if source_labels and i < len(source_labels) else f"source_{i}"
            
            for memory in memory_set:
                # Add source tracking
                memory_with_source = Memory(
                    domain=memory.domain,
                    key=memory.key,
                    value=memory.value,
                    confidence=memory.confidence,
                    source=memory.source or source_label,
                    timestamp=memory.timestamp,
                    metadata={
                        **memory.metadata,
                        "merge_source": source_label,
                    },
                )
                all_memories.append(memory_with_source)
                sources.add(source_label)
        
        total_input = len(all_memories)
        
        # Step 1: Deduplicate
        dedupe_result = self.deduplicator.deduplicate(all_memories)
        
        # Boost corroborated memories (found in multiple sources)
        if self.boost_corroborated:
            self._boost_corroborated(dedupe_result)
        
        # Step 2: Resolve conflicts
        resolution_result = self.conflict_resolver.resolve(dedupe_result.unique_memories)
        
        return MergeResult(
            merged_memories=resolution_result.resolved_memories,
            total_input=total_input,
            duplicates_removed=dedupe_result.duplicates_removed,
            conflicts_resolved=resolution_result.conflicts_resolved,
            sources_merged=len(sources),
        )
    
    def _boost_corroborated(self, dedupe_result: DeduplicationResult) -> None:
        """
        Boost confidence for memories that were corroborated by multiple sources.
        
        Modifies memories in place.
        """
        for group in dedupe_result.duplicate_groups:
            if len(group) < 2:
                continue
            
            # Check if duplicates came from different sources
            sources = set()
            for memory in group:
                source = memory.metadata.get("merge_source", memory.source)
                if source:
                    sources.add(source)
            
            if len(sources) > 1:
                # Corroborated by multiple sources - boost the surviving memory
                for memory in dedupe_result.unique_memories:
                    if memory.similar_to(group[0]):
                        # Apply boost
                        boost = self.corroboration_boost * (len(sources) - 1)
                        memory.confidence = min(1.0, memory.confidence + boost)
                        memory.metadata["corroborated_by"] = list(sources)
                        memory.metadata["corroboration_count"] = len(sources)
                        break
    
    def merge_into(
        self,
        base_memories: list[Memory],
        new_memories: list[Memory],
        source_label: Optional[str] = None
    ) -> MergeResult:
        """
        Merge new memories into an existing base.
        
        Args:
            base_memories: Existing memory store
            new_memories: New memories to merge in
            source_label: Label for the new memories source
            
        Returns:
            MergeResult with complete merged memory set
        """
        return self.merge(
            base_memories,
            new_memories,
            source_labels=["base", source_label or "new"]
        )
    
    def incremental_merge(
        self,
        existing: list[Memory],
        incoming: Memory
    ) -> tuple[list[Memory], Optional[Memory]]:
        """
        Merge a single incoming memory into existing memories.
        
        More efficient for real-time memory updates.
        
        Args:
            existing: Current memory store
            incoming: New memory to merge
            
        Returns:
            Tuple of (updated memories, replaced memory if any)
        """
        # Check for duplicates
        duplicates = self.deduplicator.find_duplicates(incoming, existing)
        
        if duplicates:
            # Found duplicates - check if we should replace
            best_existing = max(duplicates, key=lambda m: m.confidence)
            
            if incoming.confidence > best_existing.confidence:
                # Replace with new memory
                result = [m for m in existing if m not in duplicates]
                
                # Boost if corroborated
                if self.boost_corroborated:
                    incoming.confidence = min(1.0, incoming.confidence + self.corroboration_boost)
                    incoming.metadata["corroborated"] = True
                
                result.append(incoming)
                return result, best_existing
            else:
                # Keep existing, maybe boost confidence
                if self.boost_corroborated:
                    best_existing.confidence = min(1.0, best_existing.confidence + self.corroboration_boost)
                    best_existing.metadata["corroborated"] = True
                return existing, None
        
        # Check for conflicts
        conflicts = self.conflict_resolver.find_conflicts(incoming, existing)
        
        if conflicts:
            # Resolve conflict
            resolved = self.conflict_resolver.resolve_conflict(conflicts[0])
            
            result = [m for m in existing if m != conflicts[0].memory1]
            result.append(resolved)
            return result, conflicts[0].memory1
        
        # No duplicates or conflicts - just add
        return existing + [incoming], None


class MultiSourceMerger:
    """
    Specialized merger for handling memories from multiple agents/sources.
    
    Tracks provenance and supports weighted source trust.
    """
    
    def __init__(
        self,
        source_weights: Optional[dict[str, float]] = None,
        default_weight: float = 1.0,
    ):
        """
        Initialize multi-source merger.
        
        Args:
            source_weights: Trust weights per source (higher = more trusted)
            default_weight: Default weight for unknown sources
        """
        self.source_weights = source_weights or {}
        self.default_weight = default_weight
        self.merger = MemoryMerger()
    
    def merge_with_weights(
        self,
        memories_by_source: dict[str, list[Memory]]
    ) -> MergeResult:
        """
        Merge memories with source-based weighting.
        
        Confidence is adjusted by source trust weight before merging.
        
        Args:
            memories_by_source: Dict mapping source names to memory lists
            
        Returns:
            MergeResult with weighted merged memories
        """
        weighted_memories: dict[str, list[Memory]] = {}
        
        for source, memories in memories_by_source.items():
            weight = self.source_weights.get(source, self.default_weight)
            
            weighted = []
            for memory in memories:
                # Apply weight to confidence
                weighted_memory = Memory(
                    domain=memory.domain,
                    key=memory.key,
                    value=memory.value,
                    confidence=memory.confidence * weight,
                    source=source,
                    timestamp=memory.timestamp,
                    metadata={
                        **memory.metadata,
                        "original_confidence": memory.confidence,
                        "source_weight": weight,
                    },
                )
                weighted.append(weighted_memory)
            
            weighted_memories[source] = weighted
        
        # Merge all weighted memories
        all_sets = list(weighted_memories.values())
        source_labels = list(weighted_memories.keys())
        
        return self.merger.merge(*all_sets, source_labels=source_labels)
