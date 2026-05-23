"""
Memory Conflict Resolution

Handles conflicting memories with configurable strategies:
- Latest wins: Most recent memory takes precedence
- Highest confidence: Memory with highest confidence wins
- Manual: Requires explicit resolution
- Merge: Combine values when possible
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from .domains import CognitiveDomain, Memory


class ConflictStrategy(Enum):
    """Strategies for resolving memory conflicts."""
    
    LATEST_WINS = "latest_wins"         # Most recent memory wins
    CONFIDENCE_WINS = "confidence_wins"  # Highest confidence wins
    KEEP_BOTH = "keep_both"             # Keep both with versions
    MERGE = "merge"                     # Attempt to merge values
    MANUAL = "manual"                   # Require manual resolution


@dataclass
class Conflict:
    """Represents a conflict between two memories."""
    
    memory1: Memory
    memory2: Memory
    conflict_type: str  # "value", "confidence", "temporal"
    description: str
    resolution: Optional[Memory] = None
    resolved: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "memory1": self.memory1.to_dict(),
            "memory2": self.memory2.to_dict(),
            "conflict_type": self.conflict_type,
            "description": self.description,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "resolved": self.resolved,
        }


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""
    
    resolved_memories: list[Memory]
    conflicts_found: int
    conflicts_resolved: int
    unresolved_conflicts: list[Conflict]
    
    def __len__(self) -> int:
        return len(self.resolved_memories)


class ConflictResolver:
    """
    Resolve conflicts between memories.
    
    A conflict occurs when two memories have the same domain and key
    but different values.
    """
    
    def __init__(
        self,
        strategy: ConflictStrategy = ConflictStrategy.CONFIDENCE_WINS,
        merge_fn: Optional[Callable[[Memory, Memory], Memory]] = None,
    ):
        """
        Initialize conflict resolver.
        
        Args:
            strategy: Default resolution strategy
            merge_fn: Custom function for merging memories (for MERGE strategy)
        """
        self.strategy = strategy
        self.merge_fn = merge_fn or self._default_merge
    
    def resolve(
        self,
        memories: list[Memory],
        strategy: Optional[ConflictStrategy] = None
    ) -> ResolutionResult:
        """
        Resolve conflicts in a list of memories.
        
        Args:
            memories: List of memories potentially containing conflicts
            strategy: Override default strategy for this resolution
            
        Returns:
            ResolutionResult with resolved memories and conflict info
        """
        strategy = strategy or self.strategy
        
        # Group by domain + key
        groups: dict[tuple[CognitiveDomain, str], list[Memory]] = {}
        for memory in memories:
            key = (memory.domain, memory.key.lower())
            if key not in groups:
                groups[key] = []
            groups[key].append(memory)
        
        resolved = []
        conflicts = []
        
        for (domain, key), group in groups.items():
            if len(group) == 1:
                resolved.append(group[0])
            else:
                conflict_result = self._resolve_group(group, strategy)
                resolved.extend(conflict_result["resolved"])
                conflicts.extend(conflict_result["conflicts"])
        
        return ResolutionResult(
            resolved_memories=resolved,
            conflicts_found=len(conflicts),
            conflicts_resolved=sum(1 for c in conflicts if c.resolved),
            unresolved_conflicts=[c for c in conflicts if not c.resolved],
        )
    
    def _resolve_group(
        self,
        memories: list[Memory],
        strategy: ConflictStrategy
    ) -> dict[str, Any]:
        """Resolve conflicts within a group of same-key memories."""
        conflicts = []
        
        # Check for actual value conflicts
        unique_values = set(m.value.lower() for m in memories)
        
        if len(unique_values) == 1:
            # No value conflict, just keep highest confidence
            best = max(memories, key=lambda m: m.confidence)
            return {"resolved": [best], "conflicts": []}
        
        # We have a conflict
        sorted_memories = sorted(memories, key=lambda m: (m.timestamp, m.confidence))
        
        for i in range(len(sorted_memories) - 1):
            for j in range(i + 1, len(sorted_memories)):
                m1, m2 = sorted_memories[i], sorted_memories[j]
                if m1.value.lower() != m2.value.lower():
                    conflicts.append(Conflict(
                        memory1=m1,
                        memory2=m2,
                        conflict_type="value",
                        description=f"Different values for {m1.key}: '{m1.value}' vs '{m2.value}'",
                    ))
        
        # Resolve based on strategy
        if strategy == ConflictStrategy.LATEST_WINS:
            winner = max(memories, key=lambda m: m.timestamp)
            for c in conflicts:
                c.resolution = winner
                c.resolved = True
            return {"resolved": [winner], "conflicts": conflicts}
        
        elif strategy == ConflictStrategy.CONFIDENCE_WINS:
            winner = max(memories, key=lambda m: m.confidence)
            for c in conflicts:
                c.resolution = winner
                c.resolved = True
            return {"resolved": [winner], "conflicts": conflicts}
        
        elif strategy == ConflictStrategy.KEEP_BOTH:
            # Add version numbers to values
            versioned = []
            for i, memory in enumerate(sorted_memories, 1):
                new_memory = Memory(
                    domain=memory.domain,
                    key=f"{memory.key}_v{i}",
                    value=memory.value,
                    confidence=memory.confidence,
                    source=memory.source,
                    timestamp=memory.timestamp,
                    metadata={**memory.metadata, "version": i, "original_key": memory.key},
                )
                versioned.append(new_memory)
            for c in conflicts:
                c.resolved = True
                c.resolution = versioned[-1]  # Latest version
            return {"resolved": versioned, "conflicts": conflicts}
        
        elif strategy == ConflictStrategy.MERGE:
            # Try to merge the memories
            merged = sorted_memories[0]
            for other in sorted_memories[1:]:
                merged = self.merge_fn(merged, other)
            for c in conflicts:
                c.resolution = merged
                c.resolved = True
            return {"resolved": [merged], "conflicts": conflicts}
        
        else:  # MANUAL
            # Don't resolve, return all as unresolved
            return {"resolved": memories, "conflicts": conflicts}
    
    def _default_merge(self, memory1: Memory, memory2: Memory) -> Memory:
        """
        Default merge function: combine values, take highest confidence.
        """
        # If one value contains the other, use the longer one
        v1, v2 = memory1.value, memory2.value
        
        if v1.lower() in v2.lower():
            merged_value = v2
        elif v2.lower() in v1.lower():
            merged_value = v1
        else:
            # Combine values
            merged_value = f"{v1}; {v2}"
        
        return Memory(
            domain=memory1.domain,
            key=memory1.key,
            value=merged_value,
            confidence=max(memory1.confidence, memory2.confidence),
            source=memory1.source or memory2.source,
            timestamp=max(memory1.timestamp, memory2.timestamp),
            metadata={
                **memory1.metadata,
                **memory2.metadata,
                "merged_from": [memory1.memory_id, memory2.memory_id],
            },
        )
    
    def find_conflicts(
        self,
        new_memory: Memory,
        existing_memories: list[Memory]
    ) -> list[Conflict]:
        """
        Find conflicts between a new memory and existing ones.
        
        Args:
            new_memory: The new memory to check
            existing_memories: List of existing memories
            
        Returns:
            List of conflicts found
        """
        conflicts = []
        
        for existing in existing_memories:
            if (existing.domain == new_memory.domain and
                existing.key.lower() == new_memory.key.lower() and
                existing.value.lower() != new_memory.value.lower()):
                
                conflicts.append(Conflict(
                    memory1=existing,
                    memory2=new_memory,
                    conflict_type="value",
                    description=(
                        f"New value '{new_memory.value}' conflicts with "
                        f"existing value '{existing.value}' for key '{existing.key}'"
                    ),
                ))
        
        return conflicts
    
    def resolve_conflict(
        self,
        conflict: Conflict,
        strategy: Optional[ConflictStrategy] = None
    ) -> Memory:
        """
        Resolve a single conflict.
        
        Args:
            conflict: The conflict to resolve
            strategy: Strategy to use (defaults to instance strategy)
            
        Returns:
            The resolved memory
        """
        strategy = strategy or self.strategy
        
        if strategy == ConflictStrategy.LATEST_WINS:
            winner = max([conflict.memory1, conflict.memory2], key=lambda m: m.timestamp)
        elif strategy == ConflictStrategy.CONFIDENCE_WINS:
            winner = max([conflict.memory1, conflict.memory2], key=lambda m: m.confidence)
        elif strategy == ConflictStrategy.MERGE:
            winner = self.merge_fn(conflict.memory1, conflict.memory2)
        else:
            # Default to higher confidence
            winner = max([conflict.memory1, conflict.memory2], key=lambda m: m.confidence)
        
        conflict.resolution = winner
        conflict.resolved = True
        return winner
