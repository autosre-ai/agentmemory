"""
Deduplication strategies for memory consolidation.

Implements various strategies for handling duplicate memories:
- Keep newest
- Keep oldest
- Keep highest confidence
- Merge all
- Keep most accessed
"""

import logging
from datetime import datetime
from typing import Any, Optional
import uuid

from .models import DeduplicationStrategy, SimilarityScore
from .similarity import MemoryData, SimilarityDetector

logger = logging.getLogger(__name__)


class DeduplicationResult:
    """Result of a deduplication operation."""
    
    def __init__(
        self,
        kept_memory_ids: list[str],
        removed_memory_ids: list[str],
        duplicate_groups: list[list[str]],
        strategy_used: DeduplicationStrategy,
        details: Optional[dict[str, Any]] = None,
    ):
        self.kept_memory_ids = kept_memory_ids
        self.removed_memory_ids = removed_memory_ids
        self.duplicate_groups = duplicate_groups
        self.strategy_used = strategy_used
        self.details = details or {}
    
    @property
    def total_removed(self) -> int:
        return len(self.removed_memory_ids)
    
    @property
    def total_kept(self) -> int:
        return len(self.kept_memory_ids)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "kept_memory_ids": self.kept_memory_ids,
            "removed_memory_ids": self.removed_memory_ids,
            "duplicate_groups": self.duplicate_groups,
            "strategy_used": self.strategy_used.value,
            "total_removed": self.total_removed,
            "total_kept": self.total_kept,
            "details": self.details,
        }


class Deduplicator:
    """
    Remove duplicate memories using configurable strategies.
    
    Strategies:
    - KEEP_NEWEST: Keep the most recently created/updated memory
    - KEEP_OLDEST: Keep the original (oldest) memory
    - KEEP_HIGHEST_CONFIDENCE: Keep memory with highest confidence score
    - MERGE_ALL: Merge duplicate content into a single memory
    - KEEP_MOST_ACCESSED: Keep memory with highest access count
    """
    
    def __init__(
        self,
        strategy: DeduplicationStrategy = DeduplicationStrategy.KEEP_HIGHEST_CONFIDENCE,
        detector: Optional[SimilarityDetector] = None,
        duplicate_threshold: float = 0.95,
    ):
        """
        Initialize deduplicator.
        
        Args:
            strategy: Default deduplication strategy
            detector: SimilarityDetector for finding duplicates
            duplicate_threshold: Minimum similarity to consider as duplicate
        """
        self.strategy = strategy
        self.detector = detector or SimilarityDetector(
            duplicate_threshold=duplicate_threshold
        )
        self.duplicate_threshold = duplicate_threshold
    
    def deduplicate(
        self,
        memories: list[MemoryData],
        strategy: Optional[DeduplicationStrategy] = None,
    ) -> DeduplicationResult:
        """
        Remove duplicates from a list of memories.
        
        Args:
            memories: List of memories to deduplicate
            strategy: Strategy to use (defaults to instance strategy)
            
        Returns:
            DeduplicationResult with kept and removed memories
        """
        strategy = strategy or self.strategy
        
        # Find duplicate groups
        duplicate_groups = self._find_duplicate_groups(memories)
        
        if not duplicate_groups:
            return DeduplicationResult(
                kept_memory_ids=[m.id for m in memories],
                removed_memory_ids=[],
                duplicate_groups=[],
                strategy_used=strategy,
            )
        
        # Apply strategy to each group
        kept_ids = set()
        removed_ids = set()
        
        # First, add all non-duplicate memories
        duplicate_member_ids = set()
        for group in duplicate_groups:
            duplicate_member_ids.update(group)
        
        for memory in memories:
            if memory.id not in duplicate_member_ids:
                kept_ids.add(memory.id)
        
        # Then process duplicate groups
        for group in duplicate_groups:
            group_memories = [m for m in memories if m.id in group]
            
            keeper = self._select_keeper(group_memories, strategy)
            kept_ids.add(keeper.id)
            
            for m in group_memories:
                if m.id != keeper.id:
                    removed_ids.add(m.id)
        
        return DeduplicationResult(
            kept_memory_ids=list(kept_ids),
            removed_memory_ids=list(removed_ids),
            duplicate_groups=duplicate_groups,
            strategy_used=strategy,
            details={
                "total_groups": len(duplicate_groups),
                "largest_group": max(len(g) for g in duplicate_groups) if duplicate_groups else 0,
            },
        )
    
    def _find_duplicate_groups(
        self,
        memories: list[MemoryData],
    ) -> list[list[str]]:
        """Find groups of duplicate memories."""
        # First try exact hash groups for efficiency
        hash_groups = self.detector.quick_hash_groups(memories)
        
        # Convert to list format
        groups = list(hash_groups.values())
        
        # For remaining memories, do fuzzy/semantic matching
        hashed_ids = set()
        for ids in hash_groups.values():
            hashed_ids.update(ids)
        
        remaining = [m for m in memories if m.id not in hashed_ids]
        
        if remaining:
            # Find similar pairs among remaining
            similar_pairs = self.detector.find_duplicates(remaining)
            
            # Build groups using union-find
            parent: dict[str, str] = {}
            
            def find(x: str) -> str:
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x: str, y: str) -> None:
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
            
            for pair in similar_pairs:
                union(pair.memory1_id, pair.memory2_id)
            
            # Collect groups
            group_map: dict[str, list[str]] = {}
            for mid in parent:
                root = find(mid)
                if root not in group_map:
                    group_map[root] = []
                group_map[root].append(mid)
            
            # Add groups with multiple members
            for group_ids in group_map.values():
                if len(group_ids) > 1:
                    groups.append(group_ids)
        
        return groups
    
    def _select_keeper(
        self,
        memories: list[MemoryData],
        strategy: DeduplicationStrategy,
    ) -> MemoryData:
        """Select which memory to keep based on strategy."""
        if not memories:
            raise ValueError("No memories to select from")
        
        if len(memories) == 1:
            return memories[0]
        
        if strategy == DeduplicationStrategy.KEEP_NEWEST:
            return max(
                memories,
                key=lambda m: self._get_timestamp(m, use_updated=True)
            )
        
        elif strategy == DeduplicationStrategy.KEEP_OLDEST:
            return min(
                memories,
                key=lambda m: self._get_timestamp(m, use_updated=False)
            )
        
        elif strategy == DeduplicationStrategy.KEEP_HIGHEST_CONFIDENCE:
            return max(
                memories,
                key=lambda m: self._get_confidence(m)
            )
        
        elif strategy == DeduplicationStrategy.KEEP_MOST_ACCESSED:
            return max(
                memories,
                key=lambda m: self._get_access_count(m)
            )
        
        else:  # MERGE_ALL or fallback
            # For merge, we return the one with highest confidence
            # Actual merging is done separately
            return max(
                memories,
                key=lambda m: self._get_confidence(m)
            )
    
    def _get_timestamp(
        self,
        memory: MemoryData,
        use_updated: bool = True,
    ) -> datetime:
        """Get timestamp from memory metadata."""
        if memory.metadata:
            if use_updated and "updated_at" in memory.metadata:
                ts = memory.metadata["updated_at"]
                if isinstance(ts, str):
                    return datetime.fromisoformat(ts)
                return ts
            if "created_at" in memory.metadata:
                ts = memory.metadata["created_at"]
                if isinstance(ts, str):
                    return datetime.fromisoformat(ts)
                return ts
        return datetime.min
    
    def _get_confidence(self, memory: MemoryData) -> float:
        """Get confidence score from memory metadata."""
        if memory.metadata and "confidence" in memory.metadata:
            return memory.metadata["confidence"]
        return 1.0
    
    def _get_access_count(self, memory: MemoryData) -> int:
        """Get access count from memory metadata."""
        if memory.metadata and "access_count" in memory.metadata:
            return memory.metadata["access_count"]
        return 0
    
    def find_and_remove_exact_duplicates(
        self,
        memories: list[MemoryData],
    ) -> tuple[list[MemoryData], list[str]]:
        """
        Quick pass to remove exact duplicates.
        
        Args:
            memories: List of memories
            
        Returns:
            Tuple of (unique memories, removed IDs)
        """
        hash_groups = self.detector.quick_hash_groups(memories)
        
        # For each group, keep one and mark rest for removal
        removed_ids = []
        kept_ids = set()
        
        for group_ids in hash_groups.values():
            # Keep first, remove rest
            kept_ids.add(group_ids[0])
            removed_ids.extend(group_ids[1:])
        
        # Also keep all non-duplicate memories
        duplicate_ids = set()
        for ids in hash_groups.values():
            duplicate_ids.update(ids)
        
        unique = []
        for m in memories:
            if m.id not in duplicate_ids or m.id in kept_ids:
                unique.append(m)
        
        return unique, removed_ids
    
    def estimate_duplicates(
        self,
        memories: list[MemoryData],
    ) -> dict[str, Any]:
        """
        Estimate number of duplicates without full processing.
        
        Args:
            memories: List of memories
            
        Returns:
            Statistics about potential duplicates
        """
        # Quick hash-based estimate
        hash_groups = self.detector.quick_hash_groups(memories)
        
        exact_duplicates = sum(len(g) - 1 for g in hash_groups.values())
        
        # Sample-based estimate for fuzzy duplicates
        sample_size = min(100, len(memories))
        if sample_size < 10:
            fuzzy_estimate = 0
        else:
            import random
            sample = random.sample(memories, sample_size)
            sample_pairs = self.detector.find_all_similar_pairs(
                sample, min_score=self.duplicate_threshold
            )
            
            # Extrapolate
            total_pairs = len(memories) * (len(memories) - 1) // 2
            sample_pairs_possible = sample_size * (sample_size - 1) // 2
            
            if sample_pairs_possible > 0:
                fuzzy_rate = len(sample_pairs) / sample_pairs_possible
                fuzzy_estimate = int(fuzzy_rate * total_pairs)
            else:
                fuzzy_estimate = 0
        
        return {
            "total_memories": len(memories),
            "exact_duplicates": exact_duplicates,
            "exact_duplicate_groups": len(hash_groups),
            "estimated_fuzzy_duplicates": fuzzy_estimate,
            "estimated_total_duplicates": exact_duplicates + fuzzy_estimate,
            "estimated_reduction_percent": (
                (exact_duplicates + fuzzy_estimate) / max(len(memories), 1) * 100
            ),
        }
