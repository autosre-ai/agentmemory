"""Semantic Memory Deduplication - Eliminate duplicate and similar memories.

This module provides semantic deduplication capabilities to identify and
consolidate similar memories, reducing storage footprint while preserving
information integrity.

Deduplication strategies:
- Exact: Remove identical content
- Fuzzy: Remove near-identical content (edit distance)
- Semantic: Remove semantically similar content (embeddings)
- Hybrid: Combine multiple strategies
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional, Protocol, Sequence, TypeVar

import numpy as np


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class DeduplicationStrategy(str, Enum):
    """Available deduplication strategies."""
    EXACT = "exact"           # Exact content match
    FUZZY = "fuzzy"           # Fuzzy text matching
    SEMANTIC = "semantic"     # Semantic similarity
    HYBRID = "hybrid"         # Combine strategies


@dataclass
class MemoryItem:
    """A memory item for deduplication."""
    memory_id: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    
    def __hash__(self) -> int:
        return hash(self.memory_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryItem):
            return False
        return self.memory_id == other.memory_id


@dataclass
class DuplicateGroup:
    """A group of duplicate or similar memories."""
    canonical: MemoryItem  # The memory to keep
    duplicates: list[MemoryItem]  # Memories to remove/merge
    similarity_scores: dict[str, float]  # memory_id -> similarity
    strategy_used: DeduplicationStrategy
    merge_suggestion: Optional[str] = None  # Suggested merged content
    
    @property
    def total_count(self) -> int:
        """Total memories in group including canonical."""
        return 1 + len(self.duplicates)
    
    @property
    def average_similarity(self) -> float:
        """Average similarity score."""
        if not self.similarity_scores:
            return 1.0
        return sum(self.similarity_scores.values()) / len(self.similarity_scores)


@dataclass
class DeduplicationResult:
    """Result of a deduplication operation."""
    original_count: int
    deduplicated_count: int
    duplicate_groups: list[DuplicateGroup]
    removed_ids: list[str]
    merged_memories: list[MemoryItem]  # New memories from merging
    stats: dict[str, Any]
    
    @property
    def duplicates_found(self) -> int:
        """Total duplicates found."""
        return sum(len(g.duplicates) for g in self.duplicate_groups)
    
    @property
    def reduction_percent(self) -> float:
        """Percentage reduction in memory count."""
        if self.original_count == 0:
            return 0.0
        return (self.duplicates_found / self.original_count) * 100


class DeduplicationMatcher(ABC):
    """Abstract base class for deduplication matchers."""
    
    @property
    @abstractmethod
    def strategy(self) -> DeduplicationStrategy:
        """The strategy used by this matcher."""
        ...
    
    @abstractmethod
    def find_duplicates(
        self,
        memories: list[MemoryItem],
        threshold: float,
    ) -> list[DuplicateGroup]:
        """Find groups of duplicate memories.
        
        Args:
            memories: List of memories to check
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of duplicate groups
        """
        ...


class ExactMatcher(DeduplicationMatcher):
    """Find exact duplicate content."""
    
    def __init__(self, normalize: bool = True):
        """Initialize exact matcher.
        
        Args:
            normalize: Normalize content before comparing
        """
        self.normalize = normalize
    
    @property
    def strategy(self) -> DeduplicationStrategy:
        return DeduplicationStrategy.EXACT
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison."""
        if not self.normalize:
            return content
        
        # Lowercase, strip whitespace, normalize spaces
        normalized = content.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _content_hash(self, content: str) -> str:
        """Compute hash of normalized content."""
        normalized = self._normalize_content(content)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def find_duplicates(
        self,
        memories: list[MemoryItem],
        threshold: float = 1.0,  # Not used for exact matching
    ) -> list[DuplicateGroup]:
        """Find exact duplicate groups."""
        # Group by content hash
        hash_groups: dict[str, list[MemoryItem]] = defaultdict(list)
        
        for memory in memories:
            content_hash = self._content_hash(memory.content)
            hash_groups[content_hash].append(memory)
        
        # Create duplicate groups for groups with more than one memory
        groups = []
        for hash_val, group_memories in hash_groups.items():
            if len(group_memories) > 1:
                # Select canonical (prefer higher importance, then older)
                sorted_mems = sorted(
                    group_memories,
                    key=lambda m: (-m.importance, m.created_at),
                )
                canonical = sorted_mems[0]
                duplicates = sorted_mems[1:]
                
                groups.append(DuplicateGroup(
                    canonical=canonical,
                    duplicates=duplicates,
                    similarity_scores={m.memory_id: 1.0 for m in duplicates},
                    strategy_used=self.strategy,
                ))
        
        return groups


class FuzzyMatcher(DeduplicationMatcher):
    """Find near-duplicate content using fuzzy matching."""
    
    def __init__(
        self,
        min_length: int = 20,
        use_jaro_winkler: bool = False,
    ):
        """Initialize fuzzy matcher.
        
        Args:
            min_length: Minimum content length to consider
            use_jaro_winkler: Use Jaro-Winkler instead of SequenceMatcher
        """
        self.min_length = min_length
        self.use_jaro_winkler = use_jaro_winkler
    
    @property
    def strategy(self) -> DeduplicationStrategy:
        return DeduplicationStrategy.FUZZY
    
    def _jaro_winkler(self, s1: str, s2: str) -> float:
        """Compute Jaro-Winkler similarity."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Simple Jaro implementation
        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches / len1 + matches / len2 + 
                (matches - transpositions // 2) / matches) / 3
        
        # Winkler modification
        prefix = 0
        for i in range(min(4, len1, len2)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        return jaro + prefix * 0.1 * (1 - jaro)
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Compute similarity between two strings."""
        if self.use_jaro_winkler:
            return self._jaro_winkler(s1, s2)
        return SequenceMatcher(None, s1, s2).ratio()
    
    def find_duplicates(
        self,
        memories: list[MemoryItem],
        threshold: float = 0.85,
    ) -> list[DuplicateGroup]:
        """Find fuzzy duplicate groups."""
        if len(memories) < 2:
            return []
        
        # Filter by minimum length
        valid_memories = [
            m for m in memories 
            if len(m.content) >= self.min_length
        ]
        
        if len(valid_memories) < 2:
            return []
        
        # Track which memories have been grouped
        grouped: set[str] = set()
        groups: list[DuplicateGroup] = []
        
        for i, mem1 in enumerate(valid_memories):
            if mem1.memory_id in grouped:
                continue
            
            duplicates = []
            scores = {}
            
            for j, mem2 in enumerate(valid_memories[i + 1:], start=i + 1):
                if mem2.memory_id in grouped:
                    continue
                
                similarity = self._similarity(
                    mem1.content.lower(),
                    mem2.content.lower(),
                )
                
                if similarity >= threshold:
                    duplicates.append(mem2)
                    scores[mem2.memory_id] = similarity
                    grouped.add(mem2.memory_id)
            
            if duplicates:
                grouped.add(mem1.memory_id)
                groups.append(DuplicateGroup(
                    canonical=mem1,
                    duplicates=duplicates,
                    similarity_scores=scores,
                    strategy_used=self.strategy,
                ))
        
        return groups


class SemanticMatcher(DeduplicationMatcher):
    """Find semantically similar content using embeddings."""
    
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        similarity_metric: str = "cosine",
    ):
        """Initialize semantic matcher.
        
        Args:
            embedding_provider: Provider for generating embeddings
            similarity_metric: Similarity metric ('cosine' or 'euclidean')
        """
        self.embedding_provider = embedding_provider
        self.similarity_metric = similarity_metric
    
    @property
    def strategy(self) -> DeduplicationStrategy:
        return DeduplicationStrategy.SEMANTIC
    
    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between vectors."""
        a = np.array(v1)
        b = np.array(v2)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _euclidean_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Compute similarity from Euclidean distance."""
        a = np.array(v1)
        b = np.array(v2)
        
        distance = np.linalg.norm(a - b)
        # Convert distance to similarity (0-1 range)
        return float(1 / (1 + distance))
    
    def _similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute similarity between embeddings."""
        if self.similarity_metric == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        else:
            return self._euclidean_similarity(embedding1, embedding2)
    
    def _ensure_embeddings(self, memories: list[MemoryItem]) -> list[MemoryItem]:
        """Ensure all memories have embeddings."""
        if not self.embedding_provider:
            return memories
        
        # Find memories without embeddings
        missing = [m for m in memories if m.embedding is None]
        
        if missing:
            # Batch embed
            texts = [m.content for m in missing]
            embeddings = self.embedding_provider.embed_batch(texts)
            
            for mem, emb in zip(missing, embeddings):
                mem.embedding = emb
        
        return memories
    
    def find_duplicates(
        self,
        memories: list[MemoryItem],
        threshold: float = 0.9,
    ) -> list[DuplicateGroup]:
        """Find semantically similar groups."""
        if len(memories) < 2:
            return []
        
        # Ensure embeddings
        memories = self._ensure_embeddings(memories)
        
        # Filter to memories with embeddings
        valid_memories = [m for m in memories if m.embedding is not None]
        
        if len(valid_memories) < 2:
            return []
        
        # Track which memories have been grouped
        grouped: set[str] = set()
        groups: list[DuplicateGroup] = []
        
        for i, mem1 in enumerate(valid_memories):
            if mem1.memory_id in grouped:
                continue
            
            duplicates = []
            scores = {}
            
            for j, mem2 in enumerate(valid_memories[i + 1:], start=i + 1):
                if mem2.memory_id in grouped:
                    continue
                
                assert mem1.embedding is not None
                assert mem2.embedding is not None
                
                similarity = self._similarity(mem1.embedding, mem2.embedding)
                
                if similarity >= threshold:
                    duplicates.append(mem2)
                    scores[mem2.memory_id] = similarity
                    grouped.add(mem2.memory_id)
            
            if duplicates:
                grouped.add(mem1.memory_id)
                groups.append(DuplicateGroup(
                    canonical=mem1,
                    duplicates=duplicates,
                    similarity_scores=scores,
                    strategy_used=self.strategy,
                ))
        
        return groups


class HybridMatcher(DeduplicationMatcher):
    """Combine multiple deduplication strategies."""
    
    def __init__(
        self,
        matchers: Optional[list[DeduplicationMatcher]] = None,
        merge_mode: str = "union",  # 'union' or 'intersection'
    ):
        """Initialize hybrid matcher.
        
        Args:
            matchers: List of matchers to combine
            merge_mode: How to combine results ('union' or 'intersection')
        """
        self.matchers = matchers or [ExactMatcher(), FuzzyMatcher()]
        self.merge_mode = merge_mode
    
    @property
    def strategy(self) -> DeduplicationStrategy:
        return DeduplicationStrategy.HYBRID
    
    def find_duplicates(
        self,
        memories: list[MemoryItem],
        threshold: float = 0.85,
    ) -> list[DuplicateGroup]:
        """Find duplicates using combined strategies."""
        if not self.matchers or len(memories) < 2:
            return []
        
        # Collect groups from all matchers
        all_groups: list[DuplicateGroup] = []
        for matcher in self.matchers:
            groups = matcher.find_duplicates(memories, threshold)
            all_groups.extend(groups)
        
        if not all_groups:
            return []
        
        # Merge groups by canonical memory
        canonical_groups: dict[str, DuplicateGroup] = {}
        
        for group in all_groups:
            canonical_id = group.canonical.memory_id
            
            if canonical_id in canonical_groups:
                existing = canonical_groups[canonical_id]
                # Merge duplicates
                existing_dup_ids = {m.memory_id for m in existing.duplicates}
                
                for dup in group.duplicates:
                    if dup.memory_id not in existing_dup_ids:
                        existing.duplicates.append(dup)
                        existing.similarity_scores[dup.memory_id] = \
                            group.similarity_scores.get(dup.memory_id, 0.0)
            else:
                canonical_groups[canonical_id] = DuplicateGroup(
                    canonical=group.canonical,
                    duplicates=list(group.duplicates),
                    similarity_scores=dict(group.similarity_scores),
                    strategy_used=self.strategy,
                )
        
        return list(canonical_groups.values())


@dataclass
class DeduplicationConfig:
    """Configuration for semantic deduplication."""
    
    # Strategy selection
    strategy: DeduplicationStrategy = DeduplicationStrategy.FUZZY
    
    # Similarity thresholds
    exact_threshold: float = 1.0
    fuzzy_threshold: float = 0.85
    semantic_threshold: float = 0.9
    
    # Processing options
    normalize_content: bool = True
    min_content_length: int = 20
    
    # Merging options
    auto_merge: bool = False
    merge_strategy: str = "keep_newest"  # 'keep_newest', 'keep_oldest', 'merge_content'
    
    # Embedding options
    embedding_dimension: int = 1536  # OpenAI default
    
    # Performance
    batch_size: int = 100


class SemanticDeduplicator:
    """Intelligent semantic deduplication engine.
    
    Identifies and consolidates duplicate or similar memories
    to reduce storage footprint while preserving information.
    
    Example:
        >>> deduplicator = SemanticDeduplicator()
        >>> 
        >>> # Find duplicates
        >>> memories = [
        ...     MemoryItem("1", "Python is a programming language"),
        ...     MemoryItem("2", "python is a programming language"),  # Near-duplicate
        ...     MemoryItem("3", "JavaScript is also a language"),
        ... ]
        >>> result = deduplicator.deduplicate(memories)
        >>> 
        >>> print(f"Found {result.duplicates_found} duplicates")
        >>> print(f"Reduction: {result.reduction_percent:.1f}%")
    
    Advanced usage with embeddings:
        >>> from my_embeddings import MyEmbeddingProvider
        >>> 
        >>> config = DeduplicationConfig(
        ...     strategy=DeduplicationStrategy.SEMANTIC,
        ...     semantic_threshold=0.92,
        ... )
        >>> deduplicator = SemanticDeduplicator(
        ...     config=config,
        ...     embedding_provider=MyEmbeddingProvider(),
        ... )
    """
    
    def __init__(
        self,
        config: Optional[DeduplicationConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        """Initialize the deduplicator.
        
        Args:
            config: Deduplication configuration
            embedding_provider: Provider for semantic embeddings
        """
        self.config = config or DeduplicationConfig()
        self.embedding_provider = embedding_provider
        
        # Initialize matchers
        self._matchers: dict[DeduplicationStrategy, DeduplicationMatcher] = {
            DeduplicationStrategy.EXACT: ExactMatcher(
                normalize=self.config.normalize_content
            ),
            DeduplicationStrategy.FUZZY: FuzzyMatcher(
                min_length=self.config.min_content_length
            ),
            DeduplicationStrategy.SEMANTIC: SemanticMatcher(
                embedding_provider=embedding_provider
            ),
        }
        
        # Add hybrid matcher
        self._matchers[DeduplicationStrategy.HYBRID] = HybridMatcher(
            matchers=[
                self._matchers[DeduplicationStrategy.EXACT],
                self._matchers[DeduplicationStrategy.FUZZY],
            ]
        )
    
    def _get_threshold(self, strategy: DeduplicationStrategy) -> float:
        """Get threshold for strategy."""
        return {
            DeduplicationStrategy.EXACT: self.config.exact_threshold,
            DeduplicationStrategy.FUZZY: self.config.fuzzy_threshold,
            DeduplicationStrategy.SEMANTIC: self.config.semantic_threshold,
            DeduplicationStrategy.HYBRID: self.config.fuzzy_threshold,
        }.get(strategy, 0.85)
    
    def _merge_memories(self, group: DuplicateGroup) -> Optional[MemoryItem]:
        """Merge a group of duplicate memories."""
        if not self.config.auto_merge:
            return None
        
        all_mems = [group.canonical] + group.duplicates
        
        if self.config.merge_strategy == "keep_newest":
            newest = max(all_mems, key=lambda m: m.created_at)
            return newest
        
        elif self.config.merge_strategy == "keep_oldest":
            oldest = min(all_mems, key=lambda m: m.created_at)
            return oldest
        
        elif self.config.merge_strategy == "merge_content":
            # Create merged memory with combined metadata
            base = group.canonical
            
            # Collect unique content pieces
            all_content = set()
            all_metadata: dict[str, Any] = {}
            
            for mem in all_mems:
                all_content.add(mem.content.strip())
                all_metadata.update(mem.metadata)
            
            # Use canonical content if all are very similar
            if len(all_content) == 1:
                merged_content = base.content
            else:
                # Sort by length, use longest
                merged_content = max(all_content, key=len)
            
            return MemoryItem(
                memory_id=f"merged_{base.memory_id}",
                content=merged_content,
                created_at=min(m.created_at for m in all_mems),
                embedding=base.embedding,
                metadata=all_metadata,
                importance=max(m.importance for m in all_mems),
            )
        
        return None
    
    def find_duplicates(
        self,
        memories: list[MemoryItem],
        strategy: Optional[DeduplicationStrategy] = None,
        threshold: Optional[float] = None,
    ) -> list[DuplicateGroup]:
        """Find duplicate memory groups.
        
        Args:
            memories: Memories to check
            strategy: Override strategy
            threshold: Override similarity threshold
            
        Returns:
            List of duplicate groups
        """
        used_strategy = strategy or self.config.strategy
        used_threshold = threshold or self._get_threshold(used_strategy)
        
        matcher = self._matchers.get(used_strategy)
        if not matcher:
            raise ValueError(f"Unknown strategy: {used_strategy}")
        
        return matcher.find_duplicates(memories, used_threshold)
    
    def deduplicate(
        self,
        memories: list[MemoryItem],
        strategy: Optional[DeduplicationStrategy] = None,
        threshold: Optional[float] = None,
        dry_run: bool = False,
    ) -> DeduplicationResult:
        """Deduplicate memories.
        
        Args:
            memories: Memories to deduplicate
            strategy: Override strategy
            threshold: Override similarity threshold
            dry_run: Only find duplicates, don't merge
            
        Returns:
            DeduplicationResult with deduplication details
        """
        import time
        start_time = time.perf_counter()
        
        original_count = len(memories)
        
        # Find duplicates
        groups = self.find_duplicates(memories, strategy, threshold)
        
        # Calculate removals and merges
        removed_ids: list[str] = []
        merged_memories: list[MemoryItem] = []
        
        for group in groups:
            # Mark duplicates for removal
            removed_ids.extend(m.memory_id for m in group.duplicates)
            
            # Optionally merge
            if not dry_run and self.config.auto_merge:
                merged = self._merge_memories(group)
                if merged:
                    merged_memories.append(merged)
                    group.merge_suggestion = merged.content
        
        # Calculate final count
        deduplicated_count = original_count - len(removed_ids)
        if merged_memories:
            # Merging replaces the group, so add 1 for each merge, subtract canonical
            deduplicated_count = (
                original_count 
                - len(removed_ids) 
                - len(groups)  # Remove canonicals too
                + len(merged_memories)  # Add merged memories
            )
        
        end_time = time.perf_counter()
        
        return DeduplicationResult(
            original_count=original_count,
            deduplicated_count=deduplicated_count,
            duplicate_groups=groups,
            removed_ids=removed_ids,
            merged_memories=merged_memories,
            stats={
                "processing_time_ms": (end_time - start_time) * 1000,
                "strategy_used": (strategy or self.config.strategy).value,
                "threshold_used": threshold or self._get_threshold(
                    strategy or self.config.strategy
                ),
                "groups_found": len(groups),
                "dry_run": dry_run,
            },
        )
    
    def find_similar(
        self,
        query_memory: MemoryItem,
        memories: list[MemoryItem],
        top_k: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[MemoryItem, float]]:
        """Find memories similar to a query.
        
        Args:
            query_memory: Memory to find similar to
            memories: Candidate memories
            top_k: Maximum results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (memory, similarity) tuples, sorted by similarity
        """
        results: list[tuple[MemoryItem, float]] = []
        
        semantic_matcher = self._matchers.get(DeduplicationStrategy.SEMANTIC)
        fuzzy_matcher = self._matchers.get(DeduplicationStrategy.FUZZY)
        
        for mem in memories:
            if mem.memory_id == query_memory.memory_id:
                continue
            
            # Try semantic similarity first
            similarity = 0.0
            
            if (isinstance(semantic_matcher, SemanticMatcher) and 
                query_memory.embedding and mem.embedding):
                similarity = semantic_matcher._similarity(
                    query_memory.embedding, 
                    mem.embedding
                )
            elif isinstance(fuzzy_matcher, FuzzyMatcher):
                similarity = fuzzy_matcher._similarity(
                    query_memory.content.lower(),
                    mem.content.lower(),
                )
            
            if similarity >= threshold:
                results.append((mem, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def compute_similarity_matrix(
        self,
        memories: list[MemoryItem],
    ) -> tuple[list[str], list[list[float]]]:
        """Compute pairwise similarity matrix.
        
        Args:
            memories: Memories to compare
            
        Returns:
            Tuple of (memory_ids, similarity_matrix)
        """
        n = len(memories)
        ids = [m.memory_id for m in memories]
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        fuzzy_matcher = self._matchers.get(DeduplicationStrategy.FUZZY)
        
        for i in range(n):
            matrix[i][i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                if isinstance(fuzzy_matcher, FuzzyMatcher):
                    sim = fuzzy_matcher._similarity(
                        memories[i].content.lower(),
                        memories[j].content.lower(),
                    )
                else:
                    sim = 0.0
                
                matrix[i][j] = sim
                matrix[j][i] = sim  # Symmetric
        
        return ids, matrix
    
    def estimate_reduction(
        self,
        memories: list[MemoryItem],
        strategy: Optional[DeduplicationStrategy] = None,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """Estimate potential reduction from deduplication.
        
        Args:
            memories: Memories to analyze
            strategy: Strategy to use
            threshold: Similarity threshold
            
        Returns:
            Dictionary with reduction estimates
        """
        result = self.deduplicate(memories, strategy, threshold, dry_run=True)
        
        # Estimate bytes saved (assuming average content length)
        avg_content_size = (
            sum(len(m.content.encode()) for m in memories) / len(memories)
            if memories else 0
        )
        
        return {
            "original_count": result.original_count,
            "projected_count": result.deduplicated_count,
            "duplicates_found": result.duplicates_found,
            "reduction_percent": result.reduction_percent,
            "estimated_bytes_saved": int(result.duplicates_found * avg_content_size),
            "groups_found": len(result.duplicate_groups),
            "strategy": (strategy or self.config.strategy).value,
        }


# Convenience functions

def find_duplicates(
    memories: list[MemoryItem],
    strategy: DeduplicationStrategy = DeduplicationStrategy.FUZZY,
    threshold: float = 0.85,
) -> list[DuplicateGroup]:
    """Find duplicate memories.
    
    Convenience function for one-off deduplication.
    
    Args:
        memories: Memories to check
        strategy: Deduplication strategy
        threshold: Similarity threshold
        
    Returns:
        List of duplicate groups
    """
    deduplicator = SemanticDeduplicator()
    return deduplicator.find_duplicates(memories, strategy, threshold)


def deduplicate_memories(
    memories: list[MemoryItem],
    strategy: DeduplicationStrategy = DeduplicationStrategy.FUZZY,
    threshold: float = 0.85,
) -> DeduplicationResult:
    """Deduplicate memories.
    
    Convenience function for one-off deduplication.
    
    Args:
        memories: Memories to deduplicate
        strategy: Deduplication strategy
        threshold: Similarity threshold
        
    Returns:
        DeduplicationResult
    """
    deduplicator = SemanticDeduplicator()
    return deduplicator.deduplicate(memories, strategy, threshold)
