"""
Similarity detection between memories.

Implements multiple strategies for finding similar memories:
- Exact match
- Fuzzy text matching (Levenshtein distance)
- Semantic similarity (embedding-based)
- Hybrid approaches
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Optional
import uuid

from .models import (
    ConsolidationStrategy,
    SimilarityScore,
    SimilarityCluster,
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryData:
    """Minimal memory representation for similarity detection."""
    
    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: Optional[dict] = None


class SimilarityDetector:
    """
    Detect similar memories using configurable strategies.
    
    Supports:
    - Exact match: Hash-based duplicate detection
    - Fuzzy match: Text similarity using SequenceMatcher
    - Semantic match: Embedding cosine similarity
    - Hybrid: Combines fuzzy and semantic scores
    """
    
    def __init__(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID,
        similarity_threshold: float = 0.85,
        duplicate_threshold: float = 0.95,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        fuzzy_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ):
        """
        Initialize similarity detector.
        
        Args:
            strategy: Detection strategy to use
            similarity_threshold: Minimum score to consider memories similar
            duplicate_threshold: Minimum score to consider memories duplicates
            embedding_fn: Function to generate embeddings for semantic matching
            fuzzy_weight: Weight for fuzzy score in hybrid mode
            semantic_weight: Weight for semantic score in hybrid mode
        """
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold
        self.embedding_fn = embedding_fn
        self.fuzzy_weight = fuzzy_weight
        self.semantic_weight = semantic_weight
    
    def compute_similarity(
        self,
        memory1: MemoryData,
        memory2: MemoryData,
    ) -> SimilarityScore:
        """
        Compute similarity between two memories.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            SimilarityScore with computed score and details
        """
        details = {}
        
        if self.strategy == ConsolidationStrategy.EXACT_MATCH:
            score = self._exact_match(memory1.content, memory2.content)
            details["hash_match"] = score == 1.0
            
        elif self.strategy == ConsolidationStrategy.FUZZY_MATCH:
            score = self._fuzzy_match(memory1.content, memory2.content)
            details["fuzzy_score"] = score
            
        elif self.strategy == ConsolidationStrategy.SEMANTIC_MATCH:
            score = self._semantic_match(memory1, memory2)
            details["semantic_score"] = score
            
        else:  # HYBRID
            fuzzy_score = self._fuzzy_match(memory1.content, memory2.content)
            semantic_score = self._semantic_match(memory1, memory2)
            
            # Weighted combination
            score = (
                self.fuzzy_weight * fuzzy_score +
                self.semantic_weight * semantic_score
            )
            
            details["fuzzy_score"] = fuzzy_score
            details["semantic_score"] = semantic_score
            details["weights"] = {
                "fuzzy": self.fuzzy_weight,
                "semantic": self.semantic_weight,
            }
        
        return SimilarityScore(
            memory1_id=memory1.id,
            memory2_id=memory2.id,
            score=score,
            match_type=self.strategy,
            details=details,
        )
    
    def _exact_match(self, content1: str, content2: str) -> float:
        """Check for exact content match using hash."""
        hash1 = hashlib.sha256(content1.strip().lower().encode()).hexdigest()
        hash2 = hashlib.sha256(content2.strip().lower().encode()).hexdigest()
        return 1.0 if hash1 == hash2 else 0.0
    
    def _fuzzy_match(self, content1: str, content2: str) -> float:
        """Compute fuzzy text similarity using SequenceMatcher."""
        # Normalize content
        c1 = content1.strip().lower()
        c2 = content2.strip().lower()
        
        # Use SequenceMatcher for efficient similarity
        matcher = SequenceMatcher(None, c1, c2)
        return matcher.ratio()
    
    def _semantic_match(
        self,
        memory1: MemoryData,
        memory2: MemoryData,
    ) -> float:
        """Compute semantic similarity using embeddings."""
        # Get or compute embeddings
        emb1 = memory1.embedding
        emb2 = memory2.embedding
        
        if emb1 is None and self.embedding_fn:
            emb1 = self.embedding_fn(memory1.content)
        
        if emb2 is None and self.embedding_fn:
            emb2 = self.embedding_fn(memory2.content)
        
        if emb1 is None or emb2 is None:
            # Fall back to fuzzy if no embeddings
            logger.debug("No embeddings available, falling back to fuzzy match")
            return self._fuzzy_match(memory1.content, memory2.content)
        
        return self._cosine_similarity(emb1, emb2)
    
    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def find_similar(
        self,
        target: MemoryData,
        candidates: list[MemoryData],
        min_score: Optional[float] = None,
    ) -> list[SimilarityScore]:
        """
        Find memories similar to a target memory.
        
        Args:
            target: The target memory to match against
            candidates: List of candidate memories
            min_score: Minimum similarity score (defaults to threshold)
            
        Returns:
            List of SimilarityScore objects, sorted by score descending
        """
        min_score = min_score or self.similarity_threshold
        results = []
        
        for candidate in candidates:
            if candidate.id == target.id:
                continue
            
            score = self.compute_similarity(target, candidate)
            if score.score >= min_score:
                results.append(score)
        
        return sorted(results, key=lambda s: s.score, reverse=True)
    
    def find_all_similar_pairs(
        self,
        memories: list[MemoryData],
        min_score: Optional[float] = None,
    ) -> list[SimilarityScore]:
        """
        Find all pairs of similar memories.
        
        Args:
            memories: List of memories to compare
            min_score: Minimum similarity score
            
        Returns:
            List of SimilarityScore objects for all similar pairs
        """
        min_score = min_score or self.similarity_threshold
        results = []
        seen_pairs = set()
        
        for i, m1 in enumerate(memories):
            for m2 in memories[i + 1:]:
                pair_key = tuple(sorted([m1.id, m2.id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                score = self.compute_similarity(m1, m2)
                if score.score >= min_score:
                    results.append(score)
        
        return sorted(results, key=lambda s: s.score, reverse=True)
    
    def find_duplicates(
        self,
        memories: list[MemoryData],
    ) -> list[SimilarityScore]:
        """
        Find duplicate memories (score >= duplicate_threshold).
        
        Args:
            memories: List of memories to check
            
        Returns:
            List of SimilarityScore objects for duplicates
        """
        return self.find_all_similar_pairs(memories, self.duplicate_threshold)
    
    def cluster_similar(
        self,
        memories: list[MemoryData],
        min_cluster_score: Optional[float] = None,
    ) -> list[SimilarityCluster]:
        """
        Group similar memories into clusters.
        
        Uses single-linkage clustering - memories are in the same cluster
        if they're similar to any memory already in the cluster.
        
        Args:
            memories: List of memories to cluster
            min_cluster_score: Minimum score to group memories
            
        Returns:
            List of SimilarityCluster objects
        """
        min_score = min_cluster_score or self.similarity_threshold
        
        # Build adjacency list
        similar_pairs = self.find_all_similar_pairs(memories, min_score)
        
        adjacency: dict[str, set[str]] = defaultdict(set)
        pair_scores: dict[tuple, float] = {}
        
        for pair in similar_pairs:
            adjacency[pair.memory1_id].add(pair.memory2_id)
            adjacency[pair.memory2_id].add(pair.memory1_id)
            key = tuple(sorted([pair.memory1_id, pair.memory2_id]))
            pair_scores[key] = pair.score
        
        # Find connected components using DFS
        visited = set()
        clusters = []
        memory_map = {m.id: m for m in memories}
        
        for memory in memories:
            if memory.id in visited:
                continue
            if memory.id not in adjacency:
                continue  # No similar memories
            
            # DFS to find cluster
            cluster_ids = []
            stack = [memory.id]
            
            while stack:
                mid = stack.pop()
                if mid in visited:
                    continue
                visited.add(mid)
                cluster_ids.append(mid)
                
                for neighbor in adjacency[mid]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            if len(cluster_ids) < 2:
                continue
            
            # Compute cluster statistics
            scores = []
            for i, id1 in enumerate(cluster_ids):
                for id2 in cluster_ids[i + 1:]:
                    key = tuple(sorted([id1, id2]))
                    if key in pair_scores:
                        scores.append(pair_scores[key])
            
            if not scores:
                continue
            
            # Find centroid (memory with highest avg similarity to others)
            centroid_id = self._find_centroid(cluster_ids, pair_scores)
            
            cluster = SimilarityCluster(
                cluster_id=str(uuid.uuid4()),
                memory_ids=cluster_ids,
                centroid_id=centroid_id,
                avg_similarity=sum(scores) / len(scores),
                min_similarity=min(scores),
                max_similarity=max(scores),
            )
            clusters.append(cluster)
        
        return sorted(clusters, key=lambda c: c.avg_similarity, reverse=True)
    
    def _find_centroid(
        self,
        cluster_ids: list[str],
        pair_scores: dict[tuple, float],
    ) -> str:
        """Find the centroid (most representative) memory in a cluster."""
        best_id = cluster_ids[0]
        best_avg = 0.0
        
        for mid in cluster_ids:
            scores = []
            for other in cluster_ids:
                if other == mid:
                    continue
                key = tuple(sorted([mid, other]))
                if key in pair_scores:
                    scores.append(pair_scores[key])
            
            if scores:
                avg = sum(scores) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_id = mid
        
        return best_id
    
    def quick_hash_groups(
        self,
        memories: list[MemoryData],
    ) -> dict[str, list[str]]:
        """
        Quickly group memories by content hash for exact duplicate detection.
        
        Args:
            memories: List of memories
            
        Returns:
            Dict mapping content hash to list of memory IDs
        """
        groups: dict[str, list[str]] = defaultdict(list)
        
        for memory in memories:
            content_hash = hashlib.sha256(
                memory.content.strip().lower().encode()
            ).hexdigest()
            groups[content_hash].append(memory.id)
        
        # Filter to only groups with duplicates
        return {
            h: ids for h, ids in groups.items()
            if len(ids) > 1
        }
    
    # Alias methods for compatibility
    def calculate_similarity(
        self,
        memory1: MemoryData,
        memory2: MemoryData,
    ) -> SimilarityScore:
        """Alias for compute_similarity."""
        return self.compute_similarity(memory1, memory2)
    
    def is_duplicate(
        self,
        memory1: MemoryData,
        memory2: MemoryData,
    ) -> bool:
        """Check if two memories are duplicates."""
        score = self.compute_similarity(memory1, memory2)
        return score.score >= self.duplicate_threshold
    
    def _content_hash(self, content: str) -> str:
        """Compute normalized content hash."""
        return hashlib.sha256(
            content.strip().lower().encode()
        ).hexdigest()
