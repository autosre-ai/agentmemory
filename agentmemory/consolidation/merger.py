"""
Auto-merge related memories.

Provides intelligent merging of similar memories with:
- Content combination strategies
- Confidence aggregation
- Metadata preservation
- Source tracking
"""

import logging
from datetime import datetime
from typing import Any, Callable, Optional
import uuid

from .models import (
    MergeCandidate,
    MergeResult,
    SimilarityCluster,
    DeduplicationStrategy,
)
from .similarity import MemoryData, SimilarityDetector

logger = logging.getLogger(__name__)


class MemoryAutoMerger:
    """
    Automatically merge related memories.
    
    Merging strategies:
    - Concatenate: Join content from all memories
    - Summarize: Create a summary (requires LLM)
    - Keep best: Keep highest quality, reference others
    - Union facts: Extract and combine unique facts
    """
    
    def __init__(
        self,
        merge_threshold: float = 0.90,
        max_cluster_size: int = 5,
        summarize_fn: Optional[Callable[[list[str]], str]] = None,
        confidence_aggregation: str = "max",  # "max", "avg", "sum_capped"
    ):
        """
        Initialize auto-merger.
        
        Args:
            merge_threshold: Minimum similarity to merge
            max_cluster_size: Maximum memories to merge at once
            summarize_fn: Optional function to summarize multiple texts
            confidence_aggregation: How to aggregate confidence scores
        """
        self.merge_threshold = merge_threshold
        self.max_cluster_size = max_cluster_size
        self.summarize_fn = summarize_fn
        self.confidence_aggregation = confidence_aggregation
    
    def find_merge_candidates(
        self,
        memories: list[MemoryData],
        detector: SimilarityDetector,
    ) -> list[MergeCandidate]:
        """
        Find pairs of memories that could be merged.
        
        Args:
            memories: List of memories to analyze
            detector: SimilarityDetector instance
            
        Returns:
            List of MergeCandidate objects
        """
        candidates = []
        similar_pairs = detector.find_all_similar_pairs(
            memories, min_score=self.merge_threshold
        )
        
        for pair in similar_pairs:
            # Estimate merge quality
            estimated_quality = self._estimate_merge_quality(
                pair.memory1_id,
                pair.memory2_id,
                pair.score,
                memories,
            )
            
            # Determine merge strategy
            strategy = self._select_merge_strategy(pair.score)
            
            candidates.append(MergeCandidate(
                memory1_id=pair.memory1_id,
                memory2_id=pair.memory2_id,
                similarity_score=pair.score,
                merge_strategy=strategy,
                estimated_quality=estimated_quality,
            ))
        
        # Sort by estimated quality
        return sorted(candidates, key=lambda c: c.estimated_quality, reverse=True)
    
    def _estimate_merge_quality(
        self,
        memory1_id: str,
        memory2_id: str,
        similarity: float,
        memories: list[MemoryData],
    ) -> float:
        """Estimate the quality of a potential merge."""
        memory_map = {m.id: m for m in memories}
        m1 = memory_map.get(memory1_id)
        m2 = memory_map.get(memory2_id)
        
        if not m1 or not m2:
            return 0.0
        
        # Base quality on similarity
        quality = similarity
        
        # Boost if content lengths are similar
        len_ratio = min(len(m1.content), len(m2.content)) / max(len(m1.content), len(m2.content))
        quality *= (0.8 + 0.2 * len_ratio)
        
        return min(quality, 1.0)
    
    def _select_merge_strategy(self, similarity: float) -> str:
        """Select merge strategy based on similarity."""
        if similarity >= 0.98:
            return "keep_best"  # Nearly identical
        elif similarity >= 0.95:
            return "concatenate_unique"  # Some unique content
        else:
            return "union_facts"  # More diverse
    
    def merge_memories(
        self,
        memories: list[MemoryData],
        strategy: str = "auto",
        preserve_originals: bool = True,
    ) -> MergeResult:
        """
        Merge multiple memories into one.
        
        Args:
            memories: List of memories to merge
            strategy: Merge strategy ("auto", "concatenate", "keep_best", "summarize")
            preserve_originals: Whether to keep original memory references
            
        Returns:
            MergeResult with merged memory
        """
        if not memories:
            raise ValueError("No memories to merge")
        
        if len(memories) == 1:
            return MergeResult(
                source_memory_ids=[memories[0].id],
                merged_memory_id=memories[0].id,
                merged_content=memories[0].content,
                merged_confidence=self._get_confidence(memories[0]),
                merge_strategy="identity",
                quality_score=1.0,
            )
        
        # Limit cluster size
        if len(memories) > self.max_cluster_size:
            logger.warning(
                f"Cluster size {len(memories)} exceeds max {self.max_cluster_size}, "
                "truncating to best matches"
            )
            memories = memories[:self.max_cluster_size]
        
        # Auto-select strategy
        if strategy == "auto":
            strategy = self._auto_select_strategy(memories)
        
        # Execute merge
        if strategy == "keep_best":
            return self._merge_keep_best(memories)
        elif strategy == "concatenate" or strategy == "concatenate_unique":
            return self._merge_concatenate(memories)
        elif strategy == "summarize" and self.summarize_fn:
            return self._merge_summarize(memories)
        elif strategy == "union_facts":
            return self._merge_union_facts(memories)
        else:
            return self._merge_concatenate(memories)
    
    def _auto_select_strategy(self, memories: list[MemoryData]) -> str:
        """Auto-select best merge strategy for the memories."""
        # If summarize_fn is provided, prefer summarize strategy
        if self.summarize_fn:
            return "summarize"
            
        # Check content lengths
        lengths = [len(m.content) for m in memories]
        avg_length = sum(lengths) / len(lengths)
        
        # Check similarity of lengths
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        if length_variance < 100:  # Similar lengths
            # Check content similarity
            unique_words = set()
            for m in memories:
                unique_words.update(m.content.lower().split())
            
            total_words = sum(len(m.content.split()) for m in memories)
            uniqueness = len(unique_words) / max(total_words, 1)
            
            if uniqueness < 0.3:  # Mostly same words
                return "keep_best"
            elif uniqueness < 0.6:
                return "concatenate_unique"
            else:
                return "union_facts"
        else:
            return "union_facts"
    
    def _merge_keep_best(self, memories: list[MemoryData]) -> MergeResult:
        """Keep the best memory, reference others."""
        # Find best by content length and confidence
        best = max(
            memories,
            key=lambda m: (len(m.content), self._get_confidence(m))
        )
        
        source_ids = [m.id for m in memories]
        
        return MergeResult(
            source_memory_ids=source_ids,
            merged_memory_id=str(uuid.uuid4()),
            merged_content=best.content,
            merged_confidence=self._aggregate_confidence(memories),
            merge_strategy="keep_best",
            quality_score=1.0,
            metadata={
                "best_source_id": best.id,
                "alternative_count": len(memories) - 1,
            },
        )
    
    def _merge_concatenate(self, memories: list[MemoryData]) -> MergeResult:
        """Concatenate unique content from memories."""
        seen_content = set()
        unique_parts = []
        
        for memory in memories:
            # Normalize for comparison
            normalized = memory.content.strip().lower()
            
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique_parts.append(memory.content.strip())
        
        # Join with appropriate separator
        merged_content = " ".join(unique_parts)
        source_ids = [m.id for m in memories]
        
        return MergeResult(
            source_memory_ids=source_ids,
            merged_memory_id=str(uuid.uuid4()),
            merged_content=merged_content,
            merged_confidence=self._aggregate_confidence(memories),
            merge_strategy="concatenate",
            quality_score=0.9,
            metadata={
                "unique_parts": len(unique_parts),
                "total_sources": len(memories),
            },
        )
    
    def _merge_summarize(self, memories: list[MemoryData]) -> MergeResult:
        """Summarize content using provided function."""
        if not self.summarize_fn:
            return self._merge_concatenate(memories)
        
        contents = [m.content for m in memories]
        summarized = self.summarize_fn(contents)
        source_ids = [m.id for m in memories]
        
        return MergeResult(
            source_memory_ids=source_ids,
            merged_memory_id=str(uuid.uuid4()),
            merged_content=summarized,
            merged_confidence=self._aggregate_confidence(memories),
            merge_strategy="summarize",
            quality_score=0.85,
            metadata={
                "original_count": len(memories),
                "summarization_method": "llm",
            },
        )
    
    def _merge_union_facts(self, memories: list[MemoryData]) -> MergeResult:
        """Extract and combine unique facts from memories."""
        # Simple fact extraction: split by sentences
        all_facts = []
        seen_facts = set()
        
        for memory in memories:
            # Split into sentences
            sentences = self._split_sentences(memory.content)
            
            for sentence in sentences:
                normalized = sentence.strip().lower()
                if normalized and normalized not in seen_facts:
                    seen_facts.add(normalized)
                    all_facts.append(sentence.strip())
        
        merged_content = ". ".join(all_facts)
        if merged_content and not merged_content.endswith("."):
            merged_content += "."
        
        source_ids = [m.id for m in memories]
        
        return MergeResult(
            source_memory_ids=source_ids,
            merged_memory_id=str(uuid.uuid4()),
            merged_content=merged_content,
            merged_confidence=self._aggregate_confidence(memories),
            merge_strategy="union_facts",
            quality_score=0.8,
            metadata={
                "unique_facts": len(all_facts),
                "total_sources": len(memories),
            },
        )
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_confidence(self, memory: MemoryData) -> float:
        """Get confidence score from memory metadata."""
        if memory.metadata and "confidence" in memory.metadata:
            return memory.metadata["confidence"]
        return 1.0
    
    def _aggregate_confidence(self, memories: list[MemoryData]) -> float:
        """Aggregate confidence scores based on strategy."""
        confidences = [self._get_confidence(m) for m in memories]
        
        if self.confidence_aggregation == "max":
            return max(confidences)
        elif self.confidence_aggregation == "avg":
            return sum(confidences) / len(confidences)
        elif self.confidence_aggregation == "sum_capped":
            return min(1.0, sum(confidences) / len(confidences) + 0.1 * len(confidences))
        else:
            return max(confidences)
    
    def merge_cluster(
        self,
        cluster: "SimilarityCluster | list[str]",
        memories: list[MemoryData],
    ) -> MergeResult:
        """
        Merge all memories in a cluster.
        
        Args:
            cluster: SimilarityCluster to merge, or list of memory IDs
            memories: Full list of memories (to look up by ID)
            
        Returns:
            MergeResult
        """
        # Handle both SimilarityCluster and list of IDs
        if isinstance(cluster, list):
            memory_ids = cluster
        else:
            memory_ids = cluster.memory_ids
            
        memory_map = {m.id: m for m in memories}
        cluster_memories = [
            memory_map[mid]
            for mid in memory_ids
            if mid in memory_map
        ]
        
        if not cluster_memories:
            raise ValueError("No valid memories in cluster")
        
        return self.merge_memories(cluster_memories, strategy="auto")
    
    def merge_all_clusters(
        self,
        clusters: list[SimilarityCluster],
        memories: list[MemoryData],
    ) -> list[MergeResult]:
        """
        Merge all provided clusters.
        
        Args:
            clusters: List of clusters to merge
            memories: Full list of memories
            
        Returns:
            List of MergeResult objects
        """
        results = []
        
        for cluster in clusters:
            try:
                result = self.merge_cluster(cluster, memories)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to merge cluster {cluster.cluster_id}: {e}")
        
        return results
    
    def should_merge(
        self,
        memory1: MemoryData,
        memory2: MemoryData,
        similarity: float,
    ) -> bool:
        """
        Check if two memories should be merged based on similarity.
        
        Args:
            memory1: First memory
            memory2: Second memory
            similarity: Similarity score between memories
            
        Returns:
            True if memories should be merged
        """
        return similarity >= self.merge_threshold
    
    def _calculate_merged_confidence(
        self,
        memories: list[MemoryData],
    ) -> float:
        """
        Calculate merged confidence score.
        
        Alias for _aggregate_confidence.
        """
        return self._aggregate_confidence(memories)
