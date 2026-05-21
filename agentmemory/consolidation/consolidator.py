"""
Main Memory Consolidation Engine.

Coordinates similarity detection, deduplication, conflict detection,
and auto-merging to consolidate memory stores.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional
import uuid

from .models import (
    ConsolidationConfig,
    ConsolidationResult,
    ConsolidationStrategy,
    DeduplicationStrategy,
)
from .similarity import MemoryData, SimilarityDetector
from .deduplication import Deduplicator, DeduplicationResult
from .merger import MemoryAutoMerger
from .conflict_detector import ConflictDetector, MemoryConflict

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """
    Main consolidation engine for memory stores.
    
    Provides:
    - Similarity detection between memories
    - Deduplication with configurable strategies
    - Conflict detection and resolution
    - Auto-merging of related memories
    - Batch processing with progress tracking
    """
    
    def __init__(
        self,
        config: Optional[ConsolidationConfig] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        summarize_fn: Optional[Callable[[list[str]], str]] = None,
    ):
        """
        Initialize consolidator.
        
        Args:
            config: Consolidation configuration
            embedding_fn: Function to generate embeddings
            summarize_fn: Function to summarize multiple texts
        """
        self.config = config or ConsolidationConfig()
        self.embedding_fn = embedding_fn
        
        # Initialize components
        self.similarity_detector = SimilarityDetector(
            strategy=self.config.strategy,
            similarity_threshold=self.config.similarity_threshold,
            duplicate_threshold=self.config.duplicate_threshold,
            embedding_fn=embedding_fn,
        )
        
        self.deduplicator = Deduplicator(
            strategy=self.config.dedup_strategy,
            detector=self.similarity_detector,
            duplicate_threshold=self.config.duplicate_threshold,
        )
        
        self.merger = MemoryAutoMerger(
            merge_threshold=self.config.merge_threshold,
            max_cluster_size=self.config.max_merge_cluster_size,
            summarize_fn=summarize_fn,
        )
        
        self.conflict_detector = ConflictDetector(
            similarity_detector=self.similarity_detector,
        )
    
    def consolidate(
        self,
        memories: list[MemoryData],
        dry_run: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> ConsolidationResult:
        """
        Run full consolidation on memory set.
        
        Args:
            memories: List of memories to consolidate
            dry_run: If True, analyze but don't modify
            progress_callback: Called with (stage, current, total)
            
        Returns:
            ConsolidationResult with statistics and changes
        """
        run_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        errors = []
        
        logger.info(f"Starting consolidation run {run_id} with {len(memories)} memories")
        
        results = {
            "duplicates_found": 0,
            "duplicates_removed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "merges_performed": 0,
            "clusters_found": 0,
        }
        
        try:
            # Phase 1: Deduplication
            if progress_callback:
                progress_callback("deduplication", 0, 4)
            
            dedup_result = self.deduplicator.deduplicate(memories)
            results["duplicates_found"] = len(dedup_result.removed_memory_ids)
            
            if not dry_run:
                results["duplicates_removed"] = len(dedup_result.removed_memory_ids)
            
            logger.info(f"Found {results['duplicates_found']} duplicates")
            
            # Phase 2: Conflict Detection
            if progress_callback:
                progress_callback("conflict_detection", 1, 4)
            
            if self.config.detect_conflicts:
                conflicts = self.conflict_detector.detect_all_conflicts(memories)
                results["conflicts_detected"] = len(conflicts)
                
                if self.config.auto_resolve_conflicts and not dry_run:
                    resolved = self._resolve_conflicts(conflicts, memories)
                    results["conflicts_resolved"] = resolved
                
                logger.info(f"Found {results['conflicts_detected']} conflicts")
            
            # Phase 3: Similarity Clustering
            if progress_callback:
                progress_callback("clustering", 2, 4)
            
            # Get remaining memories after dedup
            remaining_ids = set(dedup_result.kept_memory_ids)
            remaining = [m for m in memories if m.id in remaining_ids]
            
            clusters = self.similarity_detector.cluster_similar(remaining)
            results["clusters_found"] = len(clusters)
            
            logger.info(f"Found {results['clusters_found']} similarity clusters")
            
            # Phase 4: Auto-merge
            if progress_callback:
                progress_callback("merging", 3, 4)
            
            if self.config.auto_merge and not dry_run:
                merge_results = self.merger.merge_all_clusters(clusters, remaining)
                results["merges_performed"] = len(merge_results)
                
                logger.info(f"Performed {results['merges_performed']} merges")
            
            if progress_callback:
                progress_callback("complete", 4, 4)
            
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Consolidation error: {e}")
        
        completed_at = datetime.utcnow()
        processing_time = (completed_at - started_at).total_seconds()
        
        return ConsolidationResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            memories_analyzed=len(memories),
            duplicates_found=results["duplicates_found"],
            duplicates_removed=results["duplicates_removed"],
            conflicts_detected=results["conflicts_detected"],
            conflicts_resolved=results["conflicts_resolved"],
            merges_performed=results["merges_performed"],
            clusters_found=results["clusters_found"],
            processing_time_seconds=processing_time,
            errors=errors,
        )
    
    def _resolve_conflicts(
        self,
        conflicts: list[MemoryConflict],
        memories: list[MemoryData],
    ) -> int:
        """Resolve conflicts based on strategy."""
        resolved = 0
        strategy = self.config.conflict_resolution_strategy
        memory_map = {m.id: m for m in memories}
        
        for conflict in conflicts:
            if conflict.resolved:
                continue
            
            m1 = memory_map.get(conflict.memory1_id)
            m2 = memory_map.get(conflict.memory2_id)
            
            if not m1 or not m2:
                continue
            
            winner = None
            
            if strategy == "confidence_wins":
                conf1 = self._get_confidence(m1)
                conf2 = self._get_confidence(m2)
                winner = m1 if conf1 >= conf2 else m2
                
            elif strategy == "latest_wins":
                ts1 = self._get_timestamp(m1)
                ts2 = self._get_timestamp(m2)
                winner = m1 if ts1 >= ts2 else m2
                
            elif strategy == "merge":
                # Let merger handle it
                pass
            
            if winner:
                conflict.resolved = True
                conflict.resolution_memory_id = winner.id
                conflict.resolution = f"Resolved using {strategy}"
                resolved += 1
        
        return resolved
    
    def _get_confidence(self, memory: MemoryData) -> float:
        """Get confidence from memory."""
        if memory.metadata and "confidence" in memory.metadata:
            return memory.metadata["confidence"]
        return 1.0
    
    def _get_timestamp(self, memory: MemoryData) -> datetime:
        """Get timestamp from memory."""
        if memory.metadata and "updated_at" in memory.metadata:
            ts = memory.metadata["updated_at"]
            if isinstance(ts, str):
                return datetime.fromisoformat(ts)
            return ts
        return datetime.min
    
    def analyze(
        self,
        memories: list[MemoryData],
    ) -> dict[str, Any]:
        """
        Analyze memory set without making changes.
        
        Args:
            memories: Memories to analyze
            
        Returns:
            Analysis report
        """
        result = self.consolidate(memories, dry_run=True)
        
        # Add detailed analysis
        dedup_estimate = self.deduplicator.estimate_duplicates(memories)
        
        return {
            "summary": result.to_dict(),
            "deduplication_estimate": dedup_estimate,
            "recommendations": self._generate_recommendations(result, dedup_estimate),
        }
    
    def _generate_recommendations(
        self,
        result: ConsolidationResult,
        dedup_estimate: dict[str, Any],
    ) -> list[str]:
        """Generate consolidation recommendations."""
        recommendations = []
        
        if dedup_estimate["estimated_total_duplicates"] > 0:
            recommendations.append(
                f"Found {dedup_estimate['estimated_total_duplicates']} potential duplicates. "
                f"Run consolidation to remove them."
            )
        
        if result.conflicts_detected > 0:
            recommendations.append(
                f"Detected {result.conflicts_detected} conflicts. "
                f"Review and resolve to maintain data consistency."
            )
        
        if result.clusters_found > 0:
            recommendations.append(
                f"Found {result.clusters_found} similarity clusters. "
                f"Consider merging related memories."
            )
        
        if not recommendations:
            recommendations.append("Memory store is well-organized. No action needed.")
        
        return recommendations
    
    def consolidate_incremental(
        self,
        new_memory: MemoryData,
        existing_memories: list[MemoryData],
    ) -> dict[str, Any]:
        """
        Consolidate a single new memory against existing ones.
        
        Args:
            new_memory: The new memory to consolidate
            existing_memories: Existing memories to check against
            
        Returns:
            Dict with consolidation actions taken
        """
        actions = {
            "is_duplicate": False,
            "duplicate_of": None,
            "has_conflicts": False,
            "conflicts": [],
            "similar_memories": [],
            "recommended_action": "add",
        }
        
        # Check for duplicates
        duplicates = self.similarity_detector.find_similar(
            new_memory,
            existing_memories,
            min_score=self.config.duplicate_threshold,
        )
        
        if duplicates:
            actions["is_duplicate"] = True
            actions["duplicate_of"] = duplicates[0].memory2_id
            actions["recommended_action"] = "skip"
            return actions
        
        # Check for conflicts
        conflicts = self.conflict_detector.detect_conflicts_for_memory(
            new_memory, existing_memories
        )
        
        if conflicts:
            actions["has_conflicts"] = True
            actions["conflicts"] = [c.to_dict() for c in conflicts]
            actions["recommended_action"] = "review"
        
        # Find similar memories
        similar = self.similarity_detector.find_similar(
            new_memory,
            existing_memories,
            min_score=self.config.similarity_threshold,
        )
        
        if similar:
            actions["similar_memories"] = [s.to_dict() for s in similar[:5]]
            
            if not actions["has_conflicts"]:
                actions["recommended_action"] = "add_and_link"
        
        return actions
    
    def batch_process(
        self,
        memories: list[MemoryData],
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[ConsolidationResult]:
        """
        Process memories in batches.
        
        Args:
            memories: All memories to process
            batch_size: Size of each batch
            progress_callback: Called with (batch_num, total_batches)
            
        Returns:
            List of ConsolidationResult for each batch
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        total_batches = (len(memories) + batch_size - 1) // batch_size
        
        for i in range(0, len(memories), batch_size):
            batch_num = i // batch_size + 1
            batch = memories[i:i + batch_size]
            
            if progress_callback:
                progress_callback(batch_num, total_batches)
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            result = self.consolidate(batch)
            results.append(result)
        
        return results
