"""
Data models for memory consolidation.

Defines types for similarity scores, merge operations, conflicts,
and consolidation results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ConsolidationStrategy(Enum):
    """Strategies for consolidating memories."""
    
    EXACT_MATCH = "exact_match"           # Identical content
    FUZZY_MATCH = "fuzzy_match"           # Similar text (Levenshtein)
    SEMANTIC_MATCH = "semantic_match"     # Embedding-based similarity
    HYBRID = "hybrid"                     # Combine fuzzy + semantic


class DeduplicationStrategy(Enum):
    """Strategies for handling duplicate memories."""
    
    KEEP_NEWEST = "keep_newest"           # Keep most recent
    KEEP_OLDEST = "keep_oldest"           # Keep oldest (original)
    KEEP_HIGHEST_CONFIDENCE = "keep_highest_confidence"  # Keep most confident
    MERGE_ALL = "merge_all"               # Merge into single memory
    KEEP_MOST_ACCESSED = "keep_most_accessed"  # Keep most used


class ConflictType(Enum):
    """Types of conflicts between memories."""
    
    VALUE_CONFLICT = "value_conflict"           # Same key, different values
    TEMPORAL_CONFLICT = "temporal_conflict"     # Conflicting time references
    SOURCE_CONFLICT = "source_conflict"         # Conflicting source information
    CONFIDENCE_GAP = "confidence_gap"           # Large confidence difference
    SEMANTIC_CONTRADICTION = "semantic_contradiction"  # Contradictory meaning


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    
    LOW = "low"           # Minor inconsistency
    MEDIUM = "medium"     # Needs attention
    HIGH = "high"         # Significant contradiction
    CRITICAL = "critical"  # Must be resolved


@dataclass
class SimilarityScore:
    """Represents similarity between two memories."""
    
    memory1_id: str
    memory2_id: str
    score: float  # 0.0 to 1.0
    match_type: ConsolidationStrategy
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_similar(self) -> bool:
        """Check if memories are similar based on score."""
        return self.score >= 0.8
    
    @property
    def is_duplicate(self) -> bool:
        """Check if memories are near-duplicates."""
        return self.score >= 0.95
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "memory1_id": self.memory1_id,
            "memory2_id": self.memory2_id,
            "score": self.score,
            "match_type": self.match_type.value,
            "is_similar": self.is_similar,
            "is_duplicate": self.is_duplicate,
            "details": self.details,
        }


@dataclass
class SimilarityCluster:
    """A group of similar memories."""
    
    cluster_id: str
    memory_ids: list[str]
    centroid_id: str  # The "representative" memory
    avg_similarity: float
    min_similarity: float
    max_similarity: float
    
    def __len__(self) -> int:
        return len(self.memory_ids)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "memory_ids": self.memory_ids,
            "centroid_id": self.centroid_id,
            "size": len(self),
            "avg_similarity": self.avg_similarity,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
        }


@dataclass
class MemoryConflict:
    """Represents a conflict between memories."""
    
    conflict_id: str
    memory1_id: str
    memory2_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution: Optional[str] = None
    resolution_memory_id: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "memory1_id": self.memory1_id,
            "memory2_id": self.memory2_id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolution_memory_id": self.resolution_memory_id,
        }


@dataclass
class MergeCandidate:
    """Candidate pair of memories for merging."""
    
    memory1_id: str
    memory2_id: str
    similarity_score: float
    merge_strategy: str
    estimated_quality: float  # Expected quality of merged memory
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "memory1_id": self.memory1_id,
            "memory2_id": self.memory2_id,
            "similarity_score": self.similarity_score,
            "merge_strategy": self.merge_strategy,
            "estimated_quality": self.estimated_quality,
        }


@dataclass
class MergeResult:
    """Result of a memory merge operation."""
    
    source_memory_ids: list[str]
    merged_memory_id: str
    merged_content: str
    merged_confidence: float
    merge_strategy: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_memory_ids": self.source_memory_ids,
            "merged_memory_id": self.merged_memory_id,
            "merged_content": self.merged_content,
            "merged_confidence": self.merged_confidence,
            "merge_strategy": self.merge_strategy,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


@dataclass
class ConsolidationResult:
    """Result of a full consolidation run."""
    
    run_id: str
    started_at: datetime
    completed_at: datetime
    memories_analyzed: int
    duplicates_found: int
    duplicates_removed: int
    conflicts_detected: int
    conflicts_resolved: int
    merges_performed: int
    clusters_found: int
    processing_time_seconds: float
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    @property
    def summary(self) -> str:
        return (
            f"Consolidation completed: analyzed {self.memories_analyzed} memories, "
            f"removed {self.duplicates_removed} duplicates, "
            f"resolved {self.conflicts_resolved}/{self.conflicts_detected} conflicts, "
            f"performed {self.merges_performed} merges"
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "memories_analyzed": self.memories_analyzed,
            "duplicates_found": self.duplicates_found,
            "duplicates_removed": self.duplicates_removed,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "merges_performed": self.merges_performed,
            "clusters_found": self.clusters_found,
            "processing_time_seconds": self.processing_time_seconds,
            "success": self.success,
            "summary": self.summary,
            "errors": self.errors,
            "details": self.details,
        }


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""
    
    # Similarity detection
    strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID
    similarity_threshold: float = 0.85
    duplicate_threshold: float = 0.95
    
    # Deduplication
    dedup_strategy: DeduplicationStrategy = DeduplicationStrategy.KEEP_HIGHEST_CONFIDENCE
    
    # Conflict detection
    detect_conflicts: bool = True
    auto_resolve_conflicts: bool = False
    conflict_resolution_strategy: str = "confidence_wins"
    
    # Auto-merge
    auto_merge: bool = False
    merge_threshold: float = 0.90
    max_merge_cluster_size: int = 5
    
    # Background process
    batch_size: int = 100
    max_workers: int = 4
    
    # Scheduling
    run_interval_hours: int = 24
    max_run_time_seconds: int = 3600
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "similarity_threshold": self.similarity_threshold,
            "duplicate_threshold": self.duplicate_threshold,
            "dedup_strategy": self.dedup_strategy.value,
            "detect_conflicts": self.detect_conflicts,
            "auto_resolve_conflicts": self.auto_resolve_conflicts,
            "conflict_resolution_strategy": self.conflict_resolution_strategy,
            "auto_merge": self.auto_merge,
            "merge_threshold": self.merge_threshold,
            "max_merge_cluster_size": self.max_merge_cluster_size,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "run_interval_hours": self.run_interval_hours,
            "max_run_time_seconds": self.max_run_time_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsolidationConfig":
        return cls(
            strategy=ConsolidationStrategy(data.get("strategy", "hybrid")),
            similarity_threshold=data.get("similarity_threshold", 0.85),
            duplicate_threshold=data.get("duplicate_threshold", 0.95),
            dedup_strategy=DeduplicationStrategy(
                data.get("dedup_strategy", "keep_highest_confidence")
            ),
            detect_conflicts=data.get("detect_conflicts", True),
            auto_resolve_conflicts=data.get("auto_resolve_conflicts", False),
            conflict_resolution_strategy=data.get(
                "conflict_resolution_strategy", "confidence_wins"
            ),
            auto_merge=data.get("auto_merge", False),
            merge_threshold=data.get("merge_threshold", 0.90),
            max_merge_cluster_size=data.get("max_merge_cluster_size", 5),
            batch_size=data.get("batch_size", 100),
            max_workers=data.get("max_workers", 4),
            run_interval_hours=data.get("run_interval_hours", 24),
            max_run_time_seconds=data.get("max_run_time_seconds", 3600),
        )
