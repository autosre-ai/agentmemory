"""
Memory Consolidation Module

Provides tools for consolidating AI agent memories through:
- Similarity detection between memories
- Deduplication with multiple strategies
- Conflict detection and resolution
- Auto-merging of related memories
- Background consolidation process
- CLI command 'amt consolidate'

Usage:
    from agentmemory.consolidation import MemoryConsolidator, ConsolidationConfig
    
    config = ConsolidationConfig(
        similarity_threshold=0.85,
        auto_merge=True,
    )
    
    consolidator = MemoryConsolidator(config)
    result = consolidator.consolidate(memories)
"""

from .models import (
    ConsolidationStrategy,
    DeduplicationStrategy,
    ConflictType,
    ConflictSeverity,
    SimilarityScore,
    SimilarityCluster,
    MemoryConflict,
    MergeCandidate,
    MergeResult,
    ConsolidationResult,
    ConsolidationConfig,
)

from .similarity import (
    MemoryData,
    SimilarityDetector,
)

from .deduplication import (
    Deduplicator,
    DeduplicationResult,
)

from .merger import (
    MemoryAutoMerger,
)

from .conflict_detector import (
    ConflictDetector,
)

from .consolidator import (
    MemoryConsolidator,
)

from .background import (
    ConsolidationScheduler,
    ConsolidationDaemon,
)

__all__ = [
    # Models
    "ConsolidationStrategy",
    "DeduplicationStrategy",
    "ConflictType",
    "ConflictSeverity",
    "SimilarityScore",
    "SimilarityCluster",
    "MemoryConflict",
    "MergeCandidate",
    "MergeResult",
    "ConsolidationResult",
    "ConsolidationConfig",
    # Similarity
    "MemoryData",
    "SimilarityDetector",
    # Deduplication
    "Deduplicator",
    "DeduplicationResult",
    # Merger
    "MemoryAutoMerger",
    # Conflict Detection
    "ConflictDetector",
    # Main Consolidator
    "MemoryConsolidator",
    # Background Process
    "ConsolidationScheduler",
    "ConsolidationDaemon",
]
