"""Temporal memory system for Agent Memory Toolkit.

This module provides cognitively-inspired temporal memory capabilities including:
- Timeline-based memory organization with chronological retrieval
- Episodic memory with decay and forgetting curves
- Working memory with limited capacity and attention mechanisms
- Memory consolidation from short-term to long-term storage

Quick Start:
    >>> from agent_memory_toolkit.temporal import (
    ...     Timeline,
    ...     TimeWindow,
    ...     TemporalMemory,
    ...     EpisodicMemoryStore,
    ...     WorkingMemory,
    ...     MemoryConsolidator,
    ... )
    >>> 
    >>> # Timeline-based organization
    >>> timeline = Timeline()
    >>> timeline.add(TemporalMemory(
    ...     memory_id="1",
    ...     content="User asked about Python",
    ...     timestamp=datetime.now(),
    ... ))
    >>> recent = timeline.query(time_window=TimeWindow.LAST_HOUR)
    >>> 
    >>> # Episodic memory with decay
    >>> episodes = EpisodicMemoryStore()
    >>> item = episodes.add_item("Important fact", importance=0.9)
    >>> episodes.apply_decay()  # Apply forgetting curve
    >>> 
    >>> # Working memory with capacity limits
    >>> wm = WorkingMemory()
    >>> wm.add("Task: Book flight", priority=0.8)
    >>> wm.focus(item.item_id)  # Focus attention
    >>> 
    >>> # Consolidate to long-term memory
    >>> consolidator = MemoryConsolidator(working_memory=wm)
    >>> result = consolidator.consolidate()
"""

from datetime import datetime, timedelta

# Timeline module
from .timeline import (
    Timeline,
    TimelineConfig,
    TimelineQuery,
    TimelineResult,
    TemporalMemory,
    TemporalCluster,
    TemporalIndex,
    TimeWindow,
    RecencyDecay,
    RecencyScorer,
    create_timeline,
)

# Episodes module
from .episodes import (
    EpisodicMemoryStore,
    EpisodicMemoryConfig,
    EpisodicMemoryItem,
    Episode,
    EpisodeType,
    DecayModel,
    DecayCalculator,
    EmotionalValence,
    create_episodic_store,
)

# Working memory module
from .working import (
    WorkingMemory,
    WorkingMemoryConfig,
    WorkingMemoryItem,
    WorkingMemoryContext,
    DisplacementStrategy,
    DisplacementResult,
    AttentionMode,
    AttentionAllocator,
    create_working_memory,
)

# Consolidation module
from .consolidation import (
    MemoryConsolidator,
    ConsolidationConfig,
    ConsolidationResult,
    ConsolidationTrigger,
    ConsolidationStrategy,
    ConsolidationCandidate,
    ConsolidatedMemory,
    Schema,
    LongTermStore,
    InMemoryLongTermStore,
    ConsolidationScorer,
    SemanticGrouper,
    create_consolidator,
)

__all__ = [
    # Timeline
    "Timeline",
    "TimelineConfig",
    "TimelineQuery",
    "TimelineResult",
    "TemporalMemory",
    "TemporalCluster",
    "TemporalIndex",
    "TimeWindow",
    "RecencyDecay",
    "RecencyScorer",
    "create_timeline",
    
    # Episodes
    "EpisodicMemoryStore",
    "EpisodicMemoryConfig",
    "EpisodicMemoryItem",
    "Episode",
    "EpisodeType",
    "DecayModel",
    "DecayCalculator",
    "EmotionalValence",
    "create_episodic_store",
    
    # Working memory
    "WorkingMemory",
    "WorkingMemoryConfig",
    "WorkingMemoryItem",
    "WorkingMemoryContext",
    "DisplacementStrategy",
    "DisplacementResult",
    "AttentionMode",
    "AttentionAllocator",
    "create_working_memory",
    
    # Consolidation
    "MemoryConsolidator",
    "ConsolidationConfig",
    "ConsolidationResult",
    "ConsolidationTrigger",
    "ConsolidationStrategy",
    "ConsolidationCandidate",
    "ConsolidatedMemory",
    "Schema",
    "LongTermStore",
    "InMemoryLongTermStore",
    "ConsolidationScorer",
    "SemanticGrouper",
    "create_consolidator",
]
