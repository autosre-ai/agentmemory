"""Memory consolidation from short-term to long-term storage.

This module implements cognitively-inspired memory consolidation:
- Transfer from working memory to long-term storage
- Semantic organization during consolidation
- Sleep-like consolidation cycles
- Memory integration and schema formation
- Importance-based selective consolidation
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable
from collections import defaultdict

from .working import WorkingMemory, WorkingMemoryItem, DisplacementResult
from .episodes import EpisodicMemoryStore, EpisodicMemoryItem, Episode

logger = logging.getLogger(__name__)


class ConsolidationTrigger(Enum):
    """What triggers consolidation."""
    TIME_BASED = "time_based"       # Regular intervals
    CAPACITY = "capacity"           # Working memory full
    EPISODE_END = "episode_end"     # Episode boundary
    EXPLICIT = "explicit"           # Explicitly triggered
    SLEEP_CYCLE = "sleep_cycle"     # Sleep-like consolidation


class ConsolidationStrategy(Enum):
    """Strategy for selecting what to consolidate."""
    ALL = "all"                     # Consolidate everything
    IMPORTANT = "important"         # Only important items
    FREQUENT = "frequent"           # Frequently accessed
    EMOTIONAL = "emotional"         # Emotionally salient
    COMBINED = "combined"           # Weighted combination


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""
    
    # Triggers
    consolidation_interval: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    capacity_trigger_threshold: float = 0.8  # Trigger when WM at 80% capacity
    
    # Selection
    strategy: ConsolidationStrategy = ConsolidationStrategy.COMBINED
    min_importance: float = 0.3
    min_activation: float = 0.2
    min_access_count: int = 1
    
    # Weights for combined strategy
    importance_weight: float = 0.4
    activation_weight: float = 0.3
    access_weight: float = 0.2
    emotional_weight: float = 0.1
    
    # Processing
    batch_size: int = 10
    enable_semantic_grouping: bool = True
    enable_schema_integration: bool = True
    
    # Sleep-like consolidation
    enable_sleep_consolidation: bool = True
    sleep_consolidation_boost: float = 0.3
    
    # Callbacks
    transform_on_consolidation: bool = True


@dataclass
class ConsolidationCandidate:
    """A candidate item for consolidation."""
    item: WorkingMemoryItem | EpisodicMemoryItem
    score: float
    source: str  # 'working_memory', 'episode', etc.
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Grouping
    semantic_cluster: str | None = None
    related_candidates: list[str] = field(default_factory=list)


@dataclass
class ConsolidatedMemory:
    """A memory that has been consolidated to long-term storage."""
    memory_id: str
    content: str
    original_content: str
    timestamp: datetime
    consolidated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Consolidation info
    source_ids: list[str] = field(default_factory=list)
    consolidation_score: float = 0.0
    is_integrated: bool = False
    schema_id: str | None = None
    
    # Memory properties
    strength: float = 1.0
    importance: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "original_content": self.original_content,
            "timestamp": self.timestamp.isoformat(),
            "consolidated_at": self.consolidated_at.isoformat(),
            "metadata": self.metadata,
            "source_ids": self.source_ids,
            "consolidation_score": self.consolidation_score,
            "is_integrated": self.is_integrated,
            "schema_id": self.schema_id,
            "strength": self.strength,
            "importance": self.importance,
        }


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    success: bool
    consolidated_count: int
    skipped_count: int
    memories: list[ConsolidatedMemory]
    trigger: ConsolidationTrigger
    timestamp: datetime = field(default_factory=datetime.now)
    duration: timedelta | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class Schema:
    """A schema represents organized knowledge structure."""
    schema_id: str
    name: str
    description: str | None = None
    created: datetime = field(default_factory=datetime.now)
    
    # Schema structure
    concepts: list[str] = field(default_factory=list)
    relationships: dict[str, list[str]] = field(default_factory=dict)
    
    # Linked memories
    memory_ids: list[str] = field(default_factory=list)
    
    # Statistics
    activation_count: int = 0
    last_activated: datetime | None = None


@runtime_checkable
class LongTermStore(Protocol):
    """Protocol for long-term memory storage backends."""
    
    def store(self, memory: ConsolidatedMemory) -> str:
        """Store a consolidated memory."""
        ...
    
    def retrieve(self, memory_id: str) -> ConsolidatedMemory | None:
        """Retrieve a memory by ID."""
        ...
    
    def search(self, query: str, limit: int = 10) -> list[ConsolidatedMemory]:
        """Search for memories."""
        ...


class InMemoryLongTermStore:
    """Simple in-memory implementation of long-term storage."""
    
    def __init__(self):
        self._memories: dict[str, ConsolidatedMemory] = {}
        self._schemas: dict[str, Schema] = {}
    
    def store(self, memory: ConsolidatedMemory) -> str:
        """Store a consolidated memory."""
        self._memories[memory.memory_id] = memory
        return memory.memory_id
    
    def retrieve(self, memory_id: str) -> ConsolidatedMemory | None:
        """Retrieve a memory by ID."""
        return self._memories.get(memory_id)
    
    def search(self, query: str, limit: int = 10) -> list[ConsolidatedMemory]:
        """Simple substring search."""
        query_lower = query.lower()
        matches = [
            m for m in self._memories.values()
            if query_lower in m.content.lower()
        ]
        return matches[:limit]
    
    def store_schema(self, schema: Schema) -> str:
        """Store a schema."""
        self._schemas[schema.schema_id] = schema
        return schema.schema_id
    
    def get_schema(self, schema_id: str) -> Schema | None:
        """Get a schema by ID."""
        return self._schemas.get(schema_id)
    
    def list_memories(self, limit: int = 100) -> list[ConsolidatedMemory]:
        """List all memories."""
        memories = sorted(
            self._memories.values(),
            key=lambda m: m.consolidated_at,
            reverse=True,
        )
        return memories[:limit]
    
    def __len__(self) -> int:
        return len(self._memories)


class ConsolidationScorer:
    """Scores items for consolidation priority."""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
    
    def score(self, item: WorkingMemoryItem | EpisodicMemoryItem) -> float:
        """
        Calculate consolidation score for an item.
        
        Higher scores = higher priority for consolidation.
        
        Args:
            item: The item to score
            
        Returns:
            Score between 0 and 1
        """
        strategy = self.config.strategy
        
        if strategy == ConsolidationStrategy.ALL:
            return 1.0
        
        # Get item properties
        importance = getattr(item, 'importance', 0.5)
        activation = getattr(item, 'activation', getattr(item, 'strength', 0.5))
        access_count = getattr(item, 'access_count', 0)
        emotional_valence = getattr(item, 'emotional_valence', None)
        
        if strategy == ConsolidationStrategy.IMPORTANT:
            return importance
        
        elif strategy == ConsolidationStrategy.FREQUENT:
            # Normalize access count
            return min(1.0, access_count / 10)
        
        elif strategy == ConsolidationStrategy.EMOTIONAL:
            if emotional_valence:
                return abs(emotional_valence.value) / 2
            return 0.5
        
        else:  # COMBINED
            config = self.config
            
            # Importance component
            imp_score = importance * config.importance_weight
            
            # Activation component
            act_score = activation * config.activation_weight
            
            # Access frequency component
            acc_score = min(1.0, access_count / 10) * config.access_weight
            
            # Emotional component
            if emotional_valence:
                emo_score = (abs(emotional_valence.value) / 2) * config.emotional_weight
            else:
                emo_score = 0.5 * config.emotional_weight
            
            return imp_score + act_score + acc_score + emo_score
    
    def should_consolidate(self, item: WorkingMemoryItem | EpisodicMemoryItem) -> bool:
        """Check if item meets consolidation criteria."""
        importance = getattr(item, 'importance', 0.5)
        activation = getattr(item, 'activation', getattr(item, 'strength', 0.5))
        access_count = getattr(item, 'access_count', 0)
        
        return (
            importance >= self.config.min_importance or
            activation >= self.config.min_activation or
            access_count >= self.config.min_access_count
        )


class SemanticGrouper:
    """Groups related memories for consolidated storage."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
    
    def group(
        self,
        candidates: list[ConsolidationCandidate],
    ) -> dict[str, list[ConsolidationCandidate]]:
        """
        Group candidates by semantic similarity.
        
        Args:
            candidates: Candidates to group
            
        Returns:
            Dict mapping cluster_id to candidates
        """
        if not candidates:
            return {}
        
        # Simple grouping by shared context tags or content overlap
        # In production, use embeddings for semantic grouping
        groups: dict[str, list[ConsolidationCandidate]] = defaultdict(list)
        
        for candidate in candidates:
            item = candidate.item
            
            # Get context tags if available
            tags = getattr(item, 'context_tags', [])
            
            if tags:
                # Use first tag as cluster key
                cluster_key = tags[0]
            else:
                # Use content-based key (first few words)
                content = item.content[:50]
                cluster_key = f"content_{hash(content) % 1000}"
            
            candidate.semantic_cluster = cluster_key
            groups[cluster_key].append(candidate)
        
        return dict(groups)


class MemoryConsolidator:
    """
    Memory consolidation system for transferring memories to long-term storage.
    
    Implements cognitive psychology concepts:
    - Selective consolidation based on importance
    - Semantic organization during storage
    - Sleep-like consolidation cycles
    - Schema integration for knowledge organization
    
    Example:
        >>> from agent_memory_toolkit.temporal import (
        ...     MemoryConsolidator,
        ...     WorkingMemory,
        ...     InMemoryLongTermStore,
        ... )
        >>> 
        >>> wm = WorkingMemory()
        >>> ltm = InMemoryLongTermStore()
        >>> consolidator = MemoryConsolidator(
        ...     working_memory=wm,
        ...     long_term_store=ltm,
        ... )
        >>> 
        >>> # Add items to working memory
        >>> wm.add("Important fact", priority=0.9)
        >>> wm.add("Another fact", priority=0.7)
        >>> 
        >>> # Consolidate
        >>> result = consolidator.consolidate()
        >>> print(f"Consolidated {result.consolidated_count} memories")
    """
    
    def __init__(
        self,
        working_memory: WorkingMemory | None = None,
        episodic_store: EpisodicMemoryStore | None = None,
        long_term_store: LongTermStore | None = None,
        config: ConsolidationConfig | None = None,
    ):
        """
        Initialize the consolidator.
        
        Args:
            working_memory: Working memory source
            episodic_store: Episodic memory source
            long_term_store: Long-term storage destination
            config: Consolidation configuration
        """
        self.working_memory = working_memory
        self.episodic_store = episodic_store
        self.long_term_store = long_term_store or InMemoryLongTermStore()
        self.config = config or ConsolidationConfig()
        
        self._scorer = ConsolidationScorer(self.config)
        self._grouper = SemanticGrouper() if self.config.enable_semantic_grouping else None
        
        # State
        self._last_consolidation: datetime | None = None
        self._consolidation_history: list[ConsolidationResult] = []
        
        # Callbacks
        self._transform_callback: Callable[[str], str] | None = None
        
        # Auto-connect to working memory displacement
        if working_memory:
            working_memory.on_displacement(self._handle_displacement)
    
    def _handle_displacement(self, result: DisplacementResult) -> None:
        """Handle displaced items from working memory."""
        logger.debug(f"Handling {len(result.displaced_items)} displaced items")
        
        # Queue displaced items for consolidation
        candidates = []
        for item in result.displaced_items:
            if self._scorer.should_consolidate(item):
                candidates.append(ConsolidationCandidate(
                    item=item,
                    score=self._scorer.score(item),
                    source="working_memory_displacement",
                ))
        
        if candidates:
            self._consolidate_candidates(candidates, ConsolidationTrigger.CAPACITY)
    
    def consolidate(
        self,
        trigger: ConsolidationTrigger = ConsolidationTrigger.EXPLICIT,
        force: bool = False,
    ) -> ConsolidationResult:
        """
        Run consolidation from short-term to long-term memory.
        
        Args:
            trigger: What triggered this consolidation
            force: Force consolidation even if interval not elapsed
            
        Returns:
            ConsolidationResult with details
        """
        start_time = datetime.now()
        
        # Check interval
        if not force and self._last_consolidation:
            elapsed = start_time - self._last_consolidation
            if elapsed < self.config.consolidation_interval:
                return ConsolidationResult(
                    success=True,
                    consolidated_count=0,
                    skipped_count=0,
                    memories=[],
                    trigger=trigger,
                    errors=["Consolidation interval not elapsed"],
                )
        
        # Gather candidates
        candidates = self._gather_candidates()
        
        # Run consolidation
        result = self._consolidate_candidates(candidates, trigger)
        
        # Update state
        self._last_consolidation = datetime.now()
        result.duration = datetime.now() - start_time
        self._consolidation_history.append(result)
        
        return result
    
    def _gather_candidates(self) -> list[ConsolidationCandidate]:
        """Gather candidates from all sources."""
        candidates = []
        
        # From working memory
        if self.working_memory:
            for item in self.working_memory:
                if self._scorer.should_consolidate(item):
                    candidates.append(ConsolidationCandidate(
                        item=item,
                        score=self._scorer.score(item),
                        source="working_memory",
                    ))
        
        # From episodic store (strong memories)
        if self.episodic_store:
            for item in self.episodic_store.get_strong_memories(min_strength=0.5):
                if self._scorer.should_consolidate(item):
                    candidates.append(ConsolidationCandidate(
                        item=item,
                        score=self._scorer.score(item),
                        source="episodic_store",
                    ))
        
        # Sort by score
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        return candidates
    
    def _consolidate_candidates(
        self,
        candidates: list[ConsolidationCandidate],
        trigger: ConsolidationTrigger,
    ) -> ConsolidationResult:
        """Process candidates for consolidation."""
        if not candidates:
            return ConsolidationResult(
                success=True,
                consolidated_count=0,
                skipped_count=0,
                memories=[],
                trigger=trigger,
            )
        
        consolidated: list[ConsolidatedMemory] = []
        skipped = 0
        errors: list[str] = []
        
        # Group if enabled
        if self._grouper:
            groups = self._grouper.group(candidates)
        else:
            groups = {"default": candidates}
        
        # Process each group
        for cluster_id, group_candidates in groups.items():
            try:
                for candidate in group_candidates[:self.config.batch_size]:
                    memory = self._consolidate_single(candidate, cluster_id)
                    if memory:
                        consolidated.append(memory)
                    else:
                        skipped += 1
            except Exception as e:
                errors.append(f"Error consolidating cluster {cluster_id}: {str(e)}")
                logger.exception(f"Consolidation error: {e}")
        
        return ConsolidationResult(
            success=len(errors) == 0,
            consolidated_count=len(consolidated),
            skipped_count=skipped,
            memories=consolidated,
            trigger=trigger,
            errors=errors,
        )
    
    def _consolidate_single(
        self,
        candidate: ConsolidationCandidate,
        cluster_id: str | None = None,
    ) -> ConsolidatedMemory | None:
        """Consolidate a single candidate."""
        item = candidate.item
        
        # Transform content if enabled
        content = item.content
        if self.config.transform_on_consolidation and self._transform_callback:
            try:
                content = self._transform_callback(content)
            except Exception as e:
                logger.warning(f"Transform failed: {e}")
        
        # Create consolidated memory
        memory = ConsolidatedMemory(
            memory_id=str(uuid.uuid4()),
            content=content,
            original_content=item.content,
            timestamp=item.timestamp,
            consolidated_at=datetime.now(),
            metadata=getattr(item, 'metadata', {}),
            source_ids=[item.item_id],
            consolidation_score=candidate.score,
            is_integrated=False,
            schema_id=cluster_id,
            strength=getattr(item, 'strength', getattr(item, 'activation', 0.8)),
            importance=getattr(item, 'importance', 0.5),
        )
        
        # Store to long-term memory
        try:
            self.long_term_store.store(memory)
            
            # Remove from working memory if applicable
            if candidate.source == "working_memory" and self.working_memory:
                self.working_memory.remove(item.item_id)
            
            return memory
        except Exception as e:
            logger.error(f"Failed to store consolidated memory: {e}")
            return None
    
    def run_sleep_consolidation(self) -> ConsolidationResult:
        """
        Run a "sleep-like" consolidation cycle.
        
        This applies additional memory strengthening and schema integration.
        
        Returns:
            ConsolidationResult
        """
        if not self.config.enable_sleep_consolidation:
            return ConsolidationResult(
                success=False,
                consolidated_count=0,
                skipped_count=0,
                memories=[],
                trigger=ConsolidationTrigger.SLEEP_CYCLE,
                errors=["Sleep consolidation is disabled"],
            )
        
        # First, run normal consolidation
        result = self.consolidate(
            trigger=ConsolidationTrigger.SLEEP_CYCLE,
            force=True,
        )
        
        # Apply additional strengthening to consolidated memories
        if isinstance(self.long_term_store, InMemoryLongTermStore):
            for memory in self.long_term_store.list_memories():
                boost = self.config.sleep_consolidation_boost
                memory.strength = min(1.0, memory.strength + boost)
        
        return result
    
    def set_transform_callback(
        self,
        callback: Callable[[str], str],
    ) -> None:
        """
        Set a callback to transform content during consolidation.
        
        Useful for summarization, cleaning, or enrichment.
        
        Args:
            callback: Function that takes content and returns transformed content
        """
        self._transform_callback = callback
    
    def get_history(self, limit: int = 10) -> list[ConsolidationResult]:
        """Get recent consolidation history."""
        return self._consolidation_history[-limit:]
    
    def get_stats(self) -> dict[str, Any]:
        """Get consolidation statistics."""
        total_consolidated = sum(r.consolidated_count for r in self._consolidation_history)
        total_skipped = sum(r.skipped_count for r in self._consolidation_history)
        
        return {
            "total_consolidations": len(self._consolidation_history),
            "total_memories_consolidated": total_consolidated,
            "total_skipped": total_skipped,
            "last_consolidation": self._last_consolidation.isoformat() if self._last_consolidation else None,
            "long_term_store_size": len(self.long_term_store) if hasattr(self.long_term_store, '__len__') else None,
        }


def create_consolidator(
    working_memory: WorkingMemory | None = None,
    episodic_store: EpisodicMemoryStore | None = None,
    strategy: ConsolidationStrategy = ConsolidationStrategy.COMBINED,
    **config_kwargs,
) -> MemoryConsolidator:
    """
    Create a memory consolidator with specified settings.
    
    Args:
        working_memory: Working memory source
        episodic_store: Episodic memory source
        strategy: Consolidation selection strategy
        **config_kwargs: Additional config parameters
        
    Returns:
        Configured MemoryConsolidator
        
    Example:
        >>> wm = WorkingMemory()
        >>> consolidator = create_consolidator(
        ...     working_memory=wm,
        ...     strategy=ConsolidationStrategy.IMPORTANT,
        ...     min_importance=0.5,
        ... )
    """
    config = ConsolidationConfig(
        strategy=strategy,
        **config_kwargs,
    )
    return MemoryConsolidator(
        working_memory=working_memory,
        episodic_store=episodic_store,
        config=config,
    )
