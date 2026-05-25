"""Timeline-based memory organization for temporal retrieval.

This module provides timeline-based organization of memories, enabling:
- Chronological ordering and navigation
- Time-windowed retrieval (recent, today, this week, etc.)
- Temporal clustering of related events
- Recency-weighted search boosting
- Historical snapshots and time-travel queries
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Generic, Iterator, TypeVar
from collections import defaultdict
import heapq
import bisect

logger = logging.getLogger(__name__)


class TimeWindow(Enum):
    """Predefined time windows for memory retrieval."""
    LAST_MINUTE = "last_minute"
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    ALL_TIME = "all_time"
    
    def to_timedelta(self) -> timedelta | None:
        """Convert to timedelta for filtering."""
        mapping = {
            TimeWindow.LAST_MINUTE: timedelta(minutes=1),
            TimeWindow.LAST_HOUR: timedelta(hours=1),
            TimeWindow.LAST_DAY: timedelta(days=1),
            TimeWindow.LAST_WEEK: timedelta(weeks=1),
            TimeWindow.LAST_MONTH: timedelta(days=30),
            TimeWindow.LAST_QUARTER: timedelta(days=90),
            TimeWindow.LAST_YEAR: timedelta(days=365),
            TimeWindow.ALL_TIME: None,
        }
        return mapping.get(self)


class RecencyDecay(Enum):
    """Decay functions for recency-based scoring."""
    NONE = "none"               # No decay
    LINEAR = "linear"           # Linear decay
    EXPONENTIAL = "exponential" # Exponential decay
    LOGARITHMIC = "logarithmic" # Logarithmic decay (slow decay)
    STEP = "step"               # Step function (sharp cutoffs)


@dataclass
class TimelineConfig:
    """Configuration for timeline memory organization."""
    
    # Time granularity for clustering
    cluster_granularity: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    # Recency scoring
    recency_decay: RecencyDecay = RecencyDecay.EXPONENTIAL
    decay_half_life: timedelta = field(default_factory=lambda: timedelta(days=7))
    max_recency_boost: float = 1.0
    
    # Temporal indexing
    index_by_day: bool = True
    index_by_week: bool = True
    index_by_month: bool = True
    
    # Navigation
    default_page_size: int = 50
    max_page_size: int = 500


@dataclass
class TemporalMemory:
    """A memory with temporal metadata."""
    memory_id: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    
    # Temporal annotations
    duration: timedelta | None = None  # For events with duration
    end_timestamp: datetime | None = None
    temporal_references: list[str] = field(default_factory=list)  # e.g., "yesterday", "last week"
    
    def __lt__(self, other: "TemporalMemory") -> bool:
        """Enable heap operations based on timestamp."""
        return self.timestamp < other.timestamp
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "embedding": self.embedding,
            "duration": self.duration.total_seconds() if self.duration else None,
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
            "temporal_references": self.temporal_references,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalMemory":
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            duration=timedelta(seconds=data["duration"]) if data.get("duration") else None,
            end_timestamp=datetime.fromisoformat(data["end_timestamp"]) if data.get("end_timestamp") else None,
            temporal_references=data.get("temporal_references", []),
        )


@dataclass
class TemporalCluster:
    """A cluster of temporally related memories."""
    cluster_id: str
    start_time: datetime
    end_time: datetime
    memories: list[TemporalMemory] = field(default_factory=list)
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    
    @property
    def duration(self) -> timedelta:
        """Get cluster duration."""
        return self.end_time - self.start_time
    
    @property
    def memory_count(self) -> int:
        """Get number of memories in cluster."""
        return len(self.memories)
    
    def add_memory(self, memory: TemporalMemory) -> None:
        """Add a memory to the cluster, updating boundaries."""
        self.memories.append(memory)
        if memory.timestamp < self.start_time:
            self.start_time = memory.timestamp
        if memory.timestamp > self.end_time:
            self.end_time = memory.timestamp


@dataclass
class TimelineQuery:
    """Query parameters for timeline retrieval."""
    start_time: datetime | None = None
    end_time: datetime | None = None
    time_window: TimeWindow | None = None
    
    # Filtering
    content_filter: str | None = None
    tags: list[str] | None = None
    metadata_filters: dict[str, Any] | None = None
    
    # Scoring
    recency_weight: float = 0.0  # 0 = no recency boost, 1 = strong recency boost
    
    # Pagination
    limit: int = 50
    offset: int = 0
    
    # Ordering
    order_descending: bool = True  # Most recent first
    
    def get_time_bounds(self, reference_time: datetime | None = None) -> tuple[datetime | None, datetime | None]:
        """Get effective start and end times."""
        ref = reference_time or datetime.now()
        
        if self.time_window:
            delta = self.time_window.to_timedelta()
            if delta:
                return ref - delta, ref
            return None, None
        
        return self.start_time, self.end_time


@dataclass
class TimelineResult:
    """Result of a timeline query."""
    memories: list[TemporalMemory]
    total_count: int
    time_range: tuple[datetime | None, datetime | None]
    clusters: list[TemporalCluster] | None = None
    
    # Pagination info
    has_more: bool = False
    next_offset: int | None = None


class RecencyScorer:
    """Calculates recency-based scores for memories."""
    
    def __init__(self, config: TimelineConfig):
        self.config = config
    
    def score(
        self, 
        memory: TemporalMemory, 
        reference_time: datetime | None = None,
    ) -> float:
        """
        Calculate recency score for a memory.
        
        Args:
            memory: The memory to score
            reference_time: Reference time (default: now)
            
        Returns:
            Score between 0 and max_recency_boost
        """
        ref = reference_time or datetime.now()
        age = ref - memory.timestamp
        
        if age.total_seconds() < 0:
            # Future timestamp - return max boost
            return self.config.max_recency_boost
        
        decay = self.config.recency_decay
        half_life_seconds = self.config.decay_half_life.total_seconds()
        age_seconds = age.total_seconds()
        
        if decay == RecencyDecay.NONE:
            return self.config.max_recency_boost
        
        elif decay == RecencyDecay.LINEAR:
            # Linear decay over 2x half-life
            decay_window = half_life_seconds * 2
            score = max(0, 1 - (age_seconds / decay_window))
        
        elif decay == RecencyDecay.EXPONENTIAL:
            # Exponential decay: score = 0.5^(age/half_life)
            import math
            score = math.pow(0.5, age_seconds / half_life_seconds)
        
        elif decay == RecencyDecay.LOGARITHMIC:
            # Logarithmic decay (slow)
            import math
            score = 1.0 / (1 + math.log1p(age_seconds / half_life_seconds))
        
        elif decay == RecencyDecay.STEP:
            # Step function: full boost within half-life, then drops
            if age_seconds <= half_life_seconds:
                score = 1.0
            elif age_seconds <= half_life_seconds * 2:
                score = 0.5
            elif age_seconds <= half_life_seconds * 4:
                score = 0.25
            else:
                score = 0.0
        
        else:
            score = 1.0
        
        return score * self.config.max_recency_boost


class TemporalIndex:
    """
    Index for efficient temporal queries.
    
    Maintains sorted lists and hash indexes for fast:
    - Range queries (between time A and B)
    - Window queries (last N hours/days)
    - Point-in-time queries
    """
    
    def __init__(self, config: TimelineConfig | None = None):
        self.config = config or TimelineConfig()
        
        # Primary index: sorted by timestamp
        self._memories: list[TemporalMemory] = []
        self._timestamps: list[datetime] = []  # Parallel list for bisect
        
        # Secondary indexes
        self._by_day: dict[str, list[str]] = defaultdict(list)  # date -> memory_ids
        self._by_week: dict[str, list[str]] = defaultdict(list)  # year-week -> memory_ids
        self._by_month: dict[str, list[str]] = defaultdict(list)  # year-month -> memory_ids
        
        # ID lookup
        self._by_id: dict[str, TemporalMemory] = {}
    
    def _date_key(self, dt: datetime) -> str:
        """Get day key for a datetime."""
        return dt.strftime("%Y-%m-%d")
    
    def _week_key(self, dt: datetime) -> str:
        """Get week key for a datetime."""
        return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
    
    def _month_key(self, dt: datetime) -> str:
        """Get month key for a datetime."""
        return dt.strftime("%Y-%m")
    
    def add(self, memory: TemporalMemory) -> None:
        """Add a memory to the index."""
        # Insert in sorted order
        idx = bisect.bisect_left(self._timestamps, memory.timestamp)
        self._timestamps.insert(idx, memory.timestamp)
        self._memories.insert(idx, memory)
        
        # Update secondary indexes
        self._by_id[memory.memory_id] = memory
        
        if self.config.index_by_day:
            self._by_day[self._date_key(memory.timestamp)].append(memory.memory_id)
        if self.config.index_by_week:
            self._by_week[self._week_key(memory.timestamp)].append(memory.memory_id)
        if self.config.index_by_month:
            self._by_month[self._month_key(memory.timestamp)].append(memory.memory_id)
    
    def remove(self, memory_id: str) -> bool:
        """Remove a memory from the index."""
        if memory_id not in self._by_id:
            return False
        
        memory = self._by_id[memory_id]
        
        # Remove from primary index
        idx = bisect.bisect_left(self._timestamps, memory.timestamp)
        while idx < len(self._memories):
            if self._memories[idx].memory_id == memory_id:
                self._memories.pop(idx)
                self._timestamps.pop(idx)
                break
            idx += 1
        
        # Remove from secondary indexes
        del self._by_id[memory_id]
        
        day_key = self._date_key(memory.timestamp)
        if memory_id in self._by_day[day_key]:
            self._by_day[day_key].remove(memory_id)
        
        week_key = self._week_key(memory.timestamp)
        if memory_id in self._by_week[week_key]:
            self._by_week[week_key].remove(memory_id)
        
        month_key = self._month_key(memory.timestamp)
        if memory_id in self._by_month[month_key]:
            self._by_month[month_key].remove(memory_id)
        
        return True
    
    def get(self, memory_id: str) -> TemporalMemory | None:
        """Get a memory by ID."""
        return self._by_id.get(memory_id)
    
    def range_query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        descending: bool = True,
    ) -> list[TemporalMemory]:
        """
        Query memories within a time range.
        
        Args:
            start_time: Start of range (inclusive)
            end_time: End of range (inclusive)
            limit: Maximum results to return
            descending: If True, most recent first
            
        Returns:
            List of memories in the range
        """
        # Find range bounds
        if start_time:
            start_idx = bisect.bisect_left(self._timestamps, start_time)
        else:
            start_idx = 0
        
        if end_time:
            end_idx = bisect.bisect_right(self._timestamps, end_time)
        else:
            end_idx = len(self._memories)
        
        # Extract slice
        result = self._memories[start_idx:end_idx]
        
        # Order
        if descending:
            result = list(reversed(result))
        
        # Limit
        if limit:
            result = result[:limit]
        
        return result
    
    def get_by_day(self, date: datetime) -> list[TemporalMemory]:
        """Get all memories from a specific day."""
        key = self._date_key(date)
        memory_ids = self._by_day.get(key, [])
        return [self._by_id[mid] for mid in memory_ids if mid in self._by_id]
    
    def get_by_week(self, date: datetime) -> list[TemporalMemory]:
        """Get all memories from the week containing the date."""
        key = self._week_key(date)
        memory_ids = self._by_week.get(key, [])
        return [self._by_id[mid] for mid in memory_ids if mid in self._by_id]
    
    def get_by_month(self, date: datetime) -> list[TemporalMemory]:
        """Get all memories from the month containing the date."""
        key = self._month_key(date)
        memory_ids = self._by_month.get(key, [])
        return [self._by_id[mid] for mid in memory_ids if mid in self._by_id]
    
    def get_recent(self, n: int = 10) -> list[TemporalMemory]:
        """Get the N most recent memories."""
        return list(reversed(self._memories[-n:]))
    
    def __len__(self) -> int:
        """Get total number of indexed memories."""
        return len(self._memories)
    
    def __iter__(self) -> Iterator[TemporalMemory]:
        """Iterate over all memories in chronological order."""
        return iter(self._memories)


class Timeline:
    """
    Timeline-based memory organization system.
    
    Provides chronological organization, temporal clustering, recency scoring,
    and efficient time-based retrieval of memories.
    
    Example:
        >>> from agent_memory_toolkit.temporal import Timeline, TimeWindow
        >>> 
        >>> timeline = Timeline()
        >>> 
        >>> # Add memories
        >>> timeline.add(TemporalMemory(
        ...     memory_id="1",
        ...     content="User asked about password reset",
        ...     timestamp=datetime.now(),
        ... ))
        >>> 
        >>> # Query by time window
        >>> recent = timeline.query(time_window=TimeWindow.LAST_HOUR)
        >>> 
        >>> # Get with recency scoring
        >>> scored = timeline.get_with_recency_scores(memories)
    """
    
    def __init__(self, config: TimelineConfig | None = None):
        """
        Initialize the timeline.
        
        Args:
            config: Timeline configuration
        """
        self.config = config or TimelineConfig()
        self._index = TemporalIndex(self.config)
        self._recency_scorer = RecencyScorer(self.config)
        self._clusters: dict[str, TemporalCluster] = {}
    
    def add(self, memory: TemporalMemory) -> None:
        """
        Add a memory to the timeline.
        
        Args:
            memory: The temporal memory to add
        """
        self._index.add(memory)
        logger.debug(f"Added memory {memory.memory_id} at {memory.timestamp}")
    
    def add_batch(self, memories: list[TemporalMemory]) -> int:
        """
        Add multiple memories to the timeline.
        
        Args:
            memories: List of memories to add
            
        Returns:
            Number of memories added
        """
        for memory in memories:
            self._index.add(memory)
        return len(memories)
    
    def remove(self, memory_id: str) -> bool:
        """
        Remove a memory from the timeline.
        
        Args:
            memory_id: ID of memory to remove
            
        Returns:
            True if removed, False if not found
        """
        return self._index.remove(memory_id)
    
    def get(self, memory_id: str) -> TemporalMemory | None:
        """Get a memory by ID."""
        return self._index.get(memory_id)
    
    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        time_window: TimeWindow | None = None,
        limit: int | None = None,
        offset: int = 0,
        descending: bool = True,
        include_clusters: bool = False,
        reference_time: datetime | None = None,
    ) -> TimelineResult:
        """
        Query memories from the timeline.
        
        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            time_window: Predefined time window (overrides start/end)
            limit: Maximum results
            offset: Skip first N results
            descending: Most recent first if True
            include_clusters: Include temporal clusters in result
            reference_time: Reference time for time_window calculation
            
        Returns:
            TimelineResult with matching memories
        """
        ref = reference_time or datetime.now()
        
        # Resolve time window
        effective_start = start_time
        effective_end = end_time
        
        if time_window:
            delta = time_window.to_timedelta()
            if delta:
                effective_start = ref - delta
                effective_end = ref
        
        # Query index
        all_results = self._index.range_query(
            start_time=effective_start,
            end_time=effective_end,
            descending=descending,
        )
        
        total_count = len(all_results)
        
        # Apply pagination
        if offset:
            all_results = all_results[offset:]
        if limit:
            paginated = all_results[:limit]
            has_more = len(all_results) > limit
        else:
            paginated = all_results
            has_more = False
        
        # Build clusters if requested
        clusters = None
        if include_clusters:
            clusters = self._cluster_memories(paginated)
        
        return TimelineResult(
            memories=paginated,
            total_count=total_count,
            time_range=(effective_start, effective_end),
            clusters=clusters,
            has_more=has_more,
            next_offset=offset + len(paginated) if has_more else None,
        )
    
    def get_recent(self, n: int = 10) -> list[TemporalMemory]:
        """Get the N most recent memories."""
        return self._index.get_recent(n)
    
    def get_by_day(self, date: datetime) -> list[TemporalMemory]:
        """Get all memories from a specific day."""
        return self._index.get_by_day(date)
    
    def get_by_week(self, date: datetime) -> list[TemporalMemory]:
        """Get all memories from the week containing the date."""
        return self._index.get_by_week(date)
    
    def get_by_month(self, date: datetime) -> list[TemporalMemory]:
        """Get all memories from the month containing the date."""
        return self._index.get_by_month(date)
    
    def score_recency(
        self, 
        memory: TemporalMemory,
        reference_time: datetime | None = None,
    ) -> float:
        """
        Get the recency score for a memory.
        
        Args:
            memory: Memory to score
            reference_time: Reference time (default: now)
            
        Returns:
            Recency score between 0 and max_recency_boost
        """
        return self._recency_scorer.score(memory, reference_time)
    
    def get_with_recency_scores(
        self,
        memories: list[TemporalMemory],
        reference_time: datetime | None = None,
    ) -> list[tuple[TemporalMemory, float]]:
        """
        Get memories with their recency scores.
        
        Args:
            memories: Memories to score
            reference_time: Reference time for scoring
            
        Returns:
            List of (memory, score) tuples sorted by score descending
        """
        scored = [
            (memory, self._recency_scorer.score(memory, reference_time))
            for memory in memories
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def boost_search_results(
        self,
        results: list[tuple[TemporalMemory, float]],
        recency_weight: float = 0.3,
        reference_time: datetime | None = None,
    ) -> list[tuple[TemporalMemory, float]]:
        """
        Boost search results with recency scores.
        
        Args:
            results: List of (memory, relevance_score) tuples
            recency_weight: Weight of recency vs relevance (0-1)
            reference_time: Reference time for recency calculation
            
        Returns:
            Re-ranked results with combined scores
        """
        if recency_weight <= 0:
            return results
        
        relevance_weight = 1 - recency_weight
        boosted = []
        
        for memory, relevance_score in results:
            recency_score = self._recency_scorer.score(memory, reference_time)
            combined = (relevance_weight * relevance_score) + (recency_weight * recency_score)
            boosted.append((memory, combined))
        
        return sorted(boosted, key=lambda x: x[1], reverse=True)
    
    def _cluster_memories(
        self, 
        memories: list[TemporalMemory],
    ) -> list[TemporalCluster]:
        """Cluster memories by temporal proximity."""
        if not memories:
            return []
        
        # Sort by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.timestamp)
        
        clusters: list[TemporalCluster] = []
        current_cluster: TemporalCluster | None = None
        
        for memory in sorted_memories:
            if current_cluster is None:
                # Start new cluster
                current_cluster = TemporalCluster(
                    cluster_id=f"cluster_{memory.memory_id}",
                    start_time=memory.timestamp,
                    end_time=memory.timestamp,
                    memories=[memory],
                )
            else:
                # Check if within granularity
                gap = memory.timestamp - current_cluster.end_time
                if gap <= self.config.cluster_granularity:
                    current_cluster.add_memory(memory)
                else:
                    # Save current cluster and start new one
                    clusters.append(current_cluster)
                    current_cluster = TemporalCluster(
                        cluster_id=f"cluster_{memory.memory_id}",
                        start_time=memory.timestamp,
                        end_time=memory.timestamp,
                        memories=[memory],
                    )
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def get_timeline_stats(self) -> dict[str, Any]:
        """
        Get statistics about the timeline.
        
        Returns:
            Dictionary with timeline statistics
        """
        if len(self._index) == 0:
            return {
                "total_memories": 0,
                "earliest": None,
                "latest": None,
                "span": None,
            }
        
        memories = list(self._index)
        earliest = memories[0].timestamp
        latest = memories[-1].timestamp
        
        return {
            "total_memories": len(self._index),
            "earliest": earliest.isoformat(),
            "latest": latest.isoformat(),
            "span": str(latest - earliest),
            "days_with_memories": len(self._index._by_day),
            "weeks_with_memories": len(self._index._by_week),
            "months_with_memories": len(self._index._by_month),
        }
    
    def __len__(self) -> int:
        """Get total number of memories."""
        return len(self._index)
    
    def __iter__(self) -> Iterator[TemporalMemory]:
        """Iterate over all memories chronologically."""
        return iter(self._index)


def create_timeline(
    memories: list[dict[str, Any]] | None = None,
    config: TimelineConfig | None = None,
) -> Timeline:
    """
    Create a timeline from existing memories.
    
    Args:
        memories: List of memory dictionaries with 'timestamp' field
        config: Timeline configuration
        
    Returns:
        Configured Timeline instance
        
    Example:
        >>> memories = [
        ...     {"memory_id": "1", "content": "Hello", "timestamp": datetime.now()},
        ...     {"memory_id": "2", "content": "World", "timestamp": datetime.now()},
        ... ]
        >>> timeline = create_timeline(memories)
    """
    timeline = Timeline(config)
    
    if memories:
        for mem_data in memories:
            if isinstance(mem_data.get("timestamp"), str):
                mem_data["timestamp"] = datetime.fromisoformat(mem_data["timestamp"])
            memory = TemporalMemory.from_dict(mem_data)
            timeline.add(memory)
    
    return timeline
