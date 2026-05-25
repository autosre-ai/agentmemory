"""Episodic memory system with decay and forgetting curves.

This module implements cognitively-inspired episodic memory:
- Episode boundaries detection
- Memory strength with decay over time
- Forgetting curves (Ebbinghaus-inspired)
- Spaced repetition for memory reinforcement
- Emotional salience and significance weighting
- Autobiographical memory organization
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Iterator
from collections import defaultdict

logger = logging.getLogger(__name__)


class DecayModel(Enum):
    """Models for memory decay over time."""
    EBBINGHAUS = "ebbinghaus"       # Classic forgetting curve
    POWER_LAW = "power_law"         # Power law of forgetting
    EXPONENTIAL = "exponential"     # Simple exponential decay
    MCM = "mcm"                     # Memory Chain Model
    ACT_R = "act_r"                 # ACT-R base-level learning


class EpisodeType(Enum):
    """Types of episodic memories."""
    INTERACTION = "interaction"     # User interaction episode
    TASK = "task"                   # Task execution episode
    LEARNING = "learning"           # Learning/information episode
    EVENT = "event"                 # External event
    REFLECTION = "reflection"       # Agent reflection/reasoning


class EmotionalValence(Enum):
    """Emotional valence of memories."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class EpisodicMemoryConfig:
    """Configuration for the episodic memory system."""
    
    # Decay settings
    decay_model: DecayModel = DecayModel.EBBINGHAUS
    base_retention_rate: float = 0.9  # Initial retention probability
    decay_rate: float = 0.1  # How fast memories decay
    
    # Forgetting
    forgetting_threshold: float = 0.1  # Below this strength, memory is "forgotten"
    enable_forgetting: bool = True  # Whether to actually remove forgotten memories
    
    # Reinforcement
    rehearsal_boost: float = 0.2  # Strength boost from rehearsal
    retrieval_boost: float = 0.1  # Strength boost from retrieval
    max_strength: float = 1.0  # Maximum memory strength
    
    # Episode detection
    episode_gap_threshold: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    min_episode_events: int = 2
    max_episode_duration: timedelta = field(default_factory=lambda: timedelta(hours=4))
    
    # Salience
    emotional_salience_weight: float = 0.3
    importance_weight: float = 0.4
    recency_weight: float = 0.3
    
    # Consolidation
    consolidation_interval: timedelta = field(default_factory=lambda: timedelta(hours=6))


@dataclass
class EpisodicMemoryItem:
    """A single item within an episode."""
    item_id: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    
    # Memory properties
    strength: float = 1.0  # Current memory strength [0, 1]
    initial_strength: float = 1.0
    importance: float = 0.5  # Rated importance [0, 1]
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL
    
    # Tracking
    creation_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime | None = None
    access_count: int = 0
    rehearsal_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "embedding": self.embedding,
            "strength": self.strength,
            "initial_strength": self.initial_strength,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence.value,
            "creation_time": self.creation_time.isoformat(),
            "last_access_time": self.last_access_time.isoformat() if self.last_access_time else None,
            "access_count": self.access_count,
            "rehearsal_count": self.rehearsal_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodicMemoryItem":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            strength=data.get("strength", 1.0),
            initial_strength=data.get("initial_strength", 1.0),
            importance=data.get("importance", 0.5),
            emotional_valence=EmotionalValence(data.get("emotional_valence", 0)),
            creation_time=datetime.fromisoformat(data["creation_time"]) if data.get("creation_time") else datetime.now(),
            last_access_time=datetime.fromisoformat(data["last_access_time"]) if data.get("last_access_time") else None,
            access_count=data.get("access_count", 0),
            rehearsal_count=data.get("rehearsal_count", 0),
        )


@dataclass
class Episode:
    """
    An episode is a coherent sequence of related memory items.
    
    Episodes have boundaries determined by temporal gaps, context shifts,
    or explicit markers. They form the building blocks of autobiographical memory.
    """
    episode_id: str
    episode_type: EpisodeType
    start_time: datetime
    end_time: datetime
    items: list[EpisodicMemoryItem] = field(default_factory=list)
    
    # Episode metadata
    title: str | None = None
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    
    # Episode-level properties
    overall_importance: float = 0.5
    emotional_peak: EmotionalValence = EmotionalValence.NEUTRAL
    strength: float = 1.0  # Aggregate episode strength
    
    # Linking
    related_episodes: list[str] = field(default_factory=list)
    
    @property
    def duration(self) -> timedelta:
        """Get episode duration."""
        return self.end_time - self.start_time
    
    @property
    def item_count(self) -> int:
        """Get number of items in episode."""
        return len(self.items)
    
    def add_item(self, item: EpisodicMemoryItem) -> None:
        """Add an item to the episode."""
        self.items.append(item)
        if item.timestamp < self.start_time:
            self.start_time = item.timestamp
        if item.timestamp > self.end_time:
            self.end_time = item.timestamp
        
        # Update emotional peak
        if abs(item.emotional_valence.value) > abs(self.emotional_peak.value):
            self.emotional_peak = item.emotional_valence
        
        # Update importance
        self._recalculate_importance()
    
    def _recalculate_importance(self) -> None:
        """Recalculate overall episode importance."""
        if not self.items:
            self.overall_importance = 0.5
            return
        
        # Weighted average with emphasis on high-importance items
        weights = [item.importance ** 2 for item in self.items]
        total_weight = sum(weights)
        if total_weight > 0:
            self.overall_importance = sum(
                item.importance * w for item, w in zip(self.items, weights)
            ) / total_weight
    
    def get_strength(self) -> float:
        """Get aggregate episode strength based on items."""
        if not self.items:
            return 0.0
        # Use max strength of contained items
        return max(item.strength for item in self.items)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "episode_type": self.episode_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "items": [item.to_dict() for item in self.items],
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "context": self.context,
            "overall_importance": self.overall_importance,
            "emotional_peak": self.emotional_peak.value,
            "strength": self.strength,
            "related_episodes": self.related_episodes,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        """Create from dictionary."""
        episode = cls(
            episode_id=data["episode_id"],
            episode_type=EpisodeType(data["episode_type"]),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            title=data.get("title"),
            summary=data.get("summary"),
            tags=data.get("tags", []),
            context=data.get("context", {}),
            overall_importance=data.get("overall_importance", 0.5),
            emotional_peak=EmotionalValence(data.get("emotional_peak", 0)),
            strength=data.get("strength", 1.0),
            related_episodes=data.get("related_episodes", []),
        )
        episode.items = [
            EpisodicMemoryItem.from_dict(item_data) 
            for item_data in data.get("items", [])
        ]
        return episode


class DecayCalculator:
    """
    Calculates memory decay using various cognitive models.
    
    Implements several forgetting curves from cognitive psychology research.
    """
    
    def __init__(self, config: EpisodicMemoryConfig):
        self.config = config
    
    def calculate_retention(
        self,
        item: EpisodicMemoryItem,
        current_time: datetime | None = None,
    ) -> float:
        """
        Calculate current retention/strength of a memory item.
        
        Args:
            item: The memory item
            current_time: Current time (default: now)
            
        Returns:
            Retention probability/strength [0, 1]
        """
        now = current_time or datetime.now()
        age = now - item.creation_time
        age_hours = age.total_seconds() / 3600
        
        if age_hours <= 0:
            return item.initial_strength
        
        model = self.config.decay_model
        
        if model == DecayModel.EBBINGHAUS:
            return self._ebbinghaus_decay(item, age_hours)
        elif model == DecayModel.POWER_LAW:
            return self._power_law_decay(item, age_hours)
        elif model == DecayModel.EXPONENTIAL:
            return self._exponential_decay(item, age_hours)
        elif model == DecayModel.ACT_R:
            return self._act_r_decay(item, age_hours)
        else:
            return self._exponential_decay(item, age_hours)
    
    def _ebbinghaus_decay(self, item: EpisodicMemoryItem, age_hours: float) -> float:
        """
        Ebbinghaus forgetting curve: R = e^(-t/S)
        where t is time and S is memory stability.
        """
        # Stability increases with rehearsals
        stability = 24 * (1 + item.rehearsal_count * 0.5)
        retention = math.exp(-age_hours / stability)
        
        # Apply importance modifier
        retention = retention * (0.5 + 0.5 * item.importance)
        
        return max(0, min(1, retention * item.initial_strength))
    
    def _power_law_decay(self, item: EpisodicMemoryItem, age_hours: float) -> float:
        """
        Power law of forgetting: R = a * t^(-b)
        More gradual decay than exponential.
        """
        a = item.initial_strength
        b = self.config.decay_rate / (1 + item.rehearsal_count * 0.3)
        
        retention = a * math.pow(max(1, age_hours), -b)
        
        # Apply importance modifier
        retention = retention * (0.5 + 0.5 * item.importance)
        
        return max(0, min(1, retention))
    
    def _exponential_decay(self, item: EpisodicMemoryItem, age_hours: float) -> float:
        """
        Simple exponential decay: R = R0 * e^(-λt)
        """
        lambda_decay = self.config.decay_rate / (1 + item.rehearsal_count * 0.5)
        retention = item.initial_strength * math.exp(-lambda_decay * age_hours / 24)
        
        # Apply importance modifier
        retention = retention * (0.5 + 0.5 * item.importance)
        
        return max(0, min(1, retention))
    
    def _act_r_decay(self, item: EpisodicMemoryItem, age_hours: float) -> float:
        """
        ACT-R base-level learning equation.
        Accounts for both decay and practice effects.
        """
        # Base-level activation decays with time
        d = 0.5  # Decay parameter
        
        # Each access creates a "trace" that decays
        # Here we use a simplified version
        n = max(1, item.access_count + item.rehearsal_count)
        
        # Base-level activation
        B = math.log(n) - d * math.log(age_hours + 1)
        
        # Convert to probability via logistic function
        retention = 1 / (1 + math.exp(-B))
        
        # Apply importance modifier  
        retention = retention * (0.5 + 0.5 * item.importance)
        
        return max(0, min(1, retention * item.initial_strength))
    
    def is_forgotten(self, item: EpisodicMemoryItem, current_time: datetime | None = None) -> bool:
        """Check if a memory item has been forgotten."""
        retention = self.calculate_retention(item, current_time)
        return retention < self.config.forgetting_threshold


class EpisodicMemoryStore:
    """
    Episodic memory store with decay, forgetting, and episode management.
    
    This implements a cognitively-inspired episodic memory system where:
    - Memories decay over time following forgetting curves
    - Retrieval and rehearsal strengthen memories
    - Memories are organized into episodes
    - Emotional salience affects retention
    
    Example:
        >>> from agent_memory_toolkit.temporal import EpisodicMemoryStore
        >>> 
        >>> store = EpisodicMemoryStore()
        >>> 
        >>> # Add memory items
        >>> item = store.add_item(
        ...     content="User asked about Python decorators",
        ...     importance=0.8,
        ...     emotional_valence=EmotionalValence.POSITIVE,
        ... )
        >>> 
        >>> # Retrieve and strengthen
        >>> memories = store.retrieve("Python decorators")
        >>> 
        >>> # Apply decay
        >>> store.apply_decay()
        >>> 
        >>> # Get active episodes
        >>> episodes = store.get_episodes(min_strength=0.3)
    """
    
    def __init__(self, config: EpisodicMemoryConfig | None = None):
        """
        Initialize the episodic memory store.
        
        Args:
            config: Configuration for the memory system
        """
        self.config = config or EpisodicMemoryConfig()
        self._decay_calculator = DecayCalculator(self.config)
        
        # Storage
        self._items: dict[str, EpisodicMemoryItem] = {}
        self._episodes: dict[str, Episode] = {}
        
        # Current episode tracking
        self._current_episode: Episode | None = None
        self._last_item_time: datetime | None = None
        
        # Statistics
        self._total_items_added: int = 0
        self._items_forgotten: int = 0
    
    def add_item(
        self,
        content: str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        importance: float = 0.5,
        emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL,
        episode_type: EpisodeType = EpisodeType.INTERACTION,
    ) -> EpisodicMemoryItem:
        """
        Add a new memory item.
        
        Args:
            content: Memory content
            timestamp: Event timestamp (default: now)
            metadata: Additional metadata
            embedding: Content embedding
            importance: Importance rating [0, 1]
            emotional_valence: Emotional valence
            episode_type: Type of episode for auto-grouping
            
        Returns:
            The created memory item
        """
        now = timestamp or datetime.now()
        
        item = EpisodicMemoryItem(
            item_id=str(uuid.uuid4()),
            content=content,
            timestamp=now,
            metadata=metadata or {},
            embedding=embedding,
            importance=importance,
            emotional_valence=emotional_valence,
            creation_time=now,
        )
        
        self._items[item.item_id] = item
        self._total_items_added += 1
        
        # Auto-episode management
        self._add_to_episode(item, episode_type)
        
        logger.debug(f"Added episodic memory item {item.item_id}")
        return item
    
    def _add_to_episode(self, item: EpisodicMemoryItem, episode_type: EpisodeType) -> None:
        """Add item to current or new episode based on temporal boundaries."""
        should_start_new = False
        
        if self._current_episode is None:
            should_start_new = True
        elif self._last_item_time:
            gap = item.timestamp - self._last_item_time
            if gap > self.config.episode_gap_threshold:
                should_start_new = True
            elif (item.timestamp - self._current_episode.start_time) > self.config.max_episode_duration:
                should_start_new = True
        
        if should_start_new:
            # Finalize current episode if exists
            if self._current_episode and self._current_episode.item_count >= self.config.min_episode_events:
                self._episodes[self._current_episode.episode_id] = self._current_episode
            
            # Start new episode
            self._current_episode = Episode(
                episode_id=str(uuid.uuid4()),
                episode_type=episode_type,
                start_time=item.timestamp,
                end_time=item.timestamp,
            )
        
        self._current_episode.add_item(item)
        self._last_item_time = item.timestamp
    
    def get_item(self, item_id: str, update_access: bool = True) -> EpisodicMemoryItem | None:
        """
        Get a memory item by ID.
        
        Args:
            item_id: Item ID
            update_access: Whether to update access time and count
            
        Returns:
            The memory item or None
        """
        item = self._items.get(item_id)
        if item and update_access:
            self._access_item(item)
        return item
    
    def _access_item(self, item: EpisodicMemoryItem) -> None:
        """Record an access to the item, strengthening the memory."""
        item.last_access_time = datetime.now()
        item.access_count += 1
        
        # Retrieval strengthening
        boost = self.config.retrieval_boost
        item.strength = min(self.config.max_strength, item.strength + boost)
    
    def rehearse(self, item_id: str) -> bool:
        """
        Rehearse a memory item, strengthening it more than retrieval.
        
        Args:
            item_id: Item to rehearse
            
        Returns:
            True if item exists and was rehearsed
        """
        item = self._items.get(item_id)
        if not item:
            return False
        
        item.rehearsal_count += 1
        item.last_access_time = datetime.now()
        
        # Rehearsal provides stronger boost
        boost = self.config.rehearsal_boost
        item.strength = min(self.config.max_strength, item.strength + boost)
        item.initial_strength = min(self.config.max_strength, item.initial_strength + boost * 0.5)
        
        return True
    
    def apply_decay(self, current_time: datetime | None = None) -> int:
        """
        Apply decay to all memory items.
        
        Args:
            current_time: Current time for decay calculation
            
        Returns:
            Number of items that dropped below forgetting threshold
        """
        now = current_time or datetime.now()
        forgotten_count = 0
        items_to_remove = []
        
        for item_id, item in self._items.items():
            new_strength = self._decay_calculator.calculate_retention(item, now)
            item.strength = new_strength
            
            if self._decay_calculator.is_forgotten(item, now):
                forgotten_count += 1
                if self.config.enable_forgetting:
                    items_to_remove.append(item_id)
        
        # Remove forgotten items
        for item_id in items_to_remove:
            del self._items[item_id]
            self._items_forgotten += 1
        
        return forgotten_count
    
    def get_strong_memories(
        self, 
        min_strength: float = 0.5,
        limit: int | None = None,
    ) -> list[EpisodicMemoryItem]:
        """
        Get memories above a strength threshold.
        
        Args:
            min_strength: Minimum strength threshold
            limit: Maximum results
            
        Returns:
            List of strong memories sorted by strength descending
        """
        strong = [
            item for item in self._items.values()
            if item.strength >= min_strength
        ]
        strong.sort(key=lambda x: x.strength, reverse=True)
        
        if limit:
            return strong[:limit]
        return strong
    
    def get_important_memories(
        self,
        min_importance: float = 0.7,
        limit: int | None = None,
    ) -> list[EpisodicMemoryItem]:
        """
        Get memories above an importance threshold.
        
        Args:
            min_importance: Minimum importance threshold
            limit: Maximum results
            
        Returns:
            List of important memories
        """
        important = [
            item for item in self._items.values()
            if item.importance >= min_importance
        ]
        important.sort(key=lambda x: (x.importance, x.strength), reverse=True)
        
        if limit:
            return important[:limit]
        return important
    
    def get_episodes(
        self,
        min_strength: float = 0.0,
        episode_type: EpisodeType | None = None,
        limit: int | None = None,
    ) -> list[Episode]:
        """
        Get episodes matching criteria.
        
        Args:
            min_strength: Minimum episode strength
            episode_type: Filter by episode type
            limit: Maximum results
            
        Returns:
            List of matching episodes
        """
        episodes = list(self._episodes.values())
        
        # Include current episode if it meets criteria
        if self._current_episode and self._current_episode.item_count >= self.config.min_episode_events:
            episodes.append(self._current_episode)
        
        # Filter by strength
        episodes = [e for e in episodes if e.get_strength() >= min_strength]
        
        # Filter by type
        if episode_type:
            episodes = [e for e in episodes if e.episode_type == episode_type]
        
        # Sort by recency
        episodes.sort(key=lambda x: x.end_time, reverse=True)
        
        if limit:
            return episodes[:limit]
        return episodes
    
    def get_episode(self, episode_id: str) -> Episode | None:
        """Get an episode by ID."""
        return self._episodes.get(episode_id)
    
    def calculate_salience(self, item: EpisodicMemoryItem) -> float:
        """
        Calculate overall salience of a memory item.
        
        Combines importance, emotional valience, and recency.
        
        Args:
            item: Memory item
            
        Returns:
            Salience score [0, 1]
        """
        config = self.config
        
        # Emotional component
        emotional = (abs(item.emotional_valence.value) / 2) * config.emotional_salience_weight
        
        # Importance component
        importance = item.importance * config.importance_weight
        
        # Recency component (based on current strength as proxy)
        recency = item.strength * config.recency_weight
        
        return emotional + importance + recency
    
    def get_salient_memories(self, limit: int = 10) -> list[tuple[EpisodicMemoryItem, float]]:
        """
        Get the most salient memories.
        
        Args:
            limit: Maximum results
            
        Returns:
            List of (item, salience_score) tuples
        """
        scored = [
            (item, self.calculate_salience(item))
            for item in self._items.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
    
    def finalize_current_episode(self, summary: str | None = None) -> Episode | None:
        """
        Finalize the current episode and store it.
        
        Args:
            summary: Optional summary for the episode
            
        Returns:
            The finalized episode or None
        """
        if not self._current_episode:
            return None
        
        if self._current_episode.item_count < self.config.min_episode_events:
            return None
        
        if summary:
            self._current_episode.summary = summary
        
        self._episodes[self._current_episode.episode_id] = self._current_episode
        episode = self._current_episode
        self._current_episode = None
        
        return episode
    
    def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        return {
            "total_items": len(self._items),
            "total_items_added": self._total_items_added,
            "items_forgotten": self._items_forgotten,
            "episode_count": len(self._episodes),
            "has_current_episode": self._current_episode is not None,
            "current_episode_items": self._current_episode.item_count if self._current_episode else 0,
            "avg_strength": (
                sum(item.strength for item in self._items.values()) / len(self._items)
                if self._items else 0
            ),
        }
    
    def __len__(self) -> int:
        """Get number of stored items."""
        return len(self._items)
    
    def __iter__(self) -> Iterator[EpisodicMemoryItem]:
        """Iterate over all items."""
        return iter(self._items.values())


def create_episodic_store(
    decay_model: DecayModel = DecayModel.EBBINGHAUS,
    enable_forgetting: bool = True,
    **config_kwargs,
) -> EpisodicMemoryStore:
    """
    Create an episodic memory store with specified settings.
    
    Args:
        decay_model: Which forgetting curve to use
        enable_forgetting: Whether to remove forgotten memories
        **config_kwargs: Additional config parameters
        
    Returns:
        Configured EpisodicMemoryStore
        
    Example:
        >>> store = create_episodic_store(
        ...     decay_model=DecayModel.POWER_LAW,
        ...     enable_forgetting=False,
        ...     forgetting_threshold=0.2,
        ... )
    """
    config = EpisodicMemoryConfig(
        decay_model=decay_model,
        enable_forgetting=enable_forgetting,
        **config_kwargs,
    )
    return EpisodicMemoryStore(config)
