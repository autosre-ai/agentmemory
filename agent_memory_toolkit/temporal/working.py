"""Working memory system with capacity limits and attention mechanisms.

This module implements a cognitively-inspired working memory system:
- Limited capacity (Miller's 7±2 chunks)
- Attention-based prioritization
- Rehearsal to maintain items
- Displacement of low-priority items
- Context maintenance for active task
- Integration with long-term memory
"""

from __future__ import annotations

import logging
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Generic, Iterator, TypeVar
import uuid

logger = logging.getLogger(__name__)


class DisplacementStrategy(Enum):
    """Strategy for removing items when capacity is exceeded."""
    OLDEST = "oldest"           # Remove oldest items first
    LOWEST_PRIORITY = "lowest_priority"  # Remove lowest priority
    LEAST_RECENT = "least_recent"  # Remove least recently accessed
    COMBINED = "combined"       # Weighted combination


class AttentionMode(Enum):
    """Mode of attention allocation."""
    FOCUSED = "focused"         # Strong focus on few items
    DISTRIBUTED = "distributed" # Distributed attention
    AUTOMATIC = "automatic"     # Based on task demands


@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory."""
    
    # Capacity
    capacity: int = 7  # Miller's magic number
    soft_capacity: int = 9  # Allow slight overflow with decay
    
    # Attention
    attention_mode: AttentionMode = AttentionMode.AUTOMATIC
    attention_decay_rate: float = 0.1  # Per second
    attention_refresh_threshold: float = 0.3
    
    # Displacement
    displacement_strategy: DisplacementStrategy = DisplacementStrategy.COMBINED
    priority_weight: float = 0.4
    recency_weight: float = 0.3
    relevance_weight: float = 0.3
    
    # Rehearsal
    auto_rehearsal: bool = True
    rehearsal_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    max_rehearsal_items: int = 3
    
    # Decay
    item_decay_rate: float = 0.05  # Per second of inattention
    min_activation: float = 0.1


@dataclass
class WorkingMemoryItem:
    """An item in working memory."""
    item_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Activation and attention
    activation: float = 1.0  # Current activation level [0, 1]
    priority: float = 0.5  # Base priority [0, 1]
    attention: float = 0.0  # Current attention allocated [0, 1]
    
    # Tracking
    last_access: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    rehearsal_count: int = 0
    
    # Context linkage
    context_tags: list[str] = field(default_factory=list)
    linked_items: list[str] = field(default_factory=list)
    
    # Source tracking
    source: str = "direct"  # 'direct', 'retrieved', 'inferred'
    source_memory_id: str | None = None  # Link to long-term memory
    
    def __lt__(self, other: "WorkingMemoryItem") -> bool:
        """For heap operations - lower activation = lower priority in heap."""
        return self.activation < other.activation
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "activation": self.activation,
            "priority": self.priority,
            "attention": self.attention,
            "last_access": self.last_access.isoformat(),
            "access_count": self.access_count,
            "rehearsal_count": self.rehearsal_count,
            "context_tags": self.context_tags,
            "linked_items": self.linked_items,
            "source": self.source,
            "source_memory_id": self.source_memory_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemoryItem":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            metadata=data.get("metadata", {}),
            activation=data.get("activation", 1.0),
            priority=data.get("priority", 0.5),
            attention=data.get("attention", 0.0),
            last_access=datetime.fromisoformat(data["last_access"]) if data.get("last_access") else datetime.now(),
            access_count=data.get("access_count", 0),
            rehearsal_count=data.get("rehearsal_count", 0),
            context_tags=data.get("context_tags", []),
            linked_items=data.get("linked_items", []),
            source=data.get("source", "direct"),
            source_memory_id=data.get("source_memory_id"),
        )


@dataclass
class WorkingMemoryContext:
    """Context for the current working memory state."""
    context_id: str
    name: str
    description: str | None = None
    created: datetime = field(default_factory=datetime.now)
    
    # Active goals/tasks
    active_goals: list[str] = field(default_factory=list)
    
    # Context metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Items associated with this context
    item_ids: list[str] = field(default_factory=list)


@dataclass
class DisplacementResult:
    """Result of a displacement operation."""
    displaced_items: list[WorkingMemoryItem]
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class AttentionAllocator:
    """Allocates attention across working memory items."""
    
    def __init__(self, config: WorkingMemoryConfig):
        self.config = config
    
    def allocate(
        self,
        items: list[WorkingMemoryItem],
        focus_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Allocate attention across items.
        
        Args:
            items: Items to allocate attention to
            focus_ids: IDs of items to focus on (if any)
            
        Returns:
            Dict mapping item_id to attention allocation [0, 1]
        """
        if not items:
            return {}
        
        mode = self.config.attention_mode
        
        if mode == AttentionMode.FOCUSED and focus_ids:
            return self._focused_allocation(items, focus_ids)
        elif mode == AttentionMode.DISTRIBUTED:
            return self._distributed_allocation(items)
        else:  # AUTOMATIC
            return self._automatic_allocation(items, focus_ids)
    
    def _focused_allocation(
        self,
        items: list[WorkingMemoryItem],
        focus_ids: list[str],
    ) -> dict[str, float]:
        """Focused attention on specific items."""
        allocations = {}
        focus_set = set(focus_ids)
        
        # Give 80% attention to focused items, 20% to rest
        focused = [i for i in items if i.item_id in focus_set]
        unfocused = [i for i in items if i.item_id not in focus_set]
        
        focus_share = 0.8 / max(1, len(focused))
        rest_share = 0.2 / max(1, len(unfocused))
        
        for item in focused:
            allocations[item.item_id] = focus_share
        for item in unfocused:
            allocations[item.item_id] = rest_share
        
        return allocations
    
    def _distributed_allocation(
        self,
        items: list[WorkingMemoryItem],
    ) -> dict[str, float]:
        """Evenly distributed attention."""
        share = 1.0 / len(items)
        return {item.item_id: share for item in items}
    
    def _automatic_allocation(
        self,
        items: list[WorkingMemoryItem],
        focus_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Automatic allocation based on activation and priority."""
        if not items:
            return {}
        
        # Calculate raw weights
        weights = {}
        for item in items:
            base = item.activation * item.priority
            if focus_ids and item.item_id in focus_ids:
                base *= 2.0
            weights[item.item_id] = base
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        else:
            return self._distributed_allocation(items)


class WorkingMemory:
    """
    Working memory system with limited capacity and attention mechanisms.
    
    Implements cognitive psychology concepts:
    - Limited capacity (7±2 items)
    - Attention-based prioritization
    - Decay without rehearsal
    - Displacement when full
    
    Example:
        >>> from agent_memory_toolkit.temporal import WorkingMemory
        >>> 
        >>> wm = WorkingMemory()
        >>> 
        >>> # Add items
        >>> item = wm.add("User wants to book a flight", priority=0.8)
        >>> wm.add("Departure: New York", context_tags=["flight_booking"])
        >>> wm.add("Destination: London", context_tags=["flight_booking"])
        >>> 
        >>> # Focus attention
        >>> wm.focus(item.item_id)
        >>> 
        >>> # Get active items
        >>> active = wm.get_active(min_activation=0.5)
        >>> 
        >>> # Apply decay
        >>> wm.decay()
    """
    
    def __init__(self, config: WorkingMemoryConfig | None = None):
        """
        Initialize working memory.
        
        Args:
            config: Configuration settings
        """
        self.config = config or WorkingMemoryConfig()
        self._attention = AttentionAllocator(self.config)
        
        # Storage
        self._items: dict[str, WorkingMemoryItem] = {}
        self._contexts: dict[str, WorkingMemoryContext] = {}
        
        # Current state
        self._current_context: WorkingMemoryContext | None = None
        self._focused_ids: list[str] = []
        
        # Tracking
        self._displaced: list[DisplacementResult] = []
        self._last_decay: datetime = datetime.now()
        self._last_rehearsal: datetime = datetime.now()
        
        # Callbacks
        self._on_displacement: Callable[[DisplacementResult], None] | None = None
    
    def add(
        self,
        content: str,
        priority: float = 0.5,
        context_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "direct",
        source_memory_id: str | None = None,
    ) -> WorkingMemoryItem:
        """
        Add an item to working memory.
        
        If capacity is exceeded, lowest activation items are displaced.
        
        Args:
            content: Item content
            priority: Base priority [0, 1]
            context_tags: Tags for context association
            metadata: Additional metadata
            source: Source of the item
            source_memory_id: Link to long-term memory
            
        Returns:
            The added item
        """
        item = WorkingMemoryItem(
            item_id=str(uuid.uuid4()),
            content=content,
            priority=priority,
            context_tags=context_tags or [],
            metadata=metadata or {},
            source=source,
            source_memory_id=source_memory_id,
        )
        
        # Check capacity and displace if needed
        if len(self._items) >= self.config.capacity:
            self._displace(1)
        
        self._items[item.item_id] = item
        
        # Update context
        if self._current_context:
            self._current_context.item_ids.append(item.item_id)
        
        # Reallocate attention
        self._reallocate_attention()
        
        logger.debug(f"Added working memory item {item.item_id}")
        return item
    
    def get(self, item_id: str) -> WorkingMemoryItem | None:
        """
        Get an item by ID, refreshing its activation.
        
        Args:
            item_id: Item ID
            
        Returns:
            The item or None
        """
        item = self._items.get(item_id)
        if item:
            self._access(item)
        return item
    
    def _access(self, item: WorkingMemoryItem) -> None:
        """Record access to an item."""
        item.last_access = datetime.now()
        item.access_count += 1
        # Boost activation on access
        item.activation = min(1.0, item.activation + 0.2)
    
    def remove(self, item_id: str) -> bool:
        """
        Remove an item from working memory.
        
        Args:
            item_id: Item to remove
            
        Returns:
            True if removed
        """
        if item_id in self._items:
            del self._items[item_id]
            if item_id in self._focused_ids:
                self._focused_ids.remove(item_id)
            return True
        return False
    
    def focus(self, *item_ids: str) -> None:
        """
        Focus attention on specific items.
        
        Args:
            *item_ids: IDs of items to focus on
        """
        self._focused_ids = [
            id for id in item_ids 
            if id in self._items
        ]
        self._reallocate_attention()
        
        # Boost activation of focused items
        for item_id in self._focused_ids:
            item = self._items.get(item_id)
            if item:
                item.activation = min(1.0, item.activation + 0.3)
    
    def unfocus(self) -> None:
        """Remove focus from all items."""
        self._focused_ids = []
        self._reallocate_attention()
    
    def _reallocate_attention(self) -> None:
        """Reallocate attention across all items."""
        items = list(self._items.values())
        allocations = self._attention.allocate(items, self._focused_ids)
        
        for item_id, attention in allocations.items():
            if item_id in self._items:
                self._items[item_id].attention = attention
    
    def decay(self, elapsed: timedelta | None = None) -> int:
        """
        Apply decay to all items based on elapsed time and attention.
        
        Args:
            elapsed: Time since last decay (auto-calculated if None)
            
        Returns:
            Number of items that fell below minimum activation
        """
        now = datetime.now()
        if elapsed is None:
            elapsed = now - self._last_decay
        self._last_decay = now
        
        elapsed_seconds = elapsed.total_seconds()
        decay_rate = self.config.item_decay_rate
        min_activation = self.config.min_activation
        
        low_activation_count = 0
        
        for item in self._items.values():
            # Items with attention decay slower
            effective_decay = decay_rate * (1 - item.attention * 0.5)
            
            # Apply decay
            item.activation = max(
                min_activation,
                item.activation - (effective_decay * elapsed_seconds)
            )
            
            if item.activation <= min_activation:
                low_activation_count += 1
        
        # Auto-rehearsal if enabled
        if self.config.auto_rehearsal:
            time_since_rehearsal = now - self._last_rehearsal
            if time_since_rehearsal >= self.config.rehearsal_interval:
                self._auto_rehearse()
                self._last_rehearsal = now
        
        return low_activation_count
    
    def _auto_rehearse(self) -> None:
        """Automatically rehearse important items."""
        # Sort by priority * activation
        items = sorted(
            self._items.values(),
            key=lambda i: i.priority * i.activation,
            reverse=True,
        )
        
        # Rehearse top items
        for item in items[:self.config.max_rehearsal_items]:
            self.rehearse(item.item_id)
    
    def rehearse(self, item_id: str) -> bool:
        """
        Rehearse an item, boosting its activation.
        
        Args:
            item_id: Item to rehearse
            
        Returns:
            True if item exists
        """
        item = self._items.get(item_id)
        if not item:
            return False
        
        item.activation = min(1.0, item.activation + 0.3)
        item.rehearsal_count += 1
        item.last_access = datetime.now()
        
        return True
    
    def _displace(self, count: int) -> DisplacementResult:
        """
        Displace items to make room.
        
        Args:
            count: Number of items to displace
            
        Returns:
            DisplacementResult with displaced items
        """
        strategy = self.config.displacement_strategy
        
        if strategy == DisplacementStrategy.OLDEST:
            items = sorted(self._items.values(), key=lambda i: i.timestamp)
        elif strategy == DisplacementStrategy.LOWEST_PRIORITY:
            items = sorted(self._items.values(), key=lambda i: i.priority)
        elif strategy == DisplacementStrategy.LEAST_RECENT:
            items = sorted(self._items.values(), key=lambda i: i.last_access)
        else:  # COMBINED
            # Weighted score (lower = more likely to be displaced)
            def displacement_score(item: WorkingMemoryItem) -> float:
                config = self.config
                return (
                    item.priority * config.priority_weight +
                    item.activation * config.relevance_weight +
                    (1.0 / (1 + (datetime.now() - item.last_access).total_seconds())) * config.recency_weight
                )
            items = sorted(self._items.values(), key=displacement_score)
        
        displaced = []
        for item in items[:count]:
            displaced.append(item)
            del self._items[item.item_id]
            if item.item_id in self._focused_ids:
                self._focused_ids.remove(item.item_id)
        
        result = DisplacementResult(
            displaced_items=displaced,
            reason=f"Capacity exceeded, used {strategy.value} strategy",
        )
        
        self._displaced.append(result)
        
        if self._on_displacement:
            self._on_displacement(result)
        
        return result
    
    def get_active(
        self,
        min_activation: float = 0.0,
        limit: int | None = None,
    ) -> list[WorkingMemoryItem]:
        """
        Get active items above an activation threshold.
        
        Args:
            min_activation: Minimum activation level
            limit: Maximum results
            
        Returns:
            List of active items sorted by activation
        """
        active = [
            item for item in self._items.values()
            if item.activation >= min_activation
        ]
        active.sort(key=lambda i: i.activation, reverse=True)
        
        if limit:
            return active[:limit]
        return active
    
    def get_by_context(self, *tags: str) -> list[WorkingMemoryItem]:
        """
        Get items matching context tags.
        
        Args:
            *tags: Context tags to match (any)
            
        Returns:
            Items with any matching tag
        """
        tag_set = set(tags)
        return [
            item for item in self._items.values()
            if tag_set.intersection(item.context_tags)
        ]
    
    def set_context(
        self,
        name: str,
        description: str | None = None,
        goals: list[str] | None = None,
    ) -> WorkingMemoryContext:
        """
        Set the current working memory context.
        
        Args:
            name: Context name
            description: Context description
            goals: Active goals in this context
            
        Returns:
            The created context
        """
        context = WorkingMemoryContext(
            context_id=str(uuid.uuid4()),
            name=name,
            description=description,
            active_goals=goals or [],
        )
        
        self._current_context = context
        self._contexts[context.context_id] = context
        
        return context
    
    def get_context(self) -> WorkingMemoryContext | None:
        """Get the current context."""
        return self._current_context
    
    def clear_context(self) -> None:
        """Clear the current context."""
        self._current_context = None
    
    def snapshot(self) -> dict[str, Any]:
        """
        Get a snapshot of current working memory state.
        
        Returns:
            Dictionary representing current state
        """
        return {
            "items": [item.to_dict() for item in self._items.values()],
            "focused_ids": self._focused_ids.copy(),
            "current_context": self._current_context.name if self._current_context else None,
            "capacity_used": len(self._items),
            "capacity": self.config.capacity,
            "avg_activation": (
                sum(i.activation for i in self._items.values()) / len(self._items)
                if self._items else 0
            ),
        }
    
    def clear(self) -> list[WorkingMemoryItem]:
        """
        Clear all items from working memory.
        
        Returns:
            List of cleared items
        """
        items = list(self._items.values())
        self._items.clear()
        self._focused_ids.clear()
        return items
    
    def on_displacement(
        self, 
        callback: Callable[[DisplacementResult], None],
    ) -> None:
        """
        Set callback for when items are displaced.
        
        Useful for saving displaced items to long-term memory.
        
        Args:
            callback: Function called with DisplacementResult
        """
        self._on_displacement = callback
    
    def get_displaced(self, limit: int = 10) -> list[DisplacementResult]:
        """Get recent displacement events."""
        return self._displaced[-limit:]
    
    @property
    def size(self) -> int:
        """Current number of items."""
        return len(self._items)
    
    @property
    def capacity(self) -> int:
        """Maximum capacity."""
        return self.config.capacity
    
    @property
    def is_full(self) -> bool:
        """Whether at capacity."""
        return len(self._items) >= self.config.capacity
    
    def __len__(self) -> int:
        """Get number of items."""
        return len(self._items)
    
    def __iter__(self) -> Iterator[WorkingMemoryItem]:
        """Iterate over items."""
        return iter(self._items.values())
    
    def __contains__(self, item_id: str) -> bool:
        """Check if item is in working memory."""
        return item_id in self._items


def create_working_memory(
    capacity: int = 7,
    auto_rehearsal: bool = True,
    displacement_strategy: DisplacementStrategy = DisplacementStrategy.COMBINED,
    **config_kwargs,
) -> WorkingMemory:
    """
    Create a working memory instance with specified settings.
    
    Args:
        capacity: Maximum item capacity
        auto_rehearsal: Whether to auto-rehearse important items
        displacement_strategy: How to choose items for displacement
        **config_kwargs: Additional config parameters
        
    Returns:
        Configured WorkingMemory instance
        
    Example:
        >>> wm = create_working_memory(
        ...     capacity=5,
        ...     auto_rehearsal=True,
        ...     displacement_strategy=DisplacementStrategy.LOWEST_PRIORITY,
        ... )
    """
    config = WorkingMemoryConfig(
        capacity=capacity,
        auto_rehearsal=auto_rehearsal,
        displacement_strategy=displacement_strategy,
        **config_kwargs,
    )
    return WorkingMemory(config)
