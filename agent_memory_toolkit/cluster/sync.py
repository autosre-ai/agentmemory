"""
Multi-node synchronization for distributed memory.

Provides vector clocks, conflict resolution, and state synchronization
across cluster nodes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable

from .exceptions import ClusterError, SyncConflictError

logger = logging.getLogger(__name__)


# Optional Redis imports
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore


class SyncEventType(Enum):
    """Types of synchronization events."""

    MEMORY_CREATED = auto()
    MEMORY_UPDATED = auto()
    MEMORY_DELETED = auto()
    FULL_SYNC_STARTED = auto()
    FULL_SYNC_COMPLETED = auto()
    CONFLICT_DETECTED = auto()
    CONFLICT_RESOLVED = auto()
    NODE_SYNC_STARTED = auto()
    NODE_SYNC_COMPLETED = auto()


class ConflictResolution(Enum):
    """Strategies for resolving sync conflicts."""

    LAST_WRITE_WINS = auto()  # Most recent update wins
    FIRST_WRITE_WINS = auto()  # First update wins
    HIGHEST_VERSION_WINS = auto()  # Highest version number wins
    MERGE = auto()  # Attempt to merge changes
    MANUAL = auto()  # Require manual resolution
    CUSTOM = auto()  # Use custom resolver function


class SyncState(Enum):
    """State of synchronization."""

    IDLE = auto()
    SYNCING = auto()
    PAUSED = auto()
    ERROR = auto()


@dataclass
class VectorClock:
    """
    Vector clock for tracking causality across nodes.

    Each node maintains a counter that is incremented on each update.
    By comparing vector clocks, we can determine:
    - If one event happened before another
    - If events are concurrent (conflict)
    """

    clocks: dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> None:
        """Increment the clock for a node."""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def merge(self, other: VectorClock) -> VectorClock:
        """Merge with another vector clock (element-wise max)."""
        merged = VectorClock(clocks=dict(self.clocks))
        for node_id, counter in other.clocks.items():
            merged.clocks[node_id] = max(merged.clocks.get(node_id, 0), counter)
        return merged

    def is_before(self, other: VectorClock) -> bool:
        """Check if this clock happened before another."""
        if not self.clocks:
            return bool(other.clocks)

        before = False
        for node_id, counter in other.clocks.items():
            my_counter = self.clocks.get(node_id, 0)
            if my_counter > counter:
                return False
            if my_counter < counter:
                before = True

        return before

    def is_concurrent(self, other: VectorClock) -> bool:
        """Check if this clock is concurrent with another (neither before nor after)."""
        return not self.is_before(other) and not other.is_before(self)

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return dict(self.clocks)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> VectorClock:
        """Create from dictionary."""
        return cls(clocks=dict(data))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return False
        return self.clocks == other.clocks


@dataclass
class SyncEvent:
    """An event during synchronization."""

    id: str
    type: SyncEventType
    timestamp: datetime
    node_id: str
    memory_id: str | None
    data: dict[str, Any]
    vector_clock: VectorClock

    @classmethod
    def create(
        cls,
        event_type: SyncEventType,
        node_id: str,
        memory_id: str | None = None,
        data: dict[str, Any] | None = None,
        vector_clock: VectorClock | None = None,
    ) -> SyncEvent:
        """Create a new sync event."""
        return cls(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.utcnow(),
            node_id=node_id,
            memory_id=memory_id,
            data=data or {},
            vector_clock=vector_clock or VectorClock(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "node_id": self.node_id,
            "memory_id": self.memory_id,
            "data": self.data,
            "vector_clock": self.vector_clock.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncEvent:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=SyncEventType[data["type"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            node_id=data["node_id"],
            memory_id=data.get("memory_id"),
            data=data.get("data", {}),
            vector_clock=VectorClock.from_dict(data.get("vector_clock", {})),
        )


@dataclass
class SyncResult:
    """Result of a synchronization operation."""

    success: bool
    events_sent: int = 0
    events_received: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "events_sent": self.events_sent,
            "events_received": self.events_received,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


@dataclass
class SyncMetrics:
    """Metrics for synchronization performance."""

    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    total_events_sent: int = 0
    total_events_received: int = 0
    total_conflicts: int = 0
    avg_sync_duration_ms: float = 0.0
    last_sync_time: datetime | None = None

    def record_sync(self, result: SyncResult) -> None:
        """Record a sync result."""
        self.total_syncs += 1
        if result.success:
            self.successful_syncs += 1
        else:
            self.failed_syncs += 1

        self.total_events_sent += result.events_sent
        self.total_events_received += result.events_received
        self.total_conflicts += result.conflicts_detected

        # Update average duration
        if self.total_syncs == 1:
            self.avg_sync_duration_ms = result.duration_ms
        else:
            self.avg_sync_duration_ms = (
                self.avg_sync_duration_ms * (self.total_syncs - 1) + result.duration_ms
            ) / self.total_syncs

        self.last_sync_time = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_syncs": self.total_syncs,
            "successful_syncs": self.successful_syncs,
            "failed_syncs": self.failed_syncs,
            "total_events_sent": self.total_events_sent,
            "total_events_received": self.total_events_received,
            "total_conflicts": self.total_conflicts,
            "avg_sync_duration_ms": self.avg_sync_duration_ms,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
        }


@dataclass
class SyncConfig:
    """Configuration for cluster synchronization."""

    # Node identification
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cluster_name: str = "agent-memory-cluster"

    # Sync settings
    sync_interval: float = 5.0  # Seconds between sync cycles
    batch_size: int = 100  # Events per batch
    max_retries: int = 3
    retry_delay: float = 1.0

    # Conflict resolution
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS

    # Performance
    max_concurrent_syncs: int = 5
    timeout: float = 30.0

    # Event log retention
    event_retention_seconds: int = 3600  # 1 hour


# Type for conflict resolver functions
ConflictResolver = Callable[
    [dict[str, Any], dict[str, Any], VectorClock, VectorClock],
    dict[str, Any],
]


class ClusterSync:
    """
    Multi-node synchronization manager.

    Handles:
    - Event log management
    - Vector clock synchronization
    - Conflict detection and resolution
    - Peer-to-peer sync protocol

    Example:
        >>> config = SyncConfig(node_id="node-1")
        >>> sync = ClusterSync(redis_url="redis://localhost:6379", config=config)
        >>> await sync.start()
        >>>
        >>> # Record a change
        >>> await sync.record_change(
        ...     memory_id="mem-123",
        ...     change_type=SyncEventType.MEMORY_UPDATED,
        ...     data={"content": "Updated content"},
        ... )
        >>>
        >>> # Sync with peers
        >>> result = await sync.sync_with_peers()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: SyncConfig | None = None,
        conflict_resolver: ConflictResolver | None = None,
        on_event: Callable[[SyncEvent], None] | None = None,
    ):
        """
        Initialize cluster sync.

        Args:
            redis_url: Redis connection URL
            config: Sync configuration
            conflict_resolver: Custom conflict resolution function
            on_event: Callback for sync events
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required for cluster sync. "
                "Install with: pip install redis[hiredis]"
            )

        self.redis_url = redis_url
        self.config = config or SyncConfig()
        self._conflict_resolver = conflict_resolver
        self._on_event = on_event

        # State
        self._redis: redis.Redis | None = None
        self._state = SyncState.IDLE
        self._vector_clock = VectorClock()
        self._metrics = SyncMetrics()

        # Background tasks
        self._sync_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    @property
    def state(self) -> SyncState:
        """Get current sync state."""
        return self._state

    @property
    def metrics(self) -> SyncMetrics:
        """Get sync metrics."""
        return self._metrics

    @property
    def vector_clock(self) -> VectorClock:
        """Get current vector clock."""
        return self._vector_clock

    async def start(self) -> None:
        """Start the sync service."""
        self._redis = redis.from_url(
            self.redis_url,
            socket_timeout=self.config.timeout,
        )

        # Test connection
        await self._redis.ping()

        # Load vector clock from storage
        await self._load_vector_clock()

        # Start background sync
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._state = SyncState.SYNCING
        logger.info(f"Cluster sync started for node {self.config.node_id}")

    async def stop(self) -> None:
        """Stop the sync service."""
        self._state = SyncState.IDLE

        # Cancel background tasks
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Save vector clock
        await self._save_vector_clock()

        # Close Redis
        if self._redis:
            await self._redis.close()

        logger.info(f"Cluster sync stopped for node {self.config.node_id}")

    async def __aenter__(self) -> ClusterSync:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # ==================== Event Management ====================

    async def record_change(
        self,
        memory_id: str,
        change_type: SyncEventType,
        data: dict[str, Any],
    ) -> SyncEvent:
        """
        Record a memory change for synchronization.

        Args:
            memory_id: ID of the affected memory
            change_type: Type of change
            data: Change data (new content, metadata, etc.)

        Returns:
            The created SyncEvent
        """
        self._ensure_connected()

        # Increment vector clock
        self._vector_clock.increment(self.config.node_id)

        # Create event
        event = SyncEvent.create(
            event_type=change_type,
            node_id=self.config.node_id,
            memory_id=memory_id,
            data=data,
            vector_clock=VectorClock.from_dict(self._vector_clock.to_dict()),
        )

        # Store in event log
        await self._append_event(event)

        # Notify callback
        if self._on_event:
            self._on_event(event)

        return event

    async def get_events_since(
        self,
        since_clock: VectorClock | None = None,
        limit: int = 100,
    ) -> list[SyncEvent]:
        """
        Get events since a given vector clock.

        Args:
            since_clock: Get events after this clock (None for all)
            limit: Maximum events to return

        Returns:
            List of events
        """
        self._ensure_connected()

        key = self._event_log_key()
        raw_events = await self._redis.lrange(key, 0, -1)

        events = []
        for raw in raw_events:
            if isinstance(raw, bytes):
                raw = raw.decode()
            event = SyncEvent.from_dict(json.loads(raw))

            # Filter by vector clock
            if since_clock is None or event.vector_clock.is_concurrent(since_clock):
                events.append(event)
                if len(events) >= limit:
                    break

        return events

    async def apply_remote_events(
        self,
        events: list[SyncEvent],
        memory_getter: Callable[[str], Any] | None = None,
        memory_setter: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> SyncResult:
        """
        Apply events from remote nodes.

        Args:
            events: Events to apply
            memory_getter: Function to get current memory state
            memory_setter: Function to set memory state

        Returns:
            SyncResult with details
        """
        self._ensure_connected()

        start_time = time.time()
        conflicts_detected = 0
        conflicts_resolved = 0
        errors = []

        for event in events:
            try:
                # Check for conflicts using vector clock
                if event.vector_clock.is_concurrent(self._vector_clock):
                    conflicts_detected += 1

                    # Resolve conflict
                    if memory_getter and memory_setter and event.memory_id:
                        try:
                            local_state = await memory_getter(event.memory_id)
                            resolved = await self._resolve_conflict(
                                local_state,
                                event.data,
                                self._vector_clock,
                                event.vector_clock,
                            )
                            await memory_setter(event.memory_id, resolved)
                            conflicts_resolved += 1
                        except Exception as e:
                            errors.append(f"Conflict resolution failed: {e}")

                # Merge vector clocks
                self._vector_clock = self._vector_clock.merge(event.vector_clock)

                # Emit event notification
                if self._on_event:
                    self._on_event(event)

            except Exception as e:
                errors.append(f"Failed to apply event {event.id}: {e}")

        duration_ms = (time.time() - start_time) * 1000

        result = SyncResult(
            success=len(errors) == 0,
            events_received=len(events),
            conflicts_detected=conflicts_detected,
            conflicts_resolved=conflicts_resolved,
            errors=errors,
            duration_ms=duration_ms,
        )

        self._metrics.record_sync(result)
        return result

    # ==================== Peer Synchronization ====================

    async def sync_with_peers(self) -> SyncResult:
        """
        Synchronize with all peer nodes.

        Returns:
            Combined SyncResult
        """
        self._ensure_connected()

        start_time = time.time()
        total_sent = 0
        total_received = 0
        total_conflicts = 0
        errors = []

        # Get peer nodes
        peer_nodes = await self._get_peer_nodes()

        # Sync with each peer
        for peer_id in peer_nodes:
            if peer_id == self.config.node_id:
                continue

            try:
                result = await self._sync_with_peer(peer_id)
                total_sent += result.events_sent
                total_received += result.events_received
                total_conflicts += result.conflicts_detected
                errors.extend(result.errors)
            except Exception as e:
                errors.append(f"Sync with {peer_id} failed: {e}")

        duration_ms = (time.time() - start_time) * 1000

        result = SyncResult(
            success=len(errors) == 0,
            events_sent=total_sent,
            events_received=total_received,
            conflicts_detected=total_conflicts,
            errors=errors,
            duration_ms=duration_ms,
        )

        self._metrics.record_sync(result)
        return result

    async def _sync_with_peer(self, peer_id: str) -> SyncResult:
        """Sync with a specific peer."""
        start_time = time.time()

        # Get peer's clock
        peer_clock = await self._get_peer_clock(peer_id)

        # Get our events that peer doesn't have
        our_events = await self._get_events_for_peer(peer_clock)

        # Send our events to peer's inbox
        events_sent = 0
        for event in our_events:
            inbox_key = self._peer_inbox_key(peer_id)
            await self._redis.lpush(inbox_key, json.dumps(event.to_dict()))
            events_sent += 1

        # Get events from our inbox
        my_inbox_key = self._peer_inbox_key(self.config.node_id)
        events_received = 0
        conflicts = 0

        while True:
            raw = await self._redis.rpop(my_inbox_key)
            if raw is None:
                break

            if isinstance(raw, bytes):
                raw = raw.decode()

            event = SyncEvent.from_dict(json.loads(raw))
            events_received += 1

            if event.vector_clock.is_concurrent(self._vector_clock):
                conflicts += 1

            self._vector_clock = self._vector_clock.merge(event.vector_clock)

        duration_ms = (time.time() - start_time) * 1000

        return SyncResult(
            success=True,
            events_sent=events_sent,
            events_received=events_received,
            conflicts_detected=conflicts,
            duration_ms=duration_ms,
        )

    async def _get_events_for_peer(self, peer_clock: VectorClock) -> list[SyncEvent]:
        """Get events that peer doesn't have."""
        events = await self.get_events_since(peer_clock, limit=self.config.batch_size)
        return events

    async def _get_peer_clock(self, peer_id: str) -> VectorClock:
        """Get a peer's vector clock."""
        key = f"{self.config.cluster_name}:clock:{peer_id}"
        data = await self._redis.get(key)

        if data:
            if isinstance(data, bytes):
                data = data.decode()
            return VectorClock.from_dict(json.loads(data))

        return VectorClock()

    async def _get_peer_nodes(self) -> list[str]:
        """Get list of peer node IDs."""
        key = f"{self.config.cluster_name}:nodes"
        nodes = await self._redis.smembers(key)
        return [n.decode() if isinstance(n, bytes) else n for n in nodes]

    # ==================== Conflict Resolution ====================

    async def _resolve_conflict(
        self,
        local_state: dict[str, Any],
        remote_state: dict[str, Any],
        local_clock: VectorClock,
        remote_clock: VectorClock,
    ) -> dict[str, Any]:
        """
        Resolve a conflict between local and remote state.

        Args:
            local_state: Current local state
            remote_state: Incoming remote state
            local_clock: Local vector clock
            remote_clock: Remote vector clock

        Returns:
            Resolved state
        """
        resolution = self.config.conflict_resolution

        if resolution == ConflictResolution.CUSTOM and self._conflict_resolver:
            return self._conflict_resolver(
                local_state, remote_state, local_clock, remote_clock
            )

        if resolution == ConflictResolution.LAST_WRITE_WINS:
            # Compare timestamps
            local_time = local_state.get("updated_at", "")
            remote_time = remote_state.get("updated_at", "")
            return remote_state if remote_time > local_time else local_state

        if resolution == ConflictResolution.FIRST_WRITE_WINS:
            local_time = local_state.get("created_at", "")
            remote_time = remote_state.get("created_at", "")
            return local_state if local_time <= remote_time else remote_state

        if resolution == ConflictResolution.HIGHEST_VERSION_WINS:
            local_version = local_state.get("version", 0)
            remote_version = remote_state.get("version", 0)
            return remote_state if remote_version > local_version else local_state

        if resolution == ConflictResolution.MERGE:
            # Simple merge - remote values override local
            merged = dict(local_state)
            merged.update(remote_state)
            merged["version"] = max(
                local_state.get("version", 0),
                remote_state.get("version", 0),
            ) + 1
            return merged

        if resolution == ConflictResolution.MANUAL:
            raise SyncConflictError(
                "Manual conflict resolution required",
                memory_id=local_state.get("id", "unknown"),
                local_version=local_state.get("version", 0),
                remote_version=remote_state.get("version", 0),
            )

        # Default to last write wins
        return remote_state

    # ==================== Internal Methods ====================

    def _ensure_connected(self) -> None:
        """Ensure Redis is connected."""
        if self._redis is None:
            raise ClusterError("Sync service not started")

    def _event_log_key(self) -> str:
        """Get Redis key for event log."""
        return f"{self.config.cluster_name}:events:{self.config.node_id}"

    def _peer_inbox_key(self, peer_id: str) -> str:
        """Get Redis key for peer inbox."""
        return f"{self.config.cluster_name}:inbox:{peer_id}"

    async def _append_event(self, event: SyncEvent) -> None:
        """Append event to log."""
        key = self._event_log_key()
        await self._redis.lpush(key, json.dumps(event.to_dict()))

        # Trim to max size
        await self._redis.ltrim(key, 0, self.config.batch_size * 10)

    async def _load_vector_clock(self) -> None:
        """Load vector clock from storage."""
        key = f"{self.config.cluster_name}:clock:{self.config.node_id}"
        data = await self._redis.get(key)

        if data:
            if isinstance(data, bytes):
                data = data.decode()
            self._vector_clock = VectorClock.from_dict(json.loads(data))
        else:
            self._vector_clock = VectorClock()

    async def _save_vector_clock(self) -> None:
        """Save vector clock to storage."""
        key = f"{self.config.cluster_name}:clock:{self.config.node_id}"
        await self._redis.set(key, json.dumps(self._vector_clock.to_dict()))

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._state == SyncState.SYNCING:
            try:
                await asyncio.sleep(self.config.sync_interval)
                await self.sync_with_peers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Sync loop error: {e}")
                self._state = SyncState.ERROR

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._state != SyncState.IDLE:
            try:
                await asyncio.sleep(self.config.event_retention_seconds / 10)

                # Trim old events
                key = self._event_log_key()
                await self._redis.ltrim(key, 0, self.config.batch_size * 100)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cleanup loop error: {e}")
