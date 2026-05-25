"""
Memory replication across cluster nodes.

Provides configurable replication strategies for durability and
read performance in distributed memory clusters.
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
from typing import Any, Callable, AsyncIterator

from .exceptions import ReplicationError, QuorumNotReachedError, ClusterError

logger = logging.getLogger(__name__)


# Optional Redis imports
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore


class ReplicationStrategy(Enum):
    """Strategies for memory replication."""

    SYNC = auto()  # Synchronous replication (wait for all replicas)
    ASYNC = auto()  # Asynchronous replication (don't wait)
    SEMI_SYNC = auto()  # Wait for quorum, async for rest
    CHAIN = auto()  # Chain replication (primary -> replica1 -> replica2)


class ReplicationState(Enum):
    """State of a replica."""

    SYNCING = auto()  # Catching up with primary
    IN_SYNC = auto()  # Fully synchronized
    LAGGING = auto()  # Behind but recoverable
    FAILED = auto()  # Failed, needs rebuild


class ReplicationEventType(Enum):
    """Types of replication events."""

    REPLICA_ADDED = auto()
    REPLICA_REMOVED = auto()
    REPLICATION_STARTED = auto()
    REPLICATION_COMPLETED = auto()
    REPLICATION_FAILED = auto()
    REPLICA_CAUGHT_UP = auto()
    REPLICA_FELL_BEHIND = auto()
    FAILOVER_STARTED = auto()
    FAILOVER_COMPLETED = auto()


@dataclass
class ReplicationEvent:
    """An event during replication."""

    id: str
    type: ReplicationEventType
    timestamp: datetime
    source_node: str
    target_node: str | None
    memory_id: str | None
    data: dict[str, Any]

    @classmethod
    def create(
        cls,
        event_type: ReplicationEventType,
        source_node: str,
        target_node: str | None = None,
        memory_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> ReplicationEvent:
        """Create a new replication event."""
        return cls(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.utcnow(),
            source_node=source_node,
            target_node=target_node,
            memory_id=memory_id,
            data=data or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "source_node": self.source_node,
            "target_node": self.target_node,
            "memory_id": self.memory_id,
            "data": self.data,
        }


@dataclass
class ReplicaInfo:
    """Information about a replica node."""

    node_id: str
    state: ReplicationState
    lag_bytes: int = 0
    lag_messages: int = 0
    last_sync_time: datetime | None = None
    last_ack_time: datetime | None = None
    sync_position: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "state": self.state.name,
            "lag_bytes": self.lag_bytes,
            "lag_messages": self.lag_messages,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "last_ack_time": self.last_ack_time.isoformat() if self.last_ack_time else None,
            "sync_position": self.sync_position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplicaInfo:
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            state=ReplicationState[data["state"]],
            lag_bytes=data.get("lag_bytes", 0),
            lag_messages=data.get("lag_messages", 0),
            last_sync_time=(
                datetime.fromisoformat(data["last_sync_time"])
                if data.get("last_sync_time")
                else None
            ),
            last_ack_time=(
                datetime.fromisoformat(data["last_ack_time"])
                if data.get("last_ack_time")
                else None
            ),
            sync_position=data.get("sync_position", 0),
        )


@dataclass
class ReplicationConfig:
    """Configuration for memory replication."""

    # Node identification
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cluster_name: str = "agent-memory-cluster"

    # Replication settings
    replication_factor: int = 2  # Total copies (including primary)
    strategy: ReplicationStrategy = ReplicationStrategy.SEMI_SYNC
    min_sync_replicas: int = 1  # Minimum replicas to ack before commit

    # Performance
    batch_size: int = 100
    max_lag_messages: int = 1000  # Max messages before replica is LAGGING
    max_lag_seconds: float = 30.0  # Max seconds before replica is LAGGING

    # Timeouts
    ack_timeout: float = 5.0  # Seconds to wait for replica ack
    sync_timeout: float = 60.0  # Seconds to wait for full sync
    heartbeat_interval: float = 1.0

    # Failover
    auto_failover: bool = True
    failover_timeout: float = 10.0


@dataclass
class ReplicationMetrics:
    """Metrics for replication performance."""

    total_replicated: int = 0
    total_acks: int = 0
    total_failures: int = 0
    avg_replication_latency_ms: float = 0.0
    current_lag_messages: int = 0
    replicas_in_sync: int = 0
    replicas_lagging: int = 0
    replicas_failed: int = 0
    last_replication_time: datetime | None = None

    def record_replication(self, latency_ms: float, success: bool = True) -> None:
        """Record a replication attempt."""
        self.total_replicated += 1
        if success:
            self.total_acks += 1
        else:
            self.total_failures += 1

        # Update average latency
        if self.total_replicated == 1:
            self.avg_replication_latency_ms = latency_ms
        else:
            self.avg_replication_latency_ms = (
                self.avg_replication_latency_ms * (self.total_replicated - 1) + latency_ms
            ) / self.total_replicated

        self.last_replication_time = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_replicated": self.total_replicated,
            "total_acks": self.total_acks,
            "total_failures": self.total_failures,
            "avg_replication_latency_ms": self.avg_replication_latency_ms,
            "current_lag_messages": self.current_lag_messages,
            "replicas_in_sync": self.replicas_in_sync,
            "replicas_lagging": self.replicas_lagging,
            "replicas_failed": self.replicas_failed,
            "last_replication_time": (
                self.last_replication_time.isoformat()
                if self.last_replication_time
                else None
            ),
        }


class ReplicationManager:
    """
    Manager for memory replication across cluster nodes.

    Features:
    - Configurable replication strategies (sync, async, semi-sync)
    - Automatic replica health monitoring
    - Quorum-based writes
    - Automatic failover support
    - Replication lag tracking

    Example:
        >>> config = ReplicationConfig(
        ...     node_id="node-1",
        ...     replication_factor=3,
        ...     strategy=ReplicationStrategy.SEMI_SYNC,
        ... )
        >>> manager = ReplicationManager(
        ...     redis_url="redis://localhost:6379",
        ...     config=config,
        ... )
        >>> await manager.start()
        >>>
        >>> # Replicate a memory
        >>> acks = await manager.replicate(memory_data)
        >>> if acks >= config.min_sync_replicas:
        ...     print("Memory safely replicated")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: ReplicationConfig | None = None,
        on_event: Callable[[ReplicationEvent], None] | None = None,
    ):
        """
        Initialize replication manager.

        Args:
            redis_url: Redis connection URL
            config: Replication configuration
            on_event: Callback for replication events
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required for replication. "
                "Install with: pip install redis[hiredis]"
            )

        self.redis_url = redis_url
        self.config = config or ReplicationConfig()
        self._on_event = on_event

        # State
        self._redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._replicas: dict[str, ReplicaInfo] = {}
        self._metrics = ReplicationMetrics()
        self._position = 0  # Current replication position

        # Background tasks
        self._listener_task: asyncio.Task | None = None
        self._health_task: asyncio.Task | None = None
        self._running = False

    @property
    def metrics(self) -> ReplicationMetrics:
        """Get replication metrics."""
        return self._metrics

    @property
    def replicas(self) -> dict[str, ReplicaInfo]:
        """Get replica information."""
        return dict(self._replicas)

    @property
    def in_sync_count(self) -> int:
        """Get count of in-sync replicas."""
        return sum(1 for r in self._replicas.values() if r.state == ReplicationState.IN_SYNC)

    async def start(self) -> None:
        """Start replication manager."""
        self._redis = redis.from_url(
            self.redis_url,
            socket_timeout=self.config.ack_timeout,
        )

        # Test connection
        await self._redis.ping()

        self._running = True

        # Register as replica source
        await self._register_node()

        # Start listener for incoming replications
        self._listener_task = asyncio.create_task(self._listen_for_replications())

        # Start health checker
        self._health_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Replication manager started for node {self.config.node_id}")

    async def stop(self) -> None:
        """Stop replication manager."""
        self._running = False

        # Unregister
        await self._unregister_node()

        # Cancel background tasks
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Close pubsub
        if self._pubsub:
            await self._pubsub.close()

        # Close Redis
        if self._redis:
            await self._redis.close()

        logger.info(f"Replication manager stopped for node {self.config.node_id}")

    async def __aenter__(self) -> ReplicationManager:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # ==================== Replication Operations ====================

    async def replicate(
        self,
        memory_data: dict[str, Any],
        wait_for_acks: bool | None = None,
    ) -> int:
        """
        Replicate a memory to replica nodes.

        Args:
            memory_data: Memory data to replicate
            wait_for_acks: Whether to wait for acknowledgments

        Returns:
            Number of successful acknowledgments

        Raises:
            QuorumNotReachedError: If quorum cannot be reached
        """
        self._ensure_connected()

        # Determine if we should wait based on strategy
        if wait_for_acks is None:
            wait_for_acks = self.config.strategy in (
                ReplicationStrategy.SYNC,
                ReplicationStrategy.SEMI_SYNC,
            )

        start_time = time.time()
        self._position += 1

        # Prepare replication message
        message = {
            "type": "replicate",
            "source_node": self.config.node_id,
            "position": self._position,
            "timestamp": datetime.utcnow().isoformat(),
            "memory": memory_data,
        }

        # Publish to replication channel
        channel = self._replication_channel()
        await self._redis.publish(channel, json.dumps(message))

        acks = 0
        if wait_for_acks:
            # Wait for acknowledgments
            acks = await self._wait_for_acks(
                self._position,
                self.config.min_sync_replicas,
            )

            # Check quorum
            if self.config.strategy == ReplicationStrategy.SYNC:
                required = self.config.replication_factor - 1
            else:
                required = self.config.min_sync_replicas

            if acks < required:
                raise QuorumNotReachedError(
                    required=required,
                    available=acks,
                    operation="replication",
                )

        latency_ms = (time.time() - start_time) * 1000
        self._metrics.record_replication(latency_ms, success=True)

        # Emit event
        self._emit_event(
            ReplicationEventType.REPLICATION_COMPLETED,
            memory_id=memory_data.get("id"),
            data={"acks": acks, "latency_ms": latency_ms},
        )

        return acks

    async def replicate_batch(
        self,
        memories: list[dict[str, Any]],
    ) -> int:
        """
        Replicate a batch of memories.

        Args:
            memories: List of memory data to replicate

        Returns:
            Number of successful acknowledgments for last batch
        """
        # Process in batches
        total_acks = 0
        for i in range(0, len(memories), self.config.batch_size):
            batch = memories[i : i + self.config.batch_size]
            for memory in batch:
                acks = await self.replicate(memory)
                total_acks = max(total_acks, acks)

        return total_acks

    async def _wait_for_acks(
        self,
        position: int,
        required: int,
    ) -> int:
        """Wait for acknowledgments from replicas."""
        ack_key = self._ack_key(position)
        deadline = time.time() + self.config.ack_timeout
        acks = 0

        while time.time() < deadline and acks < required:
            # Check for new acks
            count = await self._redis.scard(ack_key)
            acks = count if count else 0

            if acks >= required:
                break

            await asyncio.sleep(0.01)  # Small delay

        # Cleanup
        await self._redis.delete(ack_key)

        return acks

    async def acknowledge(self, source_node: str, position: int) -> None:
        """
        Acknowledge a replication from source.

        Args:
            source_node: Source node ID
            position: Replication position
        """
        self._ensure_connected()

        ack_key = f"{self.config.cluster_name}:replication:{source_node}:ack:{position}"
        await self._redis.sadd(ack_key, self.config.node_id)
        await self._redis.expire(ack_key, 60)  # Cleanup after 60 seconds

        # Update replica info
        if source_node in self._replicas:
            self._replicas[source_node].sync_position = position
            self._replicas[source_node].last_ack_time = datetime.utcnow()

    # ==================== Replica Management ====================

    async def add_replica(self, node_id: str) -> ReplicaInfo:
        """
        Add a new replica node.

        Args:
            node_id: Node ID to add as replica

        Returns:
            ReplicaInfo for the new replica
        """
        self._ensure_connected()

        replica = ReplicaInfo(
            node_id=node_id,
            state=ReplicationState.SYNCING,
        )

        self._replicas[node_id] = replica

        # Store in Redis
        await self._save_replica_info(replica)

        self._emit_event(
            ReplicationEventType.REPLICA_ADDED,
            target_node=node_id,
        )

        return replica

    async def remove_replica(self, node_id: str) -> None:
        """
        Remove a replica node.

        Args:
            node_id: Node ID to remove
        """
        self._ensure_connected()

        if node_id in self._replicas:
            del self._replicas[node_id]

        # Remove from Redis
        key = self._replica_key(node_id)
        await self._redis.delete(key)

        self._emit_event(
            ReplicationEventType.REPLICA_REMOVED,
            target_node=node_id,
        )

    async def get_replica_lag(self, node_id: str) -> int:
        """
        Get replication lag for a replica.

        Args:
            node_id: Replica node ID

        Returns:
            Number of messages behind
        """
        if node_id not in self._replicas:
            return -1

        replica = self._replicas[node_id]
        return self._position - replica.sync_position

    async def sync_replica(
        self,
        node_id: str,
        memory_getter: Callable[[], AsyncIterator[dict[str, Any]]],
    ) -> bool:
        """
        Full sync a replica from scratch.

        Args:
            node_id: Replica to sync
            memory_getter: Async generator yielding all memories

        Returns:
            True if sync successful
        """
        self._ensure_connected()

        if node_id not in self._replicas:
            raise ReplicationError(
                f"Unknown replica: {node_id}",
                source_node=self.config.node_id,
                target_node=node_id,
            )

        replica = self._replicas[node_id]
        replica.state = ReplicationState.SYNCING

        self._emit_event(
            ReplicationEventType.REPLICATION_STARTED,
            target_node=node_id,
        )

        try:
            batch = []
            async for memory in memory_getter():
                batch.append(memory)
                if len(batch) >= self.config.batch_size:
                    await self._send_sync_batch(node_id, batch)
                    batch = []

            # Send remaining
            if batch:
                await self._send_sync_batch(node_id, batch)

            replica.state = ReplicationState.IN_SYNC
            replica.sync_position = self._position
            replica.last_sync_time = datetime.utcnow()

            await self._save_replica_info(replica)

            self._emit_event(
                ReplicationEventType.REPLICA_CAUGHT_UP,
                target_node=node_id,
            )

            return True

        except Exception as e:
            replica.state = ReplicationState.FAILED
            await self._save_replica_info(replica)

            self._emit_event(
                ReplicationEventType.REPLICATION_FAILED,
                target_node=node_id,
                data={"error": str(e)},
            )

            raise ReplicationError(
                f"Sync failed: {e}",
                source_node=self.config.node_id,
                target_node=node_id,
            )

    async def _send_sync_batch(self, node_id: str, memories: list[dict[str, Any]]) -> None:
        """Send a batch of memories to a replica for sync."""
        channel = f"{self.config.cluster_name}:sync:{node_id}"
        message = {
            "type": "sync_batch",
            "source_node": self.config.node_id,
            "memories": memories,
        }
        await self._redis.publish(channel, json.dumps(message))

    # ==================== Failover ====================

    async def initiate_failover(
        self,
        failed_node: str,
        new_primary: str | None = None,
    ) -> str:
        """
        Initiate failover from a failed node.

        Args:
            failed_node: ID of failed node
            new_primary: Preferred new primary (or auto-select)

        Returns:
            ID of new primary node
        """
        self._ensure_connected()

        self._emit_event(
            ReplicationEventType.FAILOVER_STARTED,
            target_node=failed_node,
        )

        # Select new primary
        if new_primary is None:
            # Select replica with least lag
            candidates = [
                (r.node_id, r.lag_messages)
                for r in self._replicas.values()
                if r.state in (ReplicationState.IN_SYNC, ReplicationState.LAGGING)
            ]
            if not candidates:
                raise ReplicationError(
                    "No healthy replicas available for failover",
                    source_node=failed_node,
                )

            candidates.sort(key=lambda x: x[1])
            new_primary = candidates[0][0]

        # Promote new primary
        promotion_key = f"{self.config.cluster_name}:primary"
        await self._redis.set(promotion_key, new_primary)

        # Mark old node as failed
        if failed_node in self._replicas:
            self._replicas[failed_node].state = ReplicationState.FAILED
            await self._save_replica_info(self._replicas[failed_node])

        self._emit_event(
            ReplicationEventType.FAILOVER_COMPLETED,
            target_node=new_primary,
            data={"failed_node": failed_node},
        )

        logger.info(f"Failover completed: {failed_node} -> {new_primary}")
        return new_primary

    # ==================== Internal Methods ====================

    def _ensure_connected(self) -> None:
        """Ensure Redis is connected."""
        if self._redis is None:
            raise ClusterError("Replication manager not started")

    def _replication_channel(self) -> str:
        """Get Redis channel for replication."""
        return f"{self.config.cluster_name}:replication:{self.config.node_id}"

    def _ack_key(self, position: int) -> str:
        """Get Redis key for acks."""
        return f"{self.config.cluster_name}:replication:{self.config.node_id}:ack:{position}"

    def _replica_key(self, node_id: str) -> str:
        """Get Redis key for replica info."""
        return f"{self.config.cluster_name}:replica:{node_id}"

    async def _register_node(self) -> None:
        """Register this node as replication source."""
        key = f"{self.config.cluster_name}:replication:sources"
        await self._redis.sadd(key, self.config.node_id)

    async def _unregister_node(self) -> None:
        """Unregister this node."""
        key = f"{self.config.cluster_name}:replication:sources"
        await self._redis.srem(key, self.config.node_id)

    async def _save_replica_info(self, replica: ReplicaInfo) -> None:
        """Save replica info to Redis."""
        key = self._replica_key(replica.node_id)
        await self._redis.set(key, json.dumps(replica.to_dict()))

    async def _load_replicas(self) -> None:
        """Load replica info from Redis."""
        pattern = f"{self.config.cluster_name}:replica:*"
        cursor = 0

        while True:
            cursor, keys = await self._redis.scan(cursor, match=pattern)
            for key in keys:
                data = await self._redis.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode()
                    replica = ReplicaInfo.from_dict(json.loads(data))
                    self._replicas[replica.node_id] = replica

            if cursor == 0:
                break

    def _emit_event(
        self,
        event_type: ReplicationEventType,
        target_node: str | None = None,
        memory_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a replication event."""
        event = ReplicationEvent.create(
            event_type=event_type,
            source_node=self.config.node_id,
            target_node=target_node,
            memory_id=memory_id,
            data=data,
        )

        if self._on_event:
            try:
                self._on_event(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")

    async def _listen_for_replications(self) -> None:
        """Listen for incoming replication messages."""
        # Subscribe to replication channels
        self._pubsub = self._redis.pubsub()

        # Listen to all replication channels we should receive
        pattern = f"{self.config.cluster_name}:replication:*"
        await self._pubsub.psubscribe(pattern)

        # Also listen to our sync channel
        sync_channel = f"{self.config.cluster_name}:sync:{self.config.node_id}"
        await self._pubsub.subscribe(sync_channel)

        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message and message["type"] in ("message", "pmessage"):
                    await self._handle_replication_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Replication listener error: {e}")

    async def _handle_replication_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming replication message."""
        try:
            data = message.get("data")
            if isinstance(data, bytes):
                data = data.decode()

            payload = json.loads(data)
            msg_type = payload.get("type")

            if msg_type == "replicate":
                # Acknowledge receipt
                source = payload["source_node"]
                position = payload["position"]
                await self.acknowledge(source, position)

                # Store memory locally (would need store reference)
                # For now, just log
                logger.debug(f"Received replication from {source} at position {position}")

            elif msg_type == "sync_batch":
                # Handle sync batch
                source = payload["source_node"]
                memories = payload["memories"]
                logger.debug(f"Received sync batch from {source}: {len(memories)} memories")

        except Exception as e:
            logger.error(f"Error handling replication message: {e}")

    async def _health_check_loop(self) -> None:
        """Background health check for replicas."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                in_sync = 0
                lagging = 0
                failed = 0

                for replica in self._replicas.values():
                    # Check lag
                    lag = self._position - replica.sync_position

                    if lag > self.config.max_lag_messages:
                        if replica.state != ReplicationState.LAGGING:
                            replica.state = ReplicationState.LAGGING
                            self._emit_event(
                                ReplicationEventType.REPLICA_FELL_BEHIND,
                                target_node=replica.node_id,
                            )

                    # Check last ack time
                    if replica.last_ack_time:
                        elapsed = (datetime.utcnow() - replica.last_ack_time).total_seconds()
                        if elapsed > self.config.max_lag_seconds:
                            replica.state = ReplicationState.FAILED
                            failed += 1
                            continue

                    # Count states
                    if replica.state == ReplicationState.IN_SYNC:
                        in_sync += 1
                    elif replica.state == ReplicationState.LAGGING:
                        lagging += 1
                    elif replica.state == ReplicationState.FAILED:
                        failed += 1

                    replica.lag_messages = lag

                # Update metrics
                self._metrics.replicas_in_sync = in_sync
                self._metrics.replicas_lagging = lagging
                self._metrics.replicas_failed = failed
                self._metrics.current_lag_messages = max(
                    r.lag_messages for r in self._replicas.values()
                ) if self._replicas else 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check error: {e}")
