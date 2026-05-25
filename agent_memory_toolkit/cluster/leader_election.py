"""
Leader election for write coordination in distributed memory clusters.

Uses Redis-based distributed locks for leader election with
automatic failover and fencing.
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

from .exceptions import LeaderElectionError, ClusterError

logger = logging.getLogger(__name__)


# Optional Redis imports
try:
    import redis.asyncio as redis
    from redis.asyncio.lock import Lock as RedisLock

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore
    RedisLock = None  # type: ignore


class LeaderState(Enum):
    """State of a node in leader election."""

    FOLLOWER = auto()  # Not the leader
    CANDIDATE = auto()  # Attempting to become leader
    LEADER = auto()  # Current leader


class LeaderEventType(Enum):
    """Types of leader election events."""

    ELECTION_STARTED = auto()
    BECAME_LEADER = auto()
    LOST_LEADERSHIP = auto()
    LEADER_CHANGED = auto()
    HEARTBEAT = auto()
    FENCING_DETECTED = auto()


@dataclass
class LeaderEvent:
    """An event during leader election."""

    id: str
    type: LeaderEventType
    timestamp: datetime
    node_id: str
    leader_id: str | None
    term: int
    data: dict[str, Any]

    @classmethod
    def create(
        cls,
        event_type: LeaderEventType,
        node_id: str,
        leader_id: str | None = None,
        term: int = 0,
        data: dict[str, Any] | None = None,
    ) -> LeaderEvent:
        """Create a new leader event."""
        return cls(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.utcnow(),
            node_id=node_id,
            leader_id=leader_id,
            term=term,
            data=data or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "node_id": self.node_id,
            "leader_id": self.leader_id,
            "term": self.term,
            "data": self.data,
        }


@dataclass
class LeaderConfig:
    """Configuration for leader election."""

    # Node identification
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cluster_name: str = "agent-memory-cluster"

    # Election settings
    election_timeout: float = 10.0  # Seconds to wait for election
    heartbeat_interval: float = 3.0  # Seconds between heartbeats
    lease_duration: float = 15.0  # Leader lease duration

    # Fencing
    enable_fencing: bool = True  # Enable fencing tokens
    fencing_token_key: str = "fencing_token"

    # Callbacks
    on_become_leader: Callable[[], None] | None = None
    on_lose_leadership: Callable[[], None] | None = None
    on_leader_change: Callable[[str], None] | None = None


class LeaderElection:
    """
    Leader election with Redis distributed locks.

    Features:
    - Automatic failover when leader fails
    - Fencing tokens to prevent split-brain
    - Leader lease with automatic renewal
    - Event notifications for state changes

    Example:
        >>> config = LeaderConfig(
        ...     node_id="node-1",
        ...     on_become_leader=lambda: print("I am the leader!"),
        ... )
        >>> election = LeaderElection(
        ...     redis_url="redis://localhost:6379",
        ...     config=config,
        ... )
        >>> await election.start()
        >>>
        >>> # Check if this node is leader
        >>> if election.is_leader:
        ...     await perform_leader_only_operation()
        >>>
        >>> # Use as context manager
        >>> async with LeaderElection(redis_url, config) as leader:
        ...     if leader.is_leader:
        ...         await coordinated_write()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: LeaderConfig | None = None,
        on_event: Callable[[LeaderEvent], None] | None = None,
    ):
        """
        Initialize leader election.

        Args:
            redis_url: Redis connection URL
            config: Leader election configuration
            on_event: Callback for election events
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required for leader election. "
                "Install with: pip install redis[hiredis]"
            )

        self.redis_url = redis_url
        self.config = config or LeaderConfig()
        self._on_event = on_event

        # State
        self._redis: redis.Redis | None = None
        self._lock: RedisLock | None = None
        self._state = LeaderState.FOLLOWER
        self._current_leader: str | None = None
        self._term = 0
        self._fencing_token = 0

        # Background tasks
        self._election_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._running = False

    @property
    def is_leader(self) -> bool:
        """Check if this node is currently the leader."""
        return self._state == LeaderState.LEADER

    @property
    def state(self) -> LeaderState:
        """Get current election state."""
        return self._state

    @property
    def current_leader(self) -> str | None:
        """Get ID of current leader."""
        return self._current_leader

    @property
    def term(self) -> int:
        """Get current election term."""
        return self._term

    @property
    def fencing_token(self) -> int:
        """Get current fencing token (valid only for leader)."""
        return self._fencing_token if self.is_leader else 0

    async def start(self) -> None:
        """Start leader election process."""
        self._redis = redis.from_url(
            self.redis_url,
            socket_timeout=self.config.election_timeout,
        )

        # Test connection
        await self._redis.ping()

        self._running = True

        # Start election loop
        self._election_task = asyncio.create_task(self._election_loop())

        logger.info(f"Leader election started for node {self.config.node_id}")

    async def stop(self) -> None:
        """Stop leader election and release leadership."""
        self._running = False

        # Release leadership if we have it
        if self.is_leader:
            await self._release_leadership()

        # Cancel background tasks
        if self._election_task:
            self._election_task.cancel()
            try:
                await self._election_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close Redis
        if self._redis:
            await self._redis.close()

        logger.info(f"Leader election stopped for node {self.config.node_id}")

    async def __aenter__(self) -> LeaderElection:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # ==================== Leadership Operations ====================

    async def try_become_leader(self) -> bool:
        """
        Attempt to become the leader.

        Returns:
            True if successfully became leader
        """
        self._ensure_connected()

        self._state = LeaderState.CANDIDATE
        self._emit_event(LeaderEventType.ELECTION_STARTED)

        try:
            # Try to acquire the leader lock
            lock_key = self._leader_lock_key()
            acquired = await self._redis.set(
                lock_key,
                self.config.node_id,
                nx=True,
                ex=int(self.config.lease_duration),
            )

            if acquired:
                await self._on_became_leader()
                return True

            # Check who the current leader is
            leader = await self._redis.get(lock_key)
            if leader:
                self._current_leader = leader.decode() if isinstance(leader, bytes) else leader

            self._state = LeaderState.FOLLOWER
            return False

        except Exception as e:
            logger.error(f"Election failed: {e}")
            self._state = LeaderState.FOLLOWER
            return False

    async def _on_became_leader(self) -> None:
        """Handle becoming the leader."""
        self._state = LeaderState.LEADER
        self._current_leader = self.config.node_id
        self._term += 1

        # Increment fencing token
        if self.config.enable_fencing:
            self._fencing_token = await self._get_next_fencing_token()

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Emit event
        self._emit_event(LeaderEventType.BECAME_LEADER)

        # Callback
        if self.config.on_become_leader:
            try:
                self.config.on_become_leader()
            except Exception as e:
                logger.error(f"on_become_leader callback failed: {e}")

        logger.info(
            f"Node {self.config.node_id} became leader "
            f"(term={self._term}, fencing_token={self._fencing_token})"
        )

    async def _release_leadership(self) -> None:
        """Release leadership."""
        if not self.is_leader:
            return

        try:
            # Use Lua script to atomically check and delete
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            lock_key = self._leader_lock_key()
            await self._redis.eval(script, 1, lock_key, self.config.node_id)

        except Exception as e:
            logger.warning(f"Error releasing leadership: {e}")

        self._state = LeaderState.FOLLOWER
        self._emit_event(LeaderEventType.LOST_LEADERSHIP)

        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Callback
        if self.config.on_lose_leadership:
            try:
                self.config.on_lose_leadership()
            except Exception as e:
                logger.error(f"on_lose_leadership callback failed: {e}")

        logger.info(f"Node {self.config.node_id} released leadership")

    async def renew_lease(self) -> bool:
        """
        Renew the leader lease.

        Returns:
            True if lease was renewed
        """
        if not self.is_leader:
            return False

        try:
            # Atomic check-and-extend using Lua script
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("expire", KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            lock_key = self._leader_lock_key()
            result = await self._redis.eval(
                script, 1, lock_key, self.config.node_id, int(self.config.lease_duration)
            )

            if result == 0:
                # Lost leadership
                await self._on_lost_leadership()
                return False

            return True

        except Exception as e:
            logger.error(f"Lease renewal failed: {e}")
            await self._on_lost_leadership()
            return False

    async def _on_lost_leadership(self) -> None:
        """Handle losing leadership unexpectedly."""
        if not self.is_leader:
            return

        self._state = LeaderState.FOLLOWER
        self._emit_event(LeaderEventType.LOST_LEADERSHIP)

        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Callback
        if self.config.on_lose_leadership:
            try:
                self.config.on_lose_leadership()
            except Exception as e:
                logger.error(f"on_lose_leadership callback failed: {e}")

        logger.warning(f"Node {self.config.node_id} lost leadership unexpectedly")

    async def get_leader(self) -> str | None:
        """
        Get the current leader ID.

        Returns:
            Leader node ID or None if no leader
        """
        self._ensure_connected()

        lock_key = self._leader_lock_key()
        leader = await self._redis.get(lock_key)

        if leader:
            leader_id = leader.decode() if isinstance(leader, bytes) else leader
            if leader_id != self._current_leader:
                old_leader = self._current_leader
                self._current_leader = leader_id
                if old_leader:
                    self._emit_event(LeaderEventType.LEADER_CHANGED)
                    if self.config.on_leader_change:
                        self.config.on_leader_change(leader_id)
            return leader_id

        self._current_leader = None
        return None

    # ==================== Fencing ====================

    async def _get_next_fencing_token(self) -> int:
        """Get the next fencing token (monotonically increasing)."""
        key = f"{self.config.cluster_name}:{self.config.fencing_token_key}"
        token = await self._redis.incr(key)
        return token

    async def validate_fencing_token(self, token: int) -> bool:
        """
        Validate a fencing token.

        Args:
            token: Token to validate

        Returns:
            True if token is valid (current or newer)
        """
        if not self.config.enable_fencing:
            return True

        key = f"{self.config.cluster_name}:{self.config.fencing_token_key}"
        current = await self._redis.get(key)

        if current is None:
            return True

        current_token = int(current)
        return token >= current_token

    def get_fencing_header(self) -> dict[str, str]:
        """
        Get fencing token as HTTP header for distributed requests.

        Returns:
            Dict with fencing token header
        """
        if self.is_leader and self.config.enable_fencing:
            return {"X-Fencing-Token": str(self._fencing_token)}
        return {}

    # ==================== Internal Methods ====================

    def _ensure_connected(self) -> None:
        """Ensure Redis is connected."""
        if self._redis is None:
            raise ClusterError("Leader election not started")

    def _leader_lock_key(self) -> str:
        """Get Redis key for leader lock."""
        return f"{self.config.cluster_name}:leader:lock"

    def _emit_event(self, event_type: LeaderEventType) -> None:
        """Emit a leader event."""
        event = LeaderEvent.create(
            event_type=event_type,
            node_id=self.config.node_id,
            leader_id=self._current_leader,
            term=self._term,
        )

        if self._on_event:
            try:
                self._on_event(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")

    async def _election_loop(self) -> None:
        """Background election loop."""
        while self._running:
            try:
                if not self.is_leader:
                    # Check if there's a leader
                    leader = await self.get_leader()

                    if leader is None:
                        # No leader, try to become one
                        await self.try_become_leader()

                await asyncio.sleep(self.config.election_timeout / 2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Election loop error: {e}")
                await asyncio.sleep(1)

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop for leader."""
        while self._running and self.is_leader:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                # Renew lease
                renewed = await self.renew_lease()
                if not renewed:
                    break

                self._emit_event(LeaderEventType.HEARTBEAT)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")


class LeaderGuard:
    """
    Context manager for leader-only operations.

    Ensures operation only proceeds if this node is the leader
    and uses fencing tokens for safety.

    Example:
        >>> async with LeaderGuard(election) as guard:
        ...     if guard.is_leader:
        ...         # Safe to perform write
        ...         await store.add("Important memory")
    """

    def __init__(self, election: LeaderElection, require_leader: bool = True):
        """
        Initialize leader guard.

        Args:
            election: LeaderElection instance
            require_leader: Raise exception if not leader (default True)
        """
        self.election = election
        self.require_leader = require_leader
        self._initial_token = 0

    @property
    def is_leader(self) -> bool:
        """Check if this node is currently the leader."""
        return self.election.is_leader

    @property
    def fencing_token(self) -> int:
        """Get fencing token for this operation."""
        return self._initial_token

    async def __aenter__(self) -> LeaderGuard:
        if self.require_leader and not self.election.is_leader:
            raise LeaderElectionError(
                f"Node {self.election.config.node_id} is not the leader"
            )

        self._initial_token = self.election.fencing_token
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Verify we still have the same fencing token (no leadership change)
        if self.election.is_leader and self.election.fencing_token != self._initial_token:
            logger.warning("Fencing token changed during operation - leadership may have changed")
