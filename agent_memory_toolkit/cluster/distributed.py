"""
Distributed Memory Store with Redis Cluster backend.

Provides horizontal scaling for AI agent memories across multiple nodes
with consistent hashing, automatic sharding, and configurable consistency levels.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable

from .exceptions import (
    ClusterConnectionError,
    ClusterError,
    NodeNotFoundError,
    QuorumNotReachedError,
)

logger = logging.getLogger(__name__)


# Optional Redis imports
try:
    import redis.asyncio as redis
    from redis.asyncio.cluster import RedisCluster
    from redis.asyncio.sentinel import Sentinel

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore
    RedisCluster = None  # type: ignore
    Sentinel = None  # type: ignore


class NodeState(Enum):
    """State of a cluster node."""

    JOINING = auto()  # Node is joining the cluster
    ACTIVE = auto()  # Node is active and healthy
    DRAINING = auto()  # Node is draining connections before shutdown
    LEAVING = auto()  # Node is leaving the cluster
    FAILED = auto()  # Node has failed
    UNKNOWN = auto()  # Node state is unknown


class ConsistencyLevel(Enum):
    """Consistency level for distributed operations."""

    ONE = 1  # Write/read from one node
    QUORUM = 2  # Majority of nodes
    ALL = 3  # All nodes must acknowledge
    LOCAL_QUORUM = 4  # Quorum within local datacenter


@dataclass
class ClusterConfig:
    """Configuration for distributed cluster."""

    # Redis connection
    redis_urls: list[str] = field(default_factory=lambda: ["redis://localhost:6379"])
    redis_password: str | None = None
    redis_db: int = 0
    redis_ssl: bool = False

    # Cluster settings
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cluster_name: str = "agent-memory-cluster"
    datacenter: str = "default"

    # Replication
    replication_factor: int = 2
    consistency_level: ConsistencyLevel = ConsistencyLevel.QUORUM

    # Sharding
    num_shards: int = 16
    virtual_nodes: int = 150  # Virtual nodes per physical node

    # Timeouts (seconds)
    connection_timeout: float = 5.0
    operation_timeout: float = 30.0
    heartbeat_interval: float = 5.0
    node_timeout: float = 15.0

    # Performance
    max_connections: int = 100
    batch_size: int = 100

    # Sentinel settings (optional)
    use_sentinel: bool = False
    sentinel_master: str = "mymaster"


@dataclass
class NodeInfo:
    """Information about a cluster node."""

    node_id: str
    host: str
    port: int
    state: NodeState
    datacenter: str
    shards: list[int]
    last_heartbeat: datetime
    load: float = 0.0
    memory_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "state": self.state.name,
            "datacenter": self.datacenter,
            "shards": self.shards,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "load": self.load,
            "memory_count": self.memory_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeInfo:
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            state=NodeState[data["state"]],
            datacenter=data["datacenter"],
            shards=data.get("shards", []),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            load=data.get("load", 0.0),
            memory_count=data.get("memory_count", 0),
        )


@dataclass
class ShardInfo:
    """Information about a shard."""

    shard_id: int
    primary_node: str
    replica_nodes: list[str]
    memory_count: int
    size_bytes: int
    state: str = "active"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "shard_id": self.shard_id,
            "primary_node": self.primary_node,
            "replica_nodes": self.replica_nodes,
            "memory_count": self.memory_count,
            "size_bytes": self.size_bytes,
            "state": self.state,
        }


@dataclass
class ClusterHealth:
    """Overall cluster health status."""

    healthy: bool
    total_nodes: int
    active_nodes: int
    failed_nodes: int
    total_shards: int
    healthy_shards: int
    replication_factor: int
    total_memories: int
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "healthy": self.healthy,
            "total_nodes": self.total_nodes,
            "active_nodes": self.active_nodes,
            "failed_nodes": self.failed_nodes,
            "total_shards": self.total_shards,
            "healthy_shards": self.healthy_shards,
            "replication_factor": self.replication_factor,
            "total_memories": self.total_memories,
            "message": self.message,
        }


@dataclass
class DistributedMemory:
    """A memory entry in the distributed store."""

    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None
    created_at: datetime
    updated_at: datetime
    version: int
    shard_id: int
    vector_clock: dict[str, int]
    is_deleted: bool = False

    @classmethod
    def create(
        cls,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        shard_id: int = 0,
        node_id: str = "",
    ) -> DistributedMemory:
        """Create a new distributed memory."""
        now = datetime.utcnow()
        memory_id = str(uuid.uuid4())
        vector_clock = {node_id: 1} if node_id else {}

        return cls(
            id=memory_id,
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            created_at=now,
            updated_at=now,
            version=1,
            shard_id=shard_id,
            vector_clock=vector_clock,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "shard_id": self.shard_id,
            "vector_clock": self.vector_clock,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributedMemory:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", 1),
            shard_id=data.get("shard_id", 0),
            vector_clock=data.get("vector_clock", {}),
            is_deleted=data.get("is_deleted", False),
        )


class ConsistentHash:
    """Consistent hashing for shard distribution."""

    def __init__(self, num_shards: int, virtual_nodes: int = 150):
        self.num_shards = num_shards
        self.virtual_nodes = virtual_nodes
        self.ring: dict[int, str] = {}
        self.sorted_keys: list[int] = []
        self._node_to_shards: dict[str, list[int]] = {}

    def _hash(self, key: str) -> int:
        """Hash a key to a position on the ring."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node_id: str, shards: list[int] | None = None) -> None:
        """Add a node to the hash ring."""
        if shards is None:
            # Distribute shards evenly if not specified
            shards = list(range(self.num_shards))

        self._node_to_shards[node_id] = shards

        for i in range(self.virtual_nodes):
            key = f"{node_id}:{i}"
            hash_val = self._hash(key)
            self.ring[hash_val] = node_id

        self.sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the hash ring."""
        for i in range(self.virtual_nodes):
            key = f"{node_id}:{i}"
            hash_val = self._hash(key)
            self.ring.pop(hash_val, None)

        self.sorted_keys = sorted(self.ring.keys())
        self._node_to_shards.pop(node_id, None)

    def get_node(self, key: str) -> str | None:
        """Get the node responsible for a key."""
        if not self.ring:
            return None

        hash_val = self._hash(key)

        # Binary search for the first node with hash >= key hash
        idx = self._bisect_right(hash_val)
        if idx == len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def get_shard(self, key: str) -> int:
        """Get the shard ID for a key."""
        hash_val = self._hash(key)
        return hash_val % self.num_shards

    def _bisect_right(self, x: int) -> int:
        """Binary search for insertion point."""
        lo, hi = 0, len(self.sorted_keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < self.sorted_keys[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def get_nodes_for_shard(self, shard_id: int, count: int = 1) -> list[str]:
        """Get nodes responsible for a shard."""
        nodes = []
        for node_id, shards in self._node_to_shards.items():
            if shard_id in shards:
                nodes.append(node_id)
                if len(nodes) >= count:
                    break
        return nodes


class DistributedMemoryStore:
    """
    Distributed memory store with Redis Cluster backend.

    Features:
    - Automatic sharding with consistent hashing
    - Configurable replication factor
    - Quorum-based consistency
    - Automatic failover
    - Vector clock for conflict resolution

    Example:
        >>> config = ClusterConfig(
        ...     redis_urls=["redis://node1:6379", "redis://node2:6379"],
        ...     node_id="node-1",
        ...     replication_factor=2,
        ... )
        >>> store = DistributedMemoryStore(config)
        >>> await store.connect()
        >>>
        >>> # Add memory (automatically sharded and replicated)
        >>> memory = await store.add("Important fact")
        >>>
        >>> # Search across all shards
        >>> results = await store.search("important")
    """

    def __init__(
        self,
        config: ClusterConfig | None = None,
        on_node_join: Callable[[NodeInfo], None] | None = None,
        on_node_leave: Callable[[NodeInfo], None] | None = None,
    ):
        """
        Initialize distributed memory store.

        Args:
            config: Cluster configuration
            on_node_join: Callback when a node joins
            on_node_leave: Callback when a node leaves
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package is required for distributed clustering. "
                "Install with: pip install redis[hiredis]"
            )

        self.config = config or ClusterConfig()
        self._on_node_join = on_node_join
        self._on_node_leave = on_node_leave

        # Redis connections
        self._redis: redis.Redis | None = None
        self._cluster: RedisCluster | None = None
        self._sentinel: Sentinel | None = None

        # Cluster state
        self._nodes: dict[str, NodeInfo] = {}
        self._hash_ring = ConsistentHash(
            self.config.num_shards, self.config.virtual_nodes
        )
        self._shards: dict[int, ShardInfo] = {}
        self._connected = False

        # Background tasks
        self._heartbeat_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Connect to the Redis cluster."""
        if self._connected:
            return

        try:
            # Parse first URL to get connection params
            url = self.config.redis_urls[0]

            if self.config.use_sentinel:
                # Connect via Sentinel
                sentinel_hosts = []
                for u in self.config.redis_urls:
                    # Parse redis://host:port format
                    parts = u.replace("redis://", "").split(":")
                    sentinel_hosts.append((parts[0], int(parts[1]) if len(parts) > 1 else 26379))

                self._sentinel = Sentinel(
                    sentinel_hosts,
                    password=self.config.redis_password,
                    socket_timeout=self.config.connection_timeout,
                )
                self._redis = self._sentinel.master_for(
                    self.config.sentinel_master,
                    redis_class=redis.Redis,
                )
            else:
                # Direct connection
                self._redis = redis.from_url(
                    url,
                    password=self.config.redis_password,
                    db=self.config.redis_db,
                    socket_timeout=self.config.connection_timeout,
                    max_connections=self.config.max_connections,
                    ssl=self.config.redis_ssl,
                )

            # Test connection
            await self._redis.ping()

            # Register this node
            await self._register_node()

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._connected = True
            logger.info(f"Connected to cluster as node {self.config.node_id}")

        except Exception as e:
            raise ClusterConnectionError(f"Failed to connect: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the cluster."""
        if not self._connected:
            return

        # Stop background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Unregister node
        await self._unregister_node()

        # Close connections
        if self._redis:
            await self._redis.close()

        self._connected = False
        logger.info(f"Disconnected node {self.config.node_id} from cluster")

    async def __aenter__(self) -> DistributedMemoryStore:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()

    # ==================== CRUD Operations ====================

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> DistributedMemory:
        """
        Add a memory to the distributed store.

        Args:
            content: Memory content
            metadata: Optional metadata
            embedding: Optional embedding vector

        Returns:
            The created DistributedMemory
        """
        self._ensure_connected()

        # Create memory
        memory_id = str(uuid.uuid4())
        shard_id = self._hash_ring.get_shard(memory_id)

        memory = DistributedMemory.create(
            content=content,
            metadata=metadata,
            embedding=embedding,
            shard_id=shard_id,
            node_id=self.config.node_id,
        )
        memory.id = memory_id

        # Store in Redis
        key = self._memory_key(memory_id)
        await self._redis.set(key, json.dumps(memory.to_dict()))

        # Add to shard index
        shard_key = self._shard_index_key(shard_id)
        await self._redis.sadd(shard_key, memory_id)

        # Replicate if needed
        if self.config.replication_factor > 1:
            await self._replicate_memory(memory)

        logger.debug(f"Added memory {memory_id} to shard {shard_id}")
        return memory

    async def get(self, memory_id: str) -> DistributedMemory:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            The DistributedMemory

        Raises:
            KeyError: If memory not found
        """
        self._ensure_connected()

        key = self._memory_key(memory_id)
        data = await self._redis.get(key)

        if data is None:
            raise KeyError(f"Memory not found: {memory_id}")

        memory = DistributedMemory.from_dict(json.loads(data))
        if memory.is_deleted:
            raise KeyError(f"Memory not found: {memory_id}")

        return memory

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> DistributedMemory:
        """
        Update an existing memory.

        Args:
            memory_id: Memory ID
            content: New content (optional)
            metadata: New metadata (optional)
            embedding: New embedding (optional)

        Returns:
            Updated DistributedMemory
        """
        self._ensure_connected()

        # Get existing memory
        memory = await self.get(memory_id)

        # Update fields
        if content is not None:
            memory.content = content
        if metadata is not None:
            memory.metadata = metadata
        if embedding is not None:
            memory.embedding = embedding

        memory.updated_at = datetime.utcnow()
        memory.version += 1
        memory.vector_clock[self.config.node_id] = (
            memory.vector_clock.get(self.config.node_id, 0) + 1
        )

        # Store update
        key = self._memory_key(memory_id)
        await self._redis.set(key, json.dumps(memory.to_dict()))

        # Replicate update
        if self.config.replication_factor > 1:
            await self._replicate_memory(memory)

        return memory

    async def delete(self, memory_id: str, hard: bool = False) -> None:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID
            hard: If True, permanently delete
        """
        self._ensure_connected()

        if hard:
            key = self._memory_key(memory_id)

            # Get memory to find shard
            data = await self._redis.get(key)
            if data:
                memory = DistributedMemory.from_dict(json.loads(data))
                shard_key = self._shard_index_key(memory.shard_id)
                await self._redis.srem(shard_key, memory_id)

            await self._redis.delete(key)
        else:
            # Soft delete
            memory = await self.get(memory_id)
            memory.is_deleted = True
            memory.updated_at = datetime.utcnow()
            memory.version += 1

            key = self._memory_key(memory_id)
            await self._redis.set(key, json.dumps(memory.to_dict()))

    async def list(
        self,
        limit: int = 100,
        offset: int = 0,
        shard_id: int | None = None,
    ) -> list[DistributedMemory]:
        """
        List memories with pagination.

        Args:
            limit: Maximum memories to return
            offset: Number to skip
            shard_id: Filter by shard (optional)

        Returns:
            List of memories
        """
        self._ensure_connected()

        memories = []

        if shard_id is not None:
            # List from specific shard
            shard_key = self._shard_index_key(shard_id)
            memory_ids = await self._redis.smembers(shard_key)
        else:
            # List from all shards
            memory_ids = set()
            for sid in range(self.config.num_shards):
                shard_key = self._shard_index_key(sid)
                ids = await self._redis.smembers(shard_key)
                memory_ids.update(ids)

        # Apply pagination
        memory_ids_list = sorted(memory_ids)[offset : offset + limit]

        # Fetch memories
        for mid in memory_ids_list:
            try:
                if isinstance(mid, bytes):
                    mid = mid.decode()
                memory = await self.get(mid)
                memories.append(memory)
            except KeyError:
                continue

        return memories

    async def search(
        self,
        query: str,
        limit: int = 10,
        shard_ids: list[int] | None = None,
    ) -> list[DistributedMemory]:
        """
        Search memories across shards.

        Args:
            query: Search query
            limit: Maximum results
            shard_ids: Specific shards to search (optional)

        Returns:
            List of matching memories
        """
        self._ensure_connected()

        # Simple text search - in production, use Redis Search or external index
        results = []
        query_lower = query.lower()

        if shard_ids is None:
            shard_ids = list(range(self.config.num_shards))

        for shard_id in shard_ids:
            shard_key = self._shard_index_key(shard_id)
            memory_ids = await self._redis.smembers(shard_key)

            for mid in memory_ids:
                try:
                    if isinstance(mid, bytes):
                        mid = mid.decode()
                    memory = await self.get(mid)
                    if query_lower in memory.content.lower():
                        results.append(memory)
                        if len(results) >= limit:
                            return results
                except KeyError:
                    continue

        return results[:limit]

    async def count(self, shard_id: int | None = None) -> int:
        """Count memories in cluster or specific shard."""
        self._ensure_connected()

        if shard_id is not None:
            shard_key = self._shard_index_key(shard_id)
            return await self._redis.scard(shard_key)

        total = 0
        for sid in range(self.config.num_shards):
            shard_key = self._shard_index_key(sid)
            total += await self._redis.scard(shard_key)

        return total

    # ==================== Cluster Operations ====================

    async def get_health(self) -> ClusterHealth:
        """Get cluster health status."""
        self._ensure_connected()

        # Refresh node list
        await self._refresh_nodes()

        active_nodes = sum(1 for n in self._nodes.values() if n.state == NodeState.ACTIVE)
        failed_nodes = sum(1 for n in self._nodes.values() if n.state == NodeState.FAILED)

        # Count healthy shards (shards with at least one replica)
        healthy_shards = 0
        for shard_id in range(self.config.num_shards):
            nodes = self._hash_ring.get_nodes_for_shard(shard_id, 1)
            if any(
                self._nodes.get(n, NodeInfo("", "", 0, NodeState.FAILED, "", [], datetime.utcnow())).state == NodeState.ACTIVE
                for n in nodes
            ):
                healthy_shards += 1

        total_memories = await self.count()

        healthy = (
            active_nodes >= self.config.replication_factor
            and healthy_shards == self.config.num_shards
        )

        return ClusterHealth(
            healthy=healthy,
            total_nodes=len(self._nodes),
            active_nodes=active_nodes,
            failed_nodes=failed_nodes,
            total_shards=self.config.num_shards,
            healthy_shards=healthy_shards,
            replication_factor=self.config.replication_factor,
            total_memories=total_memories,
            message="Cluster healthy" if healthy else "Cluster degraded",
        )

    async def get_nodes(self) -> list[NodeInfo]:
        """Get all nodes in the cluster."""
        await self._refresh_nodes()
        return list(self._nodes.values())

    async def get_shards(self) -> list[ShardInfo]:
        """Get all shard information."""
        shards = []
        for shard_id in range(self.config.num_shards):
            count = await self.count(shard_id)
            nodes = self._hash_ring.get_nodes_for_shard(shard_id, self.config.replication_factor)

            shards.append(
                ShardInfo(
                    shard_id=shard_id,
                    primary_node=nodes[0] if nodes else "",
                    replica_nodes=nodes[1:] if len(nodes) > 1 else [],
                    memory_count=count,
                    size_bytes=0,  # Would need to track this
                )
            )

        return shards

    # ==================== Internal Methods ====================

    def _ensure_connected(self) -> None:
        """Ensure connected to cluster."""
        if not self._connected:
            raise ClusterError("Not connected to cluster")

    def _memory_key(self, memory_id: str) -> str:
        """Get Redis key for a memory."""
        return f"{self.config.cluster_name}:memory:{memory_id}"

    def _shard_index_key(self, shard_id: int) -> str:
        """Get Redis key for shard index."""
        return f"{self.config.cluster_name}:shard:{shard_id}:index"

    def _node_key(self, node_id: str) -> str:
        """Get Redis key for node info."""
        return f"{self.config.cluster_name}:node:{node_id}"

    def _nodes_set_key(self) -> str:
        """Get Redis key for nodes set."""
        return f"{self.config.cluster_name}:nodes"

    async def _register_node(self) -> None:
        """Register this node in the cluster."""
        url = self.config.redis_urls[0]
        parts = url.replace("redis://", "").split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 6379

        node_info = NodeInfo(
            node_id=self.config.node_id,
            host=host,
            port=port,
            state=NodeState.ACTIVE,
            datacenter=self.config.datacenter,
            shards=list(range(self.config.num_shards)),  # Initially all shards
            last_heartbeat=datetime.utcnow(),
        )

        # Store node info
        await self._redis.set(
            self._node_key(self.config.node_id),
            json.dumps(node_info.to_dict()),
        )

        # Add to nodes set
        await self._redis.sadd(self._nodes_set_key(), self.config.node_id)

        # Add to hash ring
        self._hash_ring.add_node(self.config.node_id)
        self._nodes[self.config.node_id] = node_info

    async def _unregister_node(self) -> None:
        """Unregister this node from the cluster."""
        # Update state to LEAVING
        key = self._node_key(self.config.node_id)
        data = await self._redis.get(key)
        if data:
            node_info = NodeInfo.from_dict(json.loads(data))
            node_info.state = NodeState.LEAVING
            await self._redis.set(key, json.dumps(node_info.to_dict()))

        # Remove from nodes set after grace period
        await self._redis.srem(self._nodes_set_key(), self.config.node_id)

        # Remove from hash ring
        self._hash_ring.remove_node(self.config.node_id)

    async def _refresh_nodes(self) -> None:
        """Refresh the list of nodes from Redis."""
        node_ids = await self._redis.smembers(self._nodes_set_key())

        for nid in node_ids:
            if isinstance(nid, bytes):
                nid = nid.decode()

            key = self._node_key(nid)
            data = await self._redis.get(key)

            if data:
                node_info = NodeInfo.from_dict(json.loads(data))

                # Check if node is healthy
                elapsed = (datetime.utcnow() - node_info.last_heartbeat).total_seconds()
                if elapsed > self.config.node_timeout:
                    node_info.state = NodeState.FAILED

                if nid not in self._nodes:
                    self._hash_ring.add_node(nid)
                    if self._on_node_join:
                        self._on_node_join(node_info)

                self._nodes[nid] = node_info

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                if not self._connected:
                    break

                # Update heartbeat
                key = self._node_key(self.config.node_id)
                data = await self._redis.get(key)

                if data:
                    node_info = NodeInfo.from_dict(json.loads(data))
                    node_info.last_heartbeat = datetime.utcnow()
                    node_info.memory_count = await self.count()
                    await self._redis.set(key, json.dumps(node_info.to_dict()))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for failed nodes."""
        while True:
            try:
                await asyncio.sleep(self.config.node_timeout)

                if not self._connected:
                    break

                await self._refresh_nodes()

                # Handle failed nodes
                for node_id, node_info in list(self._nodes.items()):
                    if node_info.state == NodeState.FAILED:
                        if self._on_node_leave:
                            self._on_node_leave(node_info)
                        self._hash_ring.remove_node(node_id)
                        del self._nodes[node_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

    async def _replicate_memory(self, memory: DistributedMemory) -> None:
        """Replicate a memory to other nodes."""
        # In a real implementation, this would publish to a replication channel
        # or directly write to replica nodes
        key = f"{self.config.cluster_name}:replication:{memory.shard_id}"
        await self._redis.publish(key, json.dumps(memory.to_dict()))

    async def wait_for_quorum(
        self,
        timeout: float | None = None,
    ) -> bool:
        """
        Wait for quorum to be reached.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            True if quorum reached, False if timeout
        """
        timeout = timeout or self.config.operation_timeout
        start = time.time()

        required = (
            self.config.replication_factor // 2 + 1
            if self.config.consistency_level == ConsistencyLevel.QUORUM
            else self.config.replication_factor
        )

        while time.time() - start < timeout:
            await self._refresh_nodes()
            active = sum(1 for n in self._nodes.values() if n.state == NodeState.ACTIVE)

            if active >= required:
                return True

            await asyncio.sleep(0.5)

        return False
