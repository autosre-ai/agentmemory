"""
Distributed Clustering Module for Agent Memory Toolkit.

Enterprise-ready horizontal scaling with:
- Redis Cluster-backed distributed memory
- Multi-node synchronization with vector clocks
- Leader election for write coordination
- Memory replication across nodes
- Automatic failover and recovery

Example:
    >>> from agent_memory_toolkit.cluster import (
    ...     DistributedMemoryStore,
    ...     ClusterConfig,
    ...     LeaderElection,
    ... )
    >>>
    >>> # Create a distributed memory store
    >>> config = ClusterConfig(
    ...     redis_urls=["redis://node1:6379", "redis://node2:6379"],
    ...     node_id="node-1",
    ...     replication_factor=2,
    ... )
    >>> store = DistributedMemoryStore(config)
    >>>
    >>> # Add memories (automatically replicated)
    >>> memory = await store.add("Important fact")
    >>>
    >>> # Leader election for coordinated writes
    >>> async with LeaderElection(config) as leader:
    ...     if leader.is_leader:
    ...         await store.batch_add(memories)
"""

from .distributed import (
    DistributedMemoryStore,
    ClusterConfig,
    NodeInfo,
    NodeState,
    ClusterHealth,
    DistributedMemory,
    ShardInfo,
    ConsistencyLevel,
)

from .sync import (
    ClusterSync,
    SyncConfig,
    SyncState,
    SyncEvent,
    SyncEventType,
    VectorClock,
    ConflictResolution,
    SyncResult,
    SyncMetrics,
)

from .leader_election import (
    LeaderElection,
    LeaderConfig,
    LeaderState,
    LeaderEvent,
    LeaderEventType,
)

from .replication import (
    ReplicationManager,
    ReplicationConfig,
    ReplicaInfo,
    ReplicationState,
    ReplicationEvent,
    ReplicationEventType,
    ReplicationStrategy,
)

from .exceptions import (
    ClusterError,
    NodeNotFoundError,
    LeaderElectionError,
    ReplicationError,
    SyncConflictError,
    QuorumNotReachedError,
    ClusterConnectionError,
    ShardMigrationError,
)

__all__ = [
    # Distributed Store
    "DistributedMemoryStore",
    "ClusterConfig",
    "NodeInfo",
    "NodeState",
    "ClusterHealth",
    "DistributedMemory",
    "ShardInfo",
    "ConsistencyLevel",
    # Sync
    "ClusterSync",
    "SyncConfig",
    "SyncState",
    "SyncEvent",
    "SyncEventType",
    "VectorClock",
    "ConflictResolution",
    "SyncResult",
    "SyncMetrics",
    # Leader Election
    "LeaderElection",
    "LeaderConfig",
    "LeaderState",
    "LeaderEvent",
    "LeaderEventType",
    # Replication
    "ReplicationManager",
    "ReplicationConfig",
    "ReplicaInfo",
    "ReplicationState",
    "ReplicationEvent",
    "ReplicationEventType",
    "ReplicationStrategy",
    # Exceptions
    "ClusterError",
    "NodeNotFoundError",
    "LeaderElectionError",
    "ReplicationError",
    "SyncConflictError",
    "QuorumNotReachedError",
    "ClusterConnectionError",
    "ShardMigrationError",
]
