# Distributed Clustering Guide

This guide covers horizontal scaling of Agent Memory Toolkit across multiple nodes using Redis Cluster.

## Overview

The clustering module provides enterprise-ready distributed memory with:

- **Redis Cluster backend** - Distributed storage with automatic sharding
- **Consistent hashing** - Even distribution of memories across nodes
- **Vector clocks** - Causality tracking for conflict resolution
- **Leader election** - Coordinated writes with fencing tokens
- **Memory replication** - Configurable durability and read performance
- **Automatic failover** - High availability with quorum-based writes

## Installation

Install with Redis support:

```bash
pip install agent-memory-toolkit[cluster]

# Or with Redis extras
pip install agent-memory-toolkit redis[hiredis]
```

## Quick Start

### Basic Distributed Store

```python
import asyncio
from agent_memory_toolkit.cluster import (
    DistributedMemoryStore,
    ClusterConfig,
    ConsistencyLevel,
)

async def main():
    # Configure cluster
    config = ClusterConfig(
        redis_urls=["redis://node1:6379", "redis://node2:6379", "redis://node3:6379"],
        node_id="node-1",
        cluster_name="my-agent-cluster",
        replication_factor=2,
        consistency_level=ConsistencyLevel.QUORUM,
    )

    # Connect to cluster
    async with DistributedMemoryStore(config) as store:
        # Add memories (automatically sharded and replicated)
        memory = await store.add(
            "The quarterly report is due on Friday",
            metadata={"category": "deadlines", "priority": "high"},
        )
        print(f"Created memory: {memory.id} on shard {memory.shard_id}")

        # Search across all shards
        results = await store.search("quarterly report")
        for r in results:
            print(f"Found: {r.content}")

        # Check cluster health
        health = await store.get_health()
        print(f"Cluster healthy: {health.healthy}")
        print(f"Active nodes: {health.active_nodes}/{health.total_nodes}")

asyncio.run(main())
```

### Multi-Node Synchronization

```python
from agent_memory_toolkit.cluster import (
    ClusterSync,
    SyncConfig,
    SyncEventType,
    ConflictResolution,
)

async def main():
    config = SyncConfig(
        node_id="node-1",
        sync_interval=5.0,  # Sync every 5 seconds
        conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
    )

    async with ClusterSync("redis://localhost:6379", config) as sync:
        # Record local changes
        await sync.record_change(
            memory_id="mem-123",
            change_type=SyncEventType.MEMORY_UPDATED,
            data={"content": "Updated content"},
        )

        # Manually trigger sync
        result = await sync.sync_with_peers()
        print(f"Sent: {result.events_sent}, Received: {result.events_received}")
        print(f"Conflicts detected: {result.conflicts_detected}")

asyncio.run(main())
```

### Leader Election for Coordinated Writes

```python
from agent_memory_toolkit.cluster import (
    LeaderElection,
    LeaderConfig,
)

async def on_become_leader():
    print("This node is now the leader!")

async def on_lose_leadership():
    print("Leadership lost, switching to follower mode")

async def main():
    config = LeaderConfig(
        node_id="node-1",
        election_timeout=10.0,
        heartbeat_interval=3.0,
        on_become_leader=on_become_leader,
        on_lose_leadership=on_lose_leadership,
    )

    async with LeaderElection("redis://localhost:6379", config) as election:
        # Check if we're the leader
        if election.is_leader:
            print(f"I am leader! Term: {election.term}")
            print(f"Fencing token: {election.fencing_token}")

            # Perform coordinated write
            # ...

        # Wait for leadership (or perform follower duties)
        await asyncio.sleep(60)

asyncio.run(main())
```

## Configuration Reference

### ClusterConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `redis_urls` | `list[str]` | `["redis://localhost:6379"]` | Redis node URLs |
| `redis_password` | `str | None` | `None` | Redis password |
| `redis_ssl` | `bool` | `False` | Enable SSL/TLS |
| `node_id` | `str` | Auto-generated | Unique node identifier |
| `cluster_name` | `str` | `"agent-memory-cluster"` | Cluster namespace |
| `datacenter` | `str` | `"default"` | Datacenter for locality |
| `replication_factor` | `int` | `2` | Number of copies per memory |
| `consistency_level` | `ConsistencyLevel` | `QUORUM` | Read/write consistency |
| `num_shards` | `int` | `16` | Number of shards |
| `virtual_nodes` | `int` | `150` | Virtual nodes for consistent hashing |
| `connection_timeout` | `float` | `5.0` | Connection timeout (seconds) |
| `operation_timeout` | `float` | `30.0` | Operation timeout (seconds) |
| `heartbeat_interval` | `float` | `5.0` | Node heartbeat interval |
| `node_timeout` | `float` | `15.0` | Time before node is marked failed |

### ConsistencyLevel

| Level | Description |
|-------|-------------|
| `ONE` | Write/read from one node (fastest, lowest durability) |
| `QUORUM` | Majority of nodes must acknowledge (balanced) |
| `ALL` | All nodes must acknowledge (slowest, highest durability) |
| `LOCAL_QUORUM` | Quorum within local datacenter |

### SyncConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_id` | `str` | Auto-generated | Unique node identifier |
| `sync_interval` | `float` | `5.0` | Seconds between sync cycles |
| `batch_size` | `int` | `100` | Events per sync batch |
| `conflict_resolution` | `ConflictResolution` | `LAST_WRITE_WINS` | Conflict resolution strategy |
| `max_retries` | `int` | `3` | Max retry attempts |
| `event_retention_seconds` | `int` | `3600` | Event log retention |

### ConflictResolution

| Strategy | Description |
|----------|-------------|
| `LAST_WRITE_WINS` | Most recently updated memory wins |
| `FIRST_WRITE_WINS` | First update wins (preserves original) |
| `HIGHEST_VERSION_WINS` | Memory with highest version number wins |
| `MERGE` | Attempt to merge changes (last values override) |
| `MANUAL` | Raise exception for manual resolution |
| `CUSTOM` | Use custom resolver function |

### ReplicationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `replication_factor` | `int` | `2` | Total copies including primary |
| `strategy` | `ReplicationStrategy` | `SEMI_SYNC` | Replication strategy |
| `min_sync_replicas` | `int` | `1` | Minimum replicas to ack |
| `ack_timeout` | `float` | `5.0` | Timeout for replica acks |
| `auto_failover` | `bool` | `True` | Enable automatic failover |

### ReplicationStrategy

| Strategy | Description |
|----------|-------------|
| `SYNC` | Wait for all replicas before returning |
| `ASYNC` | Return immediately, replicate in background |
| `SEMI_SYNC` | Wait for quorum, async for remaining |
| `CHAIN` | Chain replication (primary -> r1 -> r2) |

## Architecture

### Consistent Hashing

The cluster uses consistent hashing with virtual nodes to distribute memories evenly:

```
                    ┌──────────────────────────────────────┐
                    │           Hash Ring                   │
                    │                                       │
                    │    ┌──────┐                          │
                    │    │ VN1  │───┐                      │
                    │    └──────┘   │                      │
                    │               ▼                      │
                    │  ┌──────┐  ┌──────────┐  ┌──────┐   │
                    │  │ VN4  │  │  Node 1  │  │ VN2  │   │
                    │  └──────┘  └──────────┘  └──────┘   │
                    │     │                        │       │
                    │     │    ┌──────────┐       │       │
                    │     └───▶│  Node 2  │◀──────┘       │
                    │          └──────────┘                │
                    │               │                      │
                    │               ▼                      │
                    │          ┌──────────┐                │
                    │          │  Node 3  │                │
                    │          └──────────┘                │
                    └──────────────────────────────────────┘

Memory ID hash determines placement on ring
Virtual nodes (VN) provide even distribution
```

### Vector Clocks

Vector clocks track causality across nodes:

```python
# Node 1 creates memory
vc1 = {"node-1": 1}

# Node 2 updates (concurrent)
vc2 = {"node-2": 1}

# These are concurrent - conflict!
# Resolution needed

# After merge
merged = {"node-1": 1, "node-2": 1}
```

### Leader Election

Redis-based distributed locks with fencing:

```
┌─────────────┐    Leader Lock    ┌─────────────┐
│   Node 1    │◄─────────────────►│    Redis    │
│  (Leader)   │    heartbeat      │             │
└─────────────┘                   └─────────────┘
       │
       │ Fencing Token: 42
       ▼
┌─────────────┐
│  Writes OK  │
│  token=42   │
└─────────────┘

If leadership changes:
- New leader gets token 43
- Old writes with token 42 are rejected
```

## Deployment Patterns

### Single Datacenter

```python
config = ClusterConfig(
    redis_urls=[
        "redis://redis1:6379",
        "redis://redis2:6379",
        "redis://redis3:6379",
    ],
    replication_factor=3,
    consistency_level=ConsistencyLevel.QUORUM,
)
```

### Multi-Datacenter

```python
# Datacenter 1 nodes
config_dc1 = ClusterConfig(
    redis_urls=["redis://redis-dc1-1:6379"],
    datacenter="dc1",
    consistency_level=ConsistencyLevel.LOCAL_QUORUM,
)

# Datacenter 2 nodes
config_dc2 = ClusterConfig(
    redis_urls=["redis://redis-dc2-1:6379"],
    datacenter="dc2",
    consistency_level=ConsistencyLevel.LOCAL_QUORUM,
)
```

### Redis Sentinel (High Availability)

```python
config = ClusterConfig(
    redis_urls=[
        "redis://sentinel1:26379",
        "redis://sentinel2:26379",
        "redis://sentinel3:26379",
    ],
    use_sentinel=True,
    sentinel_master="mymaster",
)
```

## Monitoring

### Health Checks

```python
async def health_check():
    health = await store.get_health()

    # Prometheus-style metrics
    print(f"cluster_healthy {1 if health.healthy else 0}")
    print(f"cluster_nodes_total {health.total_nodes}")
    print(f"cluster_nodes_active {health.active_nodes}")
    print(f"cluster_nodes_failed {health.failed_nodes}")
    print(f"cluster_shards_healthy {health.healthy_shards}")
    print(f"cluster_memories_total {health.total_memories}")
```

### Replication Metrics

```python
metrics = replication_manager.metrics

print(f"replication_total {metrics.total_replicated}")
print(f"replication_acks {metrics.total_acks}")
print(f"replication_failures {metrics.total_failures}")
print(f"replication_latency_ms {metrics.avg_replication_latency_ms}")
print(f"replicas_in_sync {metrics.replicas_in_sync}")
print(f"replicas_lagging {metrics.replicas_lagging}")
```

### Sync Metrics

```python
metrics = cluster_sync.metrics

print(f"sync_total {metrics.total_syncs}")
print(f"sync_successful {metrics.successful_syncs}")
print(f"sync_events_sent {metrics.total_events_sent}")
print(f"sync_events_received {metrics.total_events_received}")
print(f"sync_conflicts {metrics.total_conflicts}")
```

## Error Handling

### Common Exceptions

```python
from agent_memory_toolkit.cluster import (
    ClusterError,
    QuorumNotReachedError,
    LeaderElectionError,
    ReplicationError,
    SyncConflictError,
)

try:
    await store.add("Important memory")
except QuorumNotReachedError as e:
    print(f"Quorum not reached: need {e.required}, have {e.available}")
    # Maybe retry with lower consistency
except ClusterConnectionError as e:
    print(f"Connection failed: {e}")
    # Reconnect or failover
except ReplicationError as e:
    print(f"Replication failed to {e.target_node}")
    # Check replica health
```

### Graceful Degradation

```python
async def add_with_fallback(content: str):
    try:
        # Try with quorum
        return await store.add(content)
    except QuorumNotReachedError:
        # Fall back to single node
        store.config.consistency_level = ConsistencyLevel.ONE
        return await store.add(content)
```

## Best Practices

### 1. Choose Appropriate Consistency

- Use `QUORUM` for most workloads (balance of speed and durability)
- Use `ALL` for critical data that must not be lost
- Use `ONE` for high-throughput, loss-tolerant data

### 2. Size Your Cluster

- Minimum 3 nodes for quorum (can lose 1 node)
- 5 nodes for better fault tolerance (can lose 2 nodes)
- Use `replication_factor >= 2` for any production data

### 3. Monitor Replication Lag

```python
async def check_replicas():
    for replica in replication_manager.replicas.values():
        if replica.lag_messages > 1000:
            alert(f"Replica {replica.node_id} is lagging!")
```

### 4. Handle Leadership Changes

```python
config = LeaderConfig(
    on_become_leader=lambda: start_coordination(),
    on_lose_leadership=lambda: stop_coordination(),
)
```

### 5. Use Fencing Tokens

```python
# Include fencing token in distributed requests
headers = election.get_fencing_header()
# {"X-Fencing-Token": "42"}
```

## Troubleshooting

### Node Won't Join Cluster

1. Check Redis connectivity: `redis-cli -h <host> ping`
2. Verify firewall rules allow Redis port (default 6379)
3. Check `cluster_name` matches across all nodes
4. Ensure clock sync (NTP) across nodes

### High Replication Lag

1. Check network latency between nodes
2. Increase `batch_size` for better throughput
3. Consider `ASYNC` replication for non-critical data
4. Monitor Redis memory and CPU

### Split Brain

1. Always use odd number of nodes (3, 5, 7)
2. Enable fencing tokens
3. Use `QUORUM` consistency
4. Monitor leader election events

### Conflicts During Sync

1. Choose appropriate `ConflictResolution` strategy
2. Implement custom resolver for complex merges
3. Monitor `conflicts_detected` metric
4. Review conflict logs for patterns

## API Reference

See the full API documentation:

- [DistributedMemoryStore](api-reference.md#distributedmemorystore)
- [ClusterSync](api-reference.md#clustersync)
- [LeaderElection](api-reference.md#leaderelection)
- [ReplicationManager](api-reference.md#replicationmanager)
