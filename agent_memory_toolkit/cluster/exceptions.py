"""Exceptions for the distributed clustering module."""

from __future__ import annotations


class ClusterError(Exception):
    """Base exception for cluster-related errors."""

    def __init__(self, message: str, node_id: str | None = None):
        super().__init__(message)
        self.node_id = node_id
        self.message = message


class NodeNotFoundError(ClusterError):
    """Raised when a node is not found in the cluster."""

    def __init__(self, node_id: str):
        super().__init__(f"Node not found: {node_id}", node_id=node_id)


class LeaderElectionError(ClusterError):
    """Raised when leader election fails."""

    def __init__(self, message: str = "Failed to elect leader"):
        super().__init__(message)


class ReplicationError(ClusterError):
    """Raised when replication fails."""

    def __init__(
        self,
        message: str,
        source_node: str | None = None,
        target_node: str | None = None,
        memory_id: str | None = None,
    ):
        super().__init__(message)
        self.source_node = source_node
        self.target_node = target_node
        self.memory_id = memory_id


class SyncConflictError(ClusterError):
    """Raised when there's a conflict during synchronization."""

    def __init__(
        self,
        message: str,
        memory_id: str,
        local_version: int,
        remote_version: int,
    ):
        super().__init__(message)
        self.memory_id = memory_id
        self.local_version = local_version
        self.remote_version = remote_version


class QuorumNotReachedError(ClusterError):
    """Raised when quorum cannot be reached for an operation."""

    def __init__(
        self,
        required: int,
        available: int,
        operation: str = "write",
    ):
        super().__init__(
            f"Quorum not reached for {operation}: required {required}, available {available}"
        )
        self.required = required
        self.available = available
        self.operation = operation


class ClusterConnectionError(ClusterError):
    """Raised when connection to cluster fails."""

    def __init__(self, message: str, host: str | None = None, port: int | None = None):
        super().__init__(message)
        self.host = host
        self.port = port


class ShardMigrationError(ClusterError):
    """Raised when shard migration fails."""

    def __init__(
        self,
        message: str,
        shard_id: int,
        source_node: str,
        target_node: str,
    ):
        super().__init__(message)
        self.shard_id = shard_id
        self.source_node = source_node
        self.target_node = target_node
