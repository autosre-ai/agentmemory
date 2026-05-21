"""Custom exceptions for Team Memory Protocol."""


class TeamMemoryError(Exception):
    """Base exception for team memory errors."""
    pass


class MemoryNotFoundError(TeamMemoryError):
    """Raised when a memory is not found."""
    
    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class BranchNotFoundError(TeamMemoryError):
    """Raised when a branch is not found."""
    
    def __init__(self, branch_name: str):
        self.branch_name = branch_name
        super().__init__(f"Branch not found: {branch_name}")


class BranchExistsError(TeamMemoryError):
    """Raised when trying to create a branch that already exists."""
    
    def __init__(self, branch_name: str):
        self.branch_name = branch_name
        super().__init__(f"Branch already exists: {branch_name}")


class CommitNotFoundError(TeamMemoryError):
    """Raised when a commit is not found."""
    
    def __init__(self, commit_id: str):
        self.commit_id = commit_id
        super().__init__(f"Commit not found: {commit_id}")


class MergeConflictError(TeamMemoryError):
    """Raised when merge conflicts require manual resolution."""
    
    def __init__(self, conflicts: list):
        self.conflicts = conflicts
        super().__init__(
            f"Merge conflicts detected: {len(conflicts)} conflict(s) require manual resolution"
        )


class PermissionDeniedError(TeamMemoryError):
    """Raised when an agent doesn't have permission for an operation."""
    
    def __init__(self, agent_id: str, operation: str, namespace: str | None = None):
        self.agent_id = agent_id
        self.operation = operation
        self.namespace = namespace
        msg = f"Permission denied for agent '{agent_id}' to {operation}"
        if namespace:
            msg += f" in namespace '{namespace}'"
        super().__init__(msg)


class SyncError(TeamMemoryError):
    """Raised when sync operation fails."""
    
    def __init__(self, message: str, errors: list[str] | None = None):
        self.errors = errors or []
        super().__init__(message)


class LockError(TeamMemoryError):
    """Raised when unable to acquire a lock."""
    
    def __init__(self, resource: str, timeout: float | None = None):
        self.resource = resource
        self.timeout = timeout
        msg = f"Failed to acquire lock on '{resource}'"
        if timeout:
            msg += f" within {timeout}s"
        super().__init__(msg)


class NamespaceNotFoundError(TeamMemoryError):
    """Raised when a namespace is not found."""
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        super().__init__(f"Namespace not found: {namespace}")
