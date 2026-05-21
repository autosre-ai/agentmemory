"""Custom exceptions for the memory store."""


class MemoryStoreError(Exception):
    """Base exception for memory store errors."""

    pass


class MemoryNotFoundError(MemoryStoreError):
    """Raised when a memory is not found."""

    def __init__(self, memory_id: str):
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class BranchNotFoundError(MemoryStoreError):
    """Raised when a branch is not found."""

    def __init__(self, branch_name: str):
        self.branch_name = branch_name
        super().__init__(f"Branch not found: {branch_name}")


class CommitNotFoundError(MemoryStoreError):
    """Raised when a commit is not found."""

    def __init__(self, commit_id: str):
        self.commit_id = commit_id
        super().__init__(f"Commit not found: {commit_id}")


class MergeConflictError(MemoryStoreError):
    """Raised when there's a merge conflict between branches."""

    def __init__(self, source_branch: str, target_branch: str, conflicts: list[str]):
        self.source_branch = source_branch
        self.target_branch = target_branch
        self.conflicts = conflicts
        super().__init__(
            f"Merge conflict between '{source_branch}' and '{target_branch}': {conflicts}"
        )
