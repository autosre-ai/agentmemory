"""Agent Memory Toolkit - Local-first memory layer for AI agents."""

__version__ = "0.2.0"

# Import from store module
from .store import (
    MemoryStore,
    Memory as StoreMemory,
    MemoryMetadata,
    SearchResult,
    Branch,
    Commit,
    MemoryStoreError,
    MemoryNotFoundError as StoreMemoryNotFoundError,
    BranchNotFoundError as StoreBranchNotFoundError,
    CommitNotFoundError,
    MergeConflictError as StoreMergeConflictError,
)

# Import from extraction module
from .extraction import (
    CognitiveDomain,
    Memory,
    ExtractionResult,
    MemoryExtractor,
    RuleBasedExtractor,
    LLMExtractor,
    MemoryDeduplicator,
    ConflictResolver,
    MemoryMerger,
)

# Import from compression module
from .compression import (
    ContextCompressor,
    CompressionStrategy,
    CompressionResult,
    CompressionConfig,
    CompressionMode,
    Message,
    MessageRole,
    TokenCounter,
    ImportanceRanker,
    ImportanceFactors,
    TruncateStrategy,
    SummarizeStrategy,
    ExtractKeyFactsStrategy,
    TieredCompressionStrategy,
)

# Import from security module
from .security import (
    MemoryGuard,
    ValidationResult,
    SecurityLevel,
    SecurityConfig,
    PoisonDetector,
    InjectionPattern,
    DetectionResult,
    ConfidenceScorer,
    UncertaintyDetector,
    ConfidenceResult,
    SourceValidator,
    SourceTrust,
    SourceValidationResult,
    AuditLogger,
    AuditEvent,
    AuditEventType,
)

# Import from team module
from .team import (
    TeamMemoryStore,
    TeamMemory,
    TeamMemoryMetadata,
    TeamBranch,
    TeamCommit,
    ConflictResolution,
    Permission,
    MergeConflict,
    SyncResult,
    Event,
    EventType,
    EventHook,
    AccessRule,
    AccessControl,
    SyncProtocol,
    TeamMemoryError,
    MemoryNotFoundError as TeamMemoryNotFoundError,
    BranchNotFoundError as TeamBranchNotFoundError,
    BranchExistsError,
    CommitNotFoundError as TeamCommitNotFoundError,
    MergeConflictError as TeamMergeConflictError,
    PermissionDeniedError,
    SyncError,
    LockError,
    NamespaceNotFoundError,
)

__all__ = [
    # Version
    "__version__",
    # Store
    "MemoryStore",
    "StoreMemory",
    "MemoryMetadata",
    "SearchResult",
    "Branch",
    "Commit",
    "MemoryStoreError",
    "StoreMemoryNotFoundError",
    "StoreBranchNotFoundError",
    "CommitNotFoundError",
    "StoreMergeConflictError",
    # Extraction
    "CognitiveDomain",
    "Memory",
    "ExtractionResult",
    "MemoryExtractor",
    "RuleBasedExtractor",
    "LLMExtractor",
    "MemoryDeduplicator",
    "ConflictResolver",
    "MemoryMerger",
    # Compression
    "ContextCompressor",
    "CompressionStrategy",
    "CompressionResult",
    "CompressionConfig",
    "CompressionMode",
    "Message",
    "MessageRole",
    "TokenCounter",
    "ImportanceRanker",
    "ImportanceFactors",
    "TruncateStrategy",
    "SummarizeStrategy",
    "ExtractKeyFactsStrategy",
    "TieredCompressionStrategy",
    # Security
    "MemoryGuard",
    "ValidationResult",
    "SecurityLevel",
    "SecurityConfig",
    "PoisonDetector",
    "InjectionPattern",
    "DetectionResult",
    "ConfidenceScorer",
    "UncertaintyDetector",
    "ConfidenceResult",
    "SourceValidator",
    "SourceTrust",
    "SourceValidationResult",
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    # Team
    "TeamMemoryStore",
    "TeamMemory",
    "TeamMemoryMetadata",
    "TeamBranch",
    "TeamCommit",
    "ConflictResolution",
    "Permission",
    "MergeConflict",
    "SyncResult",
    "Event",
    "EventType",
    "EventHook",
    "AccessRule",
    "AccessControl",
    "SyncProtocol",
    "TeamMemoryError",
    "TeamMemoryNotFoundError",
    "TeamBranchNotFoundError",
    "BranchExistsError",
    "TeamCommitNotFoundError",
    "TeamMergeConflictError",
    "PermissionDeniedError",
    "SyncError",
    "LockError",
    "NamespaceNotFoundError",
    # Hermes plugin
    "hermes_plugin",
]

# Lazy import for hermes_plugin to avoid circular imports
def __getattr__(name: str):
    if name == "hermes_plugin":
        from . import hermes_plugin
        return hermes_plugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
