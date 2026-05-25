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
    # Search module
    "search",
    # Temporal module
    "temporal",
    # Observability module
    "observability",
    # Graph module
    "graph",
    # IO/Persistence module
    "io",
    # LLM module
    "llm",
]

# Lazy import for hermes_plugin, cluster, and search to avoid circular imports
import importlib as _importlib
import sys as _sys

_LAZY_MODULES = {"hermes_plugin", "cluster", "search", "temporal", "observability", "graph", "io", "llm"}
_LAZY_LOADING = set()  # Prevent recursion during loading

def __getattr__(name: str):
    if name not in _LAZY_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    # Prevent recursion
    if name in _LAZY_LOADING:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    _LAZY_LOADING.add(name)
    try:
        module = _importlib.import_module(f".{name}", __name__)
        # Register in sys.modules and as module attribute
        setattr(_sys.modules[__name__], name, module)
        return module
    finally:
        _LAZY_LOADING.discard(name)
