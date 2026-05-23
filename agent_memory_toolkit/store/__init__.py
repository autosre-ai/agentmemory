"""Memory store module with SQLite + FTS5 backend."""

from .memory_store import MemoryStore, HybridSearchConfig, SearchMethod
from .models import Memory, MemoryMetadata, SearchResult, Branch, Commit
from .exceptions import (
    MemoryStoreError,
    MemoryNotFoundError,
    BranchNotFoundError,
    CommitNotFoundError,
    MergeConflictError,
)
from .embeddings import (
    EmbeddingProvider,
    RerankerProvider,
    SentenceTransformerProvider,
    CrossEncoderReranker,
    EmbeddingConfig,
    RerankerConfig,
    SimilarityMetric,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    CROSS_ENCODER_AVAILABLE,
    embedding_to_blob,
    blob_to_embedding,
    cosine_similarity,
    batch_cosine_similarity,
)

__all__ = [
    # Store
    "MemoryStore",
    "HybridSearchConfig",
    "SearchMethod",
    # Models
    "Memory",
    "MemoryMetadata",
    "SearchResult",
    "Branch",
    "Commit",
    # Exceptions
    "MemoryStoreError",
    "MemoryNotFoundError",
    "BranchNotFoundError",
    "CommitNotFoundError",
    "MergeConflictError",
    # Embeddings
    "EmbeddingProvider",
    "RerankerProvider",
    "SentenceTransformerProvider",
    "CrossEncoderReranker",
    "EmbeddingConfig",
    "RerankerConfig",
    "SimilarityMetric",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "CROSS_ENCODER_AVAILABLE",
    "embedding_to_blob",
    "blob_to_embedding",
    "cosine_similarity",
    "batch_cosine_similarity",
]
