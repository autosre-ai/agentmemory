"""Context Compression Engine - Intelligent compression for LLM context windows.

This module provides tools for compressing conversation context to fit within
token budgets while preserving critical information.

Components:
- Context Compression: Compress conversation context within token budgets
- Lossless Compression: Reduce storage with zlib/gzip/brotli/lz4
- Semantic Deduplication: Eliminate duplicate and similar memories
- AI Summarization: Consolidate memories with extractive/abstractive summarization
"""

from .compressor import (
    ContextCompressor,
    CompressionStrategy,
    CompressionResult,
    CompressionConfig,
    CompressionMode,
    Message,
    MessageRole,
)
from .token_counter import TokenCounter
from .importance import ImportanceRanker, ImportanceFactors
from .strategies import (
    TruncateStrategy,
    SummarizeStrategy,
    ExtractKeyFactsStrategy,
    TieredCompressionStrategy,
)

# Lossless compression
from .lossless import (
    CompressionAlgorithm,
    CompressionStats,
    CompressedMemory,
    LosslessCompressor,
    ZlibCompressor,
    GzipCompressor,
    BrotliCompressor,
    Lz4Compressor,
    NoCompressor,
    MemoryCompressionConfig,
    MemoryCompressor,
    compress_memory,
    decompress_memory,
)

# Semantic deduplication
from .semantic import (
    DeduplicationStrategy,
    MemoryItem,
    DuplicateGroup,
    DeduplicationResult,
    DeduplicationMatcher,
    ExactMatcher,
    FuzzyMatcher,
    SemanticMatcher,
    HybridMatcher,
    DeduplicationConfig,
    SemanticDeduplicator,
    find_duplicates,
    deduplicate_memories,
)

# AI summarization
from .summarization import (
    SummarizationStrategy,
    SummaryLevel,
    MemoryEntry,
    Summary,
    HierarchicalSummary,
    SummarizationResult,
    Summarizer,
    ExtractiveSummarizer,
    AbstractiveSummarizer,
    HierarchicalSummarizer,
    IncrementalSummarizer,
    SummarizationConfig,
    MemorySummarizer,
    summarize_memories,
    create_hierarchical_summary,
)

__all__ = [
    # Main compressor
    "ContextCompressor",
    "CompressionStrategy",
    "CompressionResult",
    "CompressionConfig",
    "CompressionMode",
    "Message",
    "MessageRole",
    # Token counting
    "TokenCounter",
    # Importance ranking
    "ImportanceRanker",
    "ImportanceFactors",
    # Compression strategies
    "TruncateStrategy",
    "SummarizeStrategy",
    "ExtractKeyFactsStrategy",
    "TieredCompressionStrategy",
    # Lossless compression
    "CompressionAlgorithm",
    "CompressionStats",
    "CompressedMemory",
    "LosslessCompressor",
    "ZlibCompressor",
    "GzipCompressor",
    "BrotliCompressor",
    "Lz4Compressor",
    "NoCompressor",
    "MemoryCompressionConfig",
    "MemoryCompressor",
    "compress_memory",
    "decompress_memory",
    # Semantic deduplication
    "DeduplicationStrategy",
    "MemoryItem",
    "DuplicateGroup",
    "DeduplicationResult",
    "DeduplicationMatcher",
    "ExactMatcher",
    "FuzzyMatcher",
    "SemanticMatcher",
    "HybridMatcher",
    "DeduplicationConfig",
    "SemanticDeduplicator",
    "find_duplicates",
    "deduplicate_memories",
    # AI summarization
    "SummarizationStrategy",
    "SummaryLevel",
    "MemoryEntry",
    "Summary",
    "HierarchicalSummary",
    "SummarizationResult",
    "Summarizer",
    "ExtractiveSummarizer",
    "AbstractiveSummarizer",
    "HierarchicalSummarizer",
    "IncrementalSummarizer",
    "SummarizationConfig",
    "MemorySummarizer",
    "summarize_memories",
    "create_hierarchical_summary",
]
