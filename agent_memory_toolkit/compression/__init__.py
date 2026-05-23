"""Context Compression Engine - Intelligent compression for LLM context windows.

This module provides tools for compressing conversation context to fit within
token budgets while preserving critical information.
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
]
