"""Context Compression Engine - Main compressor class.

Provides the ContextCompressor class for intelligent context compression
with configurable strategies, token budgets, and importance ranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, Union

from .token_counter import TokenCounter
from .importance import ImportanceRanker, ScoredMessage, ImportanceFactors
from .strategies import (
    CompressionStrategy,
    CompressionResult,
    TruncateStrategy,
    SummarizeStrategy,
    ExtractKeyFactsStrategy,
    TieredCompressionStrategy,
)


class MessageRole(str, Enum):
    """Standard message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Structured message for compression."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Compression-related flags
    is_critical: bool = False  # Never compress this message
    importance_boost: float = 0.0  # Custom importance boost
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to standard message dict."""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.metadata:
            d["metadata"] = self.metadata
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """Create from dict."""
        role = MessageRole(d.get("role", "user"))
        return cls(
            role=role,
            content=d.get("content", ""),
            name=d.get("name"),
            metadata=d.get("metadata", {}),
        )


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion."""
        ...


class CompressionMode(str, Enum):
    """Compression mode presets."""
    AGGRESSIVE = "aggressive"  # Maximum compression, may lose some info
    BALANCED = "balanced"  # Balance between compression and retention
    CONSERVATIVE = "conservative"  # Preserve more information
    LOSSLESS = "lossless"  # Only truncate, no summarization


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    
    # Token budget
    max_tokens: int = 4000
    reserve_tokens: int = 500  # Reserve for response
    
    # Strategy selection
    mode: CompressionMode = CompressionMode.BALANCED
    strategy: Optional[str] = None  # Override: truncate, summarize, extract_key_facts, tiered
    
    # Preservation settings
    preserve_system: bool = True
    preserve_recent: int = 4  # Recent messages to keep intact
    preserve_critical: bool = True  # Messages with critical markers
    
    # Tiered compression settings
    recent_count: int = 4  # Full fidelity
    medium_count: int = 8  # Summarized
    
    # Message limits
    max_message_tokens: int = 1000  # Truncate long individual messages
    
    # Custom settings
    critical_patterns: Optional[list[str]] = None
    
    @property
    def effective_budget(self) -> int:
        """Token budget minus reserve."""
        return self.max_tokens - self.reserve_tokens
    
    def get_strategy_name(self) -> str:
        """Get strategy name based on mode or override."""
        if self.strategy:
            return self.strategy
        
        return {
            CompressionMode.AGGRESSIVE: "tiered",
            CompressionMode.BALANCED: "tiered",
            CompressionMode.CONSERVATIVE: "summarize",
            CompressionMode.LOSSLESS: "truncate",
        }.get(self.mode, "tiered")


class ContextCompressor:
    """Intelligent context compressor for LLM conversations.
    
    Provides multiple strategies for compressing conversation context
    to fit within token budgets while preserving critical information.
    
    Features:
    - Multiple compression strategies (truncate, summarize, extract, tiered)
    - Importance-based ranking of messages
    - Critical information preservation
    - Configurable token budgets
    - Support for both rule-based and LLM-based compression
    
    Example:
        >>> compressor = ContextCompressor(max_tokens=4000)
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ...     # ... many more messages
        ... ]
        >>> result = compressor.compress(messages)
        >>> print(f"Compressed {result.original_tokens} -> {result.compressed_tokens}")
        >>> print(result.messages)
    
    Advanced usage with LLM:
        >>> from my_llm import MyLLMProvider
        >>> compressor = ContextCompressor(
        ...     max_tokens=8000,
        ...     llm_provider=MyLLMProvider(),
        ...     mode=CompressionMode.BALANCED,
        ... )
        >>> result = compressor.compress(messages)
    """
    
    # Critical information markers to always preserve
    CRITICAL_MARKERS = [
        r"\[CRITICAL\]",
        r"\[IMPORTANT\]",
        r"\[REMEMBER\]",
        r"\[PRESERVE\]",
        r"\[KEY\]",
    ]
    
    def __init__(
        self,
        max_tokens: int = 4000,
        model: str = "gpt-4",
        llm_provider: Optional[LLMProvider] = None,
        mode: CompressionMode = CompressionMode.BALANCED,
        config: Optional[CompressionConfig] = None,
    ):
        """Initialize the context compressor.
        
        Args:
            max_tokens: Maximum tokens for compressed context
            model: Model name for token counting
            llm_provider: Optional LLM for summarization
            mode: Compression mode preset
            config: Full configuration (overrides other params)
        """
        # Create or use config
        if config:
            self.config = config
        else:
            self.config = CompressionConfig(
                max_tokens=max_tokens,
                mode=mode,
            )
        
        # Initialize components
        self.token_counter = TokenCounter(model=model)
        self.importance_ranker = ImportanceRanker(
            critical_patterns=self.config.critical_patterns,
        )
        self.llm_provider = llm_provider
        
        # Build critical marker regex
        patterns = self.CRITICAL_MARKERS + (self.config.critical_patterns or [])
        self._critical_re = re.compile(
            "|".join(patterns),
            re.IGNORECASE
        )
        
        # Initialize strategies
        self._strategies: dict[str, CompressionStrategy] = {
            "truncate": TruncateStrategy(
                preserve_system=self.config.preserve_system,
                preserve_recent=self.config.preserve_recent,
                max_message_tokens=self.config.max_message_tokens,
            ),
            "summarize": SummarizeStrategy(
                llm_provider=llm_provider,
                preserve_system=self.config.preserve_system,
                preserve_recent=self.config.preserve_recent,
            ),
            "extract_key_facts": ExtractKeyFactsStrategy(
                llm_provider=llm_provider,
                preserve_system=self.config.preserve_system,
                preserve_recent=self.config.preserve_recent,
            ),
            "tiered": TieredCompressionStrategy(
                llm_provider=llm_provider,
                recent_count=self.config.recent_count,
                medium_count=self.config.medium_count,
                preserve_system=self.config.preserve_system,
                importance_ranker=self.importance_ranker,
            ),
        }
    
    def _is_critical(self, message: dict) -> bool:
        """Check if message contains critical markers."""
        content = message.get("content", "")
        return bool(self._critical_re.search(content))
    
    def _extract_critical_info(self, messages: list[dict]) -> list[str]:
        """Extract critical information that must be preserved."""
        critical_info = []
        
        for msg in messages:
            content = msg.get("content", "")
            for match in self._critical_re.finditer(content):
                # Get the line containing the critical marker
                start = content.rfind("\n", 0, match.start()) + 1
                end = content.find("\n", match.end())
                if end == -1:
                    end = len(content)
                line = content[start:end].strip()
                if line:
                    critical_info.append(line)
        
        return critical_info
    
    def _preserve_critical_messages(
        self,
        messages: list[dict],
        compressed: list[dict],
    ) -> list[dict]:
        """Ensure critical information is preserved in output."""
        if not self.config.preserve_critical:
            return compressed
        
        # Extract critical info from original
        critical_info = self._extract_critical_info(messages)
        
        if not critical_info:
            return compressed
        
        # Check if critical info is in compressed output
        compressed_text = " ".join(m.get("content", "") for m in compressed)
        missing_critical = [
            info for info in critical_info
            if info not in compressed_text
        ]
        
        if missing_critical:
            # Add critical info as a system message
            critical_message = {
                "role": "system",
                "content": "[PRESERVED CRITICAL INFO]\n" + "\n".join(missing_critical),
            }
            # Insert after first system message or at start
            insert_idx = 0
            for i, msg in enumerate(compressed):
                if msg.get("role") == "system":
                    insert_idx = i + 1
                    break
            compressed.insert(insert_idx, critical_message)
        
        return compressed
    
    def count_tokens(self, messages: list[dict]) -> int:
        """Count tokens in messages.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Total token count
        """
        return self.token_counter.count_messages(messages)
    
    def needs_compression(self, messages: list[dict]) -> bool:
        """Check if messages need compression.
        
        Args:
            messages: List of message dicts
            
        Returns:
            True if token count exceeds budget
        """
        return self.count_tokens(messages) > self.config.effective_budget
    
    def rank_messages(
        self,
        messages: list[dict],
    ) -> list[ScoredMessage]:
        """Rank messages by importance.
        
        Args:
            messages: List of message dicts
            
        Returns:
            List of ScoredMessage with importance scores
        """
        return self.importance_ranker.rank(messages)
    
    def compress(
        self,
        messages: list[dict],
        strategy: Optional[str] = None,
        token_budget: Optional[int] = None,
    ) -> CompressionResult:
        """Compress messages to fit within token budget.
        
        Args:
            messages: List of message dicts to compress
            strategy: Override strategy name
            token_budget: Override token budget
            
        Returns:
            CompressionResult with compressed messages
        """
        budget = token_budget or self.config.effective_budget
        strategy_name = strategy or self.config.get_strategy_name()
        
        # Get the strategy
        if strategy_name not in self._strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(self._strategies.keys())}"
            )
        
        compressor = self._strategies[strategy_name]
        
        # Compress
        result = compressor.compress(
            messages,
            budget,
            self.token_counter,
        )
        
        # Preserve critical information
        result.messages = self._preserve_critical_messages(
            messages,
            result.messages,
        )
        
        # Update token count after preservation
        result.compressed_tokens = self.token_counter.count_messages(result.messages)
        result.compression_ratio = 1.0 - (
            result.compressed_tokens / result.original_tokens
        ) if result.original_tokens > 0 else 0.0
        
        return result
    
    def compress_auto(
        self,
        messages: list[dict],
        token_budget: Optional[int] = None,
    ) -> CompressionResult:
        """Automatically select best compression strategy.
        
        Tries strategies in order of increasing aggressiveness until
        the token budget is met.
        
        Args:
            messages: List of message dicts
            token_budget: Override token budget
            
        Returns:
            CompressionResult with compressed messages
        """
        budget = token_budget or self.config.effective_budget
        original_tokens = self.token_counter.count_messages(messages)
        
        # If already within budget, no compression needed
        if original_tokens <= budget:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                messages=messages.copy(),
                compression_ratio=0.0,
                strategy_used="none",
            )
        
        # Try strategies in order of aggressiveness
        strategy_order = ["truncate", "summarize", "extract_key_facts", "tiered"]
        
        result: Optional[CompressionResult] = None
        for strategy_name in strategy_order:
            result = self.compress(messages, strategy=strategy_name, token_budget=budget)
            
            if result.compressed_tokens <= budget:
                return result
        
        # If tiered didn't work, return best effort
        assert result is not None  # At least one iteration happened
        return result
    
    def add_strategy(
        self,
        name: str,
        strategy: CompressionStrategy,
    ) -> None:
        """Add a custom compression strategy.
        
        Args:
            name: Strategy name
            strategy: CompressionStrategy instance
        """
        self._strategies[name] = strategy
    
    def get_compression_stats(
        self,
        messages: list[dict],
    ) -> dict[str, Any]:
        """Get statistics about potential compression.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Dictionary with compression statistics
        """
        current_tokens = self.count_tokens(messages)
        budget = self.config.effective_budget
        
        # Rank messages
        scored = self.rank_messages(messages)
        
        # Calculate stats
        system_tokens = sum(
            self.token_counter.count(m.get("content", ""))
            for m in messages
            if m.get("role") == "system"
        )
        
        # Find compression candidates
        candidates = self.importance_ranker.get_compression_candidates(
            messages,
            target_reduction=0.5,
            preserve_system=self.config.preserve_system,
            preserve_recent=self.config.preserve_recent,
        )
        
        return {
            "current_tokens": current_tokens,
            "token_budget": budget,
            "over_budget": current_tokens > budget,
            "tokens_over": max(0, current_tokens - budget),
            "compression_needed": current_tokens / budget if budget > 0 else 0,
            "message_count": len(messages),
            "system_tokens": system_tokens,
            "avg_message_importance": sum(s.score for s in scored) / len(scored) if scored else 0,
            "compression_candidates": len(candidates),
            "candidate_indices": candidates,
        }


# Convenience functions


def compress_context(
    messages: list[dict],
    max_tokens: int = 4000,
    model: str = "gpt-4",
    strategy: str = "tiered",
) -> CompressionResult:
    """Compress conversation context.
    
    Convenience function for one-off compression.
    
    Args:
        messages: List of message dicts
        max_tokens: Maximum tokens
        model: Model for token counting
        strategy: Compression strategy
        
    Returns:
        CompressionResult with compressed messages
    """
    compressor = ContextCompressor(max_tokens=max_tokens, model=model)
    return compressor.compress(messages, strategy=strategy)


def needs_compression(
    messages: list[dict],
    max_tokens: int = 4000,
    model: str = "gpt-4",
) -> bool:
    """Check if messages need compression.
    
    Args:
        messages: List of message dicts
        max_tokens: Maximum tokens
        model: Model for token counting
        
    Returns:
        True if compression needed
    """
    compressor = ContextCompressor(max_tokens=max_tokens, model=model)
    return compressor.needs_compression(messages)
