"""Compression strategies for context compression.

Provides multiple strategies for compressing messages:
- TruncateStrategy: Simple truncation to token limit
- SummarizeStrategy: LLM-based summarization
- ExtractKeyFactsStrategy: Extract key facts only
- TieredCompressionStrategy: Combine strategies based on position
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .token_counter import TokenCounter
from .importance import ImportanceRanker, ScoredMessage


class LLMProvider(Protocol):
    """Protocol for LLM providers used in summarization."""
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for prompt."""
        ...


@dataclass
class CompressionResult:
    """Result of compressing messages."""
    
    original_tokens: int
    compressed_tokens: int
    messages: list[dict]
    compression_ratio: float
    strategy_used: str
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens_saved(self) -> int:
        """Tokens saved by compression."""
        return self.original_tokens - self.compressed_tokens


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...
    
    @abstractmethod
    def compress(
        self,
        messages: list[dict],
        token_budget: int,
        counter: TokenCounter,
        **kwargs,
    ) -> CompressionResult:
        """Compress messages to fit within token budget.
        
        Args:
            messages: List of message dicts
            token_budget: Maximum tokens allowed
            counter: Token counter instance
            **kwargs: Strategy-specific arguments
            
        Returns:
            CompressionResult with compressed messages
        """
        ...


class TruncateStrategy(CompressionStrategy):
    """Simple truncation strategy.
    
    Removes oldest messages first until token budget is met.
    Optionally truncates long individual messages.
    """
    
    def __init__(
        self,
        preserve_system: bool = True,
        preserve_recent: int = 2,
        truncate_long_messages: bool = True,
        max_message_tokens: int = 1000,
    ):
        """Initialize truncate strategy.
        
        Args:
            preserve_system: Keep system messages
            preserve_recent: Number of recent messages to preserve
            truncate_long_messages: Truncate individual long messages
            max_message_tokens: Maximum tokens per message
        """
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent
        self.truncate_long_messages = truncate_long_messages
        self.max_message_tokens = max_message_tokens
    
    @property
    def name(self) -> str:
        return "truncate"
    
    def compress(
        self,
        messages: list[dict],
        token_budget: int,
        counter: TokenCounter,
        **kwargs,
    ) -> CompressionResult:
        """Compress by removing/truncating messages."""
        original_tokens = counter.count_messages(messages)
        
        if original_tokens <= token_budget:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                messages=messages.copy(),
                compression_ratio=0.0,
                strategy_used=self.name,
            )
        
        # First pass: truncate long individual messages
        working_messages = []
        for msg in messages:
            new_msg = msg.copy()
            content = msg.get("content", "")
            
            if self.truncate_long_messages and content:
                msg_tokens = counter.count(content)
                if msg_tokens > self.max_message_tokens:
                    new_msg["content"] = counter.truncate_to_tokens(
                        content,
                        self.max_message_tokens,
                        truncation_marker="... [truncated]"
                    )
            
            working_messages.append(new_msg)
        
        # Check if truncation was enough
        current_tokens = counter.count_messages(working_messages)
        if current_tokens <= token_budget:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=current_tokens,
                messages=working_messages,
                compression_ratio=1.0 - (current_tokens / original_tokens),
                strategy_used=self.name,
                details={"truncated_messages": True, "removed_messages": 0},
            )
        
        # Second pass: remove old messages
        n_messages = len(working_messages)
        preserved_indices = set()
        
        # Mark messages to preserve
        for i, msg in enumerate(working_messages):
            if self.preserve_system and msg.get("role") == "system":
                preserved_indices.add(i)
            if i >= n_messages - self.preserve_recent:
                preserved_indices.add(i)
        
        # Remove from oldest non-preserved first
        result_messages = []
        removed = 0
        
        for i, msg in enumerate(working_messages):
            if i in preserved_indices:
                result_messages.append(msg)
            else:
                # Check if we can include this message
                test_messages = result_messages + [msg] + [
                    working_messages[j] 
                    for j in range(i + 1, n_messages) 
                    if j in preserved_indices
                ]
                if counter.count_messages(test_messages) <= token_budget:
                    result_messages.append(msg)
                else:
                    removed += 1
        
        # Add preserved messages at the end
        for i in range(n_messages - self.preserve_recent, n_messages):
            if i not in [m.get("_idx") for m in result_messages] and i < n_messages:
                pass  # Already handled above
        
        compressed_tokens = counter.count_messages(result_messages)
        
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            messages=result_messages,
            compression_ratio=1.0 - (compressed_tokens / original_tokens),
            strategy_used=self.name,
            details={
                "truncated_messages": self.truncate_long_messages,
                "removed_messages": removed,
            },
        )


class SummarizeStrategy(CompressionStrategy):
    """LLM-based summarization strategy.
    
    Uses an LLM to summarize groups of messages into concise summaries.
    """
    
    DEFAULT_PROMPT = """Summarize the following conversation excerpt concisely.
Preserve all critical information, decisions, and key facts.
Keep names, dates, numbers, and technical details.
Format as a brief narrative or bullet points.

CONVERSATION:
{conversation}

SUMMARY:"""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt_template: Optional[str] = None,
        chunk_size: int = 5,
        preserve_system: bool = True,
        preserve_recent: int = 2,
        summary_max_tokens: int = 200,
    ):
        """Initialize summarize strategy.
        
        Args:
            llm_provider: LLM provider for summarization
            prompt_template: Custom prompt template with {conversation}
            chunk_size: Number of messages to summarize together
            preserve_system: Keep system messages intact
            preserve_recent: Recent messages to keep intact
            summary_max_tokens: Max tokens for summary
        """
        self.llm_provider = llm_provider
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.chunk_size = chunk_size
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent
        self.summary_max_tokens = summary_max_tokens
    
    @property
    def name(self) -> str:
        return "summarize"
    
    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for summarization prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def _create_summary_message(self, summary: str, original_count: int) -> dict:
        """Create a summary message."""
        return {
            "role": "system",
            "content": f"[CONVERSATION SUMMARY - {original_count} messages]\n{summary}",
        }
    
    def _rule_based_summarize(self, messages: list[dict]) -> str:
        """Simple rule-based summarization when no LLM available."""
        key_sentences = []
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            # Extract first sentence or key information
            sentences = re.split(r'[.!?]\s+', content)
            if sentences:
                first = sentences[0].strip()
                if len(first) > 20:
                    key_sentences.append(f"{role}: {first[:100]}...")
        
        if not key_sentences:
            return f"[{len(messages)} messages exchanged]"
        
        return "\n".join(key_sentences[:3])
    
    def compress(
        self,
        messages: list[dict],
        token_budget: int,
        counter: TokenCounter,
        **kwargs,
    ) -> CompressionResult:
        """Compress by summarizing message groups."""
        original_tokens = counter.count_messages(messages)
        
        if original_tokens <= token_budget:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                messages=messages.copy(),
                compression_ratio=0.0,
                strategy_used=self.name,
            )
        
        n_messages = len(messages)
        result_messages = []
        summarized_chunks = 0
        
        # Separate system messages, old messages, and recent messages
        system_messages = []
        old_messages = []
        recent_messages = []
        
        for i, msg in enumerate(messages):
            if self.preserve_system and msg.get("role") == "system":
                system_messages.append(msg)
            elif i >= n_messages - self.preserve_recent:
                recent_messages.append(msg)
            else:
                old_messages.append(msg)
        
        # Keep system messages
        result_messages.extend(system_messages)
        
        # Summarize old messages in chunks
        if old_messages:
            chunks = [
                old_messages[i:i + self.chunk_size]
                for i in range(0, len(old_messages), self.chunk_size)
            ]
            
            for chunk in chunks:
                if self.llm_provider:
                    prompt = self.prompt_template.format(
                        conversation=self._format_messages(chunk)
                    )
                    summary = self.llm_provider.complete(
                        prompt, 
                        max_tokens=self.summary_max_tokens
                    )
                else:
                    summary = self._rule_based_summarize(chunk)
                
                result_messages.append(
                    self._create_summary_message(summary, len(chunk))
                )
                summarized_chunks += 1
        
        # Add recent messages
        result_messages.extend(recent_messages)
        
        compressed_tokens = counter.count_messages(result_messages)
        
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            messages=result_messages,
            compression_ratio=1.0 - (compressed_tokens / original_tokens),
            strategy_used=self.name,
            details={
                "summarized_chunks": summarized_chunks,
                "preserved_recent": len(recent_messages),
                "llm_used": self.llm_provider is not None,
            },
        )


class ExtractKeyFactsStrategy(CompressionStrategy):
    """Extract key facts from messages.
    
    Identifies and preserves only the most important facts from each message.
    """
    
    # Patterns for extracting key facts
    KEY_FACT_PATTERNS = [
        (r"\b(?:is|are|was|were)\s+(\d+)", "number"),
        (r"\b(?:named?|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "name"),
        (r"\b(?:on|at|by)\s+(\d{4}[-/]\d{2}[-/]\d{2})", "date"),
        (r"\b(?:decided|concluded|determined)\s+(?:that\s+)?(.+?)[.!?]", "decision"),
        (r"\[(?:CRITICAL|IMPORTANT|KEY)\](.+?)(?:\[|$|\n)", "critical"),
        (r"(?:result|answer|output)(?:\s+is)?:\s*(.+?)(?:[.!?\n]|$)", "result"),
        (r"```[\w]*\n?([\s\S]*?)```", "code"),
    ]
    
    DEFAULT_PROMPT = """Extract the key facts from this text.
Return only the most important information as bullet points.
Preserve: names, dates, numbers, decisions, code, and critical information.

TEXT:
{text}

KEY FACTS:"""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt_template: Optional[str] = None,
        preserve_system: bool = True,
        preserve_recent: int = 2,
        max_facts_per_message: int = 3,
    ):
        """Initialize extract key facts strategy.
        
        Args:
            llm_provider: Optional LLM for extraction
            prompt_template: Custom prompt template
            preserve_system: Keep system messages
            preserve_recent: Recent messages to preserve
            max_facts_per_message: Max facts to extract per message
        """
        self.llm_provider = llm_provider
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent
        self.max_facts_per_message = max_facts_per_message
        
        self._patterns = [
            (re.compile(p, re.IGNORECASE), t) 
            for p, t in self.KEY_FACT_PATTERNS
        ]
    
    @property
    def name(self) -> str:
        return "extract_key_facts"
    
    def _extract_facts_rule_based(self, content: str) -> list[str]:
        """Extract key facts using regex patterns."""
        facts = []
        
        for pattern, fact_type in self._patterns:
            matches = pattern.findall(content)
            for match in matches[:self.max_facts_per_message]:
                if isinstance(match, tuple):
                    match = match[0]
                if match and len(match.strip()) > 5:
                    facts.append(f"[{fact_type}] {match.strip()[:100]}")
        
        return facts[:self.max_facts_per_message]
    
    def _create_facts_message(self, facts: list[str], role: str) -> dict:
        """Create a message containing extracted facts."""
        return {
            "role": role,
            "content": "KEY FACTS:\n" + "\n".join(f"• {f}" for f in facts),
        }
    
    def compress(
        self,
        messages: list[dict],
        token_budget: int,
        counter: TokenCounter,
        **kwargs,
    ) -> CompressionResult:
        """Compress by extracting key facts."""
        original_tokens = counter.count_messages(messages)
        
        if original_tokens <= token_budget:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                messages=messages.copy(),
                compression_ratio=0.0,
                strategy_used=self.name,
            )
        
        n_messages = len(messages)
        result_messages = []
        facts_extracted = 0
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Preserve system messages
            if self.preserve_system and role == "system":
                result_messages.append(msg.copy())
                continue
            
            # Preserve recent messages
            if i >= n_messages - self.preserve_recent:
                result_messages.append(msg.copy())
                continue
            
            # Extract facts from older messages
            if self.llm_provider:
                prompt = self.prompt_template.format(text=content)
                facts_text = self.llm_provider.complete(prompt, max_tokens=150)
                facts = [f.strip() for f in facts_text.split("\n") if f.strip()]
            else:
                facts = self._extract_facts_rule_based(content)
            
            if facts:
                result_messages.append(self._create_facts_message(facts, role))
                facts_extracted += len(facts)
            # Skip messages with no extractable facts
        
        compressed_tokens = counter.count_messages(result_messages)
        
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            messages=result_messages,
            compression_ratio=1.0 - (compressed_tokens / original_tokens),
            strategy_used=self.name,
            details={
                "facts_extracted": facts_extracted,
                "messages_processed": n_messages - self.preserve_recent,
                "llm_used": self.llm_provider is not None,
            },
        )


class TieredCompressionStrategy(CompressionStrategy):
    """Tiered compression with different strategies for different message ages.
    
    Applies increasingly aggressive compression to older messages:
    - Recent messages: No compression (full fidelity)
    - Medium-age messages: Light compression (summarize)
    - Old messages: Heavy compression (extract key facts only)
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        recent_count: int = 4,
        medium_count: int = 8,
        preserve_system: bool = True,
        importance_ranker: Optional[ImportanceRanker] = None,
    ):
        """Initialize tiered compression.
        
        Args:
            llm_provider: Optional LLM for summarization
            recent_count: Messages to keep at full fidelity
            medium_count: Messages to summarize
            preserve_system: Keep all system messages
            importance_ranker: Custom importance ranker
        """
        self.llm_provider = llm_provider
        self.recent_count = recent_count
        self.medium_count = medium_count
        self.preserve_system = preserve_system
        self.importance_ranker = importance_ranker or ImportanceRanker()
        
        # Initialize sub-strategies
        self.summarize_strategy = SummarizeStrategy(
            llm_provider=llm_provider,
            preserve_system=False,  # We handle this ourselves
            preserve_recent=0,
        )
        self.extract_strategy = ExtractKeyFactsStrategy(
            llm_provider=llm_provider,
            preserve_system=False,
            preserve_recent=0,
        )
    
    @property
    def name(self) -> str:
        return "tiered"
    
    def compress(
        self,
        messages: list[dict],
        token_budget: int,
        counter: TokenCounter,
        **kwargs,
    ) -> CompressionResult:
        """Apply tiered compression based on message age and importance."""
        original_tokens = counter.count_messages(messages)
        
        if original_tokens <= token_budget:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                messages=messages.copy(),
                compression_ratio=0.0,
                strategy_used=self.name,
            )
        
        n_messages = len(messages)
        
        # Categorize messages
        system_messages = []
        old_messages = []  # Will get key facts extracted
        medium_messages = []  # Will be summarized
        recent_messages = []  # Kept as-is
        
        old_threshold = n_messages - self.recent_count - self.medium_count
        medium_threshold = n_messages - self.recent_count
        
        for i, msg in enumerate(messages):
            if self.preserve_system and msg.get("role") == "system":
                system_messages.append(msg)
            elif i < old_threshold:
                old_messages.append(msg)
            elif i < medium_threshold:
                medium_messages.append(msg)
            else:
                recent_messages.append(msg)
        
        result_messages = []
        
        # Add system messages first
        result_messages.extend(system_messages)
        
        # Process old messages - extract key facts
        if old_messages:
            old_result = self.extract_strategy.compress(
                old_messages,
                token_budget // 4,  # Allocate 1/4 budget to old
                counter,
            )
            if old_result.messages:
                # Consolidate into single summary
                all_facts = []
                for msg in old_result.messages:
                    content = msg.get("content", "")
                    if content:
                        all_facts.append(content)
                if all_facts:
                    result_messages.append({
                        "role": "system",
                        "content": f"[EARLY CONTEXT - {len(old_messages)} messages]\n" + 
                                  "\n".join(all_facts),
                    })
        
        # Process medium-age messages - summarize
        if medium_messages:
            medium_result = self.summarize_strategy.compress(
                medium_messages,
                token_budget // 3,  # Allocate 1/3 budget to medium
                counter,
            )
            result_messages.extend(medium_result.messages)
        
        # Add recent messages as-is
        result_messages.extend(recent_messages)
        
        # Final check - if still over budget, truncate
        compressed_tokens = counter.count_messages(result_messages)
        if compressed_tokens > token_budget:
            truncate = TruncateStrategy(
                preserve_system=True,
                preserve_recent=self.recent_count,
            )
            final_result = truncate.compress(
                result_messages,
                token_budget,
                counter,
            )
            result_messages = final_result.messages
            compressed_tokens = final_result.compressed_tokens
        
        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            messages=result_messages,
            compression_ratio=1.0 - (compressed_tokens / original_tokens),
            strategy_used=self.name,
            details={
                "system_messages": len(system_messages),
                "old_messages_processed": len(old_messages),
                "medium_messages_processed": len(medium_messages),
                "recent_messages_preserved": len(recent_messages),
            },
        )
