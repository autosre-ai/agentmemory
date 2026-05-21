"""Importance ranking for context messages.

Provides algorithms to rank the importance of messages in a conversation
to guide compression decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Protocol


class MessageRole(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


@dataclass
class ImportanceFactors:
    """Factors that contribute to message importance.
    
    Each factor is a float from 0.0 to 1.0, where higher means more important.
    """
    # Position factors
    recency: float = 0.0  # How recent the message is
    is_first_message: bool = False  # First message often sets context
    is_last_exchange: bool = False  # Last user-assistant pair
    
    # Content factors
    has_critical_marker: bool = False  # Contains [CRITICAL] or similar
    has_code: bool = False  # Contains code blocks
    has_structured_data: bool = False  # Contains JSON, lists, etc.
    has_names_entities: bool = False  # Contains names, dates, numbers
    has_decisions: bool = False  # Contains decisions or conclusions
    has_questions: bool = False  # Contains unanswered questions
    
    # Role factors
    is_system: bool = False  # System messages are usually important
    is_tool_result: bool = False  # Tool results often contain key info
    
    # Reference factors
    is_referenced: bool = False  # Referenced by later messages
    reference_count: int = 0  # How many times referenced
    
    # Custom factors
    custom_score: float = 0.0  # User-defined importance boost
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recency": self.recency,
            "is_first_message": self.is_first_message,
            "is_last_exchange": self.is_last_exchange,
            "has_critical_marker": self.has_critical_marker,
            "has_code": self.has_code,
            "has_structured_data": self.has_structured_data,
            "has_names_entities": self.has_names_entities,
            "has_decisions": self.has_decisions,
            "has_questions": self.has_questions,
            "is_system": self.is_system,
            "is_tool_result": self.is_tool_result,
            "is_referenced": self.is_referenced,
            "reference_count": self.reference_count,
            "custom_score": self.custom_score,
        }


@dataclass
class ScoredMessage:
    """A message with its importance score and factors."""
    index: int
    role: str
    content: str
    score: float
    factors: ImportanceFactors
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "ScoredMessage") -> bool:
        return self.score < other.score


class ImportanceRanker:
    """Rank messages by importance for compression decisions.
    
    Uses a weighted scoring system that considers:
    - Position (recency, first/last messages)
    - Content (critical markers, code, structured data)
    - Role (system messages, tool results)
    - References (messages referenced by later messages)
    
    Example:
        >>> ranker = ImportanceRanker()
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hi there"},
        ...     {"role": "assistant", "content": "Hello!"},
        ... ]
        >>> scored = ranker.rank(messages)
        >>> [s.score for s in scored]
        [0.85, 0.45, 0.7]
    """
    
    # Critical markers that indicate important information
    CRITICAL_MARKERS = [
        r"\[CRITICAL\]",
        r"\[IMPORTANT\]",
        r"\[REMEMBER\]",
        r"\[KEY\]",
        r"\[NOTE\]",
        r"\[PRESERVE\]",
        r"(?:^|\s)IMPORTANT:",
        r"(?:^|\s)NOTE:",
        r"(?:^|\s)CRITICAL:",
        r"must remember",
        r"don't forget",
        r"key point",
        r"important:",
    ]
    
    # Patterns for detecting code
    CODE_PATTERNS = [
        r"```[\s\S]*?```",
        r"`[^`]+`",
        r"^\s{4,}\S",  # Indented code
    ]
    
    # Patterns for structured data
    STRUCTURED_PATTERNS = [
        r"\{[\s\S]*\}",  # JSON-like
        r"\[[\s\S]*\]",  # Arrays
        r"^\s*[-*]\s",  # Bullet lists
        r"^\s*\d+\.\s",  # Numbered lists
        r"\|.*\|",  # Tables
    ]
    
    # Patterns for entities
    ENTITY_PATTERNS = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Names
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",  # Dates
        r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b",  # Numbers
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
        r"https?://\S+",  # URLs
    ]
    
    # Decision indicators
    DECISION_PATTERNS = [
        r"\b(?:decided|decision|conclude|concluded|therefore|thus)\b",
        r"\b(?:will|should|must|need to)\b",
        r"\b(?:answer is|solution is|result is)\b",
        r"\b(?:in summary|to summarize|in conclusion)\b",
    ]
    
    # Default weights for scoring
    DEFAULT_WEIGHTS = {
        "recency": 0.20,
        "is_first_message": 0.15,
        "is_last_exchange": 0.10,
        "has_critical_marker": 0.20,
        "has_code": 0.08,
        "has_structured_data": 0.05,
        "has_names_entities": 0.05,
        "has_decisions": 0.05,
        "has_questions": 0.03,
        "is_system": 0.15,
        "is_tool_result": 0.08,
        "is_referenced": 0.05,
        "reference_count": 0.02,
        "custom_score": 1.00,  # Direct add, not weighted
    }
    
    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        critical_patterns: Optional[list[str]] = None,
        custom_scorer: Optional[Callable[[dict, int], float]] = None,
    ):
        """Initialize importance ranker.
        
        Args:
            weights: Custom weights for scoring factors
            critical_patterns: Additional patterns to mark as critical
            custom_scorer: Custom scoring function (message, index) -> score
        """
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.critical_patterns = self.CRITICAL_MARKERS + (critical_patterns or [])
        self.custom_scorer = custom_scorer
        
        # Compile regex patterns
        self._critical_re = re.compile(
            "|".join(self.critical_patterns), 
            re.IGNORECASE | re.MULTILINE
        )
        self._code_re = re.compile(
            "|".join(self.CODE_PATTERNS),
            re.MULTILINE
        )
        self._structured_re = re.compile(
            "|".join(self.STRUCTURED_PATTERNS),
            re.MULTILINE
        )
        self._entity_re = re.compile(
            "|".join(self.ENTITY_PATTERNS),
            re.IGNORECASE
        )
        self._decision_re = re.compile(
            "|".join(self.DECISION_PATTERNS),
            re.IGNORECASE
        )
        self._question_re = re.compile(r"\?(?:\s|$)")
    
    def analyze_factors(
        self,
        message: dict,
        index: int,
        total_messages: int,
        context: Optional[dict] = None,
    ) -> ImportanceFactors:
        """Analyze importance factors for a single message.
        
        Args:
            message: Message dict with 'role' and 'content'
            index: Position in message list
            total_messages: Total number of messages
            context: Additional context (e.g., reference info)
            
        Returns:
            ImportanceFactors with all analyzed factors
        """
        content = message.get("content", "")
        role = message.get("role", "user")
        context = context or {}
        
        factors = ImportanceFactors()
        
        # Position factors
        factors.recency = index / max(total_messages - 1, 1) if total_messages > 1 else 1.0
        factors.is_first_message = index == 0
        factors.is_last_exchange = index >= total_messages - 2
        
        # Content analysis
        if content:
            factors.has_critical_marker = bool(self._critical_re.search(content))
            factors.has_code = bool(self._code_re.search(content))
            factors.has_structured_data = bool(self._structured_re.search(content))
            factors.has_names_entities = bool(self._entity_re.search(content))
            factors.has_decisions = bool(self._decision_re.search(content))
            factors.has_questions = bool(self._question_re.search(content))
        
        # Role factors
        factors.is_system = role == "system"
        factors.is_tool_result = role in ("tool", "function")
        
        # Reference factors from context
        factors.is_referenced = context.get("is_referenced", False)
        factors.reference_count = context.get("reference_count", 0)
        
        # Custom scoring
        if self.custom_scorer:
            factors.custom_score = self.custom_scorer(message, index)
        
        return factors
    
    def compute_score(self, factors: ImportanceFactors) -> float:
        """Compute importance score from factors.
        
        Args:
            factors: ImportanceFactors to score
            
        Returns:
            Float score from 0.0 to 1.0+
        """
        score = 0.0
        
        # Position factors
        score += factors.recency * self.weights["recency"]
        if factors.is_first_message:
            score += self.weights["is_first_message"]
        if factors.is_last_exchange:
            score += self.weights["is_last_exchange"]
        
        # Content factors
        if factors.has_critical_marker:
            score += self.weights["has_critical_marker"]
        if factors.has_code:
            score += self.weights["has_code"]
        if factors.has_structured_data:
            score += self.weights["has_structured_data"]
        if factors.has_names_entities:
            score += self.weights["has_names_entities"]
        if factors.has_decisions:
            score += self.weights["has_decisions"]
        if factors.has_questions:
            score += self.weights["has_questions"]
        
        # Role factors
        if factors.is_system:
            score += self.weights["is_system"]
        if factors.is_tool_result:
            score += self.weights["is_tool_result"]
        
        # Reference factors
        if factors.is_referenced:
            score += self.weights["is_referenced"]
        score += min(factors.reference_count * self.weights["reference_count"], 0.1)
        
        # Custom score (added directly)
        score += factors.custom_score
        
        return score
    
    def rank(
        self,
        messages: list[dict],
        context: Optional[dict[int, dict]] = None,
    ) -> list[ScoredMessage]:
        """Rank all messages by importance.
        
        Args:
            messages: List of message dicts
            context: Optional per-message context by index
            
        Returns:
            List of ScoredMessage in original order
        """
        context = context or {}
        total = len(messages)
        scored = []
        
        for i, msg in enumerate(messages):
            factors = self.analyze_factors(
                msg, i, total, 
                context.get(i, {})
            )
            score = self.compute_score(factors)
            
            scored.append(ScoredMessage(
                index=i,
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                score=score,
                factors=factors,
                metadata=msg.get("metadata", {}),
            ))
        
        return scored
    
    def rank_sorted(
        self,
        messages: list[dict],
        context: Optional[dict[int, dict]] = None,
        ascending: bool = False,
    ) -> list[ScoredMessage]:
        """Rank messages and return sorted by score.
        
        Args:
            messages: List of message dicts
            context: Optional per-message context by index
            ascending: Sort low to high (default: high to low)
            
        Returns:
            List of ScoredMessage sorted by score
        """
        scored = self.rank(messages, context)
        return sorted(scored, reverse=not ascending)
    
    def get_compression_candidates(
        self,
        messages: list[dict],
        target_reduction: float = 0.5,
        preserve_system: bool = True,
        preserve_recent: int = 2,
    ) -> list[int]:
        """Get indices of messages that are candidates for compression.
        
        Args:
            messages: List of message dicts
            target_reduction: Target fraction of messages to mark (0.0-1.0)
            preserve_system: Never mark system messages
            preserve_recent: Number of recent messages to preserve
            
        Returns:
            List of message indices suitable for compression
        """
        scored = self.rank(messages)
        n_total = len(scored)
        n_target = int(n_total * target_reduction)
        
        # Sort by score (ascending = lowest importance first)
        sorted_scored = sorted(scored)
        
        candidates = []
        for sm in sorted_scored:
            if len(candidates) >= n_target:
                break
            
            # Skip system messages if preserving
            if preserve_system and sm.role == "system":
                continue
            
            # Skip recent messages
            if sm.index >= n_total - preserve_recent:
                continue
            
            candidates.append(sm.index)
        
        return sorted(candidates)  # Return in original order
