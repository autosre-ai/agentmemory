"""AI-Powered Memory Summarization - Consolidate and compress memories intelligently.

This module provides AI-based summarization capabilities for memory consolidation,
enabling significant memory footprint reduction while preserving key information.

Summarization strategies:
- Extractive: Select key sentences/facts from memories
- Abstractive: Generate new summaries using LLM
- Hierarchical: Create multi-level summary hierarchies
- Incremental: Update summaries as new memories arrive
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence, TypeVar


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate completion for prompt."""
        ...
    
    def complete_with_system(
        self, 
        system: str, 
        prompt: str, 
        max_tokens: int = 500
    ) -> str:
        """Generate completion with system prompt."""
        ...


class SummarizationStrategy(str, Enum):
    """Available summarization strategies."""
    EXTRACTIVE = "extractive"     # Select key sentences
    ABSTRACTIVE = "abstractive"   # Generate new summary
    HIERARCHICAL = "hierarchical" # Multi-level summaries
    INCREMENTAL = "incremental"   # Streaming summarization


class SummaryLevel(str, Enum):
    """Levels of summary detail."""
    BRIEF = "brief"       # 1-2 sentences
    STANDARD = "standard" # Paragraph
    DETAILED = "detailed" # Multiple paragraphs
    FULL = "full"         # Comprehensive


@dataclass
class MemoryEntry:
    """A memory entry for summarization."""
    memory_id: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5
    category: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Summary:
    """A generated summary."""
    summary_id: str
    content: str
    source_memory_ids: list[str]
    level: SummaryLevel
    strategy: SummarizationStrategy
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Summary metadata
    word_count: int = 0
    compression_ratio: float = 0.0
    key_topics: list[str] = field(default_factory=list)
    preserved_facts: list[str] = field(default_factory=list)
    
    # Quality metrics
    coverage_score: float = 0.0  # How much original info is captured
    coherence_score: float = 0.0  # How well the summary flows
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary_id": self.summary_id,
            "content": self.content,
            "source_memory_ids": self.source_memory_ids,
            "level": self.level.value,
            "strategy": self.strategy.value,
            "created_at": self.created_at.isoformat(),
            "word_count": self.word_count,
            "compression_ratio": self.compression_ratio,
            "key_topics": self.key_topics,
            "preserved_facts": self.preserved_facts,
            "coverage_score": self.coverage_score,
            "coherence_score": self.coherence_score,
        }


@dataclass
class HierarchicalSummary:
    """A hierarchical summary with multiple levels."""
    root: Summary
    children: list["HierarchicalSummary"]
    depth: int = 0
    
    def flatten(self) -> list[Summary]:
        """Flatten to list of summaries."""
        result = [self.root]
        for child in self.children:
            result.extend(child.flatten())
        return result
    
    def get_level(self, target_depth: int) -> list[Summary]:
        """Get summaries at a specific depth."""
        if self.depth == target_depth:
            return [self.root]
        
        result = []
        for child in self.children:
            result.extend(child.get_level(target_depth))
        return result


@dataclass
class SummarizationResult:
    """Result of a summarization operation."""
    summaries: list[Summary]
    original_memory_count: int
    original_word_count: int
    summarized_word_count: int
    compression_ratio: float
    processing_time_ms: float
    strategy_used: SummarizationStrategy
    stats: dict[str, Any] = field(default_factory=dict)
    
    @property
    def words_saved(self) -> int:
        """Words saved through summarization."""
        return self.original_word_count - self.summarized_word_count
    
    @property
    def reduction_percent(self) -> float:
        """Percentage reduction in word count."""
        if self.original_word_count == 0:
            return 0.0
        return (self.words_saved / self.original_word_count) * 100


class Summarizer(ABC):
    """Abstract base class for summarizers."""
    
    @property
    @abstractmethod
    def strategy(self) -> SummarizationStrategy:
        """The summarization strategy used."""
        ...
    
    @abstractmethod
    def summarize(
        self,
        memories: list[MemoryEntry],
        level: SummaryLevel,
        **kwargs,
    ) -> Summary:
        """Summarize memories.
        
        Args:
            memories: Memories to summarize
            level: Detail level for summary
            **kwargs: Strategy-specific arguments
            
        Returns:
            Summary object
        """
        ...


class ExtractiveSummarizer(Summarizer):
    """Extractive summarization - select key sentences."""
    
    # Patterns for important information
    IMPORTANCE_PATTERNS = [
        (r'\b(?:important|critical|key|essential|must|should)\b', 2.0),
        (r'\b(?:decided|concluded|determined|agreed|resolved)\b', 1.8),
        (r'\b(?:because|therefore|thus|hence|consequently)\b', 1.5),
        (r'\b(?:first|finally|in summary|to summarize)\b', 1.5),
        (r'\b(?:\d{4}[-/]\d{2}[-/]\d{2})\b', 1.3),  # Dates
        (r'\b(?:\$\d+|\d+%)\b', 1.3),  # Numbers
        (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', 1.2),  # Named entities
    ]
    
    def __init__(
        self,
        max_sentences: int = 10,
        min_sentence_length: int = 20,
        preserve_dates: bool = True,
        preserve_numbers: bool = True,
    ):
        """Initialize extractive summarizer.
        
        Args:
            max_sentences: Maximum sentences to extract
            min_sentence_length: Minimum sentence length
            preserve_dates: Always include sentences with dates
            preserve_numbers: Always include sentences with numbers
        """
        self.max_sentences = max_sentences
        self.min_sentence_length = min_sentence_length
        self.preserve_dates = preserve_dates
        self.preserve_numbers = preserve_numbers
        
        self._patterns = [
            (re.compile(p, re.IGNORECASE), w) 
            for p, w in self.IMPORTANCE_PATTERNS
        ]
    
    @property
    def strategy(self) -> SummarizationStrategy:
        return SummarizationStrategy.EXTRACTIVE
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_sentence_length]
    
    def _score_sentence(self, sentence: str, position: float) -> float:
        """Score a sentence for importance."""
        score = 1.0
        
        # Position bonus (first and last sentences)
        if position < 0.2:
            score += 0.5
        elif position > 0.8:
            score += 0.3
        
        # Pattern matching
        for pattern, weight in self._patterns:
            if pattern.search(sentence):
                score *= weight
        
        # Length penalty for very long sentences
        words = len(sentence.split())
        if words > 50:
            score *= 0.8
        
        return score
    
    def _get_target_sentences(self, level: SummaryLevel) -> int:
        """Get target sentence count for level."""
        return {
            SummaryLevel.BRIEF: min(2, self.max_sentences),
            SummaryLevel.STANDARD: min(5, self.max_sentences),
            SummaryLevel.DETAILED: min(10, self.max_sentences),
            SummaryLevel.FULL: self.max_sentences,
        }.get(level, self.max_sentences)
    
    def summarize(
        self,
        memories: list[MemoryEntry],
        level: SummaryLevel = SummaryLevel.STANDARD,
        **kwargs,
    ) -> Summary:
        """Extract key sentences from memories."""
        import uuid
        
        # Combine all content
        all_sentences: list[tuple[str, float, str]] = []  # (sentence, score, source_id)
        
        for memory in memories:
            sentences = self._split_sentences(memory.content)
            n_sentences = len(sentences)
            
            for i, sentence in enumerate(sentences):
                position = i / n_sentences if n_sentences > 0 else 0.5
                score = self._score_sentence(sentence, position)
                
                # Importance boost from memory
                score *= (0.5 + memory.importance)
                
                all_sentences.append((sentence, score, memory.memory_id))
        
        # Sort by score and select top sentences
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        
        target_count = self._get_target_sentences(level)
        selected = all_sentences[:target_count]
        
        # Reconstruct in original order (roughly)
        # Sort by position within the combined text
        indices = {s[0]: i for i, s in enumerate(all_sentences)}
        selected.sort(key=lambda x: indices[x[0]])
        
        # Build summary
        summary_content = " ".join(s[0] for s in selected)
        source_ids = list(set(s[2] for s in selected))
        
        # Extract key facts (sentences with high scores)
        key_facts = [s[0] for s in all_sentences[:3]]
        
        # Original word count
        original_words = sum(len(m.content.split()) for m in memories)
        summary_words = len(summary_content.split())
        
        return Summary(
            summary_id=f"sum_{uuid.uuid4().hex[:12]}",
            content=summary_content,
            source_memory_ids=source_ids,
            level=level,
            strategy=self.strategy,
            word_count=summary_words,
            compression_ratio=summary_words / original_words if original_words > 0 else 1.0,
            key_topics=self._extract_topics(memories),
            preserved_facts=key_facts,
            coverage_score=len(source_ids) / len(memories) if memories else 0.0,
        )
    
    def _extract_topics(self, memories: list[MemoryEntry]) -> list[str]:
        """Extract key topics from memories."""
        topics = set()
        for mem in memories:
            if mem.category:
                topics.add(mem.category)
            topics.update(mem.tags)
        return list(topics)[:10]


class AbstractiveSummarizer(Summarizer):
    """Abstractive summarization using LLM."""
    
    SYSTEM_PROMPT = """You are an expert summarization assistant. Create concise, 
accurate summaries that preserve all critical information, decisions, and facts.
Maintain factual accuracy and never add information not present in the source."""
    
    SUMMARY_PROMPTS = {
        SummaryLevel.BRIEF: """Summarize the following memories in 1-2 sentences:

{memories}

Brief Summary:""",
        
        SummaryLevel.STANDARD: """Summarize the following memories in a single paragraph,
preserving key facts, decisions, and important details:

{memories}

Summary:""",
        
        SummaryLevel.DETAILED: """Create a detailed summary of the following memories.
Include all important facts, decisions, context, and relationships between topics.
Use multiple paragraphs if needed:

{memories}

Detailed Summary:""",
        
        SummaryLevel.FULL: """Create a comprehensive summary of the following memories.
Preserve all significant information, including:
- Key facts and data points
- Decisions and conclusions
- Important context
- Relationships between topics
- Any action items or next steps

{memories}

Comprehensive Summary:""",
    }
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        custom_prompts: Optional[dict[SummaryLevel, str]] = None,
        max_input_tokens: int = 3000,
    ):
        """Initialize abstractive summarizer.
        
        Args:
            llm_provider: LLM provider for generation
            custom_prompts: Custom prompt templates
            max_input_tokens: Maximum input tokens
        """
        self.llm_provider = llm_provider
        self.prompts = custom_prompts or self.SUMMARY_PROMPTS
        self.max_input_tokens = max_input_tokens
    
    @property
    def strategy(self) -> SummarizationStrategy:
        return SummarizationStrategy.ABSTRACTIVE
    
    def _format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format memories for prompt."""
        parts = []
        for i, mem in enumerate(memories, 1):
            timestamp = mem.created_at.strftime("%Y-%m-%d %H:%M")
            tags = f" [{', '.join(mem.tags)}]" if mem.tags else ""
            parts.append(f"[Memory {i} - {timestamp}]{tags}\n{mem.content}")
        return "\n\n".join(parts)
    
    def _get_max_tokens(self, level: SummaryLevel) -> int:
        """Get max output tokens for level."""
        return {
            SummaryLevel.BRIEF: 100,
            SummaryLevel.STANDARD: 250,
            SummaryLevel.DETAILED: 500,
            SummaryLevel.FULL: 1000,
        }.get(level, 250)
    
    def _rule_based_summary(
        self, 
        memories: list[MemoryEntry],
        level: SummaryLevel,
    ) -> str:
        """Fallback rule-based summary when no LLM available."""
        # Use extractive summarizer as fallback
        extractor = ExtractiveSummarizer()
        result = extractor.summarize(memories, level)
        return result.content
    
    def summarize(
        self,
        memories: list[MemoryEntry],
        level: SummaryLevel = SummaryLevel.STANDARD,
        **kwargs,
    ) -> Summary:
        """Generate summary using LLM."""
        import uuid
        
        memories_text = self._format_memories(memories)
        prompt = self.prompts[level].format(memories=memories_text)
        
        # Generate summary
        if self.llm_provider:
            try:
                summary_content = self.llm_provider.complete_with_system(
                    self.SYSTEM_PROMPT,
                    prompt,
                    max_tokens=self._get_max_tokens(level),
                )
            except Exception:
                # Try simple completion
                try:
                    summary_content = self.llm_provider.complete(
                        f"{self.SYSTEM_PROMPT}\n\n{prompt}",
                        max_tokens=self._get_max_tokens(level),
                    )
                except Exception:
                    summary_content = self._rule_based_summary(memories, level)
        else:
            summary_content = self._rule_based_summary(memories, level)
        
        # Clean up
        summary_content = summary_content.strip()
        
        # Calculate metrics
        original_words = sum(len(m.content.split()) for m in memories)
        summary_words = len(summary_content.split())
        
        return Summary(
            summary_id=f"sum_{uuid.uuid4().hex[:12]}",
            content=summary_content,
            source_memory_ids=[m.memory_id for m in memories],
            level=level,
            strategy=self.strategy,
            word_count=summary_words,
            compression_ratio=summary_words / original_words if original_words > 0 else 1.0,
            key_topics=list(set(
                tag for m in memories for tag in m.tags
            ))[:10],
            coverage_score=1.0,  # Abstractive covers all input
        )


class HierarchicalSummarizer(Summarizer):
    """Hierarchical summarization - multi-level summary trees."""
    
    def __init__(
        self,
        base_summarizer: Optional[Summarizer] = None,
        chunk_size: int = 5,
        max_depth: int = 3,
    ):
        """Initialize hierarchical summarizer.
        
        Args:
            base_summarizer: Summarizer for each level
            chunk_size: Memories per chunk
            max_depth: Maximum hierarchy depth
        """
        self.base_summarizer = base_summarizer or ExtractiveSummarizer()
        self.chunk_size = chunk_size
        self.max_depth = max_depth
    
    @property
    def strategy(self) -> SummarizationStrategy:
        return SummarizationStrategy.HIERARCHICAL
    
    def _chunk_memories(
        self, 
        memories: list[MemoryEntry]
    ) -> list[list[MemoryEntry]]:
        """Split memories into chunks."""
        return [
            memories[i:i + self.chunk_size]
            for i in range(0, len(memories), self.chunk_size)
        ]
    
    def summarize(
        self,
        memories: list[MemoryEntry],
        level: SummaryLevel = SummaryLevel.STANDARD,
        **kwargs,
    ) -> Summary:
        """Create hierarchical summary."""
        hierarchy = self.create_hierarchy(memories, level)
        return hierarchy.root
    
    def create_hierarchy(
        self,
        memories: list[MemoryEntry],
        level: SummaryLevel = SummaryLevel.STANDARD,
        current_depth: int = 0,
    ) -> HierarchicalSummary:
        """Create full hierarchical summary structure.
        
        Args:
            memories: Memories to summarize
            level: Detail level
            current_depth: Current depth in hierarchy
            
        Returns:
            HierarchicalSummary tree
        """
        # Base case: few memories, summarize directly
        if len(memories) <= self.chunk_size or current_depth >= self.max_depth:
            summary = self.base_summarizer.summarize(memories, level)
            return HierarchicalSummary(
                root=summary,
                children=[],
                depth=current_depth,
            )
        
        # Recursive case: chunk and summarize each chunk
        chunks = self._chunk_memories(memories)
        children = []
        child_summaries = []
        
        for chunk in chunks:
            child_hierarchy = self.create_hierarchy(
                chunk,
                level,
                current_depth + 1,
            )
            children.append(child_hierarchy)
            child_summaries.append(child_hierarchy.root)
        
        # Create parent summary from child summaries
        # Convert summaries to memory entries for summarization
        summary_memories = [
            MemoryEntry(
                memory_id=s.summary_id,
                content=s.content,
                created_at=s.created_at,
            )
            for s in child_summaries
        ]
        
        root_summary = self.base_summarizer.summarize(
            summary_memories,
            SummaryLevel.STANDARD,  # Parent is more compressed
        )
        
        # Update source IDs to include all original memories
        root_summary.source_memory_ids = [
            m.memory_id for m in memories
        ]
        
        return HierarchicalSummary(
            root=root_summary,
            children=children,
            depth=current_depth,
        )


class IncrementalSummarizer(Summarizer):
    """Incremental summarization - update summaries as memories arrive."""
    
    UPDATE_PROMPT = """You have an existing summary and new information. 
Update the summary to incorporate the new information while maintaining 
coherence and avoiding redundancy.

EXISTING SUMMARY:
{existing_summary}

NEW INFORMATION:
{new_content}

UPDATED SUMMARY:"""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        base_summarizer: Optional[Summarizer] = None,
        max_summary_length: int = 500,
    ):
        """Initialize incremental summarizer.
        
        Args:
            llm_provider: LLM provider for updates
            base_summarizer: Summarizer for initial summary
            max_summary_length: Maximum summary word count
        """
        self.llm_provider = llm_provider
        self.base_summarizer = base_summarizer or ExtractiveSummarizer()
        self.max_summary_length = max_summary_length
        
        # Cache for running summaries
        self._running_summaries: dict[str, Summary] = {}
    
    @property
    def strategy(self) -> SummarizationStrategy:
        return SummarizationStrategy.INCREMENTAL
    
    def summarize(
        self,
        memories: list[MemoryEntry],
        level: SummaryLevel = SummaryLevel.STANDARD,
        **kwargs,
    ) -> Summary:
        """Create initial summary."""
        return self.base_summarizer.summarize(memories, level)
    
    def update_summary(
        self,
        existing_summary: Summary,
        new_memories: list[MemoryEntry],
        level: SummaryLevel = SummaryLevel.STANDARD,
    ) -> Summary:
        """Update an existing summary with new memories.
        
        Args:
            existing_summary: Current summary
            new_memories: New memories to incorporate
            level: Summary detail level
            
        Returns:
            Updated summary
        """
        import uuid
        
        # Format new content
        new_content = "\n\n".join(m.content for m in new_memories)
        
        # Generate updated summary
        if self.llm_provider:
            prompt = self.UPDATE_PROMPT.format(
                existing_summary=existing_summary.content,
                new_content=new_content,
            )
            
            try:
                updated_content = self.llm_provider.complete(prompt, max_tokens=500)
            except Exception:
                # Fallback: concatenate and re-summarize
                combined_memories = [
                    MemoryEntry(
                        memory_id="existing",
                        content=existing_summary.content,
                    )
                ] + new_memories
                return self.base_summarizer.summarize(combined_memories, level)
        else:
            # Rule-based: combine and re-summarize
            combined_memories = [
                MemoryEntry(
                    memory_id="existing",
                    content=existing_summary.content,
                )
            ] + new_memories
            return self.base_summarizer.summarize(combined_memories, level)
        
        # Update source IDs
        all_source_ids = existing_summary.source_memory_ids + [
            m.memory_id for m in new_memories
        ]
        
        # Recalculate metrics
        summary_words = len(updated_content.split())
        
        return Summary(
            summary_id=f"sum_{uuid.uuid4().hex[:12]}",
            content=updated_content.strip(),
            source_memory_ids=all_source_ids,
            level=level,
            strategy=self.strategy,
            word_count=summary_words,
            compression_ratio=existing_summary.compression_ratio,  # Approximate
            key_topics=existing_summary.key_topics + list(set(
                tag for m in new_memories for tag in m.tags
            )),
        )
    
    def get_or_create_summary(
        self,
        summary_key: str,
        memories: list[MemoryEntry],
        level: SummaryLevel = SummaryLevel.STANDARD,
    ) -> Summary:
        """Get existing summary or create new one.
        
        Args:
            summary_key: Unique key for the summary
            memories: Memories for initial summary
            level: Summary detail level
            
        Returns:
            Summary (cached or new)
        """
        if summary_key in self._running_summaries:
            return self._running_summaries[summary_key]
        
        summary = self.summarize(memories, level)
        self._running_summaries[summary_key] = summary
        return summary
    
    def add_memories(
        self,
        summary_key: str,
        new_memories: list[MemoryEntry],
        level: SummaryLevel = SummaryLevel.STANDARD,
    ) -> Summary:
        """Add memories to an existing summary.
        
        Args:
            summary_key: Key of summary to update
            new_memories: New memories to add
            level: Summary detail level
            
        Returns:
            Updated summary
        """
        if summary_key not in self._running_summaries:
            # Create new summary
            summary = self.summarize(new_memories, level)
            self._running_summaries[summary_key] = summary
            return summary
        
        existing = self._running_summaries[summary_key]
        updated = self.update_summary(existing, new_memories, level)
        self._running_summaries[summary_key] = updated
        return updated
    
    def clear_cache(self) -> None:
        """Clear summary cache."""
        self._running_summaries.clear()


@dataclass
class SummarizationConfig:
    """Configuration for memory summarization."""
    
    # Strategy selection
    strategy: SummarizationStrategy = SummarizationStrategy.EXTRACTIVE
    default_level: SummaryLevel = SummaryLevel.STANDARD
    
    # Extractive settings
    max_sentences: int = 10
    min_sentence_length: int = 20
    
    # Abstractive settings
    max_input_tokens: int = 3000
    
    # Hierarchical settings
    chunk_size: int = 5
    max_depth: int = 3
    
    # Incremental settings
    max_summary_length: int = 500
    
    # Quality settings
    preserve_dates: bool = True
    preserve_numbers: bool = True
    preserve_names: bool = True


class MemorySummarizer:
    """Intelligent memory summarization engine.
    
    Provides AI-powered summarization for memory consolidation,
    enabling significant footprint reduction while preserving key information.
    
    Example:
        >>> summarizer = MemorySummarizer()
        >>> 
        >>> # Summarize memories
        >>> memories = [
        ...     MemoryEntry("1", "Meeting discussed project timeline..."),
        ...     MemoryEntry("2", "Action items from meeting: ..."),
        ...     MemoryEntry("3", "Follow-up notes on timeline..."),
        ... ]
        >>> result = summarizer.summarize(memories)
        >>> 
        >>> print(f"Compressed {result.original_word_count} words to {result.summarized_word_count}")
        >>> print(f"Summary: {result.summaries[0].content}")
    
    Advanced usage with LLM:
        >>> from my_llm import MyLLMProvider
        >>> 
        >>> config = SummarizationConfig(
        ...     strategy=SummarizationStrategy.ABSTRACTIVE,
        ...     default_level=SummaryLevel.DETAILED,
        ... )
        >>> summarizer = MemorySummarizer(
        ...     config=config,
        ...     llm_provider=MyLLMProvider(),
        ... )
    """
    
    def __init__(
        self,
        config: Optional[SummarizationConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        """Initialize the summarizer.
        
        Args:
            config: Summarization configuration
            llm_provider: LLM provider for abstractive summarization
        """
        self.config = config or SummarizationConfig()
        self.llm_provider = llm_provider
        
        # Initialize summarizers
        self._summarizers: dict[SummarizationStrategy, Summarizer] = {}
        self._init_summarizers()
    
    def _init_summarizers(self) -> None:
        """Initialize all summarizers."""
        # Extractive
        self._summarizers[SummarizationStrategy.EXTRACTIVE] = ExtractiveSummarizer(
            max_sentences=self.config.max_sentences,
            min_sentence_length=self.config.min_sentence_length,
            preserve_dates=self.config.preserve_dates,
            preserve_numbers=self.config.preserve_numbers,
        )
        
        # Abstractive
        self._summarizers[SummarizationStrategy.ABSTRACTIVE] = AbstractiveSummarizer(
            llm_provider=self.llm_provider,
            max_input_tokens=self.config.max_input_tokens,
        )
        
        # Hierarchical
        base_summarizer = (
            self._summarizers[SummarizationStrategy.ABSTRACTIVE]
            if self.llm_provider
            else self._summarizers[SummarizationStrategy.EXTRACTIVE]
        )
        self._summarizers[SummarizationStrategy.HIERARCHICAL] = HierarchicalSummarizer(
            base_summarizer=base_summarizer,
            chunk_size=self.config.chunk_size,
            max_depth=self.config.max_depth,
        )
        
        # Incremental
        self._summarizers[SummarizationStrategy.INCREMENTAL] = IncrementalSummarizer(
            llm_provider=self.llm_provider,
            base_summarizer=base_summarizer,
            max_summary_length=self.config.max_summary_length,
        )
    
    def summarize(
        self,
        memories: list[MemoryEntry],
        strategy: Optional[SummarizationStrategy] = None,
        level: Optional[SummaryLevel] = None,
    ) -> SummarizationResult:
        """Summarize memories.
        
        Args:
            memories: Memories to summarize
            strategy: Override strategy
            level: Override summary level
            
        Returns:
            SummarizationResult
        """
        import time
        start_time = time.perf_counter()
        
        used_strategy = strategy or self.config.strategy
        used_level = level or self.config.default_level
        
        summarizer = self._summarizers.get(used_strategy)
        if not summarizer:
            raise ValueError(f"Unknown strategy: {used_strategy}")
        
        # Generate summary
        summary = summarizer.summarize(memories, used_level)
        
        # Calculate metrics
        original_words = sum(len(m.content.split()) for m in memories)
        
        end_time = time.perf_counter()
        
        return SummarizationResult(
            summaries=[summary],
            original_memory_count=len(memories),
            original_word_count=original_words,
            summarized_word_count=summary.word_count,
            compression_ratio=summary.compression_ratio,
            processing_time_ms=(end_time - start_time) * 1000,
            strategy_used=used_strategy,
            stats={
                "level_used": used_level.value,
                "key_topics": summary.key_topics,
                "source_memory_count": len(summary.source_memory_ids),
            },
        )
    
    def summarize_by_category(
        self,
        memories: list[MemoryEntry],
        level: Optional[SummaryLevel] = None,
    ) -> dict[str, SummarizationResult]:
        """Summarize memories grouped by category.
        
        Args:
            memories: Memories to summarize
            level: Summary level
            
        Returns:
            Dictionary mapping category to summarization result
        """
        # Group by category
        by_category: dict[str, list[MemoryEntry]] = defaultdict(list)
        for mem in memories:
            category = mem.category or "uncategorized"
            by_category[category].append(mem)
        
        # Summarize each category
        results = {}
        for category, category_memories in by_category.items():
            results[category] = self.summarize(category_memories, level=level)
        
        return results
    
    def create_hierarchy(
        self,
        memories: list[MemoryEntry],
        level: Optional[SummaryLevel] = None,
    ) -> HierarchicalSummary:
        """Create hierarchical summary structure.
        
        Args:
            memories: Memories to summarize
            level: Summary level
            
        Returns:
            HierarchicalSummary tree
        """
        hierarchical = self._summarizers.get(SummarizationStrategy.HIERARCHICAL)
        if not isinstance(hierarchical, HierarchicalSummarizer):
            raise ValueError("Hierarchical summarizer not configured")
        
        return hierarchical.create_hierarchy(
            memories,
            level or self.config.default_level,
        )
    
    def update_incrementally(
        self,
        summary_key: str,
        new_memories: list[MemoryEntry],
        level: Optional[SummaryLevel] = None,
    ) -> Summary:
        """Update summary incrementally with new memories.
        
        Args:
            summary_key: Unique key for the summary
            new_memories: New memories to incorporate
            level: Summary level
            
        Returns:
            Updated summary
        """
        incremental = self._summarizers.get(SummarizationStrategy.INCREMENTAL)
        if not isinstance(incremental, IncrementalSummarizer):
            raise ValueError("Incremental summarizer not configured")
        
        return incremental.add_memories(
            summary_key,
            new_memories,
            level or self.config.default_level,
        )
    
    def estimate_reduction(
        self,
        memories: list[MemoryEntry],
        strategy: Optional[SummarizationStrategy] = None,
        level: Optional[SummaryLevel] = None,
    ) -> dict[str, Any]:
        """Estimate compression from summarization.
        
        Args:
            memories: Memories to analyze
            strategy: Strategy to use
            level: Summary level
            
        Returns:
            Dictionary with reduction estimates
        """
        result = self.summarize(memories, strategy, level)
        
        return {
            "original_memory_count": result.original_memory_count,
            "original_word_count": result.original_word_count,
            "projected_word_count": result.summarized_word_count,
            "reduction_percent": result.reduction_percent,
            "compression_ratio": result.compression_ratio,
            "strategy": result.strategy_used.value,
            "key_topics": result.stats.get("key_topics", []),
        }


# Convenience functions

def summarize_memories(
    memories: list[MemoryEntry],
    strategy: SummarizationStrategy = SummarizationStrategy.EXTRACTIVE,
    level: SummaryLevel = SummaryLevel.STANDARD,
) -> Summary:
    """Summarize memories.
    
    Convenience function for one-off summarization.
    
    Args:
        memories: Memories to summarize
        strategy: Summarization strategy
        level: Summary detail level
        
    Returns:
        Summary object
    """
    summarizer = MemorySummarizer()
    result = summarizer.summarize(memories, strategy, level)
    return result.summaries[0] if result.summaries else Summary(
        summary_id="empty",
        content="",
        source_memory_ids=[],
        level=level,
        strategy=strategy,
    )


def create_hierarchical_summary(
    memories: list[MemoryEntry],
    level: SummaryLevel = SummaryLevel.STANDARD,
) -> HierarchicalSummary:
    """Create hierarchical summary.
    
    Convenience function for one-off hierarchical summarization.
    
    Args:
        memories: Memories to summarize
        level: Summary detail level
        
    Returns:
        HierarchicalSummary tree
    """
    summarizer = MemorySummarizer()
    return summarizer.create_hierarchy(memories, level)
