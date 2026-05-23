"""
Memory Extractor

Main entry point for structured memory extraction.
Combines rule-based and LLM-based extraction with
deduplication and conflict resolution.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from .domains import CognitiveDomain, Memory, ExtractionResult
from .rule_extractor import RuleBasedExtractor
from .llm_extractor import LLMExtractor, LLMClient, MockLLMClient
from .deduplication import MemoryDeduplicator
from .conflict_resolver import ConflictResolver, ConflictStrategy
from .merger import MemoryMerger


class MemoryExtractor:
    """
    Production-ready memory extractor.
    
    Extracts structured memories from text using:
    - Rule-based extraction (fast, offline)
    - LLM-based extraction (accurate, requires API)
    - Hybrid extraction (best of both)
    
    Includes automatic deduplication and conflict resolution.
    
    Example:
        >>> extractor = MemoryExtractor()
        >>> result = extractor.extract("My name is John. I work at Google.")
        >>> for memory in result.memories:
        ...     print(f"{memory.domain.value}: {memory.key} = {memory.value}")
        biography: name = John
        work: company = Google
    """
    
    def __init__(
        self,
        mode: str = "rule",
        llm_client: Optional[LLMClient] = None,
        llm_provider: str = "openai",
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        deduplication_strategy: str = "fuzzy",
        similarity_threshold: float = 0.85,
        conflict_strategy: ConflictStrategy = ConflictStrategy.CONFIDENCE_WINS,
        auto_dedupe: bool = True,
        auto_resolve_conflicts: bool = True,
    ):
        """
        Initialize memory extractor.
        
        Args:
            mode: Extraction mode ("rule", "llm", or "hybrid")
            llm_client: Pre-configured LLM client (optional)
            llm_provider: LLM provider ("openai" or "anthropic")
            llm_api_key: API key for LLM provider
            llm_model: Model to use for LLM extraction
            deduplication_strategy: Strategy for deduplication ("exact", "fuzzy", "semantic")
            similarity_threshold: Threshold for fuzzy/semantic deduplication
            conflict_strategy: Strategy for resolving conflicts
            auto_dedupe: Automatically deduplicate extracted memories
            auto_resolve_conflicts: Automatically resolve conflicts
        """
        self.mode = mode.lower()
        self.auto_dedupe = auto_dedupe
        self.auto_resolve_conflicts = auto_resolve_conflicts
        
        # Initialize extractors
        self._rule_extractor = RuleBasedExtractor()
        self._llm_extractor: Optional[LLMExtractor] = None
        
        if self.mode in ("llm", "hybrid"):
            self._llm_extractor = LLMExtractor(
                client=llm_client,
                provider=llm_provider,
                api_key=llm_api_key,
                model=llm_model,
            )
        
        # Initialize processors
        self._deduplicator = MemoryDeduplicator(
            strategy=deduplication_strategy,
            similarity_threshold=similarity_threshold,
        )
        self._conflict_resolver = ConflictResolver(strategy=conflict_strategy)
        self._merger = MemoryMerger(
            deduplicator=self._deduplicator,
            conflict_resolver=self._conflict_resolver,
        )
    
    def extract(
        self,
        text: str,
        source: Optional[str] = None,
        domains: Optional[list[CognitiveDomain]] = None,
    ) -> ExtractionResult:
        """
        Extract memories from text.
        
        Args:
            text: Input text to analyze
            source: Optional source identifier (e.g., message ID)
            domains: Specific domains to extract (None = all)
            
        Returns:
            ExtractionResult containing extracted memories
        """
        start_time = time.time()
        errors: list[str] = []
        memories: list[Memory] = []
        method = self.mode
        
        try:
            if self.mode == "rule":
                memories = self._extract_rule(text, source)
            elif self.mode == "llm":
                memories = self._extract_llm(text, source, domains)
            elif self.mode == "hybrid":
                memories = self._extract_hybrid(text, source, domains)
                method = "hybrid"
            else:
                errors.append(f"Unknown extraction mode: {self.mode}")
        except Exception as e:
            errors.append(f"Extraction error: {str(e)}")
        
        # Filter by domains if specified
        if domains and memories:
            memories = [m for m in memories if m.domain in domains]
        
        # Post-processing
        if memories:
            if self.auto_dedupe:
                dedupe_result = self._deduplicator.deduplicate(memories)
                memories = dedupe_result.unique_memories
            
            if self.auto_resolve_conflicts:
                resolution_result = self._conflict_resolver.resolve(memories)
                memories = resolution_result.resolved_memories
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return ExtractionResult(
            memories=memories,
            text=text,
            method=method,
            processing_time_ms=processing_time,
            errors=errors,
        )
    
    def _extract_rule(self, text: str, source: Optional[str]) -> list[Memory]:
        """Extract using rule-based patterns."""
        memories = self._rule_extractor.extract(text, source)
        
        # Also try key-value extraction
        kv_memories = self._rule_extractor.extract_from_key_value(text, source)
        memories.extend(kv_memories)
        
        return memories
    
    def _extract_llm(
        self,
        text: str,
        source: Optional[str],
        domains: Optional[list[CognitiveDomain]] = None
    ) -> list[Memory]:
        """Extract using LLM."""
        if not self._llm_extractor:
            return []
        
        if domains and len(domains) == 1:
            # Domain-specific extraction
            return self._llm_extractor.extract_domain(text, domains[0], source)
        else:
            # General extraction
            return self._llm_extractor.extract(text, source)
    
    def _extract_hybrid(
        self,
        text: str,
        source: Optional[str],
        domains: Optional[list[CognitiveDomain]] = None
    ) -> list[Memory]:
        """
        Hybrid extraction: rule-based + LLM.
        
        Uses rule-based for explicit patterns, LLM for implicit/complex.
        Merges results with deduplication.
        """
        # Rule-based extraction
        rule_memories = self._extract_rule(text, source)
        
        # LLM extraction
        llm_memories = self._extract_llm(text, source, domains)
        
        # Mark extraction method in metadata
        for m in rule_memories:
            m.metadata["hybrid_source"] = "rule"
        for m in llm_memories:
            m.metadata["hybrid_source"] = "llm"
        
        # Merge results
        merge_result = self._merger.merge(
            rule_memories,
            llm_memories,
            source_labels=["rule", "llm"]
        )
        
        return merge_result.merged_memories
    
    def extract_conversation(
        self,
        messages: list[dict[str, str]],
        source: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract memories from a conversation.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            source: Optional conversation identifier
            
        Returns:
            ExtractionResult with memories from entire conversation
        """
        start_time = time.time()
        all_memories: list[Memory] = []
        errors: list[str] = []
        
        for i, message in enumerate(messages):
            content = message.get("content", "")
            role = message.get("role", "unknown")
            msg_source = f"{source}:{i}" if source else f"msg_{i}"
            
            try:
                result = self.extract(content, source=msg_source)
                
                # Add conversation context to metadata
                for memory in result.memories:
                    memory.metadata["message_role"] = role
                    memory.metadata["message_index"] = i
                
                all_memories.extend(result.memories)
                errors.extend(result.errors)
            except Exception as e:
                errors.append(f"Error processing message {i}: {str(e)}")
        
        # Deduplicate across entire conversation
        if all_memories and self.auto_dedupe:
            dedupe_result = self._deduplicator.deduplicate(all_memories)
            all_memories = dedupe_result.unique_memories
        
        # Resolve conflicts
        if all_memories and self.auto_resolve_conflicts:
            resolution_result = self._conflict_resolver.resolve(all_memories)
            all_memories = resolution_result.resolved_memories
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            memories=all_memories,
            text=str(messages),
            method=f"{self.mode}_conversation",
            processing_time_ms=processing_time,
            errors=errors,
        )
    
    def extract_by_domain(
        self,
        text: str,
        source: Optional[str] = None,
    ) -> dict[CognitiveDomain, list[Memory]]:
        """
        Extract and organize memories by domain.
        
        Args:
            text: Input text
            source: Optional source identifier
            
        Returns:
            Dict mapping each domain to its extracted memories
        """
        result = self.extract(text, source)
        
        by_domain: dict[CognitiveDomain, list[Memory]] = {
            domain: [] for domain in CognitiveDomain
        }
        
        for memory in result.memories:
            by_domain[memory.domain].append(memory)
        
        return by_domain
    
    def merge_memories(
        self,
        *memory_sets: list[Memory],
        source_labels: Optional[list[str]] = None
    ) -> list[Memory]:
        """
        Merge multiple sets of memories.
        
        Args:
            *memory_sets: Variable number of memory lists
            source_labels: Optional labels for sources
            
        Returns:
            Merged and deduplicated memory list
        """
        result = self._merger.merge(*memory_sets, source_labels=source_labels)
        return result.merged_memories
    
    def add_memory(
        self,
        existing: list[Memory],
        new_memory: Memory
    ) -> tuple[list[Memory], bool]:
        """
        Add a new memory to existing memories.
        
        Handles deduplication and conflict resolution.
        
        Args:
            existing: Existing memory list
            new_memory: New memory to add
            
        Returns:
            Tuple of (updated memories, whether memory was added/updated)
        """
        updated, replaced = self._merger.incremental_merge(existing, new_memory)
        was_updated = replaced is not None or new_memory in updated
        return updated, was_updated


# Convenience function for quick extraction
def extract_memories(
    text: str,
    mode: str = "rule",
    source: Optional[str] = None,
) -> list[Memory]:
    """
    Quick extraction function.
    
    Args:
        text: Text to extract from
        mode: Extraction mode ("rule", "llm", "hybrid")
        source: Optional source identifier
        
    Returns:
        List of extracted memories
    """
    extractor = MemoryExtractor(mode=mode)
    result = extractor.extract(text, source=source)
    return result.memories
