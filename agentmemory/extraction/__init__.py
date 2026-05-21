"""
Structured Memory Extraction Module

Extracts structured memories from conversations into six cognitive domains:
- Biography: personal details, background, identity
- Preferences: likes, dislikes, styles, choices
- Work: projects, skills, tools, professional context
- Social: relationships, contacts, connections
- Temporal: schedules, events, deadlines, time-based info
- Procedural: how-tos, workflows, processes

Supports both LLM-based and rule-based extraction.
"""

from .domains import CognitiveDomain, Memory, ExtractionResult
from .extractor import MemoryExtractor
from .rule_extractor import RuleBasedExtractor
from .llm_extractor import LLMExtractor
from .deduplication import MemoryDeduplicator
from .conflict_resolver import ConflictResolver
from .merger import MemoryMerger

__all__ = [
    "CognitiveDomain",
    "Memory",
    "ExtractionResult",
    "MemoryExtractor",
    "RuleBasedExtractor",
    "LLMExtractor",
    "MemoryDeduplicator",
    "ConflictResolver",
    "MemoryMerger",
]
