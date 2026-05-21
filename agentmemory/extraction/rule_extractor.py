"""
Rule-Based Memory Extractor

Lightweight extraction using regex patterns and heuristics.
Works completely offline without LLM dependencies.
"""

import re
from datetime import datetime
from typing import Optional
from .domains import CognitiveDomain, Memory


class RuleBasedExtractor:
    """
    Extract structured memories using rule-based patterns.
    
    Provides fast, offline extraction with moderate accuracy.
    Best for common patterns and explicit statements.
    """
    
    def __init__(self):
        """Initialize with pattern definitions."""
        self._patterns = self._build_patterns()
    
    def extract(self, text: str, source: Optional[str] = None) -> list[Memory]:
        """
        Extract memories from text using rule-based patterns.
        
        Args:
            text: Input text to analyze
            source: Optional source identifier
            
        Returns:
            List of extracted Memory objects
        """
        memories = []
        text_lower = text.lower()
        
        # Apply domain-specific extraction
        memories.extend(self._extract_biography(text, text_lower, source))
        memories.extend(self._extract_preferences(text, text_lower, source))
        memories.extend(self._extract_work(text, text_lower, source))
        memories.extend(self._extract_social(text, text_lower, source))
        memories.extend(self._extract_temporal(text, text_lower, source))
        memories.extend(self._extract_procedural(text, text_lower, source))
        
        return memories
    
    def _build_patterns(self) -> dict[str, list[tuple[re.Pattern, str, float]]]:
        """Build regex patterns for each domain."""
        return {
            "biography": [
                # Name patterns
                (re.compile(r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", re.I), "name", 0.9),
                (re.compile(r"(?:i'm|i am)\s+(\d{1,3})\s+years?\s+old", re.I), "age", 0.85),
                (re.compile(r"(?:i live in|i'm from|based in|living in)\s+([A-Z][a-zA-Z\s,]+)", re.I), "location", 0.8),
                (re.compile(r"(?:graduated from|went to|studied at)\s+([A-Z][a-zA-Z\s]+(?:University|College|Institute|School))", re.I), "education", 0.85),
                (re.compile(r"(?:my birthday is|born on|birthdate:?)\s+([A-Za-z]+\s+\d{1,2}(?:,?\s+\d{4})?)", re.I), "birthday", 0.9),
            ],
            "preferences": [
                # Likes/dislikes
                (re.compile(r"(?:i prefer|i like|i love|i enjoy)\s+([^.!?\n]+)", re.I), "likes", 0.75),
                (re.compile(r"(?:i hate|i dislike|i don't like|i can't stand)\s+([^.!?\n]+)", re.I), "dislikes", 0.75),
                (re.compile(r"(?:my favorite|my preferred)\s+(\w+)\s+is\s+([^.!?\n]+)", re.I), "favorite_{0}", 0.8),
                (re.compile(r"(?:i always|i usually|i tend to)\s+([^.!?\n]+)", re.I), "habit", 0.7),
                # Programming preferences
                (re.compile(r"(?:i prefer|i use|i like)\s+(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|Ruby|Swift)", re.I), "preferred_language", 0.85),
                (re.compile(r"(?:using|prefer|like)\s+(VS Code|VSCode|Vim|Neovim|Emacs|IntelliJ|PyCharm|Cursor)", re.I), "preferred_editor", 0.85),
            ],
            "work": [
                # Job/role
                (re.compile(r"(?:i work as|i'm a|i am a|my role is|my job is)\s+([^.!?\n]+)", re.I), "role", 0.85),
                (re.compile(r"(?:i work at|i work for|employed at|employed by)\s+([A-Z][a-zA-Z\s&]+)", re.I), "company", 0.85),
                (re.compile(r"(?:i'm working on|working on|my project is|current project:?)\s+([^.!?\n]+)", re.I), "current_project", 0.8),
                # Skills
                (re.compile(r"(?:i know|i can use|i'm proficient in|skilled in|experience with)\s+([^.!?\n]+)", re.I), "skill", 0.75),
                (re.compile(r"(?:using|built with|tech stack:?)\s+(React|Vue|Angular|Django|Flask|FastAPI|Express|Next\.js)", re.I), "technology", 0.85),
            ],
            "social": [
                # Relationships
                (re.compile(r"(?:my (?:wife|husband|spouse|partner) is|married to)\s+([A-Z][a-zA-Z]+)", re.I), "spouse", 0.9),
                (re.compile(r"(?:my (?:boss|manager) is|report to)\s+([A-Z][a-zA-Z\s]+)", re.I), "manager", 0.85),
                (re.compile(r"(?:my (?:team|colleagues?) include)\s+([^.!?\n]+)", re.I), "team_members", 0.75),
                (re.compile(r"(?:my (?:friend|buddy|pal))\s+([A-Z][a-zA-Z]+)", re.I), "friend", 0.7),
            ],
            "temporal": [
                # Schedules
                (re.compile(r"(?:i work|my hours are|available)\s+(?:from\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:to|-)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", re.I), "work_hours", 0.85),
                (re.compile(r"(?:my timezone is|i'm in|timezone:?)\s*((?:PST|EST|CST|MST|UTC|GMT)(?:[+-]\d+)?|[A-Z][a-z]+/[A-Z][a-z_]+)", re.I), "timezone", 0.9),
                (re.compile(r"(?:meeting|call|appointment)\s+(?:on|at)\s+([A-Za-z]+day(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)", re.I), "meeting", 0.8),
                (re.compile(r"(?:deadline|due date|due by)\s+(?:is\s+)?([A-Za-z]+\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I), "deadline", 0.85),
            ],
            "procedural": [
                # Processes
                (re.compile(r"(?:i always|always)\s+(start|begin|end|finish)\s+(?:by|with)\s+([^.!?\n]+)", re.I), "workflow_step", 0.75),
                (re.compile(r"(?:step \d+:?|first|then|next|finally)[,:]?\s+([^.!?\n]+)", re.I), "process_step", 0.7),
                (re.compile(r"(?:my process|my workflow|i usually)\s+(?:is\s+)?([^.!?\n]+)", re.I), "workflow", 0.75),
                (re.compile(r"(?:before|after)\s+(?:i|we)\s+([^.!?,\n]+)", re.I), "conditional_action", 0.7),
                (re.compile(r"(?:make sure to|don't forget to|remember to|always)\s+([^.!?\n]+)", re.I), "reminder", 0.75),
            ],
        }
    
    def _extract_biography(self, text: str, text_lower: str, source: Optional[str]) -> list[Memory]:
        """Extract biography memories."""
        memories = []
        for pattern, key, confidence in self._patterns["biography"]:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value and len(value) > 1:
                    memories.append(Memory(
                        domain=CognitiveDomain.BIOGRAPHY,
                        key=key,
                        value=value,
                        confidence=confidence,
                        source=source,
                        metadata={"extraction_method": "rule", "pattern": pattern.pattern[:50]},
                    ))
        return memories
    
    def _extract_preferences(self, text: str, text_lower: str, source: Optional[str]) -> list[Memory]:
        """Extract preference memories."""
        memories = []
        for pattern, key_template, confidence in self._patterns["preferences"]:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) == 2:
                    # Pattern like "my favorite {thing} is {value}"
                    key = key_template.format(groups[0].lower())
                    value = groups[1].strip()
                else:
                    key = key_template
                    value = groups[0].strip()
                
                if value and len(value) > 1:
                    memories.append(Memory(
                        domain=CognitiveDomain.PREFERENCES,
                        key=key,
                        value=value,
                        confidence=confidence,
                        source=source,
                        metadata={"extraction_method": "rule"},
                    ))
        return memories
    
    def _extract_work(self, text: str, text_lower: str, source: Optional[str]) -> list[Memory]:
        """Extract work-related memories."""
        memories = []
        for pattern, key, confidence in self._patterns["work"]:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value and len(value) > 1:
                    memories.append(Memory(
                        domain=CognitiveDomain.WORK,
                        key=key,
                        value=value,
                        confidence=confidence,
                        source=source,
                        metadata={"extraction_method": "rule"},
                    ))
        return memories
    
    def _extract_social(self, text: str, text_lower: str, source: Optional[str]) -> list[Memory]:
        """Extract social relationship memories."""
        memories = []
        for pattern, key, confidence in self._patterns["social"]:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if value and len(value) > 1:
                    memories.append(Memory(
                        domain=CognitiveDomain.SOCIAL,
                        key=key,
                        value=value,
                        confidence=confidence,
                        source=source,
                        metadata={"extraction_method": "rule"},
                    ))
        return memories
    
    def _extract_temporal(self, text: str, text_lower: str, source: Optional[str]) -> list[Memory]:
        """Extract temporal memories (schedules, times, events)."""
        memories = []
        for pattern, key, confidence in self._patterns["temporal"]:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) == 2:
                    # Work hours pattern
                    value = f"{groups[0]} to {groups[1]}"
                else:
                    value = groups[0].strip()
                
                if value and len(value) > 1:
                    memories.append(Memory(
                        domain=CognitiveDomain.TEMPORAL,
                        key=key,
                        value=value,
                        confidence=confidence,
                        source=source,
                        metadata={"extraction_method": "rule"},
                    ))
        return memories
    
    def _extract_procedural(self, text: str, text_lower: str, source: Optional[str]) -> list[Memory]:
        """Extract procedural memories (workflows, how-tos)."""
        memories = []
        for pattern, key, confidence in self._patterns["procedural"]:
            for match in pattern.finditer(text):
                value = match.group(1).strip() if match.lastindex == 1 else f"{match.group(1)} {match.group(2)}".strip()
                if value and len(value) > 2:
                    memories.append(Memory(
                        domain=CognitiveDomain.PROCEDURAL,
                        key=key,
                        value=value,
                        confidence=confidence,
                        source=source,
                        metadata={"extraction_method": "rule"},
                    ))
        return memories
    
    def extract_from_key_value(self, text: str, source: Optional[str] = None) -> list[Memory]:
        """
        Extract from explicit key-value format.
        Useful for structured input like "name: John" or "role: developer".
        """
        memories = []
        
        # Match patterns like "key: value" or "key = value"
        kv_pattern = re.compile(r"^([a-zA-Z_]+)\s*[:=]\s*(.+)$", re.MULTILINE)
        
        domain_hints = {
            "name": CognitiveDomain.BIOGRAPHY,
            "age": CognitiveDomain.BIOGRAPHY,
            "location": CognitiveDomain.BIOGRAPHY,
            "education": CognitiveDomain.BIOGRAPHY,
            "birthday": CognitiveDomain.BIOGRAPHY,
            "likes": CognitiveDomain.PREFERENCES,
            "dislikes": CognitiveDomain.PREFERENCES,
            "favorite": CognitiveDomain.PREFERENCES,
            "preferred": CognitiveDomain.PREFERENCES,
            "role": CognitiveDomain.WORK,
            "company": CognitiveDomain.WORK,
            "skill": CognitiveDomain.WORK,
            "project": CognitiveDomain.WORK,
            "technology": CognitiveDomain.WORK,
            "team": CognitiveDomain.SOCIAL,
            "manager": CognitiveDomain.SOCIAL,
            "colleague": CognitiveDomain.SOCIAL,
            "friend": CognitiveDomain.SOCIAL,
            "schedule": CognitiveDomain.TEMPORAL,
            "timezone": CognitiveDomain.TEMPORAL,
            "deadline": CognitiveDomain.TEMPORAL,
            "meeting": CognitiveDomain.TEMPORAL,
            "workflow": CognitiveDomain.PROCEDURAL,
            "process": CognitiveDomain.PROCEDURAL,
            "step": CognitiveDomain.PROCEDURAL,
        }
        
        for match in kv_pattern.finditer(text):
            key = match.group(1).lower()
            value = match.group(2).strip()
            
            # Determine domain from key
            domain = CognitiveDomain.BIOGRAPHY  # default
            for hint_key, hint_domain in domain_hints.items():
                if hint_key in key:
                    domain = hint_domain
                    break
            
            if value:
                memories.append(Memory(
                    domain=domain,
                    key=key,
                    value=value,
                    confidence=0.95,  # High confidence for explicit key-value
                    source=source,
                    metadata={"extraction_method": "key_value"},
                ))
        
        return memories
