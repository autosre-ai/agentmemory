"""
Cognitive Domain Models

Defines the six cognitive domains for memory extraction and
data models for structured memories.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import hashlib
import json


class CognitiveDomain(Enum):
    """Six cognitive domains for memory categorization."""
    
    BIOGRAPHY = "biography"       # Personal details: name, role, background, identity
    PREFERENCES = "preferences"   # Likes, dislikes, styles, choices
    WORK = "work"                 # Projects, skills, tools, professional context
    SOCIAL = "social"             # Relationships, contacts, connections
    TEMPORAL = "temporal"         # Schedules, events, deadlines, time-based info
    PROCEDURAL = "procedural"     # How-tos, workflows, processes, instructions
    
    @classmethod
    def from_string(cls, value: str) -> "CognitiveDomain":
        """Parse domain from string, case-insensitive."""
        value = value.lower().strip()
        for domain in cls:
            if domain.value == value:
                return domain
        raise ValueError(f"Unknown domain: {value}")


@dataclass
class Memory:
    """
    A structured memory unit extracted from text.
    
    Attributes:
        domain: The cognitive domain this memory belongs to
        key: A short identifier/label for the memory
        value: The actual content/fact
        confidence: Confidence score (0.0 to 1.0)
        source: Optional source reference (e.g., message ID)
        timestamp: When the memory was extracted
        metadata: Additional context
        memory_id: Unique hash-based identifier
    """
    
    domain: CognitiveDomain
    key: str
    value: str
    confidence: float = 1.0
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: str = field(default="")
    
    def __post_init__(self):
        """Generate memory_id if not provided."""
        if not self.memory_id:
            self.memory_id = self._generate_id()
        # Clamp confidence
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def _generate_id(self) -> str:
        """Generate unique ID based on domain, key, and value."""
        content = f"{self.domain.value}:{self.key}:{self.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "domain": self.domain.value,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Create Memory from dictionary."""
        return cls(
            domain=CognitiveDomain.from_string(data["domain"]),
            key=data["key"],
            value=data["value"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            metadata=data.get("metadata", {}),
            memory_id=data.get("memory_id", ""),
        )
    
    def __eq__(self, other: Any) -> bool:
        """Two memories are equal if they have the same domain, key, and value."""
        if not isinstance(other, Memory):
            return False
        return (
            self.domain == other.domain
            and self.key.lower() == other.key.lower()
            and self.value.lower() == other.value.lower()
        )
    
    def __hash__(self) -> int:
        """Hash based on domain, key, and value."""
        return hash((self.domain, self.key.lower(), self.value.lower()))
    
    def similar_to(self, other: "Memory", threshold: float = 0.8) -> bool:
        """
        Check if this memory is similar to another.
        Uses simple text overlap for fast comparison.
        """
        if self.domain != other.domain:
            return False
        
        # Compare keys
        key_sim = self._jaccard_similarity(self.key.lower(), other.key.lower())
        
        # Compare values
        value_sim = self._jaccard_similarity(self.value.lower(), other.value.lower())
        
        # Weighted average (value is more important)
        similarity = 0.3 * key_sim + 0.7 * value_sim
        return similarity >= threshold
    
    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)


@dataclass
class ExtractionResult:
    """
    Result of memory extraction from text.
    
    Attributes:
        memories: List of extracted memories
        text: Original input text
        method: Extraction method used ("llm" or "rule")
        processing_time_ms: Time taken to process
        errors: Any errors encountered
    """
    
    memories: list[Memory] = field(default_factory=list)
    text: str = ""
    method: str = "rule"
    processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memories": [m.to_dict() for m in self.memories],
            "text": self.text,
            "method": self.method,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionResult":
        """Create from dictionary."""
        return cls(
            memories=[Memory.from_dict(m) for m in data.get("memories", [])],
            text=data.get("text", ""),
            method=data.get("method", "rule"),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            errors=data.get("errors", []),
        )
    
    def filter_by_domain(self, domain: CognitiveDomain) -> list[Memory]:
        """Get memories for a specific domain."""
        return [m for m in self.memories if m.domain == domain]
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> list[Memory]:
        """Get memories above a confidence threshold."""
        return [m for m in self.memories if m.confidence >= min_confidence]
    
    def __len__(self) -> int:
        return len(self.memories)
    
    def __iter__(self):
        return iter(self.memories)


# Domain-specific extraction prompts for LLM-based extraction
DOMAIN_PROMPTS = {
    CognitiveDomain.BIOGRAPHY: """
Extract BIOGRAPHY information: personal identity, background, demographic details.
Look for:
- Name, nicknames, titles
- Age, birthday, location
- Education, degrees, institutions
- Cultural background, nationality
- Personal history, life events

Format each as: KEY: VALUE (with confidence 0-1)
Example: "name: John Smith (0.95)"
""",
    
    CognitiveDomain.PREFERENCES: """
Extract PREFERENCES information: likes, dislikes, personal choices, styles.
Look for:
- Favorite things (colors, foods, activities)
- Dislikes and aversions
- Communication style preferences
- Work style preferences
- Aesthetic preferences

Format each as: KEY: VALUE (with confidence 0-1)
Example: "preferred_language: Python (0.9)"
""",
    
    CognitiveDomain.WORK: """
Extract WORK information: professional context, skills, projects.
Look for:
- Job title, role, company
- Skills and expertise
- Current projects
- Tools and technologies used
- Professional goals

Format each as: KEY: VALUE (with confidence 0-1)
Example: "primary_language: Python (0.95)"
""",
    
    CognitiveDomain.SOCIAL: """
Extract SOCIAL information: relationships, contacts, connections.
Look for:
- Family members and relations
- Colleagues, mentors, friends
- Professional connections
- Team members
- Community affiliations

Format each as: KEY: VALUE (with confidence 0-1)
Example: "manager: Sarah Chen (0.85)"
""",
    
    CognitiveDomain.TEMPORAL: """
Extract TEMPORAL information: schedules, events, deadlines.
Look for:
- Regular schedules (work hours, meetings)
- Upcoming events and deadlines
- Recurring appointments
- Time zone, availability
- Important dates

Format each as: KEY: VALUE (with confidence 0-1)
Example: "timezone: PST (0.9)"
""",
    
    CognitiveDomain.PROCEDURAL: """
Extract PROCEDURAL information: workflows, processes, how-tos.
Look for:
- Preferred workflows
- Step-by-step processes
- Best practices mentioned
- Automation rules
- Decision procedures

Format each as: KEY: VALUE (with confidence 0-1)
Example: "code_review_process: PR must have 2 approvals (0.85)"
""",
}
