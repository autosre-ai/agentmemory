"""
Conflict detection for memory consolidation.

Detects various types of conflicts between memories:
- Value conflicts (same topic, different values)
- Temporal conflicts (conflicting time references)
- Source conflicts (contradicting sources)
- Semantic contradictions (opposing meanings)
"""

import logging
import re
from datetime import datetime
from typing import Any, Callable, Optional
import uuid

from .models import (
    MemoryConflict,
    ConflictType,
    ConflictSeverity,
)
from .similarity import MemoryData, SimilarityDetector

logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    Detect conflicts between memories.
    
    Conflict types:
    - VALUE_CONFLICT: Same subject, different values
    - TEMPORAL_CONFLICT: Inconsistent time references
    - SOURCE_CONFLICT: Contradicting source information
    - CONFIDENCE_GAP: Large confidence difference for similar content
    - SEMANTIC_CONTRADICTION: Semantically opposing statements
    """
    
    def __init__(
        self,
        similarity_detector: Optional[SimilarityDetector] = None,
        contradiction_detector: Optional[Callable[[str, str], float]] = None,
        confidence_gap_threshold: float = 0.5,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize conflict detector.
        
        Args:
            similarity_detector: For finding related memories
            contradiction_detector: Function to detect semantic contradictions
            confidence_gap_threshold: Minimum gap to flag confidence conflicts
            similarity_threshold: Minimum similarity for conflict checks
        """
        self.similarity_detector = similarity_detector or SimilarityDetector()
        self.contradiction_detector = contradiction_detector
        self.confidence_gap_threshold = confidence_gap_threshold
        self.similarity_threshold = similarity_threshold
        
        # Temporal patterns
        self.date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
        ]
        
        # Negation patterns for simple contradiction detection
        self.negation_patterns = [
            (r"is not", r"is"),
            (r"isn't", r"is"),
            (r"are not", r"are"),
            (r"aren't", r"are"),
            (r"was not", r"was"),
            (r"wasn't", r"was"),
            (r"were not", r"were"),
            (r"weren't", r"were"),
            (r"does not", r"does"),
            (r"doesn't", r"does"),
            (r"do not", r"do"),
            (r"don't", r"do"),
            (r"cannot", r"can"),
            (r"can't", r"can"),
            (r"never", r"always"),
            (r"no longer", r"still"),
        ]
    
    def detect_all_conflicts(
        self,
        memories: list[MemoryData],
    ) -> list[MemoryConflict]:
        """
        Detect all conflicts among memories.
        
        Args:
            memories: List of memories to check
            
        Returns:
            List of MemoryConflict objects
        """
        conflicts = []
        
        # Find similar memory pairs first
        similar_pairs = self.similarity_detector.find_all_similar_pairs(
            memories, min_score=self.similarity_threshold
        )
        
        memory_map = {m.id: m for m in memories}
        
        for pair in similar_pairs:
            m1 = memory_map.get(pair.memory1_id)
            m2 = memory_map.get(pair.memory2_id)
            
            if not m1 or not m2:
                continue
            
            # Check for various conflict types
            pair_conflicts = self._check_pair_conflicts(m1, m2, pair.score)
            conflicts.extend(pair_conflicts)
        
        # Deduplicate conflicts (same pair might have multiple types)
        return self._dedupe_conflicts(conflicts)
    
    def _check_pair_conflicts(
        self,
        m1: MemoryData,
        m2: MemoryData,
        similarity: float,
    ) -> list[MemoryConflict]:
        """Check for conflicts between a pair of memories."""
        conflicts = []
        
        # Value conflict check
        value_conflict = self._check_value_conflict(m1, m2, similarity)
        if value_conflict:
            conflicts.append(value_conflict)
        
        # Temporal conflict check
        temporal_conflict = self._check_temporal_conflict(m1, m2)
        if temporal_conflict:
            conflicts.append(temporal_conflict)
        
        # Confidence gap check
        confidence_conflict = self._check_confidence_gap(m1, m2, similarity)
        if confidence_conflict:
            conflicts.append(confidence_conflict)
        
        # Semantic contradiction check
        contradiction = self._check_semantic_contradiction(m1, m2)
        if contradiction:
            conflicts.append(contradiction)
        
        # Source conflict check
        source_conflict = self._check_source_conflict(m1, m2, similarity)
        if source_conflict:
            conflicts.append(source_conflict)
        
        return conflicts
    
    def _check_value_conflict(
        self,
        m1: MemoryData,
        m2: MemoryData,
        similarity: float,
    ) -> Optional[MemoryConflict]:
        """Check if memories have conflicting values for same subject."""
        # Look for key-value patterns like "X is Y" or "X = Y"
        patterns = [
            r"(\w+(?:\s+\w+)?)\s+(?:is|are|was|were|=|:)\s+(.+)",
            r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:equals?|is\s+set\s+to)\s+(.+)",
        ]
        
        for pattern in patterns:
            match1 = re.search(pattern, m1.content, re.IGNORECASE)
            match2 = re.search(pattern, m2.content, re.IGNORECASE)
            
            if match1 and match2:
                key1, value1 = match1.group(1).lower(), match1.group(2).strip()
                key2, value2 = match2.group(1).lower(), match2.group(2).strip()
                
                # Same key, different values
                if key1 == key2 and value1.lower() != value2.lower():
                    severity = self._assess_value_conflict_severity(
                        value1, value2, m1, m2
                    )
                    
                    return MemoryConflict(
                        conflict_id=str(uuid.uuid4()),
                        memory1_id=m1.id,
                        memory2_id=m2.id,
                        conflict_type=ConflictType.VALUE_CONFLICT,
                        severity=severity,
                        description=(
                            f"Conflicting values for '{key1}': "
                            f"'{value1}' vs '{value2}'"
                        ),
                    )
        
        return None
    
    def _assess_value_conflict_severity(
        self,
        value1: str,
        value2: str,
        m1: MemoryData,
        m2: MemoryData,
    ) -> ConflictSeverity:
        """Assess severity of a value conflict."""
        # Check confidence levels
        conf1 = self._get_confidence(m1)
        conf2 = self._get_confidence(m2)
        
        # Both high confidence = critical
        if conf1 > 0.8 and conf2 > 0.8:
            return ConflictSeverity.CRITICAL
        
        # One high, one low = medium
        if (conf1 > 0.8 or conf2 > 0.8) and (conf1 < 0.5 or conf2 < 0.5):
            return ConflictSeverity.MEDIUM
        
        # Check if values are similar (might be formatting difference)
        if self._values_similar(value1, value2):
            return ConflictSeverity.LOW
        
        return ConflictSeverity.HIGH
    
    def _values_similar(self, v1: str, v2: str) -> bool:
        """Check if two values are similar but not identical."""
        # Normalize and compare
        n1 = re.sub(r'\s+', ' ', v1.lower().strip())
        n2 = re.sub(r'\s+', ' ', v2.lower().strip())
        
        # One contains the other
        if n1 in n2 or n2 in n1:
            return True
        
        # Simple fuzzy match
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, n1, n2).ratio()
        return ratio > 0.8
    
    def _check_temporal_conflict(
        self,
        m1: MemoryData,
        m2: MemoryData,
    ) -> Optional[MemoryConflict]:
        """Check for conflicting temporal references."""
        dates1 = self._extract_dates(m1.content)
        dates2 = self._extract_dates(m2.content)
        
        if not dates1 or not dates2:
            return None
        
        # Check for conflicting date references for same events
        # This is a simplified check - a full implementation would need NLP
        
        # Look for overlapping context
        context_overlap = self._get_context_overlap(m1.content, m2.content)
        
        if context_overlap and dates1 != dates2:
            # Check if dates are significantly different
            if self._dates_conflict(dates1, dates2):
                return MemoryConflict(
                    conflict_id=str(uuid.uuid4()),
                    memory1_id=m1.id,
                    memory2_id=m2.id,
                    conflict_type=ConflictType.TEMPORAL_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    description=(
                        f"Conflicting dates for similar context: "
                        f"{dates1} vs {dates2}"
                    ),
                )
        
        return None
    
    def _extract_dates(self, text: str) -> list[str]:
        """Extract date strings from text."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        return dates
    
    def _get_context_overlap(self, text1: str, text2: str) -> bool:
        """Check if texts share significant context (non-date words)."""
        # Remove dates and compare remaining words
        for pattern in self.date_patterns:
            text1 = re.sub(pattern, '', text1)
            text2 = re.sub(pattern, '', text2)
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at'}
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap > 0.5
    
    def _dates_conflict(self, dates1: list[str], dates2: list[str]) -> bool:
        """Check if date lists represent conflicting information."""
        # Simple check: different dates
        return set(dates1) != set(dates2)
    
    def _check_confidence_gap(
        self,
        m1: MemoryData,
        m2: MemoryData,
        similarity: float,
    ) -> Optional[MemoryConflict]:
        """Check for large confidence gap between similar memories."""
        conf1 = self._get_confidence(m1)
        conf2 = self._get_confidence(m2)
        
        gap = abs(conf1 - conf2)
        
        if gap >= self.confidence_gap_threshold and similarity > 0.9:
            return MemoryConflict(
                conflict_id=str(uuid.uuid4()),
                memory1_id=m1.id,
                memory2_id=m2.id,
                conflict_type=ConflictType.CONFIDENCE_GAP,
                severity=ConflictSeverity.LOW,
                description=(
                    f"Large confidence gap ({gap:.2f}) between "
                    f"similar memories ({similarity:.2f} similarity)"
                ),
            )
        
        return None
    
    def _check_semantic_contradiction(
        self,
        m1: MemoryData,
        m2: MemoryData,
    ) -> Optional[MemoryConflict]:
        """Check for semantic contradictions between memories."""
        # Use custom detector if provided
        if self.contradiction_detector:
            score = self.contradiction_detector(m1.content, m2.content)
            if score > 0.7:
                return MemoryConflict(
                    conflict_id=str(uuid.uuid4()),
                    memory1_id=m1.id,
                    memory2_id=m2.id,
                    conflict_type=ConflictType.SEMANTIC_CONTRADICTION,
                    severity=ConflictSeverity.HIGH,
                    description=f"Semantic contradiction detected (score: {score:.2f})",
                )
        
        # Simple pattern-based contradiction detection
        contradiction = self._simple_contradiction_check(m1.content, m2.content)
        if contradiction:
            return MemoryConflict(
                conflict_id=str(uuid.uuid4()),
                memory1_id=m1.id,
                memory2_id=m2.id,
                conflict_type=ConflictType.SEMANTIC_CONTRADICTION,
                severity=ConflictSeverity.MEDIUM,
                description=contradiction,
            )
        
        return None
    
    def _simple_contradiction_check(
        self,
        text1: str,
        text2: str,
    ) -> Optional[str]:
        """Simple pattern-based contradiction detection."""
        t1_lower = text1.lower()
        t2_lower = text2.lower()
        
        for neg_pattern, pos_pattern in self.negation_patterns:
            # Check if one text has negation and other has positive
            has_neg1 = re.search(neg_pattern, t1_lower) is not None
            has_pos1 = re.search(pos_pattern, t1_lower) is not None
            has_neg2 = re.search(neg_pattern, t2_lower) is not None
            has_pos2 = re.search(pos_pattern, t2_lower) is not None
            
            if (has_neg1 and has_pos2 and not has_neg2) or \
               (has_neg2 and has_pos1 and not has_neg1):
                return f"Potential contradiction: negation pattern '{neg_pattern}' vs '{pos_pattern}'"
        
        return None
    
    def _check_source_conflict(
        self,
        m1: MemoryData,
        m2: MemoryData,
        similarity: float,
    ) -> Optional[MemoryConflict]:
        """Check for source conflicts."""
        if not m1.metadata or not m2.metadata:
            return None
        
        source1 = m1.metadata.get("source", "")
        source2 = m2.metadata.get("source", "")
        
        if not source1 or not source2:
            return None
        
        # Same content, different sources with different trust levels
        if source1 != source2 and similarity > 0.95:
            trust1 = m1.metadata.get("source_trust", 1.0)
            trust2 = m2.metadata.get("source_trust", 1.0)
            
            if abs(trust1 - trust2) > 0.3:
                return MemoryConflict(
                    conflict_id=str(uuid.uuid4()),
                    memory1_id=m1.id,
                    memory2_id=m2.id,
                    conflict_type=ConflictType.SOURCE_CONFLICT,
                    severity=ConflictSeverity.LOW,
                    description=(
                        f"Same content from different sources: "
                        f"'{source1}' (trust: {trust1}) vs "
                        f"'{source2}' (trust: {trust2})"
                    ),
                )
        
        return None
    
    def _get_confidence(self, memory: MemoryData) -> float:
        """Get confidence score from memory metadata."""
        if memory.metadata and "confidence" in memory.metadata:
            return memory.metadata["confidence"]
        return 1.0
    
    def _dedupe_conflicts(
        self,
        conflicts: list[MemoryConflict],
    ) -> list[MemoryConflict]:
        """Remove duplicate conflicts."""
        seen = set()
        unique = []
        
        for conflict in conflicts:
            key = (
                tuple(sorted([conflict.memory1_id, conflict.memory2_id])),
                conflict.conflict_type,
            )
            if key not in seen:
                seen.add(key)
                unique.append(conflict)
        
        return unique
    
    def detect_conflicts_for_memory(
        self,
        target: MemoryData,
        existing: list[MemoryData],
    ) -> list[MemoryConflict]:
        """
        Check if a new memory conflicts with existing ones.
        
        Args:
            target: The new memory to check
            existing: List of existing memories
            
        Returns:
            List of conflicts found
        """
        conflicts = []
        
        # Find similar memories
        similar = self.similarity_detector.find_similar(
            target, existing, min_score=self.similarity_threshold
        )
        
        memory_map = {m.id: m for m in existing}
        
        for score in similar:
            other = memory_map.get(score.memory2_id)
            if other:
                pair_conflicts = self._check_pair_conflicts(
                    target, other, score.score
                )
                conflicts.extend(pair_conflicts)
        
        return self._dedupe_conflicts(conflicts)
    
    def get_conflict_summary(
        self,
        conflicts: list[MemoryConflict],
    ) -> dict[str, Any]:
        """Get summary statistics for conflicts."""
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        
        for conflict in conflicts:
            type_key = conflict.conflict_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            sev_key = conflict.severity.value
            by_severity[sev_key] = by_severity.get(sev_key, 0) + 1
        
        return {
            "total_conflicts": len(conflicts),
            "by_type": by_type,
            "by_severity": by_severity,
            "unresolved": sum(1 for c in conflicts if not c.resolved),
            "critical_count": by_severity.get("critical", 0),
            "high_count": by_severity.get("high", 0),
        }
