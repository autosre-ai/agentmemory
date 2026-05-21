"""
Confidence Scoring and Uncertainty Detection

Provides confidence scoring for memories and detects when the system
should say "I don't know" rather than use uncertain information.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class UncertaintyType(Enum):
    """Types of uncertainty detected in content."""
    
    EXPLICIT_UNCERTAINTY = "explicit_uncertainty"  # "I think", "maybe"
    HEDGING_LANGUAGE = "hedging_language"  # "possibly", "might"
    CONTRADICTION = "contradiction"  # Conflicting information
    TEMPORAL_UNCERTAINTY = "temporal_uncertainty"  # Outdated info
    SOURCE_UNCERTAINTY = "source_uncertainty"  # Unknown source
    VAGUE_QUANTIFICATION = "vague_quantification"  # "some", "many"
    CONDITIONAL = "conditional"  # "if", "depends"
    SPECULATIVE = "speculative"  # Guesses, assumptions


@dataclass
class ConfidenceResult:
    """
    Result of confidence scoring and uncertainty detection.
    
    Attributes:
        confidence: Overall confidence score (0.0 to 1.0)
        should_defer: Whether the system should say "I don't know"
        uncertainties: List of detected uncertainty types
        uncertainty_indicators: Specific phrases that triggered detection
        adjustments: Breakdown of confidence adjustments
        recommendation: Human-readable recommendation
    """
    
    confidence: float = 1.0
    should_defer: bool = False
    uncertainties: list[UncertaintyType] = field(default_factory=list)
    uncertainty_indicators: list[str] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confidence": self.confidence,
            "should_defer": self.should_defer,
            "uncertainties": [u.value for u in self.uncertainties],
            "uncertainty_indicators": self.uncertainty_indicators,
            "adjustments": self.adjustments,
            "recommendation": self.recommendation,
        }


class UncertaintyDetector:
    """
    Detects uncertainty and "I don't know" cases in content.
    
    Identifies when information is too uncertain to be trusted,
    allowing the system to defer rather than provide unreliable answers.
    
    Example:
        detector = UncertaintyDetector()
        result = detector.detect("I think his name might be John")
        
        if result.should_defer:
            print("Better to say 'I don't know'")
    """
    
    def __init__(
        self,
        defer_threshold: float = 0.5,
        strict_mode: bool = False,
    ):
        """
        Initialize uncertainty detector.
        
        Args:
            defer_threshold: Confidence below which to recommend deferring
            strict_mode: More aggressive uncertainty detection
        """
        self.defer_threshold = defer_threshold
        self.strict_mode = strict_mode
        self._build_patterns()
    
    def _build_patterns(self) -> None:
        """Build uncertainty detection patterns."""
        self._explicit_uncertainty = re.compile(
            r"\b(i\s+think|i\s+believe|i\s+assume|i\s+guess|"
            r"not\s+sure|unsure|uncertain|don'?t\s+know|"
            r"can'?t\s+remember|don'?t\s+recall|forgot|unclear)\b",
            re.IGNORECASE
        )
        
        self._hedging = re.compile(
            r"\b(maybe|perhaps|possibly|probably|might|could\s+be|"
            r"seems?\s+like|appears?\s+to|looks?\s+like|"
            r"sort\s+of|kind\s+of|somewhat|roughly|approximately|"
            r"more\s+or\s+less|in\s+a\s+way|to\s+some\s+extent)\b",
            re.IGNORECASE
        )
        
        self._conditional = re.compile(
            r"\b(if|unless|depending|depends|when|provided\s+that|"
            r"assuming|in\s+case|should\s+be|would\s+be|could\s+be)\b",
            re.IGNORECASE
        )
        
        self._speculative = re.compile(
            r"\b(supposedly|allegedly|reportedly|rumor|heard\s+that|"
            r"someone\s+said|they\s+say|word\s+is|speculation|"
            r"guessing|estimate|ballpark|guesstimate)\b",
            re.IGNORECASE
        )
        
        self._vague_quantifiers = re.compile(
            r"\b(some|many|few|several|most|often|sometimes|"
            r"rarely|occasionally|frequently|various|certain|"
            r"a\s+lot|bunch\s+of|tons\s+of|handful)\b",
            re.IGNORECASE
        )
        
        self._contradiction_markers = re.compile(
            r"\b(but|however|although|yet|on\s+the\s+other\s+hand|"
            r"alternatively|conversely|in\s+contrast|then\s+again|"
            r"actually|wait|no|rather|instead)\b",
            re.IGNORECASE
        )
        
        self._temporal_uncertainty = re.compile(
            r"\b(used\s+to|was|were|before|previously|formerly|"
            r"back\s+then|at\s+the\s+time|outdated|old|"
            r"last\s+(year|month|week)|ago|recently\s+changed)\b",
            re.IGNORECASE
        )
        
        self._negation = re.compile(
            r"\b(no|not|never|none|nothing|nowhere|neither|"
            r"don'?t|doesn'?t|didn'?t|won'?t|wouldn'?t|"
            r"can'?t|couldn'?t|shouldn'?t|isn'?t|aren'?t|wasn'?t)\b",
            re.IGNORECASE
        )
    
    def detect(self, content: str) -> ConfidenceResult:
        """
        Detect uncertainty in content.
        
        Args:
            content: The content to analyze
            
        Returns:
            ConfidenceResult with uncertainty analysis
        """
        uncertainties: list[UncertaintyType] = []
        indicators: list[str] = []
        adjustments: dict[str, float] = {}
        base_confidence = 1.0
        
        # Check explicit uncertainty
        explicit_matches = list(self._explicit_uncertainty.finditer(content))
        if explicit_matches:
            uncertainties.append(UncertaintyType.EXPLICIT_UNCERTAINTY)
            indicators.extend([m.group() for m in explicit_matches[:3]])
            penalty = min(0.5, len(explicit_matches) * 0.2)
            adjustments["explicit_uncertainty"] = -penalty
            base_confidence -= penalty
        
        # Check hedging language
        hedging_matches = list(self._hedging.finditer(content))
        if hedging_matches:
            uncertainties.append(UncertaintyType.HEDGING_LANGUAGE)
            indicators.extend([m.group() for m in hedging_matches[:3]])
            penalty = min(0.3, len(hedging_matches) * 0.1)
            adjustments["hedging"] = -penalty
            base_confidence -= penalty
        
        # Check conditional statements
        conditional_matches = list(self._conditional.finditer(content))
        if conditional_matches:
            uncertainties.append(UncertaintyType.CONDITIONAL)
            indicators.extend([m.group() for m in conditional_matches[:3]])
            penalty = min(0.2, len(conditional_matches) * 0.05)
            adjustments["conditional"] = -penalty
            base_confidence -= penalty
        
        # Check speculative language
        speculative_matches = list(self._speculative.finditer(content))
        if speculative_matches:
            uncertainties.append(UncertaintyType.SPECULATIVE)
            indicators.extend([m.group() for m in speculative_matches[:3]])
            penalty = min(0.4, len(speculative_matches) * 0.15)
            adjustments["speculative"] = -penalty
            base_confidence -= penalty
        
        # Check vague quantifiers
        vague_matches = list(self._vague_quantifiers.finditer(content))
        if vague_matches:
            uncertainties.append(UncertaintyType.VAGUE_QUANTIFICATION)
            indicators.extend([m.group() for m in vague_matches[:3]])
            penalty = min(0.15, len(vague_matches) * 0.03)
            adjustments["vague_quantifiers"] = -penalty
            base_confidence -= penalty
        
        # Check contradiction markers
        contradiction_matches = list(self._contradiction_markers.finditer(content))
        if contradiction_matches and len(contradiction_matches) > 1:
            uncertainties.append(UncertaintyType.CONTRADICTION)
            indicators.extend([m.group() for m in contradiction_matches[:3]])
            penalty = min(0.25, len(contradiction_matches) * 0.08)
            adjustments["contradictions"] = -penalty
            base_confidence -= penalty
        
        # Check temporal uncertainty
        temporal_matches = list(self._temporal_uncertainty.finditer(content))
        if temporal_matches:
            uncertainties.append(UncertaintyType.TEMPORAL_UNCERTAINTY)
            indicators.extend([m.group() for m in temporal_matches[:3]])
            penalty = min(0.2, len(temporal_matches) * 0.05)
            adjustments["temporal"] = -penalty
            base_confidence -= penalty
        
        # Strict mode: penalize heavily for negations (less certain context)
        if self.strict_mode:
            negation_matches = list(self._negation.finditer(content))
            if negation_matches:
                penalty = min(0.1, len(negation_matches) * 0.02)
                adjustments["negation_strict"] = -penalty
                base_confidence -= penalty
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, base_confidence))
        
        # Determine if we should defer
        should_defer = (
            confidence < self.defer_threshold
            or UncertaintyType.EXPLICIT_UNCERTAINTY in uncertainties
            or (UncertaintyType.SPECULATIVE in uncertainties and confidence < 0.6)
        )
        
        # Generate recommendation
        if should_defer:
            recommendation = "Defer: Recommend saying 'I don't know' or qualifying the response"
        elif confidence < 0.7:
            recommendation = "Caution: Use with qualification (e.g., 'based on uncertain information...')"
        else:
            recommendation = "OK: Information appears reasonably certain"
        
        return ConfidenceResult(
            confidence=confidence,
            should_defer=should_defer,
            uncertainties=uncertainties,
            uncertainty_indicators=list(set(indicators))[:10],
            adjustments=adjustments,
            recommendation=recommendation,
        )


class ConfidenceScorer:
    """
    Scores confidence for memories based on multiple factors.
    
    Combines content analysis, source reliability, recency, and
    corroboration to produce a final confidence score.
    
    Example:
        scorer = ConfidenceScorer()
        result = scorer.score(
            memory_content="User prefers Python",
            source_confidence=0.9,
            recency_days=5,
            corroborating_count=2,
        )
        print(f"Final confidence: {result.confidence}")
    """
    
    def __init__(
        self,
        content_weight: float = 0.4,
        source_weight: float = 0.3,
        recency_weight: float = 0.15,
        corroboration_weight: float = 0.15,
    ):
        """
        Initialize confidence scorer.
        
        Args:
            content_weight: Weight for content-based confidence
            source_weight: Weight for source reliability
            recency_weight: Weight for recency
            corroboration_weight: Weight for corroboration
        """
        self.content_weight = content_weight
        self.source_weight = source_weight
        self.recency_weight = recency_weight
        self.corroboration_weight = corroboration_weight
        
        # Normalize weights
        total = content_weight + source_weight + recency_weight + corroboration_weight
        self.content_weight /= total
        self.source_weight /= total
        self.recency_weight /= total
        self.corroboration_weight /= total
        
        self._uncertainty_detector = UncertaintyDetector()
    
    def score(
        self,
        memory_content: str,
        source_confidence: float = 1.0,
        recency_days: Optional[int] = None,
        corroborating_count: int = 0,
        base_confidence: Optional[float] = None,
    ) -> ConfidenceResult:
        """
        Score confidence for a memory.
        
        Args:
            memory_content: The content of the memory
            source_confidence: Confidence in the source (0.0 to 1.0)
            recency_days: Days since the memory was created
            corroborating_count: Number of corroborating memories
            base_confidence: Override base confidence from content analysis
            
        Returns:
            ConfidenceResult with scoring details
        """
        adjustments: dict[str, float] = {}
        
        # 1. Content-based confidence (from uncertainty detection)
        if base_confidence is not None:
            content_confidence = base_confidence
            uncertainties = []
            uncertainty_indicators = []
        else:
            uncertainty_result = self._uncertainty_detector.detect(memory_content)
            content_confidence = uncertainty_result.confidence
            uncertainties = uncertainty_result.uncertainties
            uncertainty_indicators = uncertainty_result.uncertainty_indicators
            adjustments.update(uncertainty_result.adjustments)
        
        # 2. Source confidence
        source_score = max(0.0, min(1.0, source_confidence))
        
        # 3. Recency score
        if recency_days is not None:
            # Decay: 50% confidence after 365 days
            recency_score = 1.0 / (1.0 + (recency_days / 365.0))
            adjustments["recency_decay"] = recency_score - 1.0
        else:
            recency_score = 1.0
        
        # 4. Corroboration bonus
        if corroborating_count > 0:
            # Diminishing returns: max 20% bonus
            corroboration_score = min(1.2, 1.0 + (corroborating_count * 0.1))
            adjustments["corroboration_bonus"] = corroboration_score - 1.0
        else:
            corroboration_score = 1.0
        
        # Calculate weighted average
        weighted_score = (
            self.content_weight * content_confidence +
            self.source_weight * source_score +
            self.recency_weight * recency_score +
            self.corroboration_weight * min(1.0, corroboration_score)
        )
        
        # Apply corroboration bonus on top
        final_confidence = min(1.0, weighted_score * corroboration_score)
        
        # Determine if we should defer
        should_defer = (
            final_confidence < 0.4
            or UncertaintyType.EXPLICIT_UNCERTAINTY in uncertainties
        )
        
        # Generate recommendation
        if should_defer:
            recommendation = "Defer: Low confidence, recommend verification or deferral"
        elif final_confidence < 0.6:
            recommendation = "Caution: Moderate confidence, qualify usage"
        elif final_confidence < 0.8:
            recommendation = "OK: Good confidence, suitable for most uses"
        else:
            recommendation = "High: Very confident, reliable for critical uses"
        
        return ConfidenceResult(
            confidence=final_confidence,
            should_defer=should_defer,
            uncertainties=uncertainties,
            uncertainty_indicators=uncertainty_indicators,
            adjustments=adjustments,
            recommendation=recommendation,
        )
    
    def quick_confidence(self, memory_content: str) -> float:
        """
        Quick confidence score based on content only.
        
        Returns just the confidence value for fast filtering.
        """
        result = self._uncertainty_detector.detect(memory_content)
        return result.confidence
    
    def adjust_for_source(
        self,
        base_confidence: float,
        source_trust: float,
    ) -> float:
        """
        Adjust confidence based on source trust.
        
        Args:
            base_confidence: Current confidence score
            source_trust: Trust level of the source (0.0 to 1.0)
            
        Returns:
            Adjusted confidence score
        """
        # Source trust can only reduce confidence, not increase it
        # (except for very high trust sources)
        if source_trust >= 0.9:
            adjustment = 1.0 + (source_trust - 0.9) * 0.2
        else:
            adjustment = source_trust
        
        return min(1.0, base_confidence * adjustment)
