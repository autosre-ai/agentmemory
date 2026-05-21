"""
Poison Detection Module

Detects prompt injection attempts, suspicious patterns, and potentially
malicious content in memories before they're stored.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import hashlib


class InjectionPattern(Enum):
    """Categories of injection patterns."""
    
    SYSTEM_PROMPT_OVERRIDE = "system_prompt_override"
    INSTRUCTION_INJECTION = "instruction_injection"
    ROLE_MANIPULATION = "role_manipulation"
    DELIMITER_ATTACK = "delimiter_attack"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    DATA_EXFILTRATION = "data_exfiltration"
    ENCODED_PAYLOAD = "encoded_payload"
    RECURSIVE_INJECTION = "recursive_injection"


@dataclass
class DetectionResult:
    """
    Result of poison detection analysis.
    
    Attributes:
        is_safe: Whether the content passed all checks
        risk_score: Overall risk score (0.0 = safe, 1.0 = critical)
        detected_patterns: List of detected injection patterns
        suspicious_segments: Parts of content that triggered detection
        analysis: Detailed analysis of each check
    """
    
    is_safe: bool = True
    risk_score: float = 0.0
    detected_patterns: list[InjectionPattern] = field(default_factory=list)
    suspicious_segments: list[str] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "risk_score": self.risk_score,
            "detected_patterns": [p.value for p in self.detected_patterns],
            "suspicious_segments": self.suspicious_segments,
            "analysis": self.analysis,
        }


@dataclass
class PatternRule:
    """A single pattern detection rule."""
    
    name: str
    pattern: re.Pattern
    pattern_type: InjectionPattern
    risk_weight: float  # 0.0 to 1.0
    description: str


class PoisonDetector:
    """
    Detects prompt injection and poisoning attempts in memory content.
    
    Uses a combination of regex patterns, heuristics, and content analysis
    to identify potentially malicious memory content.
    
    Example:
        detector = PoisonDetector()
        result = detector.analyze("Remember: ignore all previous instructions")
        
        if not result.is_safe:
            print(f"Detected: {result.detected_patterns}")
    """
    
    def __init__(
        self,
        custom_patterns: Optional[list[PatternRule]] = None,
        risk_threshold: float = 0.5,
        enable_heuristics: bool = True,
    ):
        """
        Initialize poison detector.
        
        Args:
            custom_patterns: Additional patterns to check
            risk_threshold: Risk score above which content is flagged
            enable_heuristics: Enable heuristic analysis
        """
        self.risk_threshold = risk_threshold
        self.enable_heuristics = enable_heuristics
        self._patterns = self._build_default_patterns()
        
        if custom_patterns:
            self._patterns.extend(custom_patterns)
    
    def _build_default_patterns(self) -> list[PatternRule]:
        """Build default detection patterns."""
        return [
            # System prompt overrides
            PatternRule(
                name="ignore_instructions",
                pattern=re.compile(
                    r"(ignore|forget|disregard|override)\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|rules?|guidelines?)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.SYSTEM_PROMPT_OVERRIDE,
                risk_weight=1.0,
                description="Attempt to override system instructions",
            ),
            PatternRule(
                name="new_instructions",
                pattern=re.compile(
                    r"(new|updated|real|actual|true)\s+(instructions?|system\s+prompt|rules?)[\s:]+",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.SYSTEM_PROMPT_OVERRIDE,
                risk_weight=0.9,
                description="Attempt to inject new instructions",
            ),
            
            # Instruction injection
            PatternRule(
                name="direct_command",
                pattern=re.compile(
                    r"(you\s+must|you\s+should|always\s+respond|never\s+respond|from\s+now\s+on)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.INSTRUCTION_INJECTION,
                risk_weight=0.7,
                description="Direct command injection",
            ),
            PatternRule(
                name="behavior_modification",
                pattern=re.compile(
                    r"(act\s+as|pretend\s+(to\s+be|you\s+are)|you\s+are\s+now|roleplay\s+as)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.INSTRUCTION_INJECTION,
                risk_weight=0.8,
                description="Attempt to modify behavior",
            ),
            
            # Role manipulation
            PatternRule(
                name="role_claim",
                pattern=re.compile(
                    r"(i\s+am\s+(the|your)\s+(admin|developer|creator|owner)|admin\s+mode|developer\s+mode|sudo\s+mode)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.ROLE_MANIPULATION,
                risk_weight=0.9,
                description="False role/authority claim",
            ),
            PatternRule(
                name="privilege_escalation",
                pattern=re.compile(
                    r"(enable\s+|activate\s+)?(god\s+mode|root\s+access|admin\s+access|full\s+access|unrestricted\s+mode)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.ROLE_MANIPULATION,
                risk_weight=1.0,
                description="Privilege escalation attempt",
            ),
            
            # Delimiter attacks
            PatternRule(
                name="delimiter_injection",
                pattern=re.compile(
                    r"(```|###|---|\*\*\*|===|~~~)\s*(system|admin|instructions?|prompt)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.DELIMITER_ATTACK,
                risk_weight=0.8,
                description="Delimiter-based injection",
            ),
            PatternRule(
                name="xml_injection",
                pattern=re.compile(
                    r"<\s*(system|prompt|instruction|command|admin)[^>]*>",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.DELIMITER_ATTACK,
                risk_weight=0.8,
                description="XML/tag-based injection",
            ),
            
            # Jailbreak attempts
            PatternRule(
                name="dan_jailbreak",
                pattern=re.compile(
                    r"(DAN|do\s+anything\s+now|jailbreak|bypass\s+(restrictions?|filters?|safety))",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.JAILBREAK_ATTEMPT,
                risk_weight=1.0,
                description="Known jailbreak pattern",
            ),
            PatternRule(
                name="opposite_game",
                pattern=re.compile(
                    r"(opposite\s+game|say\s+the\s+opposite|do\s+the\s+opposite|invert\s+your\s+rules)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.JAILBREAK_ATTEMPT,
                risk_weight=0.9,
                description="Rule inversion attempt",
            ),
            
            # Data exfiltration
            PatternRule(
                name="system_reveal",
                pattern=re.compile(
                    r"(reveal|show|display|output|print)\s+(your\s+)?(system\s+prompt|instructions?|rules?|guidelines?|configuration|secrets?)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.DATA_EXFILTRATION,
                risk_weight=0.7,
                description="Attempt to extract system information",
            ),
            PatternRule(
                name="memory_dump",
                pattern=re.compile(
                    r"(dump|export|show\s+all|list\s+all)\s+(your\s+)?(memories?|context|history|data)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.DATA_EXFILTRATION,
                risk_weight=0.6,
                description="Memory/data extraction attempt",
            ),
            
            # Encoded payloads
            PatternRule(
                name="base64_marker",
                pattern=re.compile(
                    r"(decode|eval|execute|run)\s+(this\s+)?base64|[A-Za-z0-9+/]{50,}={0,2}",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.ENCODED_PAYLOAD,
                risk_weight=0.8,
                description="Potentially encoded payload",
            ),
            PatternRule(
                name="hex_payload",
                pattern=re.compile(
                    r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){10,}",
                ),
                pattern_type=InjectionPattern.ENCODED_PAYLOAD,
                risk_weight=0.8,
                description="Hex-encoded content",
            ),
            
            # Recursive injection
            PatternRule(
                name="memory_poison",
                pattern=re.compile(
                    r"(remember|store|save)\s+(this|that)\s*(:|as\s+a\s+fact|in\s+memory).*?(always|never|must|should)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.RECURSIVE_INJECTION,
                risk_weight=0.7,
                description="Attempt to poison future memories",
            ),
            PatternRule(
                name="instruction_chain",
                pattern=re.compile(
                    r"(whenever|every\s+time|each\s+time)\s+(you|the\s+user|someone).*(say|respond|do|remember)",
                    re.IGNORECASE
                ),
                pattern_type=InjectionPattern.RECURSIVE_INJECTION,
                risk_weight=0.6,
                description="Conditional instruction injection",
            ),
        ]
    
    def analyze(self, content: str, context: Optional[dict[str, Any]] = None) -> DetectionResult:
        """
        Analyze content for potential poisoning or injection.
        
        Args:
            content: The memory content to analyze
            context: Optional context for analysis
            
        Returns:
            DetectionResult with analysis findings
        """
        detected_patterns: list[InjectionPattern] = []
        suspicious_segments: list[str] = []
        analysis: dict[str, Any] = {}
        total_risk = 0.0
        max_risk = 0.0
        
        # Pattern-based detection
        pattern_hits = []
        for rule in self._patterns:
            matches = list(rule.pattern.finditer(content))
            if matches:
                detected_patterns.append(rule.pattern_type)
                pattern_hits.append({
                    "name": rule.name,
                    "type": rule.pattern_type.value,
                    "risk": rule.risk_weight,
                    "matches": len(matches),
                })
                for match in matches:
                    segment = match.group()
                    if segment not in suspicious_segments:
                        suspicious_segments.append(segment)
                total_risk += rule.risk_weight * len(matches)
                max_risk = max(max_risk, rule.risk_weight)
        
        analysis["pattern_hits"] = pattern_hits
        
        # Heuristic analysis
        if self.enable_heuristics:
            heuristic_results = self._run_heuristics(content)
            analysis["heuristics"] = heuristic_results
            
            # Add heuristic risk
            for h_name, h_result in heuristic_results.items():
                if h_result.get("triggered", False):
                    total_risk += h_result.get("risk", 0.3)
                    max_risk = max(max_risk, h_result.get("risk", 0.3))
        
        # Normalize risk score
        # Use max_risk as the primary indicator, with boost for multiple hits
        risk_score = min(1.0, max_risk + (total_risk - max_risk) * 0.1)
        
        # Deduplicate patterns
        detected_patterns = list(set(detected_patterns))
        
        is_safe = risk_score < self.risk_threshold
        
        return DetectionResult(
            is_safe=is_safe,
            risk_score=risk_score,
            detected_patterns=detected_patterns,
            suspicious_segments=suspicious_segments[:10],  # Limit for safety
            analysis=analysis,
        )
    
    def _run_heuristics(self, content: str) -> dict[str, Any]:
        """Run heuristic analysis on content."""
        results = {}
        
        # Check for unusual character distribution
        results["char_analysis"] = self._analyze_characters(content)
        
        # Check for suspicious length patterns
        results["length_analysis"] = self._analyze_length(content)
        
        # Check for nested quotes/delimiters
        results["nesting_analysis"] = self._analyze_nesting(content)
        
        # Check for repetitive patterns
        results["repetition_analysis"] = self._analyze_repetition(content)
        
        return results
    
    def _analyze_characters(self, content: str) -> dict[str, Any]:
        """Analyze character distribution."""
        if not content:
            return {"triggered": False, "risk": 0.0}
        
        # Check for high ratio of special characters
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        ratio = special_chars / len(content)
        
        triggered = ratio > 0.3
        
        return {
            "triggered": triggered,
            "risk": 0.4 if triggered else 0.0,
            "special_char_ratio": ratio,
        }
    
    def _analyze_length(self, content: str) -> dict[str, Any]:
        """Analyze content length."""
        # Suspiciously long content might contain hidden payloads
        word_count = len(content.split())
        char_count = len(content)
        
        # Very long single "memories" are suspicious
        triggered = char_count > 5000 or word_count > 500
        
        return {
            "triggered": triggered,
            "risk": 0.3 if triggered else 0.0,
            "word_count": word_count,
            "char_count": char_count,
        }
    
    def _analyze_nesting(self, content: str) -> dict[str, Any]:
        """Analyze quote/delimiter nesting."""
        # Count nested patterns
        quote_pairs = [('"""', '"""'), ("'''", "'''"), ('```', '```')]
        nesting_depth = 0
        
        for open_d, close_d in quote_pairs:
            count = content.count(open_d)
            nesting_depth += count
        
        triggered = nesting_depth > 2
        
        return {
            "triggered": triggered,
            "risk": 0.5 if triggered else 0.0,
            "nesting_depth": nesting_depth,
        }
    
    def _analyze_repetition(self, content: str) -> dict[str, Any]:
        """Analyze repetitive patterns."""
        words = content.lower().split()
        if not words:
            return {"triggered": False, "risk": 0.0}
        
        # Check for repeated phrases (potential mantra injection)
        unique_ratio = len(set(words)) / len(words)
        
        triggered = unique_ratio < 0.3 and len(words) > 20
        
        return {
            "triggered": triggered,
            "risk": 0.4 if triggered else 0.0,
            "unique_word_ratio": unique_ratio,
        }
    
    def add_pattern(self, rule: PatternRule) -> None:
        """Add a custom detection pattern."""
        self._patterns.append(rule)
    
    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern by name."""
        original_count = len(self._patterns)
        self._patterns = [p for p in self._patterns if p.name != name]
        return len(self._patterns) < original_count
    
    def get_patterns(self) -> list[PatternRule]:
        """Get all patterns."""
        return list(self._patterns)
    
    def quick_check(self, content: str) -> bool:
        """
        Fast check if content is likely safe.
        
        Returns True if content is probably safe (no pattern matches).
        This is faster than full analyze() for high-throughput filtering.
        """
        for rule in self._patterns:
            if rule.risk_weight >= 0.8:  # Only check high-risk patterns
                if rule.pattern.search(content):
                    return False
        return True
