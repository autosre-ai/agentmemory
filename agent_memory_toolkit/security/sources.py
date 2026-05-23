"""
Source Validation Module

Validates memory sources and assigns trust levels based on source type,
verification status, and historical reliability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import hashlib


class SourceTrust(Enum):
    """Trust levels for memory sources."""
    
    VERIFIED = "verified"  # Cryptographically verified source
    TRUSTED = "trusted"  # Known trusted source
    KNOWN = "known"  # Known but not fully trusted
    UNKNOWN = "unknown"  # Unknown source
    SUSPICIOUS = "suspicious"  # Previously flagged source
    BLOCKED = "blocked"  # Blocked source
    
    @property
    def trust_score(self) -> float:
        """Get numeric trust score for this level."""
        scores = {
            SourceTrust.VERIFIED: 1.0,
            SourceTrust.TRUSTED: 0.9,
            SourceTrust.KNOWN: 0.7,
            SourceTrust.UNKNOWN: 0.5,
            SourceTrust.SUSPICIOUS: 0.2,
            SourceTrust.BLOCKED: 0.0,
        }
        return scores[self]


@dataclass
class SourceProfile:
    """
    Profile of a memory source.
    
    Tracks historical reliability, verification status, and metadata.
    """
    
    source_id: str
    name: str
    trust_level: SourceTrust = SourceTrust.UNKNOWN
    verified: bool = False
    verified_at: Optional[datetime] = None
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    memory_count: int = 0
    rejection_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def reliability_score(self) -> float:
        """Calculate reliability based on history."""
        if self.memory_count == 0:
            return 0.5  # Neutral for new sources
        
        # Rejection ratio impacts reliability
        rejection_ratio = self.rejection_count / self.memory_count
        history_score = 1.0 - (rejection_ratio * 2)  # Heavy penalty for rejections
        
        # Combine with trust level
        combined = (self.trust_level.trust_score + max(0, history_score)) / 2
        return max(0.0, min(1.0, combined))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "trust_level": self.trust_level.value,
            "verified": self.verified,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "memory_count": self.memory_count,
            "rejection_count": self.rejection_count,
            "reliability_score": self.reliability_score,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceProfile":
        """Create from dictionary."""
        return cls(
            source_id=data["source_id"],
            name=data["name"],
            trust_level=SourceTrust(data["trust_level"]),
            verified=data.get("verified", False),
            verified_at=datetime.fromisoformat(data["verified_at"]) if data.get("verified_at") else None,
            first_seen=datetime.fromisoformat(data["first_seen"]) if data.get("first_seen") else datetime.utcnow(),
            last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else datetime.utcnow(),
            memory_count=data.get("memory_count", 0),
            rejection_count=data.get("rejection_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SourceValidationResult:
    """
    Result of source validation.
    
    Attributes:
        is_valid: Whether the source is valid for use
        source_profile: Profile of the source
        trust_score: Numeric trust score (0.0 to 1.0)
        reasons: Reasons for validation result
        recommendations: Actions to take
    """
    
    is_valid: bool = True
    source_profile: Optional[SourceProfile] = None
    trust_score: float = 0.5
    reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "source_profile": self.source_profile.to_dict() if self.source_profile else None,
            "trust_score": self.trust_score,
            "reasons": self.reasons,
            "recommendations": self.recommendations,
        }


class SourceValidator:
    """
    Validates memory sources and manages source trust.
    
    Tracks source profiles, verifies sources, and provides trust
    recommendations based on historical behavior.
    
    Example:
        validator = SourceValidator()
        validator.register_source("user_123", "User Chat", SourceTrust.TRUSTED)
        
        result = validator.validate("user_123")
        if result.is_valid:
            print(f"Trust score: {result.trust_score}")
    """
    
    def __init__(
        self,
        default_trust: SourceTrust = SourceTrust.UNKNOWN,
        block_unknown: bool = False,
        min_trust_threshold: float = 0.3,
    ):
        """
        Initialize source validator.
        
        Args:
            default_trust: Default trust level for new sources
            block_unknown: Block unknown sources entirely
            min_trust_threshold: Minimum trust score to validate
        """
        self._profiles: dict[str, SourceProfile] = {}
        self.default_trust = default_trust
        self.block_unknown = block_unknown
        self.min_trust_threshold = min_trust_threshold
        
        # Built-in trusted sources
        self._builtin_trusted: set[str] = set()
        self._builtin_blocked: set[str] = set()
    
    def register_source(
        self,
        source_id: str,
        name: str,
        trust_level: SourceTrust = SourceTrust.UNKNOWN,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SourceProfile:
        """
        Register a new source or update existing.
        
        Args:
            source_id: Unique source identifier
            name: Human-readable source name
            trust_level: Initial trust level
            metadata: Additional source metadata
            
        Returns:
            The source profile
        """
        if source_id in self._profiles:
            profile = self._profiles[source_id]
            profile.trust_level = trust_level
            profile.last_seen = datetime.utcnow()
            if metadata:
                profile.metadata.update(metadata)
        else:
            profile = SourceProfile(
                source_id=source_id,
                name=name,
                trust_level=trust_level,
                metadata=metadata or {},
            )
            self._profiles[source_id] = profile
        
        return profile
    
    def get_profile(self, source_id: str) -> Optional[SourceProfile]:
        """Get source profile by ID."""
        return self._profiles.get(source_id)
    
    def validate(
        self,
        source_id: Optional[str],
        create_if_missing: bool = True,
    ) -> SourceValidationResult:
        """
        Validate a source.
        
        Args:
            source_id: Source identifier to validate
            create_if_missing: Create profile if source doesn't exist
            
        Returns:
            Validation result with trust information
        """
        reasons: list[str] = []
        recommendations: list[str] = []
        
        # Handle None/empty source
        if not source_id:
            if self.block_unknown:
                return SourceValidationResult(
                    is_valid=False,
                    trust_score=0.0,
                    reasons=["No source provided"],
                    recommendations=["Require source identification"],
                )
            return SourceValidationResult(
                is_valid=True,
                trust_score=0.5,
                reasons=["No source - using default trust"],
                recommendations=["Consider requiring source identification"],
            )
        
        # Check built-in lists first
        if source_id in self._builtin_blocked:
            return SourceValidationResult(
                is_valid=False,
                trust_score=0.0,
                reasons=["Source is on block list"],
                recommendations=["Contact administrator to remove from block list"],
            )
        
        if source_id in self._builtin_trusted:
            profile = self._profiles.get(source_id)
            if not profile:
                profile = self.register_source(
                    source_id, f"Builtin:{source_id}", SourceTrust.TRUSTED
                )
            return SourceValidationResult(
                is_valid=True,
                source_profile=profile,
                trust_score=1.0,
                reasons=["Source is on trusted list"],
            )
        
        # Get or create profile
        profile = self._profiles.get(source_id)
        
        if not profile:
            if not create_if_missing:
                if self.block_unknown:
                    return SourceValidationResult(
                        is_valid=False,
                        trust_score=0.0,
                        reasons=["Unknown source and block_unknown is enabled"],
                        recommendations=["Register source before use"],
                    )
                return SourceValidationResult(
                    is_valid=True,
                    trust_score=self.default_trust.trust_score,
                    reasons=["New source - using default trust"],
                    recommendations=["Consider registering source for tracking"],
                )
            
            # Create new profile
            profile = self.register_source(
                source_id,
                f"Auto:{source_id[:20]}",
                self.default_trust,
            )
            reasons.append("New source profile created")
        
        # Update last seen
        profile.last_seen = datetime.utcnow()
        
        # Calculate trust score
        trust_score = profile.reliability_score
        
        # Check blocked trust level
        if profile.trust_level == SourceTrust.BLOCKED:
            return SourceValidationResult(
                is_valid=False,
                source_profile=profile,
                trust_score=0.0,
                reasons=["Source is blocked"],
                recommendations=["Contact administrator to unblock"],
            )
        
        # Check suspicious
        if profile.trust_level == SourceTrust.SUSPICIOUS:
            reasons.append("Source is flagged as suspicious")
            recommendations.append("Review source history")
            trust_score *= 0.5
        
        # Check minimum threshold
        if trust_score < self.min_trust_threshold:
            return SourceValidationResult(
                is_valid=False,
                source_profile=profile,
                trust_score=trust_score,
                reasons=reasons + [f"Trust score {trust_score:.2f} below threshold {self.min_trust_threshold}"],
                recommendations=recommendations + ["Build trust through successful validations"],
            )
        
        # Add trust level reason
        reasons.append(f"Trust level: {profile.trust_level.value}")
        
        # Add reliability info
        if profile.memory_count > 0:
            reasons.append(f"History: {profile.memory_count} memories, {profile.rejection_count} rejections")
        
        return SourceValidationResult(
            is_valid=True,
            source_profile=profile,
            trust_score=trust_score,
            reasons=reasons,
            recommendations=recommendations,
        )
    
    def record_memory(self, source_id: str, rejected: bool = False) -> None:
        """
        Record a memory from a source.
        
        Args:
            source_id: Source identifier
            rejected: Whether the memory was rejected
        """
        profile = self._profiles.get(source_id)
        if profile:
            profile.memory_count += 1
            if rejected:
                profile.rejection_count += 1
            profile.last_seen = datetime.utcnow()
            
            # Auto-downgrade if too many rejections
            if profile.memory_count >= 10:
                rejection_ratio = profile.rejection_count / profile.memory_count
                if rejection_ratio > 0.5 and profile.trust_level != SourceTrust.SUSPICIOUS:
                    profile.trust_level = SourceTrust.SUSPICIOUS
    
    def verify_source(self, source_id: str, verifier: str = "system") -> bool:
        """
        Mark a source as verified.
        
        Args:
            source_id: Source to verify
            verifier: Who/what verified the source
            
        Returns:
            True if verification succeeded
        """
        profile = self._profiles.get(source_id)
        if not profile:
            return False
        
        profile.verified = True
        profile.verified_at = datetime.utcnow()
        profile.trust_level = SourceTrust.VERIFIED
        profile.metadata["verified_by"] = verifier
        
        return True
    
    def block_source(self, source_id: str, reason: str = "") -> None:
        """Block a source."""
        self._builtin_blocked.add(source_id)
        profile = self._profiles.get(source_id)
        if profile:
            profile.trust_level = SourceTrust.BLOCKED
            profile.metadata["blocked_reason"] = reason
            profile.metadata["blocked_at"] = datetime.utcnow().isoformat()
    
    def trust_source(self, source_id: str) -> None:
        """Add source to trusted list."""
        self._builtin_trusted.add(source_id)
        profile = self._profiles.get(source_id)
        if profile:
            profile.trust_level = SourceTrust.TRUSTED
    
    def get_all_profiles(self) -> list[SourceProfile]:
        """Get all source profiles."""
        return list(self._profiles.values())
    
    def get_suspicious_sources(self) -> list[SourceProfile]:
        """Get all suspicious sources."""
        return [
            p for p in self._profiles.values()
            if p.trust_level in (SourceTrust.SUSPICIOUS, SourceTrust.BLOCKED)
        ]
    
    def export_profiles(self) -> list[dict[str, Any]]:
        """Export all profiles as dictionaries."""
        return [p.to_dict() for p in self._profiles.values()]
    
    def import_profiles(self, profiles: list[dict[str, Any]]) -> int:
        """
        Import profiles from dictionaries.
        
        Returns number of profiles imported.
        """
        count = 0
        for data in profiles:
            try:
                profile = SourceProfile.from_dict(data)
                self._profiles[profile.source_id] = profile
                count += 1
            except (KeyError, ValueError):
                continue
        return count
