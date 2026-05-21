"""
Tests for Structured Memory Extraction module.
"""

import json
import pytest
from datetime import datetime, timedelta
from typing import Any

from agentmemory.extraction import (
    CognitiveDomain,
    Memory,
    ExtractionResult,
    MemoryExtractor,
    RuleBasedExtractor,
    LLMExtractor,
    MemoryDeduplicator,
    ConflictResolver,
    MemoryMerger,
)
from agentmemory.extraction.conflict_resolver import ConflictStrategy, Conflict
from agentmemory.extraction.llm_extractor import MockLLMClient


class TestMemory:
    """Tests for Memory dataclass."""
    
    def test_memory_creation(self):
        """Test basic memory creation."""
        memory = Memory(
            domain=CognitiveDomain.BIOGRAPHY,
            key="name",
            value="John Doe",
            confidence=0.95,
        )
        
        assert memory.domain == CognitiveDomain.BIOGRAPHY
        assert memory.key == "name"
        assert memory.value == "John Doe"
        assert memory.confidence == 0.95
        assert memory.memory_id  # Auto-generated
    
    def test_memory_id_generation(self):
        """Test that memory IDs are consistent."""
        m1 = Memory(
            domain=CognitiveDomain.WORK,
            key="role",
            value="developer",
        )
        m2 = Memory(
            domain=CognitiveDomain.WORK,
            key="role",
            value="developer",
        )
        
        assert m1.memory_id == m2.memory_id
    
    def test_memory_equality(self):
        """Test memory equality comparison."""
        m1 = Memory(
            domain=CognitiveDomain.PREFERENCES,
            key="language",
            value="Python",
            confidence=0.8,
        )
        m2 = Memory(
            domain=CognitiveDomain.PREFERENCES,
            key="Language",  # Case insensitive
            value="PYTHON",  # Case insensitive
            confidence=0.9,  # Different confidence OK
        )
        
        assert m1 == m2
    
    def test_memory_serialization(self):
        """Test to_dict and from_dict."""
        original = Memory(
            domain=CognitiveDomain.SOCIAL,
            key="manager",
            value="Jane Smith",
            confidence=0.85,
            source="conversation_123",
            metadata={"extraction_method": "rule"},
        )
        
        data = original.to_dict()
        restored = Memory.from_dict(data)
        
        assert restored.domain == original.domain
        assert restored.key == original.key
        assert restored.value == original.value
        assert restored.confidence == original.confidence
        assert restored.source == original.source
    
    def test_memory_similarity(self):
        """Test similar_to method."""
        m1 = Memory(
            domain=CognitiveDomain.WORK,
            key="project",
            value="Building a web application with React",
        )
        m2 = Memory(
            domain=CognitiveDomain.WORK,
            key="project",
            value="Building a web application with React and Node",
        )
        m3 = Memory(
            domain=CognitiveDomain.WORK,
            key="project",
            value="Machine learning research paper",
        )
        
        assert m1.similar_to(m2, threshold=0.6)
        assert not m1.similar_to(m3, threshold=0.6)
    
    def test_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1]."""
        m1 = Memory(
            domain=CognitiveDomain.BIOGRAPHY,
            key="test",
            value="test",
            confidence=1.5,  # Should clamp to 1.0
        )
        m2 = Memory(
            domain=CognitiveDomain.BIOGRAPHY,
            key="test",
            value="test",
            confidence=-0.5,  # Should clamp to 0.0
        )
        
        assert m1.confidence == 1.0
        assert m2.confidence == 0.0


class TestCognitiveDomain:
    """Tests for CognitiveDomain enum."""
    
    def test_all_domains_exist(self):
        """Verify all six domains are defined."""
        domains = list(CognitiveDomain)
        assert len(domains) == 6
        assert CognitiveDomain.BIOGRAPHY in domains
        assert CognitiveDomain.PREFERENCES in domains
        assert CognitiveDomain.WORK in domains
        assert CognitiveDomain.SOCIAL in domains
        assert CognitiveDomain.TEMPORAL in domains
        assert CognitiveDomain.PROCEDURAL in domains
    
    def test_from_string(self):
        """Test parsing domain from string."""
        assert CognitiveDomain.from_string("biography") == CognitiveDomain.BIOGRAPHY
        assert CognitiveDomain.from_string("WORK") == CognitiveDomain.WORK
        assert CognitiveDomain.from_string("  preferences  ") == CognitiveDomain.PREFERENCES
        
        with pytest.raises(ValueError):
            CognitiveDomain.from_string("invalid")


class TestRuleBasedExtractor:
    """Tests for rule-based extraction."""
    
    def test_biography_extraction(self):
        """Test biography domain extraction."""
        extractor = RuleBasedExtractor()
        
        text = "My name is John Smith. I'm 35 years old and I live in San Francisco."
        memories = extractor.extract(text)
        
        # Should extract name, age, location
        domains = [m.domain for m in memories]
        assert CognitiveDomain.BIOGRAPHY in domains
        
        names = [m for m in memories if m.key == "name"]
        assert len(names) > 0
        assert "John" in names[0].value
    
    def test_preferences_extraction(self):
        """Test preferences domain extraction."""
        extractor = RuleBasedExtractor()
        
        text = "I prefer Python over JavaScript. I love dark mode."
        memories = extractor.extract(text)
        
        prefs = [m for m in memories if m.domain == CognitiveDomain.PREFERENCES]
        assert len(prefs) > 0
    
    def test_work_extraction(self):
        """Test work domain extraction."""
        extractor = RuleBasedExtractor()
        
        text = "I work as a software engineer at Google. I'm working on a search project."
        memories = extractor.extract(text)
        
        work = [m for m in memories if m.domain == CognitiveDomain.WORK]
        assert len(work) > 0
        
        # Check for role or company
        keys = [m.key for m in work]
        assert "role" in keys or "company" in keys or "current_project" in keys
    
    def test_social_extraction(self):
        """Test social domain extraction."""
        extractor = RuleBasedExtractor()
        
        text = "My manager is Sarah Chen. My wife is Emily."
        memories = extractor.extract(text)
        
        social = [m for m in memories if m.domain == CognitiveDomain.SOCIAL]
        assert len(social) > 0
    
    def test_temporal_extraction(self):
        """Test temporal domain extraction."""
        extractor = RuleBasedExtractor()
        
        text = "My timezone is PST. I work from 9am to 5pm."
        memories = extractor.extract(text)
        
        temporal = [m for m in memories if m.domain == CognitiveDomain.TEMPORAL]
        assert len(temporal) > 0
    
    def test_procedural_extraction(self):
        """Test procedural domain extraction."""
        extractor = RuleBasedExtractor()
        
        text = "I always start with tests first. Remember to commit often."
        memories = extractor.extract(text)
        
        proc = [m for m in memories if m.domain == CognitiveDomain.PROCEDURAL]
        assert len(proc) > 0
    
    def test_key_value_extraction(self):
        """Test explicit key-value format."""
        extractor = RuleBasedExtractor()
        
        # Key-value format requires keys at start of lines (no leading whitespace)
        text = """name: John Doe
role: Developer
timezone: UTC"""
        memories = extractor.extract_from_key_value(text)
        
        assert len(memories) == 3
        names = [m for m in memories if m.key == "name"]
        assert len(names) == 1
        assert names[0].value == "John Doe"
    
    def test_source_tracking(self):
        """Test that source is properly tracked."""
        extractor = RuleBasedExtractor()
        
        text = "My name is Jane"
        memories = extractor.extract(text, source="msg_001")
        
        for memory in memories:
            assert memory.source == "msg_001"


class TestLLMExtractor:
    """Tests for LLM-based extraction."""
    
    def test_with_mock_client(self):
        """Test extraction with mock LLM client."""
        mock_client = MockLLMClient()
        extractor = LLMExtractor(client=mock_client)
        
        text = "Hi, I'm John. I use Python and prefer dark mode."
        memories = extractor.extract(text)
        
        # MockLLMClient returns specific patterns
        assert len(memories) > 0
        assert any(m.domain == CognitiveDomain.BIOGRAPHY for m in memories)
        assert any(m.domain == CognitiveDomain.WORK for m in memories)
    
    def test_custom_mock_responses(self):
        """Test with custom mock responses."""
        mock_client = MockLLMClient(responses={
            "alice": json.dumps([
                {"domain": "biography", "key": "name", "value": "Alice", "confidence": 0.95},
                {"domain": "work", "key": "role", "value": "Designer", "confidence": 0.9},
            ])
        })
        extractor = LLMExtractor(client=mock_client)
        
        memories = extractor.extract("My name is Alice and I'm a designer.")
        
        assert len(memories) == 2
        assert memories[0].value == "Alice"
        assert memories[1].value == "Designer"
    
    def test_empty_text(self):
        """Test extraction with empty text."""
        mock_client = MockLLMClient()
        extractor = LLMExtractor(client=mock_client)
        
        memories = extractor.extract("")
        assert memories == []
    
    def test_domain_specific_extraction(self):
        """Test extraction for specific domain."""
        mock_client = MockLLMClient(responses={
            "work": json.dumps([
                {"domain": "work", "key": "skill", "value": "Python", "confidence": 0.9},
            ])
        })
        extractor = LLMExtractor(client=mock_client)
        
        memories = extractor.extract_domain(
            "I'm experienced with Python and JavaScript",
            CognitiveDomain.WORK
        )
        
        # Should only return work domain
        assert all(m.domain == CognitiveDomain.WORK for m in memories)


class TestMemoryDeduplicator:
    """Tests for memory deduplication."""
    
    def test_exact_deduplication(self):
        """Test exact match deduplication."""
        deduplicator = MemoryDeduplicator(strategy="exact")
        
        memories = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John"),
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John"),  # Duplicate
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="Jane"),  # Different value
        ]
        
        result = deduplicator.deduplicate(memories)
        
        assert len(result.unique_memories) == 2
        assert result.duplicates_removed == 1
    
    def test_fuzzy_deduplication(self):
        """Test fuzzy match deduplication."""
        deduplicator = MemoryDeduplicator(strategy="fuzzy", similarity_threshold=0.7)
        
        memories = [
            Memory(domain=CognitiveDomain.WORK, key="project", value="Building a web app with React"),
            Memory(domain=CognitiveDomain.WORK, key="project", value="Building a web app with React and Node"),  # Similar
            Memory(domain=CognitiveDomain.WORK, key="project", value="Machine learning research"),
        ]
        
        result = deduplicator.deduplicate(memories)
        
        # First two should be deduplicated
        assert len(result.unique_memories) == 2
    
    def test_keep_highest_confidence(self):
        """Test that highest confidence duplicate is kept."""
        deduplicator = MemoryDeduplicator(strategy="exact")
        
        memories = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John", confidence=0.7),
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John", confidence=0.95),
        ]
        
        result = deduplicator.deduplicate(memories)
        
        assert len(result.unique_memories) == 1
        assert result.unique_memories[0].confidence == 0.95
    
    def test_find_duplicates(self):
        """Test finding duplicates for a new memory."""
        deduplicator = MemoryDeduplicator(strategy="fuzzy", similarity_threshold=0.7)
        
        existing = [
            Memory(domain=CognitiveDomain.WORK, key="skill", value="Python programming language"),
            Memory(domain=CognitiveDomain.WORK, key="role", value="Developer"),
        ]
        
        # Create a very similar memory
        new_memory = Memory(domain=CognitiveDomain.WORK, key="skill", value="Python programming")
        
        duplicates = deduplicator.find_duplicates(new_memory, existing)
        
        assert len(duplicates) == 1
        assert "Python" in duplicates[0].value


class TestConflictResolver:
    """Tests for conflict resolution."""
    
    def test_confidence_wins(self):
        """Test confidence-based resolution."""
        resolver = ConflictResolver(strategy=ConflictStrategy.CONFIDENCE_WINS)
        
        memories = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="location", value="NYC", confidence=0.7),
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="location", value="SF", confidence=0.9),
        ]
        
        result = resolver.resolve(memories)
        
        assert len(result.resolved_memories) == 1
        assert result.resolved_memories[0].value == "SF"
    
    def test_latest_wins(self):
        """Test timestamp-based resolution."""
        resolver = ConflictResolver(strategy=ConflictStrategy.LATEST_WINS)
        
        old_time = datetime.utcnow() - timedelta(days=1)
        new_time = datetime.utcnow()
        
        memories = [
            Memory(domain=CognitiveDomain.WORK, key="company", value="OldCo", timestamp=old_time),
            Memory(domain=CognitiveDomain.WORK, key="company", value="NewCo", timestamp=new_time),
        ]
        
        result = resolver.resolve(memories)
        
        assert len(result.resolved_memories) == 1
        assert result.resolved_memories[0].value == "NewCo"
    
    def test_keep_both(self):
        """Test keeping both conflicting values with versions."""
        resolver = ConflictResolver(strategy=ConflictStrategy.KEEP_BOTH)
        
        memories = [
            Memory(domain=CognitiveDomain.PREFERENCES, key="editor", value="Vim"),
            Memory(domain=CognitiveDomain.PREFERENCES, key="editor", value="VSCode"),
        ]
        
        result = resolver.resolve(memories)
        
        # Should keep both with version numbers
        assert len(result.resolved_memories) == 2
        assert any("v1" in m.key for m in result.resolved_memories)
        assert any("v2" in m.key for m in result.resolved_memories)
    
    def test_merge_strategy(self):
        """Test merging conflicting values."""
        resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)
        
        memories = [
            Memory(domain=CognitiveDomain.WORK, key="skills", value="Python"),
            Memory(domain=CognitiveDomain.WORK, key="skills", value="JavaScript"),
        ]
        
        result = resolver.resolve(memories)
        
        assert len(result.resolved_memories) == 1
        # Merged value should contain both
        assert "Python" in result.resolved_memories[0].value
        assert "JavaScript" in result.resolved_memories[0].value
    
    def test_no_conflict_same_value(self):
        """Test that same values don't create conflicts."""
        resolver = ConflictResolver()
        
        memories = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John", confidence=0.7),
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John", confidence=0.9),
        ]
        
        result = resolver.resolve(memories)
        
        assert len(result.resolved_memories) == 1
        assert result.conflicts_found == 0  # No value conflict


class TestMemoryMerger:
    """Tests for memory merging."""
    
    def test_basic_merge(self):
        """Test merging two memory sets."""
        merger = MemoryMerger()
        
        set1 = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John"),
            Memory(domain=CognitiveDomain.WORK, key="role", value="Developer"),
        ]
        set2 = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="location", value="NYC"),
            Memory(domain=CognitiveDomain.WORK, key="company", value="Acme"),
        ]
        
        result = merger.merge(set1, set2)
        
        assert len(result.merged_memories) == 4
        assert result.sources_merged == 2
    
    def test_merge_with_duplicates(self):
        """Test merging with deduplication."""
        merger = MemoryMerger()
        
        set1 = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John"),
        ]
        set2 = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John"),  # Duplicate
            Memory(domain=CognitiveDomain.WORK, key="role", value="Developer"),
        ]
        
        result = merger.merge(set1, set2)
        
        assert len(result.merged_memories) == 2
        assert result.duplicates_removed == 1
    
    def test_corroboration_boost(self):
        """Test confidence boost for corroborated facts."""
        merger = MemoryMerger(boost_corroborated=True, corroboration_boost=0.1)
        
        set1 = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John", confidence=0.8),
        ]
        set2 = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John", confidence=0.85),
        ]
        
        result = merger.merge(set1, set2, source_labels=["source1", "source2"])
        
        # Higher confidence + boost
        assert result.merged_memories[0].confidence >= 0.85 + 0.1 - 0.01  # Allow small float error
    
    def test_incremental_merge(self):
        """Test adding single memory to existing set."""
        merger = MemoryMerger()
        
        existing = [
            Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John"),
        ]
        new_memory = Memory(domain=CognitiveDomain.WORK, key="role", value="Developer")
        
        updated, replaced = merger.incremental_merge(existing, new_memory)
        
        assert len(updated) == 2
        assert replaced is None


class TestMemoryExtractor:
    """Tests for main MemoryExtractor class."""
    
    def test_rule_mode_extraction(self):
        """Test rule-based extraction mode."""
        extractor = MemoryExtractor(mode="rule")
        
        text = "My name is John Smith. I work at Google as a developer."
        result = extractor.extract(text)
        
        assert len(result.memories) > 0
        assert result.method == "rule"
        assert result.processing_time_ms > 0
    
    def test_llm_mode_extraction(self):
        """Test LLM extraction with mock client."""
        mock_client = MockLLMClient()
        extractor = MemoryExtractor(mode="llm", llm_client=mock_client)
        
        text = "I'm John and I love Python."
        result = extractor.extract(text)
        
        assert result.method == "llm"
    
    def test_hybrid_mode(self):
        """Test hybrid extraction mode."""
        mock_client = MockLLMClient()
        extractor = MemoryExtractor(mode="hybrid", llm_client=mock_client)
        
        text = "My name is John. I prefer Python over JavaScript."
        result = extractor.extract(text)
        
        assert result.method == "hybrid"
    
    def test_conversation_extraction(self):
        """Test extracting from conversation format."""
        extractor = MemoryExtractor(mode="rule")
        
        messages = [
            {"role": "user", "content": "Hi, my name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "I work at Meta."},
        ]
        
        result = extractor.extract_conversation(messages)
        
        assert len(result.memories) > 0
        # Should find name and company
        keys = [m.key for m in result.memories]
        assert "name" in keys or "company" in keys
    
    def test_domain_filtering(self):
        """Test extracting specific domains only."""
        extractor = MemoryExtractor(mode="rule")
        
        text = "My name is John. I work at Google. My timezone is PST."
        result = extractor.extract(text, domains=[CognitiveDomain.WORK])
        
        # Should only have work domain
        assert all(m.domain == CognitiveDomain.WORK for m in result.memories)
    
    def test_extract_by_domain(self):
        """Test organizing results by domain."""
        extractor = MemoryExtractor(mode="rule")
        
        text = "My name is John. I love Python. I work at Google."
        by_domain = extractor.extract_by_domain(text)
        
        assert isinstance(by_domain, dict)
        assert all(domain in by_domain for domain in CognitiveDomain)
    
    def test_source_tracking(self):
        """Test that source is tracked through extraction."""
        extractor = MemoryExtractor(mode="rule")
        
        result = extractor.extract("My name is Jane", source="test_source")
        
        for memory in result.memories:
            assert memory.source == "test_source"
    
    def test_auto_deduplication(self):
        """Test automatic deduplication."""
        extractor = MemoryExtractor(mode="rule", auto_dedupe=True)
        
        # Text that might produce duplicate extractions
        text = "My name is John. Call me John."
        result = extractor.extract(text)
        
        # Should be deduplicated
        names = [m for m in result.memories if m.key == "name"]
        assert len(names) <= 1
    
    def test_merge_memories(self):
        """Test merging memory sets."""
        extractor = MemoryExtractor(mode="rule")
        
        set1 = [Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John")]
        set2 = [Memory(domain=CognitiveDomain.WORK, key="role", value="Developer")]
        
        merged = extractor.merge_memories(set1, set2)
        
        assert len(merged) == 2


class TestExtractionResult:
    """Tests for ExtractionResult."""
    
    def test_filter_by_domain(self):
        """Test filtering results by domain."""
        result = ExtractionResult(
            memories=[
                Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John"),
                Memory(domain=CognitiveDomain.WORK, key="role", value="Dev"),
                Memory(domain=CognitiveDomain.BIOGRAPHY, key="age", value="30"),
            ]
        )
        
        bio = result.filter_by_domain(CognitiveDomain.BIOGRAPHY)
        assert len(bio) == 2
        
        work = result.filter_by_domain(CognitiveDomain.WORK)
        assert len(work) == 1
    
    def test_filter_by_confidence(self):
        """Test filtering by confidence threshold."""
        result = ExtractionResult(
            memories=[
                Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John", confidence=0.9),
                Memory(domain=CognitiveDomain.BIOGRAPHY, key="age", value="30", confidence=0.4),
                Memory(domain=CognitiveDomain.WORK, key="role", value="Dev", confidence=0.7),
            ]
        )
        
        high_conf = result.filter_by_confidence(min_confidence=0.8)
        assert len(high_conf) == 1
        assert high_conf[0].key == "name"
    
    def test_serialization(self):
        """Test result serialization."""
        result = ExtractionResult(
            memories=[Memory(domain=CognitiveDomain.BIOGRAPHY, key="name", value="John")],
            text="My name is John",
            method="rule",
            processing_time_ms=5.5,
        )
        
        data = result.to_dict()
        restored = ExtractionResult.from_dict(data)
        
        assert len(restored.memories) == 1
        assert restored.method == "rule"


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_extraction_pipeline(self):
        """Test complete extraction with all features."""
        extractor = MemoryExtractor(
            mode="rule",
            deduplication_strategy="fuzzy",
            conflict_strategy=ConflictStrategy.CONFIDENCE_WINS,
            auto_dedupe=True,
            auto_resolve_conflicts=True,
        )
        
        conversation = [
            {"role": "user", "content": "Hi, I'm Alice Johnson. I work as a software engineer at Meta."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "I prefer Python and use VS Code. My timezone is PST."},
            {"role": "user", "content": "I work from 9am to 5pm."},
        ]
        
        result = extractor.extract_conversation(conversation)
        
        # Should extract multiple domains
        domains = set(m.domain for m in result.memories)
        assert len(domains) >= 2
        
        # Check for key facts
        all_text = " ".join(m.value.lower() for m in result.memories)
        assert "alice" in all_text or any("alice" in m.value.lower() for m in result.memories)
    
    def test_memory_update_workflow(self):
        """Test updating memories over time."""
        extractor = MemoryExtractor(mode="rule")
        
        # Initial extraction
        memories = extractor.extract("I work at Google").memories
        
        # Update with new info
        new_memory = Memory(
            domain=CognitiveDomain.WORK,
            key="company",
            value="Meta",
            confidence=0.9,
        )
        
        updated, was_updated = extractor.add_memory(memories, new_memory)
        
        # Check update happened
        assert was_updated or len(updated) > len(memories)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
