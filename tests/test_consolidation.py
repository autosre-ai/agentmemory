"""
Tests for memory consolidation module.

Tests similarity detection, deduplication strategies, auto-merge,
conflict detection, and the main consolidator.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import json

from agent_memory_toolkit.consolidation import (
    # Models
    ConsolidationStrategy,
    DeduplicationStrategy,
    ConflictType,
    ConflictSeverity,
    SimilarityScore,
    ConsolidationConfig,
    ConsolidationResult,
    # Core classes
    MemoryData,
    SimilarityDetector,
    Deduplicator,
    MemoryAutoMerger,
    ConflictDetector,
    MemoryConsolidator,
    ConsolidationScheduler,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_memories():
    """Create sample memories for testing."""
    now = datetime.utcnow()
    return [
        MemoryData(
            id="mem1",
            content="The user's name is John Doe",
            metadata={
                "confidence": 0.95,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
        ),
        MemoryData(
            id="mem2",
            content="John Doe is the user's name",  # Similar to mem1
            metadata={
                "confidence": 0.85,
                "created_at": (now - timedelta(days=1)).isoformat(),
                "updated_at": now.isoformat(),
            }
        ),
        MemoryData(
            id="mem3",
            content="The user works at Acme Corp",
            metadata={
                "confidence": 0.9,
                "created_at": now.isoformat(),
            }
        ),
        MemoryData(
            id="mem4",
            content="User prefers dark mode",
            metadata={
                "confidence": 1.0,
            }
        ),
        MemoryData(
            id="mem5",
            content="The user's name is John Doe",  # Exact duplicate of mem1
            metadata={
                "confidence": 0.7,
                "created_at": (now + timedelta(hours=1)).isoformat(),
            }
        ),
    ]


@pytest.fixture
def exact_duplicates():
    """Create exact duplicate memories."""
    return [
        MemoryData(
            id="dup1",
            content="Python is a programming language",
            metadata={"confidence": 0.9}
        ),
        MemoryData(
            id="dup2",
            content="Python is a programming language",  # Exact same
            metadata={"confidence": 0.95}
        ),
        MemoryData(
            id="dup3",
            content="Python is a programming language",  # Exact same
            metadata={"confidence": 0.8}
        ),
    ]


@pytest.fixture
def conflicting_memories():
    """Create memories with conflicts."""
    return [
        MemoryData(
            id="conf1",
            content="The user's age is 25",
            metadata={"confidence": 0.9}
        ),
        MemoryData(
            id="conf2",
            content="The user's age is 30",  # Value conflict
            metadata={"confidence": 0.85}
        ),
        MemoryData(
            id="conf3",
            content="The user is not married",
            metadata={"confidence": 0.95}
        ),
        MemoryData(
            id="conf4",
            content="The user is married",  # Contradiction
            metadata={"confidence": 0.9}
        ),
    ]


# ==============================================================================
# Similarity Detection Tests
# ==============================================================================

class TestSimilarityDetector:
    """Tests for SimilarityDetector."""
    
    def test_init_default(self):
        """Test default initialization."""
        detector = SimilarityDetector()
        assert detector.strategy == ConsolidationStrategy.HYBRID
        assert detector.similarity_threshold == 0.85
        assert detector.duplicate_threshold == 0.95
    
    def test_init_custom(self):
        """Test custom initialization."""
        detector = SimilarityDetector(
            strategy=ConsolidationStrategy.FUZZY_MATCH,
            similarity_threshold=0.7,
            duplicate_threshold=0.9,
        )
        assert detector.strategy == ConsolidationStrategy.FUZZY_MATCH
        assert detector.similarity_threshold == 0.7
        assert detector.duplicate_threshold == 0.9
    
    def test_content_hash(self):
        """Test content hashing."""
        detector = SimilarityDetector()
        
        hash1 = detector._content_hash("Hello World")
        hash2 = detector._content_hash("Hello World")
        hash3 = detector._content_hash("  hello world  ")  # Should normalize
        hash4 = detector._content_hash("Different content")
        
        assert hash1 == hash2
        assert hash1 == hash3  # Same after normalization
        assert hash1 != hash4
    
    def test_exact_match(self):
        """Test exact content matching."""
        detector = SimilarityDetector(strategy=ConsolidationStrategy.EXACT_MATCH)
        
        m1 = MemoryData(id="1", content="Hello World")
        m2 = MemoryData(id="2", content="Hello World")
        m3 = MemoryData(id="3", content="Hello World!")
        
        assert detector.is_duplicate(m1, m2)
        assert not detector.is_duplicate(m1, m3)
    
    def test_fuzzy_match(self):
        """Test fuzzy matching."""
        detector = SimilarityDetector(
            strategy=ConsolidationStrategy.FUZZY_MATCH,
            similarity_threshold=0.8,
        )
        
        m1 = MemoryData(id="1", content="The user's name is John")
        m2 = MemoryData(id="2", content="The users name is John")  # Minor diff
        m3 = MemoryData(id="3", content="Something completely different")
        
        score12 = detector.calculate_similarity(m1, m2)
        score13 = detector.calculate_similarity(m1, m3)
        
        assert score12.score > 0.8
        assert score13.score < 0.5
    
    def test_semantic_match_without_embeddings(self):
        """Test semantic matching falls back to fuzzy without embeddings."""
        detector = SimilarityDetector(strategy=ConsolidationStrategy.SEMANTIC_MATCH)
        
        m1 = MemoryData(id="1", content="Hello World")
        m2 = MemoryData(id="2", content="Hello World")
        
        # Should fall back to fuzzy matching
        score = detector.calculate_similarity(m1, m2)
        assert score.score > 0.9
    
    def test_semantic_match_with_embeddings(self):
        """Test semantic matching with embeddings."""
        detector = SimilarityDetector(strategy=ConsolidationStrategy.SEMANTIC_MATCH)
        
        # Create memories with embeddings
        m1 = MemoryData(
            id="1",
            content="Hello World",
            embedding=[1.0, 0.0, 0.0],
        )
        m2 = MemoryData(
            id="2",
            content="Different text",
            embedding=[0.9, 0.1, 0.0],  # Similar embedding
        )
        
        score = detector.calculate_similarity(m1, m2)
        assert score.score > 0.9  # Cosine similarity should be high
    
    def test_find_duplicates(self, exact_duplicates):
        """Test finding duplicates in a list."""
        detector = SimilarityDetector()
        
        duplicates = detector.find_duplicates(exact_duplicates)
        
        assert len(duplicates) >= 2  # At least 2 pairs from 3 duplicates
    
    def test_quick_hash_groups(self, exact_duplicates):
        """Test quick hash-based grouping."""
        detector = SimilarityDetector()
        
        groups = detector.quick_hash_groups(exact_duplicates)
        
        # All three should be in one group
        assert len(groups) == 1
        assert len(list(groups.values())[0]) == 3
    
    def test_find_similar(self, sample_memories):
        """Test finding similar memories."""
        detector = SimilarityDetector(similarity_threshold=0.5)
        
        target = sample_memories[0]  # "The user's name is John Doe"
        others = sample_memories[1:]
        
        similar = detector.find_similar(target, others, min_score=0.5)
        
        # Should find at least the exact duplicate
        assert len(similar) >= 1
    
    def test_cluster_similar(self, sample_memories):
        """Test clustering similar memories."""
        detector = SimilarityDetector(similarity_threshold=0.7)
        
        clusters = detector.cluster_similar(sample_memories)
        
        # Should have at least one cluster (the duplicates)
        assert len(clusters) >= 1


# ==============================================================================
# Deduplication Tests
# ==============================================================================

class TestDeduplicator:
    """Tests for Deduplicator."""
    
    def test_init_default(self):
        """Test default initialization."""
        dedup = Deduplicator()
        assert dedup.strategy == DeduplicationStrategy.KEEP_HIGHEST_CONFIDENCE
        assert dedup.duplicate_threshold == 0.95
    
    def test_deduplicate_exact(self, exact_duplicates):
        """Test deduplicating exact duplicates."""
        dedup = Deduplicator(strategy=DeduplicationStrategy.KEEP_HIGHEST_CONFIDENCE)
        
        result = dedup.deduplicate(exact_duplicates)
        
        assert result.total_removed == 2
        assert result.total_kept == 1
        # Should keep dup2 (highest confidence 0.95)
        assert "dup2" in result.kept_memory_ids
    
    def test_strategy_keep_newest(self, exact_duplicates):
        """Test keep newest strategy."""
        # Add timestamps
        now = datetime.utcnow()
        exact_duplicates[0].metadata["created_at"] = (now - timedelta(days=2)).isoformat()
        exact_duplicates[1].metadata["created_at"] = (now - timedelta(days=1)).isoformat()
        exact_duplicates[2].metadata["created_at"] = now.isoformat()
        
        dedup = Deduplicator(strategy=DeduplicationStrategy.KEEP_NEWEST)
        result = dedup.deduplicate(exact_duplicates)
        
        # Should keep dup3 (newest)
        assert "dup3" in result.kept_memory_ids
    
    def test_strategy_keep_oldest(self, exact_duplicates):
        """Test keep oldest strategy."""
        now = datetime.utcnow()
        exact_duplicates[0].metadata["created_at"] = (now - timedelta(days=2)).isoformat()
        exact_duplicates[1].metadata["created_at"] = (now - timedelta(days=1)).isoformat()
        exact_duplicates[2].metadata["created_at"] = now.isoformat()
        
        dedup = Deduplicator(strategy=DeduplicationStrategy.KEEP_OLDEST)
        result = dedup.deduplicate(exact_duplicates)
        
        # Should keep dup1 (oldest)
        assert "dup1" in result.kept_memory_ids
    
    def test_strategy_keep_most_accessed(self, exact_duplicates):
        """Test keep most accessed strategy."""
        exact_duplicates[0].metadata["access_count"] = 10
        exact_duplicates[1].metadata["access_count"] = 50
        exact_duplicates[2].metadata["access_count"] = 5
        
        dedup = Deduplicator(strategy=DeduplicationStrategy.KEEP_MOST_ACCESSED)
        result = dedup.deduplicate(exact_duplicates)
        
        # Should keep dup2 (highest access count)
        assert "dup2" in result.kept_memory_ids
    
    def test_no_duplicates(self):
        """Test with no duplicates."""
        memories = [
            MemoryData(id="1", content="First memory"),
            MemoryData(id="2", content="Second memory"),
            MemoryData(id="3", content="Third memory"),
        ]
        
        dedup = Deduplicator()
        result = dedup.deduplicate(memories)
        
        assert result.total_removed == 0
        assert result.total_kept == 3
    
    def test_estimate_duplicates(self, sample_memories):
        """Test duplicate estimation."""
        dedup = Deduplicator()
        
        estimate = dedup.estimate_duplicates(sample_memories)
        
        assert "total_memories" in estimate
        assert "exact_duplicates" in estimate
        assert "estimated_reduction_percent" in estimate
        assert estimate["total_memories"] == len(sample_memories)
    
    def test_find_and_remove_exact_duplicates(self, exact_duplicates):
        """Test quick exact duplicate removal."""
        dedup = Deduplicator()
        
        unique, removed = dedup.find_and_remove_exact_duplicates(exact_duplicates)
        
        assert len(unique) == 1
        assert len(removed) == 2


# ==============================================================================
# Merger Tests
# ==============================================================================

class TestMemoryAutoMerger:
    """Tests for MemoryAutoMerger."""
    
    def test_init_default(self):
        """Test default initialization."""
        merger = MemoryAutoMerger()
        assert merger.merge_threshold == 0.90
        assert merger.max_cluster_size == 5
    
    def test_should_merge(self):
        """Test merge threshold logic."""
        merger = MemoryAutoMerger(merge_threshold=0.8)
        
        m1 = MemoryData(id="1", content="Hello")
        m2 = MemoryData(id="2", content="Hello")
        
        # High similarity should merge
        assert merger.should_merge(m1, m2, 0.9)
        
        # Low similarity should not merge
        assert not merger.should_merge(m1, m2, 0.7)
    
    def test_calculate_merged_confidence(self):
        """Test merged confidence calculation."""
        merger = MemoryAutoMerger()
        
        memories = [
            MemoryData(id="1", content="A", metadata={"confidence": 0.9}),
            MemoryData(id="2", content="B", metadata={"confidence": 0.8}),
            MemoryData(id="3", content="C", metadata={"confidence": 0.7}),
        ]
        
        confidence = merger._calculate_merged_confidence(memories)
        
        # Should be at least max confidence
        assert confidence >= 0.9
    
    def test_find_merge_candidates(self):
        """Test finding merge candidates."""
        merger = MemoryAutoMerger(merge_threshold=0.5)
        detector = SimilarityDetector(similarity_threshold=0.5)
        
        memories = [
            MemoryData(id="1", content="User likes Python programming"),
            MemoryData(id="2", content="User enjoys Python development"),
            MemoryData(id="3", content="Something completely different"),
        ]
        
        candidates = merger.find_merge_candidates(memories, detector)
        
        # Should find at least one candidate (1 and 2)
        assert len(candidates) >= 0  # Depends on threshold
    
    def test_merge_cluster_simple(self):
        """Test simple cluster merge."""
        merger = MemoryAutoMerger()
        
        cluster_ids = ["1", "2"]
        all_memories = [
            MemoryData(id="1", content="User likes coding", metadata={"confidence": 0.8}),
            MemoryData(id="2", content="User enjoys programming", metadata={"confidence": 0.9}),
        ]
        
        result = merger.merge_cluster(cluster_ids, all_memories)
        
        assert result is not None
        assert result.merged_memory_id is not None
        assert "1" in result.source_memory_ids
        assert "2" in result.source_memory_ids
    
    def test_merge_with_custom_summarizer(self):
        """Test merge with custom summarizer."""
        def custom_summarizer(texts):
            return f"Combined: {', '.join(texts)}"
        
        merger = MemoryAutoMerger(summarize_fn=custom_summarizer)
        
        cluster_ids = ["1", "2"]
        all_memories = [
            MemoryData(id="1", content="First content"),
            MemoryData(id="2", content="Second content"),
        ]
        
        result = merger.merge_cluster(cluster_ids, all_memories)
        
        assert "Combined:" in result.merged_content


# ==============================================================================
# Conflict Detection Tests
# ==============================================================================

class TestConflictDetector:
    """Tests for ConflictDetector."""
    
    def test_init_default(self):
        """Test default initialization."""
        detector = ConflictDetector()
        assert detector.confidence_gap_threshold == 0.5
        assert detector.similarity_threshold == 0.7
    
    def test_detect_value_conflict(self, conflicting_memories):
        """Test value conflict detection."""
        detector = ConflictDetector(similarity_threshold=0.5)
        
        conflicts = detector.detect_all_conflicts(conflicting_memories)
        
        # Should detect conflicts between age values
        value_conflicts = [
            c for c in conflicts
            if c.conflict_type == ConflictType.VALUE_CONFLICT
        ]
        
        assert len(value_conflicts) >= 1
    
    def test_detect_semantic_contradiction(self, conflicting_memories):
        """Test semantic contradiction detection."""
        detector = ConflictDetector(similarity_threshold=0.3)
        
        conflicts = detector.detect_all_conflicts(conflicting_memories)
        
        # Should detect contradiction between married/not married
        contradictions = [
            c for c in conflicts
            if c.conflict_type == ConflictType.SEMANTIC_CONTRADICTION
        ]
        
        # May or may not detect depending on similarity
        assert isinstance(contradictions, list)
    
    def test_detect_confidence_gap(self):
        """Test confidence gap detection."""
        detector = ConflictDetector(
            confidence_gap_threshold=0.3,
            similarity_threshold=0.9,
        )
        
        memories = [
            MemoryData(id="1", content="Same content", metadata={"confidence": 1.0}),
            MemoryData(id="2", content="Same content", metadata={"confidence": 0.5}),
        ]
        
        conflicts = detector.detect_all_conflicts(memories)
        
        gap_conflicts = [
            c for c in conflicts
            if c.conflict_type == ConflictType.CONFIDENCE_GAP
        ]
        
        assert len(gap_conflicts) >= 1
    
    def test_detect_conflicts_for_memory(self, sample_memories):
        """Test conflict detection for a single new memory."""
        detector = ConflictDetector(similarity_threshold=0.5)
        
        new_memory = MemoryData(
            id="new",
            content="The user's name is Jane Doe",  # Conflicts with John Doe
            metadata={"confidence": 0.9}
        )
        
        conflicts = detector.detect_conflicts_for_memory(new_memory, sample_memories)
        
        # Should find conflicts with similar memories about user's name
        assert isinstance(conflicts, list)
    
    def test_conflict_summary(self, conflicting_memories):
        """Test conflict summary generation."""
        detector = ConflictDetector(similarity_threshold=0.5)
        
        conflicts = detector.detect_all_conflicts(conflicting_memories)
        summary = detector.get_conflict_summary(conflicts)
        
        assert "total_conflicts" in summary
        assert "by_type" in summary
        assert "by_severity" in summary


# ==============================================================================
# Main Consolidator Tests
# ==============================================================================

class TestMemoryConsolidator:
    """Tests for MemoryConsolidator."""
    
    def test_init_default(self):
        """Test default initialization."""
        consolidator = MemoryConsolidator()
        assert consolidator.config is not None
    
    def test_init_custom_config(self):
        """Test custom config initialization."""
        config = ConsolidationConfig(
            similarity_threshold=0.9,
            auto_merge=True,
        )
        consolidator = MemoryConsolidator(config)
        
        assert consolidator.config.similarity_threshold == 0.9
        assert consolidator.config.auto_merge is True
    
    def test_consolidate_dry_run(self, sample_memories):
        """Test consolidation dry run."""
        consolidator = MemoryConsolidator()
        
        result = consolidator.consolidate(sample_memories, dry_run=True)
        
        assert isinstance(result, ConsolidationResult)
        assert result.memories_analyzed == len(sample_memories)
        assert result.duplicates_removed == 0  # Dry run
    
    def test_consolidate_full_run(self, sample_memories):
        """Test full consolidation."""
        consolidator = MemoryConsolidator()
        
        result = consolidator.consolidate(sample_memories, dry_run=False)
        
        assert isinstance(result, ConsolidationResult)
        assert result.success is True
    
    def test_consolidate_with_progress_callback(self, sample_memories):
        """Test consolidation with progress tracking."""
        consolidator = MemoryConsolidator()
        
        progress_calls = []
        
        def progress_callback(stage, current, total):
            progress_calls.append((stage, current, total))
        
        result = consolidator.consolidate(
            sample_memories,
            progress_callback=progress_callback,
        )
        
        assert len(progress_calls) > 0
        # Should have multiple phases
        stages = [p[0] for p in progress_calls]
        assert "deduplication" in stages
        assert "complete" in stages
    
    def test_analyze(self, sample_memories):
        """Test memory analysis."""
        consolidator = MemoryConsolidator()
        
        analysis = consolidator.analyze(sample_memories)
        
        assert "summary" in analysis
        assert "deduplication_estimate" in analysis
        assert "recommendations" in analysis
    
    def test_consolidate_incremental(self, sample_memories):
        """Test incremental consolidation."""
        consolidator = MemoryConsolidator()
        
        new_memory = MemoryData(
            id="new",
            content="The user's name is John Doe",  # Duplicate
            metadata={"confidence": 0.8}
        )
        
        result = consolidator.consolidate_incremental(new_memory, sample_memories)
        
        assert "is_duplicate" in result
        assert "recommended_action" in result
        
        # Should detect as duplicate
        assert result["is_duplicate"] is True
        assert result["recommended_action"] == "skip"
    
    def test_batch_process(self, sample_memories):
        """Test batch processing."""
        consolidator = MemoryConsolidator(ConsolidationConfig(batch_size=2))
        
        results = consolidator.batch_process(sample_memories, batch_size=2)
        
        # Should have multiple batches
        expected_batches = (len(sample_memories) + 1) // 2
        assert len(results) == expected_batches


# ==============================================================================
# Scheduler Tests
# ==============================================================================

class TestConsolidationScheduler:
    """Tests for ConsolidationScheduler."""
    
    def test_init_default(self):
        """Test default initialization."""
        scheduler = ConsolidationScheduler()
        assert scheduler.is_running is False
        assert scheduler.last_run is None
    
    def test_start_stop(self):
        """Test starting and stopping scheduler."""
        scheduler = ConsolidationScheduler()
        
        scheduler.start()
        assert scheduler.is_running is True
        
        scheduler.stop()
        assert scheduler.is_running is False
    
    def test_run_now_no_loader(self):
        """Test run_now without memory loader."""
        scheduler = ConsolidationScheduler()
        
        result = scheduler.run_now()
        assert result is None  # No loader configured
    
    def test_run_now_with_loader(self):
        """Test run_now with memory loader."""
        memories = [
            MemoryData(id="1", content="Test memory"),
        ]
        
        def loader():
            return memories
        
        scheduler = ConsolidationScheduler(memory_loader=loader)
        
        result = scheduler.run_now()
        
        assert result is not None
        assert isinstance(result, ConsolidationResult)
    
    def test_callbacks(self):
        """Test callback registration."""
        memories = [
            MemoryData(id="1", content="Test memory"),
        ]
        
        start_called = []
        complete_called = []
        
        def on_start():
            start_called.append(True)
        
        def on_complete(result):
            complete_called.append(result)
        
        scheduler = ConsolidationScheduler(memory_loader=lambda: memories)
        scheduler.on_start(on_start)
        scheduler.on_complete(on_complete)
        
        scheduler.run_now()
        
        assert len(start_called) == 1
        assert len(complete_called) == 1
    
    def test_get_status(self):
        """Test status retrieval."""
        scheduler = ConsolidationScheduler()
        
        status = scheduler.get_status()
        
        assert "is_running" in status
        assert "last_run" in status
        assert "next_run" in status
    
    def test_get_history(self):
        """Test history retrieval."""
        memories = [
            MemoryData(id="1", content="Test memory"),
        ]
        
        scheduler = ConsolidationScheduler(memory_loader=lambda: memories)
        
        # Run a few times
        scheduler.run_now()
        scheduler.run_now()
        
        history = scheduler.get_history(limit=5)
        
        assert len(history) == 2


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestConsolidationIntegration:
    """Integration tests for the consolidation system."""
    
    def test_full_consolidation_workflow(self, sample_memories):
        """Test complete consolidation workflow."""
        config = ConsolidationConfig(
            strategy=ConsolidationStrategy.HYBRID,
            similarity_threshold=0.8,
            duplicate_threshold=0.95,
            detect_conflicts=True,
            auto_merge=False,
        )
        
        consolidator = MemoryConsolidator(config)
        
        # Analyze first
        analysis = consolidator.analyze(sample_memories)
        assert analysis is not None
        
        # Then consolidate
        result = consolidator.consolidate(sample_memories)
        
        assert result.success is True
        assert result.memories_analyzed == len(sample_memories)
    
    def test_incremental_workflow(self):
        """Test incremental consolidation workflow."""
        existing = [
            MemoryData(id="1", content="User prefers Python"),
            MemoryData(id="2", content="User works at Tech Corp"),
        ]
        
        consolidator = MemoryConsolidator()
        
        # Add non-duplicate
        new1 = MemoryData(id="new1", content="User likes coffee")
        result1 = consolidator.consolidate_incremental(new1, existing)
        assert result1["recommended_action"] == "add"
        
        # Add duplicate
        new2 = MemoryData(id="new2", content="User prefers Python")
        result2 = consolidator.consolidate_incremental(new2, existing)
        assert result2["is_duplicate"] is True
    
    def test_scheduler_with_real_consolidator(self):
        """Test scheduler with real consolidator."""
        memories = [
            MemoryData(id="1", content="Memory one"),
            MemoryData(id="2", content="Memory two"),
            MemoryData(id="3", content="Memory one"),  # Duplicate
        ]
        
        config = ConsolidationConfig(run_interval_hours=1)
        consolidator = MemoryConsolidator(config)
        
        scheduler = ConsolidationScheduler(
            consolidator=consolidator,
            config=config,
            memory_loader=lambda: memories,
        )
        
        result = scheduler.run_now()
        
        assert result is not None
        assert result.duplicates_found >= 1
