"""Comprehensive tests for enhanced semantic search in the memory store."""

import pytest
import math
from typing import List
from unittest.mock import Mock, patch, MagicMock

from agent_memory_toolkit.store import (
    MemoryStore,
    Memory,
    MemoryMetadata,
    SearchResult,
    HybridSearchConfig,
    SearchMethod,
    EmbeddingConfig,
    RerankerConfig,
    SimilarityMetric,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    CROSS_ENCODER_AVAILABLE,
    embedding_to_blob,
    blob_to_embedding,
    cosine_similarity,
    batch_cosine_similarity,
)
from agent_memory_toolkit.store.embeddings import (
    EmbeddingProvider,
    RerankerProvider,
    normalize_embedding,
    dot_product_similarity,
    euclidean_similarity,
    euclidean_distance,
)


# ==================== Mock Providers ====================


class MockEmbeddingProvider:
    """Mock embedding provider for testing without sentence-transformers."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Create deterministic embeddings based on text content
            base = hash(text) % 1000 / 1000.0
            embedding = [
                math.sin(base * (i + 1)) * 0.5 + 0.5
                for i in range(self._dimension)
            ]
            # Normalize
            norm = sum(x * x for x in embedding) ** 0.5
            embedding = [x / norm for x in embedding]
            embeddings.append(embedding)
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


class MockReranker:
    """Mock reranker for testing without cross-encoder."""
    
    def rerank(
        self, 
        query: str, 
        documents: List[str],
        top_k: int | None = None,
    ) -> List[tuple[int, float]]:
        """Rerank based on simple word overlap."""
        query_words = set(query.lower().split())
        
        scores = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            scores = scores[:top_k]
        
        return scores


# ==================== Fixtures ====================


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider."""
    return MockEmbeddingProvider(dimension=384)


@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""
    return MockReranker()


@pytest.fixture
def store():
    """Create an in-memory store without embeddings."""
    return MemoryStore(":memory:")


@pytest.fixture
def store_with_embeddings(mock_embedding_provider):
    """Create an in-memory store with mock embedding provider."""
    return MemoryStore(
        ":memory:",
        embedding_provider=mock_embedding_provider,
        auto_embed=True,
    )


@pytest.fixture
def store_with_reranker(mock_embedding_provider, mock_reranker):
    """Create an in-memory store with embeddings and reranker."""
    return MemoryStore(
        ":memory:",
        embedding_provider=mock_embedding_provider,
        reranker=mock_reranker,
        auto_embed=True,
    )


@pytest.fixture
def populated_store(store_with_embeddings):
    """Create a store with test data."""
    store = store_with_embeddings
    
    # Add sample memories
    memories = [
        ("The capital of France is Paris.", {"source": "geography"}),
        ("Python is a popular programming language.", {"source": "tech"}),
        ("Machine learning is a subset of AI.", {"source": "tech"}),
        ("The Eiffel Tower is in Paris, France.", {"source": "geography"}),
        ("Natural language processing enables computers to understand text.", {"source": "tech"}),
        ("Berlin is the capital of Germany.", {"source": "geography"}),
        ("JavaScript is used for web development.", {"source": "tech"}),
        ("Tokyo is the capital of Japan.", {"source": "geography"}),
    ]
    
    for content, meta_dict in memories:
        store.add(content, metadata=MemoryMetadata(source=meta_dict["source"]))
    
    return store


# ==================== Embedding Utility Tests ====================


class TestEmbeddingUtilities:
    """Test embedding utility functions."""
    
    def test_embedding_to_blob_and_back(self):
        """Test round-trip serialization of embeddings."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = embedding_to_blob(embedding)
        result = blob_to_embedding(blob)
        
        assert len(result) == len(embedding)
        for a, b in zip(result, embedding):
            assert abs(a - b) < 1e-6
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = [1.0, 0.0, 0.0, 0.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert abs(cosine_similarity(vec1, vec2)) < 1e-6
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        assert abs(cosine_similarity(vec1, vec2) - (-1.0)) < 1e-6
    
    def test_cosine_similarity_mismatched_dimensions(self):
        """Test that mismatched dimensions raise error."""
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError):
            cosine_similarity(vec1, vec2)
    
    def test_batch_cosine_similarity(self):
        """Test batch cosine similarity computation."""
        query = [1.0, 0.0, 0.0]
        embeddings = [
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [0.707, 0.707, 0.0],  # 45 degrees
        ]
        
        similarities = batch_cosine_similarity(query, embeddings)
        
        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-5
        assert abs(similarities[1]) < 1e-5
        assert abs(similarities[2] - 0.707) < 0.01
    
    def test_normalize_embedding(self):
        """Test embedding normalization."""
        embedding = [3.0, 4.0]
        normalized = normalize_embedding(embedding)
        
        norm = sum(x * x for x in normalized) ** 0.5
        assert abs(norm - 1.0) < 1e-6
        
        # Check direction preserved
        assert normalized[0] / normalized[1] == pytest.approx(3.0 / 4.0)
    
    def test_dot_product_similarity(self):
        """Test dot product similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]
        
        result = dot_product_similarity(vec1, vec2)
        expected = 1*4 + 2*5 + 3*6  # 32
        
        assert result == expected
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        vec1 = [0.0, 0.0]
        vec2 = [3.0, 4.0]
        
        result = euclidean_distance(vec1, vec2)
        assert result == 5.0
    
    def test_euclidean_similarity(self):
        """Test Euclidean similarity (distance to similarity conversion)."""
        vec1 = [0.0, 0.0]
        vec2 = [0.0, 0.0]
        
        # Identical vectors have similarity 1
        assert euclidean_similarity(vec1, vec2) == 1.0
        
        # Distant vectors have lower similarity
        vec3 = [3.0, 4.0]
        sim = euclidean_similarity(vec1, vec3)
        assert 0 < sim < 1


# ==================== Mock Provider Tests ====================


class TestMockProviders:
    """Test that mock providers work correctly."""
    
    def test_mock_embedding_provider(self, mock_embedding_provider):
        """Test mock embedding provider generates consistent embeddings."""
        texts = ["hello world", "goodbye moon"]
        
        embeddings1 = mock_embedding_provider.encode(texts)
        embeddings2 = mock_embedding_provider.encode(texts)
        
        # Should be deterministic
        assert embeddings1 == embeddings2
        
        # Should have correct dimension
        assert len(embeddings1[0]) == 384
        
        # Should be normalized (unit length)
        for emb in embeddings1:
            norm = sum(x * x for x in emb) ** 0.5
            assert abs(norm - 1.0) < 1e-6
    
    def test_mock_reranker(self, mock_reranker):
        """Test mock reranker produces expected rankings."""
        query = "capital of France"
        docs = [
            "Python programming",
            "The capital of France is Paris",
            "Berlin is a city",
        ]
        
        results = mock_reranker.rerank(query, docs)
        
        # "capital of France" should rank higher than others
        assert results[0][0] == 1  # Second doc has most overlap
        assert len(results) == 3


# ==================== FTS Search Tests ====================


class TestFTSSearch:
    """Test full-text search functionality."""
    
    def test_basic_fts_search(self, populated_store):
        """Test basic FTS search."""
        results = populated_store.search_fts("capital France")
        
        assert len(results) > 0
        assert any("France" in r.memory.content for r in results)
        assert all(r.match_type == "fts" for r in results)
    
    def test_fts_search_limit(self, populated_store):
        """Test FTS search respects limit."""
        results = populated_store.search_fts("capital", limit=2)
        assert len(results) <= 2
    
    def test_fts_search_empty_results(self, populated_store):
        """Test FTS search with no matches."""
        results = populated_store.search_fts("xyznonexistent")
        assert len(results) == 0
    
    def test_fts_search_boost_recent(self, store_with_embeddings):
        """Test FTS search with recency boost."""
        # Add some memories
        store_with_embeddings.add("Old test content")
        store_with_embeddings.add("New test content")
        
        # Search with boost
        results = store_with_embeddings.search_fts(
            "test content", 
            boost_recent=True
        )
        
        assert len(results) >= 2
        # Newer should have higher score due to boost
    
    def test_fts_search_boost_confidence(self, store_with_embeddings):
        """Test FTS search with confidence boost."""
        store_with_embeddings.add(
            "High confidence content",
            metadata=MemoryMetadata(confidence=1.0)
        )
        store_with_embeddings.add(
            "Low confidence content",
            metadata=MemoryMetadata(confidence=0.1)
        )
        
        results = store_with_embeddings.search_fts(
            "confidence content",
            boost_confidence=True
        )
        
        assert len(results) >= 2
        # Higher confidence should rank higher


# ==================== Vector Search Tests ====================


class TestVectorSearch:
    """Test vector similarity search functionality."""
    
    def test_basic_vector_search(self, populated_store):
        """Test basic vector search."""
        results = populated_store.search_vector("What is the capital of France?")
        
        assert len(results) > 0
        assert all(r.match_type == "vector" for r in results)
        assert all(0 <= r.score <= 1 for r in results)
    
    def test_vector_search_with_embedding(self, populated_store):
        """Test vector search with pre-computed embedding."""
        # Get an embedding for a query
        embedding = populated_store._embedding_provider.encode(["France capital"])[0]
        
        results = populated_store.search_vector(embedding)
        
        assert len(results) > 0
    
    def test_vector_search_limit(self, populated_store):
        """Test vector search respects limit."""
        results = populated_store.search_vector("programming", limit=3)
        assert len(results) <= 3
    
    def test_vector_search_without_provider(self, store):
        """Test vector search fails without embedding provider."""
        from agent_memory_toolkit.store.exceptions import MemoryStoreError
        
        with pytest.raises(MemoryStoreError):
            store.search_vector("test query")
    
    def test_vector_search_different_metrics(self, populated_store):
        """Test vector search with different similarity metrics."""
        query = "programming language"
        
        # Cosine similarity
        results_cosine = populated_store.search_vector(
            query, 
            similarity_metric=SimilarityMetric.COSINE
        )
        
        # Dot product
        results_dot = populated_store.search_vector(
            query,
            similarity_metric=SimilarityMetric.DOT_PRODUCT
        )
        
        # Euclidean
        results_euclidean = populated_store.search_vector(
            query,
            similarity_metric=SimilarityMetric.EUCLIDEAN
        )
        
        # All should return results
        assert len(results_cosine) > 0
        assert len(results_dot) > 0
        assert len(results_euclidean) > 0


# ==================== Hybrid Search Tests ====================


class TestHybridSearch:
    """Test hybrid search combining FTS and vector similarity."""
    
    def test_basic_hybrid_search(self, populated_store):
        """Test basic hybrid search."""
        results = populated_store.search_hybrid("capital of France")
        
        assert len(results) > 0
        assert all(r.match_type == "hybrid" for r in results)
    
    def test_hybrid_search_weights(self, populated_store):
        """Test hybrid search with custom weights."""
        # FTS-heavy search
        results_fts = populated_store.search_hybrid(
            "capital",
            fts_weight=0.9,
            vector_weight=0.1
        )
        
        # Vector-heavy search
        results_vec = populated_store.search_hybrid(
            "capital",
            fts_weight=0.1,
            vector_weight=0.9
        )
        
        # Both should return results
        assert len(results_fts) > 0
        assert len(results_vec) > 0
    
    def test_hybrid_search_config(self, populated_store):
        """Test hybrid search with HybridSearchConfig."""
        config = HybridSearchConfig(
            fts_weight=0.7,
            vector_weight=0.3,
            normalize_scores=True,
            min_score_threshold=0.0
        )
        
        results = populated_store.search_hybrid(
            "programming language",
            config=config
        )
        
        assert len(results) > 0
    
    def test_hybrid_search_with_reranking(self, store_with_reranker):
        """Test hybrid search with cross-encoder reranking."""
        # Add test data
        store_with_reranker.add("The capital of France is Paris.")
        store_with_reranker.add("Paris is a beautiful city in France.")
        store_with_reranker.add("Germany has many cities.")
        
        results = store_with_reranker.search_hybrid(
            "capital France",
            rerank=True
        )
        
        assert len(results) > 0
        assert any("reranked" in r.match_type for r in results)
    
    def test_hybrid_search_limit(self, populated_store):
        """Test hybrid search respects limit."""
        results = populated_store.search_hybrid("capital", limit=2)
        assert len(results) <= 2


# ==================== Main Search Method Tests ====================


class TestSearchMethod:
    """Test the main search method with automatic method selection."""
    
    def test_search_auto_with_embeddings(self, populated_store):
        """Test auto search uses hybrid when embeddings available."""
        results = populated_store.search("capital France", method="auto")
        
        assert len(results) > 0
        # Should use hybrid when embeddings are available
        assert all(r.match_type in ["hybrid", "hybrid_reranked"] for r in results)
    
    def test_search_auto_without_embeddings(self, store):
        """Test auto search uses FTS when no embeddings."""
        store.add("The capital of France is Paris.")
        
        results = store.search("capital France", method="auto")
        
        assert len(results) > 0
        assert all(r.match_type == "fts" for r in results)
    
    def test_search_explicit_method(self, populated_store):
        """Test search with explicit method selection."""
        # FTS
        results_fts = populated_store.search("capital", method="fts")
        assert all(r.match_type == "fts" for r in results_fts)
        
        # Vector
        results_vec = populated_store.search("capital", method="vector")
        assert all(r.match_type == "vector" for r in results_vec)
        
        # Hybrid
        results_hybrid = populated_store.search("capital", method="hybrid")
        assert all(r.match_type.startswith("hybrid") for r in results_hybrid)
    
    def test_search_method_enum(self, populated_store):
        """Test search with SearchMethod enum."""
        results = populated_store.search(
            "programming",
            method=SearchMethod.HYBRID
        )
        
        assert len(results) > 0
    
    def test_search_with_reranking(self, store_with_reranker):
        """Test search with reranking flag."""
        store_with_reranker.add("Python programming language")
        store_with_reranker.add("JavaScript for web")
        
        results = store_with_reranker.search(
            "programming language",
            rerank=True
        )
        
        assert len(results) > 0


# ==================== Batch Embedding Tests ====================


class TestBatchEmbedding:
    """Test batch embedding functionality."""
    
    def test_embed_all(self, mock_embedding_provider):
        """Test embed_all function."""
        store = MemoryStore(
            ":memory:",
            embedding_provider=mock_embedding_provider,
            auto_embed=False,  # Don't auto-embed
        )
        
        # Add memories without embeddings
        store.add("First memory")
        store.add("Second memory")
        store.add("Third memory")
        
        # Embed all
        count = store.embed_all()
        
        assert count == 3
        
        # Verify embeddings exist
        memories = store.list()
        assert all(m.embedding is not None for m in memories)
    
    def test_embed_all_with_progress(self, mock_embedding_provider):
        """Test embed_all with progress callback."""
        store = MemoryStore(
            ":memory:",
            embedding_provider=mock_embedding_provider,
            auto_embed=False,
        )
        
        # Add memories
        for i in range(5):
            store.add(f"Memory number {i}")
        
        # Track progress
        progress = []
        
        def callback(processed, total):
            progress.append((processed, total))
        
        store.embed_all(batch_size=2, progress_callback=callback)
        
        # Should have progress updates
        assert len(progress) > 0
        assert progress[-1][0] == progress[-1][1]  # Final: processed == total
    
    def test_embed_all_skip_existing(self, mock_embedding_provider):
        """Test embed_all skips existing embeddings."""
        store = MemoryStore(
            ":memory:",
            embedding_provider=mock_embedding_provider,
            auto_embed=True,  # Auto-embed first memory
        )
        
        store.add("First memory")  # Gets embedded
        
        # Turn off auto_embed and add more
        store._auto_embed = False
        store.add("Second memory")  # No embedding
        store.add("Third memory")  # No embedding
        
        # Embed only missing
        count = store.embed_all()
        
        assert count == 2  # Only the two without embeddings
    
    def test_embed_all_overwrite(self, mock_embedding_provider):
        """Test embed_all with overwrite flag."""
        store = MemoryStore(
            ":memory:",
            embedding_provider=mock_embedding_provider,
            auto_embed=True,
        )
        
        store.add("First memory")
        store.add("Second memory")
        
        # Embed all with overwrite
        count = store.embed_all(overwrite=True)
        
        assert count == 2  # All memories re-embedded


# ==================== Configuration Tests ====================


class TestConfigurations:
    """Test configuration classes."""
    
    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        config = EmbeddingConfig()
        
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.normalize == True
    
    def test_embedding_config_presets(self):
        """Test EmbeddingConfig preset configurations."""
        fast = EmbeddingConfig.fast()
        quality = EmbeddingConfig.quality()
        qa = EmbeddingConfig.qa_optimized()
        
        assert fast.batch_size == 64
        assert "mpnet" in quality.model_name
        assert "qa" in qa.model_name
    
    def test_reranker_config_defaults(self):
        """Test RerankerConfig default values."""
        config = RerankerConfig()
        
        assert "ms-marco" in config.model_name
        assert config.batch_size == 16
    
    def test_hybrid_search_config_normalization(self):
        """Test HybridSearchConfig weight normalization."""
        config = HybridSearchConfig(fts_weight=3, vector_weight=1)
        
        # Weights should be normalized
        assert config.fts_weight + config.vector_weight == pytest.approx(1.0)
        assert config.fts_weight == pytest.approx(0.75)
        assert config.vector_weight == pytest.approx(0.25)


# ==================== Property Tests ====================


class TestStoreProperties:
    """Test MemoryStore property methods."""
    
    def test_has_embeddings(self, store, store_with_embeddings):
        """Test has_embeddings property."""
        assert store.has_embeddings == False
        assert store_with_embeddings.has_embeddings == True
    
    def test_has_reranker(self, store_with_embeddings, store_with_reranker):
        """Test has_reranker property."""
        assert store_with_embeddings.has_reranker == False
        assert store_with_reranker.has_reranker == True
    
    def test_embedding_dimension(self, store, store_with_embeddings):
        """Test embedding_dimension property."""
        assert store.embedding_dimension is None
        assert store_with_embeddings.embedding_dimension == 384


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for the complete search pipeline."""
    
    def test_full_search_pipeline(self, mock_embedding_provider, mock_reranker):
        """Test complete search pipeline from add to search."""
        store = MemoryStore(
            ":memory:",
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
            auto_embed=True,
        )
        
        # Add memories
        store.add(
            "Paris is the capital of France.",
            metadata=MemoryMetadata(source="geography", confidence=0.95)
        )
        store.add(
            "The Eiffel Tower is located in Paris.",
            metadata=MemoryMetadata(source="geography", confidence=0.9)
        )
        store.add(
            "Python is a programming language.",
            metadata=MemoryMetadata(source="tech", confidence=0.85)
        )
        
        # Search with different methods
        fts_results = store.search("capital Paris", method="fts")
        vector_results = store.search("What is the capital of France?", method="vector")
        hybrid_results = store.search("France capital city", method="hybrid")
        reranked_results = store.search("capital of France", method="hybrid", rerank=True)
        
        # Verify all methods work
        assert len(fts_results) > 0
        assert len(vector_results) > 0
        assert len(hybrid_results) > 0
        assert len(reranked_results) > 0
        
        # Verify result types
        assert fts_results[0].match_type == "fts"
        assert vector_results[0].match_type == "vector"
        assert hybrid_results[0].match_type.startswith("hybrid")
    
    def test_search_accuracy(self, populated_store):
        """Test that search returns relevant results."""
        # Search for programming-related content
        results = populated_store.search("programming language Python")
        
        # Should find Python-related memories
        contents = [r.memory.content.lower() for r in results]
        assert any("python" in c for c in contents)
        
        # Search for geography
        results = populated_store.search("capital city Germany")
        contents = [r.memory.content.lower() for r in results]
        assert any("germany" in c or "capital" in c for c in contents)


# ==================== Conditional Tests (require sentence-transformers) ====================


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE, 
    reason="sentence-transformers not installed"
)
class TestSentenceTransformers:
    """Tests that require sentence-transformers to be installed."""
    
    def test_real_embedding_provider(self):
        """Test with real SentenceTransformer model."""
        from agent_memory_toolkit.store import SentenceTransformerProvider
        
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        
        embeddings = provider.encode(["Hello world", "Goodbye world"])
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        
        # Check normalization
        for emb in embeddings:
            norm = sum(x * x for x in emb) ** 0.5
            assert abs(norm - 1.0) < 0.01
    
    def test_store_with_real_embeddings(self):
        """Test MemoryStore with real embeddings."""
        from agent_memory_toolkit.store import SentenceTransformerProvider
        
        provider = SentenceTransformerProvider()
        store = MemoryStore(
            ":memory:",
            embedding_provider=provider,
            auto_embed=True,
        )
        
        store.add("The quick brown fox jumps over the lazy dog.")
        store.add("A fast auburn fox leaps above a sleepy canine.")
        store.add("Python programming is fun.")
        
        # Vector search should find semantically similar content
        results = store.search_vector("rapid fox jumping")
        
        assert len(results) > 0
        # Both fox sentences should rank higher than Python
        fox_count = sum(1 for r in results[:2] if "fox" in r.memory.content.lower())
        assert fox_count >= 1


@pytest.mark.skipif(
    not CROSS_ENCODER_AVAILABLE,
    reason="CrossEncoder not installed"
)
class TestCrossEncoder:
    """Tests that require CrossEncoder to be installed."""
    
    def test_real_reranker(self):
        """Test with real CrossEncoder model."""
        from agent_memory_toolkit.store import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        
        results = reranker.rerank(
            "What is the capital of France?",
            [
                "Berlin is in Germany",
                "Paris is the capital of France",
                "Python is a programming language",
            ]
        )
        
        # Paris should rank first
        assert results[0][0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
