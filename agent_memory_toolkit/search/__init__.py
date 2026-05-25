"""Advanced search capabilities for Agent Memory Toolkit.

This module provides production-grade search functionality including:
- Semantic search with embeddings and query expansion
- Hybrid BM25 + vector search with configurable fusion
- Faceted search with multi-dimensional filtering
- Re-ranking with cross-encoders and diversity (MMR)

Quick Start:
    >>> from agent_memory_toolkit.search import (
    ...     SemanticSearchEngine,
    ...     HybridSearchEngine,
    ...     FacetedSearchEngine,
    ...     FilterBuilder,
    ... )
    >>> 
    >>> # Semantic search
    >>> semantic = SemanticSearchEngine(embedding_provider=provider)
    >>> results = semantic.search("authentication issues", documents)
    >>> 
    >>> # Hybrid search (BM25 + vector)
    >>> hybrid = HybridSearchEngine(embedding_provider=provider)
    >>> hybrid.index(documents)
    >>> results = hybrid.search("reset password", top_k=10)
    >>> 
    >>> # Faceted search with filters
    >>> filters = (FilterBuilder()
    ...     .has_tag("security")
    ...     .min_confidence(0.8)
    ...     .created_after("2024-01-01")
    ...     .build()
    ... )
    >>> faceted = FacetedSearchEngine()
    >>> results = faceted.search(documents, filters=filters)
    >>> 
    >>> # Re-ranking for improved precision
    >>> from agent_memory_toolkit.search import CrossEncoderReranker
    >>> reranker = CrossEncoderReranker()
    >>> reranked = reranker.rerank(query, [r.content for r in results])
"""

from .semantic import (
    SemanticSearchEngine,
    SemanticSearchConfig,
    SemanticMatch,
    QueryAnalyzer,
    QueryAnalysis,
    QueryType,
    QueryExpander,
    SynonymQueryExpander,
    LLMQueryExpander,
    EmbeddingCache,
    MultiVectorSearch,
)

from .hybrid import (
    HybridSearchEngine,
    HybridSearchConfig,
    HybridMatch,
    FusionStrategy,
    BM25,
    ScoreNormalizer,
    ScoreFuser,
    create_hybrid_engine,
)

from .faceted import (
    FacetedSearchEngine,
    FacetedSearchConfig,
    FacetedSearchResult,
    FilterBuilder,
    FilterGroup,
    FilterCondition,
    FilterOperator,
    BooleanOperator,
    Facet,
    FacetValue,
    FacetExtractor,
    filter_documents,
    build_filter,
)

from .ranking import (
    CrossEncoderReranker,
    RRFFusion,
    MMRDiversifier,
    ScoreCalibrator,
    RankingEvaluator,
    RerankingConfig,
    RankedItem,
    RankingMetric,
    rerank_with_cross_encoder,
    fuse_rankings,
    diversify_results,
)

__all__ = [
    # Semantic search
    "SemanticSearchEngine",
    "SemanticSearchConfig",
    "SemanticMatch",
    "QueryAnalyzer",
    "QueryAnalysis",
    "QueryType",
    "QueryExpander",
    "SynonymQueryExpander",
    "LLMQueryExpander",
    "EmbeddingCache",
    "MultiVectorSearch",
    
    # Hybrid search
    "HybridSearchEngine",
    "HybridSearchConfig",
    "HybridMatch",
    "FusionStrategy",
    "BM25",
    "ScoreNormalizer",
    "ScoreFuser",
    "create_hybrid_engine",
    
    # Faceted search
    "FacetedSearchEngine",
    "FacetedSearchConfig",
    "FacetedSearchResult",
    "FilterBuilder",
    "FilterGroup",
    "FilterCondition",
    "FilterOperator",
    "BooleanOperator",
    "Facet",
    "FacetValue",
    "FacetExtractor",
    "filter_documents",
    "build_filter",
    
    # Ranking
    "CrossEncoderReranker",
    "RRFFusion",
    "MMRDiversifier",
    "ScoreCalibrator",
    "RankingEvaluator",
    "RerankingConfig",
    "RankedItem",
    "RankingMetric",
    "rerank_with_cross_encoder",
    "fuse_rankings",
    "diversify_results",
]
