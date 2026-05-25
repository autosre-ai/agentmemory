"""Hybrid search combining BM25 lexical search with vector semantic search.

This module provides state-of-the-art hybrid search that combines:
- BM25 lexical/keyword matching (good for exact terms)
- Vector semantic similarity (good for concepts/meaning)
- Configurable fusion strategies (linear, RRF, learned)
- Score normalization and calibration
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from enum import Enum
from collections import Counter

logger = logging.getLogger(__name__)


# Feature availability
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


class FusionStrategy(Enum):
    """Strategies for combining lexical and semantic scores."""
    LINEAR = "linear"           # Weighted linear combination
    RRF = "rrf"                 # Reciprocal Rank Fusion
    DISTRIBUTION = "distribution"  # Distribution-based score calibration
    CONVEX = "convex"           # Convex combination with learned alpha
    MAX_SCORE = "max_score"     # Take maximum of normalized scores


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    
    # Weight allocation
    lexical_weight: float = 0.5
    semantic_weight: float = 0.5
    
    # Fusion settings
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    rrf_k: int = 60  # RRF smoothing parameter
    
    # Score normalization
    normalize_scores: bool = True
    normalization_method: str = "minmax"  # "minmax", "zscore", "percentile"
    
    # Candidate retrieval
    lexical_candidates: int = 100  # Number of candidates from lexical search
    semantic_candidates: int = 100  # Number of candidates from semantic search
    
    # BM25 parameters
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Length normalization
    
    # Result settings
    top_k: int = 10
    min_score: float = 0.0
    
    # Boosting
    boost_recency: bool = False
    recency_half_life_days: float = 30.0
    boost_confidence: bool = False
    
    def __post_init__(self):
        """Normalize weights to sum to 1."""
        total = self.lexical_weight + self.semantic_weight
        if total > 0:
            self.lexical_weight /= total
            self.semantic_weight /= total


@dataclass
class HybridMatch:
    """A hybrid search match with detailed scoring breakdown."""
    memory_id: str
    content: str
    combined_score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Additional boost factors
    recency_boost: float = 0.0
    confidence_boost: float = 0.0
    
    @property
    def source(self) -> str:
        """Identify primary source of match."""
        if self.lexical_rank is not None and self.semantic_rank is not None:
            return "hybrid"
        elif self.lexical_rank is not None:
            return "lexical"
        elif self.semantic_rank is not None:
            return "semantic"
        return "unknown"


class BM25:
    """
    BM25 implementation for lexical search.
    
    BM25 is a probabilistic retrieval function that ranks documents
    based on term frequency, inverse document frequency, and document length.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Document length normalization (0.75 typical)
            epsilon: Floor for IDF calculation
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Corpus statistics (computed on fit)
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_freqs: dict[str, int] = {}  # term -> doc frequency
        self.idf: dict[str, float] = {}
        self.doc_lens: list[int] = []
        self.doc_term_freqs: list[dict[str, int]] = []
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization with lowercasing."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def fit(self, documents: list[str]) -> "BM25":
        """
        Fit BM25 on a corpus of documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            Self for chaining
        """
        self.corpus_size = len(documents)
        self.doc_lens = []
        self.doc_term_freqs = []
        self.doc_freqs = Counter()
        
        # Process documents
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            
            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(dict(term_freqs))
            
            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] += 1
        
        # Compute average document length
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Compute IDF values
        self._compute_idf()
        
        return self
    
    def _compute_idf(self) -> None:
        """Compute IDF values for all terms."""
        self.idf = {}
        
        for term, freq in self.doc_freqs.items():
            # Standard IDF with log
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            # Apply epsilon floor
            self.idf[term] = max(self.epsilon, idf)
    
    def get_scores(self, query: str) -> list[float]:
        """
        Calculate BM25 scores for all documents.
        
        Args:
            query: Search query
            
        Returns:
            List of scores for each document in corpus
        """
        query_terms = self._tokenize(query)
        scores = []
        
        for i, doc_tf in enumerate(self.doc_term_freqs):
            score = 0.0
            doc_len = self.doc_lens[i]
            
            for term in query_terms:
                if term not in doc_tf:
                    continue
                
                tf = doc_tf[term]
                idf = self.idf.get(term, self.epsilon)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                
                score += idf * numerator / denominator
            
            scores.append(score)
        
        return scores
    
    def get_top_k(
        self, 
        query: str, 
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Get top-k documents by BM25 score.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (doc_index, score) tuples sorted by score descending
        """
        scores = self.get_scores(query)
        
        # Get top-k indices
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:k]


class ScoreNormalizer:
    """Normalize and calibrate search scores."""
    
    @staticmethod
    def minmax_normalize(scores: list[float]) -> list[float]:
        """Min-max normalization to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score
        
        if range_score == 0:
            return [0.5] * len(scores)
        
        return [(s - min_score) / range_score for s in scores]
    
    @staticmethod
    def zscore_normalize(scores: list[float]) -> list[float]:
        """Z-score normalization (standardization)."""
        if not scores or len(scores) < 2:
            return [0.5] * len(scores) if scores else []
        
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        
        if std == 0:
            return [0.5] * len(scores)
        
        # Z-score then sigmoid to bound to [0, 1]
        z_scores = [(s - mean) / std for s in scores]
        return [1 / (1 + math.exp(-z)) for z in z_scores]
    
    @staticmethod
    def percentile_normalize(scores: list[float]) -> list[float]:
        """Percentile rank normalization."""
        if not scores:
            return []
        
        n = len(scores)
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1])
        
        result = [0.0] * n
        for rank, (idx, _) in enumerate(sorted_scores):
            result[idx] = rank / (n - 1) if n > 1 else 0.5
        
        return result
    
    def normalize(
        self, 
        scores: list[float], 
        method: str = "minmax",
    ) -> list[float]:
        """Normalize scores using specified method."""
        if method == "minmax":
            return self.minmax_normalize(scores)
        elif method == "zscore":
            return self.zscore_normalize(scores)
        elif method == "percentile":
            return self.percentile_normalize(scores)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class ScoreFuser:
    """Fuse scores from multiple retrieval sources."""
    
    def __init__(self, config: HybridSearchConfig | None = None):
        self.config = config or HybridSearchConfig()
        self.normalizer = ScoreNormalizer()
    
    def fuse_linear(
        self,
        lexical_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Linear weighted combination of scores.
        
        Args:
            lexical_results: List of (id, score) from lexical search
            semantic_results: List of (id, score) from semantic search
            
        Returns:
            List of (id, combined_score) sorted by score descending
        """
        # Collect all unique IDs
        all_ids = set(id for id, _ in lexical_results)
        all_ids.update(id for id, _ in semantic_results)
        
        # Build score dicts
        lex_scores = {id: score for id, score in lexical_results}
        sem_scores = {id: score for id, score in semantic_results}
        
        # Normalize scores
        if self.config.normalize_scores:
            lex_values = list(lex_scores.values())
            sem_values = list(sem_scores.values())
            
            lex_norm = self.normalizer.normalize(lex_values, self.config.normalization_method)
            sem_norm = self.normalizer.normalize(sem_values, self.config.normalization_method)
            
            lex_scores = dict(zip(lex_scores.keys(), lex_norm))
            sem_scores = dict(zip(sem_scores.keys(), sem_norm))
        
        # Combine scores
        combined = []
        for id in all_ids:
            lex = lex_scores.get(id, 0.0)
            sem = sem_scores.get(id, 0.0)
            
            score = (
                lex * self.config.lexical_weight +
                sem * self.config.semantic_weight
            )
            combined.append((id, score))
        
        # Sort by score descending
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined
    
    def fuse_rrf(
        self,
        lexical_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion.
        
        RRF is rank-based and doesn't require score calibration.
        Score = sum(1 / (k + rank)) across all rankings.
        
        Args:
            lexical_results: List of (id, score) from lexical search
            semantic_results: List of (id, score) from semantic search
            
        Returns:
            List of (id, rrf_score) sorted by score descending
        """
        k = self.config.rrf_k
        
        # Get rankings
        lexical_ranks = {id: rank for rank, (id, _) in enumerate(lexical_results)}
        semantic_ranks = {id: rank for rank, (id, _) in enumerate(semantic_results)}
        
        # Collect all IDs
        all_ids = set(lexical_ranks.keys())
        all_ids.update(semantic_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = []
        for id in all_ids:
            score = 0.0
            
            if id in lexical_ranks:
                score += self.config.lexical_weight / (k + lexical_ranks[id] + 1)
            
            if id in semantic_ranks:
                score += self.config.semantic_weight / (k + semantic_ranks[id] + 1)
            
            rrf_scores.append((id, score))
        
        # Sort by score descending
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        
        return rrf_scores
    
    def fuse_distribution(
        self,
        lexical_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Distribution-based fusion with score calibration.
        
        Maps scores to a common distribution before combining.
        """
        # Normalize to percentile ranks
        lex_norm = self.normalizer.normalize(
            [s for _, s in lexical_results], 
            "percentile"
        )
        sem_norm = self.normalizer.normalize(
            [s for _, s in semantic_results], 
            "percentile"
        )
        
        # Build score dicts with normalized values
        lex_scores = {}
        for i, (id, _) in enumerate(lexical_results):
            lex_scores[id] = lex_norm[i]
        
        sem_scores = {}
        for i, (id, _) in enumerate(semantic_results):
            sem_scores[id] = sem_norm[i]
        
        # Combine using geometric mean for better calibration
        all_ids = set(lex_scores.keys())
        all_ids.update(sem_scores.keys())
        
        combined = []
        for id in all_ids:
            lex = lex_scores.get(id, 0.0)
            sem = sem_scores.get(id, 0.0)
            
            # Weighted geometric mean
            if lex > 0 and sem > 0:
                score = (
                    (lex ** self.config.lexical_weight) *
                    (sem ** self.config.semantic_weight)
                )
            else:
                # Fall back to weighted sum for zero values
                score = (
                    lex * self.config.lexical_weight +
                    sem * self.config.semantic_weight
                )
            
            combined.append((id, score))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined
    
    def fuse_max_score(
        self,
        lexical_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Take the maximum normalized score from either source.
        
        Good when you trust either signal equally.
        """
        # Normalize both
        lex_norm = self.normalizer.minmax_normalize([s for _, s in lexical_results])
        sem_norm = self.normalizer.minmax_normalize([s for _, s in semantic_results])
        
        lex_scores = {}
        for i, (id, _) in enumerate(lexical_results):
            lex_scores[id] = lex_norm[i]
        
        sem_scores = {}
        for i, (id, _) in enumerate(semantic_results):
            sem_scores[id] = sem_norm[i]
        
        all_ids = set(lex_scores.keys())
        all_ids.update(sem_scores.keys())
        
        combined = []
        for id in all_ids:
            lex = lex_scores.get(id, 0.0)
            sem = sem_scores.get(id, 0.0)
            score = max(lex, sem)
            combined.append((id, score))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined
    
    def fuse(
        self,
        lexical_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
        strategy: FusionStrategy | None = None,
    ) -> list[tuple[str, float]]:
        """
        Fuse results using the configured or specified strategy.
        
        Args:
            lexical_results: Results from lexical search
            semantic_results: Results from semantic search
            strategy: Override fusion strategy
            
        Returns:
            Fused results sorted by combined score
        """
        strategy = strategy or self.config.fusion_strategy
        
        if strategy == FusionStrategy.LINEAR:
            return self.fuse_linear(lexical_results, semantic_results)
        elif strategy == FusionStrategy.RRF:
            return self.fuse_rrf(lexical_results, semantic_results)
        elif strategy == FusionStrategy.DISTRIBUTION:
            return self.fuse_distribution(lexical_results, semantic_results)
        elif strategy == FusionStrategy.MAX_SCORE:
            return self.fuse_max_score(lexical_results, semantic_results)
        else:
            # Default to RRF
            return self.fuse_rrf(lexical_results, semantic_results)


class HybridSearchEngine:
    """
    Production-grade hybrid search combining lexical and semantic retrieval.
    
    This engine provides:
    - BM25 lexical search for exact term matching
    - Vector semantic search for meaning-based retrieval
    - Multiple fusion strategies (RRF, linear, distribution-based)
    - Score normalization and calibration
    - Configurable weighting between search types
    
    Example:
        >>> from agent_memory_toolkit.search import HybridSearchEngine
        >>> 
        >>> engine = HybridSearchEngine(
        ...     embedding_provider=provider,
        ...     config=HybridSearchConfig(
        ...         lexical_weight=0.3,
        ...         semantic_weight=0.7,
        ...         fusion_strategy=FusionStrategy.RRF,
        ...     )
        ... )
        >>> 
        >>> # Index documents
        >>> engine.index(documents)
        >>> 
        >>> # Search
        >>> results = engine.search("How do I configure authentication?")
    """
    
    def __init__(
        self,
        embedding_provider: Any = None,
        config: HybridSearchConfig | None = None,
        bm25: BM25 | None = None,
    ):
        """
        Initialize the hybrid search engine.
        
        Args:
            embedding_provider: Provider for generating embeddings
            config: Search configuration
            bm25: Pre-configured BM25 instance
        """
        self.embedding_provider = embedding_provider
        self.config = config or HybridSearchConfig()
        self.bm25 = bm25 or BM25(k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.fuser = ScoreFuser(self.config)
        
        # Document storage
        self.documents: list[dict[str, Any]] = []
        self.document_embeddings: list[list[float]] = []
        self.indexed = False
    
    def index(
        self,
        documents: list[dict[str, Any]],
        content_field: str = "content",
        embeddings: list[list[float]] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Index documents for hybrid search.
        
        Args:
            documents: List of documents to index
            content_field: Field containing document text
            embeddings: Pre-computed embeddings (optional)
            progress_callback: Called with (processed, total) for progress
        """
        self.documents = documents
        contents = [doc.get(content_field, "") for doc in documents]
        
        # Fit BM25
        self.bm25.fit(contents)
        
        # Generate embeddings
        if embeddings is not None:
            self.document_embeddings = embeddings
        elif self.embedding_provider is not None:
            self.document_embeddings = self.embedding_provider.encode(contents)
        else:
            self.document_embeddings = []
        
        self.indexed = True
        
        if progress_callback:
            progress_callback(len(documents), len(documents))
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        content_field: str = "content",
        id_field: str = "id",
        metadata_field: str = "metadata",
        lexical_weight: float | None = None,
        semantic_weight: float | None = None,
        fusion_strategy: FusionStrategy | None = None,
    ) -> list[HybridMatch]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            content_field: Field containing document text
            id_field: Field containing document ID
            metadata_field: Field containing metadata
            lexical_weight: Override lexical weight
            semantic_weight: Override semantic weight
            fusion_strategy: Override fusion strategy
            
        Returns:
            List of HybridMatch results sorted by combined score
        """
        if not self.indexed:
            raise RuntimeError("Must call index() before search()")
        
        top_k = top_k or self.config.top_k
        
        # Override weights if provided
        effective_lex_weight = lexical_weight if lexical_weight is not None else self.config.lexical_weight
        effective_sem_weight = semantic_weight if semantic_weight is not None else self.config.semantic_weight
        
        # Normalize weights
        total = effective_lex_weight + effective_sem_weight
        if total > 0:
            effective_lex_weight /= total
            effective_sem_weight /= total
        
        # Get lexical results
        lexical_results = self._search_lexical(
            query, 
            top_k=self.config.lexical_candidates,
            id_field=id_field,
        )
        
        # Get semantic results
        semantic_results = self._search_semantic(
            query,
            top_k=self.config.semantic_candidates,
            id_field=id_field,
        )
        
        # Create temporary config with effective weights
        temp_config = HybridSearchConfig(
            lexical_weight=effective_lex_weight,
            semantic_weight=effective_sem_weight,
            fusion_strategy=fusion_strategy or self.config.fusion_strategy,
            rrf_k=self.config.rrf_k,
            normalize_scores=self.config.normalize_scores,
            normalization_method=self.config.normalization_method,
        )
        temp_fuser = ScoreFuser(temp_config)
        
        # Fuse results
        fused = temp_fuser.fuse(lexical_results, semantic_results)
        
        # Build rank lookups
        lexical_ranks = {id: rank for rank, (id, _) in enumerate(lexical_results)}
        semantic_ranks = {id: rank for rank, (id, _) in enumerate(semantic_results)}
        
        lexical_scores = {id: score for id, score in lexical_results}
        semantic_scores = {id: score for id, score in semantic_results}
        
        # Build document lookup
        doc_lookup = {doc.get(id_field, str(i)): doc for i, doc in enumerate(self.documents)}
        
        # Build results
        results = []
        for doc_id, combined_score in fused[:top_k]:
            doc = doc_lookup.get(doc_id)
            if doc is None:
                continue
            
            match = HybridMatch(
                memory_id=doc_id,
                content=doc.get(content_field, ""),
                combined_score=combined_score,
                lexical_score=lexical_scores.get(doc_id, 0.0),
                semantic_score=semantic_scores.get(doc_id, 0.0),
                lexical_rank=lexical_ranks.get(doc_id),
                semantic_rank=semantic_ranks.get(doc_id),
                metadata=doc.get(metadata_field, {}),
            )
            
            # Apply boosts
            if self.config.boost_recency:
                match.recency_boost = self._compute_recency_boost(doc)
                match.combined_score += match.recency_boost
            
            if self.config.boost_confidence:
                match.confidence_boost = self._compute_confidence_boost(doc, metadata_field)
                match.combined_score += match.confidence_boost
            
            if match.combined_score >= self.config.min_score:
                results.append(match)
        
        # Re-sort if boosts were applied
        if self.config.boost_recency or self.config.boost_confidence:
            results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results[:top_k]
    
    def _search_lexical(
        self,
        query: str,
        top_k: int,
        id_field: str,
    ) -> list[tuple[str, float]]:
        """Perform BM25 lexical search."""
        top_results = self.bm25.get_top_k(query, k=top_k)
        
        results = []
        for idx, score in top_results:
            doc = self.documents[idx]
            doc_id = doc.get(id_field, str(idx))
            results.append((doc_id, score))
        
        return results
    
    def _search_semantic(
        self,
        query: str,
        top_k: int,
        id_field: str,
    ) -> list[tuple[str, float]]:
        """Perform semantic vector search."""
        if not self.document_embeddings or self.embedding_provider is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_provider.encode([query])[0]
        
        # Compute similarities
        similarities = self._compute_similarities(query_embedding, self.document_embeddings)
        
        # Get top-k
        indexed_scores = list(enumerate(similarities))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:top_k]:
            doc = self.documents[idx]
            doc_id = doc.get(id_field, str(idx))
            results.append((doc_id, score))
        
        return results
    
    def _compute_similarities(
        self, 
        query_embedding: list[float],
        doc_embeddings: list[list[float]],
    ) -> list[float]:
        """Compute cosine similarities."""
        if NUMPY_AVAILABLE:
            q = np.array(query_embedding, dtype=np.float32)  # type: ignore
            docs = np.array(doc_embeddings, dtype=np.float32)  # type: ignore
            
            # Normalize
            q_norm = q / (np.linalg.norm(q) + 1e-10)  # type: ignore
            doc_norms = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-10)  # type: ignore
            
            similarities = np.dot(doc_norms, q_norm)  # type: ignore
            return similarities.tolist()
        else:
            # Pure Python fallback
            results = []
            for doc_emb in doc_embeddings:
                dot = sum(a * b for a, b in zip(query_embedding, doc_emb))
                norm_q = sum(a * a for a in query_embedding) ** 0.5
                norm_d = sum(b * b for b in doc_emb) ** 0.5
                
                if norm_q == 0 or norm_d == 0:
                    sim = 0.0
                else:
                    sim = dot / (norm_q * norm_d)
                
                results.append(sim)
            return results
    
    def _compute_recency_boost(self, doc: dict[str, Any]) -> float:
        """Compute recency boost for a document."""
        from datetime import datetime, timezone
        
        created_at = doc.get("created_at")
        if created_at is None:
            return 0.0
        
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                return 0.0
        
        now = datetime.now(timezone.utc) if created_at.tzinfo else datetime.utcnow()
        age_days = (now - created_at).days
        
        # Exponential decay with half-life
        decay = 0.5 ** (age_days / self.config.recency_half_life_days)
        
        return decay * 0.1  # Max 10% boost for very recent
    
    def _compute_confidence_boost(
        self, 
        doc: dict[str, Any],
        metadata_field: str,
    ) -> float:
        """Compute confidence boost from metadata."""
        metadata = doc.get(metadata_field, {})
        confidence = metadata.get("confidence", 1.0) if isinstance(metadata, dict) else 1.0
        
        # Scale confidence to a small boost
        return (confidence - 0.5) * 0.1  # -5% to +5%


def create_hybrid_engine(
    documents: list[dict[str, Any]],
    embedding_provider: Any = None,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
    fusion_strategy: FusionStrategy = FusionStrategy.RRF,
    content_field: str = "content",
) -> HybridSearchEngine:
    """
    Factory function to create and index a hybrid search engine.
    
    Args:
        documents: Documents to index
        embedding_provider: Provider for embeddings
        lexical_weight: Weight for BM25 scores
        semantic_weight: Weight for semantic scores
        fusion_strategy: Strategy for combining scores
        content_field: Field containing document text
        
    Returns:
        Indexed HybridSearchEngine ready for search
    """
    config = HybridSearchConfig(
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
        fusion_strategy=fusion_strategy,
    )
    
    engine = HybridSearchEngine(
        embedding_provider=embedding_provider,
        config=config,
    )
    
    engine.index(documents, content_field=content_field)
    
    return engine
