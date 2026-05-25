"""Advanced ranking and score fusion for search results.

This module provides:
- Cross-encoder re-ranking for improved precision
- Reciprocal Rank Fusion (RRF) for combining multiple rankings
- Learning-to-Rank features
- Score calibration and normalization
- Diversity-aware re-ranking (MMR)
- Contextual re-ranking with query understanding
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar, Generic
from enum import Enum

logger = logging.getLogger(__name__)

# Feature availability
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


T = TypeVar("T")


class RankingMetric(Enum):
    """Metrics for evaluating ranking quality."""
    NDCG = "ndcg"         # Normalized Discounted Cumulative Gain
    MRR = "mrr"           # Mean Reciprocal Rank
    MAP = "map"           # Mean Average Precision
    RECALL_AT_K = "recall_at_k"
    PRECISION_AT_K = "precision_at_k"


@dataclass
class RankedItem(Generic[T]):
    """An item with ranking information."""
    item: T
    score: float
    rank: int
    original_rank: int | None = None
    
    # Feature breakdown
    features: dict[str, float] = field(default_factory=dict)
    
    # Diversity metrics
    diversity_penalty: float = 0.0
    
    @property
    def final_score(self) -> float:
        """Get final score after diversity penalty."""
        return self.score - self.diversity_penalty


@dataclass 
class RerankingConfig:
    """Configuration for re-ranking operations."""
    
    # Cross-encoder settings
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_batch_size: int = 32
    
    # RRF settings
    rrf_k: int = 60  # Smoothing parameter
    
    # Diversity settings
    enable_mmr: bool = False
    mmr_lambda: float = 0.7  # Balance relevance vs diversity
    mmr_top_k: int = 20  # Apply MMR to top K results
    
    # Score calibration
    calibrate_scores: bool = True
    calibration_method: str = "platt"  # "platt", "isotonic", "minmax"
    
    # General
    max_rerank_candidates: int = 100
    return_scores: bool = True


class RerankerProtocol(Protocol):
    """Protocol for re-ranker implementations."""
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Re-rank documents by relevance to query.
        
        Returns list of (original_index, score) sorted by score descending.
        """
        ...


class CrossEncoderReranker:
    """
    Re-ranker using cross-encoder models.
    
    Cross-encoders jointly encode query and document pairs, enabling
    much more accurate relevance scoring than bi-encoder approaches.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            batch_size: Batch size for inference
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            self._initialized = True
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            )
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Re-rank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: List of document texts to re-rank
            top_k: Return only top K results
            
        Returns:
            List of (original_index, score) sorted by score descending
        """
        if not documents:
            return []
        
        self._ensure_initialized()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores in batches
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        
        # Create indexed scores and sort
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores
    
    def rerank_with_details(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[RankedItem[str]]:
        """
        Re-rank with detailed ranking information.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Return only top K results
            
        Returns:
            List of RankedItem with full details
        """
        rankings = self.rerank(query, documents, top_k)
        
        return [
            RankedItem(
                item=documents[idx],
                score=score,
                rank=rank,
                original_rank=idx,
                features={"cross_encoder_score": score},
            )
            for rank, (idx, score) in enumerate(rankings)
        ]


class RRFFusion:
    """
    Reciprocal Rank Fusion for combining multiple rankings.
    
    RRF is a simple yet effective method for fusing multiple ranking lists.
    It's robust to score miscalibration across different rankers.
    
    Score = sum(1 / (k + rank_i)) for each ranking list
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.
        
        Args:
            k: Smoothing parameter (typically 60)
        """
        self.k = k
    
    def fuse(
        self,
        rankings: list[list[tuple[str, float]]],
        weights: list[float] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Fuse multiple rankings using RRF.
        
        Args:
            rankings: List of rankings, each as [(id, score), ...]
            weights: Optional weights for each ranking
            top_k: Return only top K results
            
        Returns:
            Fused ranking as [(id, rrf_score), ...]
        """
        if not rankings:
            return []
        
        # Default equal weights
        if weights is None:
            weights = [1.0] * len(rankings)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        
        for ranking, weight in zip(rankings, weights):
            for rank, (item_id, _) in enumerate(ranking):
                rrf_contribution = weight / (self.k + rank + 1)
                
                if item_id in rrf_scores:
                    rrf_scores[item_id] += rrf_contribution
                else:
                    rrf_scores[item_id] = rrf_contribution
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        if top_k is not None:
            sorted_results = sorted_results[:top_k]
        
        return sorted_results
    
    def fuse_with_items(
        self,
        rankings: list[list[tuple[Any, float]]],
        id_func: Callable[[Any], str],
        weights: list[float] | None = None,
        top_k: int | None = None,
    ) -> list[RankedItem[Any]]:
        """
        Fuse rankings with full item information.
        
        Args:
            rankings: List of rankings with items
            id_func: Function to extract ID from item
            weights: Optional weights for each ranking
            top_k: Return only top K results
            
        Returns:
            List of RankedItem with fused scores
        """
        if not rankings:
            return []
        
        # Default equal weights
        if weights is None:
            weights = [1.0] * len(rankings)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Track items and their positions
        item_lookup: dict[str, Any] = {}
        rrf_scores: dict[str, float] = {}
        rank_info: dict[str, list[int | None]] = {}  # ID -> [rank in each list]
        
        for list_idx, (ranking, weight) in enumerate(zip(rankings, weights)):
            for rank, (item, _) in enumerate(ranking):
                item_id = id_func(item)
                
                if item_id not in item_lookup:
                    item_lookup[item_id] = item
                    rrf_scores[item_id] = 0.0
                    rank_info[item_id] = [None] * len(rankings)
                
                rrf_contribution = weight / (self.k + rank + 1)
                rrf_scores[item_id] += rrf_contribution
                rank_info[item_id][list_idx] = rank
        
        # Sort by score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True,
        )
        
        if top_k is not None:
            sorted_ids = sorted_ids[:top_k]
        
        # Build results
        results = []
        for final_rank, item_id in enumerate(sorted_ids):
            # Get original rank (first non-None rank)
            orig_rank = None
            for r in rank_info[item_id]:
                if r is not None:
                    orig_rank = r
                    break
            
            results.append(RankedItem(
                item=item_lookup[item_id],
                score=rrf_scores[item_id],
                rank=final_rank,
                original_rank=orig_rank,
                features={
                    f"rank_list_{i}": r if r is not None else -1
                    for i, r in enumerate(rank_info[item_id])
                },
            ))
        
        return results


class MMRDiversifier:
    """
    Maximal Marginal Relevance for diversity-aware ranking.
    
    MMR balances relevance with diversity by penalizing documents
    that are too similar to already selected documents.
    
    MMR(D, Q, S) = argmax[λ * Rel(D, Q) - (1-λ) * max(Sim(D, D_i))]
    """
    
    def __init__(
        self,
        lambda_param: float = 0.7,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize MMR diversifier.
        
        Args:
            lambda_param: Balance between relevance (1) and diversity (0)
            similarity_threshold: Consider documents similar above this threshold
        """
        self.lambda_param = lambda_param
        self.similarity_threshold = similarity_threshold
    
    def diversify(
        self,
        items: list[tuple[Any, float, list[float]]],
        top_k: int = 10,
        id_func: Callable[[Any], str] | None = None,
    ) -> list[RankedItem[Any]]:
        """
        Apply MMR diversification to ranked items.
        
        Args:
            items: List of (item, relevance_score, embedding) tuples
            top_k: Number of diverse items to select
            id_func: Optional function to get item ID
            
        Returns:
            Diversified list of RankedItem
        """
        if not items:
            return []
        
        # Separate components
        all_items = [item for item, _, _ in items]
        relevance_scores = [score for _, score, _ in items]
        embeddings = [emb for _, _, emb in items]
        
        # Normalize relevance scores to [0, 1]
        max_rel = max(relevance_scores) if relevance_scores else 1.0
        min_rel = min(relevance_scores) if relevance_scores else 0.0
        rel_range = max_rel - min_rel
        
        if rel_range > 0:
            norm_relevance = [(r - min_rel) / rel_range for r in relevance_scores]
        else:
            norm_relevance = [0.5] * len(relevance_scores)
        
        # MMR selection
        selected_indices: list[int] = []
        selected_embeddings: list[list[float]] = []
        remaining = set(range(len(items)))
        
        for _ in range(min(top_k, len(items))):
            best_idx = None
            best_mmr = float("-inf")
            
            for idx in remaining:
                relevance = norm_relevance[idx]
                
                # Calculate max similarity to selected items
                max_sim = 0.0
                if selected_embeddings:
                    for sel_emb in selected_embeddings:
                        sim = self._cosine_similarity(embeddings[idx], sel_emb)
                        max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(embeddings[best_idx])
                remaining.remove(best_idx)
        
        # Build results
        results = []
        for rank, idx in enumerate(selected_indices):
            # Calculate diversity penalty for reporting
            max_sim = 0.0
            for i, sel_idx in enumerate(selected_indices[:rank]):
                sim = self._cosine_similarity(embeddings[idx], embeddings[sel_idx])
                max_sim = max(max_sim, sim)
            
            diversity_penalty = (1 - self.lambda_param) * max_sim
            
            results.append(RankedItem(
                item=all_items[idx],
                score=relevance_scores[idx],
                rank=rank,
                original_rank=idx,
                features={
                    "relevance": relevance_scores[idx],
                    "max_similarity_to_selected": max_sim,
                },
                diversity_penalty=diversity_penalty,
            ))
        
        return results
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between vectors."""
        if NUMPY_AVAILABLE:
            a = np.array(vec1, dtype=np.float32)  # type: ignore
            b = np.array(vec2, dtype=np.float32)  # type: ignore
            
            dot = np.dot(a, b)  # type: ignore
            norm_a = np.linalg.norm(a)  # type: ignore
            norm_b = np.linalg.norm(b)  # type: ignore
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(dot / (norm_a * norm_b))
        else:
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot / (norm1 * norm2)


class ScoreCalibrator:
    """
    Calibrate scores to meaningful probability estimates.
    
    Raw model scores often don't reflect true relevance probabilities.
    Calibration adjusts scores to be better probability estimates.
    """
    
    def __init__(self, method: str = "platt"):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ("platt", "isotonic", "minmax")
        """
        self.method = method
        self._fitted = False
        self._params: dict[str, Any] = {}
    
    def fit(
        self, 
        scores: list[float], 
        labels: list[int],
    ) -> "ScoreCalibrator":
        """
        Fit calibrator on labeled data.
        
        Args:
            scores: Model scores
            labels: Binary relevance labels (0 or 1)
            
        Returns:
            Self for chaining
        """
        if self.method == "platt":
            self._fit_platt(scores, labels)
        elif self.method == "isotonic":
            self._fit_isotonic(scores, labels)
        elif self.method == "minmax":
            self._fit_minmax(scores)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self._fitted = True
        return self
    
    def _fit_platt(self, scores: list[float], labels: list[int]) -> None:
        """Fit Platt scaling (logistic regression)."""
        # Simple Platt scaling: P = 1 / (1 + exp(A*score + B))
        # Use simple gradient descent to find A and B
        
        import random
        
        A = 1.0
        B = 0.0
        lr = 0.1
        
        for _ in range(1000):
            grad_A = 0.0
            grad_B = 0.0
            
            for score, label in zip(scores, labels):
                p = 1 / (1 + math.exp(-A * score - B))
                error = p - label
                grad_A += error * score
                grad_B += error
            
            A -= lr * grad_A / len(scores)
            B -= lr * grad_B / len(scores)
        
        self._params = {"A": A, "B": B}
    
    def _fit_isotonic(self, scores: list[float], labels: list[int]) -> None:
        """Fit isotonic regression (pool adjacent violators)."""
        # Sort by score
        paired = sorted(zip(scores, labels))
        sorted_scores = [s for s, _ in paired]
        sorted_labels = [float(l) for _, l in paired]
        
        # Pool adjacent violators algorithm (PAV)
        n = len(sorted_labels)
        calibrated = sorted_labels.copy()
        
        i = 0
        while i < n - 1:
            if calibrated[i] > calibrated[i + 1]:
                # Violation: pool
                pool_sum = calibrated[i] + calibrated[i + 1]
                pool_count = 2
                
                j = i
                while j > 0 and calibrated[j - 1] > pool_sum / pool_count:
                    j -= 1
                    pool_sum += calibrated[j]
                    pool_count += 1
                
                pool_mean = pool_sum / pool_count
                for k in range(j, i + 2):
                    calibrated[k] = pool_mean
                
                i = j
            else:
                i += 1
        
        # Store as lookup table
        self._params = {
            "scores": sorted_scores,
            "calibrated": calibrated,
        }
    
    def _fit_minmax(self, scores: list[float]) -> None:
        """Simple min-max normalization."""
        self._params = {
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 1.0,
        }
    
    def calibrate(self, scores: list[float]) -> list[float]:
        """
        Calibrate scores.
        
        Args:
            scores: Raw model scores
            
        Returns:
            Calibrated probability estimates
        """
        if not self._fitted:
            # Fall back to sigmoid if not fitted
            return [1 / (1 + math.exp(-s)) for s in scores]
        
        if self.method == "platt":
            A = self._params["A"]
            B = self._params["B"]
            return [1 / (1 + math.exp(-A * s - B)) for s in scores]
        
        elif self.method == "isotonic":
            return [self._isotonic_lookup(s) for s in scores]
        
        elif self.method == "minmax":
            min_s = self._params["min"]
            max_s = self._params["max"]
            range_s = max_s - min_s
            if range_s == 0:
                return [0.5] * len(scores)
            return [(s - min_s) / range_s for s in scores]
        
        return scores
    
    def _isotonic_lookup(self, score: float) -> float:
        """Look up calibrated value for isotonic regression."""
        sorted_scores = self._params["scores"]
        calibrated = self._params["calibrated"]
        
        # Binary search for closest score
        if score <= sorted_scores[0]:
            return calibrated[0]
        if score >= sorted_scores[-1]:
            return calibrated[-1]
        
        # Find position
        for i in range(len(sorted_scores) - 1):
            if sorted_scores[i] <= score < sorted_scores[i + 1]:
                # Linear interpolation
                t = (score - sorted_scores[i]) / (sorted_scores[i + 1] - sorted_scores[i])
                return calibrated[i] + t * (calibrated[i + 1] - calibrated[i])
        
        return calibrated[-1]


class RankingEvaluator:
    """
    Evaluate ranking quality with standard IR metrics.
    """
    
    @staticmethod
    def ndcg(
        ranked_items: list[str], 
        relevant_items: set[str], 
        relevance_scores: dict[str, float] | None = None,
        k: int | None = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            ranked_items: Ranked list of item IDs
            relevant_items: Set of relevant item IDs
            relevance_scores: Optional graded relevance scores
            k: Calculate NDCG@k (None for full list)
            
        Returns:
            NDCG score in [0, 1]
        """
        if k is not None:
            ranked_items = ranked_items[:k]
        
        # DCG
        dcg = 0.0
        for i, item in enumerate(ranked_items):
            if relevance_scores:
                rel = relevance_scores.get(item, 0.0)
            else:
                rel = 1.0 if item in relevant_items else 0.0
            
            dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank starts at 1
        
        # Ideal DCG
        if relevance_scores:
            ideal_rels = sorted(relevance_scores.values(), reverse=True)
        else:
            ideal_rels = [1.0] * len(relevant_items)
        
        ideal_rels = ideal_rels[:len(ranked_items)]
        
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2 ** rel - 1) / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mrr(
        ranked_items: list[str],
        relevant_items: set[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            ranked_items: Ranked list of item IDs
            relevant_items: Set of relevant item IDs
            
        Returns:
            MRR score in [0, 1]
        """
        for i, item in enumerate(ranked_items):
            if item in relevant_items:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def precision_at_k(
        ranked_items: list[str],
        relevant_items: set[str],
        k: int,
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            ranked_items: Ranked list of item IDs
            relevant_items: Set of relevant item IDs
            k: Calculate P@k
            
        Returns:
            Precision score in [0, 1]
        """
        top_k = ranked_items[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(
        ranked_items: list[str],
        relevant_items: set[str],
        k: int,
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            ranked_items: Ranked list of item IDs
            relevant_items: Set of relevant item IDs
            k: Calculate R@k
            
        Returns:
            Recall score in [0, 1]
        """
        if not relevant_items:
            return 0.0
        
        top_k = ranked_items[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
        return relevant_in_top_k / len(relevant_items)
    
    @staticmethod
    def average_precision(
        ranked_items: list[str],
        relevant_items: set[str],
    ) -> float:
        """
        Calculate Average Precision.
        
        Args:
            ranked_items: Ranked list of item IDs
            relevant_items: Set of relevant item IDs
            
        Returns:
            AP score in [0, 1]
        """
        if not relevant_items:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, item in enumerate(ranked_items):
            if item in relevant_items:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_items)


def rerank_with_cross_encoder(
    query: str,
    documents: list[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """
    Convenience function to re-rank documents using cross-encoder.
    
    Args:
        query: Search query
        documents: List of document texts
        model_name: Cross-encoder model name
        top_k: Return only top K results
        
    Returns:
        List of (original_index, score) sorted by score descending
    """
    reranker = CrossEncoderReranker(model_name=model_name)
    return reranker.rerank(query, documents, top_k)


def fuse_rankings(
    rankings: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Convenience function for RRF fusion.
    
    Args:
        rankings: List of rankings to fuse
        weights: Optional weights for each ranking
        k: RRF smoothing parameter
        top_k: Return only top K results
        
    Returns:
        Fused ranking as [(id, score), ...]
    """
    fuser = RRFFusion(k=k)
    return fuser.fuse(rankings, weights, top_k)


def diversify_results(
    items: list[tuple[Any, float, list[float]]],
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> list[RankedItem[Any]]:
    """
    Convenience function for MMR diversification.
    
    Args:
        items: List of (item, score, embedding) tuples
        lambda_param: Balance relevance vs diversity
        top_k: Number of items to return
        
    Returns:
        Diversified list of RankedItem
    """
    diversifier = MMRDiversifier(lambda_param=lambda_param)
    return diversifier.diversify(items, top_k)
