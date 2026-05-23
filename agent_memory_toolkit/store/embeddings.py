"""Enhanced embedding support with sentence-transformers, batch processing, and cross-encoder reranking."""

import logging
from typing import Protocol, runtime_checkable, Callable, Any
from dataclasses import dataclass
from enum import Enum
import struct
import math

logger = logging.getLogger(__name__)

# Feature availability flags
SENTENCE_TRANSFORMERS_AVAILABLE = False
CROSS_ENCODER_AVAILABLE = False
NUMPY_AVAILABLE = False

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    logger.debug("sentence-transformers not available, vector search disabled")

# Try to import cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    logger.debug("CrossEncoder not available, reranking disabled")

# Try to import numpy for faster vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    logger.debug("numpy not available, using pure Python for vector operations")


class SimilarityMetric(Enum):
    """Supported similarity metrics for vector search."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True
    show_progress: bool = False
    device: str | None = None  # 'cpu', 'cuda', 'mps', or None for auto-detect
    cache_folder: str | None = None  # Custom cache location for models
    
    # Popular model options with their dimensions:
    # - "all-MiniLM-L6-v2" (384 dim) - fast and good quality
    # - "all-mpnet-base-v2" (768 dim) - higher quality, slower
    # - "paraphrase-MiniLM-L6-v2" (384 dim) - good for paraphrase detection
    # - "multi-qa-MiniLM-L6-cos-v1" (384 dim) - optimized for Q&A
    # - "all-distilroberta-v1" (768 dim) - good balance

    @classmethod
    def fast(cls) -> "EmbeddingConfig":
        """Fast embedding configuration for development/testing."""
        return cls(
            model_name="all-MiniLM-L6-v2",
            batch_size=64,
            normalize=True,
        )
    
    @classmethod
    def quality(cls) -> "EmbeddingConfig":
        """High quality embedding configuration for production."""
        return cls(
            model_name="all-mpnet-base-v2",
            batch_size=32,
            normalize=True,
        )
    
    @classmethod
    def qa_optimized(cls) -> "EmbeddingConfig":
        """Configuration optimized for question-answering."""
        return cls(
            model_name="multi-qa-MiniLM-L6-cos-v1",
            batch_size=32,
            normalize=True,
        )


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranking."""
    
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16
    device: str | None = None
    
    # Popular cross-encoder models:
    # - "cross-encoder/ms-marco-MiniLM-L-6-v2" - fast, good for general reranking
    # - "cross-encoder/ms-marco-MiniLM-L-12-v2" - slower, more accurate
    # - "cross-encoder/stsb-roberta-base" - good for semantic similarity
    
    @classmethod
    def fast(cls) -> "RerankerConfig":
        """Fast reranking configuration."""
        return cls(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    @classmethod
    def quality(cls) -> "RerankerConfig":
        """High quality reranking configuration."""
        return cls(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embeddings."""
        ...
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Protocol for reranking providers."""
    
    def rerank(
        self, 
        query: str, 
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to query.
        
        Returns list of (original_index, score) tuples sorted by score descending.
        """
        ...


class SentenceTransformerProvider:
    """Embedding provider using sentence-transformers with enhanced features."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        config: EmbeddingConfig | None = None,
    ):
        """
        Initialize the sentence transformer provider.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            config: Optional EmbeddingConfig for advanced settings
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for vector search. "
                "Install with: pip install sentence-transformers"
            )
        
        self.config = config or EmbeddingConfig(model_name=model_name)
        self.model_name = self.config.model_name
        
        # Load model with device configuration
        device = self.config.device
        cache_folder = self.config.cache_folder
        
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            cache_folder=cache_folder,
        )
        
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Loaded embedding model: {self.model_name} "
            f"(dim={self._dimension}, device={self.model.device})"
        )
    
    def encode(
        self, 
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool | None = None,
    ) -> list[list[float]]:
        """
        Encode texts to embeddings with batch processing.
        
        Args:
            texts: List of texts to encode
            batch_size: Override config batch size
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.config.batch_size
        show_progress = show_progress if show_progress is not None else self.config.show_progress
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def encode_batch(
        self,
        texts: list[str],
        batch_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """
        Encode texts with batch-level callbacks for progress tracking.
        
        Args:
            texts: List of texts to encode
            batch_callback: Called after each batch with (batch_idx, total_batches)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = self.config.batch_size
        embeddings = []
        total_batches = math.ceil(len(texts) / batch_size)
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
            )
            
            embeddings.extend([emb.tolist() for emb in batch_embeddings])
            
            if batch_callback:
                batch_callback(batch_idx + 1, total_batches)
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class CrossEncoderReranker:
    """Cross-encoder reranker for improved search accuracy."""
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: RerankerConfig | None = None,
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
            config: Optional RerankerConfig for advanced settings
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "CrossEncoder is required for reranking. "
                "Install with: pip install sentence-transformers"
            )
        
        self.config = config or RerankerConfig(model_name=model_name)
        self.model_name = self.config.model_name
        
        self.model = CrossEncoder(
            self.model_name,
            device=self.config.device,
        )
        
        logger.info(f"Loaded cross-encoder model: {self.model_name}")
    
    def rerank(
        self, 
        query: str, 
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Return only top k results (None for all)
            
        Returns:
            List of (original_index, score) tuples sorted by score descending
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs, batch_size=self.config.batch_size)
        
        # Create index-score pairs
        indexed_scores = list(enumerate(scores))
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to proper types and optionally limit
        results = [(int(idx), float(score)) for idx, score in indexed_scores]
        
        if top_k is not None:
            results = results[:top_k]
        
        return results


# Vector operation utilities

def embedding_to_blob(embedding: list[float]) -> bytes:
    """Convert embedding to binary blob for SQLite storage."""
    return struct.pack(f'{len(embedding)}f', *embedding)


def blob_to_embedding(blob: bytes) -> list[float]:
    """Convert binary blob back to embedding list."""
    count = len(blob) // 4  # 4 bytes per float
    return list(struct.unpack(f'{count}f', blob))


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Uses numpy for faster computation if available.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")
    
    if NUMPY_AVAILABLE:
        arr1 = np.array(vec1, dtype=np.float32)
        arr2 = np.array(vec2, dtype=np.float32)
        
        dot = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    else:
        # Pure Python fallback
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


def dot_product_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate dot product similarity between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")
    
    if NUMPY_AVAILABLE:
        return float(np.dot(np.array(vec1), np.array(vec2)))
    else:
        return sum(a * b for a, b in zip(vec1, vec2))


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension")
    
    if NUMPY_AVAILABLE:
        arr1 = np.array(vec1, dtype=np.float32)
        arr2 = np.array(vec2, dtype=np.float32)
        return float(np.linalg.norm(arr1 - arr2))
    else:
        return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5


def euclidean_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Convert Euclidean distance to similarity score (0-1 range).
    
    Uses formula: 1 / (1 + distance)
    """
    distance = euclidean_distance(vec1, vec2)
    return 1.0 / (1.0 + distance)


def batch_cosine_similarity(
    query: list[float], 
    embeddings: list[list[float]],
) -> list[float]:
    """
    Calculate cosine similarities between query and multiple embeddings.
    
    Optimized batch operation using numpy when available.
    """
    if not embeddings:
        return []
    
    if NUMPY_AVAILABLE:
        query_arr = np.array(query, dtype=np.float32)
        emb_arr = np.array(embeddings, dtype=np.float32)
        
        # Normalize
        query_norm = query_arr / np.linalg.norm(query_arr)
        emb_norms = emb_arr / np.linalg.norm(emb_arr, axis=1, keepdims=True)
        
        # Batch dot product
        similarities = np.dot(emb_norms, query_norm)
        
        return similarities.tolist()
    else:
        return [cosine_similarity(query, emb) for emb in embeddings]


def normalize_embedding(embedding: list[float]) -> list[float]:
    """Normalize embedding to unit length."""
    if NUMPY_AVAILABLE:
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return embedding
        return (arr / norm).tolist()
    else:
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]
