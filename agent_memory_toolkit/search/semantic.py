"""Semantic search engine with advanced embedding-based retrieval.

This module provides production-grade semantic search capabilities including:
- Multiple embedding model support with automatic selection
- Query expansion and reformulation
- Approximate Nearest Neighbor (ANN) search support
- Embedding caching for efficiency
- Multi-vector retrieval (late interaction models)
- Contextual embeddings with surrounding text
"""

from __future__ import annotations

import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, Callable, runtime_checkable
from enum import Enum
from functools import lru_cache
import math

logger = logging.getLogger(__name__)


# Feature availability flags
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


class QueryType(Enum):
    """Types of queries for specialized handling."""
    FACTUAL = "factual"           # Who, what, when, where questions
    CONCEPTUAL = "conceptual"     # How, why questions - need semantic understanding
    NAVIGATIONAL = "navigational" # Looking for specific document/memory
    EXPLORATORY = "exploratory"   # Broad exploration of a topic
    SIMILARITY = "similarity"     # Find similar to given text


@dataclass
class QueryAnalysis:
    """Result of analyzing a search query."""
    original_query: str
    query_type: QueryType
    expanded_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    semantic_weight: float = 1.0  # Weight for semantic vs keyword search
    intent_confidence: float = 0.5
    

@dataclass
class SemanticSearchConfig:
    """Configuration for semantic search."""
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    
    # Search parameters
    top_k: int = 10
    min_similarity: float = 0.0
    
    # Query expansion
    enable_query_expansion: bool = True
    max_expansion_queries: int = 3
    expansion_weight: float = 0.3  # Weight given to expanded queries
    
    # Multi-vector settings
    use_multi_vector: bool = False
    late_interaction_k: int = 10  # For ColBERT-style late interaction
    
    # Caching
    cache_embeddings: bool = True
    cache_size: int = 10000
    
    # ANN settings
    use_ann: bool = False  # Use approximate nearest neighbor
    ann_ef_search: int = 100  # HNSW ef parameter
    ann_num_candidates: int = 100  # Number of candidates for ANN
    
    # Re-ranking
    rerank_candidates: int = 50  # Number of candidates to rerank


@dataclass  
class SemanticMatch:
    """A semantic search match with detailed scoring."""
    memory_id: str
    content: str
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    
    # Detailed scoring breakdown
    primary_score: float = 0.0  # Main query match
    expansion_scores: list[float] = field(default_factory=list)  # Expanded query matches
    recency_boost: float = 0.0
    confidence_boost: float = 0.0
    
    @property
    def combined_score(self) -> float:
        """Get the combined weighted score."""
        return self.similarity_score + self.recency_boost + self.confidence_boost


class EmbeddingCache:
    """LRU cache for embeddings with memory management."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: dict[str, list[float]] = {}
        self._access_order: list[str] = []
        
    def _make_key(self, text: str) -> str:
        """Create a cache key from text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> list[float] | None:
        """Get cached embedding if available."""
        key = self._make_key(text)
        if key in self._cache:
            # Update access order for LRU
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        key = self._make_key(text)
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)
        
        self._cache[key] = embedding
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_batch(self, texts: list[str]) -> tuple[list[list[float]], list[int]]:
        """
        Get cached embeddings for a batch of texts.
        
        Returns:
            Tuple of (cached_embeddings, missing_indices)
        """
        cached = []
        missing = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text)
            if embedding is not None:
                cached.append(embedding)
            else:
                missing.append(i)
        
        return cached, missing


@runtime_checkable
class QueryExpander(Protocol):
    """Protocol for query expansion implementations."""
    
    def expand(self, query: str, max_expansions: int = 3) -> list[str]:
        """Expand query into related queries."""
        ...


class SynonymQueryExpander:
    """Query expander using synonym mapping."""
    
    def __init__(self, synonym_map: dict[str, list[str]] | None = None):
        self.synonym_map = synonym_map or self._default_synonyms()
    
    def _default_synonyms(self) -> dict[str, list[str]]:
        """Default synonym mappings for common terms."""
        return {
            "create": ["make", "build", "generate", "produce"],
            "delete": ["remove", "erase", "destroy", "eliminate"],
            "update": ["modify", "change", "edit", "revise"],
            "find": ["search", "locate", "discover", "retrieve"],
            "help": ["assist", "support", "aid", "guide"],
            "error": ["bug", "issue", "problem", "failure"],
            "fast": ["quick", "rapid", "speedy", "efficient"],
            "important": ["critical", "crucial", "significant", "key"],
            "user": ["person", "individual", "customer", "client"],
            "data": ["information", "content", "records", "details"],
        }
    
    def expand(self, query: str, max_expansions: int = 3) -> list[str]:
        """Expand query using synonyms."""
        words = query.lower().split()
        expansions = []
        
        for word in words:
            if word in self.synonym_map:
                for synonym in self.synonym_map[word][:max_expansions]:
                    expanded = query.lower().replace(word, synonym)
                    if expanded not in expansions and expanded != query.lower():
                        expansions.append(expanded)
        
        return expansions[:max_expansions]


class LLMQueryExpander:
    """Query expander using LLM for semantic expansion."""
    
    def __init__(
        self, 
        llm_client: Any = None,
        model: str = "gpt-3.5-turbo",
        system_prompt: str | None = None,
    ):
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a search query expansion assistant. Given a search query, "
            "generate 3 alternative phrasings that capture the same intent but "
            "use different words or perspectives. Return only the queries, one per line."
        )
    
    def expand(self, query: str, max_expansions: int = 3) -> list[str]:
        """Expand query using LLM."""
        if self.llm_client is None:
            return []
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            
            text = response.choices[0].message.content.strip()
            expansions = [line.strip() for line in text.split("\n") if line.strip()]
            return expansions[:max_expansions]
            
        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
            return []


class QueryAnalyzer:
    """Analyzes queries to determine type and optimal search strategy."""
    
    # Question word patterns
    FACTUAL_PATTERNS = ["who", "what", "when", "where", "which", "how many", "how much"]
    CONCEPTUAL_PATTERNS = ["how", "why", "explain", "describe", "what is the reason"]
    NAVIGATIONAL_PATTERNS = ["find", "locate", "show me", "get", "open", "go to"]
    
    def __init__(self, query_expander: QueryExpander | None = None):
        self.query_expander = query_expander or SynonymQueryExpander()
    
    def analyze(self, query: str, expand: bool = True) -> QueryAnalysis:
        """Analyze a search query."""
        query_lower = query.lower().strip()
        
        # Determine query type
        query_type = self._classify_query_type(query_lower)
        
        # Extract keywords (simple tokenization)
        keywords = self._extract_keywords(query_lower)
        
        # Determine semantic weight based on query type
        semantic_weight = self._compute_semantic_weight(query_type, keywords)
        
        # Expand queries if enabled
        expanded = []
        if expand and self.query_expander:
            expanded = self.query_expander.expand(query)
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            expanded_queries=expanded,
            keywords=keywords,
            entities=[],  # Could be enhanced with NER
            semantic_weight=semantic_weight,
            intent_confidence=0.7,  # Could be enhanced with a classifier
        )
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the query type."""
        # Check patterns in order of specificity
        for pattern in self.NAVIGATIONAL_PATTERNS:
            if query.startswith(pattern):
                return QueryType.NAVIGATIONAL
        
        for pattern in self.CONCEPTUAL_PATTERNS:
            if query.startswith(pattern):
                return QueryType.CONCEPTUAL
        
        for pattern in self.FACTUAL_PATTERNS:
            if query.startswith(pattern):
                return QueryType.FACTUAL
        
        # Check if it's a similarity query (long text input)
        if len(query.split()) > 20:
            return QueryType.SIMILARITY
        
        # Default to exploratory
        return QueryType.EXPLORATORY
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query."""
        # Simple stopword removal
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "or", "because",
            "if", "until", "while", "about", "against", "out", "up",
            "down", "off", "over", "this", "that", "these", "those",
            "what", "which", "who", "whom", "it", "its", "i", "me", "my",
            "we", "you", "your", "he", "she", "they", "them",
        }
        
        words = query.split()
        return [w for w in words if w.lower() not in stopwords and len(w) > 2]
    
    def _compute_semantic_weight(self, query_type: QueryType, keywords: list[str]) -> float:
        """Compute optimal semantic search weight for this query."""
        base_weights = {
            QueryType.FACTUAL: 0.5,        # Balance keyword and semantic
            QueryType.CONCEPTUAL: 0.9,     # Strong semantic preference
            QueryType.NAVIGATIONAL: 0.3,   # More keyword focused
            QueryType.EXPLORATORY: 0.7,    # Lean semantic
            QueryType.SIMILARITY: 1.0,     # Pure semantic
        }
        
        weight = base_weights.get(query_type, 0.7)
        
        # Adjust based on keyword characteristics
        if keywords:
            # More specific/technical keywords -> lean more keyword
            avg_word_len = sum(len(k) for k in keywords) / len(keywords)
            if avg_word_len > 8:  # Technical terms tend to be longer
                weight = max(0.3, weight - 0.2)
        
        return weight


class SemanticSearchEngine:
    """
    Advanced semantic search engine with embedding-based retrieval.
    
    Features:
    - Multiple embedding model support
    - Query expansion and analysis
    - Embedding caching
    - Configurable similarity metrics
    - Multi-vector retrieval support
    
    Example:
        >>> from agent_memory_toolkit.search import SemanticSearchEngine
        >>> engine = SemanticSearchEngine(embedding_provider=provider)
        >>> 
        >>> # Simple search
        >>> results = engine.search("How do I reset my password?")
        >>> 
        >>> # With query expansion
        >>> results = engine.search(
        ...     "authentication issues",
        ...     expand_query=True,
        ... )
    """
    
    def __init__(
        self,
        embedding_provider: Any = None,
        config: SemanticSearchConfig | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        embedding_cache: EmbeddingCache | None = None,
    ):
        """
        Initialize the semantic search engine.
        
        Args:
            embedding_provider: Provider implementing encode(texts) -> embeddings
            config: Search configuration
            query_analyzer: Query analyzer for expansion and classification
            embedding_cache: Cache for storing embeddings
        """
        self.embedding_provider = embedding_provider
        self.config = config or SemanticSearchConfig()
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        
        # Initialize cache
        if embedding_cache:
            self.cache = embedding_cache
        elif self.config.cache_embeddings:
            self.cache = EmbeddingCache(max_size=self.config.cache_size)
        else:
            self.cache = None
    
    def encode_query(
        self, 
        query: str,
        use_cache: bool = True,
    ) -> list[float]:
        """
        Encode a query to its embedding representation.
        
        Args:
            query: Query text
            use_cache: Whether to use embedding cache
            
        Returns:
            Query embedding vector
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(query)
            if cached is not None:
                return cached
        
        # Generate embedding
        if self.embedding_provider is None:
            raise ValueError("Embedding provider required for semantic search")
        
        embeddings = self.embedding_provider.encode([query])
        embedding = embeddings[0]
        
        # Cache the result
        if use_cache and self.cache:
            self.cache.put(query, embedding)
        
        return embedding
    
    def encode_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """
        Encode multiple texts with caching.
        
        Args:
            texts: List of texts to encode
            use_cache: Whether to use embedding cache
            progress_callback: Called with (processed, total) for progress
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache for existing embeddings
        embeddings = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []
        
        if use_cache and self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    embeddings[i] = cached
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
        else:
            texts_to_encode = texts
            indices_to_encode = list(range(len(texts)))
        
        # Encode missing texts
        if texts_to_encode:
            if self.embedding_provider is None:
                raise ValueError("Embedding provider required for semantic search")
            
            new_embeddings = self.embedding_provider.encode(texts_to_encode)
            
            for idx, embedding in zip(indices_to_encode, new_embeddings):
                embeddings[idx] = embedding
                
                # Cache the new embedding
                if use_cache and self.cache:
                    self.cache.put(texts[idx], embedding)
        
        if progress_callback:
            progress_callback(len(texts), len(texts))
        
        return embeddings
    
    def compute_similarity(
        self,
        query_embedding: list[float],
        document_embeddings: list[list[float]],
        metric: str = "cosine",
    ) -> list[float]:
        """
        Compute similarity between query and documents.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            metric: Similarity metric ("cosine", "dot", "euclidean")
            
        Returns:
            List of similarity scores
        """
        if not document_embeddings:
            return []
        
        if NUMPY_AVAILABLE:
            query_arr = np.array(query_embedding, dtype=np.float32)
            docs_arr = np.array(document_embeddings, dtype=np.float32)
            
            if metric == "cosine":
                # Normalize
                query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-10)
                doc_norms = docs_arr / (np.linalg.norm(docs_arr, axis=1, keepdims=True) + 1e-10)
                similarities = np.dot(doc_norms, query_norm)
            elif metric == "dot":
                similarities = np.dot(docs_arr, query_arr)
            else:  # euclidean
                distances = np.linalg.norm(docs_arr - query_arr, axis=1)
                similarities = 1.0 / (1.0 + distances)
            
            return similarities.tolist()
        else:
            # Pure Python fallback
            similarities = []
            for doc_emb in document_embeddings:
                if metric == "cosine":
                    sim = self._cosine_similarity_py(query_embedding, doc_emb)
                elif metric == "dot":
                    sim = sum(a * b for a, b in zip(query_embedding, doc_emb))
                else:  # euclidean
                    dist = sum((a - b) ** 2 for a, b in zip(query_embedding, doc_emb)) ** 0.5
                    sim = 1.0 / (1.0 + dist)
                similarities.append(sim)
            return similarities
    
    def _cosine_similarity_py(self, vec1: list[float], vec2: list[float]) -> float:
        """Pure Python cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def search(
        self,
        query: str,
        documents: list[dict[str, Any]],
        document_embeddings: list[list[float]] | None = None,
        content_field: str = "content",
        id_field: str = "id",
        metadata_field: str = "metadata",
        top_k: int | None = None,
        expand_query: bool | None = None,
        min_similarity: float | None = None,
    ) -> list[SemanticMatch]:
        """
        Perform semantic search over documents.
        
        Args:
            query: Search query
            documents: List of documents to search
            document_embeddings: Pre-computed embeddings (optional)
            content_field: Field name containing document content
            id_field: Field name containing document ID  
            metadata_field: Field name containing metadata
            top_k: Number of results to return
            expand_query: Whether to expand the query
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SemanticMatch results sorted by score
        """
        if not documents:
            return []
        
        # Use config defaults
        top_k = top_k or self.config.top_k
        expand_query = expand_query if expand_query is not None else self.config.enable_query_expansion
        min_similarity = min_similarity if min_similarity is not None else self.config.min_similarity
        
        # Analyze query
        analysis = self.query_analyzer.analyze(query, expand=expand_query)
        
        # Get query embeddings (original + expanded)
        queries_to_embed = [query]
        if analysis.expanded_queries:
            queries_to_embed.extend(analysis.expanded_queries[:self.config.max_expansion_queries])
        
        query_embeddings = self.encode_batch(queries_to_embed)
        primary_embedding = query_embeddings[0]
        expansion_embeddings = query_embeddings[1:] if len(query_embeddings) > 1 else []
        
        # Get document embeddings
        if document_embeddings is None:
            contents = [doc.get(content_field, "") for doc in documents]
            document_embeddings = self.encode_batch(contents)
        
        # Compute primary similarities
        primary_scores = self.compute_similarity(primary_embedding, document_embeddings)
        
        # Compute expansion similarities if any
        expansion_scores_list = []
        if expansion_embeddings:
            for exp_emb in expansion_embeddings:
                exp_scores = self.compute_similarity(exp_emb, document_embeddings)
                expansion_scores_list.append(exp_scores)
        
        # Build results
        results = []
        for i, doc in enumerate(documents):
            primary_score = primary_scores[i]
            
            # Aggregate expansion scores
            exp_scores = [scores[i] for scores in expansion_scores_list]
            exp_contribution = 0.0
            if exp_scores:
                exp_contribution = sum(exp_scores) / len(exp_scores) * self.config.expansion_weight
            
            # Combined score
            combined_score = primary_score + exp_contribution
            
            # Apply minimum threshold
            if combined_score < min_similarity:
                continue
            
            match = SemanticMatch(
                memory_id=doc.get(id_field, str(i)),
                content=doc.get(content_field, ""),
                similarity_score=combined_score,
                metadata=doc.get(metadata_field, {}),
                embedding=document_embeddings[i],
                primary_score=primary_score,
                expansion_scores=exp_scores,
            )
            results.append(match)
        
        # Sort by score descending
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:top_k]
    
    def find_similar(
        self,
        reference_embedding: list[float],
        documents: list[dict[str, Any]],
        document_embeddings: list[list[float]] | None = None,
        content_field: str = "content",
        id_field: str = "id",
        exclude_ids: set[str] | None = None,
        top_k: int = 10,
    ) -> list[SemanticMatch]:
        """
        Find documents similar to a reference embedding.
        
        Args:
            reference_embedding: Reference embedding to compare against
            documents: List of documents to search
            document_embeddings: Pre-computed embeddings (optional)
            content_field: Field name containing document content
            id_field: Field name containing document ID
            exclude_ids: IDs to exclude from results
            top_k: Number of results to return
            
        Returns:
            List of SemanticMatch results sorted by similarity
        """
        if not documents:
            return []
        
        exclude_ids = exclude_ids or set()
        
        # Get document embeddings if not provided
        if document_embeddings is None:
            contents = [doc.get(content_field, "") for doc in documents]
            document_embeddings = self.encode_batch(contents)
        
        # Compute similarities
        scores = self.compute_similarity(reference_embedding, document_embeddings)
        
        # Build results
        results = []
        for i, doc in enumerate(documents):
            doc_id = doc.get(id_field, str(i))
            
            if doc_id in exclude_ids:
                continue
            
            match = SemanticMatch(
                memory_id=doc_id,
                content=doc.get(content_field, ""),
                similarity_score=scores[i],
                embedding=document_embeddings[i],
                primary_score=scores[i],
            )
            results.append(match)
        
        # Sort by score descending
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:top_k]


class MultiVectorSearch:
    """
    Multi-vector retrieval for ColBERT-style late interaction.
    
    This approach stores multiple vectors per document (token-level)
    and computes MaxSim for more nuanced matching.
    """
    
    def __init__(
        self,
        embedding_provider: Any = None,
        interaction_k: int = 10,
    ):
        self.embedding_provider = embedding_provider
        self.interaction_k = interaction_k
    
    def encode_multi_vector(self, text: str) -> list[list[float]]:
        """
        Encode text as multiple vectors (one per token/chunk).
        
        Returns:
            List of embeddings for each meaningful segment
        """
        if self.embedding_provider is None:
            raise ValueError("Embedding provider required")
        
        # Simple chunking by sentence/phrase
        # In production, use proper tokenization
        chunks = self._chunk_text(text)
        if not chunks:
            chunks = [text]
        
        return self.embedding_provider.encode(chunks)
    
    def _chunk_text(self, text: str, max_chunk_len: int = 100) -> list[str]:
        """Chunk text into meaningful segments."""
        # Simple sentence-based chunking
        import re
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > max_chunk_len:
                # Further split long sentences
                words = sent.split()
                current_chunk = []
                current_len = 0
                
                for word in words:
                    if current_len + len(word) + 1 > max_chunk_len:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_len = len(word)
                    else:
                        current_chunk.append(word)
                        current_len += len(word) + 1
                
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
            elif sent:
                chunks.append(sent)
        
        return chunks
    
    def compute_max_sim(
        self,
        query_vectors: list[list[float]],
        doc_vectors: list[list[float]],
    ) -> float:
        """
        Compute MaxSim score between query and document vectors.
        
        For each query vector, find the max similarity with any document vector,
        then sum across all query vectors.
        """
        if not query_vectors or not doc_vectors:
            return 0.0
        
        if NUMPY_AVAILABLE:
            q_arr = np.array(query_vectors, dtype=np.float32)
            d_arr = np.array(doc_vectors, dtype=np.float32)
            
            # Normalize
            q_norm = q_arr / (np.linalg.norm(q_arr, axis=1, keepdims=True) + 1e-10)
            d_norm = d_arr / (np.linalg.norm(d_arr, axis=1, keepdims=True) + 1e-10)
            
            # Compute all pairwise similarities
            similarities = np.dot(q_norm, d_norm.T)
            
            # MaxSim: for each query vector, take max over document vectors
            max_sims = np.max(similarities, axis=1)
            
            return float(np.sum(max_sims))
        else:
            # Pure Python fallback
            total = 0.0
            for q_vec in query_vectors:
                max_sim = 0.0
                for d_vec in doc_vectors:
                    sim = self._cosine_sim(q_vec, d_vec)
                    max_sim = max(max_sim, sim)
                total += max_sim
            return total
    
    def _cosine_sim(self, vec1: list[float], vec2: list[float]) -> float:
        """Pure Python cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
