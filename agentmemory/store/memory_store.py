"""Main MemoryStore class with SQLite + FTS5 backend and git-like versioning."""

from __future__ import annotations

import sqlite3
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Callable
from contextlib import contextmanager
from enum import Enum

from .models import Memory, MemoryMetadata, SearchResult, Branch, Commit
from .exceptions import (
    MemoryStoreError,
    MemoryNotFoundError,
    BranchNotFoundError,
    CommitNotFoundError,
    MergeConflictError,
)
from .schema import apply_schema, run_migrations
from .embeddings import (
    EmbeddingProvider,
    RerankerProvider,
    SentenceTransformerProvider,
    CrossEncoderReranker,
    EmbeddingConfig,
    RerankerConfig,
    SimilarityMetric,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    CROSS_ENCODER_AVAILABLE,
    embedding_to_blob,
    blob_to_embedding,
    cosine_similarity,
    batch_cosine_similarity,
    dot_product_similarity,
    euclidean_similarity,
    normalize_embedding,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search combining FTS and vector search."""
    
    fts_weight: float = 0.5
    vector_weight: float = 0.5
    fts_limit_multiplier: int = 3  # Fetch more candidates for reranking
    vector_limit_multiplier: int = 3
    normalize_scores: bool = True
    min_score_threshold: float = 0.0  # Filter out low-scoring results
    
    def __post_init__(self):
        # Normalize weights
        total = self.fts_weight + self.vector_weight
        if total > 0:
            self.fts_weight /= total
            self.vector_weight /= total


class SearchMethod(Enum):
    """Available search methods."""
    FTS = "fts"
    VECTOR = "vector"
    HYBRID = "hybrid"
    AUTO = "auto"


class MemoryStore:
    """
    Local-first memory store for AI agents with SQLite + FTS5.
    
    Features:
    - Full-text search with FTS5 (BM25 ranking)
    - Optional vector similarity search with sentence-transformers
    - Hybrid search combining FTS + vector scores with configurable weights
    - Cross-encoder reranking for improved accuracy
    - Batch embedding for efficient bulk operations
    - Configurable embedding models
    - Git-like versioning (branches, commits, rollback)
    - Memory metadata (timestamps, source, confidence, tags)
    - JSON export/import
    
    Example:
        >>> store = MemoryStore("agent_memory.db", auto_embed=True)
        >>> memory = store.add("The capital of France is Paris")
        >>> results = store.search("France capital")
        >>> print(results[0].memory.content)
        The capital of France is Paris
        
        # With reranking for better accuracy
        >>> results = store.search("What is the capital of France?", rerank=True)
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedding_provider: EmbeddingProvider | None = None,
        reranker: RerankerProvider | None = None,
        auto_embed: bool = False,
        embedding_config: EmbeddingConfig | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
    ):
        """
        Initialize the memory store.

        Args:
            db_path: Path to SQLite database, or ":memory:" for in-memory
            embedding_provider: Custom embedding provider (optional)
            reranker: Custom reranker provider (optional)
            auto_embed: Automatically generate embeddings for new memories
            embedding_config: Advanced embedding configuration
            embedding_model: Model name for default SentenceTransformer provider
            similarity_metric: Metric for vector similarity (cosine, dot_product, euclidean)
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self._conn: sqlite3.Connection | None = None
        self._current_branch = "main"
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._auto_embed = auto_embed
        self._embedding_config = embedding_config
        self._embedding_model = embedding_model
        self._similarity_metric = similarity_metric

        # Initialize database
        self._init_db()
        
        # Initialize embedding provider if auto_embed is enabled
        if self._auto_embed and self._embedding_provider is None:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                if embedding_config:
                    self._embedding_provider = SentenceTransformerProvider(
                        config=embedding_config
                    )
                else:
                    self._embedding_provider = SentenceTransformerProvider(embedding_model)
            else:
                logger.warning(
                    "auto_embed=True but sentence-transformers not installed. "
                    "Vector search will be disabled."
                )
                self._auto_embed = False

    def _init_db(self) -> None:
        """Initialize the database connection and schema."""
        self._conn = sqlite3.connect(
            self.db_path if isinstance(self.db_path, str) else str(self.db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        
        # Enable foreign keys
        self._conn.execute("PRAGMA foreign_keys = ON")
        
        # Apply schema and migrations
        apply_schema(self._conn)
        run_migrations(self._conn)
        
        # Ensure main branch exists
        self._ensure_main_branch()

    def _ensure_main_branch(self) -> None:
        """Ensure the main branch exists."""
        cursor = self._conn.execute(
            "SELECT name FROM branches WHERE name = 'main'"
        )
        if cursor.fetchone() is None:
            self._conn.execute(
                "INSERT INTO branches (name, head_commit_id, created_at, is_active) "
                "VALUES ('main', NULL, ?, 1)",
                (datetime.utcnow().isoformat(),)
            )
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for database transactions."""
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ==================== CRUD Operations ====================

    def add(
        self,
        content: str,
        metadata: MemoryMetadata | dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> Memory:
        """
        Add a new memory to the store.

        Args:
            content: The memory content text
            metadata: Optional metadata (MemoryMetadata or dict)
            embedding: Optional pre-computed embedding vector

        Returns:
            The created Memory object
        """
        if isinstance(metadata, dict):
            metadata = MemoryMetadata.from_dict(metadata)
        elif metadata is None:
            metadata = MemoryMetadata()

        # Generate embedding if auto_embed is enabled
        if embedding is None and self._auto_embed and self._embedding_provider:
            embeddings = self._embedding_provider.encode([content])
            embedding = embeddings[0]

        memory = Memory.create(content=content, metadata=metadata, embedding=embedding)
        
        with self.transaction():
            # Insert into main table
            embedding_blob = embedding_to_blob(embedding) if embedding else None
            self._conn.execute(
                """
                INSERT INTO memories 
                (id, content, metadata_json, embedding_blob, created_at, updated_at, version, is_deleted, branch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.metadata.to_json(),
                    embedding_blob,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.version,
                    int(memory.is_deleted),
                    self._current_branch,
                ),
            )
            
            # Record version
            self._record_version(memory, "create")

        logger.debug(f"Added memory: {memory.id}")
        return memory

    def get(self, memory_id: str) -> Memory:
        """
        Get a memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            The Memory object

        Raises:
            MemoryNotFoundError: If memory not found
        """
        cursor = self._conn.execute(
            """
            SELECT id, content, metadata_json, embedding_blob, created_at, updated_at, 
                   version, is_deleted
            FROM memories
            WHERE id = ? AND branch = ? AND is_deleted = 0
            """,
            (memory_id, self._current_branch),
        )
        row = cursor.fetchone()
        
        if row is None:
            raise MemoryNotFoundError(memory_id)
        
        return self._row_to_memory(row)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: MemoryMetadata | dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> Memory:
        """
        Update an existing memory.

        Args:
            memory_id: The memory ID to update
            content: New content (optional)
            metadata: New metadata (optional)
            embedding: New embedding (optional)

        Returns:
            The updated Memory object

        Raises:
            MemoryNotFoundError: If memory not found
        """
        # Get existing memory
        memory = self.get(memory_id)
        
        # Update fields
        if content is not None:
            memory.content = content
            # Regenerate embedding if auto_embed is enabled
            if self._auto_embed and self._embedding_provider and embedding is None:
                embeddings = self._embedding_provider.encode([content])
                embedding = embeddings[0]
        
        if metadata is not None:
            if isinstance(metadata, dict):
                memory.metadata = MemoryMetadata.from_dict(metadata)
            else:
                memory.metadata = metadata
        
        if embedding is not None:
            memory.embedding = embedding
        
        memory.updated_at = datetime.utcnow()
        memory.version += 1
        
        with self.transaction():
            embedding_blob = (
                embedding_to_blob(memory.embedding) if memory.embedding else None
            )
            self._conn.execute(
                """
                UPDATE memories
                SET content = ?, metadata_json = ?, embedding_blob = ?, 
                    updated_at = ?, version = ?
                WHERE id = ? AND branch = ?
                """,
                (
                    memory.content,
                    memory.metadata.to_json(),
                    embedding_blob,
                    memory.updated_at.isoformat(),
                    memory.version,
                    memory_id,
                    self._current_branch,
                ),
            )
            
            # Record version
            self._record_version(memory, "update")

        logger.debug(f"Updated memory: {memory_id}")
        return memory

    def delete(self, memory_id: str, hard: bool = False) -> None:
        """
        Delete a memory.

        Args:
            memory_id: The memory ID to delete
            hard: If True, permanently delete; if False, soft delete

        Raises:
            MemoryNotFoundError: If memory not found
        """
        memory = self.get(memory_id)
        
        with self.transaction():
            if hard:
                # Delete versions first (due to FK constraint)
                self._conn.execute(
                    "DELETE FROM memory_versions WHERE memory_id = ?",
                    (memory_id,),
                )
                # Then delete the memory
                self._conn.execute(
                    "DELETE FROM memories WHERE id = ? AND branch = ?",
                    (memory_id, self._current_branch),
                )
            else:
                memory.is_deleted = True
                memory.updated_at = datetime.utcnow()
                memory.version += 1
                
                self._conn.execute(
                    """
                    UPDATE memories
                    SET is_deleted = 1, updated_at = ?, version = ?
                    WHERE id = ? AND branch = ?
                    """,
                    (memory.updated_at.isoformat(), memory.version, memory_id, self._current_branch),
                )
                
                self._record_version(memory, "delete")

        logger.debug(f"Deleted memory: {memory_id} (hard={hard})")

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False,
        tag: str | None = None,
    ) -> list[Memory]:
        """
        List memories with pagination.

        Args:
            limit: Maximum number of memories to return
            offset: Number of memories to skip
            include_deleted: Include soft-deleted memories
            tag: Filter by tag

        Returns:
            List of Memory objects
        """
        query = """
            SELECT id, content, metadata_json, embedding_blob, created_at, updated_at,
                   version, is_deleted
            FROM memories
            WHERE branch = ?
        """
        params: list[Any] = [self._current_branch]
        
        if not include_deleted:
            query += " AND is_deleted = 0"
        
        if tag:
            query += " AND json_extract(metadata_json, '$.tags') LIKE ?"
            params.append(f'%"{tag}"%')
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = self._conn.execute(query, params)
        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def count(self, include_deleted: bool = False) -> int:
        """Count memories in the current branch."""
        query = "SELECT COUNT(*) FROM memories WHERE branch = ?"
        params: list[Any] = [self._current_branch]
        
        if not include_deleted:
            query += " AND is_deleted = 0"
        
        cursor = self._conn.execute(query, params)
        return cursor.fetchone()[0]

    def _record_version(self, memory: Memory, operation: str) -> None:
        """Record a memory version for history tracking."""
        embedding_blob = (
            embedding_to_blob(memory.embedding) if memory.embedding else None
        )
        self._conn.execute(
            """
            INSERT INTO memory_versions
            (memory_id, content, metadata_json, embedding_blob, version, created_at, operation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.content,
                memory.metadata.to_json(),
                embedding_blob,
                memory.version,
                datetime.utcnow().isoformat(),
                operation,
            ),
        )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a database row to a Memory object."""
        embedding = None
        if row["embedding_blob"]:
            embedding = blob_to_embedding(row["embedding_blob"])
        
        return Memory(
            id=row["id"],
            content=row["content"],
            metadata=MemoryMetadata.from_json(row["metadata_json"]),
            embedding=embedding,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            version=row["version"],
            is_deleted=bool(row["is_deleted"]),
        )

    # ==================== Search Operations ====================

    def search_fts(
        self,
        query: str,
        limit: int = 10,
        include_deleted: bool = False,
        boost_recent: bool = False,
        boost_confidence: bool = False,
    ) -> list[SearchResult]:
        """
        Full-text search using FTS5 with BM25 ranking.

        Args:
            query: Search query (supports FTS5 query syntax)
            limit: Maximum number of results
            include_deleted: Include soft-deleted memories
            boost_recent: Boost score based on recency
            boost_confidence: Boost score based on metadata confidence

        Returns:
            List of SearchResult objects sorted by relevance
        """
        sql = """
            SELECT m.id, m.content, m.metadata_json, m.embedding_blob,
                   m.created_at, m.updated_at, m.version, m.is_deleted,
                   bm25(memories_fts) AS score
            FROM memories_fts f
            JOIN memories m ON f.memory_id = m.id
            WHERE memories_fts MATCH ?
              AND m.branch = ?
        """
        params: list[Any] = [query, self._current_branch]
        
        if not include_deleted:
            sql += " AND m.is_deleted = 0"
        
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        
        cursor = self._conn.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            memory = self._row_to_memory(row)
            # BM25 returns negative scores (more negative = better match)
            # Convert to positive score for consistency
            score = -row["score"]
            
            # Apply optional boosts
            if boost_recent:
                # Decay factor: newer memories get higher boost
                age_days = (datetime.utcnow() - memory.created_at).days
                recency_factor = 1.0 / (1.0 + age_days / 30.0)  # Half decay at 30 days
                score *= (1.0 + recency_factor * 0.5)
            
            if boost_confidence:
                score *= memory.metadata.confidence
            
            results.append(SearchResult(memory=memory, score=score, match_type="fts"))
        
        # Re-sort if boosts were applied
        if boost_recent or boost_confidence:
            results.sort(key=lambda x: x.score, reverse=True)
        
        return results

    def search_vector(
        self,
        query: str | list[float],
        limit: int = 10,
        include_deleted: bool = False,
        similarity_metric: SimilarityMetric | None = None,
        use_batch: bool = True,
    ) -> list[SearchResult]:
        """
        Vector similarity search with configurable similarity metrics.

        Args:
            query: Search query string or embedding vector
            limit: Maximum number of results
            include_deleted: Include soft-deleted memories
            similarity_metric: Override default similarity metric
            use_batch: Use optimized batch computation

        Returns:
            List of SearchResult objects sorted by similarity

        Raises:
            MemoryStoreError: If embedding provider not configured
        """
        metric = similarity_metric or self._similarity_metric
        
        # Get query embedding
        if isinstance(query, str):
            if not self._embedding_provider:
                raise MemoryStoreError(
                    "Embedding provider required for vector search. "
                    "Either provide an embedding_provider or set auto_embed=True."
                )
            embeddings = self._embedding_provider.encode([query])
            query_embedding = embeddings[0]
        else:
            query_embedding = query
        
        # Get all memories with embeddings
        sql = """
            SELECT id, content, metadata_json, embedding_blob,
                   created_at, updated_at, version, is_deleted
            FROM memories
            WHERE branch = ? AND embedding_blob IS NOT NULL
        """
        params: list[Any] = [self._current_branch]
        
        if not include_deleted:
            sql += " AND is_deleted = 0"
        
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()
        
        if not rows:
            return []
        
        # Get similarity function
        if metric == SimilarityMetric.COSINE:
            sim_func = cosine_similarity
        elif metric == SimilarityMetric.DOT_PRODUCT:
            sim_func = dot_product_similarity
        else:  # EUCLIDEAN
            sim_func = euclidean_similarity
        
        # Calculate similarities
        if use_batch and metric == SimilarityMetric.COSINE:
            # Batch computation for cosine similarity
            memories = [self._row_to_memory(row) for row in rows]
            embeddings_list = [m.embedding for m in memories if m.embedding]
            
            if embeddings_list:
                similarities = batch_cosine_similarity(query_embedding, embeddings_list)
                results = [
                    SearchResult(memory=mem, score=sim, match_type="vector")
                    for mem, sim in zip(memories, similarities)
                ]
            else:
                results = []
        else:
            # Individual computation
            results = []
            for row in rows:
                memory = self._row_to_memory(row)
                if memory.embedding:
                    score = sim_func(query_embedding, memory.embedding)
                    results.append(
                        SearchResult(memory=memory, score=score, match_type="vector")
                    )
        
        # Sort by score descending and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        config: HybridSearchConfig | None = None,
        fts_weight: float | None = None,
        vector_weight: float | None = None,
        include_deleted: bool = False,
        rerank: bool = False,
        rerank_top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search combining FTS and vector similarity with optional reranking.

        Args:
            query: Search query string
            limit: Maximum number of results
            config: HybridSearchConfig for advanced settings
            fts_weight: Weight for FTS score (0-1), overrides config
            vector_weight: Weight for vector score (0-1), overrides config
            include_deleted: Include soft-deleted memories
            rerank: Use cross-encoder reranking for final results
            rerank_top_k: Number of candidates to pass to reranker

        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Build effective config
        if config is None:
            config = HybridSearchConfig()
        
        if fts_weight is not None:
            config.fts_weight = fts_weight
        if vector_weight is not None:
            config.vector_weight = vector_weight
        
        # Normalize weights
        total = config.fts_weight + config.vector_weight
        if total > 0:
            eff_fts_weight = config.fts_weight / total
            eff_vec_weight = config.vector_weight / total
        else:
            eff_fts_weight = 0.5
            eff_vec_weight = 0.5
        
        # Calculate fetch limits (get more candidates for better fusion)
        fts_limit = limit * config.fts_limit_multiplier
        vec_limit = limit * config.vector_limit_multiplier
        
        # Get FTS results
        fts_results = self.search_fts(
            query, limit=fts_limit, include_deleted=include_deleted
        )
        
        # Get vector results if available
        vector_results = []
        if self._embedding_provider:
            try:
                vector_results = self.search_vector(
                    query, limit=vec_limit, include_deleted=include_deleted
                )
            except MemoryStoreError:
                pass
        
        # Score normalization
        if config.normalize_scores:
            if fts_results:
                max_fts = max(r.score for r in fts_results)
                min_fts = min(r.score for r in fts_results)
                range_fts = max_fts - min_fts
                for r in fts_results:
                    r.score = (r.score - min_fts) / range_fts if range_fts > 0 else 0.5
            
            if vector_results:
                max_vec = max(r.score for r in vector_results)
                min_vec = min(r.score for r in vector_results)
                range_vec = max_vec - min_vec
                for r in vector_results:
                    r.score = (r.score - min_vec) / range_vec if range_vec > 0 else 0.5
        
        # Reciprocal Rank Fusion (RRF) with weighted scores
        scores: dict[str, float] = {}
        memories: dict[str, Memory] = {}
        fts_scores: dict[str, float] = {}
        vec_scores: dict[str, float] = {}
        
        # Track individual scores for debugging
        for i, r in enumerate(fts_results):
            rrf_score = 1.0 / (60 + i + 1)  # RRF with k=60
            weighted_score = rrf_score * eff_fts_weight + r.score * eff_fts_weight
            scores[r.memory.id] = weighted_score
            memories[r.memory.id] = r.memory
            fts_scores[r.memory.id] = r.score
        
        for i, r in enumerate(vector_results):
            rrf_score = 1.0 / (60 + i + 1)  # RRF with k=60
            weighted_score = rrf_score * eff_vec_weight + r.score * eff_vec_weight
            if r.memory.id in scores:
                scores[r.memory.id] += weighted_score
            else:
                scores[r.memory.id] = weighted_score
                memories[r.memory.id] = r.memory
            vec_scores[r.memory.id] = r.score
        
        # Filter by minimum threshold
        if config.min_score_threshold > 0:
            scores = {k: v for k, v in scores.items() if v >= config.min_score_threshold}
        
        # Sort and limit
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Rerank with cross-encoder if requested
        if rerank and self._reranker:
            rerank_k = rerank_top_k or min(limit * 3, len(sorted_ids))
            candidates = sorted_ids[:rerank_k]
            
            if candidates:
                docs = [memories[mid].content for mid in candidates]
                reranked = self._reranker.rerank(query, docs, top_k=limit)
                
                results = [
                    SearchResult(
                        memory=memories[candidates[idx]], 
                        score=score, 
                        match_type="hybrid_reranked"
                    )
                    for idx, score in reranked
                ]
                return results
        
        # Return without reranking
        return [
            SearchResult(memory=memories[mid], score=scores[mid], match_type="hybrid")
            for mid in sorted_ids[:limit]
        ]

    def search(
        self,
        query: str,
        limit: int = 10,
        method: str | SearchMethod = "auto",
        rerank: bool = False,
        **kwargs,
    ) -> list[SearchResult]:
        """
        Search memories with automatic method selection and optional reranking.

        Args:
            query: Search query string
            limit: Maximum number of results
            method: Search method ("fts", "vector", "hybrid", or "auto")
            rerank: Use cross-encoder reranking (requires reranker)
            **kwargs: Additional arguments passed to specific search methods

        Returns:
            List of SearchResult objects
        """
        # Convert string to enum if needed
        if isinstance(method, str):
            method = SearchMethod(method.lower())
        
        if method == SearchMethod.AUTO:
            # Use hybrid if vector search is available, otherwise FTS
            if self._embedding_provider:
                method = SearchMethod.HYBRID
            else:
                method = SearchMethod.FTS
        
        if method == SearchMethod.FTS:
            results = self.search_fts(query, limit=limit, **kwargs)
        elif method == SearchMethod.VECTOR:
            results = self.search_vector(query, limit=limit, **kwargs)
        elif method == SearchMethod.HYBRID:
            results = self.search_hybrid(query, limit=limit, rerank=rerank, **kwargs)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        # Apply reranking for non-hybrid methods if requested
        if rerank and self._reranker and method != SearchMethod.HYBRID:
            docs = [r.memory.content for r in results]
            if docs:
                reranked = self._reranker.rerank(query, docs, top_k=limit)
                results = [
                    SearchResult(
                        memory=results[idx].memory,
                        score=score,
                        match_type=f"{results[idx].match_type}_reranked"
                    )
                    for idx, score in reranked
                ]
        
        return results

    def embed_all(
        self,
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        overwrite: bool = False,
    ) -> int:
        """
        Generate embeddings for all memories that don't have them.

        Args:
            batch_size: Number of memories to embed at once
            progress_callback: Called with (processed, total) for progress
            overwrite: If True, regenerate embeddings even if they exist

        Returns:
            Number of memories embedded

        Raises:
            MemoryStoreError: If embedding provider not configured
        """
        if not self._embedding_provider:
            raise MemoryStoreError(
                "Embedding provider required. "
                "Either provide an embedding_provider or set auto_embed=True."
            )
        
        # Get memories to embed
        if overwrite:
            sql = """
                SELECT id, content, metadata_json, embedding_blob,
                       created_at, updated_at, version, is_deleted
                FROM memories
                WHERE branch = ? AND is_deleted = 0
            """
        else:
            sql = """
                SELECT id, content, metadata_json, embedding_blob,
                       created_at, updated_at, version, is_deleted
                FROM memories
                WHERE branch = ? AND is_deleted = 0 AND embedding_blob IS NULL
            """
        
        cursor = self._conn.execute(sql, (self._current_branch,))
        rows = cursor.fetchall()
        
        if not rows:
            return 0
        
        total = len(rows)
        processed = 0
        batch_size = batch_size or 32
        
        for i in range(0, total, batch_size):
            batch_rows = rows[i:i + batch_size]
            contents = [row["content"] for row in batch_rows]
            
            # Generate embeddings for batch
            embeddings = self._embedding_provider.encode(contents)
            
            # Update database
            with self.transaction():
                for row, embedding in zip(batch_rows, embeddings):
                    embedding_blob = embedding_to_blob(embedding)
                    self._conn.execute(
                        """
                        UPDATE memories
                        SET embedding_blob = ?
                        WHERE id = ? AND branch = ?
                        """,
                        (embedding_blob, row["id"], self._current_branch),
                    )
            
            processed += len(batch_rows)
            
            if progress_callback:
                progress_callback(processed, total)
        
        logger.info(f"Embedded {processed} memories")
        return processed

    def enable_reranker(
        self,
        reranker: RerankerProvider | None = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: RerankerConfig | None = None,
    ) -> None:
        """
        Enable cross-encoder reranking for improved search accuracy.

        Args:
            reranker: Custom reranker provider
            model_name: Model name for default CrossEncoderReranker
            config: Reranker configuration
        """
        if reranker:
            self._reranker = reranker
        elif CROSS_ENCODER_AVAILABLE:
            self._reranker = CrossEncoderReranker(model_name=model_name, config=config)
        else:
            raise MemoryStoreError(
                "CrossEncoder is required for reranking. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def has_embeddings(self) -> bool:
        """Check if embedding provider is configured."""
        return self._embedding_provider is not None

    @property
    def has_reranker(self) -> bool:
        """Check if reranker is configured."""
        return self._reranker is not None

    @property
    def embedding_dimension(self) -> int | None:
        """Get the embedding dimension if provider is configured."""
        if self._embedding_provider:
            return self._embedding_provider.dimension
        return None

    # ==================== Versioning Operations ====================

    @property
    def current_branch(self) -> str:
        """Get the current branch name."""
        return self._current_branch

    def create_branch(self, name: str, from_branch: str | None = None) -> Branch:
        """
        Create a new branch.

        Args:
            name: Name of the new branch
            from_branch: Branch to copy from (default: current branch)

        Returns:
            The created Branch object
        """
        if from_branch is None:
            from_branch = self._current_branch
        
        # Get the source branch's head commit
        cursor = self._conn.execute(
            "SELECT head_commit_id FROM branches WHERE name = ?",
            (from_branch,)
        )
        row = cursor.fetchone()
        if row is None:
            raise BranchNotFoundError(from_branch)
        
        head_commit_id = row["head_commit_id"]
        
        # Create new branch
        branch = Branch.create(name=name, head_commit_id=head_commit_id)
        
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO branches (name, head_commit_id, created_at, is_active)
                VALUES (?, ?, ?, ?)
                """,
                (branch.name, branch.head_commit_id, branch.created_at.isoformat(), 1),
            )
            
            # Copy memories from source branch
            self._conn.execute(
                """
                INSERT INTO memories (id, content, metadata_json, embedding_blob,
                                     created_at, updated_at, version, is_deleted, branch)
                SELECT id || '-' || ?, content, metadata_json, embedding_blob,
                       created_at, updated_at, version, is_deleted, ?
                FROM memories
                WHERE branch = ?
                """,
                (name, name, from_branch),
            )

        logger.info(f"Created branch '{name}' from '{from_branch}'")
        return branch

    def checkout(self, branch_name: str) -> None:
        """
        Switch to a different branch.

        Args:
            branch_name: Name of the branch to switch to

        Raises:
            BranchNotFoundError: If branch doesn't exist
        """
        cursor = self._conn.execute(
            "SELECT name FROM branches WHERE name = ? AND is_active = 1",
            (branch_name,)
        )
        if cursor.fetchone() is None:
            raise BranchNotFoundError(branch_name)
        
        self._current_branch = branch_name
        logger.info(f"Switched to branch '{branch_name}'")

    def list_branches(self) -> list[Branch]:
        """List all branches."""
        cursor = self._conn.execute(
            "SELECT name, head_commit_id, created_at, is_active FROM branches"
        )
        branches = []
        for row in cursor.fetchall():
            branches.append(Branch(
                name=row["name"],
                head_commit_id=row["head_commit_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                is_active=bool(row["is_active"]),
            ))
        return branches

    def delete_branch(self, name: str) -> None:
        """
        Delete a branch.

        Args:
            name: Name of the branch to delete

        Raises:
            BranchNotFoundError: If branch doesn't exist
            MemoryStoreError: If trying to delete main branch
        """
        if name == "main":
            raise MemoryStoreError("Cannot delete the main branch")
        
        cursor = self._conn.execute(
            "SELECT name FROM branches WHERE name = ?",
            (name,)
        )
        if cursor.fetchone() is None:
            raise BranchNotFoundError(name)
        
        with self.transaction():
            # Delete memories on this branch
            self._conn.execute("DELETE FROM memories WHERE branch = ?", (name,))
            # Delete the branch
            self._conn.execute("DELETE FROM branches WHERE name = ?", (name,))
        
        # Switch to main if we deleted the current branch
        if self._current_branch == name:
            self._current_branch = "main"
        
        logger.info(f"Deleted branch '{name}'")

    def commit(self, message: str) -> Commit:
        """
        Create a commit snapshot of the current branch state.

        Args:
            message: Commit message

        Returns:
            The created Commit object
        """
        # Get current head commit
        cursor = self._conn.execute(
            "SELECT head_commit_id FROM branches WHERE name = ?",
            (self._current_branch,)
        )
        row = cursor.fetchone()
        parent_id = row["head_commit_id"] if row else None
        
        # Get current memory snapshot
        cursor = self._conn.execute(
            "SELECT id, version FROM memories WHERE branch = ? AND is_deleted = 0",
            (self._current_branch,)
        )
        memory_snapshot = {row["id"]: row["version"] for row in cursor.fetchall()}
        
        # Create commit
        commit = Commit.create(
            branch=self._current_branch,
            parent_id=parent_id,
            message=message,
            memory_snapshot=memory_snapshot,
        )
        
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO commits (id, branch, parent_id, message, created_at, memory_snapshot_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    commit.id,
                    commit.branch,
                    commit.parent_id,
                    commit.message,
                    commit.created_at.isoformat(),
                    json.dumps(commit.memory_snapshot),
                ),
            )
            
            # Update branch head
            self._conn.execute(
                "UPDATE branches SET head_commit_id = ? WHERE name = ?",
                (commit.id, self._current_branch),
            )

        logger.info(f"Created commit {commit.id[:8]}: {message}")
        return commit

    def get_history(self, limit: int = 50) -> list[Commit]:
        """
        Get commit history for the current branch.

        Args:
            limit: Maximum number of commits to return

        Returns:
            List of Commit objects, most recent first
        """
        cursor = self._conn.execute(
            """
            SELECT id, branch, parent_id, message, created_at, memory_snapshot_json
            FROM commits
            WHERE branch = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (self._current_branch, limit),
        )
        
        commits = []
        for row in cursor.fetchall():
            commits.append(Commit(
                id=row["id"],
                branch=row["branch"],
                parent_id=row["parent_id"],
                message=row["message"],
                created_at=datetime.fromisoformat(row["created_at"]),
                memory_snapshot=json.loads(row["memory_snapshot_json"]),
            ))
        return commits

    def rollback(self, commit_id: str) -> None:
        """
        Rollback the current branch to a specific commit.

        Args:
            commit_id: The commit ID to rollback to

        Raises:
            CommitNotFoundError: If commit doesn't exist
        """
        # Get the commit
        cursor = self._conn.execute(
            "SELECT id, memory_snapshot_json FROM commits WHERE id = ? AND branch = ?",
            (commit_id, self._current_branch),
        )
        row = cursor.fetchone()
        if row is None:
            raise CommitNotFoundError(commit_id)
        
        memory_snapshot = json.loads(row["memory_snapshot_json"])
        
        with self.transaction():
            # Get all current memories
            cursor = self._conn.execute(
                "SELECT id, version FROM memories WHERE branch = ?",
                (self._current_branch,)
            )
            current_memories = {r["id"]: r["version"] for r in cursor.fetchall()}
            
            # Process each memory
            for memory_id, target_version in memory_snapshot.items():
                if memory_id in current_memories:
                    if current_memories[memory_id] != target_version:
                        # Restore from version history
                        self._restore_memory_version(memory_id, target_version)
            
            # Mark memories not in snapshot as deleted
            for memory_id in current_memories:
                if memory_id not in memory_snapshot:
                    self._conn.execute(
                        "UPDATE memories SET is_deleted = 1 WHERE id = ? AND branch = ?",
                        (memory_id, self._current_branch),
                    )

        logger.info(f"Rolled back to commit {commit_id[:8]}")

    def _restore_memory_version(self, memory_id: str, target_version: int) -> None:
        """Restore a memory to a specific version."""
        cursor = self._conn.execute(
            """
            SELECT content, metadata_json, embedding_blob
            FROM memory_versions
            WHERE memory_id = ? AND version = ?
            """,
            (memory_id, target_version),
        )
        row = cursor.fetchone()
        if row:
            self._conn.execute(
                """
                UPDATE memories
                SET content = ?, metadata_json = ?, embedding_blob = ?,
                    version = ?, updated_at = ?, is_deleted = 0
                WHERE id = ? AND branch = ?
                """,
                (
                    row["content"],
                    row["metadata_json"],
                    row["embedding_blob"],
                    target_version,
                    datetime.utcnow().isoformat(),
                    memory_id,
                    self._current_branch,
                ),
            )

    def get_memory_history(self, memory_id: str) -> list[dict[str, Any]]:
        """
        Get version history for a specific memory.

        Args:
            memory_id: The memory ID

        Returns:
            List of version records
        """
        cursor = self._conn.execute(
            """
            SELECT version, content, metadata_json, created_at, operation
            FROM memory_versions
            WHERE memory_id = ?
            ORDER BY version DESC
            """,
            (memory_id,),
        )
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "version": row["version"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
                "operation": row["operation"],
            })
        return history

    # ==================== Export/Import Operations ====================

    def export_json(self, path: str | Path | None = None) -> str:
        """
        Export all memories to JSON.

        Args:
            path: Optional file path to write to

        Returns:
            JSON string of exported data
        """
        data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "branch": self._current_branch,
            "memories": [],
            "commits": [],
        }
        
        # Export memories
        memories = self.list(limit=100000, include_deleted=True)
        data["memories"] = [m.to_dict() for m in memories]
        
        # Export commits
        commits = self.get_history(limit=10000)
        data["commits"] = [c.to_dict() for c in commits]
        
        json_str = json.dumps(data, indent=2)
        
        if path:
            Path(path).write_text(json_str)
            logger.info(f"Exported {len(memories)} memories to {path}")
        
        return json_str

    def import_json(
        self,
        data: str | dict[str, Any] | Path,
        merge: bool = False,
    ) -> int:
        """
        Import memories from JSON.

        Args:
            data: JSON string, dict, or path to JSON file
            merge: If True, merge with existing; if False, replace

        Returns:
            Number of memories imported
        """
        if isinstance(data, Path):
            data = json.loads(data.read_text())
        elif isinstance(data, str):
            data = json.loads(data)
        
        if not merge:
            # Clear existing memories on current branch
            with self.transaction():
                # Get memory IDs first
                cursor = self._conn.execute(
                    "SELECT id FROM memories WHERE branch = ?",
                    (self._current_branch,)
                )
                memory_ids = [row["id"] for row in cursor.fetchall()]
                
                # Delete versions for these memories
                for mid in memory_ids:
                    self._conn.execute(
                        "DELETE FROM memory_versions WHERE memory_id = ?",
                        (mid,)
                    )
                
                # Then delete the memories
                self._conn.execute(
                    "DELETE FROM memories WHERE branch = ?",
                    (self._current_branch,)
                )
        
        count = 0
        for mem_data in data.get("memories", []):
            memory = Memory.from_dict(mem_data)
            
            with self.transaction():
                embedding_blob = (
                    embedding_to_blob(memory.embedding) if memory.embedding else None
                )
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO memories
                    (id, content, metadata_json, embedding_blob, created_at, updated_at,
                     version, is_deleted, branch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.id,
                        memory.content,
                        memory.metadata.to_json(),
                        embedding_blob,
                        memory.created_at.isoformat(),
                        memory.updated_at.isoformat(),
                        memory.version,
                        int(memory.is_deleted),
                        self._current_branch,
                    ),
                )
            count += 1
        
        logger.info(f"Imported {count} memories")
        return count
