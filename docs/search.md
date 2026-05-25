# Search Guide

Agent Memory Toolkit provides production-grade search capabilities for finding relevant memories. This guide covers the three main search paradigms and how to combine them for best results.

## Overview

The search module (`agent_memory_toolkit.search`) offers:

| Search Type | Best For | Key Feature |
|-------------|----------|-------------|
| **Semantic** | Meaning-based retrieval | Query expansion, embeddings |
| **Hybrid** | Balanced keyword + semantic | BM25 + vector fusion |
| **Faceted** | Filtered exploration | Multi-dimensional filters |
| **Re-ranking** | Precision improvement | Cross-encoder scoring |

## Quick Start

```python
from agent_memory_toolkit.search import (
    SemanticSearchEngine,
    HybridSearchEngine,
    FilterBuilder,
    CrossEncoderReranker,
)

# Initialize with your embedding provider
from agent_memory_toolkit.store import SentenceTransformerProvider
embedding_provider = SentenceTransformerProvider("all-MiniLM-L6-v2")

# Prepare your documents
documents = [
    {"id": "1", "content": "How to reset your password", "metadata": {"tags": ["auth"]}},
    {"id": "2", "content": "Two-factor authentication setup", "metadata": {"tags": ["auth", "security"]}},
    {"id": "3", "content": "API rate limiting configuration", "metadata": {"tags": ["api"]}},
]

# Semantic search
semantic = SemanticSearchEngine(embedding_provider=embedding_provider)
results = semantic.search("login issues", documents)

# Hybrid search (recommended for most use cases)
hybrid = HybridSearchEngine(embedding_provider=embedding_provider)
hybrid.index(documents)
results = hybrid.search("authentication problems")
```

## Semantic Search

Semantic search finds memories based on meaning, not just keywords. It uses embeddings to understand query intent and find conceptually related content.

### Basic Usage

```python
from agent_memory_toolkit.search import SemanticSearchEngine, SemanticSearchConfig

# Configure search
config = SemanticSearchConfig(
    embedding_model="all-MiniLM-L6-v2",
    top_k=10,
    min_similarity=0.5,
    enable_query_expansion=True,
)

engine = SemanticSearchEngine(
    embedding_provider=embedding_provider,
    config=config,
)

# Search
results = engine.search(
    query="How do I change my password?",
    documents=memories,
)

for match in results:
    print(f"{match.memory_id}: {match.similarity_score:.3f}")
    print(f"  {match.content[:100]}...")
```

### Query Expansion

Query expansion automatically generates alternative phrasings to improve recall:

```python
from agent_memory_toolkit.search import QueryAnalyzer, SynonymQueryExpander

# Synonym-based expansion
expander = SynonymQueryExpander()
analyzer = QueryAnalyzer(query_expander=expander)

analysis = analyzer.analyze("delete user account")
print(analysis.expanded_queries)
# ['remove user account', 'erase user account', ...]

# LLM-based expansion (requires OpenAI client)
from agent_memory_toolkit.search import LLMQueryExpander

llm_expander = LLMQueryExpander(llm_client=openai_client)
analysis = analyzer.analyze("authentication errors", expand=True)
```

### Embedding Caching

For performance, embeddings are cached automatically:

```python
from agent_memory_toolkit.search import EmbeddingCache

# Custom cache configuration
cache = EmbeddingCache(max_size=50000)

engine = SemanticSearchEngine(
    embedding_provider=embedding_provider,
    embedding_cache=cache,
)
```

### Finding Similar Documents

```python
# Find documents similar to a reference
reference = memories[0]
similar = engine.find_similar(
    reference_embedding=reference["embedding"],
    documents=memories,
    exclude_ids={reference["id"]},
    top_k=5,
)
```

## Hybrid Search

Hybrid search combines BM25 lexical search with vector semantic search, giving you the best of both worlds:
- **BM25**: Excellent for exact term matches, names, codes
- **Vector**: Excellent for conceptual/semantic similarity

### Basic Usage

```python
from agent_memory_toolkit.search import HybridSearchEngine, HybridSearchConfig, FusionStrategy

config = HybridSearchConfig(
    lexical_weight=0.3,      # Weight for BM25
    semantic_weight=0.7,     # Weight for vector search
    fusion_strategy=FusionStrategy.RRF,  # Reciprocal Rank Fusion
    top_k=10,
)

engine = HybridSearchEngine(
    embedding_provider=embedding_provider,
    config=config,
)

# Index documents (required before search)
engine.index(documents, content_field="content")

# Search
results = engine.search("configure OAuth authentication")

for match in results:
    print(f"{match.memory_id}: {match.combined_score:.3f}")
    print(f"  Lexical: {match.lexical_score:.3f}, Semantic: {match.semantic_score:.3f}")
    print(f"  Source: {match.source}")  # 'hybrid', 'lexical', or 'semantic'
```

### Fusion Strategies

Choose the right fusion strategy for your use case:

```python
from agent_memory_toolkit.search import FusionStrategy

# Reciprocal Rank Fusion (default, recommended)
# Robust to score miscalibration between rankers
config = HybridSearchConfig(fusion_strategy=FusionStrategy.RRF)

# Linear combination
# When you trust score calibration
config = HybridSearchConfig(fusion_strategy=FusionStrategy.LINEAR)

# Max score
# When either signal is reliable
config = HybridSearchConfig(fusion_strategy=FusionStrategy.MAX_SCORE)

# Distribution-based
# For better score calibration
config = HybridSearchConfig(fusion_strategy=FusionStrategy.DISTRIBUTION)
```

### BM25 Configuration

Fine-tune BM25 for your corpus:

```python
config = HybridSearchConfig(
    bm25_k1=1.5,   # Term frequency saturation (1.2-2.0)
    bm25_b=0.75,   # Document length normalization (0-1)
)
```

### Standalone BM25

```python
from agent_memory_toolkit.search import BM25

bm25 = BM25(k1=1.5, b=0.75)
bm25.fit([doc["content"] for doc in documents])

# Get scores for all documents
scores = bm25.get_scores("OAuth configuration")

# Get top-k results
top_results = bm25.get_top_k("OAuth configuration", k=10)
```

## Faceted Search

Faceted search enables powerful filtering and exploration with multi-dimensional constraints.

### Building Filters

```python
from agent_memory_toolkit.search import FilterBuilder, FilterOperator

# Fluent filter building
filters = (FilterBuilder()
    .has_tag("security")                    # Tag filter
    .min_confidence(0.8)                    # Confidence threshold
    .from_source("user_conversation")       # Source filter
    .created_after("2024-01-01")           # Date filter
    .build()
)

# Or with explicit operators
filters = (FilterBuilder()
    .where("metadata.tags", FilterOperator.CONTAINS, "important")
    .where("metadata.confidence", FilterOperator.GREATER_THAN_OR_EQUAL, 0.9)
    .where("content", FilterOperator.REGEX, r"password|auth", case_sensitive=False)
    .build()
)
```

### Boolean Logic

```python
# Complex boolean expressions
filters = (FilterBuilder()
    .or_group()
        .has_tag("security")
        .has_tag("authentication")
    .end_group()
    .min_confidence(0.8)
    .build()
)

# Equivalent to: (tag=security OR tag=authentication) AND confidence >= 0.8
```

### Faceted Search Engine

```python
from agent_memory_toolkit.search import FacetedSearchEngine, FacetedSearchConfig

config = FacetedSearchConfig(
    facet_fields=["metadata.tags", "metadata.source"],
    max_facet_values=50,
    include_facets=True,
)

engine = FacetedSearchEngine(config=config)

results = engine.search(
    documents=memories,
    filters=filters,
    query="password",  # Optional text query
    include_facets=True,
)

# Access results
print(f"Total: {results.total_count}, Filtered: {results.filtered_count}")

for match in results.matches:
    print(match["content"])

# Use facets for refinement
for field, facet in results.facets.items():
    print(f"\n{facet.display_name}:")
    for value in facet.top_values(5):
        print(f"  {value.value}: {value.count}")
```

### Available Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `EQUALS` | Exact match | `name = "John"` |
| `NOT_EQUALS` | Not equal | `status != "deleted"` |
| `CONTAINS` | Contains value | `tags contains "urgent"` |
| `STARTS_WITH` | String prefix | `name starts with "Dr."` |
| `GREATER_THAN` | Numeric > | `confidence > 0.8` |
| `LESS_THAN` | Numeric < | `priority < 5` |
| `BETWEEN` | Range | `date between [start, end]` |
| `IN` | Value in list | `category in ["A", "B"]` |
| `EXISTS` | Field exists | `metadata.source exists` |
| `REGEX` | Pattern match | `content ~ /error.*/i` |

### Convenience Function

```python
from agent_memory_toolkit.search import build_filter, filter_documents

# Quick filter building
filters = build_filter(
    tags=["security", "authentication"],
    min_confidence=0.8,
    created_after="2024-01-01",
)

# Apply filters directly
filtered = filter_documents(documents, filters)
```

## Re-ranking

Re-ranking improves precision by using more expensive models on a smaller candidate set.

### Cross-Encoder Re-ranking

Cross-encoders jointly encode query and document, enabling much more accurate relevance scoring:

```python
from agent_memory_toolkit.search import CrossEncoderReranker

# Initialize reranker
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size=32,
)

# Get initial candidates (from hybrid search, etc.)
candidates = hybrid.search("OAuth configuration", top_k=50)

# Re-rank top candidates
reranked = reranker.rerank(
    query="How to configure OAuth authentication?",
    documents=[c.content for c in candidates],
    top_k=10,
)

# Results are (original_index, score) tuples
for idx, score in reranked:
    print(f"{score:.3f}: {candidates[idx].content[:80]}...")
```

### Reciprocal Rank Fusion

Combine multiple rankings (e.g., from different models or queries):

```python
from agent_memory_toolkit.search import RRFFusion, fuse_rankings

# Rankings from different sources
bm25_ranking = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.6)]
vector_ranking = [("doc2", 0.95), ("doc1", 0.8), ("doc4", 0.7)]

# Fuse with RRF
fused = fuse_rankings(
    rankings=[bm25_ranking, vector_ranking],
    weights=[0.4, 0.6],  # Optional weights
    k=60,                # RRF smoothing parameter
    top_k=10,
)

for doc_id, rrf_score in fused:
    print(f"{doc_id}: {rrf_score:.4f}")
```

### Diversity (MMR)

Maximal Marginal Relevance reduces redundancy in results:

```python
from agent_memory_toolkit.search import MMRDiversifier, diversify_results

# Prepare items with embeddings
items = [
    (doc, score, embedding)
    for doc, score, embedding in zip(documents, scores, embeddings)
]

# Diversify
diversifier = MMRDiversifier(
    lambda_param=0.7,  # 0 = max diversity, 1 = max relevance
)

diverse_results = diversifier.diversify(items, top_k=10)

for result in diverse_results:
    print(f"Score: {result.score:.3f}, Penalty: {result.diversity_penalty:.3f}")
    print(f"Final: {result.final_score:.3f}")
```

### Ranking Evaluation

Evaluate your ranking quality:

```python
from agent_memory_toolkit.search import RankingEvaluator

ranked_ids = [r.memory_id for r in results]
relevant_ids = {"doc1", "doc3", "doc5"}  # Ground truth

# NDCG@10
ndcg = RankingEvaluator.ndcg(ranked_ids, relevant_ids, k=10)

# Mean Reciprocal Rank
mrr = RankingEvaluator.mrr(ranked_ids, relevant_ids)

# Precision@5
p_at_5 = RankingEvaluator.precision_at_k(ranked_ids, relevant_ids, k=5)

# Recall@10
r_at_10 = RankingEvaluator.recall_at_k(ranked_ids, relevant_ids, k=10)

print(f"NDCG@10: {ndcg:.3f}")
print(f"MRR: {mrr:.3f}")
print(f"P@5: {p_at_5:.3f}")
print(f"R@10: {r_at_10:.3f}")
```

## Integration with MemoryStore

The search module integrates seamlessly with `MemoryStore`:

```python
from agent_memory_toolkit import MemoryStore
from agent_memory_toolkit.search import HybridSearchEngine, FilterBuilder

# Create store with auto-embedding
store = MemoryStore("memories.db", auto_embed=True)

# Add some memories
store.add("Reset password via email link", metadata={"tags": ["auth"]})
store.add("OAuth 2.0 authentication flow", metadata={"tags": ["auth", "oauth"]})
store.add("API rate limiting is 100 req/min", metadata={"tags": ["api"]})

# Use store's built-in hybrid search
results = store.search("authentication", method="hybrid")

# Or use advanced search module for more control
memories = store.list(limit=1000)
documents = [m.to_dict() for m in memories]

hybrid = HybridSearchEngine(embedding_provider=store._embedding_provider)
hybrid.index(documents, content_field="content")

# Search with filters
filters = FilterBuilder().has_tag("auth").min_confidence(0.7).build()
results = hybrid.search("OAuth setup")
```

## Performance Tips

### 1. Cache Embeddings

```python
# Embedding cache prevents recomputation
engine = SemanticSearchEngine(
    embedding_provider=provider,
    config=SemanticSearchConfig(
        cache_embeddings=True,
        cache_size=50000,
    ),
)
```

### 2. Batch Processing

```python
# Encode all documents at once during indexing
embeddings = embedding_provider.encode(
    [doc["content"] for doc in documents],
    batch_size=64,
)

hybrid.index(documents, embeddings=embeddings)
```

### 3. Limit Re-ranking Candidates

```python
# Re-rank only top candidates
results = hybrid.search(query, top_k=100)  # Get more candidates
reranked = reranker.rerank(query, [r.content for r in results], top_k=10)
```

### 4. Use RRF for Fusion

RRF is more robust than linear fusion and doesn't require score calibration:

```python
config = HybridSearchConfig(
    fusion_strategy=FusionStrategy.RRF,
    rrf_k=60,
)
```

## Model Recommendations

### Embedding Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | General purpose |
| `all-mpnet-base-v2` | 768 | Medium | Better | Production |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Good | Q&A focused |

### Cross-Encoder Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast | Good | General re-ranking |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Medium | Better | Production |
| `cross-encoder/stsb-roberta-base` | Medium | Good | Semantic similarity |

## Next Steps

- [Memory Store Documentation](./store.md) - Core memory operations
- [API Reference](./api.md) - Complete API documentation
- [Examples](../examples/) - Working code examples
