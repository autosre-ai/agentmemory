# Memory Compression

Agent Memory Toolkit provides comprehensive compression capabilities to reduce memory footprint while preserving information integrity. This module includes lossless binary compression, semantic deduplication, and AI-powered summarization.

## Overview

The compression module consists of four main components:

- **Context Compression**: Compress conversation context within token budgets
- **Lossless Compression**: Binary compression using zlib/gzip/brotli/lz4
- **Semantic Deduplication**: Identify and eliminate duplicate or similar memories
- **AI Summarization**: Consolidate memories using extractive or abstractive summarization

## Quick Start

```python
from agent_memory_toolkit.compression import (
    # Lossless compression
    MemoryCompressor,
    CompressionAlgorithm,
    # Semantic deduplication
    SemanticDeduplicator,
    DeduplicationStrategy,
    MemoryItem,
    # AI summarization
    MemorySummarizer,
    SummarizationStrategy,
    SummaryLevel,
    MemoryEntry,
)

# Lossless compression
compressor = MemoryCompressor()
memory_data = {"content": "Important memory content...", "tags": ["work"]}
compressed = compressor.compress_memory(memory_data, memory_id="mem_123")
print(f"Compressed from {compressed.original_size} to {compressed.compressed_size} bytes")

# Semantic deduplication  
deduplicator = SemanticDeduplicator()
memories = [
    MemoryItem("1", "Python is a programming language"),
    MemoryItem("2", "python is a programming language"),  # Near-duplicate
    MemoryItem("3", "JavaScript is also a language"),
]
result = deduplicator.deduplicate(memories)
print(f"Found {result.duplicates_found} duplicates")

# AI summarization
summarizer = MemorySummarizer()
entries = [
    MemoryEntry("1", "Meeting discussed project timeline..."),
    MemoryEntry("2", "Action items from meeting: ..."),
]
result = summarizer.summarize(entries)
print(f"Compressed {result.original_word_count} to {result.summarized_word_count} words")
```

## Lossless Compression

Binary compression for memory storage without any data loss.

### Compression Algorithms

```python
from agent_memory_toolkit.compression import CompressionAlgorithm, MemoryCompressionConfig

# Available algorithms
CompressionAlgorithm.ZLIB   # Good balance of speed and compression (default)
CompressionAlgorithm.GZIP   # Standard, widely compatible
CompressionAlgorithm.BROTLI # Best compression ratio, requires: pip install brotli
CompressionAlgorithm.LZ4    # Ultra-fast, lower ratio, requires: pip install lz4
CompressionAlgorithm.NONE   # No compression (pass-through)
```

### Basic Usage

```python
from agent_memory_toolkit.compression import (
    MemoryCompressor,
    MemoryCompressionConfig,
    CompressionAlgorithm,
)

# Default configuration (ZLIB)
compressor = MemoryCompressor()

# Compress memory
memory_data = {
    "content": "This is important information that needs to be stored...",
    "metadata": {"source": "user_input", "timestamp": "2024-01-15"},
}
compressed = compressor.compress_memory(memory_data, memory_id="mem_001")

print(f"Original: {compressed.original_size} bytes")
print(f"Compressed: {compressed.compressed_size} bytes")
print(f"Ratio: {compressed.compressed_size / compressed.original_size:.2%}")

# Decompress when needed
restored = compressor.decompress_memory(compressed)
assert restored == memory_data
```

### Configuration Options

```python
from agent_memory_toolkit.compression import MemoryCompressionConfig, CompressionAlgorithm

# Maximum compression (Brotli)
config = MemoryCompressionConfig(
    algorithm=CompressionAlgorithm.BROTLI,
    compression_level=11,  # Maximum (0-11)
    verify_checksum=True,
)

# Fast compression (LZ4)
config = MemoryCompressionConfig(
    algorithm=CompressionAlgorithm.LZ4,
    compression_level=0,  # Fastest
    min_size_bytes=50,  # Don't compress tiny data
)

# Auto-select algorithm based on data
config = MemoryCompressionConfig(
    auto_select=True,  # Choose best algorithm automatically
)

compressor = MemoryCompressor(config=config)
```

### Batch Compression

```python
# Compress multiple memories
memories = [
    ("mem_001", {"content": "First memory..."}),
    ("mem_002", {"content": "Second memory..."}),
    ("mem_003", {"content": "Third memory..."}),
]

compressed_list = compressor.compress_batch(memories)

# Decompress batch
restored_list = compressor.decompress_batch(compressed_list)
for memory_id, data in restored_list:
    print(f"{memory_id}: {data}")
```

### Algorithm Comparison

```python
# Compare compression algorithms for your data
sample_data = {"content": "Your typical memory content..." * 100}

estimates = compressor.estimate_compression(sample_data)

for algo, stats in estimates.items():
    print(f"{algo.value}:")
    print(f"  Ratio: {stats.compression_ratio:.2%}")
    print(f"  Time: {stats.compression_time_ms:.2f}ms")
    print(f"  Space saved: {stats.space_saved_percent:.1f}%")
```

### Statistics

```python
# Get compression statistics
stats = compressor.get_stats()

print(f"Total operations: {stats['total_operations']}")
print(f"Total bytes saved: {stats['total_saved_bytes']}")
print(f"Average ratio: {stats['average_compression_ratio']:.2%}")
```

## Semantic Deduplication

Identify and remove duplicate or similar memories to reduce redundancy.

### Deduplication Strategies

```python
from agent_memory_toolkit.compression import DeduplicationStrategy

# Available strategies
DeduplicationStrategy.EXACT    # Exact content match
DeduplicationStrategy.FUZZY    # Fuzzy text matching (edit distance)
DeduplicationStrategy.SEMANTIC # Semantic similarity (embeddings)
DeduplicationStrategy.HYBRID   # Combine multiple strategies
```

### Basic Usage

```python
from agent_memory_toolkit.compression import (
    SemanticDeduplicator,
    DeduplicationConfig,
    DeduplicationStrategy,
    MemoryItem,
)

# Create deduplicator
deduplicator = SemanticDeduplicator()

# Create memory items
memories = [
    MemoryItem("1", "Python is a programming language"),
    MemoryItem("2", "python is a programming language"),  # Near-duplicate
    MemoryItem("3", "Python is a programming language used for AI"),  # Similar
    MemoryItem("4", "JavaScript is a web programming language"),  # Different
]

# Find duplicates
result = deduplicator.deduplicate(memories)

print(f"Original: {result.original_count} memories")
print(f"After dedup: {result.deduplicated_count} memories")
print(f"Reduction: {result.reduction_percent:.1f}%")

# Examine duplicate groups
for group in result.duplicate_groups:
    print(f"\nCanonical: {group.canonical.content}")
    for dup in group.duplicates:
        score = group.similarity_scores[dup.memory_id]
        print(f"  Duplicate ({score:.2f}): {dup.content}")
```

### Configuration Options

```python
from agent_memory_toolkit.compression import DeduplicationConfig, DeduplicationStrategy

config = DeduplicationConfig(
    strategy=DeduplicationStrategy.FUZZY,
    fuzzy_threshold=0.85,  # 85% similarity threshold
    normalize_content=True,  # Normalize before comparison
    min_content_length=20,  # Ignore very short content
    auto_merge=True,  # Automatically merge duplicates
    merge_strategy="keep_newest",  # How to merge: keep_newest, keep_oldest, merge_content
)

deduplicator = SemanticDeduplicator(config=config)
```

### Semantic Deduplication with Embeddings

```python
from agent_memory_toolkit.compression import SemanticDeduplicator, DeduplicationConfig, DeduplicationStrategy

# With custom embedding provider
class MyEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        # Your embedding logic
        return [0.1, 0.2, ...]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

config = DeduplicationConfig(
    strategy=DeduplicationStrategy.SEMANTIC,
    semantic_threshold=0.92,  # Higher threshold for semantic similarity
)

deduplicator = SemanticDeduplicator(
    config=config,
    embedding_provider=MyEmbeddingProvider(),
)

# Pre-computed embeddings
memories = [
    MemoryItem("1", "Python programming", embedding=[0.1, 0.2, ...]),
    MemoryItem("2", "Python coding", embedding=[0.12, 0.21, ...]),
]

result = deduplicator.deduplicate(memories)
```

### Find Similar Memories

```python
# Find memories similar to a query
query = MemoryItem("q", "Programming languages for AI")

similar = deduplicator.find_similar(
    query_memory=query,
    memories=all_memories,
    top_k=10,
    threshold=0.7,
)

for memory, similarity in similar:
    print(f"Similarity {similarity:.2f}: {memory.content[:50]}...")
```

### Estimate Reduction

```python
# Estimate potential reduction without applying changes
estimate = deduplicator.estimate_reduction(memories)

print(f"Current count: {estimate['original_count']}")
print(f"Projected count: {estimate['projected_count']}")
print(f"Potential reduction: {estimate['reduction_percent']:.1f}%")
print(f"Estimated bytes saved: {estimate['estimated_bytes_saved']}")
```

## AI Summarization

Consolidate memories using extractive or abstractive summarization.

### Summarization Strategies

```python
from agent_memory_toolkit.compression import SummarizationStrategy, SummaryLevel

# Available strategies
SummarizationStrategy.EXTRACTIVE   # Select key sentences
SummarizationStrategy.ABSTRACTIVE  # Generate new summary (requires LLM)
SummarizationStrategy.HIERARCHICAL # Multi-level summary tree
SummarizationStrategy.INCREMENTAL  # Update summaries as memories arrive

# Summary detail levels
SummaryLevel.BRIEF      # 1-2 sentences
SummaryLevel.STANDARD   # Paragraph
SummaryLevel.DETAILED   # Multiple paragraphs
SummaryLevel.FULL       # Comprehensive
```

### Basic Usage

```python
from datetime import datetime
from agent_memory_toolkit.compression import (
    MemorySummarizer,
    SummarizationConfig,
    SummarizationStrategy,
    SummaryLevel,
    MemoryEntry,
)

# Create summarizer
summarizer = MemorySummarizer()

# Create memory entries
memories = [
    MemoryEntry(
        memory_id="1",
        content="Meeting on Monday discussed Q4 goals. Team agreed on revenue targets.",
        created_at=datetime(2024, 1, 15),
        importance=0.8,
        tags=["meeting", "goals"],
    ),
    MemoryEntry(
        memory_id="2", 
        content="Action items: John to prepare report, Sarah to contact vendors.",
        created_at=datetime(2024, 1, 15),
        importance=0.7,
        tags=["action-items"],
    ),
    MemoryEntry(
        memory_id="3",
        content="Follow-up meeting scheduled for Friday to review progress.",
        created_at=datetime(2024, 1, 15),
        importance=0.5,
        tags=["meeting"],
    ),
]

# Summarize
result = summarizer.summarize(memories, level=SummaryLevel.STANDARD)

print(f"Original: {result.original_word_count} words")
print(f"Summary: {result.summarized_word_count} words")
print(f"Reduction: {result.reduction_percent:.1f}%")
print(f"\nSummary:\n{result.summaries[0].content}")
```

### Abstractive Summarization with LLM

```python
from agent_memory_toolkit.compression import MemorySummarizer, SummarizationConfig, SummarizationStrategy

# LLM provider
class MyLLMProvider:
    def complete(self, prompt: str, max_tokens: int = 500) -> str:
        # Your LLM API call
        return "Generated summary..."
    
    def complete_with_system(self, system: str, prompt: str, max_tokens: int = 500) -> str:
        # Your LLM API call with system prompt
        return "Generated summary..."

config = SummarizationConfig(
    strategy=SummarizationStrategy.ABSTRACTIVE,
    default_level=SummaryLevel.DETAILED,
    max_input_tokens=3000,
)

summarizer = MemorySummarizer(
    config=config,
    llm_provider=MyLLMProvider(),
)

result = summarizer.summarize(memories)
```

### Hierarchical Summarization

Create multi-level summary trees for large memory collections:

```python
from agent_memory_toolkit.compression import MemorySummarizer, SummarizationConfig

config = SummarizationConfig(
    strategy=SummarizationStrategy.HIERARCHICAL,
    chunk_size=5,   # Memories per chunk
    max_depth=3,    # Maximum hierarchy depth
)

summarizer = MemorySummarizer(config=config)

# Create hierarchy
hierarchy = summarizer.create_hierarchy(memories)

# Access different levels
print(f"Root summary: {hierarchy.root.content}")

# Get all summaries at depth 1
level_1 = hierarchy.get_level(1)
for summary in level_1:
    print(f"  - {summary.content[:100]}...")

# Flatten to list
all_summaries = hierarchy.flatten()
```

### Incremental Summarization

Update summaries as new memories arrive without re-processing everything:

```python
from agent_memory_toolkit.compression import MemorySummarizer, MemoryEntry

summarizer = MemorySummarizer()

# Initial memories
initial_memories = [
    MemoryEntry("1", "First batch of information..."),
    MemoryEntry("2", "More context from initial batch..."),
]

# Create initial summary with a key
summary = summarizer.update_incrementally(
    summary_key="project_alpha",
    new_memories=initial_memories,
)

print(f"Initial summary: {summary.content}")

# Later, add new memories
new_memories = [
    MemoryEntry("3", "New information to incorporate..."),
    MemoryEntry("4", "Additional updates..."),
]

# Update the summary
updated_summary = summarizer.update_incrementally(
    summary_key="project_alpha",
    new_memories=new_memories,
)

print(f"Updated summary: {updated_summary.content}")
```

### Summarize by Category

```python
# Categorize memories
memories = [
    MemoryEntry("1", "Meeting notes...", category="meetings"),
    MemoryEntry("2", "Code review feedback...", category="development"),
    MemoryEntry("3", "Another meeting...", category="meetings"),
]

# Summarize each category
results = summarizer.summarize_by_category(memories)

for category, result in results.items():
    print(f"\n{category}:")
    print(f"  {result.summaries[0].content}")
```

### Configuration Options

```python
from agent_memory_toolkit.compression import SummarizationConfig, SummarizationStrategy, SummaryLevel

config = SummarizationConfig(
    # Strategy selection
    strategy=SummarizationStrategy.EXTRACTIVE,
    default_level=SummaryLevel.STANDARD,
    
    # Extractive settings
    max_sentences=10,
    min_sentence_length=20,
    
    # Abstractive settings (when using LLM)
    max_input_tokens=3000,
    
    # Hierarchical settings
    chunk_size=5,
    max_depth=3,
    
    # Incremental settings
    max_summary_length=500,
    
    # Quality settings
    preserve_dates=True,
    preserve_numbers=True,
    preserve_names=True,
)
```

## Combined Pipeline

Use all compression techniques together for maximum footprint reduction:

```python
from agent_memory_toolkit.compression import (
    MemoryCompressor,
    SemanticDeduplicator,
    MemorySummarizer,
    MemoryItem,
    MemoryEntry,
    CompressionAlgorithm,
    DeduplicationStrategy,
    SummarizationStrategy,
)

# Step 1: Deduplicate to remove redundancy
deduplicator = SemanticDeduplicator()
dedup_result = deduplicator.deduplicate(memory_items)
unique_items = [
    item for item in memory_items 
    if item.memory_id not in dedup_result.removed_ids
]

print(f"Dedup: {len(memory_items)} -> {len(unique_items)} memories")

# Step 2: Summarize to consolidate
summarizer = MemorySummarizer()
memory_entries = [
    MemoryEntry(m.memory_id, m.content, m.created_at)
    for m in unique_items
]
summary_result = summarizer.summarize(memory_entries)
summary = summary_result.summaries[0]

print(f"Summarized: {summary_result.original_word_count} -> {summary.word_count} words")

# Step 3: Compress for storage
compressor = MemoryCompressor()
compressed = compressor.compress_memory(
    {"summary": summary.content, "source_ids": summary.source_memory_ids},
    memory_id="consolidated_summary",
)

print(f"Compressed: {compressed.original_size} -> {compressed.compressed_size} bytes")

# Calculate total reduction
original_size = sum(len(m.content.encode()) for m in memory_items)
final_size = compressed.compressed_size
print(f"Total reduction: {(1 - final_size/original_size) * 100:.1f}%")
```

## Performance Considerations

### Lossless Compression

| Algorithm | Speed     | Ratio   | Best For                    |
|-----------|-----------|---------|------------------------------|
| LZ4       | Fastest   | Lowest  | Real-time, streaming data    |
| ZLIB      | Fast      | Good    | General purpose (default)    |
| GZIP      | Fast      | Good    | Compatibility required       |
| BROTLI    | Slowest   | Best    | Cold storage, archives       |

### Deduplication

- **Exact matching**: O(n) with hashing, best for detecting true duplicates
- **Fuzzy matching**: O(n²) comparisons, use for similar content
- **Semantic matching**: Requires embeddings, most accurate but slowest

### Summarization

- **Extractive**: Fast, no external dependencies
- **Abstractive**: Better quality, requires LLM API calls
- **Hierarchical**: Good for large collections, scales well
- **Incremental**: Efficient for streaming updates

## Best Practices

1. **Combine techniques**: Use deduplication first, then summarization, then compression
2. **Choose appropriate thresholds**: Start conservative (high similarity) and adjust
3. **Preserve important metadata**: Always keep timestamps, sources, and importance scores
4. **Verify integrity**: Enable checksum verification for compressed data
5. **Monitor stats**: Track compression ratios and deduplication effectiveness
6. **Test with real data**: Algorithm performance varies by data characteristics
