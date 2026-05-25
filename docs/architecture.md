# Architecture

Deep dive into Agent Memory Toolkit's design, components, and data flows.

---

## Overview

Agent Memory Toolkit is designed as a modular, local-first memory system for AI agents. The architecture prioritizes:

1. **Local-first operation** — No required cloud dependencies
2. **Modularity** — Each component works independently
3. **Extensibility** — Easy to add custom strategies and providers
4. **Performance** — Optimized for low-latency agent interactions
5. **Security** — Defense-in-depth for memory integrity

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Application Layer                                │
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │                         Agent Memory API                          │     │
│    │                                                                    │     │
│    │  MemoryExtractor  │  MemoryGuard  │  ContextCompressor            │     │
│    │  TeamMemoryStore  │  MemoryStore                                   │     │
│    └────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│   Extraction Module  │  │  Security Module │  │  Compression Module  │
│                      │  │                  │  │                      │
│ ┌──────────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────────┐ │
│ │  RuleExtractor   │ │  │ │PoisonDetector│ │  │ │  TokenCounter    │ │
│ └──────────────────┘ │  │ └──────────────┘ │  │ └──────────────────┘ │
│ ┌──────────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────────┐ │
│ │   LLMExtractor   │ │  │ │ConfidenceScr│ │  │ │ImportanceRanker  │ │
│ └──────────────────┘ │  │ └──────────────┘ │  │ └──────────────────┘ │
│ ┌──────────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────────┐ │
│ │  Deduplicator    │ │  │ │ SourceValid  │ │  │ │   Strategies     │ │
│ └──────────────────┘ │  │ └──────────────┘ │  │ └──────────────────┘ │
│ ┌──────────────────┐ │  │ ┌──────────────┐ │  │                      │
│ │ConflictResolver  │ │  │ │ AuditLogger  │ │  │                      │
│ └──────────────────┘ │  │ └──────────────┘ │  │                      │
└──────────────────────┘  └──────────────────┘  └──────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│          Storage Module          │  │           Team Module            │
│                                  │  │                                  │
│ ┌──────────────────────────────┐ │  │ ┌──────────────────────────────┐ │
│ │          SQLite DB           │ │  │ │       TeamMemoryStore        │ │
│ │   ┌─────────┐ ┌─────────┐   │ │  │ │                              │ │
│ │   │Memories │ │Branches │   │ │  │ │ ┌────────────────────────┐   │ │
│ │   └─────────┘ └─────────┘   │ │  │ │ │    Access Control      │   │ │
│ │   ┌─────────┐ ┌─────────┐   │ │  │ │ └────────────────────────┘   │ │
│ │   │ FTS5    │ │Versions │   │ │  │ │ ┌────────────────────────┐   │ │
│ │   └─────────┘ └─────────┘   │ │  │ │ │   Sync Protocol        │   │ │
│ └──────────────────────────────┘ │  │ │ └────────────────────────┘   │ │
│ ┌──────────────────────────────┐ │  │ │ ┌────────────────────────┐   │ │
│ │      Embedding Provider      │ │  │ │ │   Event Hooks          │   │ │
│ │   (sentence-transformers)    │ │  │ │ └────────────────────────┘   │ │
│ └──────────────────────────────┘ │  │ └──────────────────────────────┘ │
└──────────────────────────────────┘  └──────────────────────────────────┘
```

---

## Module Design

### 1. Extraction Module

Converts unstructured text into structured `Memory` objects across six cognitive domains.

#### Six Cognitive Domains

| Domain | Rationale |
|--------|-----------|
| Biography | Core identity rarely changes, high confidence |
| Preferences | Influences personalization, moderate confidence |
| Work | Professional context, frequently referenced |
| Social | Relationships, requires deduplication |
| Temporal | Time-sensitive, needs freshness tracking |
| Procedural | Complex structures, may need versioning |

#### Hybrid Extraction Flow

```
Text Input
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  │
┌─────────┐     ┌─────────┐              │
│  Rule   │     │   LLM   │              │
│Extractor│     │Extractor│              │
└────┬────┘     └────┬────┘              │
     │               │                    │
     └───────┬───────┘                    │
             ▼                            │
     ┌──────────────┐                     │
     │   Merger     │◄────────────────────┘
     │ (Dedupe +    │
     │  Conflict)   │
     └──────────────┘
             │
             ▼
     Structured Memories
```

- **Rule-based** handles explicit patterns (names, dates, locations)
- **LLM-based** captures implicit information and context
- **Hybrid** uses rules first (fast, cheap) then LLM for gaps

#### Deduplication Strategies

| Strategy | Description | Speed |
|----------|-------------|-------|
| Exact | Hash-based | Fastest |
| Fuzzy | Jaccard similarity | Medium |
| Semantic | Embedding cosine | Slowest |

---

### 2. Storage Module

Persistent, searchable memory storage using SQLite.

#### Why SQLite?

| Alternative | Why Not |
|-------------|---------|
| PostgreSQL | Requires server, not local-first |
| MongoDB | Overkill for single-agent use |
| Plain files | No search, no ACID |
| Redis | In-memory, persistence complex |

SQLite provides:

- Zero-configuration deployment
- ACID transactions
- FTS5 for full-text search
- JSON1 for flexible metadata
- WAL mode for concurrent reads

#### Schema Design

```sql
-- Core memories table
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata_json TEXT,
    embedding_blob BLOB,          -- Optional vector
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    is_deleted INTEGER DEFAULT 0, -- Soft delete
    branch TEXT DEFAULT 'main'    -- Git-like branching
);

-- FTS5 virtual table for search
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid'
);

-- Version history for rollback
CREATE TABLE memory_versions (
    id INTEGER PRIMARY KEY,
    memory_id TEXT REFERENCES memories(id),
    content TEXT,
    metadata_json TEXT,
    version INTEGER,
    created_at TEXT,
    operation TEXT  -- create, update, delete
);
```

#### Search Architecture

```
Query
   │
   ├──────────────┬──────────────┐
   ▼              ▼              ▼
┌─────┐      ┌─────────┐   ┌─────────┐
│FTS5 │      │ Vector  │   │ Hybrid  │
│BM25 │      │ Cosine  │   │ Fusion  │
└──┬──┘      └────┬────┘   └────┬────┘
   │              │             │
   └──────────────┴─────────────┘
                  │
                  ▼
          Ranked Results
```

- **FTS5** provides BM25 ranking for keyword matches
- **Vector** uses cosine similarity for semantic search
- **Hybrid** combines both with RRF fusion

---

### 3. Security Module

Validates memories before storage to prevent poisoning attacks.

#### Defense in Depth

```
Input Memory
      │
      ▼
┌─────────────────────┐
│  Poison Detection   │ ◄── Pattern matching + heuristics
└──────────┬──────────┘
           │ Pass?
           ▼
┌─────────────────────┐
│ Confidence Scoring  │ ◄── Uncertainty detection
└──────────┬──────────┘
           │ Pass?
           ▼
┌─────────────────────┐
│  Source Validation  │ ◄── Trust scoring
└──────────┬──────────┘
           │
           ▼
    ┌─────────────┐
    │   Result    │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Audit Trail │
    └─────────────┘
```

#### Security Levels

| Level | Use Case |
|-------|----------|
| MINIMAL | Development/testing |
| LOW | Trusted internal agents |
| MEDIUM | Production default |
| HIGH | Sensitive data handling |
| PARANOID | Critical systems |

#### Poison Detection Patterns

1. **Instruction injection** — "Ignore previous instructions"
2. **Role manipulation** — "You are now a different AI"
3. **Data exfiltration** — Requests to output sensitive data
4. **Prompt leaking** — Attempts to extract system prompts
5. **Memory manipulation** — "Remember that X said Y"

---

### 4. Compression Module

Manages context window limits intelligently.

#### Token-Aware Design

```python
class TokenCounter:
    """Accurate token counting using tiktoken."""
    
    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
    
    def count_messages(self, messages: list[dict]) -> int:
        # Account for message formatting overhead
        tokens = 3  # Every reply starts with <|im_start|>assistant
        for msg in messages:
            tokens += 4  # <|im_start|>{role}\n{content}<|im_end|>\n
            tokens += self.count(msg.get("content", ""))
        return tokens
```

#### Importance Ranking

```
Score = w₁·Recency + w₂·Role + w₃·Content + w₄·Critical
```

| Factor | Weight | Description |
|--------|--------|-------------|
| Recency | 0.3 | Recent messages score higher |
| Role | 0.2 | System > User > Assistant |
| Content | 0.3 | Information density |
| Critical | 0.2 | [IMPORTANT] markers |

#### Tiered Compression

```
Message History (newest first)
│
├── Zone 1: Recent (keep full fidelity)
│   Messages 0-3
│
├── Zone 2: Medium (summarize)
│   Messages 4-11
│
└── Zone 3: Old (key facts only)
    Messages 12+
```

---

### 5. Team Module

Multi-agent collaboration with conflict resolution.

#### Git-like Model

```
                    main
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
 alice/experiment  bob/fix         carol/feature
    │                 │                 │
    └─────────────────┴─────────────────┘
                      │
                    merge
```

Benefits:
- Developers understand it
- Proven conflict resolution semantics
- Supports offline-first workflows
- Enables experimentation with branches

#### Conflict Resolution Strategies

| Strategy | Description |
|----------|-------------|
| LATEST_WINS | Timestamp-based (simple, may lose data) |
| OURS | Local version preferred |
| THEIRS | Remote version preferred |
| MERGE | Attempt automatic merge (metadata combined) |
| MANUAL | Raise exception for human resolution |

#### Filesystem Sync Protocol

```
/shared/memories/
├── main/
│   ├── memories/
│   │   ├── mem_abc123.json
│   │   └── mem_def456.json
│   └── commits/
│       ├── commit_001.json
│       └── commit_002.json
└── branches.json
```

Benefits:
- Works with any shared filesystem (NFS, S3, etc.)
- Human-readable format
- Git can version the sync directory
- No server process required

#### Access Control

```
┌────────────────────────────────────────────┐
│               Namespace                    │
│  ┌──────────────────────────────────────┐  │
│  │         Permissions                  │  │
│  │                                      │  │
│  │  alice: READ, WRITE                  │  │
│  │  bob:   READ                         │  │
│  │  *:     READ (default)               │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

Permissions: `READ`, `WRITE`, `DELETE`, `ADMIN`

---

## Data Flows

### Memory Extraction Flow

```
1. Text Input
       │
       ▼
2. MemoryExtractor.extract()
       │
       ├──► RuleBasedExtractor (patterns)
       │         │
       │         ▼
       │    Raw Memories (rule)
       │
       └──► LLMExtractor (if hybrid/llm mode)
                 │
                 ▼
            Raw Memories (llm)
                 │
                 ▼
3. MemoryMerger.merge()
       │
       ├──► MemoryDeduplicator.deduplicate()
       │
       └──► ConflictResolver.resolve()
                 │
                 ▼
4. ExtractionResult
       │
       ▼
5. MemoryStore.add() [optional]
       │
       ▼
6. MemoryGuard.validate() [optional]
       │
       ▼
7. Persisted Memory
```

### Search Flow

```
1. Query
      │
      ▼
2. MemoryStore.search(mode="hybrid")
      │
      ├──► FTS5 Query
      │    │
      │    ▼
      │    BM25 Scores
      │
      └──► Vector Query
           │
           ▼
           Cosine Similarities
                │
                ▼
3. Score Fusion (RRF or weighted)
      │
      ▼
4. Ranked Results
      │
      ▼
5. Return SearchResult[]
```

### Team Sync Flow

```
1. alice.push("/shared")
      │
      ▼
2. SyncProtocol.push()
      │
      ├──► Serialize memories to JSON
      │
      └──► Write to filesystem
                │
                ▼
3. bob.pull("/shared")
      │
      ▼
4. SyncProtocol.pull()
      │
      ├──► Read from filesystem
      │
      ├──► Detect conflicts
      │
      └──► Apply conflict resolution
                │
                ▼
5. Local store updated
```

---

## Extension Points

### Custom Embedding Provider

```python
from agent_memory.store.embeddings import EmbeddingProvider

class MyEmbeddingProvider(EmbeddingProvider):
    def encode(self, texts: list[str]) -> list[list[float]]:
        # Your implementation
        return embeddings

store = MemoryStore(
    "memory.db",
    embedding_provider=MyEmbeddingProvider()
)
```

### Custom Compression Strategy

```python
from agent_memory.compression import CompressionStrategy, CompressionResult

class MyStrategy(CompressionStrategy):
    def compress(
        self,
        messages: list[dict],
        token_budget: int,
        token_counter: TokenCounter
    ) -> CompressionResult:
        # Your implementation
        return CompressionResult(...)

compressor = ContextCompressor()
compressor.add_strategy("my_strategy", MyStrategy())
```

### Custom Extraction Rules

```python
from agent_memory.extraction import RuleBasedExtractor

extractor = RuleBasedExtractor()
extractor.add_pattern(
    domain=CognitiveDomain.WORK,
    pattern=r"using\s+(?P<value>[\w\s]+)\s+framework",
    key="framework"
)
```

---

## Performance Considerations

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| MemoryStore | ~50MB baseline | SQLite in-memory |
| Embeddings | ~100MB per 10K docs | With vector store |
| Compression | ~10MB | Token encoding cache |

### Optimization Tips

1. Use `auto_embed=False` for write-heavy workloads
2. Batch operations with `transaction()` context
3. Index namespaces for large team deployments
4. Set appropriate security level — PARANOID is slow

### Concurrency

| Operation | Thread-Safe | Notes |
|-----------|-------------|-------|
| MemoryStore | Yes | Per-connection locking |
| TeamMemoryStore | Yes | RLock for branch ops |
| MemoryGuard | Yes | Stateless validation |
| ContextCompressor | Yes | No shared state |

---

## Future Enhancements

Planned improvements:

1. **Distributed sync** — CRDT-based for true multi-master
2. **Streaming extraction** — Process text incrementally
3. **Memory consolidation** — Automatic memory merging over time
4. **Vector indices** — HNSW for large-scale similarity search
5. **Plugins** — Hook system for custom processing pipelines
