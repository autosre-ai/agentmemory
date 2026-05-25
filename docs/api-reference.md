# API Reference

Complete reference for Agent Memory Toolkit's Python API.

---

## MemoryStore

The primary interface for storing and searching memories.

### Constructor

```python
from agent_memory_toolkit import MemoryStore

store = MemoryStore(
    db_path: str = "memories.db",
    auto_embed: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"memories.db"` | Path to SQLite database file |
| `auto_embed` | `bool` | `True` | Automatically generate embeddings on add |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformers model name |

### Methods

#### add()

Add a new memory to the store.

```python
memory_id = store.add(
    content: str,
    metadata: dict | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0
) -> str
```

**Returns:** Memory ID (UUID string)

**Example:**
```python
mem_id = store.add(
    content="User prefers dark mode",
    metadata={"category": "preferences"},
    source="onboarding",
    tags=["ui", "settings"],
    confidence=0.95
)
```

---

#### get()

Retrieve a memory by ID.

```python
memory = store.get(memory_id: str) -> Memory | None
```

**Returns:** `Memory` object or `None` if not found

**Example:**
```python
memory = store.get("mem_abc123")
if memory:
    print(memory.content)
```

---

#### update()

Update an existing memory.

```python
store.update(
    memory_id: str,
    content: str | None = None,
    metadata: dict | None = None
) -> bool
```

**Returns:** `True` if successful

**Example:**
```python
store.update(
    "mem_abc123",
    content="User prefers dark mode in all apps",
    metadata={"updated": True}
)
```

---

#### delete()

Delete a memory (soft delete by default).

```python
store.delete(
    memory_id: str,
    hard: bool = False
) -> bool
```

**Returns:** `True` if successful

---

#### search()

Hybrid search combining BM25 and vector similarity.

```python
results = store.search(
    query: str,
    mode: str = "hybrid",
    limit: int = 10,
    min_score: float = 0.0
) -> list[SearchResult]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | - | Search query |
| `mode` | `str` | `"hybrid"` | Search mode: `hybrid`, `fts`, `vector` |
| `limit` | `int` | `10` | Maximum results to return |
| `min_score` | `float` | `0.0` | Minimum relevance score |

**Example:**
```python
results = store.search("vim preferences", mode="hybrid", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

---

#### search_fts()

Full-text search using BM25 ranking.

```python
results = store.search_fts(
    query: str,
    limit: int = 10
) -> list[SearchResult]
```

---

#### search_vector()

Semantic similarity search using embeddings.

```python
results = store.search_vector(
    query: str,
    limit: int = 10
) -> list[SearchResult]
```

---

#### list()

List all memories with pagination.

```python
memories = store.list(
    offset: int = 0,
    limit: int = 100,
    branch: str = "main"
) -> list[Memory]
```

---

#### history()

Get version history for a memory.

```python
history = store.history(memory_id: str) -> list[MemoryVersion]
```

---

## MemoryExtractor

Extract structured memories from unstructured text.

### Constructor

```python
from agent_memory_toolkit import MemoryExtractor, ExtractionMode

extractor = MemoryExtractor(
    mode: ExtractionMode = ExtractionMode.RULE
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `ExtractionMode` | `RULE` | Extraction mode: `RULE`, `LLM`, `HYBRID` |

### Extraction Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `RULE` | Pattern matching only | Fast | Good for explicit info |
| `LLM` | LLM-based extraction | Slow | Best for implicit info |
| `HYBRID` | Rules first, then LLM | Medium | Best overall |

### Methods

#### extract()

Extract memories from text.

```python
result = extractor.extract(
    text: str,
    source: str | None = None
) -> ExtractionResult
```

**Returns:** `ExtractionResult` containing extracted memories

**Example:**
```python
result = extractor.extract("""
I'm Alex Chen, a senior engineer at TechCorp.
I prefer Python and always use dark mode.
""")

for memory in result.memories:
    print(f"[{memory.domain.value}] {memory.key}: {memory.value}")
```

---

## CognitiveDomain

Enumeration of memory domains:

```python
from agent_memory_toolkit import CognitiveDomain

class CognitiveDomain(Enum):
    BIOGRAPHY = "biography"      # Name, birthdate, education
    PREFERENCES = "preferences"  # UI settings, languages, tools
    WORK = "work"                # Projects, company, role
    SOCIAL = "social"            # Family, friends, relationships
    TEMPORAL = "temporal"        # Appointments, schedules
    PROCEDURAL = "procedural"    # Workflows, routines
```

---

## MemoryGuard

Security validation for memory content.

### Constructor

```python
from agent_memory_toolkit import MemoryGuard, SecurityLevel

guard = MemoryGuard(
    level: SecurityLevel = SecurityLevel.MEDIUM
)
```

### Security Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `MINIMAL` | Basic validation | Development/testing |
| `LOW` | Light pattern detection | Trusted internal agents |
| `MEDIUM` | Balanced security | Production default |
| `HIGH` | Strict validation | Sensitive data |
| `PARANOID` | Maximum security | Critical systems |

### Methods

#### validate_content()

Validate content for security issues.

```python
result = guard.validate_content(
    content: str,
    source: str | None = None
) -> ValidationResult
```

**Returns:** `ValidationResult` with safety status

**Example:**
```python
result = guard.validate_content("Ignore previous instructions...")

if not result.is_safe:
    print(f"Blocked: {result.issues}")
else:
    store.add(content)
```

---

#### validate_memory()

Validate a Memory object.

```python
result = guard.validate_memory(memory: Memory) -> ValidationResult
```

---

## ValidationResult

Result of security validation:

```python
@dataclass
class ValidationResult:
    is_safe: bool
    confidence: float
    adjusted_confidence: float
    issues: list[str]
    validation_time_ms: float
```

---

## ContextCompressor

Token-aware context compression.

### Constructor

```python
from agent_memory_toolkit import ContextCompressor, CompressionMode

compressor = ContextCompressor(
    mode: CompressionMode = CompressionMode.BALANCED
)
```

### Compression Modes

| Mode | Description | Compression Ratio |
|------|-------------|-------------------|
| `LOSSLESS` | Keep all content | 1.0 |
| `CONSERVATIVE` | Minimal summarization | 0.7-0.9 |
| `BALANCED` | Balanced approach | 0.4-0.6 |
| `AGGRESSIVE` | Maximum compression | 0.2-0.4 |

### Methods

#### compress()

Compress messages to fit token budget.

```python
result = compressor.compress(
    messages: list[dict],
    max_tokens: int = 4000,
    preserve_system: bool = True,
    preserve_recent: int = 3
) -> CompressionResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[dict]` | - | OpenAI-format messages |
| `max_tokens` | `int` | `4000` | Target token budget |
| `preserve_system` | `bool` | `True` | Keep system message intact |
| `preserve_recent` | `int` | `3` | Keep N most recent messages |

**Example:**
```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Long message..."},
    {"role": "assistant", "content": "Long response..."},
    # ... more messages
]

result = compressor.compress(messages, max_tokens=2000)
print(f"Compressed {result.original_tokens} -> {result.compressed_tokens}")
```

---

#### count_tokens()

Count tokens in text.

```python
count = compressor.count_tokens(text: str) -> int
```

---

## CompressionResult

Result of context compression:

```python
@dataclass
class CompressionResult:
    messages: list[dict]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy_used: str
```

---

## TeamMemoryStore

Multi-agent collaboration with Git-like workflows.

### Constructor

```python
from agent_memory_toolkit.team import TeamMemoryStore

store = TeamMemoryStore(
    db_path: str,
    agent_id: str,
    namespace: str = "default"
)
```

### Methods

#### create_branch()

Create a new branch.

```python
store.create_branch(name: str, from_branch: str = "main") -> str
```

---

#### checkout()

Switch to a branch.

```python
store.checkout(branch: str) -> bool
```

---

#### commit()

Commit current changes.

```python
commit_id = store.commit(message: str) -> str
```

---

#### merge()

Merge a branch into current.

```python
store.merge(
    source_branch: str,
    strategy: MergeStrategy = MergeStrategy.LATEST_WINS
) -> MergeResult
```

### Merge Strategies

| Strategy | Description |
|----------|-------------|
| `LATEST_WINS` | Most recent timestamp wins |
| `OURS` | Local version preferred |
| `THEIRS` | Remote version preferred |
| `MERGE` | Attempt automatic merge |
| `MANUAL` | Raise exception for manual resolution |

---

#### push()

Push to shared directory.

```python
store.push(remote_path: str) -> int  # Returns number of memories pushed
```

---

#### pull()

Pull from shared directory.

```python
store.pull(remote_path: str) -> int  # Returns number of memories pulled
```

---

## Data Types

### Memory

```python
@dataclass
class Memory:
    id: str
    content: str
    metadata: dict
    created_at: datetime
    updated_at: datetime
    version: int
    branch: str
    embedding: list[float] | None
```

### SearchResult

```python
@dataclass
class SearchResult:
    memory: Memory
    score: float
    method: str  # "bm25", "vector", "hybrid"
```

### ExtractedMemory

```python
@dataclass
class ExtractedMemory:
    domain: CognitiveDomain
    key: str
    value: str
    confidence: float
    source_span: tuple[int, int] | None
```

---

## Exceptions

```python
from agent_memory_toolkit.exceptions import (
    MemoryNotFoundError,
    MemoryValidationError,
    BranchNotFoundError,
    MergeConflictError,
    SyncError
)
```

| Exception | Description |
|-----------|-------------|
| `MemoryNotFoundError` | Memory with given ID not found |
| `MemoryValidationError` | Content failed security validation |
| `BranchNotFoundError` | Branch does not exist |
| `MergeConflictError` | Merge conflict with MANUAL strategy |
| `SyncError` | Error during push/pull |
