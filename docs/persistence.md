# Data Persistence

Agent Memory Toolkit provides comprehensive persistence options for storing, importing, and exporting memories. This module supports JSON-based import/export, SQLite for local storage, and PostgreSQL for production deployments.

## Overview

The persistence module consists of three main components:

- **JSON I/O**: Import and export memories to/from JSON and JSONL files
- **SQLite Backend**: Local-first persistence with full-text search
- **PostgreSQL Backend**: Production-ready storage with vector search support

## Quick Start

```python
from agent_memory_toolkit.io import (
    # JSON I/O
    export_to_json,
    import_from_json,
    export_to_jsonl,
    import_from_jsonl,
    # SQLite
    SQLiteBackend,
    SQLiteConfig,
    # PostgreSQL
    PostgresBackend,
    PostgresConfig,
)

# Export memories to JSON
result = export_to_json(
    memories=store.list_all(),
    output_path="backup.json",
    compress=True,
)

# Import from JSON
memories, result = import_from_json("backup.json")

# SQLite storage
backend = SQLiteBackend(SQLiteConfig(database_path="memories.db"))
backend.add({"content": "Important fact", "metadata": {"source": "user"}})

# PostgreSQL storage
config = PostgresConfig(host="localhost", database="memories", user="myuser")
backend = PostgresBackend(config)
```

## JSON Import/Export

### Basic Export

```python
from agent_memory_toolkit.io import JSONExporter, JSONIOConfig, JSONFormat

# Create exporter with custom configuration
config = JSONIOConfig(
    format=JSONFormat.PRETTY,    # Pretty-printed JSON
    compress=True,               # Gzip compress output
    include_embeddings=False,    # Exclude embeddings to save space
    include_versions=True,       # Include version history
    include_branches=True,       # Include branch information
)

exporter = JSONExporter(config=config)

# Export memories
result = exporter.export(
    memories=store.list_all(),
    output_path="memories_backup.json.gz",
    branches=store.list_branches(),
    metadata={"exported_by": "my_agent", "version": "1.0"},
)

print(f"Exported {result.memory_count} memories")
print(f"File size: {result.file_size_bytes} bytes")
print(f"Duration: {result.duration_seconds:.2f}s")
```

### Basic Import

```python
from agent_memory_toolkit.io import JSONImporter, JSONIOConfig
from agent_memory_toolkit import Memory

# Create importer
config = JSONIOConfig(
    validate_on_import=True,  # Validate memory structure
    skip_invalid=True,        # Skip invalid records instead of failing
)

importer = JSONImporter(config=config)

# Import memories
memories, result = importer.import_file(
    "memories_backup.json",
    memory_class=Memory,  # Deserialize into Memory objects
)

print(f"Imported {result.memory_count} memories")
print(f"Skipped {result.skipped_count} invalid records")

if result.warnings:
    for warning in result.warnings:
        print(f"Warning: {warning}")

# Add to store
for memory in memories:
    store.add(memory)
```

### Streaming for Large Datasets

For large datasets that don't fit in memory, use JSONL (JSON Lines) format with streaming:

```python
from agent_memory_toolkit.io import export_to_jsonl, import_from_jsonl

# Stream export - writes one memory per line
result = export_to_jsonl(
    memory_iterator=store.iter_all(),  # Iterator, not list
    output_path="large_dataset.jsonl.gz",
    compress=True,
)

# Stream import - yields batches
for batch in import_from_jsonl("large_dataset.jsonl.gz", batch_size=1000):
    print(f"Processing batch of {len(batch)} memories")
    for memory in batch:
        store.add(memory)
```

### Export Configuration Options

```python
from agent_memory_toolkit.io import JSONIOConfig, JSONFormat

config = JSONIOConfig(
    # Output format
    format=JSONFormat.PRETTY,  # COMPACT, PRETTY, or JSONL
    
    # Compression
    compress=True,  # Gzip compress output
    
    # What to include
    include_embeddings=True,   # Include embedding vectors
    include_versions=True,     # Include version history
    include_branches=True,     # Include branch metadata
    
    # File settings
    encoding="utf-8",
    
    # Streaming settings
    chunk_size=1000,  # Memories per batch for streaming
    
    # Import validation
    validate_on_import=True,
    skip_invalid=False,  # Raise error on invalid records
)
```

### Convenience Functions

```python
from agent_memory_toolkit.io import (
    export_to_json,
    import_from_json,
    export_to_jsonl,
    import_from_jsonl,
)

# Quick export
result = export_to_json(
    memories=memories,
    output_path="backup.json",
    compress=False,
    pretty=True,
    include_embeddings=True,
)

# Quick import
memories, result = import_from_json(
    "backup.json",
    memory_class=Memory,  # Optional
    validate=True,
)

# Stream export for large datasets
result = export_to_jsonl(
    memory_iterator=iter(memories),
    output_path="large.jsonl.gz",
    compress=True,
)

# Stream import
for batch in import_from_jsonl("large.jsonl.gz", batch_size=500):
    process_batch(batch)
```

## SQLite Backend

SQLite provides local-first persistence with full-text search, connection pooling, and automatic migrations.

### Basic Usage

```python
from agent_memory_toolkit.io import SQLiteBackend, SQLiteConfig

# Create backend with configuration
config = SQLiteConfig(
    database_path="memories.db",
    pool_size=5,
)

backend = SQLiteBackend(config)

# Add a memory
memory_id = backend.add({
    "content": "Python is a programming language",
    "metadata": {"source": "documentation", "tags": ["programming"]},
})

# Get a memory
memory = backend.get(memory_id)
print(memory["content"])

# Update a memory
backend.update(memory_id, {
    "content": "Python is a versatile programming language",
    "metadata": {"source": "documentation", "tags": ["programming", "python"]},
})

# Search memories
results = backend.search("programming language", limit=10)
for result in results:
    print(f"{result['_score']:.2f}: {result['content']}")

# Delete (soft delete by default)
backend.delete(memory_id)

# Hard delete
backend.delete(memory_id, hard=True)

# Close connections
backend.close()
```

### Configuration Options

```python
from agent_memory_toolkit.io import SQLiteConfig

config = SQLiteConfig(
    # Database file
    database_path="memories.db",
    
    # Connection pool
    pool_size=5,
    timeout=30.0,
    
    # Threading
    check_same_thread=False,  # Allow multi-threading
    
    # SQLite optimizations
    journal_mode="WAL",       # Write-ahead logging
    synchronous="NORMAL",     # Balance between safety and speed
    cache_size=-64000,        # 64MB cache
    auto_vacuum="INCREMENTAL",
    
    # Foreign keys
    enable_foreign_keys=True,
    
    # Auto-create database
    create_if_missing=True,
)
```

### Branching and Versioning

```python
# Add to a specific branch
backend.add(memory, branch="feature-experiment")

# Get from branch
memory = backend.get(memory_id, branch="feature-experiment")

# Search in branch
results = backend.search("query", branch="feature-experiment")

# Get version history
history = backend.get_version_history(memory_id, limit=10)
for version in history:
    print(f"v{version['version']}: {version['operation']} at {version['created_at']}")
```

### Iteration and Pagination

```python
# List all with pagination
page1 = backend.list_all(limit=100, offset=0)
page2 = backend.list_all(limit=100, offset=100)

# Iterate efficiently over large datasets
for memory in backend.iter_all(batch_size=1000):
    process(memory)

# Count memories
total = backend.count()
deleted = backend.count(include_deleted=True)
```

### Database Maintenance

```python
# Optimize database
backend.vacuum()

# Get stats
count = backend.count()
print(f"Total memories: {count}")
```

### Custom Migrations

```python
from agent_memory_toolkit.io import SQLiteBackend, SQLiteMigration

# Define custom migrations
custom_migrations = [
    SQLiteMigration(
        version=2,
        name="add_custom_column",
        up_sql="ALTER TABLE memories ADD COLUMN priority INTEGER DEFAULT 0",
        down_sql="-- SQLite doesn't support DROP COLUMN easily",
    ),
]

# Apply with default + custom migrations
backend = SQLiteBackend(
    config=config,
    migrations=DEFAULT_SQLITE_MIGRATIONS + custom_migrations,
)
```

## PostgreSQL Backend

PostgreSQL provides production-ready persistence with vector search (pgvector), full-text search, connection pooling, and async support.

### Installation

```bash
# Install PostgreSQL driver
pip install psycopg2-binary

# For async support
pip install 'psycopg[pool]'
```

### Basic Usage

```python
from agent_memory_toolkit.io import PostgresBackend, PostgresConfig

# Create backend
config = PostgresConfig(
    host="localhost",
    port=5432,
    database="agent_memories",
    user="myuser",
    password="mypassword",
)

backend = PostgresBackend(config)

# Add a memory
memory_id = backend.add({
    "content": "Machine learning transforms data into insights",
    "metadata": {"domain": "ml", "importance": 0.9},
    "embedding": [0.1, 0.2, ...],  # Optional embedding vector
})

# Full-text search
results = backend.search("machine learning", limit=10)

# Vector similarity search (requires pgvector)
results = backend.vector_search(
    embedding=query_embedding,
    limit=10,
    threshold=0.8,  # Minimum similarity
)

# Hybrid search (combines text and vector)
results = backend.hybrid_search(
    query="machine learning",
    embedding=query_embedding,
    text_weight=0.3,
    vector_weight=0.7,
    limit=10,
)

backend.close()
```

### Configuration Options

```python
from agent_memory_toolkit.io import PostgresConfig

config = PostgresConfig(
    # Connection
    host="localhost",
    port=5432,
    database="agent_memories",
    user="myuser",
    password="mypassword",
    
    # Connection pool
    min_connections=2,
    max_connections=10,
    connection_timeout=30.0,
    
    # SSL
    ssl_mode="prefer",  # disable, require, verify-ca, verify-full
    
    # Schema
    schema="public",
    
    # Identification
    application_name="agent-memory-toolkit",
    
    # Timeouts
    statement_timeout=30000,  # milliseconds
    connect_timeout=10,       # seconds
)

# Use connection string format
print(config.connection_string)
print(config.dsn)
```

### Setting Up pgvector

For vector similarity search, you need to set up pgvector:

```sql
-- Install pgvector extension (run as superuser)
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Vector Search

```python
# Store memories with embeddings
backend.add({
    "content": "Neural networks learn patterns from data",
    "embedding": model.encode("Neural networks learn patterns from data"),
})

# Search by vector similarity
query_embedding = model.encode("deep learning")
results = backend.vector_search(
    embedding=query_embedding,
    limit=10,
    threshold=0.7,  # Only return if similarity >= 0.7
)

for result in results:
    print(f"Similarity: {result['_similarity']:.3f}")
    print(f"Content: {result['content']}")
```

### Hybrid Search

Combine full-text and vector search for best results:

```python
# Hybrid search with weighted combination
results = backend.hybrid_search(
    query="neural network training",       # Text query
    embedding=model.encode("neural network training"),  # Vector query
    text_weight=0.3,   # 30% weight for text match
    vector_weight=0.7, # 70% weight for vector similarity
    limit=20,
)

for result in results:
    print(f"Score: {result['_score']:.3f}: {result['content'][:100]}...")
```

### Async Support

For high-concurrency applications, use the async backend:

```python
import asyncio
from agent_memory_toolkit.io import AsyncPostgresBackend, PostgresConfig

async def main():
    config = PostgresConfig(
        host="localhost",
        database="memories",
        user="myuser",
        password="mypassword",
    )
    
    async with AsyncPostgresBackend(config) as backend:
        # Add memory
        memory_id = await backend.add({
            "content": "Async operations improve throughput",
        })
        
        # Get memory
        memory = await backend.get(memory_id)
        
        # Search
        results = await backend.search("async", limit=10)
        
        # Iterate asynchronously
        async for memory in backend.iter_all():
            print(memory["content"])

asyncio.run(main())
```

### Connection Pooling

```python
from agent_memory_toolkit.io import PostgresConnectionPool, PostgresConfig

# Create pool
config = PostgresConfig(
    min_connections=2,
    max_connections=20,
)
pool = PostgresConnectionPool(config)

# Use pool directly
with pool.connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM memories")
        count = cur.fetchone()[0]

# Close all connections
pool.close_all()
```

### Migrations

```python
from agent_memory_toolkit.io import (
    PostgresMigrationManager,
    PostgresMigration,
    get_default_postgres_migrations,
)

# Custom migration
custom_migration = PostgresMigration(
    version=3,
    name="add_tags_table",
    up_sql="""
        CREATE TABLE tags (
            id SERIAL PRIMARY KEY,
            memory_id UUID REFERENCES memories(id),
            tag TEXT NOT NULL
        );
        CREATE INDEX idx_tags_memory ON tags(memory_id);
    """,
    down_sql="DROP TABLE tags;",
)

# Get migrations
migrations = get_default_postgres_migrations("public")
migrations.append(custom_migration)

# Apply migrations
manager = PostgresMigrationManager(pool, migrations)
applied = manager.migrate()
print(f"Applied migrations: {applied}")

# Rollback if needed
rolled_back = manager.rollback(steps=1)
```

## Best Practices

### Backup Strategy

```python
import schedule
from datetime import datetime

def backup_memories():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export full backup
    result = export_to_json(
        memories=store.list_all(),
        output_path=f"backups/full_{timestamp}.json.gz",
        compress=True,
    )
    
    print(f"Backup complete: {result.memory_count} memories")

# Schedule daily backups
schedule.every().day.at("02:00").do(backup_memories)
```

### Migration Between Backends

```python
from agent_memory_toolkit.io import SQLiteBackend, PostgresBackend

# Source: SQLite
sqlite_backend = SQLiteBackend(SQLiteConfig(database_path="old.db"))

# Target: PostgreSQL
pg_backend = PostgresBackend(PostgresConfig(
    host="production-db",
    database="memories",
))

# Migrate
count = 0
for memory in sqlite_backend.iter_all():
    pg_backend.add(memory)
    count += 1
    if count % 1000 == 0:
        print(f"Migrated {count} memories...")

print(f"Migration complete: {count} total memories")

sqlite_backend.close()
pg_backend.close()
```

### Error Handling

```python
from agent_memory_toolkit.io import SQLiteBackend

backend = SQLiteBackend(config)

try:
    memory_id = backend.add(memory)
except Exception as e:
    logger.error(f"Failed to add memory: {e}")
    raise
finally:
    # Always close connections
    backend.close()

# Or use context manager pattern
class ManagedBackend:
    def __init__(self, config):
        self.backend = SQLiteBackend(config)
    
    def __enter__(self):
        return self.backend
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.close()

with ManagedBackend(config) as backend:
    backend.add(memory)
```

## Performance Tips

### SQLite

1. **Use WAL mode**: Enables concurrent reads during writes
2. **Increase cache**: Set `cache_size=-64000` (64MB) for better performance
3. **Batch operations**: Use transactions for bulk inserts
4. **Vacuum periodically**: Run `backend.vacuum()` to optimize

### PostgreSQL

1. **Connection pooling**: Use appropriate `min_connections` and `max_connections`
2. **Use indexes**: Default migrations create indexes for common queries
3. **Vector index**: HNSW index for fast vector search
4. **Async for concurrency**: Use `AsyncPostgresBackend` for high-throughput
5. **Regular VACUUM**: Run `backend.vacuum()` periodically

### JSON I/O

1. **Stream large datasets**: Use JSONL format with iterators
2. **Compress**: Use `compress=True` for network transfer
3. **Exclude embeddings**: Set `include_embeddings=False` if not needed
4. **Batch imports**: Process in chunks to manage memory
