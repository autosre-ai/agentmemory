"""PostgreSQL persistence backend for memory storage.

This module provides a PostgreSQL-based persistence layer for memories with
connection pooling, migrations, async support, and full CRUD operations.
Suitable for production deployments with high concurrency requirements.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Callable, AsyncIterator
from contextlib import contextmanager, asynccontextmanager
import json
import logging
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Check for psycopg2/psycopg availability
try:
    import psycopg2
    import psycopg2.pool
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import psycopg
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool
    PSYCOPG3_AVAILABLE = True
except ImportError:
    PSYCOPG3_AVAILABLE = False


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL backend.
    
    Attributes:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        min_connections: Minimum connections in pool
        max_connections: Maximum connections in pool
        connection_timeout: Connection timeout in seconds
        ssl_mode: SSL mode (disable, require, verify-ca, verify-full)
        schema: Database schema name
        application_name: Application name for connection identification
        statement_timeout: Statement timeout in milliseconds
        connect_timeout: Connection establishment timeout
    """
    host: str = "localhost"
    port: int = 5432
    database: str = "agent_memory"
    user: str = "postgres"
    password: str = ""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 30.0
    ssl_mode: str = "prefer"
    schema: str = "public"
    application_name: str = "agent-memory-toolkit"
    statement_timeout: int = 30000  # 30 seconds
    connect_timeout: int = 10
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password} sslmode={self.ssl_mode} "
            f"application_name={self.application_name} "
            f"connect_timeout={self.connect_timeout}"
        )
    
    @property
    def dsn(self) -> str:
        """Generate PostgreSQL DSN for asyncpg/psycopg."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


class PostgresConnectionPool:
    """Thread-safe connection pool for PostgreSQL using psycopg2.
    
    Example:
        ```python
        from agent_memory_toolkit.io import PostgresConnectionPool, PostgresConfig
        
        config = PostgresConfig(
            host="localhost",
            database="memories",
            user="myuser",
            password="mypassword",
        )
        pool = PostgresConnectionPool(config)
        
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM memories")
                results = cur.fetchall()
        ```
    """
    
    def __init__(self, config: PostgresConfig):
        """Initialize the connection pool.
        
        Args:
            config: PostgreSQL configuration
            
        Raises:
            ImportError: If psycopg2 is not installed
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for PostgresConnectionPool. "
                "Install with: pip install psycopg2-binary"
            )
        
        self.config = config
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=config.min_connections,
            maxconn=config.max_connections,
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            sslmode=config.ssl_mode,
            application_name=config.application_name,
            connect_timeout=config.connect_timeout,
        )
    
    @contextmanager
    def connection(self):
        """Get a connection from the pool.
        
        Yields:
            PostgreSQL connection
        """
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()


@dataclass
class PostgresMigration:
    """Represents a database migration.
    
    Attributes:
        version: Migration version number
        name: Human-readable migration name
        up_sql: SQL to apply the migration
        down_sql: SQL to rollback the migration
    """
    version: int
    name: str
    up_sql: str
    down_sql: str = ""


class PostgresMigrationManager:
    """Manages database migrations for PostgreSQL.
    
    Example:
        ```python
        from agent_memory_toolkit.io import PostgresMigrationManager, PostgresMigration
        
        migrations = [
            PostgresMigration(
                version=1,
                name="initial_schema",
                up_sql="CREATE TABLE memories (...)",
                down_sql="DROP TABLE memories",
            ),
        ]
        
        manager = PostgresMigrationManager(pool, migrations)
        manager.migrate()
        ```
    """
    
    def __init__(
        self,
        pool: PostgresConnectionPool,
        migrations: list[PostgresMigration] | None = None,
        schema: str = "public",
    ):
        """Initialize migration manager.
        
        Args:
            pool: Connection pool to use
            migrations: List of migrations to manage
            schema: Database schema
        """
        self.pool = pool
        self.migrations = sorted(migrations or [], key=lambda m: m.version)
        self.schema = schema
        
        # Ensure migration tracking table exists
        self._ensure_migration_table()
    
    def _ensure_migration_table(self) -> None:
        """Create the migration tracking table if it doesn't exist."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}._migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
    
    def get_current_version(self) -> int:
        """Get the current database schema version."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT MAX(version) FROM {self.schema}._migrations"
                )
                result = cur.fetchone()
                return result[0] if result[0] is not None else 0
    
    def get_pending_migrations(self) -> list[PostgresMigration]:
        """Get list of migrations that haven't been applied."""
        current = self.get_current_version()
        return [m for m in self.migrations if m.version > current]
    
    def migrate(self, target_version: int | None = None) -> list[int]:
        """Apply pending migrations.
        
        Args:
            target_version: Optional version to migrate to (default: latest)
            
        Returns:
            List of applied migration versions
        """
        applied: list[int] = []
        pending = self.get_pending_migrations()
        
        if target_version is not None:
            pending = [m for m in pending if m.version <= target_version]
        
        for migration in pending:
            logger.info(f"Applying migration {migration.version}: {migration.name}")
            
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute(migration.up_sql)
                        cur.execute(
                            f"""
                            INSERT INTO {self.schema}._migrations (version, name)
                            VALUES (%s, %s)
                            """,
                            (migration.version, migration.name)
                        )
                        applied.append(migration.version)
                        logger.info(f"Successfully applied migration {migration.version}")
                    except Exception as e:
                        logger.error(f"Failed to apply migration {migration.version}: {e}")
                        raise
        
        return applied
    
    def rollback(self, steps: int = 1) -> list[int]:
        """Rollback the last N migrations.
        
        Args:
            steps: Number of migrations to rollback
            
        Returns:
            List of rolled back migration versions
        """
        rolled_back: list[int] = []
        
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT version FROM {self.schema}._migrations 
                    ORDER BY version DESC LIMIT %s
                    """,
                    (steps,)
                )
                versions = [row[0] for row in cur.fetchall()]
        
        for version in versions:
            migration = next(
                (m for m in self.migrations if m.version == version), None
            )
            
            if migration and migration.down_sql:
                logger.info(f"Rolling back migration {version}: {migration.name}")
                
                with self.pool.connection() as conn:
                    with conn.cursor() as cur:
                        try:
                            cur.execute(migration.down_sql)
                            cur.execute(
                                f"DELETE FROM {self.schema}._migrations WHERE version = %s",
                                (version,)
                            )
                            rolled_back.append(version)
                        except Exception as e:
                            logger.error(f"Failed to rollback migration {version}: {e}")
                            raise
            else:
                logger.warning(f"No down migration for version {version}")
        
        return rolled_back


# Default migrations for PostgreSQL schema
def get_default_postgres_migrations(schema: str = "public") -> list[PostgresMigration]:
    """Get default PostgreSQL migrations.
    
    Args:
        schema: Database schema name
        
    Returns:
        List of migrations
    """
    return [
        PostgresMigration(
            version=1,
            name="initial_schema",
            up_sql=f"""
                -- Enable required extensions
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                CREATE EXTENSION IF NOT EXISTS "pg_trgm";
                
                -- Main memories table
                CREATE TABLE IF NOT EXISTS {schema}.memories (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(1536),  -- Adjust dimension as needed
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    version INTEGER NOT NULL DEFAULT 1,
                    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
                    branch TEXT NOT NULL DEFAULT 'main'
                );
                
                -- Memory versions for history tracking
                CREATE TABLE IF NOT EXISTS {schema}.memory_versions (
                    id SERIAL PRIMARY KEY,
                    memory_id UUID NOT NULL REFERENCES {schema}.memories(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector(1536),
                    version INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    operation TEXT NOT NULL
                );
                
                -- Branches table
                CREATE TABLE IF NOT EXISTS {schema}.branches (
                    name TEXT PRIMARY KEY,
                    head_commit_id UUID,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                );
                
                -- Commits table
                CREATE TABLE IF NOT EXISTS {schema}.commits (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    branch TEXT NOT NULL REFERENCES {schema}.branches(name),
                    parent_id UUID REFERENCES {schema}.commits(id),
                    message TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    memory_snapshot JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                
                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_memories_branch ON {schema}.memories(branch);
                CREATE INDEX IF NOT EXISTS idx_memories_created_at ON {schema}.memories(created_at);
                CREATE INDEX IF NOT EXISTS idx_memories_is_deleted ON {schema}.memories(is_deleted);
                CREATE INDEX IF NOT EXISTS idx_memories_content_trgm ON {schema}.memories USING gin(content gin_trgm_ops);
                CREATE INDEX IF NOT EXISTS idx_memories_metadata ON {schema}.memories USING gin(metadata);
                CREATE INDEX IF NOT EXISTS idx_memory_versions_memory_id ON {schema}.memory_versions(memory_id);
                CREATE INDEX IF NOT EXISTS idx_commits_branch ON {schema}.commits(branch);
                
                -- Insert default main branch
                INSERT INTO {schema}.branches (name) 
                VALUES ('main') 
                ON CONFLICT (name) DO NOTHING;
                
                -- Full-text search index
                CREATE INDEX IF NOT EXISTS idx_memories_fts ON {schema}.memories 
                USING gin(to_tsvector('english', content));
            """,
            down_sql=f"""
                DROP TABLE IF EXISTS {schema}.commits CASCADE;
                DROP TABLE IF EXISTS {schema}.branches CASCADE;
                DROP TABLE IF EXISTS {schema}.memory_versions CASCADE;
                DROP TABLE IF EXISTS {schema}.memories CASCADE;
            """,
        ),
        PostgresMigration(
            version=2,
            name="add_vector_index",
            up_sql=f"""
                -- Create HNSW index for vector similarity search (requires pgvector)
                CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw 
                ON {schema}.memories 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """,
            down_sql=f"""
                DROP INDEX IF EXISTS {schema}.idx_memories_embedding_hnsw;
            """,
        ),
    ]


class PostgresBackend:
    """PostgreSQL persistence backend for memories.
    
    Provides full CRUD operations, full-text search, vector search,
    and versioning for memories using PostgreSQL as the storage backend.
    
    Example:
        ```python
        from agent_memory_toolkit.io import PostgresBackend, PostgresConfig
        
        # Create backend
        config = PostgresConfig(
            host="localhost",
            database="memories",
            user="myuser",
            password="mypassword",
        )
        backend = PostgresBackend(config)
        
        # Store a memory
        memory_id = backend.add({"content": "Important fact", "metadata": {}})
        
        # Search memories
        results = backend.search("important", limit=10)
        
        # Vector similarity search
        results = backend.vector_search(embedding=[0.1, 0.2, ...], limit=10)
        ```
    """
    
    def __init__(
        self,
        config: PostgresConfig | None = None,
        migrations: list[PostgresMigration] | None = None,
    ):
        """Initialize PostgreSQL backend.
        
        Args:
            config: PostgreSQL configuration
            migrations: Custom migrations (default: use built-in schema)
        """
        self.config = config or PostgresConfig()
        
        # Initialize connection pool
        self.pool = PostgresConnectionPool(self.config)
        
        # Run migrations
        all_migrations = migrations or get_default_postgres_migrations(self.config.schema)
        self.migration_manager = PostgresMigrationManager(
            self.pool, all_migrations, self.config.schema
        )
        
        try:
            self.migration_manager.migrate()
        except Exception as e:
            logger.warning(f"Migration failed (may need pgvector): {e}")
    
    def add(
        self,
        memory: Any,
        branch: str = "main",
    ) -> str:
        """Add a memory to the store.
        
        Args:
            memory: Memory object or dictionary
            branch: Branch to add to
            
        Returns:
            Memory ID
        """
        if hasattr(memory, "to_dict"):
            data = memory.to_dict()
        else:
            data = dict(memory)
        
        memory_id = data.get("id") or str(uuid.uuid4())
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                # Insert memory
                cur.execute(
                    f"""
                    INSERT INTO {schema}.memories 
                    (id, content, metadata, embedding, created_at, updated_at, version, is_deleted, branch)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        memory_id,
                        data.get("content", ""),
                        json.dumps(data.get("metadata", {})),
                        data.get("embedding"),
                        data.get("created_at", datetime.utcnow()),
                        data.get("updated_at", datetime.utcnow()),
                        data.get("version", 1),
                        data.get("is_deleted", False),
                        branch,
                    )
                )
                result = cur.fetchone()
                memory_id = str(result[0])
                
                # Record in version history
                cur.execute(
                    f"""
                    INSERT INTO {schema}.memory_versions 
                    (memory_id, content, metadata, embedding, version, operation)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        memory_id,
                        data.get("content", ""),
                        json.dumps(data.get("metadata", {})),
                        data.get("embedding"),
                        data.get("version", 1),
                        "create",
                    )
                )
        
        return memory_id
    
    def get(self, memory_id: str, branch: str = "main") -> dict[str, Any] | None:
        """Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            branch: Branch to search in
            
        Returns:
            Memory dictionary or None if not found
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, content, metadata, embedding, created_at,
                           updated_at, version, is_deleted, branch
                    FROM {schema}.memories 
                    WHERE id = %s AND branch = %s AND is_deleted = FALSE
                    """,
                    (memory_id, branch)
                )
                row = cur.fetchone()
                
                if row:
                    return self._row_to_dict(row)
                return None
    
    def update(
        self,
        memory_id: str,
        updates: dict[str, Any],
        branch: str = "main",
    ) -> bool:
        """Update a memory.
        
        Args:
            memory_id: Memory ID
            updates: Dictionary of fields to update
            branch: Branch to update in
            
        Returns:
            True if updated, False if not found
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get current memory
                cur.execute(
                    f"SELECT * FROM {schema}.memories WHERE id = %s AND branch = %s",
                    (memory_id, branch)
                )
                row = cur.fetchone()
                
                if not row:
                    return False
                
                current = self._row_to_dict(row)
                new_version = current["version"] + 1
                
                # Update memory
                cur.execute(
                    f"""
                    UPDATE {schema}.memories 
                    SET content = %s, metadata = %s, embedding = %s,
                        updated_at = NOW(), version = %s
                    WHERE id = %s AND branch = %s
                    """,
                    (
                        updates.get("content", current["content"]),
                        json.dumps(updates.get("metadata", current.get("metadata", {}))),
                        updates.get("embedding", current.get("embedding")),
                        new_version,
                        memory_id,
                        branch,
                    )
                )
                
                # Record in version history
                cur.execute(
                    f"""
                    INSERT INTO {schema}.memory_versions 
                    (memory_id, content, metadata, embedding, version, operation)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        memory_id,
                        updates.get("content", current["content"]),
                        json.dumps(updates.get("metadata", current.get("metadata", {}))),
                        updates.get("embedding", current.get("embedding")),
                        new_version,
                        "update",
                    )
                )
        
        return True
    
    def delete(self, memory_id: str, branch: str = "main", hard: bool = False) -> bool:
        """Delete a memory.
        
        Args:
            memory_id: Memory ID
            branch: Branch to delete from
            hard: If True, permanently delete; otherwise soft delete
            
        Returns:
            True if deleted, False if not found
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                if hard:
                    cur.execute(
                        f"DELETE FROM {schema}.memories WHERE id = %s AND branch = %s",
                        (memory_id, branch)
                    )
                else:
                    cur.execute(
                        f"""
                        UPDATE {schema}.memories SET is_deleted = TRUE, updated_at = NOW()
                        WHERE id = %s AND branch = %s
                        """,
                        (memory_id, branch)
                    )
                
                return cur.rowcount > 0
    
    def search(
        self,
        query: str,
        branch: str = "main",
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Full-text search for memories using PostgreSQL's tsvector.
        
        Args:
            query: Search query
            branch: Branch to search in
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            List of matching memories with scores
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT id, content, metadata, embedding, created_at,
                           updated_at, version, is_deleted, branch,
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as score
                    FROM {schema}.memories
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                      AND branch = %s AND is_deleted = FALSE
                    ORDER BY score DESC
                    LIMIT %s OFFSET %s
                    """,
                    (query, query, branch, limit, offset)
                )
                
                results = []
                for row in cur.fetchall():
                    memory = self._row_to_dict(row)
                    memory["_score"] = row["score"]
                    results.append(memory)
                
                return results
    
    def vector_search(
        self,
        embedding: list[float],
        branch: str = "main",
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search using pgvector.
        
        Args:
            embedding: Query embedding vector
            branch: Branch to search in
            limit: Maximum results
            threshold: Optional minimum similarity threshold (0-1)
            
        Returns:
            List of matching memories with similarity scores
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                query = f"""
                    SELECT id, content, metadata, embedding, created_at,
                           updated_at, version, is_deleted, branch,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM {schema}.memories
                    WHERE branch = %s AND is_deleted = FALSE
                      AND embedding IS NOT NULL
                """
                params: list[Any] = [embedding, branch]
                
                if threshold is not None:
                    query += " AND 1 - (embedding <=> %s::vector) >= %s"
                    params.extend([embedding, threshold])
                
                query += " ORDER BY embedding <=> %s::vector LIMIT %s"
                params.extend([embedding, limit])
                
                cur.execute(query, params)
                
                results = []
                for row in cur.fetchall():
                    memory = self._row_to_dict(row)
                    memory["_similarity"] = row["similarity"]
                    results.append(memory)
                
                return results
    
    def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        branch: str = "main",
        limit: int = 10,
        text_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining full-text and vector similarity.
        
        Args:
            query: Text query for full-text search
            embedding: Query embedding for vector search
            branch: Branch to search in
            limit: Maximum results
            text_weight: Weight for text search score (0-1)
            vector_weight: Weight for vector similarity (0-1)
            
        Returns:
            List of matching memories with combined scores
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    WITH text_results AS (
                        SELECT id, 
                               ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as text_score
                        FROM {schema}.memories
                        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                          AND branch = %s AND is_deleted = FALSE
                    ),
                    vector_results AS (
                        SELECT id,
                               1 - (embedding <=> %s::vector) as vector_score
                        FROM {schema}.memories
                        WHERE branch = %s AND is_deleted = FALSE
                          AND embedding IS NOT NULL
                    )
                    SELECT m.id, m.content, m.metadata, m.embedding, m.created_at,
                           m.updated_at, m.version, m.is_deleted, m.branch,
                           COALESCE(tr.text_score, 0) * %s + COALESCE(vr.vector_score, 0) * %s as combined_score
                    FROM {schema}.memories m
                    LEFT JOIN text_results tr ON m.id = tr.id
                    LEFT JOIN vector_results vr ON m.id = vr.id
                    WHERE m.branch = %s AND m.is_deleted = FALSE
                      AND (tr.id IS NOT NULL OR vr.id IS NOT NULL)
                    ORDER BY combined_score DESC
                    LIMIT %s
                    """,
                    (
                        query, query, branch,
                        embedding, branch,
                        text_weight, vector_weight,
                        branch, limit
                    )
                )
                
                results = []
                for row in cur.fetchall():
                    memory = self._row_to_dict(row)
                    memory["_score"] = row["combined_score"]
                    results.append(memory)
                
                return results
    
    def list_all(
        self,
        branch: str = "main",
        include_deleted: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List all memories.
        
        Args:
            branch: Branch to list from
            include_deleted: Whether to include soft-deleted memories
            limit: Maximum results (None for all)
            offset: Offset for pagination
            
        Returns:
            List of memories
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                query = f"""
                    SELECT id, content, metadata, embedding, created_at,
                           updated_at, version, is_deleted, branch
                    FROM {schema}.memories
                    WHERE branch = %s
                """
                params: list[Any] = [branch]
                
                if not include_deleted:
                    query += " AND is_deleted = FALSE"
                
                query += " ORDER BY created_at DESC"
                
                if limit is not None:
                    query += " LIMIT %s OFFSET %s"
                    params.extend([limit, offset])
                
                cur.execute(query, params)
                return [self._row_to_dict(row) for row in cur.fetchall()]
    
    def iter_all(
        self,
        branch: str = "main",
        batch_size: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over all memories using server-side cursor.
        
        Args:
            branch: Branch to iterate
            batch_size: Number of memories per batch
            
        Yields:
            Memory dictionaries
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor(
                name="memory_iterator",
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.itersize = batch_size
                cur.execute(
                    f"""
                    SELECT id, content, metadata, embedding, created_at,
                           updated_at, version, is_deleted, branch
                    FROM {schema}.memories
                    WHERE branch = %s AND is_deleted = FALSE
                    ORDER BY created_at DESC
                    """,
                    (branch,)
                )
                
                for row in cur:
                    yield self._row_to_dict(row)
    
    def count(self, branch: str = "main", include_deleted: bool = False) -> int:
        """Count memories.
        
        Args:
            branch: Branch to count
            include_deleted: Whether to include soft-deleted
            
        Returns:
            Memory count
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                query = f"SELECT COUNT(*) FROM {schema}.memories WHERE branch = %s"
                params: list[Any] = [branch]
                
                if not include_deleted:
                    query += " AND is_deleted = FALSE"
                
                cur.execute(query, params)
                return cur.fetchone()[0]
    
    def get_version_history(
        self,
        memory_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get version history for a memory.
        
        Args:
            memory_id: Memory ID
            limit: Maximum versions to return
            
        Returns:
            List of version records
        """
        schema = self.config.schema
        
        with self.pool.connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT memory_id, content, metadata, version, created_at, operation
                    FROM {schema}.memory_versions
                    WHERE memory_id = %s
                    ORDER BY version DESC
                    LIMIT %s
                    """,
                    (memory_id, limit)
                )
                
                return [
                    {
                        "memory_id": str(row["memory_id"]),
                        "content": row["content"],
                        "metadata": row["metadata"],
                        "version": row["version"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "operation": row["operation"],
                    }
                    for row in cur.fetchall()
                ]
    
    def vacuum(self, full: bool = False) -> None:
        """Optimize the database.
        
        Args:
            full: Whether to run VACUUM FULL (requires exclusive lock)
        """
        with self.pool.connection() as conn:
            old_isolation = conn.isolation_level
            conn.set_isolation_level(0)  # AUTOCOMMIT
            with conn.cursor() as cur:
                if full:
                    cur.execute("VACUUM FULL ANALYZE")
                else:
                    cur.execute("VACUUM ANALYZE")
            conn.set_isolation_level(old_isolation)
    
    def close(self) -> None:
        """Close all connections."""
        self.pool.close_all()
    
    def _row_to_dict(self, row: dict) -> dict[str, Any]:
        """Convert a database row to a dictionary."""
        return {
            "id": str(row["id"]),
            "content": row["content"],
            "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
            "embedding": row.get("embedding"),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "version": row["version"],
            "is_deleted": row["is_deleted"],
            "branch": row["branch"],
        }


class AsyncPostgresBackend:
    """Async PostgreSQL persistence backend using psycopg3.
    
    Example:
        ```python
        from agent_memory_toolkit.io import AsyncPostgresBackend, PostgresConfig
        import asyncio
        
        async def main():
            config = PostgresConfig(
                host="localhost",
                database="memories",
                user="myuser",
                password="mypassword",
            )
            
            async with AsyncPostgresBackend(config) as backend:
                # Store a memory
                memory_id = await backend.add({"content": "Important fact"})
                
                # Search memories
                results = await backend.search("important", limit=10)
        
        asyncio.run(main())
        ```
    """
    
    def __init__(
        self,
        config: PostgresConfig | None = None,
    ):
        """Initialize async PostgreSQL backend.
        
        Args:
            config: PostgreSQL configuration
            
        Raises:
            ImportError: If psycopg is not installed
        """
        if not PSYCOPG3_AVAILABLE:
            raise ImportError(
                "psycopg (v3) is required for AsyncPostgresBackend. "
                "Install with: pip install 'psycopg[pool]'"
            )
        
        self.config = config or PostgresConfig()
        self._pool: AsyncConnectionPool | None = None
    
    async def __aenter__(self) -> "AsyncPostgresBackend":
        """Async context manager entry."""
        self._pool = AsyncConnectionPool(
            self.config.dsn,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
        )
        await self._pool.open()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._pool:
            await self._pool.close()
    
    async def add(
        self,
        memory: Any,
        branch: str = "main",
    ) -> str:
        """Add a memory to the store asynchronously.
        
        Args:
            memory: Memory object or dictionary
            branch: Branch to add to
            
        Returns:
            Memory ID
        """
        if hasattr(memory, "to_dict"):
            data = memory.to_dict()
        else:
            data = dict(memory)
        
        memory_id = data.get("id") or str(uuid.uuid4())
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {schema}.memories 
                    (id, content, metadata, embedding, created_at, updated_at, version, is_deleted, branch)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        memory_id,
                        data.get("content", ""),
                        json.dumps(data.get("metadata", {})),
                        data.get("embedding"),
                        data.get("created_at", datetime.utcnow()),
                        data.get("updated_at", datetime.utcnow()),
                        data.get("version", 1),
                        data.get("is_deleted", False),
                        branch,
                    )
                )
                result = await cur.fetchone()
                memory_id = str(result[0])
                
                await cur.execute(
                    f"""
                    INSERT INTO {schema}.memory_versions 
                    (memory_id, content, metadata, embedding, version, operation)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        memory_id,
                        data.get("content", ""),
                        json.dumps(data.get("metadata", {})),
                        data.get("embedding"),
                        data.get("version", 1),
                        "create",
                    )
                )
        
        return memory_id
    
    async def get(self, memory_id: str, branch: str = "main") -> dict[str, Any] | None:
        """Get a memory by ID asynchronously.
        
        Args:
            memory_id: Memory ID
            branch: Branch to search in
            
        Returns:
            Memory dictionary or None if not found
        """
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT id, content, metadata, embedding, created_at,
                           updated_at, version, is_deleted, branch
                    FROM {schema}.memories 
                    WHERE id = %s AND branch = %s AND is_deleted = FALSE
                    """,
                    (memory_id, branch)
                )
                row = await cur.fetchone()
                
                if row:
                    return self._row_to_dict(row)
                return None
    
    async def search(
        self,
        query: str,
        branch: str = "main",
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Full-text search for memories asynchronously.
        
        Args:
            query: Search query
            branch: Branch to search in
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            List of matching memories with scores
        """
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT id, content, metadata, embedding, created_at,
                           updated_at, version, is_deleted, branch,
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as score
                    FROM {schema}.memories
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                      AND branch = %s AND is_deleted = FALSE
                    ORDER BY score DESC
                    LIMIT %s OFFSET %s
                    """,
                    (query, query, branch, limit, offset)
                )
                
                results = []
                async for row in cur:
                    memory = self._row_to_dict(row)
                    memory["_score"] = row["score"]
                    results.append(memory)
                
                return results
    
    async def iter_all(
        self,
        branch: str = "main",
        batch_size: int = 1000,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate over all memories asynchronously.
        
        Args:
            branch: Branch to iterate
            batch_size: Number of memories per fetch
            
        Yields:
            Memory dictionaries
        """
        schema = self.config.schema
        
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(
                    f"""
                    SELECT id, content, metadata, embedding, created_at,
                           updated_at, version, is_deleted, branch
                    FROM {schema}.memories
                    WHERE branch = %s AND is_deleted = FALSE
                    ORDER BY created_at DESC
                    """,
                    (branch,)
                )
                
                async for row in cur:
                    yield self._row_to_dict(row)
    
    def _row_to_dict(self, row: dict) -> dict[str, Any]:
        """Convert a database row to a dictionary."""
        return {
            "id": str(row["id"]),
            "content": row["content"],
            "metadata": row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
            "embedding": row.get("embedding"),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "version": row["version"],
            "is_deleted": row["is_deleted"],
            "branch": row["branch"],
        }
