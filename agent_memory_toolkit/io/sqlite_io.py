"""SQLite persistence backend for memory storage.

This module provides a SQLite-based persistence layer for memories with
connection pooling, migrations, and full CRUD operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Callable
from contextlib import contextmanager
from threading import Lock
import sqlite3
import json
import logging
import time
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SQLiteConfig:
    """Configuration for SQLite backend.
    
    Attributes:
        database_path: Path to the SQLite database file
        pool_size: Number of connections in the pool
        timeout: Connection timeout in seconds
        check_same_thread: Whether to check same thread (set False for multi-threading)
        journal_mode: SQLite journal mode (WAL recommended for concurrency)
        synchronous: SQLite synchronous pragma
        cache_size: SQLite cache size in pages
        auto_vacuum: SQLite auto vacuum setting
        enable_foreign_keys: Whether to enable foreign key constraints
        create_if_missing: Whether to create database if it doesn't exist
    """
    database_path: str | Path = "memories.db"
    pool_size: int = 5
    timeout: float = 30.0
    check_same_thread: bool = False
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -64000  # 64MB
    auto_vacuum: str = "INCREMENTAL"
    enable_foreign_keys: bool = True
    create_if_missing: bool = True


class SQLiteConnectionPool:
    """Thread-safe connection pool for SQLite.
    
    Example:
        ```python
        from agent_memory_toolkit.io import SQLiteConnectionPool, SQLiteConfig
        
        config = SQLiteConfig(database_path="memories.db", pool_size=10)
        pool = SQLiteConnectionPool(config)
        
        with pool.connection() as conn:
            cursor = conn.execute("SELECT * FROM memories")
            results = cursor.fetchall()
        ```
    """
    
    def __init__(self, config: SQLiteConfig):
        """Initialize the connection pool.
        
        Args:
            config: SQLite configuration
        """
        self.config = config
        self._pool: list[sqlite3.Connection] = []
        self._lock = Lock()
        self._in_use: set[int] = set()
        
        # Initialize pool
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        for _ in range(self.config.pool_size):
            conn = self._create_connection()
            self._pool.append(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with configuration."""
        conn = sqlite3.connect(
            str(self.config.database_path),
            timeout=self.config.timeout,
            check_same_thread=self.config.check_same_thread,
        )
        
        # Configure connection
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
        conn.execute(f"PRAGMA synchronous={self.config.synchronous}")
        conn.execute(f"PRAGMA cache_size={self.config.cache_size}")
        conn.execute(f"PRAGMA auto_vacuum={self.config.auto_vacuum}")
        
        if self.config.enable_foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")
        
        return conn
    
    @contextmanager
    def connection(self):
        """Get a connection from the pool.
        
        Yields:
            SQLite connection
        """
        conn = self._acquire()
        try:
            yield conn
        finally:
            self._release(conn)
    
    def _acquire(self) -> sqlite3.Connection:
        """Acquire a connection from the pool."""
        with self._lock:
            for i, conn in enumerate(self._pool):
                if id(conn) not in self._in_use:
                    self._in_use.add(id(conn))
                    return conn
            
            # Pool exhausted, create a new connection
            logger.warning("Connection pool exhausted, creating new connection")
            conn = self._create_connection()
            self._pool.append(conn)
            self._in_use.add(id(conn))
            return conn
    
    def _release(self, conn: sqlite3.Connection) -> None:
        """Release a connection back to the pool."""
        with self._lock:
            self._in_use.discard(id(conn))
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()
            self._in_use.clear()


@dataclass
class SQLiteMigration:
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


class SQLiteMigrationManager:
    """Manages database migrations for SQLite.
    
    Example:
        ```python
        from agent_memory_toolkit.io import SQLiteMigrationManager, SQLiteMigration
        
        migrations = [
            SQLiteMigration(
                version=1,
                name="initial_schema",
                up_sql="CREATE TABLE memories (...)",
                down_sql="DROP TABLE memories",
            ),
        ]
        
        manager = SQLiteMigrationManager(pool, migrations)
        manager.migrate()
        ```
    """
    
    def __init__(
        self,
        pool: SQLiteConnectionPool,
        migrations: list[SQLiteMigration] | None = None,
    ):
        """Initialize migration manager.
        
        Args:
            pool: Connection pool to use
            migrations: List of migrations to manage
        """
        self.pool = pool
        self.migrations = sorted(migrations or [], key=lambda m: m.version)
        
        # Ensure migration tracking table exists
        self._ensure_migration_table()
    
    def _ensure_migration_table(self) -> None:
        """Create the migration tracking table if it doesn't exist."""
        with self.pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def get_current_version(self) -> int:
        """Get the current database schema version."""
        with self.pool.connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(version) FROM _migrations"
            )
            result = cursor.fetchone()
            return result[0] if result[0] is not None else 0
    
    def get_pending_migrations(self) -> list[SQLiteMigration]:
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
                try:
                    conn.executescript(migration.up_sql)
                    conn.execute(
                        """
                        INSERT INTO _migrations (version, name, applied_at)
                        VALUES (?, ?, ?)
                        """,
                        (migration.version, migration.name, datetime.utcnow().isoformat())
                    )
                    conn.commit()
                    applied.append(migration.version)
                    logger.info(f"Successfully applied migration {migration.version}")
                except Exception as e:
                    conn.rollback()
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
            cursor = conn.execute(
                "SELECT version FROM _migrations ORDER BY version DESC LIMIT ?",
                (steps,)
            )
            versions = [row[0] for row in cursor.fetchall()]
        
        for version in versions:
            migration = next(
                (m for m in self.migrations if m.version == version), None
            )
            
            if migration and migration.down_sql:
                logger.info(f"Rolling back migration {version}: {migration.name}")
                
                with self.pool.connection() as conn:
                    try:
                        conn.executescript(migration.down_sql)
                        conn.execute(
                            "DELETE FROM _migrations WHERE version = ?",
                            (version,)
                        )
                        conn.commit()
                        rolled_back.append(version)
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"Failed to rollback migration {version}: {e}")
                        raise
            else:
                logger.warning(f"No down migration for version {version}")
        
        return rolled_back


# Default migrations for the memory toolkit schema
DEFAULT_SQLITE_MIGRATIONS = [
    SQLiteMigration(
        version=1,
        name="initial_schema",
        up_sql="""
            -- Main memories table
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                embedding_blob BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                branch TEXT NOT NULL DEFAULT 'main'
            );
            
            -- Memory versions for history tracking
            CREATE TABLE IF NOT EXISTS memory_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_blob BLOB,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                operation TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );
            
            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED,
                content,
                metadata_text,
                tokenize='porter unicode61'
            );
            
            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(memory_id, content, metadata_text) 
                VALUES (new.id, new.content, new.metadata_json);
            END;
            
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE memory_id = old.id;
            END;
            
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                UPDATE memories_fts 
                SET content = new.content, metadata_text = new.metadata_json
                WHERE memory_id = old.id;
            END;
            
            -- Branches table
            CREATE TABLE IF NOT EXISTS branches (
                name TEXT PRIMARY KEY,
                head_commit_id TEXT,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );
            
            -- Commits table
            CREATE TABLE IF NOT EXISTS commits (
                id TEXT PRIMARY KEY,
                branch TEXT NOT NULL,
                parent_id TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                memory_snapshot_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (branch) REFERENCES branches(name),
                FOREIGN KEY (parent_id) REFERENCES commits(id)
            );
            
            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_memories_branch ON memories(branch);
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_is_deleted ON memories(is_deleted);
            CREATE INDEX IF NOT EXISTS idx_memory_versions_memory_id ON memory_versions(memory_id);
            CREATE INDEX IF NOT EXISTS idx_commits_branch ON commits(branch);
            CREATE INDEX IF NOT EXISTS idx_commits_parent_id ON commits(parent_id);
            
            -- Insert default main branch
            INSERT OR IGNORE INTO branches (name, created_at) 
            VALUES ('main', datetime('now'));
        """,
        down_sql="""
            DROP TABLE IF EXISTS commits;
            DROP TABLE IF EXISTS branches;
            DROP TRIGGER IF EXISTS memories_ai;
            DROP TRIGGER IF EXISTS memories_ad;
            DROP TRIGGER IF EXISTS memories_au;
            DROP TABLE IF EXISTS memories_fts;
            DROP TABLE IF EXISTS memory_versions;
            DROP TABLE IF EXISTS memories;
        """,
    ),
]


class SQLiteBackend:
    """SQLite persistence backend for memories.
    
    Provides full CRUD operations, search, and versioning for memories
    using SQLite as the storage backend.
    
    Example:
        ```python
        from agent_memory_toolkit.io import SQLiteBackend, SQLiteConfig
        
        # Create backend
        config = SQLiteConfig(database_path="memories.db")
        backend = SQLiteBackend(config)
        
        # Store a memory
        backend.add(Memory.create(content="Important fact"))
        
        # Search memories
        results = backend.search("important", limit=10)
        
        # Get all memories
        for memory in backend.iter_all():
            print(memory["content"])
        ```
    """
    
    def __init__(
        self,
        config: SQLiteConfig | None = None,
        migrations: list[SQLiteMigration] | None = None,
    ):
        """Initialize SQLite backend.
        
        Args:
            config: SQLite configuration
            migrations: Custom migrations (default: use built-in schema)
        """
        self.config = config or SQLiteConfig()
        
        # Create directory if needed
        db_path = Path(self.config.database_path)
        if self.config.create_if_missing:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        self.pool = SQLiteConnectionPool(self.config)
        
        # Run migrations
        all_migrations = migrations or DEFAULT_SQLITE_MIGRATIONS
        self.migration_manager = SQLiteMigrationManager(self.pool, all_migrations)
        self.migration_manager.migrate()
    
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
        now = datetime.utcnow().isoformat()
        
        with self.pool.connection() as conn:
            # Insert memory
            conn.execute(
                """
                INSERT INTO memories (id, content, metadata_json, embedding_blob, 
                                     created_at, updated_at, version, is_deleted, branch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    data.get("content", ""),
                    json.dumps(data.get("metadata", {})),
                    self._encode_embedding(data.get("embedding")),
                    data.get("created_at", now),
                    data.get("updated_at", now),
                    data.get("version", 1),
                    int(data.get("is_deleted", False)),
                    branch,
                )
            )
            
            # Record in version history
            conn.execute(
                """
                INSERT INTO memory_versions (memory_id, content, metadata_json, 
                                            embedding_blob, version, created_at, operation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    data.get("content", ""),
                    json.dumps(data.get("metadata", {})),
                    self._encode_embedding(data.get("embedding")),
                    data.get("version", 1),
                    now,
                    "create",
                )
            )
            
            conn.commit()
        
        return memory_id
    
    def get(self, memory_id: str, branch: str = "main") -> dict[str, Any] | None:
        """Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            branch: Branch to search in
            
        Returns:
            Memory dictionary or None if not found
        """
        with self.pool.connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, content, metadata_json, embedding_blob, created_at, 
                       updated_at, version, is_deleted, branch
                FROM memories 
                WHERE id = ? AND branch = ? AND is_deleted = 0
                """,
                (memory_id, branch)
            )
            row = cursor.fetchone()
            
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
        now = datetime.utcnow().isoformat()
        
        with self.pool.connection() as conn:
            # Get current memory
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND branch = ?",
                (memory_id, branch)
            )
            row = cursor.fetchone()
            
            if not row:
                return False
            
            current = self._row_to_dict(row)
            new_version = current["version"] + 1
            
            # Update memory
            conn.execute(
                """
                UPDATE memories 
                SET content = ?, metadata_json = ?, embedding_blob = ?,
                    updated_at = ?, version = ?
                WHERE id = ? AND branch = ?
                """,
                (
                    updates.get("content", current["content"]),
                    json.dumps(updates.get("metadata", current.get("metadata", {}))),
                    self._encode_embedding(updates.get("embedding", current.get("embedding"))),
                    now,
                    new_version,
                    memory_id,
                    branch,
                )
            )
            
            # Record in version history
            conn.execute(
                """
                INSERT INTO memory_versions (memory_id, content, metadata_json,
                                            embedding_blob, version, created_at, operation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    updates.get("content", current["content"]),
                    json.dumps(updates.get("metadata", current.get("metadata", {}))),
                    self._encode_embedding(updates.get("embedding", current.get("embedding"))),
                    new_version,
                    now,
                    "update",
                )
            )
            
            conn.commit()
        
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
        with self.pool.connection() as conn:
            if hard:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE id = ? AND branch = ?",
                    (memory_id, branch)
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE memories SET is_deleted = 1, updated_at = ?
                    WHERE id = ? AND branch = ?
                    """,
                    (datetime.utcnow().isoformat(), memory_id, branch)
                )
            
            conn.commit()
            return cursor.rowcount > 0
    
    def search(
        self,
        query: str,
        branch: str = "main",
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Full-text search for memories.
        
        Args:
            query: Search query
            branch: Branch to search in
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            List of matching memories with scores
        """
        with self.pool.connection() as conn:
            cursor = conn.execute(
                """
                SELECT m.id, m.content, m.metadata_json, m.embedding_blob,
                       m.created_at, m.updated_at, m.version, m.is_deleted, m.branch,
                       bm25(memories_fts) as score
                FROM memories m
                JOIN memories_fts fts ON m.id = fts.memory_id
                WHERE memories_fts MATCH ? AND m.branch = ? AND m.is_deleted = 0
                ORDER BY score
                LIMIT ? OFFSET ?
                """,
                (query, branch, limit, offset)
            )
            
            results = []
            for row in cursor.fetchall():
                memory = self._row_to_dict(row)
                memory["_score"] = row["score"]
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
        with self.pool.connection() as conn:
            query = """
                SELECT id, content, metadata_json, embedding_blob, created_at,
                       updated_at, version, is_deleted, branch
                FROM memories
                WHERE branch = ?
            """
            params: list[Any] = [branch]
            
            if not include_deleted:
                query += " AND is_deleted = 0"
            
            query += " ORDER BY created_at DESC"
            
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def iter_all(
        self,
        branch: str = "main",
        batch_size: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over all memories in batches.
        
        Args:
            branch: Branch to iterate
            batch_size: Number of memories per batch
            
        Yields:
            Memory dictionaries
        """
        offset = 0
        while True:
            batch = self.list_all(branch=branch, limit=batch_size, offset=offset)
            if not batch:
                break
            
            for memory in batch:
                yield memory
            
            offset += batch_size
    
    def count(self, branch: str = "main", include_deleted: bool = False) -> int:
        """Count memories.
        
        Args:
            branch: Branch to count
            include_deleted: Whether to include soft-deleted
            
        Returns:
            Memory count
        """
        with self.pool.connection() as conn:
            query = "SELECT COUNT(*) FROM memories WHERE branch = ?"
            params: list[Any] = [branch]
            
            if not include_deleted:
                query += " AND is_deleted = 0"
            
            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]
    
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
        with self.pool.connection() as conn:
            cursor = conn.execute(
                """
                SELECT memory_id, content, metadata_json, version, created_at, operation
                FROM memory_versions
                WHERE memory_id = ?
                ORDER BY version DESC
                LIMIT ?
                """,
                (memory_id, limit)
            )
            
            return [
                {
                    "memory_id": row["memory_id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata_json"]),
                    "version": row["version"],
                    "created_at": row["created_at"],
                    "operation": row["operation"],
                }
                for row in cursor.fetchall()
            ]
    
    def vacuum(self) -> None:
        """Optimize the database by running VACUUM."""
        with self.pool.connection() as conn:
            conn.execute("VACUUM")
    
    def close(self) -> None:
        """Close all connections."""
        self.pool.close_all()
    
    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a dictionary."""
        return {
            "id": row["id"],
            "content": row["content"],
            "metadata": json.loads(row["metadata_json"]),
            "embedding": self._decode_embedding(row["embedding_blob"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "version": row["version"],
            "is_deleted": bool(row["is_deleted"]),
            "branch": row["branch"],
        }
    
    def _encode_embedding(self, embedding: list[float] | None) -> bytes | None:
        """Encode embedding to blob."""
        if embedding is None:
            return None
        import struct
        return struct.pack(f'{len(embedding)}f', *embedding)
    
    def _decode_embedding(self, blob: bytes | None) -> list[float] | None:
        """Decode embedding from blob."""
        if blob is None:
            return None
        import struct
        count = len(blob) // 4
        return list(struct.unpack(f'{count}f', blob))
