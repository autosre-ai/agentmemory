"""Database schema and migrations for the memory store."""

import sqlite3
import logging
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

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
    operation TEXT NOT NULL,  -- 'create', 'update', 'delete'
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

-- FTS5 virtual table for full-text search (standalone, not external content)
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
"""


def get_current_schema_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version from the database."""
    try:
        cursor = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        result = cursor.fetchone()
        return result[0] if result[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply the database schema."""
    current_version = get_current_schema_version(conn)
    
    if current_version < SCHEMA_VERSION:
        logger.info(f"Applying schema version {SCHEMA_VERSION}")
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,)
        )
        conn.commit()
        logger.info("Schema applied successfully")
    else:
        logger.debug(f"Schema is up to date (version {current_version})")


def run_migrations(conn: sqlite3.Connection) -> None:
    """Run any pending migrations."""
    current_version = get_current_schema_version(conn)
    
    # Add migration functions here as the schema evolves
    migrations: dict[int, callable] = {
        # 2: migrate_v1_to_v2,
        # 3: migrate_v2_to_v3,
    }
    
    for version in sorted(migrations.keys()):
        if current_version < version:
            logger.info(f"Running migration to version {version}")
            migrations[version](conn)
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (version,)
            )
            conn.commit()
