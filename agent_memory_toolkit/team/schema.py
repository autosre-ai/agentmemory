"""Database schema for Team Memory Protocol."""

import sqlite3


SCHEMA = """
-- Team memories table with extended fields
CREATE TABLE IF NOT EXISTS team_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    branch TEXT NOT NULL DEFAULT 'main',
    namespace TEXT NOT NULL DEFAULT 'default',
    agent_id TEXT,
    vector_clock_json TEXT NOT NULL DEFAULT '{}'
);

-- Index for branch queries
CREATE INDEX IF NOT EXISTS idx_team_memories_branch ON team_memories(branch);

-- Index for namespace queries
CREATE INDEX IF NOT EXISTS idx_team_memories_namespace ON team_memories(namespace);

-- Index for agent queries
CREATE INDEX IF NOT EXISTS idx_team_memories_agent ON team_memories(agent_id);

-- Compound index for branch + namespace
CREATE INDEX IF NOT EXISTS idx_team_memories_branch_namespace ON team_memories(branch, namespace);

-- Team branches table
CREATE TABLE IF NOT EXISTS team_branches (
    name TEXT PRIMARY KEY,
    head_commit_id TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    parent_branch TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (parent_branch) REFERENCES team_branches(name)
);

-- Team commits table
CREATE TABLE IF NOT EXISTS team_commits (
    id TEXT PRIMARY KEY,
    branch TEXT NOT NULL,
    parent_id TEXT,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT,
    memory_snapshot_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (branch) REFERENCES team_branches(name),
    FOREIGN KEY (parent_id) REFERENCES team_commits(id)
);

-- Index for branch commits
CREATE INDEX IF NOT EXISTS idx_team_commits_branch ON team_commits(branch);

-- Memory versions for history tracking
CREATE TABLE IF NOT EXISTS team_memory_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    operation TEXT NOT NULL,
    agent_id TEXT,
    FOREIGN KEY (memory_id) REFERENCES team_memories(id)
);

-- Index for version history
CREATE INDEX IF NOT EXISTS idx_team_memory_versions_memory ON team_memory_versions(memory_id);

-- Access control rules
CREATE TABLE IF NOT EXISTS team_access_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT,
    namespace TEXT,
    permission INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(agent_id, namespace)
);

-- Namespaces table
CREATE TABLE IF NOT EXISTS team_namespaces (
    name TEXT PRIMARY KEY,
    description TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

-- Event log for hooks and auditing
CREATE TABLE IF NOT EXISTS team_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    agent_id TEXT,
    data_json TEXT NOT NULL DEFAULT '{}'
);

-- Index for event queries
CREATE INDEX IF NOT EXISTS idx_team_events_type ON team_events(event_type);
CREATE INDEX IF NOT EXISTS idx_team_events_timestamp ON team_events(timestamp);

-- Sync state tracking
CREATE TABLE IF NOT EXISTS team_sync_state (
    remote_id TEXT PRIMARY KEY,
    last_sync_at TEXT NOT NULL,
    last_commit_id TEXT,
    sync_cursor TEXT
);

-- Full-text search for team memories
CREATE VIRTUAL TABLE IF NOT EXISTS team_memories_fts USING fts5(
    content,
    memory_id UNINDEXED,
    content='team_memories',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS team_memories_ai AFTER INSERT ON team_memories BEGIN
    INSERT INTO team_memories_fts(rowid, content, memory_id) 
    VALUES (NEW.rowid, NEW.content, NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS team_memories_ad AFTER DELETE ON team_memories BEGIN
    INSERT INTO team_memories_fts(team_memories_fts, rowid, content, memory_id) 
    VALUES('delete', OLD.rowid, OLD.content, OLD.id);
END;

CREATE TRIGGER IF NOT EXISTS team_memories_au AFTER UPDATE ON team_memories BEGIN
    INSERT INTO team_memories_fts(team_memories_fts, rowid, content, memory_id) 
    VALUES('delete', OLD.rowid, OLD.content, OLD.id);
    INSERT INTO team_memories_fts(rowid, content, memory_id) 
    VALUES (NEW.rowid, NEW.content, NEW.id);
END;
"""


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply the database schema."""
    conn.executescript(SCHEMA)
    conn.commit()


def run_migrations(conn: sqlite3.Connection) -> None:
    """Run any pending migrations."""
    # Create migrations table if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)
    
    # Get current version
    cursor = conn.execute(
        "SELECT MAX(version) FROM team_schema_migrations"
    )
    result = cursor.fetchone()
    current_version = result[0] if result[0] is not None else 0
    
    # Define migrations
    migrations = [
        # Migration 1: Add vector_clock_json column if not exists
        (1, """
            SELECT CASE 
                WHEN COUNT(*) = 0 THEN 1 
                ELSE 0 
            END as needs_migration
            FROM pragma_table_info('team_memories') 
            WHERE name = 'vector_clock_json'
        """, """
            ALTER TABLE team_memories ADD COLUMN vector_clock_json TEXT NOT NULL DEFAULT '{}'
        """),
    ]
    
    # Apply pending migrations
    for version, check_sql, migration_sql in migrations:
        if version > current_version:
            # Check if migration is needed
            if check_sql:
                cursor = conn.execute(check_sql)
                needs_migration = cursor.fetchone()[0]
                if needs_migration:
                    conn.executescript(migration_sql)
            else:
                conn.executescript(migration_sql)
            
            # Record migration
            conn.execute(
                "INSERT INTO team_schema_migrations (version, applied_at) VALUES (?, datetime('now'))",
                (version,)
            )
    
    conn.commit()
