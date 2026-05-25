"""Data Persistence Module - Import/Export and database persistence for memories.

This module provides tools for persisting memories to various storage backends
and importing/exporting memory data in different formats.

Components:
- JSON I/O: Import and export memories to/from JSON files
- SQLite Persistence: Persistent storage using SQLite database
- PostgreSQL Persistence: Production-ready PostgreSQL storage backend
"""

from .json_io import (
    JSONExporter,
    JSONImporter,
    JSONIOConfig,
    ExportResult,
    ImportResult,
    export_to_json,
    import_from_json,
    export_to_jsonl,
    import_from_jsonl,
)

from .sqlite_io import (
    SQLiteBackend,
    SQLiteConfig,
    SQLiteConnectionPool,
    SQLiteMigration,
    SQLiteMigrationManager,
)

from .postgres_io import (
    PostgresBackend,
    PostgresConfig,
    PostgresConnectionPool,
    PostgresMigration,
    PostgresMigrationManager,
    AsyncPostgresBackend,
)

__all__ = [
    # JSON I/O
    "JSONExporter",
    "JSONImporter",
    "JSONIOConfig",
    "ExportResult",
    "ImportResult",
    "export_to_json",
    "import_from_json",
    "export_to_jsonl",
    "import_from_jsonl",
    # SQLite persistence
    "SQLiteBackend",
    "SQLiteConfig",
    "SQLiteConnectionPool",
    "SQLiteMigration",
    "SQLiteMigrationManager",
    # PostgreSQL persistence
    "PostgresBackend",
    "PostgresConfig",
    "PostgresConnectionPool",
    "PostgresMigration",
    "PostgresMigrationManager",
    "AsyncPostgresBackend",
]
