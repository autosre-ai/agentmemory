"""Export/Import format implementations for Agent Memory Toolkit.

This module provides serialization and deserialization of memory data
in various formats for backup, migration, and interoperability.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_memory_toolkit.store import MemoryStore, Memory

# Optional parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class ExportFormat(Enum):
    """Supported export formats."""
    JSONL = "jsonl"
    PARQUET = "parquet"
    CSV = "csv"
    MARKDOWN = "markdown"
    SQLITE = "sqlite"


@dataclass
class ExportConfig:
    """Configuration for memory export operations."""
    
    include_metadata: bool = True
    include_embeddings: bool = False
    include_deleted: bool = False
    include_versions: bool = False
    branches: list[str] | None = None  # None = all branches
    date_format: str = "%Y-%m-%d %H:%M:%S"
    pretty_print: bool = False  # For JSON/Markdown
    batch_size: int = 1000  # For streaming export


@dataclass
class ImportConfig:
    """Configuration for memory import operations."""
    
    merge_strategy: str = "skip"  # skip, replace, or error
    validate_content: bool = True
    preserve_ids: bool = False  # Use original IDs vs generate new
    preserve_timestamps: bool = False
    target_branch: str | None = None  # None = current branch


@dataclass
class ExportResult:
    """Result of an export operation."""
    
    format: ExportFormat
    path: str
    memory_count: int
    file_size_bytes: int
    branches_exported: list[str]
    duration_ms: float
    errors: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """Result of an import operation."""
    
    format: ExportFormat
    path: str
    memories_imported: int
    memories_skipped: int
    memories_replaced: int
    duration_ms: float
    errors: list[str] = field(default_factory=list)


# ==============================================================================
# JSONL Export/Import
# ==============================================================================

def export_jsonl(
    store: "MemoryStore",
    output_path: str | Path,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export memories to JSON Lines format.
    
    Each line contains one memory as a JSON object. This format is ideal for:
    - Streaming processing
    - Log-style archives
    - Line-by-line imports
    
    Args:
        store: The memory store to export from
        output_path: Path to write the JSONL file
        config: Export configuration options
        
    Returns:
        ExportResult with export statistics
    """
    import time
    start_time = time.perf_counter()
    
    config = config or ExportConfig()
    output_path = Path(output_path)
    
    memory_count = 0
    branches_exported = set()
    errors = []
    
    # Determine branches to export
    target_branches = config.branches or [b.name for b in store.list_branches()]
    current_branch = store.current_branch
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for branch_name in target_branches:
                # Switch to branch
                try:
                    store.checkout(branch_name)
                except Exception as e:
                    errors.append(f"Could not checkout branch {branch_name}: {e}")
                    continue
                
                branches_exported.add(branch_name)
                
                # Export memories from this branch
                offset = 0
                while True:
                    memories = store.list(
                        limit=config.batch_size,
                        offset=offset,
                        include_deleted=config.include_deleted,
                    )
                    
                    if not memories:
                        break
                    
                    for memory in memories:
                        record = _memory_to_record(memory, branch_name, config)
                        json_line = json.dumps(record, default=str)
                        f.write(json_line + "\n")
                        memory_count += 1
                    
                    offset += config.batch_size
        
        # Restore original branch
        store.checkout(current_branch)
        
    except Exception as e:
        errors.append(f"Export failed: {e}")
        raise
    finally:
        # Ensure we restore branch even on error
        try:
            store.checkout(current_branch)
        except Exception:
            pass
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    file_size = output_path.stat().st_size
    
    return ExportResult(
        format=ExportFormat.JSONL,
        path=str(output_path),
        memory_count=memory_count,
        file_size_bytes=file_size,
        branches_exported=list(branches_exported),
        duration_ms=duration_ms,
        errors=errors,
    )


def import_jsonl(
    store: "MemoryStore",
    input_path: str | Path,
    config: ImportConfig | None = None,
) -> ImportResult:
    """Import memories from JSON Lines format.
    
    Args:
        store: The memory store to import into
        input_path: Path to the JSONL file
        config: Import configuration options
        
    Returns:
        ImportResult with import statistics
    """
    import time
    start_time = time.perf_counter()
    
    config = config or ImportConfig()
    input_path = Path(input_path)
    
    imported = 0
    skipped = 0
    replaced = 0
    errors = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                result = _import_memory_record(store, record, config)
                
                if result == "imported":
                    imported += 1
                elif result == "skipped":
                    skipped += 1
                elif result == "replaced":
                    replaced += 1
                    
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"Line {line_num}: Import failed - {e}")
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    return ImportResult(
        format=ExportFormat.JSONL,
        path=str(input_path),
        memories_imported=imported,
        memories_skipped=skipped,
        memories_replaced=replaced,
        duration_ms=duration_ms,
        errors=errors,
    )


# ==============================================================================
# CSV Export/Import
# ==============================================================================

CSV_FIELDS = [
    "id", "content", "source", "confidence", "tags",
    "created_at", "updated_at", "version", "branch"
]


def export_csv(
    store: "MemoryStore",
    output_path: str | Path,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export memories to CSV format.
    
    Exports a flat representation of memories suitable for spreadsheet analysis.
    Metadata fields are flattened, and complex objects are JSON-encoded.
    
    Args:
        store: The memory store to export from
        output_path: Path to write the CSV file
        config: Export configuration options
        
    Returns:
        ExportResult with export statistics
    """
    import time
    start_time = time.perf_counter()
    
    config = config or ExportConfig()
    output_path = Path(output_path)
    
    memory_count = 0
    branches_exported = set()
    errors = []
    
    target_branches = config.branches or [b.name for b in store.list_branches()]
    current_branch = store.current_branch
    
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            
            for branch_name in target_branches:
                try:
                    store.checkout(branch_name)
                except Exception as e:
                    errors.append(f"Could not checkout branch {branch_name}: {e}")
                    continue
                
                branches_exported.add(branch_name)
                
                offset = 0
                while True:
                    memories = store.list(
                        limit=config.batch_size,
                        offset=offset,
                        include_deleted=config.include_deleted,
                    )
                    
                    if not memories:
                        break
                    
                    for memory in memories:
                        row = _memory_to_csv_row(memory, branch_name, config)
                        writer.writerow(row)
                        memory_count += 1
                    
                    offset += config.batch_size
        
        store.checkout(current_branch)
        
    except Exception as e:
        errors.append(f"Export failed: {e}")
        raise
    finally:
        try:
            store.checkout(current_branch)
        except Exception:
            pass
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    file_size = output_path.stat().st_size
    
    return ExportResult(
        format=ExportFormat.CSV,
        path=str(output_path),
        memory_count=memory_count,
        file_size_bytes=file_size,
        branches_exported=list(branches_exported),
        duration_ms=duration_ms,
        errors=errors,
    )


def import_csv(
    store: "MemoryStore",
    input_path: str | Path,
    config: ImportConfig | None = None,
) -> ImportResult:
    """Import memories from CSV format.
    
    Args:
        store: The memory store to import into
        input_path: Path to the CSV file
        config: Import configuration options
        
    Returns:
        ImportResult with import statistics
    """
    import time
    start_time = time.perf_counter()
    
    config = config or ImportConfig()
    input_path = Path(input_path)
    
    imported = 0
    skipped = 0
    replaced = 0
    errors = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, 2):  # Start at 2 (after header)
            try:
                record = _csv_row_to_record(row)
                result = _import_memory_record(store, record, config)
                
                if result == "imported":
                    imported += 1
                elif result == "skipped":
                    skipped += 1
                elif result == "replaced":
                    replaced += 1
                    
            except Exception as e:
                errors.append(f"Row {row_num}: Import failed - {e}")
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    return ImportResult(
        format=ExportFormat.CSV,
        path=str(input_path),
        memories_imported=imported,
        memories_skipped=skipped,
        memories_replaced=replaced,
        duration_ms=duration_ms,
        errors=errors,
    )


# ==============================================================================
# Parquet Export/Import (Optional)
# ==============================================================================

def export_parquet(
    store: "MemoryStore",
    output_path: str | Path,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export memories to Apache Parquet format.
    
    Parquet provides efficient columnar storage with compression,
    ideal for large datasets and analytics workloads.
    
    Requires: pyarrow
    
    Args:
        store: The memory store to export from
        output_path: Path to write the Parquet file
        config: Export configuration options
        
    Returns:
        ExportResult with export statistics
        
    Raises:
        ImportError: If pyarrow is not installed
    """
    if not PARQUET_AVAILABLE:
        raise ImportError(
            "Parquet export requires pyarrow. "
            "Install with: pip install pyarrow"
        )
    
    import time
    start_time = time.perf_counter()
    
    config = config or ExportConfig()
    output_path = Path(output_path)
    
    errors = []
    branches_exported = set()
    
    target_branches = config.branches or [b.name for b in store.list_branches()]
    current_branch = store.current_branch
    
    # Collect all records
    records = []
    
    try:
        for branch_name in target_branches:
            try:
                store.checkout(branch_name)
            except Exception as e:
                errors.append(f"Could not checkout branch {branch_name}: {e}")
                continue
            
            branches_exported.add(branch_name)
            
            offset = 0
            while True:
                memories = store.list(
                    limit=config.batch_size,
                    offset=offset,
                    include_deleted=config.include_deleted,
                )
                
                if not memories:
                    break
                
                for memory in memories:
                    record = _memory_to_parquet_record(memory, branch_name, config)
                    records.append(record)
                
                offset += config.batch_size
        
        store.checkout(current_branch)
        
    except Exception as e:
        errors.append(f"Export failed: {e}")
        raise
    finally:
        try:
            store.checkout(current_branch)
        except Exception:
            pass
    
    # Convert to PyArrow Table
    if records:
        table = pa.Table.from_pylist(records)
        pq.write_table(table, output_path, compression="snappy")
    else:
        # Write empty table with schema
        schema = pa.schema([
            ("id", pa.string()),
            ("content", pa.string()),
            ("source", pa.string()),
            ("confidence", pa.float64()),
            ("tags", pa.string()),
            ("created_at", pa.string()),
            ("updated_at", pa.string()),
            ("version", pa.int64()),
            ("branch", pa.string()),
            ("metadata_json", pa.string()),
        ])
        table = pa.table({}, schema=schema)
        pq.write_table(table, output_path)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    file_size = output_path.stat().st_size
    
    return ExportResult(
        format=ExportFormat.PARQUET,
        path=str(output_path),
        memory_count=len(records),
        file_size_bytes=file_size,
        branches_exported=list(branches_exported),
        duration_ms=duration_ms,
        errors=errors,
    )


def import_parquet(
    store: "MemoryStore",
    input_path: str | Path,
    config: ImportConfig | None = None,
) -> ImportResult:
    """Import memories from Apache Parquet format.
    
    Requires: pyarrow
    
    Args:
        store: The memory store to import into
        input_path: Path to the Parquet file
        config: Import configuration options
        
    Returns:
        ImportResult with import statistics
        
    Raises:
        ImportError: If pyarrow is not installed
    """
    if not PARQUET_AVAILABLE:
        raise ImportError(
            "Parquet import requires pyarrow. "
            "Install with: pip install pyarrow"
        )
    
    import time
    start_time = time.perf_counter()
    
    config = config or ImportConfig()
    input_path = Path(input_path)
    
    imported = 0
    skipped = 0
    replaced = 0
    errors = []
    
    table = pq.read_table(input_path)
    
    for i in range(table.num_rows):
        try:
            record = {col: table[col][i].as_py() for col in table.column_names}
            result = _import_memory_record(store, record, config)
            
            if result == "imported":
                imported += 1
            elif result == "skipped":
                skipped += 1
            elif result == "replaced":
                replaced += 1
                
        except Exception as e:
            errors.append(f"Row {i}: Import failed - {e}")
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    return ImportResult(
        format=ExportFormat.PARQUET,
        path=str(input_path),
        memories_imported=imported,
        memories_skipped=skipped,
        memories_replaced=replaced,
        duration_ms=duration_ms,
        errors=errors,
    )


# ==============================================================================
# Markdown Export (Human-readable, export only)
# ==============================================================================

def export_markdown(
    store: "MemoryStore",
    output_path: str | Path,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export memories to human-readable Markdown format.
    
    Creates a well-formatted Markdown document suitable for:
    - Documentation
    - Manual review
    - Sharing with non-technical stakeholders
    
    Args:
        store: The memory store to export from
        output_path: Path to write the Markdown file
        config: Export configuration options
        
    Returns:
        ExportResult with export statistics
    """
    import time
    start_time = time.perf_counter()
    
    config = config or ExportConfig()
    output_path = Path(output_path)
    
    memory_count = 0
    branches_exported = set()
    errors = []
    
    target_branches = config.branches or [b.name for b in store.list_branches()]
    current_branch = store.current_branch
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("# Agent Memory Export\n\n")
            f.write(f"**Exported:** {datetime.utcnow().strftime(config.date_format)}\n")
            f.write(f"**Branches:** {', '.join(target_branches)}\n")
            f.write(f"**Format:** Markdown (Human-Readable)\n\n")
            f.write("---\n\n")
            
            for branch_name in target_branches:
                try:
                    store.checkout(branch_name)
                except Exception as e:
                    errors.append(f"Could not checkout branch {branch_name}: {e}")
                    continue
                
                branches_exported.add(branch_name)
                
                # Branch section
                f.write(f"## Branch: `{branch_name}`\n\n")
                
                branch_count = 0
                offset = 0
                
                while True:
                    memories = store.list(
                        limit=config.batch_size,
                        offset=offset,
                        include_deleted=config.include_deleted,
                    )
                    
                    if not memories:
                        break
                    
                    for memory in memories:
                        _write_memory_markdown(f, memory, config)
                        memory_count += 1
                        branch_count += 1
                    
                    offset += config.batch_size
                
                if branch_count == 0:
                    f.write("*No memories in this branch.*\n\n")
                
                f.write("---\n\n")
        
        store.checkout(current_branch)
        
    except Exception as e:
        errors.append(f"Export failed: {e}")
        raise
    finally:
        try:
            store.checkout(current_branch)
        except Exception:
            pass
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    file_size = output_path.stat().st_size
    
    return ExportResult(
        format=ExportFormat.MARKDOWN,
        path=str(output_path),
        memory_count=memory_count,
        file_size_bytes=file_size,
        branches_exported=list(branches_exported),
        duration_ms=duration_ms,
        errors=errors,
    )


# ==============================================================================
# SQLite Dump/Restore
# ==============================================================================

def export_sqlite_dump(
    store: "MemoryStore",
    output_path: str | Path,
    config: ExportConfig | None = None,
) -> ExportResult:
    """Export the entire SQLite database as a binary backup.
    
    Creates a complete SQLite database copy, including:
    - Schema (tables, indexes, triggers, FTS)
    - All data (memories, branches, commits, versions)
    
    This is the most complete export format, preserving:
    - All branches and their states
    - Version history
    - Embeddings
    - FTS index data
    
    Note: Output is a binary SQLite database file, not SQL text.
    
    Args:
        store: The memory store to export from
        output_path: Path to write the SQLite backup file
        config: Export configuration options (mostly ignored for SQLite backup)
        
    Returns:
        ExportResult with export statistics
    """
    import time
    start_time = time.perf_counter()
    
    config = config or ExportConfig()
    output_path = Path(output_path)
    
    errors = []
    
    # Get stats before backup
    memory_count = 0
    branches_exported = []
    
    try:
        cursor = store._conn.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]
        
        cursor = store._conn.execute("SELECT name FROM branches")
        branches_exported = [row[0] for row in cursor.fetchall()]
        
        # Use SQLite backup API for complete copy including FTS
        backup_conn = sqlite3.connect(str(output_path))
        store._conn.backup(backup_conn)
        backup_conn.close()
        
    except Exception as e:
        errors.append(f"Export failed: {e}")
        raise
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    file_size = output_path.stat().st_size
    
    return ExportResult(
        format=ExportFormat.SQLITE,
        path=str(output_path),
        memory_count=memory_count,
        file_size_bytes=file_size,
        branches_exported=branches_exported,
        duration_ms=duration_ms,
        errors=errors,
    )


def import_sqlite_dump(
    store: "MemoryStore",
    input_path: str | Path,
    config: ImportConfig | None = None,
) -> ImportResult:
    """Import memories from a SQLite database backup.
    
    WARNING: This replaces the entire database contents!
    Make a backup before importing.
    
    Args:
        store: The memory store to import into (will be replaced)
        input_path: Path to the SQLite backup file
        config: Import configuration options
        
    Returns:
        ImportResult with import statistics
    """
    import time
    start_time = time.perf_counter()
    
    config = config or ImportConfig()
    input_path = Path(input_path)
    
    errors = []
    
    # Get database path and close current connection
    db_path = store.db_path
    store.close()
    
    try:
        # Open the source database
        source_conn = sqlite3.connect(str(input_path))
        source_conn.row_factory = sqlite3.Row
        
        # Count memories in source
        cursor = source_conn.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]
        
        # Open target database
        target_conn = sqlite3.connect(str(db_path) if not isinstance(db_path, str) else db_path)
        
        # Use backup API to restore
        source_conn.backup(target_conn)
        
        source_conn.close()
        target_conn.close()
        
        # Reinitialize the store
        store._init_db()
        
    except Exception as e:
        errors.append(f"Import failed: {e}")
        # Try to reinitialize with fresh schema
        try:
            store._init_db()
        except Exception:
            pass
        raise
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    return ImportResult(
        format=ExportFormat.SQLITE,
        path=str(input_path),
        memories_imported=memory_count,
        memories_skipped=0,
        memories_replaced=0,
        duration_ms=duration_ms,
        errors=errors,
    )


# ==============================================================================
# Helper Functions
# ==============================================================================

def _memory_to_record(
    memory: "Memory",
    branch: str,
    config: ExportConfig,
) -> dict[str, Any]:
    """Convert a Memory object to a serializable record."""
    record = {
        "id": memory.id,
        "content": memory.content,
        "branch": branch,
        "created_at": memory.created_at.strftime(config.date_format),
        "updated_at": memory.updated_at.strftime(config.date_format),
        "version": memory.version,
        "is_deleted": memory.is_deleted,
    }
    
    if config.include_metadata:
        record["metadata"] = memory.metadata.to_dict()
    
    if config.include_embeddings and memory.embedding:
        record["embedding"] = memory.embedding
    
    return record


def _memory_to_csv_row(
    memory: "Memory",
    branch: str,
    config: ExportConfig,
) -> dict[str, str]:
    """Convert a Memory object to a CSV row dictionary."""
    metadata = memory.metadata
    
    return {
        "id": memory.id,
        "content": memory.content,
        "source": metadata.source or "",
        "confidence": str(metadata.confidence),
        "tags": json.dumps(metadata.tags),
        "created_at": memory.created_at.strftime(config.date_format),
        "updated_at": memory.updated_at.strftime(config.date_format),
        "version": str(memory.version),
        "branch": branch,
    }


def _csv_row_to_record(row: dict[str, str]) -> dict[str, Any]:
    """Convert a CSV row to an import record."""
    tags = []
    if row.get("tags"):
        try:
            tags = json.loads(row["tags"])
        except json.JSONDecodeError:
            tags = [t.strip() for t in row["tags"].split(",") if t.strip()]
    
    return {
        "id": row.get("id"),
        "content": row["content"],
        "branch": row.get("branch"),
        "metadata": {
            "source": row.get("source"),
            "confidence": float(row.get("confidence", 1.0)),
            "tags": tags,
        },
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "version": int(row.get("version", 1)),
    }


def _memory_to_parquet_record(
    memory: "Memory",
    branch: str,
    config: ExportConfig,
) -> dict[str, Any]:
    """Convert a Memory object to a Parquet-compatible record."""
    metadata = memory.metadata
    
    return {
        "id": memory.id,
        "content": memory.content,
        "source": metadata.source or "",
        "confidence": metadata.confidence,
        "tags": json.dumps(metadata.tags),
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        "version": memory.version,
        "branch": branch,
        "metadata_json": json.dumps(metadata.to_dict()),
    }


def _write_memory_markdown(
    f: TextIO,
    memory: "Memory",
    config: ExportConfig,
) -> None:
    """Write a memory entry in Markdown format."""
    f.write(f"### Memory: `{memory.id[:8]}...`\n\n")
    
    # Content block
    f.write("**Content:**\n\n")
    f.write(f"> {memory.content}\n\n")
    
    # Metadata table
    if config.include_metadata:
        metadata = memory.metadata
        f.write("| Property | Value |\n")
        f.write("|----------|-------|\n")
        f.write(f"| ID | `{memory.id}` |\n")
        f.write(f"| Created | {memory.created_at.strftime(config.date_format)} |\n")
        f.write(f"| Updated | {memory.updated_at.strftime(config.date_format)} |\n")
        f.write(f"| Version | {memory.version} |\n")
        if metadata.source:
            f.write(f"| Source | {metadata.source} |\n")
        f.write(f"| Confidence | {metadata.confidence:.2f} |\n")
        if metadata.tags:
            f.write(f"| Tags | {', '.join(metadata.tags)} |\n")
    
    f.write("\n")


def _import_memory_record(
    store: "MemoryStore",
    record: dict[str, Any],
    config: ImportConfig,
) -> str:
    """Import a single memory record.
    
    Returns:
        "imported", "skipped", or "replaced"
    """
    from agent_memory_toolkit.store import MemoryMetadata
    
    content = record.get("content")
    if not content:
        raise ValueError("Record missing required 'content' field")
    
    original_id = record.get("id")
    
    # Check if memory already exists (by ID)
    exists = False
    if original_id and config.preserve_ids:
        try:
            store.get(original_id)
            exists = True
        except Exception:
            pass
    
    # Handle existing memories
    if exists:
        if config.merge_strategy == "skip":
            return "skipped"
        elif config.merge_strategy == "error":
            raise ValueError(f"Memory {original_id} already exists")
        elif config.merge_strategy == "replace":
            # Update existing
            metadata_dict = record.get("metadata", {})
            metadata = MemoryMetadata.from_dict(metadata_dict)
            store.update(original_id, content=content, metadata=metadata)
            return "replaced"
    
    # Create new memory
    metadata_dict = record.get("metadata", {})
    metadata = MemoryMetadata.from_dict(metadata_dict)
    
    # We can't easily preserve IDs in the current MemoryStore implementation
    # as Memory.create() generates a new UUID. For now, just create new.
    store.add(content, metadata=metadata)
    
    return "imported"
