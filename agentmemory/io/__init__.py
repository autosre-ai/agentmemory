"""Agent Memory Toolkit I/O Module.

Provides export and import functionality for memory data in multiple formats:
- JSONL (JSON Lines)
- Parquet (optional, requires pyarrow)
- SQLite (dump/restore)
- CSV
- Markdown (human-readable)

Example:
    >>> from agentmemory.io import export_jsonl, import_jsonl
    >>> export_jsonl(store, "memories.jsonl")
    >>> import_jsonl(store, "memories.jsonl")
"""

from .formats import (
    # Core export/import functions
    export_jsonl,
    import_jsonl,
    export_csv,
    import_csv,
    export_markdown,
    export_sqlite_dump,
    import_sqlite_dump,
    # Parquet functions (optional)
    export_parquet,
    import_parquet,
    PARQUET_AVAILABLE,
    # Utility classes
    ExportConfig,
    ImportConfig,
    ExportResult,
    ImportResult,
    ExportFormat,
)

__all__ = [
    # Core functions
    "export_jsonl",
    "import_jsonl",
    "export_csv",
    "import_csv",
    "export_markdown",
    "export_sqlite_dump",
    "import_sqlite_dump",
    # Parquet
    "export_parquet",
    "import_parquet",
    "PARQUET_AVAILABLE",
    # Classes
    "ExportConfig",
    "ImportConfig",
    "ExportResult",
    "ImportResult",
    "ExportFormat",
]
