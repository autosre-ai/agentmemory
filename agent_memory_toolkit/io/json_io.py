"""JSON Import/Export functionality for memory persistence.

This module provides utilities for importing and exporting memories to JSON
and JSONL (JSON Lines) format, enabling easy backup, migration, and
interoperability with other systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Callable
import json
import gzip
import logging

logger = logging.getLogger(__name__)


class JSONFormat(Enum):
    """JSON output format options."""
    COMPACT = "compact"      # Minified JSON
    PRETTY = "pretty"        # Human-readable with indentation
    JSONL = "jsonl"          # JSON Lines (one record per line)


@dataclass
class JSONIOConfig:
    """Configuration for JSON import/export operations.
    
    Attributes:
        format: Output format (compact, pretty, or jsonl)
        compress: Whether to gzip compress the output
        include_embeddings: Whether to include embedding vectors
        include_versions: Whether to include version history
        include_branches: Whether to include branch information
        date_format: ISO format string for datetime serialization
        encoding: File encoding (default: utf-8)
        chunk_size: Number of memories to process at a time for streaming
        validate_on_import: Whether to validate memories on import
        skip_invalid: Whether to skip invalid records or raise error
    """
    format: JSONFormat = JSONFormat.PRETTY
    compress: bool = False
    include_embeddings: bool = True
    include_versions: bool = True
    include_branches: bool = True
    date_format: str = "iso"
    encoding: str = "utf-8"
    chunk_size: int = 1000
    validate_on_import: bool = True
    skip_invalid: bool = False


@dataclass
class ExportResult:
    """Result of an export operation.
    
    Attributes:
        path: Path to the exported file
        format: Format used for export
        memory_count: Number of memories exported
        branch_count: Number of branches exported
        version_count: Number of versions exported
        file_size_bytes: Size of the exported file
        compressed: Whether the file is compressed
        duration_seconds: Time taken for export
        errors: List of any errors encountered
    """
    path: Path
    format: JSONFormat
    memory_count: int
    branch_count: int = 0
    version_count: int = 0
    file_size_bytes: int = 0
    compressed: bool = False
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    """Result of an import operation.
    
    Attributes:
        memory_count: Number of memories imported
        branch_count: Number of branches imported
        version_count: Number of versions imported
        skipped_count: Number of records skipped
        duplicate_count: Number of duplicates detected
        duration_seconds: Time taken for import
        errors: List of any errors encountered
        warnings: List of any warnings
    """
    memory_count: int
    branch_count: int = 0
    version_count: int = 0
    skipped_count: int = 0
    duplicate_count: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class JSONExporter:
    """Export memories to JSON format.
    
    Example:
        ```python
        from agent_memory_toolkit.io import JSONExporter, JSONIOConfig, JSONFormat
        
        # Create exporter with custom config
        config = JSONIOConfig(
            format=JSONFormat.PRETTY,
            compress=True,
            include_embeddings=False,
        )
        exporter = JSONExporter(config=config)
        
        # Export memories from a MemoryStore
        result = exporter.export(
            memories=store.list_all(),
            output_path="backup.json.gz"
        )
        print(f"Exported {result.memory_count} memories")
        ```
    """
    
    def __init__(self, config: JSONIOConfig | None = None):
        """Initialize the exporter.
        
        Args:
            config: Export configuration options
        """
        self.config = config or JSONIOConfig()
    
    def export(
        self,
        memories: list[Any],
        output_path: str | Path,
        branches: list[Any] | None = None,
        versions: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExportResult:
        """Export memories to a JSON file.
        
        Args:
            memories: List of Memory objects to export
            output_path: Path to the output file
            branches: Optional list of Branch objects
            versions: Optional list of version history records
            metadata: Optional metadata to include in export
            
        Returns:
            ExportResult with export statistics
        """
        import time
        start_time = time.time()
        
        output_path = Path(output_path)
        errors: list[str] = []
        
        # Build export data structure
        export_data = {
            "format_version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "toolkit_version": self._get_toolkit_version(),
            "metadata": metadata or {},
            "memories": [],
            "branches": [],
            "versions": [],
        }
        
        # Export memories
        memory_count = 0
        for memory in memories:
            try:
                memory_dict = self._serialize_memory(memory)
                export_data["memories"].append(memory_dict)
                memory_count += 1
            except Exception as e:
                errors.append(f"Failed to serialize memory {getattr(memory, 'id', 'unknown')}: {e}")
        
        # Export branches if included
        branch_count = 0
        if self.config.include_branches and branches:
            for branch in branches:
                try:
                    branch_dict = self._serialize_branch(branch)
                    export_data["branches"].append(branch_dict)
                    branch_count += 1
                except Exception as e:
                    errors.append(f"Failed to serialize branch: {e}")
        
        # Export versions if included
        version_count = 0
        if self.config.include_versions and versions:
            for version in versions:
                try:
                    version_dict = self._serialize_version(version)
                    export_data["versions"].append(version_dict)
                    version_count += 1
                except Exception as e:
                    errors.append(f"Failed to serialize version: {e}")
        
        # Write to file
        file_size = self._write_json(export_data, output_path)
        
        duration = time.time() - start_time
        
        return ExportResult(
            path=output_path,
            format=self.config.format,
            memory_count=memory_count,
            branch_count=branch_count,
            version_count=version_count,
            file_size_bytes=file_size,
            compressed=self.config.compress,
            duration_seconds=duration,
            errors=errors,
        )
    
    def export_stream(
        self,
        memory_iterator: Iterator[Any],
        output_path: str | Path,
        total_count: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> ExportResult:
        """Export memories using streaming for large datasets.
        
        Args:
            memory_iterator: Iterator yielding Memory objects
            output_path: Path to the output file
            total_count: Optional total count for progress tracking
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            ExportResult with export statistics
        """
        import time
        start_time = time.time()
        
        output_path = Path(output_path)
        errors: list[str] = []
        
        # For streaming, we use JSONL format
        memory_count = 0
        
        open_func = gzip.open if self.config.compress else open
        mode = "wt" if not self.config.compress else "wt"
        
        with open_func(output_path, mode, encoding=self.config.encoding) as f:
            # Write header
            header = {
                "format_version": "1.0",
                "exported_at": datetime.utcnow().isoformat(),
                "type": "header",
            }
            f.write(json.dumps(header) + "\n")
            
            # Stream memories
            for memory in memory_iterator:
                try:
                    memory_dict = self._serialize_memory(memory)
                    memory_dict["type"] = "memory"
                    f.write(json.dumps(memory_dict) + "\n")
                    memory_count += 1
                    
                    if progress_callback:
                        progress_callback(memory_count, total_count)
                except Exception as e:
                    errors.append(f"Failed to serialize memory: {e}")
        
        duration = time.time() - start_time
        file_size = output_path.stat().st_size
        
        return ExportResult(
            path=output_path,
            format=JSONFormat.JSONL,
            memory_count=memory_count,
            file_size_bytes=file_size,
            compressed=self.config.compress,
            duration_seconds=duration,
            errors=errors,
        )
    
    def _serialize_memory(self, memory: Any) -> dict[str, Any]:
        """Serialize a memory object to a dictionary."""
        if hasattr(memory, "to_dict"):
            data = memory.to_dict()
        else:
            data = dict(memory) if hasattr(memory, "__iter__") else vars(memory)
        
        # Remove embeddings if not included
        if not self.config.include_embeddings and "embedding" in data:
            del data["embedding"]
        
        return data
    
    def _serialize_branch(self, branch: Any) -> dict[str, Any]:
        """Serialize a branch object to a dictionary."""
        if hasattr(branch, "to_dict"):
            return branch.to_dict()
        return dict(branch) if hasattr(branch, "__iter__") else vars(branch)
    
    def _serialize_version(self, version: Any) -> dict[str, Any]:
        """Serialize a version object to a dictionary."""
        if hasattr(version, "to_dict"):
            return version.to_dict()
        return dict(version) if hasattr(version, "__iter__") else vars(version)
    
    def _write_json(self, data: dict, path: Path) -> int:
        """Write JSON data to file, optionally compressed."""
        if self.config.format == JSONFormat.PRETTY:
            json_str = json.dumps(data, indent=2, default=str)
        else:
            json_str = json.dumps(data, default=str)
        
        if self.config.compress:
            with gzip.open(path, "wt", encoding=self.config.encoding) as f:
                f.write(json_str)
        else:
            with open(path, "w", encoding=self.config.encoding) as f:
                f.write(json_str)
        
        return path.stat().st_size
    
    def _get_toolkit_version(self) -> str:
        """Get the toolkit version string."""
        try:
            from agent_memory_toolkit import __version__
            return __version__
        except ImportError:
            return "unknown"


class JSONImporter:
    """Import memories from JSON format.
    
    Example:
        ```python
        from agent_memory_toolkit.io import JSONImporter, JSONIOConfig
        
        # Create importer
        importer = JSONImporter()
        
        # Import memories
        memories, result = importer.import_file("backup.json")
        print(f"Imported {result.memory_count} memories")
        
        # Add to store
        for memory in memories:
            store.add(memory)
        ```
    """
    
    def __init__(self, config: JSONIOConfig | None = None):
        """Initialize the importer.
        
        Args:
            config: Import configuration options
        """
        self.config = config or JSONIOConfig()
    
    def import_file(
        self,
        input_path: str | Path,
        memory_class: type | None = None,
        branch_class: type | None = None,
    ) -> tuple[list[Any], ImportResult]:
        """Import memories from a JSON file.
        
        Args:
            input_path: Path to the input file
            memory_class: Optional class to deserialize memories into
            branch_class: Optional class to deserialize branches into
            
        Returns:
            Tuple of (list of memories, ImportResult)
        """
        import time
        start_time = time.time()
        
        input_path = Path(input_path)
        errors: list[str] = []
        warnings: list[str] = []
        
        # Read file
        data = self._read_json(input_path)
        
        # Parse memories
        memories: list[Any] = []
        skipped = 0
        
        raw_memories = data.get("memories", [])
        for i, mem_data in enumerate(raw_memories):
            try:
                if self.config.validate_on_import:
                    self._validate_memory_data(mem_data)
                
                if memory_class and hasattr(memory_class, "from_dict"):
                    memory = memory_class.from_dict(mem_data)
                else:
                    memory = mem_data
                
                memories.append(memory)
            except Exception as e:
                if self.config.skip_invalid:
                    skipped += 1
                    warnings.append(f"Skipped memory {i}: {e}")
                else:
                    errors.append(f"Failed to import memory {i}: {e}")
        
        # Parse branches if included
        branches: list[Any] = []
        raw_branches = data.get("branches", [])
        for branch_data in raw_branches:
            try:
                if branch_class and hasattr(branch_class, "from_dict"):
                    branch = branch_class.from_dict(branch_data)
                else:
                    branch = branch_data
                branches.append(branch)
            except Exception as e:
                warnings.append(f"Failed to import branch: {e}")
        
        # Parse versions
        versions = data.get("versions", [])
        
        duration = time.time() - start_time
        
        result = ImportResult(
            memory_count=len(memories),
            branch_count=len(branches),
            version_count=len(versions),
            skipped_count=skipped,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings,
        )
        
        return memories, result
    
    def import_stream(
        self,
        input_path: str | Path,
        memory_class: type | None = None,
        batch_size: int | None = None,
    ) -> Iterator[list[Any]]:
        """Import memories using streaming for large files.
        
        Args:
            input_path: Path to the input JSONL file
            memory_class: Optional class to deserialize memories into
            batch_size: Number of memories per batch (default from config)
            
        Yields:
            Batches of memory objects
        """
        input_path = Path(input_path)
        batch_size = batch_size or self.config.chunk_size
        
        open_func = gzip.open if str(input_path).endswith(".gz") else open
        
        batch: list[Any] = []
        
        with open_func(input_path, "rt", encoding=self.config.encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Skip header records
                    if data.get("type") == "header":
                        continue
                    
                    # Handle memory records
                    if data.get("type") == "memory":
                        del data["type"]
                    
                    if memory_class and hasattr(memory_class, "from_dict"):
                        memory = memory_class.from_dict(data)
                    else:
                        memory = data
                    
                    batch.append(memory)
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                except Exception as e:
                    logger.warning(f"Failed to parse memory: {e}")
        
        # Yield remaining batch
        if batch:
            yield batch
    
    def _read_json(self, path: Path) -> dict[str, Any]:
        """Read JSON data from file, handling compression."""
        open_func = gzip.open if str(path).endswith(".gz") else open
        
        with open_func(path, "rt", encoding=self.config.encoding) as f:
            content = f.read()
            return json.loads(content)
    
    def _validate_memory_data(self, data: dict[str, Any]) -> None:
        """Validate memory data structure."""
        required_fields = ["id", "content"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate types
        if not isinstance(data.get("content"), str):
            raise ValueError("Content must be a string")
        
        # Validate dates if present
        for date_field in ["created_at", "updated_at"]:
            if date_field in data and data[date_field]:
                try:
                    datetime.fromisoformat(data[date_field])
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid date format for {date_field}")


# Convenience functions

def export_to_json(
    memories: list[Any],
    output_path: str | Path,
    compress: bool = False,
    pretty: bool = True,
    include_embeddings: bool = True,
) -> ExportResult:
    """Export memories to a JSON file.
    
    Args:
        memories: List of Memory objects to export
        output_path: Path to the output file
        compress: Whether to gzip compress the output
        pretty: Whether to use pretty formatting
        include_embeddings: Whether to include embedding vectors
        
    Returns:
        ExportResult with export statistics
        
    Example:
        ```python
        from agent_memory_toolkit.io import export_to_json
        
        result = export_to_json(
            memories=store.list_all(),
            output_path="memories_backup.json",
            compress=True,
        )
        ```
    """
    config = JSONIOConfig(
        format=JSONFormat.PRETTY if pretty else JSONFormat.COMPACT,
        compress=compress,
        include_embeddings=include_embeddings,
    )
    exporter = JSONExporter(config=config)
    return exporter.export(memories, output_path)


def import_from_json(
    input_path: str | Path,
    memory_class: type | None = None,
    validate: bool = True,
) -> tuple[list[Any], ImportResult]:
    """Import memories from a JSON file.
    
    Args:
        input_path: Path to the input file
        memory_class: Optional class to deserialize memories into
        validate: Whether to validate memories on import
        
    Returns:
        Tuple of (list of memories, ImportResult)
        
    Example:
        ```python
        from agent_memory_toolkit.io import import_from_json
        from agent_memory_toolkit import Memory
        
        memories, result = import_from_json(
            "memories_backup.json",
            memory_class=Memory,
        )
        ```
    """
    config = JSONIOConfig(validate_on_import=validate)
    importer = JSONImporter(config=config)
    return importer.import_file(input_path, memory_class=memory_class)


def export_to_jsonl(
    memory_iterator: Iterator[Any],
    output_path: str | Path,
    compress: bool = False,
) -> ExportResult:
    """Export memories to a JSONL (JSON Lines) file using streaming.
    
    Args:
        memory_iterator: Iterator yielding Memory objects
        output_path: Path to the output file
        compress: Whether to gzip compress the output
        
    Returns:
        ExportResult with export statistics
        
    Example:
        ```python
        from agent_memory_toolkit.io import export_to_jsonl
        
        # Stream export for large datasets
        result = export_to_jsonl(
            memory_iterator=store.iter_all(),
            output_path="memories.jsonl.gz",
            compress=True,
        )
        ```
    """
    config = JSONIOConfig(format=JSONFormat.JSONL, compress=compress)
    exporter = JSONExporter(config=config)
    return exporter.export_stream(memory_iterator, output_path)


def import_from_jsonl(
    input_path: str | Path,
    memory_class: type | None = None,
    batch_size: int = 1000,
) -> Iterator[list[Any]]:
    """Import memories from a JSONL file using streaming.
    
    Args:
        input_path: Path to the input JSONL file
        memory_class: Optional class to deserialize memories into
        batch_size: Number of memories per batch
        
    Yields:
        Batches of memory objects
        
    Example:
        ```python
        from agent_memory_toolkit.io import import_from_jsonl
        from agent_memory_toolkit import Memory
        
        # Stream import for large datasets
        for batch in import_from_jsonl("memories.jsonl", Memory):
            for memory in batch:
                store.add(memory)
        ```
    """
    config = JSONIOConfig(chunk_size=batch_size)
    importer = JSONImporter(config=config)
    yield from importer.import_stream(input_path, memory_class=memory_class)
