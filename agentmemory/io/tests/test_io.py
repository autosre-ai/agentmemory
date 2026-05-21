"""Tests for Agent Memory Toolkit I/O module.

Tests export and import functionality for all supported formats.
"""

import csv
import json
import os
import tempfile
from pathlib import Path
import pytest

from agentmemory.store import MemoryStore, MemoryMetadata
from agentmemory.io import (
    export_jsonl,
    import_jsonl,
    export_csv,
    import_csv,
    export_markdown,
    export_sqlite_dump,
    import_sqlite_dump,
    export_parquet,
    import_parquet,
    PARQUET_AVAILABLE,
    ExportConfig,
    ImportConfig,
    ExportFormat,
)


@pytest.fixture
def store():
    """Create a temporary memory store with sample data."""
    with MemoryStore(":memory:") as store:
        # Add some test memories
        store.add(
            "The capital of France is Paris",
            metadata=MemoryMetadata(
                source="geography",
                confidence=0.95,
                tags=["geography", "france", "cities"],
            ),
        )
        store.add(
            "Python was created by Guido van Rossum",
            metadata=MemoryMetadata(
                source="programming",
                confidence=0.99,
                tags=["programming", "python"],
            ),
        )
        store.add(
            "The speed of light is approximately 299,792 km/s",
            metadata=MemoryMetadata(
                source="physics",
                confidence=1.0,
                tags=["physics", "constants"],
            ),
        )
        yield store


@pytest.fixture
def store_with_branches():
    """Create a store with multiple branches."""
    with MemoryStore(":memory:") as store:
        # Add to main branch
        store.add("Memory on main branch", metadata=MemoryMetadata(source="main"))
        store.commit("Add main memory")
        
        # Create feature branch (inherits main's memory)
        store.create_branch("feature")
        store.checkout("feature")
        store.add("Memory on feature branch", metadata=MemoryMetadata(source="feature"))
        store.commit("Add feature memory")
        
        # Back to main
        store.checkout("main")
        yield store


class TestJSONLExportImport:
    """Tests for JSONL format."""
    
    def test_export_jsonl_basic(self, store):
        """Test basic JSONL export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name
        
        try:
            result = export_jsonl(store, output_path)
            
            assert result.format == ExportFormat.JSONL
            assert result.memory_count == 3
            assert result.file_size_bytes > 0
            assert "main" in result.branches_exported
            assert len(result.errors) == 0
            
            # Verify file content
            with open(output_path, "r") as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            
            for line in lines:
                record = json.loads(line)
                assert "id" in record
                assert "content" in record
                assert "metadata" in record
                assert "branch" in record
                
        finally:
            os.unlink(output_path)
    
    def test_export_jsonl_without_metadata(self, store):
        """Test JSONL export without metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name
        
        try:
            config = ExportConfig(include_metadata=False)
            result = export_jsonl(store, output_path, config)
            
            with open(output_path, "r") as f:
                line = f.readline()
            
            record = json.loads(line)
            assert "metadata" not in record
            
        finally:
            os.unlink(output_path)
    
    def test_import_jsonl_basic(self, store):
        """Test basic JSONL import."""
        # First export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name
        
        try:
            export_jsonl(store, output_path)
            
            # Import into new store
            with MemoryStore(":memory:") as new_store:
                result = import_jsonl(new_store, output_path)
                
                assert result.format == ExportFormat.JSONL
                assert result.memories_imported == 3
                assert result.memories_skipped == 0
                assert len(result.errors) == 0
                
                # Verify imported data
                memories = new_store.list()
                assert len(memories) == 3
                
        finally:
            os.unlink(output_path)
    
    def test_import_jsonl_skip_existing(self):
        """Test JSONL import with skip strategy."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write sample JSONL
            for i in range(3):
                record = {
                    "id": f"test-id-{i}",
                    "content": f"Test memory {i}",
                    "branch": "main",
                    "created_at": "2024-01-01 00:00:00",
                    "updated_at": "2024-01-01 00:00:00",
                    "version": 1,
                    "is_deleted": False,
                    "metadata": {"source": "test", "confidence": 1.0, "tags": []},
                }
                f.write(json.dumps(record) + "\n")
            output_path = f.name
        
        try:
            with MemoryStore(":memory:") as store:
                # Import once
                result1 = import_jsonl(store, output_path)
                assert result1.memories_imported == 3
                
                # Import again with skip strategy (default)
                result2 = import_jsonl(store, output_path)
                # New memories will be created (IDs aren't preserved by default)
                assert result2.memories_imported == 3
                
        finally:
            os.unlink(output_path)
    
    def test_import_jsonl_invalid_json(self):
        """Test JSONL import handles invalid JSON gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"content": "valid"}\n')
            f.write('not valid json\n')
            f.write('{"content": "also valid"}\n')
            output_path = f.name
        
        try:
            with MemoryStore(":memory:") as store:
                result = import_jsonl(store, output_path)
                
                assert result.memories_imported == 2
                assert len(result.errors) == 1
                assert "Line 2" in result.errors[0]
                
        finally:
            os.unlink(output_path)


class TestCSVExportImport:
    """Tests for CSV format."""
    
    def test_export_csv_basic(self, store):
        """Test basic CSV export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name
        
        try:
            result = export_csv(store, output_path)
            
            assert result.format == ExportFormat.CSV
            assert result.memory_count == 3
            
            # Verify CSV structure
            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 3
            
            # Check headers
            expected_fields = {"id", "content", "source", "confidence", "tags", 
                             "created_at", "updated_at", "version", "branch"}
            assert set(rows[0].keys()) == expected_fields
            
        finally:
            os.unlink(output_path)
    
    def test_import_csv_basic(self, store):
        """Test basic CSV import."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name
        
        try:
            export_csv(store, output_path)
            
            with MemoryStore(":memory:") as new_store:
                result = import_csv(new_store, output_path)
                
                assert result.memories_imported == 3
                memories = new_store.list()
                assert len(memories) == 3
                
        finally:
            os.unlink(output_path)
    
    def test_csv_tags_parsing(self):
        """Test that tags are properly serialized and parsed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "content", "source", "confidence", "tags",
                "created_at", "updated_at", "version", "branch"
            ])
            writer.writeheader()
            writer.writerow({
                "id": "test-1",
                "content": "Test memory",
                "source": "test",
                "confidence": "0.9",
                "tags": '["tag1", "tag2"]',
                "created_at": "2024-01-01 00:00:00",
                "updated_at": "2024-01-01 00:00:00",
                "version": "1",
                "branch": "main",
            })
            output_path = f.name
        
        try:
            with MemoryStore(":memory:") as store:
                result = import_csv(store, output_path)
                
                assert result.memories_imported == 1
                
                memories = store.list()
                assert len(memories) == 1
                assert "tag1" in memories[0].metadata.tags
                assert "tag2" in memories[0].metadata.tags
                
        finally:
            os.unlink(output_path)


class TestMarkdownExport:
    """Tests for Markdown format (export only)."""
    
    def test_export_markdown_basic(self, store):
        """Test basic Markdown export."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            output_path = f.name
        
        try:
            result = export_markdown(store, output_path)
            
            assert result.format == ExportFormat.MARKDOWN
            assert result.memory_count == 3
            
            with open(output_path, "r") as f:
                content = f.read()
            
            # Check structure
            assert "# Agent Memory Export" in content
            assert "## Branch: `main`" in content
            assert "### Memory:" in content
            assert "**Content:**" in content
            assert "The capital of France is Paris" in content
            
        finally:
            os.unlink(output_path)
    
    def test_export_markdown_with_branches(self, store_with_branches):
        """Test Markdown export with multiple branches."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            output_path = f.name
        
        try:
            result = export_markdown(store_with_branches, output_path)
            
            # main has 1 memory, feature has 2 (1 copied from main + 1 new)
            assert result.memory_count == 3
            assert "main" in result.branches_exported
            assert "feature" in result.branches_exported
            
            with open(output_path, "r") as f:
                content = f.read()
            
            assert "## Branch: `main`" in content
            assert "## Branch: `feature`" in content
            
        finally:
            os.unlink(output_path)


class TestSQLiteDumpRestore:
    """Tests for SQLite dump/restore."""
    
    def test_export_sqlite_dump(self, store):
        """Test SQLite backup export."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".db", delete=False) as f:
            output_path = f.name
        
        try:
            result = export_sqlite_dump(store, output_path)
            
            assert result.format == ExportFormat.SQLITE
            assert result.memory_count == 3
            
            # Should be a valid SQLite database
            import sqlite3
            conn = sqlite3.connect(output_path)
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            assert cursor.fetchone()[0] == 3
            conn.close()
            
        finally:
            os.unlink(output_path)
    
    def test_import_sqlite_dump(self):
        """Test SQLite backup import."""
        # Create source store with data
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            source_db = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            backup_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            target_db = f.name
        
        try:
            # Create and populate source store
            with MemoryStore(source_db) as source:
                source.add("Test memory 1")
                source.add("Test memory 2")
                export_sqlite_dump(source, backup_path)
            
            # Import into target
            with MemoryStore(target_db) as target:
                result = import_sqlite_dump(target, backup_path)
                
                assert result.memories_imported == 2
                
                # Verify data
                memories = target.list()
                assert len(memories) == 2
                
        finally:
            for path in [source_db, backup_path, target_db]:
                try:
                    os.unlink(path)
                except Exception:
                    pass


@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="pyarrow not installed")
class TestParquetExportImport:
    """Tests for Parquet format."""
    
    def test_export_parquet_basic(self, store):
        """Test basic Parquet export."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".parquet", delete=False) as f:
            output_path = f.name
        
        try:
            result = export_parquet(store, output_path)
            
            assert result.format == ExportFormat.PARQUET
            assert result.memory_count == 3
            
            # Verify it's a valid parquet file
            import pyarrow.parquet as pq
            table = pq.read_table(output_path)
            assert table.num_rows == 3
            
        finally:
            os.unlink(output_path)
    
    def test_import_parquet_basic(self, store):
        """Test basic Parquet import."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".parquet", delete=False) as f:
            output_path = f.name
        
        try:
            export_parquet(store, output_path)
            
            with MemoryStore(":memory:") as new_store:
                result = import_parquet(new_store, output_path)
                
                assert result.memories_imported == 3
                memories = new_store.list()
                assert len(memories) == 3
                
        finally:
            os.unlink(output_path)


class TestParquetNotAvailable:
    """Tests for when pyarrow is not installed."""
    
    @pytest.mark.skipif(PARQUET_AVAILABLE, reason="pyarrow is installed")
    def test_parquet_import_error(self, store):
        """Test that parquet functions raise ImportError when pyarrow not available."""
        with pytest.raises(ImportError) as exc_info:
            export_parquet(store, "test.parquet")
        
        assert "pyarrow" in str(exc_info.value)


class TestExportConfig:
    """Tests for ExportConfig options."""
    
    def test_export_specific_branch(self, store_with_branches):
        """Test exporting only specific branches."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name
        
        try:
            config = ExportConfig(branches=["feature"])
            result = export_jsonl(store_with_branches, output_path, config)
            
            # Feature has 2 memories: 1 copied from main + 1 new
            assert result.memory_count == 2
            assert "feature" in result.branches_exported
            assert "main" not in result.branches_exported
            
        finally:
            os.unlink(output_path)
    
    def test_export_batch_size(self, store):
        """Test export with custom batch size."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name
        
        try:
            config = ExportConfig(batch_size=1)  # Small batch for testing
            result = export_jsonl(store, output_path, config)
            
            assert result.memory_count == 3
            
        finally:
            os.unlink(output_path)


class TestImportConfig:
    """Tests for ImportConfig options."""
    
    def test_import_to_specific_branch(self):
        """Test importing to a specific branch."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "content": "Test memory",
                "metadata": {"source": "test"},
            }) + "\n")
            output_path = f.name
        
        try:
            with MemoryStore(":memory:") as store:
                config = ImportConfig(target_branch="main")
                result = import_jsonl(store, output_path, config)
                
                assert result.memories_imported == 1
                
        finally:
            os.unlink(output_path)


class TestRoundTrip:
    """Test round-trip export/import preserves data."""
    
    def test_jsonl_round_trip(self, store):
        """Test JSONL round-trip preserves content and metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name
        
        try:
            # Export
            export_jsonl(store, output_path)
            
            # Import to new store
            with MemoryStore(":memory:") as new_store:
                import_jsonl(new_store, output_path)
                
                original = store.list()
                imported = new_store.list()
                
                assert len(original) == len(imported)
                
                # Check content matches (order might differ)
                original_content = {m.content for m in original}
                imported_content = {m.content for m in imported}
                assert original_content == imported_content
                
        finally:
            os.unlink(output_path)
    
    def test_csv_round_trip(self, store):
        """Test CSV round-trip preserves content and metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name
        
        try:
            export_csv(store, output_path)
            
            with MemoryStore(":memory:") as new_store:
                import_csv(new_store, output_path)
                
                original = store.list()
                imported = new_store.list()
                
                assert len(original) == len(imported)
                
                original_content = {m.content for m in original}
                imported_content = {m.content for m in imported}
                assert original_content == imported_content
                
        finally:
            os.unlink(output_path)
