"""Tests for Export/Import CLI commands."""

import json
import os
import tempfile
from pathlib import Path
import pytest
from click.testing import CliRunner

from agentmemory.cli import cli
from agentmemory.store import MemoryStore, MemoryMetadata


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def db_with_data():
    """Create a temporary database with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    with MemoryStore(db_path) as store:
        store.add(
            "The capital of France is Paris",
            metadata=MemoryMetadata(source="geography", tags=["geography"]),
        )
        store.add(
            "Python was created by Guido van Rossum",
            metadata=MemoryMetadata(source="programming", tags=["python"]),
        )
    
    yield db_path
    
    try:
        os.unlink(db_path)
    except Exception:
        pass


class TestExportCommand:
    """Tests for 'amt export' command."""
    
    def test_export_jsonl(self, cli_runner, db_with_data):
        """Test exporting to JSONL format."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name
        
        try:
            result = cli_runner.invoke(cli, [
                "export", output_path,
                "--db", db_with_data,
                "--format", "jsonl",
            ])
            
            assert result.exit_code == 0
            assert "Exported 2 memories" in result.output
            assert "Format: jsonl" in result.output
            
            # Verify file content
            with open(output_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2
            
        finally:
            os.unlink(output_path)
    
    def test_export_csv(self, cli_runner, db_with_data):
        """Test exporting to CSV format."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name
        
        try:
            result = cli_runner.invoke(cli, [
                "export", output_path,
                "--db", db_with_data,
                "--format", "csv",
            ])
            
            assert result.exit_code == 0
            assert "Exported 2 memories" in result.output
            assert "Format: csv" in result.output
            
        finally:
            os.unlink(output_path)
    
    def test_export_markdown(self, cli_runner, db_with_data):
        """Test exporting to Markdown format."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_path = f.name
        
        try:
            result = cli_runner.invoke(cli, [
                "export", output_path,
                "--db", db_with_data,
                "--format", "markdown",
            ])
            
            assert result.exit_code == 0
            assert "Exported 2 memories" in result.output
            
            # Verify it's valid markdown
            with open(output_path, "r") as f:
                content = f.read()
            assert "# Agent Memory Export" in content
            
        finally:
            os.unlink(output_path)
    
    def test_export_sqlite(self, cli_runner, db_with_data):
        """Test exporting to SQLite format."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            output_path = f.name
        
        try:
            result = cli_runner.invoke(cli, [
                "export", output_path,
                "--db", db_with_data,
                "--format", "sqlite",
            ])
            
            assert result.exit_code == 0
            assert "Exported 2 memories" in result.output
            assert "Format: sqlite" in result.output
            
        finally:
            os.unlink(output_path)
    
    def test_export_with_branch_filter(self, cli_runner):
        """Test exporting specific branches."""
        # Create DB with multiple branches
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_path = f.name
        
        try:
            with MemoryStore(db_path) as store:
                store.add("Main memory")
                store.create_branch("feature")
                store.checkout("feature")
                store.add("Feature memory")
            
            result = cli_runner.invoke(cli, [
                "export", output_path,
                "--db", db_path,
                "--branch", "main",
            ])
            
            assert result.exit_code == 0
            assert "Branches: main" in result.output
            
        finally:
            os.unlink(db_path)
            os.unlink(output_path)


class TestImportCommand:
    """Tests for 'amt import' command."""
    
    def test_import_jsonl(self, cli_runner):
        """Test importing from JSONL format."""
        # Create JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(3):
                record = {
                    "content": f"Test memory {i}",
                    "metadata": {"source": "test"},
                }
                f.write(json.dumps(record) + "\n")
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            result = cli_runner.invoke(cli, [
                "import", input_path,
                "--db", db_path,
            ])
            
            assert result.exit_code == 0
            assert "Import complete" in result.output
            assert "Imported: 3" in result.output
            
            # Verify data was imported
            with MemoryStore(db_path) as store:
                memories = store.list()
                assert len(memories) == 3
                
        finally:
            os.unlink(input_path)
            os.unlink(db_path)
    
    def test_import_csv(self, cli_runner):
        """Test importing from CSV format."""
        # Create CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,content,source,confidence,tags,created_at,updated_at,version,branch\n")
            f.write('test-1,Test memory 1,test,1.0,"[]",2024-01-01 00:00:00,2024-01-01 00:00:00,1,main\n')
            f.write('test-2,Test memory 2,test,1.0,"[]",2024-01-01 00:00:00,2024-01-01 00:00:00,1,main\n')
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            result = cli_runner.invoke(cli, [
                "import", input_path,
                "--db", db_path,
            ])
            
            assert result.exit_code == 0
            assert "Import complete" in result.output
            assert "Imported: 2" in result.output
            
        finally:
            os.unlink(input_path)
            os.unlink(db_path)
    
    def test_import_auto_detect_format(self, cli_runner):
        """Test format auto-detection from file extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"content": "Auto-detected memory"}) + "\n")
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            # Don't specify --format, let it auto-detect from .jsonl extension
            result = cli_runner.invoke(cli, [
                "import", input_path,
                "--db", db_path,
            ])
            
            assert result.exit_code == 0
            assert "Format: jsonl" in result.output
            
        finally:
            os.unlink(input_path)
            os.unlink(db_path)
    
    def test_import_merge_strategy(self, cli_runner):
        """Test import with merge strategy."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"content": "Duplicate memory"}) + "\n")
            f.write(json.dumps({"content": "Duplicate memory"}) + "\n")
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            result = cli_runner.invoke(cli, [
                "import", input_path,
                "--db", db_path,
                "--merge", "skip",
            ])
            
            assert result.exit_code == 0
            
        finally:
            os.unlink(input_path)
            os.unlink(db_path)


class TestRoundTrip:
    """Test round-trip export/import via CLI."""
    
    def test_jsonl_round_trip(self, cli_runner, db_with_data):
        """Test exporting and re-importing via JSONL."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            export_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            new_db = f.name
        
        try:
            # Export
            result1 = cli_runner.invoke(cli, [
                "export", export_path,
                "--db", db_with_data,
                "--format", "jsonl",
            ])
            assert result1.exit_code == 0
            
            # Import to new DB
            result2 = cli_runner.invoke(cli, [
                "import", export_path,
                "--db", new_db,
            ])
            assert result2.exit_code == 0
            assert "Imported: 2" in result2.output
            
            # Verify
            with MemoryStore(new_db) as store:
                memories = store.list()
                assert len(memories) == 2
                
        finally:
            os.unlink(export_path)
            os.unlink(new_db)
