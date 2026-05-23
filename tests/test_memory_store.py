"""Comprehensive tests for the memory store."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from agent_memory_toolkit.store import (
    MemoryStore,
    Memory,
    MemoryMetadata,
    SearchResult,
    Branch,
    Commit,
    MemoryNotFoundError,
    BranchNotFoundError,
    CommitNotFoundError,
    MemoryStoreError,
)


class TestMemoryMetadata:
    """Tests for MemoryMetadata."""

    def test_default_values(self):
        meta = MemoryMetadata()
        assert meta.source is None
        assert meta.confidence == 1.0
        assert meta.tags == []
        assert meta.extra == {}

    def test_with_values(self):
        meta = MemoryMetadata(
            source="test",
            confidence=0.8,
            tags=["important", "fact"],
            extra={"key": "value"},
        )
        assert meta.source == "test"
        assert meta.confidence == 0.8
        assert meta.tags == ["important", "fact"]
        assert meta.extra == {"key": "value"}

    def test_to_dict(self):
        meta = MemoryMetadata(source="test", confidence=0.9)
        d = meta.to_dict()
        assert d["source"] == "test"
        assert d["confidence"] == 0.9

    def test_from_dict(self):
        d = {"source": "web", "confidence": 0.7, "tags": ["tag1"]}
        meta = MemoryMetadata.from_dict(d)
        assert meta.source == "web"
        assert meta.confidence == 0.7
        assert meta.tags == ["tag1"]

    def test_json_serialization(self):
        meta = MemoryMetadata(source="test", tags=["a", "b"])
        json_str = meta.to_json()
        restored = MemoryMetadata.from_json(json_str)
        assert restored.source == "test"
        assert restored.tags == ["a", "b"]


class TestMemory:
    """Tests for Memory model."""

    def test_create(self):
        memory = Memory.create(content="Test content")
        assert memory.content == "Test content"
        assert memory.id is not None
        assert len(memory.id) == 36  # UUID format
        assert memory.version == 1
        assert memory.is_deleted is False
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)

    def test_create_with_metadata(self):
        meta = MemoryMetadata(source="user", confidence=0.9)
        memory = Memory.create(content="Test", metadata=meta)
        assert memory.metadata.source == "user"
        assert memory.metadata.confidence == 0.9

    def test_create_with_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        memory = Memory.create(content="Test", embedding=embedding)
        assert memory.embedding == [0.1, 0.2, 0.3]

    def test_to_dict(self):
        memory = Memory.create(content="Test")
        d = memory.to_dict()
        assert d["content"] == "Test"
        assert "id" in d
        assert "created_at" in d
        assert "metadata" in d

    def test_from_dict(self):
        memory = Memory.create(content="Test")
        d = memory.to_dict()
        restored = Memory.from_dict(d)
        assert restored.id == memory.id
        assert restored.content == memory.content


class TestMemoryStoreCRUD:
    """Tests for CRUD operations."""

    def test_add_memory(self):
        with MemoryStore() as store:
            memory = store.add("Test memory content")
            assert memory.id is not None
            assert memory.content == "Test memory content"
            assert memory.version == 1

    def test_add_memory_with_metadata_dict(self):
        with MemoryStore() as store:
            memory = store.add(
                "Test",
                metadata={"source": "test", "confidence": 0.9}
            )
            assert memory.metadata.source == "test"
            assert memory.metadata.confidence == 0.9

    def test_add_memory_with_metadata_object(self):
        with MemoryStore() as store:
            meta = MemoryMetadata(source="test", tags=["a"])
            memory = store.add("Test", metadata=meta)
            assert memory.metadata.source == "test"
            assert memory.metadata.tags == ["a"]

    def test_add_memory_with_embedding(self):
        with MemoryStore() as store:
            embedding = [0.1, 0.2, 0.3, 0.4]
            memory = store.add("Test", embedding=embedding)
            assert memory.embedding == embedding

    def test_get_memory(self):
        with MemoryStore() as store:
            created = store.add("Test content")
            retrieved = store.get(created.id)
            assert retrieved.id == created.id
            assert retrieved.content == created.content

    def test_get_memory_not_found(self):
        with MemoryStore() as store:
            with pytest.raises(MemoryNotFoundError) as exc:
                store.get("nonexistent-id")
            assert "nonexistent-id" in str(exc.value)

    def test_update_memory_content(self):
        with MemoryStore() as store:
            memory = store.add("Original content")
            updated = store.update(memory.id, content="Updated content")
            assert updated.content == "Updated content"
            assert updated.version == 2
            assert updated.updated_at > memory.updated_at

    def test_update_memory_metadata(self):
        with MemoryStore() as store:
            memory = store.add("Test")
            updated = store.update(
                memory.id,
                metadata={"source": "updated", "tags": ["new"]}
            )
            assert updated.metadata.source == "updated"
            assert updated.metadata.tags == ["new"]

    def test_update_memory_not_found(self):
        with MemoryStore() as store:
            with pytest.raises(MemoryNotFoundError):
                store.update("nonexistent", content="test")

    def test_delete_memory_soft(self):
        with MemoryStore() as store:
            memory = store.add("Test")
            store.delete(memory.id, hard=False)
            
            with pytest.raises(MemoryNotFoundError):
                store.get(memory.id)
            
            # Should still be in deleted list
            deleted = store.list(include_deleted=True)
            assert any(m.id == memory.id for m in deleted)

    def test_delete_memory_hard(self):
        with MemoryStore() as store:
            memory = store.add("Test")
            store.delete(memory.id, hard=True)
            
            with pytest.raises(MemoryNotFoundError):
                store.get(memory.id)
            
            # Should not be in any list
            all_memories = store.list(include_deleted=True)
            assert not any(m.id == memory.id for m in all_memories)

    def test_delete_memory_not_found(self):
        with MemoryStore() as store:
            with pytest.raises(MemoryNotFoundError):
                store.delete("nonexistent")

    def test_list_memories(self):
        with MemoryStore() as store:
            store.add("Memory 1")
            store.add("Memory 2")
            store.add("Memory 3")
            
            memories = store.list()
            assert len(memories) == 3

    def test_list_memories_pagination(self):
        with MemoryStore() as store:
            for i in range(10):
                store.add(f"Memory {i}")
            
            page1 = store.list(limit=5, offset=0)
            page2 = store.list(limit=5, offset=5)
            
            assert len(page1) == 5
            assert len(page2) == 5
            assert page1[0].id != page2[0].id

    def test_list_memories_by_tag(self):
        with MemoryStore() as store:
            store.add("Memory 1", metadata={"tags": ["important"]})
            store.add("Memory 2", metadata={"tags": ["trivial"]})
            store.add("Memory 3", metadata={"tags": ["important", "work"]})
            
            important = store.list(tag="important")
            assert len(important) == 2

    def test_count_memories(self):
        with MemoryStore() as store:
            store.add("Memory 1")
            store.add("Memory 2")
            assert store.count() == 2
            
            store.add("Memory 3")
            assert store.count() == 3


class TestMemoryStoreFTS:
    """Tests for full-text search."""

    def test_fts_basic_search(self):
        with MemoryStore() as store:
            store.add("The capital of France is Paris")
            store.add("Berlin is the capital of Germany")
            store.add("Python is a programming language")
            
            results = store.search_fts("capital")
            assert len(results) == 2
            assert all(r.match_type == "fts" for r in results)

    def test_fts_phrase_search(self):
        with MemoryStore() as store:
            store.add("The quick brown fox")
            store.add("A quick red fox")
            store.add("Brown lazy dog")
            
            # FTS5 phrase search
            results = store.search_fts('"quick brown"')
            assert len(results) == 1

    def test_fts_no_results(self):
        with MemoryStore() as store:
            store.add("Test content")
            results = store.search_fts("nonexistent")
            assert len(results) == 0

    def test_fts_excludes_deleted(self):
        with MemoryStore() as store:
            m1 = store.add("Important fact about science")
            store.add("Another fact about science")
            store.delete(m1.id)
            
            results = store.search_fts("science")
            assert len(results) == 1

    def test_fts_include_deleted(self):
        with MemoryStore() as store:
            m1 = store.add("Important fact about science")
            store.add("Another fact about science")
            store.delete(m1.id)
            
            results = store.search_fts("science", include_deleted=True)
            assert len(results) == 2

    def test_fts_ranking(self):
        with MemoryStore() as store:
            store.add("Python Python Python")  # High term frequency
            store.add("Python is great")
            
            results = store.search_fts("Python", limit=2)
            # Higher score should come first
            assert results[0].score >= results[1].score


class TestMemoryStoreVector:
    """Tests for vector similarity search."""

    def test_vector_search_with_embedding(self):
        with MemoryStore() as store:
            # Add memories with embeddings
            store.add("Memory 1", embedding=[1.0, 0.0, 0.0])
            store.add("Memory 2", embedding=[0.0, 1.0, 0.0])
            store.add("Memory 3", embedding=[0.9, 0.1, 0.0])
            
            # Search with similar embedding
            results = store.search_vector([1.0, 0.0, 0.0], limit=2)
            assert len(results) == 2
            assert results[0].score > results[1].score
            assert results[0].match_type == "vector"

    def test_vector_search_without_provider(self):
        with MemoryStore() as store:
            store.add("Test")
            with pytest.raises(MemoryStoreError):
                store.search_vector("query string")


class TestMemoryStoreHybrid:
    """Tests for hybrid search."""

    def test_hybrid_search_fts_only(self):
        """Test hybrid search falls back to FTS when no embeddings."""
        with MemoryStore() as store:
            store.add("The capital of France is Paris")
            store.add("Python programming language")
            
            results = store.search_hybrid("France capital")
            assert len(results) >= 1
            assert "France" in results[0].memory.content or "capital" in results[0].memory.content

    def test_search_auto_method(self):
        with MemoryStore() as store:
            store.add("Test content")
            results = store.search("content", method="auto")
            assert len(results) == 1


class TestMemoryStoreVersioning:
    """Tests for git-like versioning."""

    def test_create_branch(self):
        with MemoryStore() as store:
            store.add("Memory on main")
            branch = store.create_branch("feature")
            
            assert branch.name == "feature"
            assert branch.is_active is True
            
            branches = store.list_branches()
            assert len(branches) == 2
            assert any(b.name == "feature" for b in branches)

    def test_checkout_branch(self):
        with MemoryStore() as store:
            store.add("Memory on main")
            store.create_branch("feature")
            
            assert store.current_branch == "main"
            store.checkout("feature")
            assert store.current_branch == "feature"

    def test_checkout_nonexistent_branch(self):
        with MemoryStore() as store:
            with pytest.raises(BranchNotFoundError):
                store.checkout("nonexistent")

    def test_branch_isolation(self):
        with MemoryStore() as store:
            store.add("Shared memory")
            store.create_branch("feature")
            store.checkout("feature")
            
            # Add memory only on feature branch
            store.add("Feature-only memory")
            
            # Check feature branch has 2 memories
            assert store.count() == 2
            
            # Switch back to main
            store.checkout("main")
            # Main should have only original memory
            assert store.count() == 1

    def test_delete_branch(self):
        with MemoryStore() as store:
            store.create_branch("feature")
            store.delete_branch("feature")
            
            branches = store.list_branches()
            assert not any(b.name == "feature" for b in branches)

    def test_delete_main_branch_fails(self):
        with MemoryStore() as store:
            with pytest.raises(MemoryStoreError):
                store.delete_branch("main")

    def test_commit(self):
        with MemoryStore() as store:
            store.add("Memory 1")
            store.add("Memory 2")
            
            commit = store.commit("Initial commit")
            
            assert commit.id is not None
            assert commit.message == "Initial commit"
            assert commit.branch == "main"
            assert len(commit.memory_snapshot) == 2

    def test_get_history(self):
        with MemoryStore() as store:
            store.add("Memory 1")
            store.commit("First commit")
            
            store.add("Memory 2")
            store.commit("Second commit")
            
            history = store.get_history()
            assert len(history) == 2
            assert history[0].message == "Second commit"
            assert history[1].message == "First commit"

    def test_rollback(self):
        with MemoryStore() as store:
            m1 = store.add("Original content")
            commit1 = store.commit("Initial state")
            
            store.update(m1.id, content="Modified content")
            store.commit("Modified state")
            
            # Verify current state
            current = store.get(m1.id)
            assert current.content == "Modified content"
            
            # Rollback
            store.rollback(commit1.id)
            
            # Verify rolled back state
            rolled_back = store.get(m1.id)
            assert rolled_back.content == "Original content"

    def test_rollback_nonexistent_commit(self):
        with MemoryStore() as store:
            with pytest.raises(CommitNotFoundError):
                store.rollback("nonexistent-commit-id")

    def test_memory_history(self):
        with MemoryStore() as store:
            memory = store.add("Version 1")
            store.update(memory.id, content="Version 2")
            store.update(memory.id, content="Version 3")
            
            history = store.get_memory_history(memory.id)
            assert len(history) == 3
            assert history[0]["version"] == 3
            assert history[0]["content"] == "Version 3"
            assert history[2]["version"] == 1
            assert history[2]["content"] == "Version 1"


class TestMemoryStoreExportImport:
    """Tests for export/import functionality."""

    def test_export_json_string(self):
        with MemoryStore() as store:
            store.add("Memory 1", metadata={"source": "test"})
            store.add("Memory 2")
            
            json_str = store.export_json()
            data = json.loads(json_str)
            
            assert data["version"] == "1.0"
            assert len(data["memories"]) == 2
            assert "exported_at" in data

    def test_export_json_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        
        try:
            with MemoryStore() as store:
                store.add("Memory 1")
                store.add("Memory 2")
                store.export_json(path)
            
            assert path.exists()
            data = json.loads(path.read_text())
            assert len(data["memories"]) == 2
        finally:
            path.unlink()

    def test_import_json_string(self):
        # Create export data
        with MemoryStore() as store1:
            store1.add("Memory 1")
            store1.add("Memory 2")
            json_str = store1.export_json()
        
        # Import into new store
        with MemoryStore() as store2:
            count = store2.import_json(json_str)
            assert count == 2
            assert store2.count() == 2

    def test_import_json_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        
        try:
            with MemoryStore() as store1:
                store1.add("Memory 1")
                store1.export_json(path)
            
            with MemoryStore() as store2:
                count = store2.import_json(path)
                assert count == 1
        finally:
            path.unlink()

    def test_import_json_merge(self):
        with MemoryStore() as store:
            store.add("Existing memory")
            
            data = {
                "memories": [
                    {
                        "id": "imported-1",
                        "content": "Imported memory",
                        "metadata": {},
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "version": 1,
                        "is_deleted": False,
                    }
                ]
            }
            
            count = store.import_json(data, merge=True)
            assert count == 1
            assert store.count() == 2

    def test_import_json_replace(self):
        with MemoryStore() as store:
            store.add("Existing memory")
            
            data = {
                "memories": [
                    {
                        "id": "imported-1",
                        "content": "Imported memory",
                        "metadata": {},
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "version": 1,
                        "is_deleted": False,
                    }
                ]
            }
            
            count = store.import_json(data, merge=False)
            assert count == 1
            assert store.count() == 1  # Only imported memory


class TestMemoryStorePersistence:
    """Tests for database persistence."""

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            # Create and populate store
            with MemoryStore(db_path) as store:
                store.add("Persistent memory")
                store.commit("Initial")
            
            # Reopen and verify
            with MemoryStore(db_path) as store:
                assert store.count() == 1
                memories = store.list()
                assert memories[0].content == "Persistent memory"
                
                history = store.get_history()
                assert len(history) == 1
        finally:
            db_path.unlink()

    def test_in_memory_database(self):
        with MemoryStore(":memory:") as store:
            store.add("Test")
            assert store.count() == 1


class TestMemoryStoreContext:
    """Tests for context manager behavior."""

    def test_context_manager(self):
        with MemoryStore() as store:
            store.add("Test")
            assert store.count() == 1
        
        # Connection should be closed after context

    def test_explicit_close(self):
        store = MemoryStore()
        store.add("Test")
        store.close()


# Run tests with: pytest tests/test_memory_store.py -v
