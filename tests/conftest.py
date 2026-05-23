"""Pytest configuration and fixtures."""

import pytest
from agent_memory_toolkit.store import MemoryStore


@pytest.fixture
def memory_store():
    """Provide an in-memory MemoryStore for tests."""
    store = MemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def populated_store():
    """Provide a MemoryStore with sample data."""
    store = MemoryStore(":memory:")
    
    # Add some sample memories
    store.add(
        "The capital of France is Paris",
        metadata={"source": "geography", "tags": ["france", "capital"]}
    )
    store.add(
        "Python is a programming language",
        metadata={"source": "tech", "tags": ["programming"]}
    )
    store.add(
        "Machine learning is a subset of artificial intelligence",
        metadata={"source": "tech", "tags": ["ml", "ai"]}
    )
    
    yield store
    store.close()
