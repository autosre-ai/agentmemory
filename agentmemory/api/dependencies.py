"""FastAPI dependencies for the REST API."""

from __future__ import annotations

from typing import Optional, Generator
import sqlite3

from .config import get_config


def get_memory_store() -> Generator:
    """
    Get a memory store instance as a FastAPI dependency.
    
    Creates a new instance for each request to avoid SQLite threading issues.
    The store is closed after the request completes.
    """
    from agentmemory.store import MemoryStore
    config = get_config()
    
    # Enable SQLite check_same_thread=False for thread safety
    store = MemoryStore(db_path=config.db_path)
    
    # Enable multi-thread access for the connection
    if store._conn:
        store._conn.close()
        store._conn = sqlite3.connect(
            str(store.db_path) if store.db_path != ":memory:" else ":memory:",
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,  # Allow multi-thread access
        )
        store._conn.row_factory = sqlite3.Row
        store._conn.execute("PRAGMA foreign_keys = ON")
    
    try:
        yield store
    finally:
        store.close()


def close_memory_store():
    """Close the memory store connection (no-op with per-request stores)."""
    pass


def reset_memory_store():
    """Reset the memory store (no-op with per-request stores)."""
    pass


class MemoryStoreManager:
    """
    Context manager for memory store lifecycle.
    
    Useful for testing or when you need explicit control.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._store = None
    
    def __enter__(self):
        from agentmemory.store import MemoryStore
        config = get_config()
        db_path = self.db_path or config.db_path
        self._store = MemoryStore(db_path=db_path)
        return self._store
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._store is not None:
            self._store.close()
            self._store = None
