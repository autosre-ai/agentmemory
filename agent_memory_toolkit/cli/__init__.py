"""Agent Memory Toolkit CLI.

Provides command-line interface for managing agent memories.

Entry points:
    - amt: Main CLI command
    - agent-memory-toolkit: Alias for amt
"""

from .main import main, cli

__all__ = ["main", "cli"]
