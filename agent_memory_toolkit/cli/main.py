"""Main CLI entry point for Agent Memory Toolkit.

Provides the main command group and CLI entry point.
"""

from __future__ import annotations

import click
import sys
from pathlib import Path
from typing import Optional

# Default database path
DEFAULT_DB = "agent_memory.db"


class Context:
    """CLI context object passed between commands."""
    
    def __init__(self, db_path: str, verbose: bool = False, json_output: bool = False):
        self.db_path = db_path
        self.verbose = verbose
        self.json_output = json_output
        self._store = None
    
    @property
    def store(self):
        """Lazy load memory store."""
        if self._store is None:
            from agent_memory_toolkit import MemoryStore
            self._store = MemoryStore(self.db_path)
        return self._store
    
    def echo(self, message: str, err: bool = False) -> None:
        """Print message respecting verbosity settings."""
        click.echo(message, err=err)
    
    def echo_verbose(self, message: str) -> None:
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            click.echo(message, err=True)


pass_context = click.make_pass_decorator(Context)


@click.group()
@click.option("--db", "-d", "db_path", default=DEFAULT_DB,
              envvar="AMT_DB_PATH",
              help=f"Path to memory database (default: {DEFAULT_DB})")
@click.option("--verbose", "-v", is_flag=True,
              help="Enable verbose output")
@click.option("--json", "-j", "json_output", is_flag=True,
              help="Output results as JSON")
@click.version_option(prog_name="amt")
@click.pass_context
def cli(ctx: click.Context, db_path: str, verbose: bool, json_output: bool) -> None:
    """Agent Memory Toolkit - Local-first memory for AI agents.
    
    Manage, search, and version agent memories from the command line.
    
    \b
    Quick Start:
        amt add "User prefers dark mode"
        amt search "preferences"
        amt list --limit 10
    
    \b
    Environment Variables:
        AMT_DB_PATH    Default database path
    
    For MCP server integration, use: amt-mcp serve
    """
    # Expand user path
    db_path_expanded = str(Path(db_path).expanduser())
    ctx.obj = Context(db_path_expanded, verbose, json_output)


# Import and register subcommands
from .commands.memory import memory, add, get, update, delete
from .commands.search import search
from .commands.store import store
from .commands.export import export_cmd, import_cmd
from .commands.info import info, stats

# Register top-level commands
cli.add_command(add)
cli.add_command(search)
cli.add_command(info)
cli.add_command(stats)

# Register command groups
cli.add_command(memory)
cli.add_command(store)
cli.add_command(export_cmd, name="export")
cli.add_command(import_cmd, name="import")


# Add 'list' as top-level alias
@cli.command("list")
@click.option("--limit", "-n", default=20, help="Maximum memories to return")
@click.option("--offset", "-o", default=0, help="Offset for pagination")
@click.option("--branch", "-b", default=None, help="Branch to list from")
@pass_context
def list_memories(ctx: Context, limit: int, offset: int, branch: Optional[str]) -> None:
    """List memories in the store."""
    import json as json_lib
    
    store = ctx.store
    
    if branch:
        store.checkout(branch)
    
    memories = list(store.list(limit=limit, offset=offset))
    
    if ctx.json_output:
        data = [
            {
                "id": m.id,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "tags": list(m.metadata.tags) if m.metadata and m.metadata.tags else [],
            }
            for m in memories
        ]
        click.echo(json_lib.dumps(data, indent=2))
    else:
        if not memories:
            click.echo("No memories found.")
            return
        
        click.echo(f"Memories ({len(memories)} shown):")
        click.echo("-" * 60)
        for m in memories:
            tags = ""
            if m.metadata and m.metadata.tags:
                tags = f" [{', '.join(m.metadata.tags)}]"
            content_preview = m.content[:80] + "..." if len(m.content) > 80 else m.content
            click.echo(f"  {m.id[:8]}  {content_preview}{tags}")


def main() -> int:
    """Entry point for the CLI."""
    try:
        cli()
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
