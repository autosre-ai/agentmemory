"""Info and stats commands."""

from __future__ import annotations

import click
import json as json_lib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..main import Context


def get_context(ctx: click.Context) -> "Context":
    """Get the CLI context object."""
    return ctx.obj


@click.command("info")
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show toolkit information and capabilities.
    
    Displays version, available features, and configuration.
    """
    context = get_context(ctx)
    
    from agent_memory_toolkit import __version__
    
    features = {}
    
    # Check core features
    try:
        from agent_memory_toolkit import MemoryStore
        features["Memory Store"] = True
    except ImportError:
        features["Memory Store"] = False
    
    try:
        from agent_memory_toolkit import MemoryExtractor
        features["Memory Extraction"] = True
    except ImportError:
        features["Memory Extraction"] = False
    
    try:
        from agent_memory_toolkit import MemoryGuard
        features["Security Guard"] = True
    except ImportError:
        features["Security Guard"] = False
    
    try:
        from agent_memory_toolkit import ContextCompressor
        features["Context Compression"] = True
    except ImportError:
        features["Context Compression"] = False
    
    try:
        from agent_memory_toolkit.store import SENTENCE_TRANSFORMERS_AVAILABLE
        features["Vector Search"] = SENTENCE_TRANSFORMERS_AVAILABLE
    except ImportError:
        features["Vector Search"] = False
    
    try:
        from agent_memory_toolkit.store import CROSS_ENCODER_AVAILABLE
        features["Cross-Encoder Reranking"] = CROSS_ENCODER_AVAILABLE
    except ImportError:
        features["Cross-Encoder Reranking"] = False
    
    try:
        import mcp
        features["MCP Protocol"] = True
    except ImportError:
        features["MCP Protocol"] = False
    
    if context.json_output:
        data = {
            "name": "Agent Memory Toolkit",
            "version": __version__,
            "db_path": str(context.db_path),
            "features": features,
        }
        click.echo(json_lib.dumps(data, indent=2))
    else:
        click.echo("Agent Memory Toolkit")
        click.echo("=" * 40)
        click.echo(f"Version:  {__version__}")
        click.echo(f"Database: {context.db_path}")
        click.echo()
        click.echo("Features:")
        for feature, available in features.items():
            status = "✓" if available else "✗"
            click.echo(f"  {status} {feature}")
        click.echo()
        click.echo("CLI Commands:")
        click.echo("  amt add         Add a memory")
        click.echo("  amt search      Search memories")
        click.echo("  amt list        List memories")
        click.echo("  amt memory      Memory CRUD operations")
        click.echo("  amt store       Branch/commit management")
        click.echo("  amt export      Export to file")
        click.echo("  amt import      Import from file")


@click.command("stats")
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show memory store statistics.
    
    Displays counts, storage info, and distribution.
    """
    context = get_context(ctx)
    store = context.store
    
    # Gather statistics
    total_memories = store.count()
    branches = store.list_branches()
    current_branch = store.current_branch
    
    # Count by tags (sample-based for large stores)
    tag_counts = {}
    source_counts = {}
    sample = list(store.list(limit=1000))
    
    for m in sample:
        if m.metadata and m.metadata.tags:
            for tag in m.metadata.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if m.metadata and m.metadata.source:
            source_counts[m.metadata.source] = source_counts.get(m.metadata.source, 0) + 1
    
    if context.json_output:
        data = {
            "total_memories": total_memories,
            "current_branch": current_branch,
            "branches": [b.name for b in branches],
            "branch_count": len(branches),
            "top_tags": dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10]),
            "top_sources": dict(sorted(source_counts.items(), key=lambda x: -x[1])[:10]),
        }
        click.echo(json_lib.dumps(data, indent=2))
    else:
        click.echo("Memory Store Statistics")
        click.echo("=" * 40)
        click.echo(f"Total Memories: {total_memories}")
        click.echo(f"Current Branch: {current_branch}")
        click.echo(f"Total Branches: {len(branches)}")
        
        if tag_counts:
            click.echo()
            click.echo("Top Tags:")
            for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:5]:
                click.echo(f"  {tag}: {count}")
        
        if source_counts:
            click.echo()
            click.echo("Top Sources:")
            for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:5]:
                click.echo(f"  {source}: {count}")
