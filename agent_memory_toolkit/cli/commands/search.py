"""Search commands."""

from __future__ import annotations

import click
import json as json_lib
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..main import Context


def get_context(ctx: click.Context) -> "Context":
    """Get the CLI context object."""
    return ctx.obj


@click.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Maximum results to return")
@click.option("--mode", "-m", type=click.Choice(["hybrid", "fts", "vector"]),
              default="hybrid", help="Search mode (default: hybrid)")
@click.option("--threshold", "-t", type=float, default=0.0,
              help="Minimum score threshold")
@click.option("--tag", multiple=True, help="Filter by tag")
@click.option("--source", "-s", default=None, help="Filter by source")
@click.option("--rerank", "-r", is_flag=True, help="Enable cross-encoder reranking")
@click.option("--branch", "-b", default=None, help="Search on specific branch")
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, mode: str, 
           threshold: float, tag: tuple, source: Optional[str],
           rerank: bool, branch: Optional[str]) -> None:
    """Search memories.
    
    Search modes:
    
    \b
        hybrid: Combines BM25 keyword + vector semantic search (recommended)
        fts:    Fast full-text search with BM25 ranking
        vector: Semantic similarity search using embeddings
    
    \b
    Examples:
        amt search "user preferences"
        amt search "API configuration" --mode fts
        amt search "similar concepts" --mode vector --limit 5
        amt search "meetings" --tag work --tag important
        amt search "deployment" --rerank  # Better accuracy
    """
    context = get_context(ctx)
    store = context.store
    
    if branch:
        store.checkout(branch)
    
    context.echo_verbose(f"Searching: {query}")
    context.echo_verbose(f"Mode: {mode}, Limit: {limit}")
    
    # Build filters
    filters = {}
    if tag:
        filters["tags"] = list(tag)
    if source:
        filters["source"] = source
    
    # Execute search based on mode
    if mode == "fts":
        results = store.search_fts(query, limit=limit)
    elif mode == "vector":
        results = store.search_vector(query, limit=limit)
    else:  # hybrid
        results = store.search(query, limit=limit, rerank=rerank)
    
    # Apply threshold filter
    if threshold > 0:
        results = [r for r in results if r.score >= threshold]
    
    # Apply tag/source filters (post-search if store doesn't support it)
    if filters.get("tags"):
        filter_tags = set(filters["tags"])
        filtered = []
        for r in results:
            if r.memory.metadata and r.memory.metadata.tags:
                if filter_tags.intersection(r.memory.metadata.tags):
                    filtered.append(r)
        results = filtered
    
    if filters.get("source"):
        results = [r for r in results 
                   if r.memory.metadata and r.memory.metadata.source == filters["source"]]
    
    if context.json_output:
        data = [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "score": round(r.score, 4),
                "tags": list(r.memory.metadata.tags) if r.memory.metadata and r.memory.metadata.tags else [],
            }
            for r in results
        ]
        click.echo(json_lib.dumps(data, indent=2))
    else:
        if not results:
            click.echo("No results found.")
            return
        
        click.echo(f"Results ({len(results)} found):")
        click.echo("-" * 60)
        for r in results:
            tags = ""
            if r.memory.metadata and r.memory.metadata.tags:
                tags = f" [{', '.join(r.memory.metadata.tags)}]"
            content_preview = r.memory.content[:70] + "..." if len(r.memory.content) > 70 else r.memory.content
            score_str = f"{r.score:.3f}"
            click.echo(f"  [{score_str}] {r.memory.id[:8]}  {content_preview}{tags}")
