"""Memory management commands."""

from __future__ import annotations

import click
import json as json_lib
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..main import Context


def get_context(ctx: click.Context) -> "Context":
    """Get the CLI context object."""
    return ctx.obj


@click.group()
def memory() -> None:
    """Memory management commands.
    
    Create, read, update, and delete memories.
    
    \b
    Examples:
        amt memory add "User prefers dark mode"
        amt memory get abc123
        amt memory update abc123 --content "Updated content"
        amt memory delete abc123
    """
    pass


@click.command("add")
@click.argument("content")
@click.option("--tags", "-t", multiple=True, help="Tags for the memory")
@click.option("--source", "-s", default=None, help="Source of the memory")
@click.option("--confidence", "-c", type=float, default=1.0, 
              help="Confidence score (0.0-1.0)")
@click.pass_context
def add(ctx: click.Context, content: str, tags: tuple, source: Optional[str], 
        confidence: float) -> None:
    """Add a new memory.
    
    \b
    Examples:
        amt add "User prefers dark mode"
        amt add "API key is xyz" --tags secret --tags config
        amt add "Meeting notes..." --source meeting --confidence 0.9
    """
    context = get_context(ctx)
    store = context.store
    
    # Build metadata
    metadata = {}
    if tags:
        metadata["tags"] = list(tags)
    if source:
        metadata["source"] = source
    if confidence != 1.0:
        metadata["confidence"] = confidence
    
    memory = store.add(content, metadata=metadata if metadata else None)
    
    if context.json_output:
        click.echo(json_lib.dumps({
            "id": memory.id,
            "content": memory.content,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
        }, indent=2))
    else:
        click.echo(f"Added memory: {memory.id}")
        context.echo_verbose(f"  Content: {content[:60]}...")


@click.command("get")
@click.argument("memory_id")
@click.pass_context
def get(ctx: click.Context, memory_id: str) -> None:
    """Get a memory by ID.
    
    \b
    Examples:
        amt memory get abc123
        amt get abc123def456  # Full ID
    """
    context = get_context(ctx)
    store = context.store
    
    try:
        memory = store.get(memory_id)
    except Exception as e:
        raise click.ClickException(f"Memory not found: {memory_id}")
    
    if context.json_output:
        data = {
            "id": memory.id,
            "content": memory.content,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            "metadata": {
                "tags": list(memory.metadata.tags) if memory.metadata and memory.metadata.tags else [],
                "source": memory.metadata.source if memory.metadata else None,
                "confidence": memory.metadata.confidence if memory.metadata else None,
            } if memory.metadata else None,
        }
        click.echo(json_lib.dumps(data, indent=2))
    else:
        click.echo(f"ID: {memory.id}")
        click.echo(f"Content: {memory.content}")
        if memory.created_at:
            click.echo(f"Created: {memory.created_at}")
        if memory.updated_at:
            click.echo(f"Updated: {memory.updated_at}")
        if memory.metadata:
            if memory.metadata.tags:
                click.echo(f"Tags: {', '.join(memory.metadata.tags)}")
            if memory.metadata.source:
                click.echo(f"Source: {memory.metadata.source}")
            if memory.metadata.confidence:
                click.echo(f"Confidence: {memory.metadata.confidence}")


@click.command("update")
@click.argument("memory_id")
@click.option("--content", "-c", default=None, help="New content")
@click.option("--tags", "-t", multiple=True, help="New tags (replaces existing)")
@click.option("--add-tag", multiple=True, help="Add tags without replacing")
@click.pass_context
def update(ctx: click.Context, memory_id: str, content: Optional[str], 
           tags: tuple, add_tag: tuple) -> None:
    """Update an existing memory.
    
    \b
    Examples:
        amt memory update abc123 --content "New content"
        amt memory update abc123 --tags new-tag
        amt memory update abc123 --add-tag extra-tag
    """
    context = get_context(ctx)
    store = context.store
    
    try:
        memory = store.get(memory_id)
    except Exception:
        raise click.ClickException(f"Memory not found: {memory_id}")
    
    # Determine new content
    new_content = content if content else memory.content
    
    # Determine new tags
    new_tags = None
    if tags:
        new_tags = list(tags)
    elif add_tag:
        existing_tags = list(memory.metadata.tags) if memory.metadata and memory.metadata.tags else []
        new_tags = existing_tags + list(add_tag)
    
    # Build metadata update
    metadata = {}
    if new_tags is not None:
        metadata["tags"] = new_tags
    
    updated = store.update(memory_id, new_content, metadata=metadata if metadata else None)
    
    if context.json_output:
        click.echo(json_lib.dumps({
            "id": updated.id,
            "content": updated.content,
            "updated_at": updated.updated_at.isoformat() if updated.updated_at else None,
        }, indent=2))
    else:
        click.echo(f"Updated memory: {updated.id}")


@click.command("delete")
@click.argument("memory_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx: click.Context, memory_id: str, force: bool) -> None:
    """Delete a memory.
    
    \b
    Examples:
        amt memory delete abc123
        amt memory delete abc123 --force
    """
    context = get_context(ctx)
    store = context.store
    
    try:
        memory = store.get(memory_id)
    except Exception:
        raise click.ClickException(f"Memory not found: {memory_id}")
    
    if not force:
        click.echo(f"Memory to delete:")
        click.echo(f"  ID: {memory.id}")
        click.echo(f"  Content: {memory.content[:60]}...")
        if not click.confirm("Delete this memory?"):
            click.echo("Cancelled.")
            return
    
    store.delete(memory_id)
    
    if context.json_output:
        click.echo(json_lib.dumps({"deleted": memory_id}))
    else:
        click.echo(f"Deleted memory: {memory_id}")


# Register subcommands to the group
memory.add_command(add)
memory.add_command(get)
memory.add_command(update)
memory.add_command(delete)
