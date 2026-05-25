"""Export and import commands."""

from __future__ import annotations

import click
import json as json_lib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..main import Context


def get_context(ctx: click.Context) -> "Context":
    """Get the CLI context object."""
    return ctx.obj


@click.command("export")
@click.argument("output", type=click.Path())
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "jsonl", "csv"]),
              default="json", help="Output format (default: json)")
@click.option("--branch", "-b", default=None, help="Branch to export")
@click.option("--query", "-q", default=None, help="Only export matching memories")
@click.option("--include-metadata", "-m", is_flag=True, default=True,
              help="Include metadata in export")
@click.pass_context
def export_cmd(ctx: click.Context, output: str, fmt: str, branch: str,
               query: str, include_metadata: bool) -> None:
    """Export memories to a file.
    
    \b
    Formats:
        json:   Single JSON file with array of memories
        jsonl:  JSON Lines (one JSON object per line)
        csv:    CSV with id, content, tags columns
    
    \b
    Examples:
        amt export memories.json
        amt export data.jsonl --format jsonl
        amt export backup.json --branch experiment
        amt export filtered.json --query "preferences"
    """
    context = get_context(ctx)
    store = context.store
    
    if branch:
        store.checkout(branch)
    
    # Get memories
    if query:
        results = store.search(query, limit=10000)
        memories = [r.memory for r in results]
    else:
        memories = list(store.list(limit=10000))
    
    output_path = Path(output).expanduser()
    
    if fmt == "json":
        data = []
        for m in memories:
            entry = {
                "id": m.id,
                "content": m.content,
            }
            if include_metadata:
                entry["created_at"] = m.created_at.isoformat() if m.created_at else None
                entry["updated_at"] = m.updated_at.isoformat() if m.updated_at else None
                if m.metadata:
                    entry["metadata"] = {
                        "tags": list(m.metadata.tags) if m.metadata.tags else [],
                        "source": m.metadata.source,
                        "confidence": m.metadata.confidence,
                    }
            data.append(entry)
        
        with open(output_path, "w") as f:
            json_lib.dump(data, f, indent=2)
    
    elif fmt == "jsonl":
        with open(output_path, "w") as f:
            for m in memories:
                entry = {
                    "id": m.id,
                    "content": m.content,
                }
                if include_metadata and m.metadata:
                    entry["tags"] = list(m.metadata.tags) if m.metadata.tags else []
                    entry["source"] = m.metadata.source
                f.write(json_lib.dumps(entry) + "\n")
    
    elif fmt == "csv":
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "content", "tags", "source"])
            for m in memories:
                tags = ""
                source = ""
                if m.metadata:
                    tags = ";".join(m.metadata.tags) if m.metadata.tags else ""
                    source = m.metadata.source or ""
                writer.writerow([m.id, m.content, tags, source])
    
    click.echo(f"Exported {len(memories)} memories to {output_path}")


@click.command("import")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "jsonl", "csv"]),
              default=None, help="Input format (auto-detected if not specified)")
@click.option("--merge", is_flag=True, help="Merge with existing (skip duplicates)")
@click.option("--dry-run", is_flag=True, help="Show what would be imported")
@click.pass_context
def import_cmd(ctx: click.Context, input_file: str, fmt: str, merge: bool,
               dry_run: bool) -> None:
    """Import memories from a file.
    
    \b
    Examples:
        amt import memories.json
        amt import data.jsonl --format jsonl
        amt import backup.json --merge
        amt import data.csv --dry-run
    """
    context = get_context(ctx)
    store = context.store
    
    input_path = Path(input_file).expanduser()
    
    # Auto-detect format
    if fmt is None:
        suffix = input_path.suffix.lower()
        if suffix == ".json":
            fmt = "json"
        elif suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".csv":
            fmt = "csv"
        else:
            raise click.ClickException(f"Cannot auto-detect format for {suffix}. Use --format")
    
    memories_to_import = []
    
    if fmt == "json":
        with open(input_path) as f:
            data = json_lib.load(f)
        
        if isinstance(data, list):
            memories_to_import = data
        else:
            raise click.ClickException("JSON file must contain an array of memories")
    
    elif fmt == "jsonl":
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    memories_to_import.append(json_lib.loads(line))
    
    elif fmt == "csv":
        import csv
        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
                    "content": row.get("content", ""),
                    "metadata": {}
                }
                if row.get("tags"):
                    entry["metadata"]["tags"] = row["tags"].split(";")
                if row.get("source"):
                    entry["metadata"]["source"] = row["source"]
                memories_to_import.append(entry)
    
    if dry_run:
        click.echo(f"Would import {len(memories_to_import)} memories:")
        for m in memories_to_import[:10]:
            content = m.get("content", "")[:60]
            click.echo(f"  - {content}...")
        if len(memories_to_import) > 10:
            click.echo(f"  ... and {len(memories_to_import) - 10} more")
        return
    
    imported = 0
    skipped = 0
    
    for m in memories_to_import:
        content = m.get("content", "")
        if not content:
            skipped += 1
            continue
        
        metadata = m.get("metadata", {})
        if not metadata and m.get("tags"):
            metadata["tags"] = m.get("tags")
        if not metadata.get("source") and m.get("source"):
            metadata["source"] = m.get("source")
        
        # Check for duplicates if merge mode
        if merge:
            results = store.search_fts(content[:100], limit=1)
            if results and results[0].memory.content == content:
                skipped += 1
                continue
        
        store.add(content, metadata=metadata if metadata else None)
        imported += 1
    
    click.echo(f"Imported: {imported} memories")
    if skipped > 0:
        click.echo(f"Skipped: {skipped} (empty or duplicate)")
