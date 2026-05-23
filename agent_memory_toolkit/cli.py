"""
Agent Memory Toolkit CLI

Command-line interface for the agent-memory-toolkit.
Provides commands for memory management, extraction, team sync,
security validation, and compression.

Usage:
    amt memory add "Your memory content here"
    amt memory query "search terms"
    amt extract text "My name is John and I work at Google"
    amt team init ./shared-memories
    amt guard check "Some memory content"
    amt compress conversation messages.json --max-tokens 4000
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import click


# Lazy imports for faster startup
def get_memory_store():
    """Lazy import MemoryStore."""
    from agent_memory_toolkit.store import MemoryStore
    return MemoryStore


def get_team_memory_store():
    """Lazy import TeamMemoryStore."""
    from agent_memory_toolkit.team import TeamMemoryStore
    return TeamMemoryStore


def get_memory_extractor():
    """Lazy import MemoryExtractor."""
    from agent_memory_toolkit.extraction import MemoryExtractor
    return MemoryExtractor


def get_memory_guard():
    """Lazy import MemoryGuard."""
    from agent_memory_toolkit.security import MemoryGuard, SecurityLevel
    return MemoryGuard, SecurityLevel


def get_context_compressor():
    """Lazy import ContextCompressor."""
    from agent_memory_toolkit.compression import ContextCompressor, CompressionConfig, Message
    return ContextCompressor, CompressionConfig, Message


def get_audit_logger():
    """Lazy import AuditLogger."""
    from agent_memory_toolkit.security import AuditLogger
    return AuditLogger


@click.group()
@click.version_option(version="0.1.0", prog_name="amt")
def cli():
    """Agent Memory Toolkit (AMT) - Local-first memory layer for AI agents.
    
    A comprehensive toolkit for managing AI agent memory with SQLite + FTS5,
    structured extraction, team collaboration, security validation, and
    intelligent context compression.
    """
    pass


# ==============================================================================
# Memory Commands
# ==============================================================================

@cli.group()
@click.option("--db", "-d", default="agent_memory.db", help="Path to SQLite database")
@click.pass_context
def memory(ctx, db: str):
    """Memory store operations (add, query, history, branch)."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


@memory.command("add")
@click.argument("content")
@click.option("--source", "-s", help="Source of the memory")
@click.option("--tags", "-t", multiple=True, help="Tags for the memory")
@click.option("--confidence", "-c", type=float, default=1.0, help="Confidence score (0-1)")
@click.pass_context
def memory_add(ctx, content: str, source: Optional[str], tags: tuple, confidence: float):
    """Add a new memory to the store."""
    MemoryStore = get_memory_store()
    
    db_path = ctx.obj["db"]
    metadata = {
        "source": source or "cli",
        "tags": list(tags) if tags else [],
        "confidence": confidence,
    }
    
    with MemoryStore(db_path) as store:
        memory = store.add(content, metadata=metadata)
        click.echo(f"✓ Added memory: {memory.id}")
        click.echo(f"  Content: {content[:50]}{'...' if len(content) > 50 else ''}")
        if tags:
            click.echo(f"  Tags: {', '.join(tags)}")


@memory.command("query")
@click.argument("search_query")
@click.option("--limit", "-l", default=10, help="Maximum results to return")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def memory_query(ctx, search_query: str, limit: int, json_output: bool):
    """Search memories using full-text search."""
    MemoryStore = get_memory_store()
    
    db_path = ctx.obj["db"]
    
    with MemoryStore(db_path) as store:
        results = store.search_fts(search_query, limit=limit)
        
        if json_output:
            output = [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "score": r.score,
                    "metadata": r.memory.metadata.to_dict() if hasattr(r.memory.metadata, 'to_dict') else {},
                }
                for r in results
            ]
            click.echo(json.dumps(output, indent=2))
        else:
            if not results:
                click.echo("No memories found.")
                return
            
            click.echo(f"Found {len(results)} memories:\n")
            for i, r in enumerate(results, 1):
                click.echo(f"{i}. [{r.score:.2f}] {r.memory.content[:80]}{'...' if len(r.memory.content) > 80 else ''}")
                click.echo(f"   ID: {r.memory.id}")
                click.echo()


@memory.command("list")
@click.option("--limit", "-l", default=20, help="Maximum memories to list")
@click.option("--offset", "-o", default=0, help="Offset for pagination")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def memory_list(ctx, limit: int, offset: int, tag: Optional[str], json_output: bool):
    """List memories with optional filtering."""
    MemoryStore = get_memory_store()
    
    db_path = ctx.obj["db"]
    
    with MemoryStore(db_path) as store:
        memories = store.list(limit=limit, offset=offset, tag=tag)
        
        if json_output:
            output = [
                {
                    "id": m.id,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                    "version": m.version,
                }
                for m in memories
            ]
            click.echo(json.dumps(output, indent=2))
        else:
            if not memories:
                click.echo("No memories found.")
                return
            
            total = store.count()
            click.echo(f"Showing {len(memories)} of {total} memories:\n")
            for m in memories:
                click.echo(f"• {m.content[:60]}{'...' if len(m.content) > 60 else ''}")
                click.echo(f"  ID: {m.id} | v{m.version} | {m.created_at.strftime('%Y-%m-%d %H:%M')}")
                click.echo()


@memory.command("history")
@click.option("--memory-id", "-m", help="Show history for specific memory")
@click.option("--limit", "-l", default=20, help="Maximum commits to show")
@click.pass_context
def memory_history(ctx, memory_id: Optional[str], limit: int):
    """Show version history for memories or commits."""
    MemoryStore = get_memory_store()
    
    db_path = ctx.obj["db"]
    
    with MemoryStore(db_path) as store:
        if memory_id:
            # Show version history for specific memory
            try:
                history = store.get_memory_history(memory_id)
                click.echo(f"Version history for memory {memory_id}:\n")
                for entry in history:
                    click.echo(f"  v{entry['version']} [{entry['operation']}] {entry['created_at']}")
                    click.echo(f"    {entry['content'][:60]}...")
                    click.echo()
            except Exception as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)
        else:
            # Show commit history
            commits = store.get_history(limit=limit)
            if not commits:
                click.echo("No commits found.")
                return
            
            click.echo(f"Commit history for branch '{store.current_branch}':\n")
            for commit in commits:
                click.echo(f"commit {commit.id[:8]}")
                click.echo(f"Date:   {commit.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                click.echo(f"        {commit.message}")
                click.echo()


@memory.command("branch")
@click.argument("action", type=click.Choice(["list", "create", "checkout", "delete"]))
@click.argument("name", required=False)
@click.option("--from", "from_branch", help="Source branch for create")
@click.pass_context
def memory_branch(ctx, action: str, name: Optional[str], from_branch: Optional[str]):
    """Manage branches (list, create, checkout, delete)."""
    MemoryStore = get_memory_store()
    
    db_path = ctx.obj["db"]
    
    with MemoryStore(db_path) as store:
        if action == "list":
            branches = store.list_branches()
            current = store.current_branch
            click.echo("Branches:")
            for b in branches:
                marker = "*" if b.name == current else " "
                click.echo(f"  {marker} {b.name} (created: {b.created_at.strftime('%Y-%m-%d')})")
        
        elif action == "create":
            if not name:
                click.echo("Error: Branch name required for create.", err=True)
                sys.exit(1)
            branch = store.create_branch(name, from_branch=from_branch)
            click.echo(f"✓ Created branch '{branch.name}'")
        
        elif action == "checkout":
            if not name:
                click.echo("Error: Branch name required for checkout.", err=True)
                sys.exit(1)
            store.checkout(name)
            click.echo(f"✓ Switched to branch '{name}'")
        
        elif action == "delete":
            if not name:
                click.echo("Error: Branch name required for delete.", err=True)
                sys.exit(1)
            store.delete_branch(name)
            click.echo(f"✓ Deleted branch '{name}'")


# ==============================================================================
# Extract Commands
# ==============================================================================

@cli.group()
def extract():
    """Extract structured memories from text or conversations."""
    pass


@extract.command("text")
@click.argument("text")
@click.option("--mode", "-m", default="rule", type=click.Choice(["rule", "llm", "hybrid"]),
              help="Extraction mode")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def extract_text(text: str, mode: str, json_output: bool):
    """Extract memories from plain text."""
    MemoryExtractor = get_memory_extractor()
    
    extractor = MemoryExtractor(mode=mode)
    result = extractor.extract(text)
    
    if json_output:
        output = {
            "memories": [
                {
                    "domain": m.domain.value,
                    "key": m.key,
                    "value": m.value,
                    "confidence": m.confidence,
                }
                for m in result.memories
            ],
            "method": result.method,
            "processing_time_ms": result.processing_time_ms,
        }
        click.echo(json.dumps(output, indent=2))
    else:
        if not result.memories:
            click.echo("No memories extracted.")
            return
        
        click.echo(f"Extracted {len(result.memories)} memories:\n")
        for m in result.memories:
            click.echo(f"  [{m.domain.value}] {m.key} = {m.value}")
            click.echo(f"      confidence: {m.confidence:.2f}")
        
        click.echo(f"\nExtraction time: {result.processing_time_ms:.1f}ms")


@extract.command("conversation")
@click.argument("file", type=click.Path(exists=True))
@click.option("--mode", "-m", default="rule", type=click.Choice(["rule", "llm", "hybrid"]),
              help="Extraction mode")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def extract_conversation(file: str, mode: str, json_output: bool):
    """Extract memories from a conversation JSON file.
    
    The file should contain a list of messages with 'role' and 'content' fields.
    """
    MemoryExtractor = get_memory_extractor()
    
    with open(file, "r") as f:
        messages = json.load(f)
    
    # Combine messages into text for extraction
    conversation_text = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in messages
    )
    
    extractor = MemoryExtractor(mode=mode)
    result = extractor.extract(conversation_text, source=f"conversation:{file}")
    
    if json_output:
        output = {
            "memories": [
                {
                    "domain": m.domain.value,
                    "key": m.key,
                    "value": m.value,
                    "confidence": m.confidence,
                }
                for m in result.memories
            ],
            "method": result.method,
            "processing_time_ms": result.processing_time_ms,
            "message_count": len(messages),
        }
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo(f"Processed {len(messages)} messages")
        click.echo(f"Extracted {len(result.memories)} memories:\n")
        
        for m in result.memories:
            click.echo(f"  [{m.domain.value}] {m.key} = {m.value}")


# ==============================================================================
# Team Commands
# ==============================================================================

@cli.group()
@click.option("--db", "-d", default="team_memory.db", help="Path to team database")
@click.option("--agent-id", "-a", help="Agent identifier")
@click.pass_context
def team(ctx, db: str, agent_id: Optional[str]):
    """Team memory collaboration (init, sync, merge)."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db
    ctx.obj["agent_id"] = agent_id


@team.command("init")
@click.argument("sync_path", type=click.Path())
@click.option("--namespace", "-n", default="default", help="Default namespace")
@click.pass_context
def team_init(ctx, sync_path: str, namespace: str):
    """Initialize team memory store with sync location."""
    TeamMemoryStore = get_team_memory_store()
    
    db_path = ctx.obj["db"]
    agent_id = ctx.obj.get("agent_id")
    
    # Create sync directory if it doesn't exist
    sync_dir = Path(sync_path)
    sync_dir.mkdir(parents=True, exist_ok=True)
    
    store = TeamMemoryStore(
        db_path=db_path,
        agent_id=agent_id,
        default_namespace=namespace,
    )
    
    click.echo(f"✓ Initialized team memory store")
    click.echo(f"  Database: {db_path}")
    click.echo(f"  Agent ID: {store.agent_id}")
    click.echo(f"  Sync path: {sync_path}")
    click.echo(f"  Namespace: {namespace}")


@team.command("sync")
@click.argument("sync_path", type=click.Path(exists=True))
@click.option("--push-only", is_flag=True, help="Only push changes")
@click.option("--pull-only", is_flag=True, help="Only pull changes")
@click.pass_context
def team_sync(ctx, sync_path: str, push_only: bool, pull_only: bool):
    """Sync memories with shared location."""
    TeamMemoryStore = get_team_memory_store()
    
    db_path = ctx.obj["db"]
    agent_id = ctx.obj.get("agent_id")
    
    with TeamMemoryStore(db_path, agent_id=agent_id) as store:
        result = store.sync(sync_path)
        
        click.echo("✓ Sync complete")
        click.echo(f"  Pushed: {result.memories_pushed} memories")
        click.echo(f"  Pulled: {result.memories_pulled} memories")
        
        if result.conflicts:
            click.echo(f"  Conflicts: {len(result.conflicts)}")
            for conflict in result.conflicts[:5]:  # Show first 5
                click.echo(f"    - {conflict.memory_id}: {conflict.conflict_type}")


@team.command("merge")
@click.argument("source_branch")
@click.option("--strategy", "-s", default="latest_wins",
              type=click.Choice(["latest_wins", "ours", "theirs", "manual"]),
              help="Conflict resolution strategy")
@click.pass_context
def team_merge(ctx, source_branch: str, strategy: str):
    """Merge a branch into current branch."""
    TeamMemoryStore = get_team_memory_store()
    from agent_memory_toolkit.team import ConflictResolution
    
    db_path = ctx.obj["db"]
    agent_id = ctx.obj.get("agent_id")
    
    strategy_map = {
        "latest_wins": ConflictResolution.LATEST_WINS,
        "ours": ConflictResolution.OURS,
        "theirs": ConflictResolution.THEIRS,
        "manual": ConflictResolution.MANUAL,
    }
    
    with TeamMemoryStore(db_path, agent_id=agent_id) as store:
        conflicts = store.merge(source_branch, conflict_strategy=strategy_map[strategy])
        
        click.echo(f"✓ Merged '{source_branch}' into '{store.current_branch}'")
        
        if conflicts:
            click.echo(f"  Unresolved conflicts: {len(conflicts)}")
            for conflict in conflicts[:5]:
                click.echo(f"    - {conflict.memory_id}: {conflict.conflict_type}")
        else:
            click.echo("  No conflicts")


@team.command("status")
@click.pass_context
def team_status(ctx):
    """Show team memory status."""
    TeamMemoryStore = get_team_memory_store()
    
    db_path = ctx.obj["db"]
    agent_id = ctx.obj.get("agent_id")
    
    try:
        with TeamMemoryStore(db_path, agent_id=agent_id) as store:
            click.echo("Team Memory Status")
            click.echo("==================")
            click.echo(f"Agent ID:       {store.agent_id}")
            click.echo(f"Current branch: {store.current_branch}")
            click.echo(f"Database:       {db_path}")
    except Exception as e:
        click.echo(f"Error: Could not open team memory store: {e}", err=True)
        sys.exit(1)


# ==============================================================================
# Guard (Security) Commands
# ==============================================================================

@cli.group()
def guard():
    """Security validation and auditing."""
    pass


@guard.command("check")
@click.argument("content")
@click.option("--level", "-l", default="medium",
              type=click.Choice(["minimal", "low", "medium", "high", "paranoid"]),
              help="Security level")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def guard_check(content: str, level: str, json_output: bool):
    """Check content for security issues."""
    MemoryGuard, SecurityLevel = get_memory_guard()
    
    level_map = {
        "minimal": SecurityLevel.MINIMAL,
        "low": SecurityLevel.LOW,
        "medium": SecurityLevel.MEDIUM,
        "high": SecurityLevel.HIGH,
        "paranoid": SecurityLevel.PARANOID,
    }
    
    guard = MemoryGuard(level=level_map[level])
    result = guard.validate_content(content)
    
    if json_output:
        output = {
            "is_safe": result.is_safe,
            "rejection_reason": result.rejection_reason,
            "adjusted_confidence": result.adjusted_confidence,
            "validation_time_ms": result.validation_time_ms,
        }
        if result.poison_result:
            output["poison_detection"] = {
                "is_safe": result.poison_result.is_safe,
                "risk_score": result.poison_result.risk_score,
                "detected_patterns": [p.value for p in result.poison_result.detected_patterns],
            }
        click.echo(json.dumps(output, indent=2))
    else:
        if result.is_safe:
            click.echo("✓ Content passed security checks")
            click.echo(f"  Confidence: {result.adjusted_confidence:.2f}")
        else:
            click.echo("✗ Content FAILED security checks", err=True)
            click.echo(f"  Reason: {result.rejection_reason}")
            
        if result.poison_result:
            if result.poison_result.detected_patterns:
                click.echo(f"  Patterns detected: {len(result.poison_result.detected_patterns)}")
                for p in result.poison_result.detected_patterns[:3]:
                    click.echo(f"    - {p.value}")
        
        click.echo(f"\nValidation time: {result.validation_time_ms:.1f}ms")


@guard.command("audit")
@click.option("--db", "-d", default="audit.db", help="Audit log database")
@click.option("--limit", "-l", default=50, help="Maximum events to show")
@click.option("--type", "event_type", help="Filter by event type")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def guard_audit(db: str, limit: int, event_type: Optional[str], json_output: bool):
    """View audit log events."""
    AuditLogger = get_audit_logger()
    
    # This assumes the audit logger has a way to query past events
    # For now, we'll show a placeholder since the actual implementation depends on storage
    click.echo(f"Audit log from: {db}")
    click.echo(f"Showing last {limit} events")
    if event_type:
        click.echo(f"Filtered by type: {event_type}")
    
    click.echo("\nNote: Full audit querying requires an active audit logger with storage backend.")


# ==============================================================================
# Compress Commands
# ==============================================================================

@cli.group()
def compress():
    """Context compression utilities."""
    pass


@compress.command("conversation")
@click.argument("file", type=click.Path(exists=True))
@click.option("--max-tokens", "-t", default=4000, help="Maximum token budget")
@click.option("--reserve", "-r", default=500, help="Reserve tokens for response")
@click.option("--mode", "-m", default="balanced",
              type=click.Choice(["aggressive", "balanced", "conservative", "lossless"]),
              help="Compression mode")
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON with stats")
def compress_conversation(file: str, max_tokens: int, reserve: int, mode: str,
                         output: Optional[str], json_output: bool):
    """Compress a conversation to fit token budget.
    
    Input file should be JSON with a list of messages.
    """
    ContextCompressor, CompressionConfig, Message = get_context_compressor()
    from agent_memory_toolkit.compression import CompressionMode
    
    mode_map = {
        "aggressive": CompressionMode.AGGRESSIVE,
        "balanced": CompressionMode.BALANCED,
        "conservative": CompressionMode.CONSERVATIVE,
        "lossless": CompressionMode.LOSSLESS,
    }
    
    with open(file, "r") as f:
        messages_data = json.load(f)
    
    config = CompressionConfig(
        max_tokens=max_tokens,
        reserve_tokens=reserve,
        mode=mode_map[mode],
    )
    
    compressor = ContextCompressor(config=config)
    result = compressor.compress(messages_data)
    
    # Messages in result are already dicts
    compressed_messages = result.messages
    
    if json_output:
        output_data = {
            "messages": compressed_messages,
            "stats": {
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "compression_ratio": result.compression_ratio,
                "strategy_used": result.strategy_used,
                "tokens_saved": result.tokens_saved,
            }
        }
        output_text = json.dumps(output_data, indent=2)
    else:
        output_text = json.dumps(compressed_messages, indent=2)
    
    if output:
        with open(output, "w") as f:
            f.write(output_text)
        click.echo(f"✓ Compressed conversation written to {output}")
        click.echo(f"  Original: {result.original_tokens} tokens")
        click.echo(f"  Compressed: {result.compressed_tokens} tokens")
        click.echo(f"  Ratio: {result.compression_ratio:.1%}")
    else:
        click.echo(output_text)


@compress.command("estimate")
@click.argument("file", type=click.Path(exists=True))
def compress_estimate(file: str):
    """Estimate token count for a conversation file."""
    from agent_memory_toolkit.compression import TokenCounter
    
    with open(file, "r") as f:
        messages_data = json.load(f)
    
    counter = TokenCounter()
    total_tokens = 0
    
    for msg in messages_data:
        content = msg.get("content", "")
        tokens = counter.count(content)
        total_tokens += tokens
    
    click.echo(f"File: {file}")
    click.echo(f"Messages: {len(messages_data)}")
    click.echo(f"Total tokens: {total_tokens}")
    click.echo(f"Avg tokens/message: {total_tokens / len(messages_data):.1f}" if messages_data else "N/A")


# ==============================================================================
# Info Commands
# ==============================================================================

@cli.command("info")
def info():
    """Show toolkit information and available modules."""
    import agent_memory_toolkit
    
    click.echo("Agent Memory Toolkit (AMT)")
    click.echo("=" * 40)
    click.echo(f"Version: {agent_memory.__version__}")
    click.echo()
    click.echo("Available modules:")
    click.echo("  • store       - SQLite + FTS5 memory storage")
    click.echo("  • extraction  - Structured memory extraction")
    click.echo("  • team        - Git-like team collaboration")
    click.echo("  • security    - Memory security & validation")
    click.echo("  • compression - Context compression engine")
    click.echo("  • mcp         - Model Context Protocol server")
    click.echo("  • api         - REST API server")
    click.echo()
    click.echo("Run 'amt <command> --help' for more information.")


# ==============================================================================
# MCP Commands
# ==============================================================================

@cli.group()
def mcp():
    """MCP (Model Context Protocol) server commands."""
    pass


def get_mcp_server():
    """Lazy import MCP server components."""
    from agent_memory_toolkit.mcp import create_mcp_server, MCPConfig
    return create_mcp_server, MCPConfig


@mcp.command("serve")
@click.option("--transport", "-t", default="stdio",
              type=click.Choice(["stdio", "sse"]),
              help="Transport type (default: stdio)")
@click.option("--db", "-d", default="agent_memory.db",
              help="Path to memory database")
@click.option("--host", default="127.0.0.1",
              help="Host for SSE server")
@click.option("--port", "-p", type=int, default=8765,
              help="Port for SSE server")
@click.option("--log-level", "-l", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
              help="Logging level")
@click.option("--security-level", "-s", default="medium",
              type=click.Choice(["minimal", "low", "medium", "high", "paranoid"]),
              help="Security level for content validation")
def mcp_serve(transport: str, db: str, host: str, port: int, log_level: str, security_level: str):
    """Start the MCP server.
    
    This exposes the memory toolkit as MCP tools that can be used by
    LLM clients like Claude Desktop, Cursor, or any MCP-compatible application.
    
    For stdio transport (default), the server communicates over stdin/stdout.
    For sse transport, it runs an HTTP server with Server-Sent Events.
    
    Example configurations for Claude Desktop (claude_desktop_config.json):
    
    \b
    {
      "mcpServers": {
        "agent-memory": {
          "command": "amt",
          "args": ["mcp", "serve", "--db", "~/agent_memory.db"]
        }
      }
    }
    """
    create_mcp_server, MCPConfig = get_mcp_server()
    
    config = MCPConfig(
        memory_db=db,
        host=host,
        port=port,
        log_level=log_level,
        security_level=security_level,
    )
    
    mcp_server = create_mcp_server(config)
    
    if transport == "stdio":
        click.echo(f"Starting MCP server (stdio transport)...", err=True)
        click.echo(f"Memory database: {db}", err=True)
        click.echo(f"Security level: {security_level}", err=True)
        mcp_server.run(transport="stdio")
    else:
        click.echo(f"Starting MCP server (SSE transport)...")
        click.echo(f"URL: http://{host}:{port}")
        click.echo(f"Memory database: {db}")
        click.echo(f"Security level: {security_level}")
        mcp_server.run()


@mcp.command("config")
@click.argument("target", type=click.Choice(["claude", "cursor"]))
@click.option("--db", "-d", default="agent_memory.db",
              help="Path to memory database")
@click.option("--output", "-o", type=click.Path(),
              help="Output file (default: stdout)")
def mcp_config(target: str, db: str, output: Optional[str]):
    """Generate MCP configuration for LLM clients.
    
    Outputs JSON configuration that can be added to your client's config file.
    
    For Claude Desktop, add to: ~/Library/Application Support/Claude/claude_desktop_config.json
    For Cursor, add to: ~/.cursor/mcp.json
    """
    import sys
    
    # Resolve database path
    db_path = Path(db).expanduser().resolve()
    
    if target == "claude":
        config = {
            "mcpServers": {
                "agent-memory-toolkit": {
                    "command": "amt",
                    "args": ["mcp", "serve", "--db", str(db_path)]
                }
            }
        }
    else:  # cursor
        config = {
            "mcpServers": {
                "agent-memory-toolkit": {
                    "command": "amt",
                    "args": ["mcp", "serve", "--db", str(db_path)],
                    "env": {}
                }
            }
        }
    
    config_json = json.dumps(config, indent=2)
    
    if output:
        with open(output, "w") as f:
            f.write(config_json)
        click.echo(f"✓ Configuration written to {output}")
    else:
        click.echo(config_json)
    
    click.echo(f"\nTo use with {target.title()}:", err=True)
    if target == "claude":
        click.echo("1. Open Claude Desktop settings", err=True)
        click.echo("2. Navigate to Developer → MCP Servers", err=True)
        click.echo("3. Add the configuration above", err=True)
    else:
        click.echo("1. Open Cursor settings (Cmd/Ctrl + ,)", err=True)
        click.echo("2. Search for 'MCP'", err=True)
        click.echo("3. Add the server configuration", err=True)


@mcp.command("tools")
def mcp_tools():
    """List available MCP tools.
    
    Shows all tools exposed by the MCP server.
    """
    click.echo("Agent Memory Toolkit MCP Tools")
    click.echo("=" * 40)
    click.echo()
    click.echo("Memory Operations:")
    click.echo("  • memory_add     - Add a new memory to the store")
    click.echo("  • memory_query   - Search memories using full-text search")
    click.echo("  • memory_get     - Get a specific memory by ID")
    click.echo("  • memory_update  - Update an existing memory")
    click.echo("  • memory_delete  - Delete a memory")
    click.echo("  • memory_list    - List memories with pagination")
    click.echo("  • memory_history - Get version history")
    click.echo()
    click.echo("Extraction:")
    click.echo("  • extract_memories - Extract structured facts from text")
    click.echo()
    click.echo("Security:")
    click.echo("  • guard_check    - Validate content for security issues")
    click.echo()
    click.echo("Compression:")
    click.echo("  • compress_context - Compress conversation to fit token budget")
    click.echo("  • count_tokens     - Count tokens in text")


# ==============================================================================
# API Commands
# ==============================================================================

@cli.group()
def api():
    """REST API server commands."""
    pass


def get_api_server():
    """Lazy import API server components."""
    from agent_memory_toolkit.api import run_server, APIConfig, set_config
    return run_server, APIConfig, set_config


@api.command("serve")
@click.option("--host", "-h", default="0.0.0.0",
              help="Host to bind to (default: 0.0.0.0)")
@click.option("--port", "-p", type=int, default=8000,
              help="Port to bind to (default: 8000)")
@click.option("--db", "-d", default="agent_memory.db",
              help="Path to memory database")
@click.option("--reload", is_flag=True,
              help="Enable auto-reload for development")
@click.option("--workers", "-w", type=int, default=1,
              help="Number of worker processes")
@click.option("--log-level", "-l", default="info",
              type=click.Choice(["debug", "info", "warning", "error", "critical"]),
              help="Logging level")
@click.option("--no-rate-limit", is_flag=True,
              help="Disable rate limiting")
def api_serve(host: str, port: int, db: str, reload: bool, workers: int,
              log_level: str, no_rate_limit: bool):
    """Start the REST API server.
    
    Exposes all memory toolkit operations as REST endpoints with JWT authentication,
    rate limiting, and OpenAPI documentation.
    
    Example:
    
    \\b
        # Start server on default port
        amt api serve
        
        # Start with custom settings
        amt api serve --port 9000 --db ~/my_memory.db
        
        # Development mode with auto-reload
        amt api serve --reload --log-level debug
    
    The server provides:
    
    \\b
        - OpenAPI docs: http://localhost:8000/docs
        - ReDoc: http://localhost:8000/redoc
        - Health check: http://localhost:8000/health
    
    Authentication:
    
    \\b
        1. Get a token: POST /api/v1/auth/token
        2. Use token: Authorization: Bearer <token>
    
    Default credentials (set via environment variables):
    
    \\b
        - admin / admin (AMT_ADMIN_PASSWORD)
        - agent / agent (AMT_AGENT_PASSWORD)
    """
    run_server, APIConfig, set_config = get_api_server()
    
    # Configure the API
    config = APIConfig(
        host=host,
        port=port,
        db_path=db,
        workers=workers,
        rate_limit_enabled=not no_rate_limit,
    )
    set_config(config)
    
    click.echo("Agent Memory Toolkit REST API")
    click.echo("=" * 40)
    click.echo(f"Host: {host}")
    click.echo(f"Port: {port}")
    click.echo(f"Database: {db}")
    click.echo(f"Workers: {workers}")
    click.echo(f"Rate limiting: {'disabled' if no_rate_limit else 'enabled'}")
    click.echo()
    click.echo(f"API docs: http://{host}:{port}/docs")
    click.echo(f"Health check: http://{host}:{port}/health")
    click.echo()
    
    run_server(
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
    )


@api.command("info")
def api_info():
    """Show API information and endpoints.
    
    Lists all available endpoints and authentication details.
    """
    click.echo("Agent Memory Toolkit REST API")
    click.echo("=" * 40)
    click.echo()
    click.echo("Endpoints:")
    click.echo()
    click.echo("  Authentication:")
    click.echo("    POST /api/v1/auth/token     - Get JWT access token")
    click.echo()
    click.echo("  Memories:")
    click.echo("    POST   /api/v1/memories           - Create memory")
    click.echo("    GET    /api/v1/memories           - List memories")
    click.echo("    GET    /api/v1/memories/search    - Search memories")
    click.echo("    GET    /api/v1/memories/{id}      - Get memory")
    click.echo("    PUT    /api/v1/memories/{id}      - Update memory")
    click.echo("    DELETE /api/v1/memories/{id}      - Delete memory")
    click.echo("    GET    /api/v1/memories/{id}/history - Get version history")
    click.echo()
    click.echo("  Branches:")
    click.echo("    GET    /api/v1/branches           - List branches")
    click.echo("    POST   /api/v1/branches           - Create branch")
    click.echo("    POST   /api/v1/branches/{name}/checkout - Checkout branch")
    click.echo("    DELETE /api/v1/branches/{name}    - Delete branch")
    click.echo("    GET    /api/v1/branches/{name}/commits - List commits")
    click.echo("    POST   /api/v1/branches/{name}/commits - Create commit")
    click.echo()
    click.echo("  Extraction:")
    click.echo("    POST /api/v1/extract/text         - Extract from text")
    click.echo("    POST /api/v1/extract/conversation - Extract from conversation")
    click.echo()
    click.echo("  Security:")
    click.echo("    POST /api/v1/security/check       - Check content security")
    click.echo("    GET  /api/v1/security/levels      - List security levels")
    click.echo()
    click.echo("  Compression:")
    click.echo("    POST /api/v1/compress/conversation - Compress conversation")
    click.echo("    POST /api/v1/compress/estimate     - Estimate tokens")
    click.echo("    GET  /api/v1/compress/modes        - List compression modes")
    click.echo()
    click.echo("  Health & Info:")
    click.echo("    GET /health   - Health check (no auth)")
    click.echo("    GET /info     - API information (no auth)")
    click.echo("    GET /docs     - OpenAPI documentation")
    click.echo("    GET /redoc    - ReDoc documentation")


# ==============================================================================
# Export/Import Commands
# ==============================================================================

def get_io_module():
    """Lazy import IO module."""
    from agent_memory_toolkit import io as io_module
    return io_module


@cli.command("export")
@click.argument("output", type=click.Path())
@click.option("--db", "-d", default="agent_memory.db", help="Path to memory database")
@click.option("--format", "-f", "fmt", 
              type=click.Choice(["jsonl", "csv", "parquet", "markdown", "sqlite"]),
              default="jsonl", help="Export format (default: jsonl)")
@click.option("--branch", "-b", multiple=True, help="Branches to export (default: all)")
@click.option("--include-deleted", is_flag=True, help="Include soft-deleted memories")
@click.option("--include-embeddings", is_flag=True, help="Include embedding vectors")
@click.option("--pretty", is_flag=True, help="Pretty-print output (JSONL, Markdown)")
def export_cmd(output: str, db: str, fmt: str, branch: tuple, include_deleted: bool,
               include_embeddings: bool, pretty: bool):
    """Export memories to file.
    
    Supports multiple formats:
    
    \b
        - jsonl:    JSON Lines format (one memory per line)
        - csv:      Comma-separated values
        - parquet:  Apache Parquet (columnar, compressed)
        - markdown: Human-readable Markdown
        - sqlite:   SQLite SQL dump (full database)
    
    Examples:
    
    \b
        # Export all memories to JSONL
        amt export memories.jsonl
        
        # Export to CSV with specific database
        amt export --db myagent.db --format csv memories.csv
        
        # Export specific branches to Markdown
        amt export --format markdown --branch main --branch feature docs.md
        
        # Full database backup
        amt export --format sqlite backup.sql
    """
    io_module = get_io_module()
    MemoryStore = get_memory_store()
    
    # Configure export
    config = io_module.ExportConfig(
        include_metadata=True,
        include_embeddings=include_embeddings,
        include_deleted=include_deleted,
        branches=list(branch) if branch else None,
        pretty_print=pretty,
    )
    
    # Determine export function
    export_funcs = {
        "jsonl": io_module.export_jsonl,
        "csv": io_module.export_csv,
        "parquet": io_module.export_parquet,
        "markdown": io_module.export_markdown,
        "sqlite": io_module.export_sqlite_dump,
    }
    
    export_func = export_funcs[fmt]
    
    # Check parquet availability
    if fmt == "parquet" and not io_module.PARQUET_AVAILABLE:
        click.echo("Error: Parquet export requires pyarrow.", err=True)
        click.echo("Install with: pip install pyarrow", err=True)
        sys.exit(1)
    
    try:
        with MemoryStore(db) as store:
            result = export_func(store, output, config)
        
        click.echo(f"✓ Exported {result.memory_count} memories to {output}")
        click.echo(f"  Format: {fmt}")
        click.echo(f"  Branches: {', '.join(result.branches_exported)}")
        click.echo(f"  Size: {result.file_size_bytes:,} bytes")
        click.echo(f"  Duration: {result.duration_ms:.1f}ms")
        
        if result.errors:
            click.echo(f"\n  Warnings/Errors: {len(result.errors)}")
            for error in result.errors[:5]:
                click.echo(f"    - {error}", err=True)
                
    except FileNotFoundError:
        click.echo(f"Error: Database not found: {db}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: Export failed: {e}", err=True)
        sys.exit(1)


@cli.command("import")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--db", "-d", default="agent_memory.db", help="Path to memory database")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["jsonl", "csv", "parquet", "sqlite"]),
              help="Import format (auto-detected from extension if not specified)")
@click.option("--merge", "-m", 
              type=click.Choice(["skip", "replace", "error"]),
              default="skip", help="How to handle existing memories")
@click.option("--validate/--no-validate", default=True, help="Validate content during import")
@click.option("--force", is_flag=True, help="Force import without confirmation (for sqlite)")
def import_cmd(input_file: str, db: str, fmt: Optional[str], merge: str, 
               validate: bool, force: bool):
    """Import memories from file.
    
    Supports multiple formats:
    
    \b
        - jsonl:    JSON Lines format
        - csv:      Comma-separated values  
        - parquet:  Apache Parquet
        - sqlite:   SQLite SQL dump (replaces entire database!)
    
    Merge strategies:
    
    \b
        - skip:    Skip memories that already exist
        - replace: Update existing memories with imported data
        - error:   Fail if any duplicates found
    
    Examples:
    
    \b
        # Import JSONL file
        amt import memories.jsonl
        
        # Import CSV with replace strategy
        amt import --merge replace data.csv
        
        # Restore from SQLite dump
        amt import --format sqlite --force backup.sql
    """
    io_module = get_io_module()
    MemoryStore = get_memory_store()
    
    # Auto-detect format from extension if not specified
    if fmt is None:
        ext = Path(input_file).suffix.lower()
        fmt_map = {
            ".jsonl": "jsonl",
            ".json": "jsonl",
            ".csv": "csv",
            ".parquet": "parquet",
            ".sql": "sqlite",
        }
        fmt = fmt_map.get(ext)
        if fmt is None:
            click.echo(f"Error: Cannot detect format from extension '{ext}'", err=True)
            click.echo("Specify format with --format", err=True)
            sys.exit(1)
    
    # Warn about SQLite replacement
    if fmt == "sqlite" and not force:
        if os.path.exists(db):
            click.echo("Warning: SQLite import will REPLACE the entire database!", err=True)
            if not click.confirm("Continue?"):
                click.echo("Aborted.")
                sys.exit(0)
    
    # Check parquet availability
    if fmt == "parquet" and not io_module.PARQUET_AVAILABLE:
        click.echo("Error: Parquet import requires pyarrow.", err=True)
        click.echo("Install with: pip install pyarrow", err=True)
        sys.exit(1)
    
    # Configure import
    config = io_module.ImportConfig(
        merge_strategy=merge,
        validate_content=validate,
    )
    
    # Determine import function
    import_funcs = {
        "jsonl": io_module.import_jsonl,
        "csv": io_module.import_csv,
        "parquet": io_module.import_parquet,
        "sqlite": io_module.import_sqlite_dump,
    }
    
    import_func = import_funcs[fmt]
    
    try:
        with MemoryStore(db) as store:
            result = import_func(store, input_file, config)
        
        click.echo(f"✓ Import complete from {input_file}")
        click.echo(f"  Format: {fmt}")
        click.echo(f"  Imported: {result.memories_imported}")
        click.echo(f"  Skipped: {result.memories_skipped}")
        click.echo(f"  Replaced: {result.memories_replaced}")
        click.echo(f"  Duration: {result.duration_ms:.1f}ms")
        
        if result.errors:
            click.echo(f"\n  Errors: {len(result.errors)}")
            for error in result.errors[:5]:
                click.echo(f"    - {error}", err=True)
            if len(result.errors) > 5:
                click.echo(f"    ... and {len(result.errors) - 5} more", err=True)
                
    except FileNotFoundError as e:
        click.echo(f"Error: File not found: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: Import failed: {e}", err=True)
        sys.exit(1)


# ==============================================================================
# Consolidation Commands
# ==============================================================================

def get_consolidation_module():
    """Lazy import consolidation module."""
    from agent_memory_toolkit.consolidation import (
        MemoryConsolidator,
        ConsolidationConfig,
        ConsolidationStrategy,
        DeduplicationStrategy,
        ConsolidationScheduler,
        ConsolidationDaemon,
        MemoryData,
    )
    return (
        MemoryConsolidator,
        ConsolidationConfig,
        ConsolidationStrategy,
        DeduplicationStrategy,
        ConsolidationScheduler,
        ConsolidationDaemon,
        MemoryData,
    )


@cli.group()
def consolidate():
    """Memory consolidation commands (dedupe, merge, conflicts)."""
    pass


@consolidate.command("run")
@click.option("--db", "-d", default="agent_memory.db", help="Path to memory database")
@click.option("--strategy", "-s", default="hybrid",
              type=click.Choice(["exact_match", "fuzzy_match", "semantic_match", "hybrid"]),
              help="Similarity detection strategy")
@click.option("--threshold", "-t", type=float, default=0.85,
              help="Similarity threshold (0-1, default: 0.85)")
@click.option("--dedup-strategy", default="keep_highest_confidence",
              type=click.Choice([
                  "keep_newest", "keep_oldest", "keep_highest_confidence",
                  "merge_all", "keep_most_accessed"
              ]),
              help="Deduplication strategy")
@click.option("--auto-merge", is_flag=True, help="Automatically merge similar memories")
@click.option("--detect-conflicts", is_flag=True, default=True, help="Detect conflicts")
@click.option("--resolve-conflicts", is_flag=True, help="Auto-resolve conflicts")
@click.option("--dry-run", is_flag=True, help="Analyze without making changes")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def consolidate_run(db: str, strategy: str, threshold: float, dedup_strategy: str,
                   auto_merge: bool, detect_conflicts: bool, resolve_conflicts: bool,
                   dry_run: bool, json_output: bool):
    """Run memory consolidation.
    
    Analyzes memories for duplicates, conflicts, and similarity clusters.
    Can automatically deduplicate and merge related memories.
    
    Examples:
    
    \\b
        # Analyze without changes
        amt consolidate run --dry-run
        
        # Deduplicate with default settings
        amt consolidate run --db myagent.db
        
        # Full consolidation with auto-merge
        amt consolidate run --auto-merge --resolve-conflicts
        
        # Use semantic matching with lower threshold
        amt consolidate run --strategy semantic_match --threshold 0.8
    """
    (
        MemoryConsolidator,
        ConsolidationConfig,
        ConsolidationStrategy,
        DeduplicationStrategy,
        _,
        _,
        MemoryData,
    ) = get_consolidation_module()
    MemoryStore = get_memory_store()
    
    strategy_map = {
        "exact_match": ConsolidationStrategy.EXACT_MATCH,
        "fuzzy_match": ConsolidationStrategy.FUZZY_MATCH,
        "semantic_match": ConsolidationStrategy.SEMANTIC_MATCH,
        "hybrid": ConsolidationStrategy.HYBRID,
    }
    
    dedup_strategy_map = {
        "keep_newest": DeduplicationStrategy.KEEP_NEWEST,
        "keep_oldest": DeduplicationStrategy.KEEP_OLDEST,
        "keep_highest_confidence": DeduplicationStrategy.KEEP_HIGHEST_CONFIDENCE,
        "merge_all": DeduplicationStrategy.MERGE_ALL,
        "keep_most_accessed": DeduplicationStrategy.KEEP_MOST_ACCESSED,
    }
    
    config = ConsolidationConfig(
        strategy=strategy_map[strategy],
        similarity_threshold=threshold,
        dedup_strategy=dedup_strategy_map[dedup_strategy],
        detect_conflicts=detect_conflicts,
        auto_resolve_conflicts=resolve_conflicts,
        auto_merge=auto_merge,
    )
    
    consolidator = MemoryConsolidator(config)
    
    def progress_callback(stage: str, current: int, total: int):
        if not json_output:
            click.echo(f"  [{current}/{total}] {stage.replace('_', ' ').title()}...")
    
    try:
        with MemoryStore(db) as store:
            # Load memories
            memories_raw = store.list(limit=10000)  # Get all memories
            
            memories = [
                MemoryData(
                    id=m.id,
                    content=m.content,
                    embedding=m.embedding,
                    metadata={
                        "confidence": m.metadata.confidence if hasattr(m.metadata, 'confidence') else 1.0,
                        "created_at": m.created_at.isoformat(),
                        "updated_at": m.updated_at.isoformat(),
                        "source": m.metadata.source if hasattr(m.metadata, 'source') else None,
                    }
                )
                for m in memories_raw
            ]
        
        if not memories:
            if json_output:
                click.echo(json.dumps({"error": "No memories found"}))
            else:
                click.echo("No memories found in database.")
            return
        
        if not json_output:
            click.echo(f"Analyzing {len(memories)} memories...")
        
        result = consolidator.consolidate(
            memories,
            dry_run=dry_run,
            progress_callback=progress_callback if not json_output else None,
        )
        
        if json_output:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo()
            if dry_run:
                click.echo("=== Dry Run Analysis ===")
            else:
                click.echo("=== Consolidation Complete ===")
            click.echo()
            click.echo(f"  Memories analyzed: {result.memories_analyzed}")
            click.echo(f"  Duplicates found:  {result.duplicates_found}")
            click.echo(f"  Duplicates removed: {result.duplicates_removed}")
            click.echo(f"  Conflicts detected: {result.conflicts_detected}")
            click.echo(f"  Conflicts resolved: {result.conflicts_resolved}")
            click.echo(f"  Merges performed:  {result.merges_performed}")
            click.echo(f"  Clusters found:    {result.clusters_found}")
            click.echo(f"  Processing time:   {result.processing_time_seconds:.2f}s")
            
            if result.errors:
                click.echo(f"\n  Errors: {len(result.errors)}")
                for err in result.errors[:3]:
                    click.echo(f"    - {err}", err=True)
            
            if dry_run and result.duplicates_found > 0:
                click.echo(f"\n  Run without --dry-run to remove {result.duplicates_found} duplicates.")
                
    except FileNotFoundError:
        click.echo(f"Error: Database not found: {db}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: Consolidation failed: {e}", err=True)
        sys.exit(1)


@consolidate.command("analyze")
@click.option("--db", "-d", default="agent_memory.db", help="Path to memory database")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def consolidate_analyze(db: str, json_output: bool):
    """Analyze memory store for consolidation opportunities.
    
    Shows statistics about potential duplicates, conflicts, and recommendations.
    Does not modify any data.
    """
    (
        MemoryConsolidator,
        ConsolidationConfig,
        _,
        _,
        _,
        _,
        MemoryData,
    ) = get_consolidation_module()
    MemoryStore = get_memory_store()
    
    consolidator = MemoryConsolidator()
    
    try:
        with MemoryStore(db) as store:
            memories_raw = store.list(limit=10000)
            
            memories = [
                MemoryData(
                    id=m.id,
                    content=m.content,
                    embedding=m.embedding,
                    metadata={
                        "confidence": m.metadata.confidence if hasattr(m.metadata, 'confidence') else 1.0,
                        "created_at": m.created_at.isoformat(),
                        "updated_at": m.updated_at.isoformat(),
                    }
                )
                for m in memories_raw
            ]
        
        if not memories:
            if json_output:
                click.echo(json.dumps({"error": "No memories found"}))
            else:
                click.echo("No memories found in database.")
            return
        
        analysis = consolidator.analyze(memories)
        
        if json_output:
            click.echo(json.dumps(analysis, indent=2))
        else:
            click.echo("=== Memory Store Analysis ===\n")
            
            summary = analysis["summary"]
            click.echo(f"  Total memories: {summary['memories_analyzed']}")
            click.echo(f"  Duplicates found: {summary['duplicates_found']}")
            click.echo(f"  Conflicts detected: {summary['conflicts_detected']}")
            click.echo(f"  Similarity clusters: {summary['clusters_found']}")
            
            dedup = analysis["deduplication_estimate"]
            click.echo(f"\n  Exact duplicates: {dedup['exact_duplicates']}")
            click.echo(f"  Estimated fuzzy duplicates: {dedup['estimated_fuzzy_duplicates']}")
            click.echo(f"  Potential reduction: {dedup['estimated_reduction_percent']:.1f}%")
            
            click.echo("\n  Recommendations:")
            for rec in analysis["recommendations"]:
                click.echo(f"    • {rec}")
                
    except FileNotFoundError:
        click.echo(f"Error: Database not found: {db}", err=True)
        sys.exit(1)


@consolidate.command("start")
@click.option("--db", "-d", default="agent_memory.db", help="Path to memory database")
@click.option("--interval", "-i", type=int, default=24,
              help="Run interval in hours (default: 24)")
@click.option("--state-file", default="~/.amt/consolidation_state.json",
              help="Path to state file")
@click.option("--daemon", is_flag=True, help="Run as background daemon")
@click.option("--pid-file", default="~/.amt/consolidation.pid",
              help="PID file for daemon mode")
def consolidate_start(db: str, interval: int, state_file: str, daemon: bool, pid_file: str):
    """Start background consolidation scheduler.
    
    Runs consolidation periodically according to the specified interval.
    
    Examples:
    
    \\b
        # Run every 24 hours (default)
        amt consolidate start
        
        # Run every 6 hours
        amt consolidate start --interval 6
        
        # Run as daemon
        amt consolidate start --daemon
    """
    (
        MemoryConsolidator,
        ConsolidationConfig,
        _,
        _,
        ConsolidationScheduler,
        ConsolidationDaemon,
        MemoryData,
    ) = get_consolidation_module()
    MemoryStore = get_memory_store()
    
    state_file = os.path.expanduser(state_file)
    pid_file = os.path.expanduser(pid_file)
    
    config = ConsolidationConfig(
        run_interval_hours=interval,
    )
    
    consolidator = MemoryConsolidator(config)
    
    def memory_loader():
        with MemoryStore(db) as store:
            memories_raw = store.list(limit=10000)
            return [
                MemoryData(
                    id=m.id,
                    content=m.content,
                    embedding=m.embedding,
                    metadata={
                        "confidence": m.metadata.confidence if hasattr(m.metadata, 'confidence') else 1.0,
                    }
                )
                for m in memories_raw
            ]
    
    scheduler = ConsolidationScheduler(
        consolidator=consolidator,
        config=config,
        memory_loader=memory_loader,
        state_file=state_file,
    )
    
    def on_complete(result):
        click.echo(f"Consolidation complete: {result.summary}")
    
    scheduler.on_complete(on_complete)
    
    if daemon:
        click.echo(f"Starting consolidation daemon (interval: {interval}h)")
        click.echo(f"PID file: {pid_file}")
        
        daemon_proc = ConsolidationDaemon(
            scheduler=scheduler,
            pid_file=pid_file,
        )
        daemon_proc.run()
    else:
        click.echo(f"Starting consolidation scheduler (interval: {interval}h)")
        click.echo("Press Ctrl+C to stop.")
        
        try:
            scheduler.start()
            # Keep running
            import time
            while scheduler.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("\nStopping scheduler...")
            scheduler.stop()


@consolidate.command("stop")
@click.option("--pid-file", default="~/.amt/consolidation.pid",
              help="PID file for daemon")
def consolidate_stop(pid_file: str):
    """Stop background consolidation daemon."""
    (_, _, _, _, _, ConsolidationDaemon, _) = get_consolidation_module()
    
    pid_file = os.path.expanduser(pid_file)
    
    if ConsolidationDaemon.is_running(pid_file):
        if ConsolidationDaemon.stop_daemon(pid_file):
            click.echo("Consolidation daemon stopped.")
        else:
            click.echo("Failed to stop daemon.", err=True)
            sys.exit(1)
    else:
        click.echo("No daemon running.")


@consolidate.command("status")
@click.option("--state-file", default="~/.amt/consolidation_state.json",
              help="Path to state file")
@click.option("--pid-file", default="~/.amt/consolidation.pid",
              help="PID file for daemon")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def consolidate_status(state_file: str, pid_file: str, json_output: bool):
    """Show consolidation scheduler status."""
    (_, _, _, _, _, ConsolidationDaemon, _) = get_consolidation_module()
    
    state_file = os.path.expanduser(state_file)
    pid_file = os.path.expanduser(pid_file)
    
    is_running = ConsolidationDaemon.is_running(pid_file)
    
    status = {
        "daemon_running": is_running,
        "last_run": None,
        "run_history": [],
    }
    
    # Load state if exists
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            status["last_run"] = state.get("last_run")
            status["run_history"] = state.get("run_history", [])[-5:]
        except Exception:
            pass
    
    if json_output:
        click.echo(json.dumps(status, indent=2))
    else:
        click.echo("=== Consolidation Status ===\n")
        click.echo(f"  Daemon running: {'Yes' if is_running else 'No'}")
        click.echo(f"  Last run: {status['last_run'] or 'Never'}")
        
        if status["run_history"]:
            click.echo("\n  Recent runs:")
            for run in status["run_history"][-5:]:
                click.echo(f"    • {run.get('timestamp', 'N/A')}: "
                          f"{run.get('memories_analyzed', 0)} analyzed, "
                          f"{run.get('duplicates_removed', 0)} removed")


# ==============================================================================
# Dashboard Commands
# ==============================================================================

def get_dashboard_module():
    """Lazy import dashboard module."""
    from agent_memory_toolkit.dashboard import DashboardServer, DashboardConfig, AnalyticsEngine
    return DashboardServer, DashboardConfig, AnalyticsEngine


@cli.group()
def dashboard():
    """Analytics dashboard commands."""
    pass


@dashboard.command("serve")
@click.option("--host", "-h", default="127.0.0.1",
              help="Host to bind to (default: 127.0.0.1)")
@click.option("--port", "-p", type=int, default=8080,
              help="Port to bind to (default: 8080)")
@click.option("--db", "-d", default="agent_memory.db",
              help="Path to memory database")
@click.option("--no-open", is_flag=True,
              help="Don't automatically open browser")
def dashboard_serve(host: str, port: int, db: str, no_open: bool):
    """Start the analytics dashboard web server.
    
    Opens a web-based analytics dashboard showing memory statistics,
    search trends, storage metrics, and branch comparisons.
    
    Examples:
    
    \\b
        # Start dashboard on default port
        amt dashboard serve
        
        # Custom port and database
        amt dashboard serve --port 9090 --db ~/my_memory.db
        
        # Without opening browser
        amt dashboard serve --no-open
    """
    DashboardServer, DashboardConfig, _ = get_dashboard_module()
    
    config = DashboardConfig(
        host=host,
        port=port,
        db_path=db,
        auto_open=not no_open,
    )
    
    click.echo("Agent Memory Toolkit - Analytics Dashboard")
    click.echo("=" * 45)
    click.echo(f"Database:  {db}")
    click.echo(f"URL:       http://{host}:{port}")
    click.echo()
    
    try:
        server = DashboardServer(config)
        server.start(blocking=True)
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@dashboard.command("stats")
@click.option("--db", "-d", default="agent_memory.db",
              help="Path to memory database")
@click.option("--days", type=int, default=30,
              help="Number of days for trends (default: 30)")
@click.option("--json-output", "-j", is_flag=True,
              help="Output as JSON")
def dashboard_stats(db: str, days: int, json_output: bool):
    """Show dashboard statistics without starting the server.
    
    Displays memory counts, search trends, and storage metrics
    in the terminal.
    
    Examples:
    
    \\b
        # Show stats
        amt dashboard stats
        
        # Get stats as JSON
        amt dashboard stats --json
        
        # Stats for last 7 days
        amt dashboard stats --days 7
    """
    _, _, AnalyticsEngine = get_dashboard_module()
    
    try:
        engine = AnalyticsEngine(db_path=db)
        data = engine.get_all_analytics(days=days)
        
        if json_output:
            click.echo(json.dumps(data, indent=2))
        else:
            stats = data["memory_stats"]
            storage = data["storage_metrics"]
            searches = data["search_trends"]
            branches = data["branch_comparison"]
            
            click.echo("Agent Memory Toolkit - Statistics")
            click.echo("=" * 40)
            click.echo()
            
            click.echo("📝 Memory Stats:")
            click.echo(f"   Total memories:  {stats['total_memories']}")
            click.echo(f"   Active:          {stats['active_memories']}")
            click.echo(f"   Deleted:         {stats['deleted_memories']}")
            click.echo(f"   Avg size:        {stats['avg_memory_size']:.1f} bytes")
            click.echo()
            
            click.echo("🌳 Branches:")
            click.echo(f"   Total branches:  {stats['total_branches']}")
            click.echo(f"   Total commits:   {stats['total_commits']}")
            for b in branches.get("branches", [])[:5]:
                marker = "* " if b.get("is_current") else "  "
                click.echo(f"   {marker}{b['name']}: {b['memory_count']} memories")
            click.echo()
            
            click.echo("🔍 Search Stats:")
            click.echo(f"   Total searches:  {searches['total_searches']}")
            click.echo(f"   Today:           {searches['searches_today']}")
            click.echo(f"   This week:       {searches['searches_this_week']}")
            if searches.get("top_queries"):
                click.echo("   Top queries:")
                for q in searches["top_queries"][:3]:
                    click.echo(f"      - \"{q['query']}\" ({q['count']})")
            click.echo()
            
            click.echo("💾 Storage:")
            size_mb = storage['total_size_bytes'] / (1024 * 1024)
            click.echo(f"   Database size:   {size_mb:.2f} MB")
            click.echo(f"   FTS index:       {storage['fts_index_size_bytes']:,} bytes")
            click.echo(f"   Embeddings:      {storage['embeddings_size_bytes']:,} bytes")
            
    except FileNotFoundError:
        click.echo(f"Error: Database not found: {db}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
