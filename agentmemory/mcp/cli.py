"""
Standalone CLI for Agent Memory Toolkit MCP Server.

Provides a dedicated entry point for MCP server operations:
    - amt-mcp serve: Start the MCP server
    - amt-mcp config: Generate configuration for LLM clients

This is an alternative to using 'amt mcp serve' from the main CLI.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="amt-mcp")
def main():
    """Agent Memory Toolkit MCP Server.
    
    A standalone MCP (Model Context Protocol) server that exposes
    memory toolkit functionality as tools for LLM clients.
    
    Use this to integrate agent memory capabilities with Claude Desktop,
    Cursor, or any MCP-compatible application.
    
    Quick start:
    
        # Start server (stdio mode for Claude Desktop)
        amt-mcp serve
        
        # Generate Claude Desktop configuration
        amt-mcp config claude
        
        # Show available tools
        amt-mcp tools
    """
    pass


def get_mcp_server():
    """Lazy import MCP server components."""
    from agentmemory.mcp import create_mcp_server, MCPConfig
    return create_mcp_server, MCPConfig


@main.command("serve")
@click.option("--transport", "-t", default="stdio",
              type=click.Choice(["stdio", "sse"]),
              help="Transport type (default: stdio)")
@click.option("--db-path", "--db", "-d", default="agent_memory.db",
              help="Path to memory database")
@click.option("--host", "-H", default="127.0.0.1",
              help="Host for SSE server (default: 127.0.0.1)")
@click.option("--port", "-p", type=int, default=8765,
              help="Port for SSE server (default: 8765)")
@click.option("--log-level", "-l", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
              help="Logging level (default: INFO)")
@click.option("--security-level", "-s", default="medium",
              type=click.Choice(["minimal", "low", "medium", "high", "paranoid"]),
              help="Security level for content validation (default: medium)")
@click.option("--disable-extraction", is_flag=True,
              help="Disable memory extraction tools")
@click.option("--disable-security", is_flag=True,
              help="Disable security validation tools")
@click.option("--disable-compression", is_flag=True,
              help="Disable context compression tools")
def serve(transport: str, db_path: str, host: str, port: int, log_level: str, 
          security_level: str, disable_extraction: bool, disable_security: bool,
          disable_compression: bool):
    """Start the MCP server.
    
    This exposes the memory toolkit as MCP tools that can be used by
    LLM clients like Claude Desktop, Cursor, or any MCP-compatible application.
    
    Transport modes:
    
    \b
        stdio: Communicate over stdin/stdout (default, use with Claude Desktop)
        sse:   Run HTTP server with Server-Sent Events
    
    Example usage:
    
    \b
        # Start with stdio transport (for Claude Desktop)
        amt-mcp serve
        
        # Start with custom database
        amt-mcp serve --db-path ~/my_memories.db
        
        # Start SSE server for debugging
        amt-mcp serve --transport sse --port 8765
        
        # High security mode
        amt-mcp serve --security-level high
    
    Claude Desktop configuration (claude_desktop_config.json):
    
    \b
        {
          "mcpServers": {
            "agent-memory": {
              "command": "amt-mcp",
              "args": ["serve", "--db-path", "~/agent_memory.db"]
            }
          }
        }
    """
    create_mcp_server, MCPConfig = get_mcp_server()
    
    # Expand user path
    db_path_expanded = str(Path(db_path).expanduser())
    
    config = MCPConfig(
        memory_db=db_path_expanded,
        host=host,
        port=port,
        log_level=log_level,
        security_level=security_level,
        enable_extraction=not disable_extraction,
        enable_security=not disable_security,
        enable_compression=not disable_compression,
    )
    
    mcp_server = create_mcp_server(config)
    
    if transport == "stdio":
        # Log to stderr so we don't interfere with MCP protocol on stdout
        click.echo(f"Starting MCP server (stdio transport)...", err=True)
        click.echo(f"Memory database: {db_path_expanded}", err=True)
        click.echo(f"Security level: {security_level}", err=True)
        click.echo(f"Features: extraction={not disable_extraction}, "
                   f"security={not disable_security}, "
                   f"compression={not disable_compression}", err=True)
        mcp_server.run(transport="stdio")
    else:
        click.echo(f"Starting MCP server (SSE transport)...")
        click.echo(f"URL: http://{host}:{port}")
        click.echo(f"Memory database: {db_path_expanded}")
        click.echo(f"Security level: {security_level}")
        mcp_server.run()


@main.command("config")
@click.argument("target", type=click.Choice(["claude", "cursor", "json"]))
@click.option("--db-path", "--db", "-d", default="agent_memory.db",
              help="Path to memory database")
@click.option("--output", "-o", type=click.Path(),
              help="Output file (default: stdout)")
@click.option("--security-level", "-s", default="medium",
              help="Security level to configure")
def config(target: str, db_path: str, output: Optional[str], security_level: str):
    """Generate MCP configuration for LLM clients.
    
    Creates ready-to-use JSON configuration for various MCP clients.
    
    Targets:
    
    \b
        claude:  Claude Desktop configuration
        cursor:  Cursor editor configuration
        json:    Generic MCP configuration
    
    Examples:
    
    \b
        # Show Claude Desktop configuration
        amt-mcp config claude
        
        # Save Cursor configuration to file
        amt-mcp config cursor --output ~/.cursor/mcp.json
        
        # Configure with custom database
        amt-mcp config claude --db-path ~/my_memories.db
    
    Configuration locations:
    
    \b
        Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json
        Cursor: ~/.cursor/mcp.json
    """
    # Resolve database path to absolute
    db_path_resolved = str(Path(db_path).expanduser().resolve())
    
    if target == "claude":
        config_data = {
            "mcpServers": {
                "agent-memory-toolkit": {
                    "command": "amt-mcp",
                    "args": ["serve", "--db-path", db_path_resolved]
                }
            }
        }
        config_file = "claude_desktop_config.json"
        config_location = "~/Library/Application Support/Claude/claude_desktop_config.json"
    elif target == "cursor":
        config_data = {
            "mcpServers": {
                "agent-memory-toolkit": {
                    "command": "amt-mcp",
                    "args": ["serve", "--db-path", db_path_resolved],
                    "env": {}
                }
            }
        }
        config_file = "mcp.json"
        config_location = "~/.cursor/mcp.json"
    else:  # json
        config_data = {
            "name": "agent-memory-toolkit",
            "command": "amt-mcp",
            "args": ["serve", "--db-path", db_path_resolved],
            "transport": "stdio"
        }
        config_file = "mcp.json"
        config_location = "your MCP configuration file"
    
    config_json = json.dumps(config_data, indent=2)
    
    if output:
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(config_json)
        click.echo(f"✓ Configuration written to {output_path}")
    else:
        click.echo(config_json)
    
    # Print usage instructions to stderr
    click.echo(f"\nTo use with {target.title()}:", err=True)
    if target == "claude":
        click.echo("1. Open Claude Desktop settings", err=True)
        click.echo("2. Navigate to Developer → MCP Servers", err=True)
        click.echo("3. Add the configuration above to your config file", err=True)
        click.echo(f"   Location: {config_location}", err=True)
    elif target == "cursor":
        click.echo("1. Open Cursor settings (Cmd/Ctrl + ,)", err=True)
        click.echo("2. Search for 'MCP'", err=True)
        click.echo("3. Add the server configuration", err=True)
        click.echo(f"   Location: {config_location}", err=True)
    else:
        click.echo(f"Add this configuration to {config_location}", err=True)


@main.command("tools")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def tools(json_output: bool):
    """List available MCP tools.
    
    Shows all tools exposed by the MCP server with descriptions.
    """
    tool_list = [
        {
            "category": "Memory Operations",
            "tools": [
                {"name": "memory_add", "description": "Add a new memory to the store"},
                {"name": "memory_query", "description": "Search memories using full-text search"},
                {"name": "memory_get", "description": "Get a specific memory by ID"},
                {"name": "memory_update", "description": "Update an existing memory"},
                {"name": "memory_delete", "description": "Delete a memory"},
                {"name": "memory_list", "description": "List memories with pagination"},
                {"name": "memory_history", "description": "Get version history"},
            ]
        },
        {
            "category": "Extraction",
            "tools": [
                {"name": "extract_memories", "description": "Extract structured facts from text"},
            ]
        },
        {
            "category": "Security",
            "tools": [
                {"name": "guard_check", "description": "Validate content for security issues"},
            ]
        },
        {
            "category": "Compression",
            "tools": [
                {"name": "compress_context", "description": "Compress conversation to fit token budget"},
                {"name": "count_tokens", "description": "Count tokens in text"},
            ]
        },
    ]
    
    if json_output:
        click.echo(json.dumps(tool_list, indent=2))
    else:
        click.echo("Agent Memory Toolkit MCP Tools")
        click.echo("=" * 40)
        click.echo()
        
        for category in tool_list:
            click.echo(f"{category['category']}:")
            for tool in category["tools"]:
                click.echo(f"  • {tool['name']:20} - {tool['description']}")
            click.echo()


@main.command("info")
def info():
    """Show server information and status.
    
    Displays version, configuration defaults, and feature availability.
    """
    click.echo("Agent Memory Toolkit MCP Server")
    click.echo("=" * 40)
    click.echo()
    click.echo("Version: 0.1.0")
    click.echo()
    click.echo("Default Configuration:")
    click.echo("  Transport:       stdio")
    click.echo("  Database:        agent_memory.db")
    click.echo("  Host (SSE):      127.0.0.1")
    click.echo("  Port (SSE):      8765")
    click.echo("  Log level:       INFO")
    click.echo("  Security level:  medium")
    click.echo()
    click.echo("Features:")
    
    # Check feature availability
    features = {
        "Memory Store": True,  # Core feature
        "Memory Extraction": True,
        "Security Validation": True,
        "Context Compression": True,
    }
    
    try:
        from agentmemory.store import MemoryStore
        features["Memory Store"] = True
    except ImportError:
        features["Memory Store"] = False
    
    try:
        from agentmemory.extraction import MemoryExtractor
        features["Memory Extraction"] = True
    except ImportError:
        features["Memory Extraction"] = False
    
    try:
        from agentmemory.security import MemoryGuard
        features["Security Validation"] = True
    except ImportError:
        features["Security Validation"] = False
    
    try:
        from agentmemory.compression import ContextCompressor
        features["Context Compression"] = True
    except ImportError:
        features["Context Compression"] = False
    
    for feature, available in features.items():
        status = "✓" if available else "✗"
        click.echo(f"  {status} {feature}")
    
    click.echo()
    click.echo("MCP Protocol:")
    
    try:
        import mcp
        click.echo(f"  ✓ MCP library installed")
    except ImportError:
        click.echo(f"  ✗ MCP library not installed")
        click.echo(f"    Install with: pip install agent-memory-toolkit[mcp]")


if __name__ == "__main__":
    main()
