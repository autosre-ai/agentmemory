# MCP Server for Agent Memory Toolkit

This document explains how to install, configure, and use the Agent Memory Toolkit MCP (Model Context Protocol) server with LLM clients like Claude Desktop and Cursor.

## Overview

The MCP server exposes the Agent Memory Toolkit's functionality as tools that can be used by any MCP-compatible LLM client. This enables AI assistants to:

- Store and retrieve persistent memories
- Search memories using full-text search
- Extract structured facts from conversations
- Validate content for security issues
- Compress conversation context to fit token budgets

## Installation

### From PyPI (recommended)

```bash
# Install with MCP support
pip install agent-memory-toolkit[mcp]
```

### From source

```bash
git clone https://github.com/agent-memory-toolkit/agent-memory-toolkit.git
cd agent-memory-toolkit
pip install -e ".[mcp]"
```

### Verify installation

```bash
# Check the CLI is available
amt-mcp --version

# Or via main CLI
amt mcp --help
```

## Quick Start

### 1. Generate configuration

```bash
# For Claude Desktop
amt-mcp config claude

# For Cursor
amt-mcp config cursor
```

### 2. Copy configuration to your client

The output is JSON that you add to your client's configuration file.

### 3. Restart your client

The memory tools will now be available in your LLM client.

## Running the Server

### Standalone CLI (amt-mcp)

```bash
# Start with default settings (stdio transport)
amt-mcp serve

# Custom database path
amt-mcp serve --db-path ~/my_memories.db

# SSE transport for debugging
amt-mcp serve --transport sse --port 8765

# High security mode
amt-mcp serve --security-level high

# View all options
amt-mcp serve --help
```

### Main CLI (amt)

```bash
# Same functionality via main CLI
amt mcp serve
amt mcp serve --db ~/my_memories.db
```

### Programmatic usage

```python
from agent_memory.mcp import create_mcp_server, MCPConfig

config = MCPConfig(
    memory_db="~/agent_memory.db",
    security_level="medium",
)

server = create_mcp_server(config)
server.run(transport="stdio")
```

## Configuring LLM Clients

### Claude Desktop

1. Open Claude Desktop settings
2. Navigate to **Developer → MCP Servers**
3. Add the following to your `claude_desktop_config.json`:

**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "agent-memory-toolkit": {
      "command": "amt-mcp",
      "args": ["serve", "--db-path", "/path/to/your/agent_memory.db"]
    }
  }
}
```

4. Restart Claude Desktop

### Cursor

1. Open Cursor settings (Cmd/Ctrl + ,)
2. Search for "MCP"
3. Add the server configuration to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agent-memory-toolkit": {
      "command": "amt-mcp",
      "args": ["serve", "--db-path", "/path/to/your/agent_memory.db"],
      "env": {}
    }
  }
}
```

4. Restart Cursor

### Custom Configuration

You can customize the server behavior with additional arguments:

```json
{
  "mcpServers": {
    "agent-memory-toolkit": {
      "command": "amt-mcp",
      "args": [
        "serve",
        "--db-path", "/path/to/agent_memory.db",
        "--security-level", "high",
        "--log-level", "DEBUG"
      ]
    }
  }
}
```

## Available Tools

The MCP server provides the following tools:

### Memory Operations

| Tool | Description |
|------|-------------|
| `memory_add` | Add a new memory to the store |
| `memory_query` | Search memories using full-text search (BM25 ranking) |
| `memory_get` | Get a specific memory by ID |
| `memory_update` | Update an existing memory |
| `memory_delete` | Delete a memory |
| `memory_list` | List memories with pagination |
| `memory_history` | Get version history for a memory or the store |

### Extraction

| Tool | Description |
|------|-------------|
| `extract_memories` | Extract structured facts from text (names, preferences, etc.) |

### Security

| Tool | Description |
|------|-------------|
| `guard_check` | Validate content for injection attacks and security issues |

### Compression

| Tool | Description |
|------|-------------|
| `compress_context` | Compress conversation context to fit token budget |
| `count_tokens` | Count tokens in text |

## Tool Usage Examples

### memory_add

Add a new memory to the store:

```
Arguments:
- content: "User prefers dark mode in all applications"
- source: "conversation" (optional)
- tags: ["preferences", "ui"] (optional)
- confidence: 0.95 (optional, default: 1.0)

Returns:
{
  "success": true,
  "memory_id": "mem_abc123",
  "message": "Memory stored successfully"
}
```

### memory_query

Search memories using natural language:

```
Arguments:
- query: "user preferences dark mode"
- limit: 10 (optional, default: 10)

Returns:
{
  "success": true,
  "query": "user preferences dark mode",
  "count": 2,
  "memories": [
    {
      "id": "mem_abc123",
      "content": "User prefers dark mode...",
      "score": 0.85
    }
  ]
}
```

### extract_memories

Extract structured facts from text:

```
Arguments:
- text: "My name is Alice and I work at Google"
- source: "onboarding" (optional)

Returns:
{
  "success": true,
  "count": 2,
  "memories": [
    {"domain": "biography", "key": "name", "value": "Alice", "confidence": 0.9},
    {"domain": "work", "key": "company", "value": "Google", "confidence": 0.85}
  ]
}
```

### guard_check

Validate content for security issues:

```
Arguments:
- content: "Some user-provided content"

Returns:
{
  "success": true,
  "is_safe": true,
  "adjusted_confidence": 0.9,
  "validation_time_ms": 5.2
}
```

### compress_context

Compress conversation to fit token budget:

```
Arguments:
- messages: [{"role": "user", "content": "..."}, ...]
- max_tokens: 4000 (optional)
- mode: "balanced" (optional: "aggressive", "balanced", "conservative", "lossless")

Returns:
{
  "success": true,
  "messages": [...compressed messages...],
  "stats": {
    "original_tokens": 5000,
    "compressed_tokens": 3500,
    "compression_ratio": 0.7
  }
}
```

## Configuration Options

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--transport` | `stdio` | Transport type: `stdio` or `sse` |
| `--db-path` | `agent_memory.db` | Path to SQLite database |
| `--host` | `127.0.0.1` | Host for SSE transport |
| `--port` | `8765` | Port for SSE transport |
| `--log-level` | `INFO` | Logging level |
| `--security-level` | `medium` | Security validation level |
| `--disable-extraction` | false | Disable extraction tools |
| `--disable-security` | false | Disable security tools |
| `--disable-compression` | false | Disable compression tools |

### Security Levels

| Level | Description |
|-------|-------------|
| `minimal` | Basic validation only |
| `low` | Light pattern detection |
| `medium` | Balanced security (recommended) |
| `high` | Strict validation |
| `paranoid` | Maximum security, may have false positives |

## Troubleshooting

### Server not starting

1. Check that the MCP dependency is installed:
   ```bash
   pip install agent-memory-toolkit[mcp]
   ```

2. Verify the CLI works:
   ```bash
   amt-mcp info
   ```

3. Check logs by running with DEBUG level:
   ```bash
   amt-mcp serve --log-level DEBUG
   ```

### Tools not appearing in client

1. Restart your LLM client after updating configuration
2. Check the configuration file syntax is valid JSON
3. Ensure the path to `amt-mcp` is in your PATH

### Database errors

1. Ensure the database directory exists and is writable
2. Use an absolute path in your configuration
3. Check file permissions

### Connection issues (SSE mode)

1. Verify the port is not in use
2. Check firewall settings
3. Try a different port: `--port 9000`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AMT_DB_PATH` | Default database path |
| `AMT_LOG_LEVEL` | Default logging level |
| `AMT_SECURITY_LEVEL` | Default security level |

## Development

### Running tests

```bash
# Run MCP tests
pytest agent_memory/mcp/tests/ -v

# Run E2E tests
pytest agent_memory/mcp/tests/test_e2e.py -v
```

### Debugging

Run the server in SSE mode for debugging:

```bash
amt-mcp serve --transport sse --log-level DEBUG
```

Then connect with an MCP client or test with curl.

## Related Documentation

- [Agent Memory Toolkit README](README.md)
- [API Documentation](API.md)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Claude Desktop MCP Setup](https://docs.anthropic.com/mcp)
