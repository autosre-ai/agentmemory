# MCP Server

Use Agent Memory Toolkit with any MCP-compatible LLM client like Claude Desktop, Cursor, or Continue.

---

## Overview

The MCP server exposes Agent Memory Toolkit's functionality as tools that can be used by AI assistants to:

- Store and retrieve persistent memories
- Search memories using hybrid retrieval (BM25 + vectors)
- Extract structured facts from conversations
- Validate content for security issues
- Compress conversation context to fit token budgets

---

## Installation

=== "From PyPI"

    ```bash
    pip install agent-memory-toolkit[mcp]
    ```

=== "From Source"

    ```bash
    git clone https://github.com/autosre-ai/agent-memory-toolkit.git
    cd agent-memory-toolkit
    pip install -e ".[mcp]"
    ```

Verify the installation:

```bash
amt-mcp --version
```

---

## Quick Start

### 1. Generate Configuration

```bash
# For Claude Desktop
amt-mcp config claude

# For Cursor
amt-mcp config cursor
```

### 2. Add to Your Client

Copy the JSON output to your client's configuration file.

### 3. Restart Your Client

The memory tools will now be available.

---

## Configuring LLM Clients

### Claude Desktop

1. Open Claude Desktop settings
2. Navigate to **Developer → MCP Servers**
3. Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "agent-memory-toolkit": {
      "command": "amt-mcp",
      "args": ["serve", "--db-path", "/path/to/agent_memory.db"]
    }
  }
}
```

4. Restart Claude Desktop

---

### Cursor

1. Open Cursor settings (Cmd/Ctrl + ,)
2. Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "agent-memory-toolkit": {
      "command": "amt-mcp",
      "args": ["serve", "--db-path", "/path/to/agent_memory.db"],
      "env": {}
    }
  }
}
```

3. Restart Cursor

---

### Custom Configuration

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

---

## Running the Server

### CLI Commands

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

### Using Main CLI

```bash
amt mcp serve
amt mcp serve --db ~/my_memories.db
```

### Programmatic Usage

```python
from agent_memory.mcp import create_mcp_server, MCPConfig

config = MCPConfig(
    memory_db="~/agent_memory.db",
    security_level="medium",
)

server = create_mcp_server(config)
server.run(transport="stdio")
```

---

## Available Tools

### Memory Operations

| Tool | Description |
|------|-------------|
| `memory_add` | Add a new memory to the store |
| `memory_query` | Search memories using hybrid retrieval |
| `memory_get` | Get a specific memory by ID |
| `memory_update` | Update an existing memory |
| `memory_delete` | Delete a memory |
| `memory_list` | List memories with pagination |
| `memory_history` | Get version history |

### Extraction

| Tool | Description |
|------|-------------|
| `extract_memories` | Extract structured facts from text |

### Security

| Tool | Description |
|------|-------------|
| `guard_check` | Validate content for injection attacks |

### Compression

| Tool | Description |
|------|-------------|
| `compress_context` | Compress conversation to fit token budget |
| `count_tokens` | Count tokens in text |

---

## Tool Examples

### memory_add

Add a new memory:

```
Arguments:
- content: "User prefers dark mode in all applications"
- source: "conversation" (optional)
- tags: ["preferences", "ui"] (optional)
- confidence: 0.95 (optional)

Returns:
{
  "success": true,
  "memory_id": "mem_abc123",
  "message": "Memory stored successfully"
}
```

---

### memory_query

Search memories:

```
Arguments:
- query: "user preferences dark mode"
- limit: 10 (optional)

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

---

### extract_memories

Extract structured facts:

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

---

### guard_check

Validate content:

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

---

### compress_context

Compress conversation:

```
Arguments:
- messages: [{"role": "user", "content": "..."}, ...]
- max_tokens: 4000 (optional)
- mode: "balanced" (optional)

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

---

## Configuration Options

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--transport` | `stdio` | Transport: `stdio` or `sse` |
| `--db-path` | `agent_memory.db` | SQLite database path |
| `--host` | `127.0.0.1` | Host for SSE transport |
| `--port` | `8765` | Port for SSE transport |
| `--log-level` | `INFO` | Logging level |
| `--security-level` | `medium` | Security validation level |
| `--disable-extraction` | false | Disable extraction tools |
| `--disable-security` | false | Disable security tools |
| `--disable-compression` | false | Disable compression tools |

---

### Security Levels

| Level | Description |
|-------|-------------|
| `minimal` | Basic validation only |
| `low` | Light pattern detection |
| `medium` | Balanced (recommended) |
| `high` | Strict validation |
| `paranoid` | Maximum security |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AMT_DB_PATH` | Default database path |
| `AMT_LOG_LEVEL` | Default logging level |
| `AMT_SECURITY_LEVEL` | Default security level |

---

## Troubleshooting

### Server Not Starting

1. Check MCP dependency is installed:
   ```bash
   pip install agent-memory-toolkit[mcp]
   ```

2. Verify CLI works:
   ```bash
   amt-mcp info
   ```

3. Check logs with DEBUG level:
   ```bash
   amt-mcp serve --log-level DEBUG
   ```

### Tools Not Appearing in Client

1. Restart your LLM client after configuration changes
2. Validate JSON syntax in configuration file
3. Ensure `amt-mcp` is in your PATH

### Database Errors

1. Ensure database directory exists and is writable
2. Use absolute path in configuration
3. Check file permissions

### Connection Issues (SSE Mode)

1. Verify port is not in use
2. Check firewall settings
3. Try a different port: `--port 9000`

---

## Development

### Running Tests

```bash
# Run MCP tests
pytest agent_memory/mcp/tests/ -v

# Run E2E tests
pytest agent_memory/mcp/tests/test_e2e.py -v
```

### Debugging

Run in SSE mode for debugging:

```bash
amt-mcp serve --transport sse --log-level DEBUG
```

Connect with an MCP client or test with curl.

---

## Related Documentation

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Claude Desktop MCP Setup](https://docs.anthropic.com/mcp)
- [API Reference](api-reference.md)
