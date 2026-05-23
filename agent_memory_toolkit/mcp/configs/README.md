# MCP Server Configuration Examples

This directory contains example configuration files for integrating the Agent Memory Toolkit MCP server with various LLM clients.

## Claude Desktop

Add to: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
         `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

See: `claude_desktop_config.json`

## Cursor

Add to: `~/.cursor/mcp.json`

See: `cursor_config.json`

## Configuration Options

### Database Location

By default, the MCP server uses `agent_memory.db` in the current directory.
Change the `--db` argument to specify a different location:

```json
{
  "args": ["mcp", "serve", "--db", "/absolute/path/to/memory.db"]
}
```

### Security Level

Set the security level for content validation:

```json
{
  "args": ["mcp", "serve", "--db", "~/memory.db", "--security-level", "high"]
}
```

Options: `minimal`, `low`, `medium` (default), `high`, `paranoid`

### Logging

Enable debug logging:

```json
{
  "args": ["mcp", "serve", "--log-level", "DEBUG"]
}
```

## Available Tools

Once configured, the following tools will be available:

### Memory Operations
- `memory_add` - Add a new memory
- `memory_query` - Search memories
- `memory_get` - Get memory by ID
- `memory_update` - Update a memory
- `memory_delete` - Delete a memory
- `memory_list` - List memories
- `memory_history` - View history

### Extraction
- `extract_memories` - Extract facts from text

### Security
- `guard_check` - Validate content safety

### Compression
- `compress_context` - Compress conversations
- `count_tokens` - Count tokens in text

## Testing

Generate configuration using the CLI:

```bash
# For Claude Desktop
amt mcp config claude

# For Cursor
amt mcp config cursor

# With custom database path
amt mcp config claude --db ~/my_memories.db
```

Test the server directly:

```bash
# Run with stdio transport (for testing)
amt mcp serve

# Run with SSE transport (for debugging)
amt mcp serve --transport sse --port 8765
```
