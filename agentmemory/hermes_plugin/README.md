# Agent Memory Toolkit - Hermes Plugin

A native Hermes Agent memory provider plugin that brings local-first, persistent memory capabilities to your AI agent.

## Features

- **SQLite-backed Storage**: Fast, reliable local storage with FTS5 full-text search
- **Vector Search**: Optional semantic similarity search using sentence-transformers
- **Structured Extraction**: Automatically extract memories across 6 cognitive domains
- **Security Validation**: Built-in poison detection and content validation
- **Context Compression**: Intelligent compression when conversations get long
- **Auto-injection**: Automatically inject relevant memories into context
- **CLI Tools**: `amt-hermes` command for memory management

## Installation

### Option 1: Quick Install (Recommended)

From the agent-memory-toolkit directory:

```bash
# Install the toolkit
pip install agent-memory-toolkit

# Install the Hermes plugin
python -m agent_memory.hermes_plugin.cli install

# Run setup wizard
python -m agent_memory.hermes_plugin.cli setup
```

### Option 2: Symlink Install (Development)

For development, symlink the plugin to your Hermes plugins directory:

```bash
mkdir -p ~/.hermes/plugins/memory/
ln -s /path/to/agent-memory-toolkit/agent_memory/hermes_plugin ~/.hermes/plugins/memory/agent-memory-toolkit
```

### Option 3: Manual Copy

1. Copy the plugin to Hermes plugins directory:

```bash
mkdir -p ~/.hermes/plugins/memory/agent-memory-toolkit
cp -r agent_memory/hermes_plugin/* ~/.hermes/plugins/memory/agent-memory-toolkit/
```

2. Add to your `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - agent-memory-toolkit

memory:
  provider: agent-memory-toolkit
```

3. Install the toolkit package:

```bash
pip install agent-memory-toolkit
# Or with all features:
pip install agent-memory-toolkit[all]
```

## Configuration

### Interactive Setup

```bash
amt-hermes setup
# Or: python -m agent_memory.hermes_plugin.cli setup
```

### Manual Configuration

Create `~/.hermes/agent_memory.json`:

```json
{
  "db_path": "~/.hermes/agent_memory.db",
  "auto_embed": false,
  "embedding_model": "all-MiniLM-L6-v2",
  "extraction_mode": "rule",
  "security_level": "medium"
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_MEMORY_DB_PATH` | Path to SQLite database | `~/.hermes/agent_memory.db` |
| `AGENT_MEMORY_AUTO_EMBED` | Enable vector embeddings | `false` |
| `AGENT_MEMORY_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `AGENT_MEMORY_EXTRACTION_MODE` | Extraction mode (rule/llm/hybrid) | `rule` |
| `AGENT_MEMORY_SECURITY_LEVEL` | Security level | `medium` |

## CLI Commands (`amt-hermes`)

### Add a Memory

```bash
amt-hermes add "User prefers Python over JavaScript"
amt-hermes add "Works at TechCorp" --tags work --tags professional
amt-hermes add "Likes dark mode" --confidence 0.95 --source user_stated
```

### Search Memories

```bash
amt-hermes search "programming preferences"
amt-hermes search "work" --tag professional
amt-hermes search "dark mode" --limit 5 --json
```

### List Memories

```bash
amt-hermes list
amt-hermes list --limit 50
amt-hermes list --json
```

### Plugin Management

```bash
amt-hermes install   # Install plugin to Hermes
amt-hermes uninstall # Remove plugin from Hermes
amt-hermes setup     # Interactive configuration
amt-hermes status    # Show plugin status
```

## Agent Tools

Once installed, Hermes gets these tools:

### memory_add

Store a new memory or fact.

```
Parameters:
- content (required): The memory to store
- tags: Optional list of tags for categorization
- confidence: Confidence level 0.0-1.0 (default: 0.9)
- source: Source of this memory
```

### memory_query

Search stored memories.

```
Parameters:
- query (required): What to search for
- mode: Search mode - fts, vector, hybrid, or auto (default: auto)
- limit: Maximum results (default: 10)
- tag: Filter by tag
```

### memory_extract

Extract structured memories from text.

```
Parameters:
- text (required): Text to extract memories from
- domains: Filter to specific domains
- auto_store: Whether to automatically store extracted memories (default: true)
```

### memory_compress

Compress conversation history.

```
Parameters:
- messages (required): Array of messages to compress
- max_tokens: Target token budget (default: 4000)
- mode: Compression mode - aggressive, balanced, conservative, lossless
```

### memory_profile

Get a summary of stored memories.

```
Parameters:
- domain: Filter to specific domain
- limit: Maximum memories per domain (default: 5)
```

## Cognitive Domains

The extraction system categorizes memories into six domains:

1. **Biography**: Personal details, identity, background
2. **Preferences**: Likes, dislikes, choices, styles
3. **Work**: Projects, skills, professional context
4. **Social**: Relationships, contacts, connections
5. **Temporal**: Schedules, deadlines, time-based info
6. **Procedural**: Workflows, processes, how-tos

## Security Levels

| Level | Description |
|-------|-------------|
| `minimal` | Basic validation only |
| `low` | Light security checks |
| `medium` | Standard validation (recommended) |
| `high` | Strict validation with source requirements |
| `paranoid` | Maximum security, blocks unknown sources |

## Context Auto-Injection

The plugin automatically retrieves and injects relevant memories into the conversation context. Configure this behavior in `agent_memory.json`:

```json
{
  "injection": {
    "enabled": true,
    "frequency": "every-turn",
    "max_memories": 5,
    "min_score": 0.3
  }
}
```

Injection frequencies:
- `every-turn`: Inject before each turn
- `first-turn`: Only on the first turn of a session
- `on-demand`: Only when explicitly requested

## Example Usage

Once installed, the agent can use memory tools naturally:

```
User: Remember that I prefer Python over JavaScript

Agent: I'll store that preference.
[Uses memory_add with content="User prefers Python over JavaScript" tags=["preferences", "programming"]]

User: What programming languages do I like?

Agent: Let me check my memories...
[Uses memory_query with query="programming language preferences"]
Based on what I remember, you prefer Python over JavaScript.
```

## Troubleshooting

### Plugin not loading

1. Check if enabled in config:
   ```bash
   grep -A5 "plugins:" ~/.hermes/config.yaml
   ```

2. Check for import errors:
   ```bash
   python -c "from agent_memory.hermes_plugin import AgentMemoryToolkitProvider"
   ```

3. Verify installation:
   ```bash
   amt-hermes status
   ```

### Database not found

The database is created automatically on first use. Ensure the directory is writable:
```bash
mkdir -p ~/.hermes
touch ~/.hermes/agent_memory.db
```

### Vector search not working

Install sentence-transformers:
```bash
pip install sentence-transformers
```

Then enable auto_embed in config:
```json
{
  "auto_embed": true
}
```

### Click not installed

The CLI requires Click:
```bash
pip install click
```

## API Reference

### AgentMemoryToolkitProvider

The main memory provider class that integrates with Hermes Agent.

```python
from agent_memory.hermes_plugin import AgentMemoryToolkitProvider

provider = AgentMemoryToolkitProvider()
provider.initialize(session_id="my-session")

# Get tool schemas
schemas = provider.get_tool_schemas()

# Handle a tool call
result = provider.handle_tool_call("memory_add", {
    "content": "User prefers dark mode"
})

# Shutdown
provider.shutdown()
```

### ContextInjector

Automatic context injection for relevant memories.

```python
from agent_memory.hermes_plugin.context_injection import ContextInjector
from agent_memory import MemoryStore

store = MemoryStore("memories.db")
injector = ContextInjector(store)

# Queue prefetch for next turn
injector.queue_prefetch("What are the user's preferences?")

# Get prefetched context
context = injector.get_context()
```

### register()

Plugin registration function called by Hermes:

```python
from agent_memory.hermes_plugin import register

# Called by Hermes plugin loader
register(hermes_context)
```

## Files

- `plugin.yaml` - Plugin manifest with tools, hooks, and configuration schema
- `__init__.py` - Main plugin code with AgentMemoryToolkitProvider
- `context_injection.py` - Automatic context injection utilities
- `cli.py` - Click-based CLI commands (amt-hermes)
- `tests/` - Plugin tests

## Contributing

See the main [Agent Memory Toolkit README](../../../README.md) for contribution guidelines.

## License

MIT License - see [LICENSE](../../../LICENSE) for details.
