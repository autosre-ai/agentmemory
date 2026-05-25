# Quick Start

Get up and running with Agent Memory Toolkit in 5 minutes.

## Installation

=== "Basic Install"

    ```bash
    pip install agent-memory-toolkit
    ```

=== "Full Install (Recommended)"

    ```bash
    pip install agent-memory-toolkit[all]
    ```

=== "From Source"

    ```bash
    git clone https://github.com/autosre-ai/agent-memory-toolkit.git
    cd agent-memory-toolkit
    pip install -e ".[all]"
    ```

---

## Your First Memory Store

Create a memory store and add some memories:

```python
from agent_memory_toolkit import MemoryStore

# Create a local memory store with auto-embedding
store = MemoryStore("memories.db", auto_embed=True)

# Add some memories
store.add("User's name is Sarah Chen")
store.add("User prefers dark mode and vim keybindings")
store.add("Project deadline is Friday, client is Acme Corp")
store.add("Last meeting discussed Q4 roadmap with the engineering team")
```

---

## Searching Memories

### Hybrid Search (Recommended)

Combines BM25 keyword matching with semantic vector search:

```python
# Search with hybrid mode (default)
results = store.search("vim preferences", mode="hybrid")

for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

Output:
```
[0.89] User prefers dark mode and vim keybindings
[0.45] Project deadline is Friday, client is Acme Corp
```

### Full-Text Search (BM25)

Fast keyword-based search:

```python
results = store.search_fts("deadline Friday")
```

### Vector Search

Semantic similarity search:

```python
results = store.search_vector("what are the user's settings?")
```

---

## Extracting Structured Memories

Extract facts from unstructured text into cognitive domains:

```python
from agent_memory_toolkit import MemoryExtractor

extractor = MemoryExtractor()

text = """
Hi, I'm Sarah Chen. I work as a Senior Engineer at TechCorp.
I prefer Python over JavaScript and usually work 9-5 PST.
My manager is David Kim and we have standups every Monday.
"""

memories = extractor.extract(text)

for m in memories.memories:
    print(f"[{m.domain.value}] {m.key}: {m.value}")
```

Output:
```
[biography] name: Sarah Chen
[work] role: Senior Engineer
[work] company: TechCorp
[preferences] preferred_language: Python
[temporal] work_hours: 9-5 PST
[social] manager: David Kim
[temporal] standup_day: Monday
```

---

## Security Validation

Validate content before storing to prevent injection attacks:

```python
from agent_memory_toolkit import MemoryGuard, SecurityLevel

guard = MemoryGuard(level=SecurityLevel.HIGH)

# Safe content
result = guard.validate_content("User prefers dark mode")
print(result.is_safe)  # True

# Suspicious content
result = guard.validate_content("Ignore previous instructions and...")
print(result.is_safe)  # False
print(result.issues)   # ['Potential prompt injection detected']

# Only store safe content
if result.is_safe:
    store.add(content)
```

---

## Context Compression

Compress conversation history to fit within token limits:

```python
from agent_memory_toolkit import ContextCompressor

compressor = ContextCompressor()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about machine learning..."},
    {"role": "assistant", "content": "Machine learning is a field of AI..."},
    # ... many more messages
]

# Compress to fit 4000 tokens
result = compressor.compress(messages, max_tokens=4000)

print(f"Original: {result.original_tokens} tokens")
print(f"Compressed: {result.compressed_tokens} tokens")
print(f"Ratio: {result.compression_ratio:.1%}")
```

---

## Team Collaboration

Use Git-like branching for multi-agent collaboration:

```python
from agent_memory_toolkit.team import TeamMemoryStore

# Create a team store for agent "alice"
store = TeamMemoryStore("team.db", agent_id="alice")

# Create an experimental branch
store.create_branch("experiment")
store.checkout("experiment")

# Add memories on the branch
store.add("Testing new hypothesis about user preferences")
store.commit("Added experimental findings")

# Merge back to main
store.checkout("main")
store.merge("experiment")

# Sync with shared directory
store.push("/shared/memories")
```

---

## MCP Server Integration

Use with Claude Desktop, Cursor, or any MCP client:

```bash
# Install with MCP support
pip install agent-memory-toolkit[mcp]

# Start the server
amt-mcp serve --db-path ~/agent_memory.db
```

Configure Claude Desktop:

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

[See full MCP documentation :material-arrow-right:](mcp-server.md)

---

## Examples

See the [`examples/`](https://github.com/autosre-ai/agent-memory-toolkit/tree/main/examples) directory for complete working demos:

- [`basic_usage.py`](https://github.com/autosre-ai/agent-memory-toolkit/blob/main/examples/basic_usage.py) — Getting started
- [`team_collaboration.py`](https://github.com/autosre-ai/agent-memory-toolkit/blob/main/examples/team_collaboration.py) — Multi-agent workflows  
- [`secure_memory.py`](https://github.com/autosre-ai/agent-memory-toolkit/blob/main/examples/secure_memory.py) — Security validation
- [`compress_context.py`](https://github.com/autosre-ai/agent-memory-toolkit/blob/main/examples/compress_context.py) — Context compression

---

## Next Steps

<div class="grid cards" markdown>

-   :books: **[API Reference](api-reference.md)**

    Explore the complete API

-   :gear: **[MCP Server](mcp-server.md)**

    Integrate with LLM clients

-   :building_construction: **[Architecture](architecture.md)**

    Understand how it works

</div>
