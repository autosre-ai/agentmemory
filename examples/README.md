# Agent Memory Toolkit Examples

This directory contains practical examples demonstrating various features and
integrations of the Agent Memory Toolkit.

## Quick Start

Before running any example, install the toolkit and required dependencies:

```bash
# Install the toolkit
pip install agent-memory-toolkit

# For framework integrations
pip install langchain langchain-openai llama-index-core llama-index-llms-openai

# For OpenAI examples
pip install openai
```

## Examples Overview

### Core Functionality

| Example | Description |
|---------|-------------|
| [basic_usage.py](basic_usage.py) | Core features: extraction, storage, search, versioning, branching |
| [compress_context.py](compress_context.py) | Context compression for managing token budgets |
| [secure_memory.py](secure_memory.py) | Security features: encryption, PII detection, access control |
| [team_collaboration.py](team_collaboration.py) | Multi-agent memory sharing and collaboration |

### Framework Integrations

| Example | Description |
|---------|-------------|
| [langchain_integration.py](langchain_integration.py) | LangChain memory backend integration |
| [langchain_example.py](langchain_example.py) | Detailed LangChain examples (conversation, search, chat) |
| [openai_integration.py](openai_integration.py) | Direct OpenAI API with persistent memory |
| [llamaindex_example.py](llamaindex_example.py) | LlamaIndex chat store and vector store |

### MCP Server

| Example | Description |
|---------|-------------|
| [mcp_quickstart.py](mcp_quickstart.py) | Model Context Protocol server for Claude Desktop/Cursor |

## Running Examples

Each example can be run directly:

```bash
# Run basic usage demo
python examples/basic_usage.py

# Run LangChain integration
python examples/langchain_integration.py

# Run OpenAI integration (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-xxx python examples/openai_integration.py
```

## Example Details

### basic_usage.py

Demonstrates the core memory operations:

- **Memory Extraction**: Extract structured memories from unstructured text
- **Persistent Storage**: Store and retrieve memories with SQLite backend
- **Search**: Full-text search and semantic search (with embeddings)
- **Version Control**: Track changes with commits and history
- **Branching**: Create experimental branches for memory exploration
- **Export/Import**: JSON export for backup and sharing

```python
from agent_memory_toolkit import MemoryStore, MemoryExtractor

store = MemoryStore("memories.db")
extractor = MemoryExtractor(mode="rule")

result = extractor.extract("My name is Alice and I work at TechCorp")
for memory in result.memories:
    store.add(f"{memory.key}: {memory.value}")
```

### langchain_integration.py

Drop-in memory backend for LangChain applications:

- **ConversationChain**: Persistent conversation memory
- **Agent Memory**: Long-term memory for LangChain agents
- **Semantic Search**: Find relevant memories with natural language

```python
from agent_memory_toolkit import MemoryStore
from agent_memory_toolkit.integrations.langchain import AgentMemoryToolkitMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

store = MemoryStore("memory.db")
memory = AgentMemoryToolkitMemory(store=store, session_id="user_123")
chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
```

### openai_integration.py

Persistent memory with the OpenAI Python SDK:

- **Chat Completions**: Memory-augmented conversations
- **Function Calling**: Tools that access and update memory
- **Multi-turn Conversations**: Maintain context across API calls

```python
from openai import OpenAI
from agent_memory_toolkit import MemoryStore

client = OpenAI()
store = MemoryStore("memory.db")

# Add relevant memories to context
memories = store.search_fts("user preferences", limit=5)
context = "\n".join(m.memory.content for m in memories)
```

### secure_memory.py

Security features for sensitive data:

- **Encryption**: AES-256 encryption for stored memories
- **PII Detection**: Automatic detection and redaction
- **Access Control**: Permission-based memory access
- **Audit Logging**: Track all memory operations

### team_collaboration.py

Multi-agent memory sharing:

- **Shared Memory Spaces**: Common knowledge base for agent teams
- **Memory Isolation**: Per-agent private memories
- **Sync & Merge**: Coordinate memory updates across agents

### mcp_quickstart.py

Model Context Protocol integration:

- **Claude Desktop**: Configure as an MCP server
- **Cursor IDE**: Memory tools for AI code assistance
- **Custom Clients**: Build MCP-compatible applications

## Environment Variables

Some examples require API keys:

```bash
# Required for OpenAI and LangChain OpenAI examples
export OPENAI_API_KEY="sk-your-key-here"

# Optional: for secure storage examples
export AMT_ENCRYPTION_KEY="your-32-byte-key"
```

## Troubleshooting

### Import Errors

Ensure you have all required packages:

```bash
pip install agent-memory-toolkit[all]
```

### Database Locked

If you see "database is locked", ensure no other process is using the same
database file, or use `:memory:` for testing:

```python
store = MemoryStore(":memory:")  # In-memory database
```

### API Rate Limits

For LLM-based examples, you may hit rate limits. The examples handle this
gracefully and skip API calls when keys aren't set.

## Contributing

We welcome new examples! Please follow the existing style:

1. Include a docstring explaining what the example demonstrates
2. Use print statements to show progress and results
3. Handle missing dependencies gracefully
4. Include cleanup code (close stores, remove temp files)

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.
