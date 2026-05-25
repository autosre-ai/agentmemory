<div align="center">

# 🧠 agent-memory-toolkit

**Hybrid retrieval memory for AI agents that actually remembers.**

[![GitHub stars](https://img.shields.io/github/stars/autosre-ai/agent-memory-toolkit?style=social)](https://github.com/autosre-ai/agent-memory-toolkit)
[![CI](https://img.shields.io/github/actions/workflow/status/autosre-ai/agent-memory-toolkit/ci.yml?branch=main&label=CI)](https://github.com/autosre-ai/agent-memory-toolkit/actions)
[![codecov](https://codecov.io/gh/autosre-ai/agent-memory-toolkit/branch/main/graph/badge.svg)](https://codecov.io/gh/autosre-ai/agent-memory-toolkit)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://autosre-ai.github.io/agent-memory-toolkit/)
[![PyPI](https://img.shields.io/pypi/v/agent-memory-toolkit?color=blue)](https://pypi.org/project/agent-memory-toolkit/)
[![Downloads](https://img.shields.io/pypi/dm/agent-memory-toolkit)](https://pypi.org/project/agent-memory-toolkit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple)](https://modelcontextprotocol.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**BM25 + Vectors + Knowledge Graph** · **RRF Fusion** · **Ebbinghaus Decay** · **Local-First**

[Features](#-features) · [Install](#-install) · [Quick Start](#-quick-start) · [MCP Server](#-mcp-server) · [REST API](#-rest-api) · [Benchmarks](#-benchmarks) · [Docs](https://autosre-ai.github.io/agent-memory-toolkit/)

</div>

---

## 🎯 Why agent-memory-toolkit?

Most agent memory is just "dump everything in a vector DB and pray." That doesn't scale.

**agent-memory-toolkit** uses **hybrid retrieval**:
- 🔍 **BM25** for exact keyword matches
- 🧬 **Vector search** for semantic similarity  
- 🕸️ **Knowledge graph** for relational context
- ⚡ **RRF fusion** to combine results intelligently
- 📉 **Ebbinghaus decay** so recent memories surface naturally

> **95.2% R@5 on LongMemEval-S** — state-of-the-art recall for long-term agent memory

---

## 🔐 Local-First. Your Data Stays Yours.

No cloud. No API calls for storage. Everything runs on SQLite.

- ✅ Works offline
- ✅ GDPR-friendly
- ✅ Airgapped environments
- ✅ Full control over your data

---

## ✨ Features

| | Feature | Description |
|---|---------|-------------|
| 🔍 | **Hybrid Retrieval** | BM25 + vectors + knowledge graph with RRF fusion |
| 📉 | **Ebbinghaus Decay** | Recent memories surface first, old ones fade naturally |
| 📝 | **Structured Extraction** | 6 cognitive domains (bio, preferences, work, social, temporal, procedural) |
| 🔒 | **Security Guard** | Poison detection, confidence scoring, source validation |
| 📦 | **Smart Compression** | Token-aware context compression for LLM context windows |
| 👥 | **Team Collaboration** | Git-like branching, merging, and sync for multi-agent systems |
| 🔄 | **Version Control** | Full history tracking with commits and rollback |
| 🔌 | **MCP Server** | Works with Claude Desktop, Cursor, and other MCP clients |
| 🌐 | **REST API** | HTTP API with JWT auth for external integrations |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         AGENT MEMORY TOOLKIT                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                           Integration Layer                              │  │
│  │                                                                          │  │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │   │   MCP   │  │  REST   │  │LangChain│  │LlamaIndex│  │   Hermes    │  │  │
│  │   │ Server  │  │  API    │  │ Adapter │  │ Adapter  │  │   Plugin    │  │  │
│  │   └─────────┘  └─────────┘  └─────────┘  └──────────┘  └─────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                            Core Modules                                  │  │
│  │                                                                          │  │
│  │  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌───────────┐ │  │
│  │  │ Extraction  │   │   Storage    │   │  Compression  │   │  Security │ │  │
│  │  │   Module    │   │    Store     │   │    Engine     │   │   Guard   │ │  │
│  │  │             │   │              │   │               │   │           │ │  │
│  │  │ • Rule-based│   │ • SQLite     │   │ • Token aware │   │ • Poison  │ │  │
│  │  │ • LLM-based │   │ • FTS5/BM25  │   │ • Importance  │   │  detection│ │  │
│  │  │ • Hybrid    │   │ • Vectors    │   │   ranking     │   │ • Source  │ │  │
│  │  │ • 6 domains │   │ • RRF Fusion │   │ • Strategies  │   │  tracking │ │  │
│  │  └─────────────┘   └──────────────┘   └───────────────┘   └───────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                     │                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        Team Collaboration Layer                          │  │
│  │                                                                          │  │
│  │   ┌───────────────┐   ┌─────────────────┐   ┌─────────────────────────┐ │  │
│  │   │  Git-like     │   │    Conflict     │   │    Filesystem Sync     │ │  │
│  │   │  Branching    │   │   Resolution    │   │     & Access Control   │ │  │
│  │   └───────────────┘   └─────────────────┘   └─────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Install

```bash
pip install agent-memory-toolkit
```

With all features:
```bash
pip install agent-memory-toolkit[all]
```

Optional extras:
```bash
pip install agent-memory-toolkit[mcp]        # MCP server support
pip install agent-memory-toolkit[api]        # REST API server
pip install agent-memory-toolkit[embeddings] # Vector embeddings
pip install agent-memory-toolkit[langchain]  # LangChain integration
pip install agent-memory-toolkit[llamaindex] # LlamaIndex integration
```

---

## 🚀 Quick Start

```python
from agent_memory_toolkit import MemoryStore

# Create a local memory store
store = MemoryStore("memories.db", auto_embed=True)

# Add memories
store.add("User prefers dark mode and vim keybindings")
store.add("Project deadline is Friday, client is Acme Corp")
store.add("Last meeting discussed Q4 roadmap")

# Hybrid search (BM25 + vectors + recency decay)
results = store.search("vim preferences", mode="hybrid")

for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

### Extract Structured Memories

```python
from agent_memory_toolkit import MemoryExtractor

extractor = MemoryExtractor()

text = """
Hi, I'm Sarah Chen. I work as a Senior Engineer at TechCorp.
I prefer Python over JavaScript and usually work 9-5 PST.
"""

memories = extractor.extract(text)

for m in memories.memories:
    print(f"[{m.domain.value}] {m.key}: {m.value}")
# [biography] name: Sarah Chen
# [work] role: Senior Engineer
# [preferences] preferred_language: Python
```

---

## 🔌 MCP Server

Use Agent Memory Toolkit with Claude Desktop, Cursor, and other MCP-compatible clients.

### Quick Setup

```bash
# Install with MCP support
pip install agent-memory-toolkit[mcp]

# Generate config for your client
amt-mcp config claude  # or: amt-mcp config cursor
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `memory_add` | Add a new memory |
| `memory_query` | Search memories (hybrid retrieval) |
| `memory_get` | Get a memory by ID |
| `memory_update` | Update an existing memory |
| `memory_delete` | Delete a memory |
| `extract_memories` | Extract structured facts from text |
| `guard_check` | Validate content for security issues |
| `compress_context` | Compress conversation to fit token budget |

See [MCP Server Documentation](docs/mcp-server.md) for detailed usage.

---

## 🌐 REST API

HTTP API for external integrations with JWT authentication.

### Quick Start

```bash
# Install with API support
pip install agent-memory-toolkit[api]

# Start the server
amt api serve --port 8000

# Or with Docker
docker run -p 8000:8000 autosre-ai/agent-memory-toolkit:latest
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/memories` | Add a new memory |
| `GET` | `/api/v1/memories` | List memories |
| `GET` | `/api/v1/memories/{id}` | Get memory by ID |
| `PUT` | `/api/v1/memories/{id}` | Update memory |
| `DELETE` | `/api/v1/memories/{id}` | Delete memory |
| `POST` | `/api/v1/search` | Search memories |
| `POST` | `/api/v1/extract` | Extract memories from text |
| `POST` | `/api/v1/compress` | Compress context |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

### Example: Add a Memory

```bash
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "tags": ["preferences"]}'
```

### Example: Search Memories

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "user preferences", "mode": "hybrid", "limit": 10}'
```

OpenAPI documentation available at `http://localhost:8000/docs`.

---

## 📖 API Reference

### MemoryStore

```python
from agent_memory_toolkit import MemoryStore

store = MemoryStore(
    db_path="memories.db",
    auto_embed=True,
    embedding_model="all-MiniLM-L6-v2"
)

# Core operations
store.add(content, metadata=None)
store.get(memory_id)
store.update(memory_id, content=None, metadata=None)
store.delete(memory_id)

# Search modes
store.search(query, mode="hybrid")  # BM25 + vectors + decay
store.search_fts(query)             # BM25 only
store.search_vector(query)          # Vectors only
```

### MemoryExtractor

```python
from agent_memory_toolkit import MemoryExtractor, CognitiveDomain

extractor = MemoryExtractor(mode="rule")  # or "llm", "hybrid"
result = extractor.extract(text)
```

### MemoryGuard

```python
from agent_memory_toolkit import MemoryGuard, SecurityLevel

guard = MemoryGuard(level=SecurityLevel.HIGH)
result = guard.validate_content(content)

if result.is_safe:
    store.add(content)
```

### TeamMemoryStore

```python
from agent_memory_toolkit.team import TeamMemoryStore

store = TeamMemoryStore("team.db", agent_id="alice")

# Git-like operations
store.create_branch("experiment")
store.checkout("experiment")
store.commit("Added new findings")
store.push("/shared/memories")
store.pull("/shared/memories")
```

---

## 📊 Benchmarks

| Metric | agent-memory-toolkit | Vector-only | BM25-only |
|--------|-------------|-------------|-----------|
| **R@5 (LongMemEval-S)** | **95.2%** | 78.4% | 71.2% |
| **Latency (p50)** | 8ms | 5ms | 0.5ms |
| **Memory Usage** | 120MB | 200MB | 40MB |

Hybrid retrieval with RRF fusion significantly outperforms single-strategy approaches.

---

## ⚡ Performance

| Operation | Time |
|-----------|------|
| Rule-based extraction | ~1ms per 1KB |
| BM25 search (FTS5) | ~0.5ms |
| Vector search | ~5ms |
| Hybrid search | ~8ms |
| Security validation | ~2ms |

---

## 📂 Examples

See [`examples/`](examples/) for working demos:

- [`basic_usage.py`](examples/basic_usage.py) — Getting started
- [`mcp_quickstart.py`](examples/mcp_quickstart.py) — MCP server integration
- [`team_collaboration.py`](examples/team_collaboration.py) — Multi-agent workflows
- [`secure_memory.py`](examples/secure_memory.py) — Security validation
- [`compress_context.py`](examples/compress_context.py) — Context compression
- [`langchain_example.py`](examples/langchain_example.py) — LangChain integration
- [`llamaindex_example.py`](examples/llamaindex_example.py) — LlamaIndex integration

---

## 🧪 Testing

```bash
pytest
pytest --cov=agent_memory_toolkit
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

**[⭐ Star us on GitHub](https://github.com/autosre-ai/agent-memory-toolkit)** — it helps!

Built with ❤️ by [autosre.ai](https://autosre.ai)

</div>
