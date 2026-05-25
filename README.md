<div align="center">

# 🧠 agent-memory-toolkit

**Hybrid retrieval memory for AI agents that actually remembers.**

[![GitHub stars](https://img.shields.io/github/stars/autosre-ai/agent-memory-toolkit?style=social)](https://github.com/autosre-ai/agent-memory-toolkit)
[![CI](https://img.shields.io/github/actions/workflow/status/autosre-ai/agent-memory-toolkit/ci.yml?branch=main&label=CI)](https://github.com/autosre-ai/agent-memory-toolkit/actions)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://autosre-ai.github.io/agent-memory-toolkit/)
[![PyPI](https://img.shields.io/pypi/v/agent-memory-toolkit?color=blue)](https://pypi.org/project/agent-memory-toolkit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**BM25 + Vectors + Knowledge Graph** · **RRF Fusion** · **Ebbinghaus Decay** · **Local-First**

[Features](#-features) · [Install](#-install) · [Quick Start](#-quick-start) · [Benchmarks](#-benchmarks) · [Docs](https://autosre-ai.github.io/agent-memory-toolkit/)

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

---

## 📦 Install

```bash
pip install agent-memory-toolkit
```

With all features:
```bash
pip install agent-memory-toolkit[all]
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

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             AGENTMEMORY                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────────┐  │
│  │ Extraction  │   │   Storage    │   │  Compression  │   │  Security   │  │
│  │   Module    │   │    Store     │   │    Engine     │   │   Guard     │  │
│  │             │   │              │   │               │   │             │  │
│  │ • Rule-based│   │ • SQLite     │   │ • Token aware │   │ • Poison    │  │
│  │ • LLM-based │   │ • FTS5/BM25  │   │ • Importance  │   │   detection │  │
│  │ • Hybrid    │   │ • Vectors    │   │   ranking     │   │ • Confidence│  │
│  │ • 6 domains │   │ • RRF Fusion │   │ • Strategies  │   │   scoring   │  │
│  └─────────────┘   └──────────────┘   └───────────────┘   └─────────────┘  │
│         │                  │                   │                  │        │
│         └──────────────────┴───────────────────┴──────────────────┘        │
│                                    │                                        │
│                        ┌───────────┴───────────┐                           │
│                        │   Team Memory Store   │                           │
│                        │                       │                           │
│                        │ • Git-like branching  │                           │
│                        │ • Conflict resolution │                           │
│                        │ • Filesystem sync     │                           │
│                        │ • Access control      │                           │
│                        └───────────────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
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

## 📂 Examples

See [`examples/`](examples/) for working demos:

- [`basic_usage.py`](examples/basic_usage.py) — Getting started
- [`team_collaboration.py`](examples/team_collaboration.py) — Multi-agent workflows
- [`secure_memory.py`](examples/secure_memory.py) — Security validation
- [`compress_context.py`](examples/compress_context.py) — Context compression

---

## 🧪 Testing

```bash
pytest
pytest --cov=agent_memory_toolkit
```

---

## 🤝 Contributing

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
