# Agent Memory Toolkit

<div class="hero" markdown>

## Hybrid retrieval memory for AI agents that actually remembers.

**BM25 + Vectors + Knowledge Graph** · **RRF Fusion** · **Ebbinghaus Decay** · **Local-First**

[Get Started](quickstart.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/autosre-ai/agent-memory-toolkit){ .md-button }

</div>

---

## Why Agent Memory Toolkit?

Most agent memory is just "dump everything in a vector DB and pray." That doesn't scale.

**Agent Memory Toolkit** uses **hybrid retrieval**:

- :mag: **BM25** for exact keyword matches
- :dna: **Vector search** for semantic similarity  
- :spider_web: **Knowledge graph** for relational context
- :zap: **RRF fusion** to combine results intelligently
- :chart_with_downwards_trend: **Ebbinghaus decay** so recent memories surface naturally

!!! success "State-of-the-Art Performance"
    **95.2% R@5 on LongMemEval-S** — best-in-class recall for long-term agent memory

---

## :lock: Local-First. Your Data Stays Yours.

No cloud. No API calls for storage. Everything runs on SQLite.

<div class="grid cards" markdown>

- :white_check_mark: **Works offline**
- :white_check_mark: **GDPR-friendly**
- :white_check_mark: **Airgapped environments**
- :white_check_mark: **Full control over your data**

</div>

---

## Features

| | Feature | Description |
|---|---------|-------------|
| :mag: | **Hybrid Retrieval** | BM25 + vectors + knowledge graph with RRF fusion |
| :chart_with_downwards_trend: | **Ebbinghaus Decay** | Recent memories surface first, old ones fade naturally |
| :memo: | **Structured Extraction** | 6 cognitive domains (bio, preferences, work, social, temporal, procedural) |
| :shield: | **Security Guard** | Poison detection, confidence scoring, source validation |
| :package: | **Smart Compression** | Token-aware context compression for LLM context windows |
| :busts_in_silhouette: | **Team Collaboration** | Git-like branching, merging, and sync for multi-agent systems |
| :arrows_counterclockwise: | **Version Control** | Full history tracking with commits and rollback |

---

## Quick Install

```bash
pip install agent-memory-toolkit
```

With all features:

```bash
pip install agent-memory-toolkit[all]
```

---

## Basic Usage

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

---

## Performance

| Operation | Time |
|-----------|------|
| Rule-based extraction | ~1ms per 1KB |
| BM25 search (FTS5) | ~0.5ms |
| Vector search | ~5ms |
| Hybrid search | ~8ms |
| Security validation | ~2ms |

---

## Architecture Overview

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

[Learn more about the architecture :material-arrow-right:](architecture.md)

---

## Next Steps

<div class="grid cards" markdown>

-   :rocket: **[Quick Start](quickstart.md)**

    Get up and running in 5 minutes

-   :books: **[API Reference](api-reference.md)**

    Complete API documentation

-   :gear: **[MCP Server](mcp-server.md)**

    Use with Claude Desktop, Cursor, and more

-   :chart_with_upwards_trend: **[Benchmarks](benchmarks.md)**

    Performance results and comparisons

</div>
