# Agent Memory Toolkit - Enhancement Spec

## Current State

The toolkit already has:
- Hybrid retrieval (BM25 + Vectors + Knowledge Graph)
- RRF fusion
- Ebbinghaus decay
- Local-first SQLite storage
- 95.2% R@5 on LongMemEval-S

## Enhancements Needed

### 1. MCP Server Support
- Expose memory operations via Model Context Protocol
- Allow any MCP-compatible agent to use memory
- Tool definitions for store, retrieve, forget

### 2. REST API
- FastAPI server for remote access
- OpenAPI spec auto-generated
- Auth via API keys

### 3. Better Documentation
- GitHub Pages site with full docs
- Interactive examples
- Benchmark comparisons

### 4. Launch Content
- README badges and shields
- Social preview images
- Monthly content calendar

## Project Structure Updates

```
agent_memory_toolkit/
├── api/                    # NEW: REST API
│   ├── __init__.py
│   ├── main.py
│   └── routes.py
├── mcp/                    # NEW: MCP server
│   ├── __init__.py
│   └── server.py
├── docs/                   # NEW: Documentation site
│   ├── index.md
│   ├── quickstart.md
│   └── api-reference.md
└── ... (existing code)
```

## Success Criteria

1. MCP server works with Claude Desktop
2. REST API with OpenAPI docs
3. GitHub Pages site live
4. PyPI package updated
5. Launch content ready
