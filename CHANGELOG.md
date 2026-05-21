# Changelog

All notable changes to Agent Memory Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-15

### Added

- **Core Memory Store** - SQLite-backed persistent storage with FTS5 full-text search
  - Version control with Git-like commits and branches
  - Optional vector embeddings for semantic search
  - Hybrid search combining FTS5 and vector similarity

- **Memory Extraction** - Extract structured facts from unstructured text
  - Rule-based extractor for fast, deterministic extraction
  - LLM-based extractor for complex understanding
  - Hybrid mode combining both approaches
  - 6 cognitive domains: biography, preferences, work, social, temporal, procedural

- **Security Guard** - Protect against memory poisoning attacks
  - Content validation with poison detection
  - Confidence scoring for memory trustworthiness
  - Source tracking and audit trails
  - Configurable security levels (minimal to paranoid)

- **Context Compression** - Fit more context into token budgets
  - Token-aware compression with tiktoken
  - Importance-based ranking to preserve critical information
  - Multiple strategies: truncate, summarize, selective
  - Tiered compression for gradual reduction

- **Team Collaboration** - Multi-agent memory sharing
  - Git-like branching and merging
  - Conflict detection and resolution
  - Filesystem-based sync protocol
  - Role-based access control

- **MCP Server** - Model Context Protocol integration
  - Full CRUD operations for memories
  - Memory extraction tool
  - Security validation tool
  - Context compression tool
  - Works with Claude Desktop, Cursor, and other MCP clients

- **Hermes Plugin** - Native Hermes Agent integration
  - Context injection for agent sessions
  - CLI commands for memory management
  - Automatic memory extraction from conversations

- **REST API** - HTTP API for external integrations
  - FastAPI-based with OpenAPI docs
  - JWT authentication
  - All memory operations exposed

- **Dashboard** - Web UI for memory analytics
  - Memory browser and search
  - Usage statistics and trends
  - Security audit viewer

- **Integrations**
  - LangChain memory adapter
  - LlamaIndex storage adapter

- **CLI Tool** - Command-line interface
  - `amt add` - Add memories
  - `amt search` - Search memories
  - `amt list` - List memories
  - `amt export/import` - Backup and restore

- **Deployment**
  - Docker and docker-compose support
  - Kubernetes manifests (deployment, service, ingress, HPA)
  - Prometheus metrics endpoint

### Technical Details

- Python 3.10+ required
- SQLite 3.35+ for FTS5 support
- 106 tests with full coverage of core modules
- Type hints throughout
- Comprehensive API documentation

## [Unreleased]

### Planned
- PostgreSQL backend option
- Redis caching layer
- Webhook notifications
- Memory expiration policies
- Multi-tenant support
