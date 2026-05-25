# Command Line Interface

Agent Memory Toolkit provides a powerful CLI for managing agent memories directly from your terminal.

## Installation

The CLI is included with the base package:

```bash
pip install agent-memory-toolkit
```

Two commands are available:

- `amt` - Main CLI (recommended)
- `agent-memory-toolkit` - Full name alias

## Quick Start

```bash
# Add a memory
amt add "User prefers dark mode and vim keybindings"

# Search memories
amt search "preferences"

# List recent memories
amt list --limit 10

# Export to JSON
amt export memories.json
```

---

## Global Options

These options work with all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--db` | `-d` | Path to memory database (default: `agent_memory.db`) |
| `--verbose` | `-v` | Enable verbose output |
| `--json` | `-j` | Output results as JSON |
| `--help` | | Show help message |
| `--version` | | Show version |

**Environment Variables:**

- `AMT_DB_PATH` - Default database path

**Examples:**

```bash
# Use custom database
amt --db ~/my_memories.db search "query"

# JSON output for scripting
amt --json list | jq '.[] | .content'

# Verbose mode for debugging
amt -v add "New memory"
```

---

## Commands

### add

Add a new memory to the store.

```bash
amt add "Memory content" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--tags` | `-t` | Tags (can be repeated) |
| `--source` | `-s` | Source identifier |
| `--confidence` | `-c` | Confidence score (0.0-1.0) |

**Examples:**

```bash
# Simple memory
amt add "User's name is Sarah"

# With tags
amt add "API key is xyz" --tags secret --tags config

# With metadata
amt add "Meeting notes from standup" --source meeting --confidence 0.9
```

---

### search

Search memories using hybrid, full-text, or vector search.

```bash
amt search "query" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-n` | Maximum results (default: 10) |
| `--mode` | `-m` | Search mode: `hybrid`, `fts`, `vector` |
| `--threshold` | `-t` | Minimum score threshold |
| `--tag` | | Filter by tag (can be repeated) |
| `--source` | `-s` | Filter by source |
| `--rerank` | `-r` | Enable cross-encoder reranking |
| `--branch` | `-b` | Search on specific branch |

**Search Modes:**

- `hybrid` (default): Combines BM25 keyword + vector semantic search
- `fts`: Fast full-text search with BM25 ranking
- `vector`: Semantic similarity using embeddings

**Examples:**

```bash
# Basic search
amt search "user preferences"

# Full-text only (faster)
amt search "API configuration" --mode fts

# Semantic search with limit
amt search "similar concepts" --mode vector --limit 5

# Filter by tags
amt search "meetings" --tag work --tag important

# High accuracy with reranking
amt search "deployment process" --rerank
```

---

### list

List memories with pagination.

```bash
amt list [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-n` | Maximum memories (default: 20) |
| `--offset` | `-o` | Offset for pagination |
| `--branch` | `-b` | List from specific branch |

**Examples:**

```bash
# List recent 20 memories
amt list

# Paginate through results
amt list --limit 50 --offset 100

# List from a branch
amt list --branch experiment
```

---

### memory

Memory CRUD operations subcommand group.

#### memory add

Same as top-level `add` command.

#### memory get

Get a specific memory by ID.

```bash
amt memory get <memory_id>
```

**Examples:**

```bash
amt memory get abc123
amt memory get abc123def456789  # Full ID
```

#### memory update

Update an existing memory.

```bash
amt memory update <memory_id> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--content` | `-c` | New content |
| `--tags` | `-t` | New tags (replaces existing) |
| `--add-tag` | | Add tag without replacing |

**Examples:**

```bash
# Update content
amt memory update abc123 --content "Updated content"

# Replace tags
amt memory update abc123 --tags new-tag

# Add a tag
amt memory update abc123 --add-tag extra-tag
```

#### memory delete

Delete a memory.

```bash
amt memory delete <memory_id> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Skip confirmation |

**Examples:**

```bash
amt memory delete abc123
amt memory delete abc123 --force
```

---

### store

Store management commands for branching and version control.

#### store branch

Manage branches.

```bash
amt store branch [name] [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--delete` | `-d` | Delete the branch |
| `--list` | `-l` | List all branches |

**Examples:**

```bash
# List branches
amt store branch

# Create a branch
amt store branch experiment

# Delete a branch
amt store branch -d old-branch
```

#### store checkout

Switch to a different branch.

```bash
amt store checkout <branch_name>
```

**Examples:**

```bash
amt store checkout main
amt store checkout experiment
```

#### store commit

Commit changes on the current branch.

```bash
amt store commit "message"
```

**Examples:**

```bash
amt store commit "Added user preferences"
amt store commit "Updated meeting notes"
```

#### store merge

Merge a branch into the current branch.

```bash
amt store merge <source_branch> [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--no-commit` | Don't auto-commit after merge |

**Examples:**

```bash
amt store merge experiment
amt store merge feature --no-commit
```

#### store log

Show commit history.

```bash
amt store log [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-n` | Number of commits (default: 10) |
| `--branch` | `-b` | Branch to show |

**Examples:**

```bash
amt store log
amt store log --limit 20
amt store log --branch experiment
```

#### store rollback

Rollback to a previous commit.

```bash
amt store rollback <commit_id> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Skip confirmation |

**Examples:**

```bash
amt store rollback abc123
amt store rollback abc123 --force
```

#### store status

Show current store status.

```bash
amt store status
```

---

### export

Export memories to a file.

```bash
amt export <output> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--format` | `-f` | Format: `json`, `jsonl`, `csv` |
| `--branch` | `-b` | Branch to export |
| `--query` | `-q` | Only export matching memories |
| `--include-metadata` | `-m` | Include metadata (default: true) |

**Formats:**

- `json` (default): Single JSON array file
- `jsonl`: JSON Lines (one object per line)
- `csv`: Comma-separated values

**Examples:**

```bash
# Export all to JSON
amt export memories.json

# Export to JSONL
amt export data.jsonl --format jsonl

# Export specific branch
amt export backup.json --branch experiment

# Export search results
amt export filtered.json --query "preferences"
```

---

### import

Import memories from a file.

```bash
amt import <input_file> [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--format` | `-f` | Format (auto-detected if not specified) |
| `--merge` | | Skip duplicates |
| `--dry-run` | | Show what would be imported |

**Examples:**

```bash
# Import from JSON
amt import memories.json

# Import with merge
amt import backup.json --merge

# Preview import
amt import data.csv --dry-run
```

---

### info

Show toolkit information and capabilities.

```bash
amt info
```

Displays:

- Version
- Database path
- Available features (vector search, MCP, etc.)
- CLI command overview

---

### stats

Show memory store statistics.

```bash
amt stats
```

Displays:

- Total memory count
- Current branch
- Branch list
- Top tags
- Top sources

---

## Shell Completion

Enable shell completion for faster command entry.

### Bash

```bash
# Add to ~/.bashrc
eval "$(_AMT_COMPLETE=bash_source amt)"
```

### Zsh

```bash
# Add to ~/.zshrc
eval "$(_AMT_COMPLETE=zsh_source amt)"
```

### Fish

```bash
# Add to ~/.config/fish/completions/amt.fish
_AMT_COMPLETE=fish_source amt | source
```

---

## Scripting Examples

### Batch Add Memories

```bash
#!/bin/bash
while IFS= read -r line; do
    amt add "$line" --source batch-import
done < memories.txt
```

### Export and Backup

```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
amt export ~/backups/memories_$DATE.json
```

### Search and Process

```bash
#!/bin/bash
# Find memories and extract IDs
amt --json search "api keys" | jq -r '.[].id' | while read id; do
    amt memory get $id
done
```

### CI/CD Integration

```bash
#!/bin/bash
# Validate memories before deployment
UNSAFE=$(amt --json search "password\|secret\|key" | jq length)
if [ "$UNSAFE" -gt 0 ]; then
    echo "Warning: Found $UNSAFE sensitive memories"
    exit 1
fi
```

---

## Related

- [Quick Start](quickstart.md) - Getting started with the toolkit
- [MCP Server](mcp-server.md) - Integrate with LLM clients
- [API Reference](api-reference.md) - Python API documentation
