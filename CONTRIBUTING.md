# Contributing to Agent Memory Toolkit

Thank you for your interest in contributing to Agent Memory Toolkit! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept responsibility for mistakes and learn from them

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- SQLite 3.35+ (for FTS5 support)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/agent-memory-toolkit.git
cd agent-memory-toolkit
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/autosre-ai/agent-memory-toolkit.git
```

## Development Setup

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Development Dependencies

```bash
pip install -e ".[dev,all]"
```

This installs:
- All package dependencies
- Development tools (pytest, black, ruff, mypy)
- All optional features (embeddings, MCP, API, etc.)

### Verify Installation

```bash
# Run tests
pytest

# Run linters
black --check .
ruff check .
mypy agent_memory_toolkit/
```

## Making Changes

### Branch Naming

Create a descriptive branch name:

```bash
git checkout -b feature/add-redis-cache
git checkout -b fix/memory-leak-in-search
git checkout -b docs/update-api-reference
```

Prefixes:
- `feature/` — New features
- `fix/` — Bug fixes
- `docs/` — Documentation changes
- `refactor/` — Code refactoring
- `test/` — Test additions or modifications
- `chore/` — Maintenance tasks

### Commit Messages

Follow conventional commit format:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types:
- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation
- `style` — Formatting (no code change)
- `refactor` — Code restructuring
- `test` — Tests
- `chore` — Maintenance

Examples:
```
feat(mcp): add SSE transport option
fix(store): prevent duplicate entries on concurrent writes
docs(readme): add MCP server section
```

### Keep Changes Focused

- One logical change per commit
- One feature or fix per PR
- Break large changes into smaller PRs

## Pull Request Process

### Before Submitting

1. **Sync with upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks:**
   ```bash
   # Format code
   black .
   
   # Lint
   ruff check . --fix
   
   # Type check
   mypy agent_memory_toolkit/
   
   # Run tests
   pytest --cov=agent_memory_toolkit
   ```

3. **Update documentation** if needed

4. **Add tests** for new features

### Submitting

1. Push your branch to your fork
2. Open a PR against `main`
3. Fill out the PR template
4. Link related issues

### PR Review Process

- Maintainers will review within 48 hours
- Address feedback promptly
- Keep discussion focused and constructive
- Once approved, a maintainer will merge

## Coding Standards

### Python Style

We use:
- **Black** for code formatting (line length: 88)
- **Ruff** for linting
- **MyPy** for type checking

Configuration is in `pyproject.toml`.

### Type Hints

All public functions must have type hints:

```python
def search(
    self,
    query: str,
    mode: str = "hybrid",
    limit: int = 10,
) -> list[SearchResult]:
    """Search memories using the specified mode."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def extract(self, text: str, source: str | None = None) -> ExtractionResult:
    """Extract structured memories from text.
    
    Args:
        text: The input text to extract memories from.
        source: Optional source identifier for tracking.
    
    Returns:
        ExtractionResult containing extracted memories and metadata.
    
    Raises:
        ExtractionError: If extraction fails.
    
    Example:
        >>> extractor = MemoryExtractor()
        >>> result = extractor.extract("My name is Alice")
        >>> print(result.memories[0].value)
        'Alice'
    """
```

### Error Handling

- Use custom exceptions from `agent_memory_toolkit.exceptions`
- Provide helpful error messages
- Don't catch broad exceptions unless necessary

```python
from agent_memory_toolkit.exceptions import MemoryNotFoundError

def get(self, memory_id: str) -> Memory:
    memory = self._fetch(memory_id)
    if memory is None:
        raise MemoryNotFoundError(f"Memory not found: {memory_id}")
    return memory
```

## Testing Guidelines

### Test Structure

Tests are in the `tests/` directory, mirroring the source structure:

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_memory_store.py
├── test_extraction.py
├── test_security.py
└── test_compression.py
```

### Writing Tests

Use pytest with clear, descriptive test names:

```python
import pytest
from agent_memory_toolkit import MemoryStore

class TestMemoryStore:
    def test_add_returns_memory_id(self, store: MemoryStore):
        """Adding a memory returns a valid ID."""
        memory_id = store.add("Test content")
        assert memory_id is not None
        assert len(memory_id) == 36  # UUID format
    
    def test_search_hybrid_combines_fts_and_vector(self, store: MemoryStore):
        """Hybrid search uses both FTS5 and vector similarity."""
        store.add("Python is a programming language")
        store.add("Snakes are reptiles")
        
        results = store.search("programming Python", mode="hybrid")
        
        assert len(results) > 0
        assert "Python" in results[0].memory.content
    
    def test_delete_nonexistent_raises_error(self, store: MemoryStore):
        """Deleting non-existent memory raises MemoryNotFoundError."""
        with pytest.raises(MemoryNotFoundError):
            store.delete("nonexistent-id")
```

### Fixtures

Define reusable fixtures in `conftest.py`:

```python
import pytest
from agent_memory_toolkit import MemoryStore

@pytest.fixture
def store(tmp_path):
    """Create a temporary memory store for testing."""
    db_path = tmp_path / "test.db"
    return MemoryStore(str(db_path), auto_embed=False)

@pytest.fixture
def populated_store(store):
    """Store with sample data."""
    store.add("User prefers dark mode")
    store.add("Project deadline is Friday")
    return store
```

### Coverage Requirements

- Maintain >80% test coverage
- New features require tests
- Bug fixes should include regression tests

Run coverage:
```bash
pytest --cov=agent_memory_toolkit --cov-report=html
open htmlcov/index.html
```

## Documentation

### Types of Documentation

1. **Docstrings** — In-code API documentation
2. **README.md** — Quick start and overview
3. **docs/** — Detailed MkDocs documentation
4. **CHANGELOG.md** — Version history

### Building Docs Locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Visit `http://localhost:8000` to preview.

### Documentation Standards

- Keep examples working and tested
- Use clear, concise language
- Include code examples
- Update when APIs change

## Issue Reporting

### Bug Reports

Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages/tracebacks
- Minimal code example

### Feature Requests

Include:
- Use case description
- Proposed API or behavior
- Alternatives considered
- Willingness to implement

### Security Issues

**Do not open public issues for security vulnerabilities.**

Email security@autosre.ai with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

## Questions?

- Open a [GitHub Discussion](https://github.com/autosre-ai/agent-memory-toolkit/discussions)
- Check existing issues and docs first
- Be specific about your question

---

Thank you for contributing to Agent Memory Toolkit! 🧠
