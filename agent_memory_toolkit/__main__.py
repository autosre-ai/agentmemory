"""Entry point for running agent_memory_toolkit as a module.

This allows running the CLI with:
    python -m agent_memory_toolkit --help
    python -m agent_memory_toolkit memory add "Some content"
"""

from agent_memory_toolkit.cli import main

if __name__ == "__main__":
    main()
