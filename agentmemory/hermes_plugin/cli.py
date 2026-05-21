"""CLI commands for Agent Memory Toolkit Hermes plugin.

Provides the `amt-hermes` command group for memory management:
- amt-hermes add <content>   - Add a memory
- amt-hermes search <query>  - Search memories
- amt-hermes list            - List recent memories
- amt-hermes install         - Install plugin into Hermes
- amt-hermes setup           - Interactive setup wizard
- amt-hermes status          - Show plugin status
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import click


def get_hermes_home() -> Path:
    """Get the Hermes home directory."""
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home)
    return Path.home() / ".hermes"


def get_plugin_source_dir() -> Path:
    """Get the source directory of this plugin."""
    return Path(__file__).parent


def _get_memory_store():
    """Get initialized memory store, or None if not available."""
    try:
        from agentmemory import MemoryStore
        
        hermes_home = get_hermes_home()
        config_path = hermes_home / "agent_memory.json"
        
        # Load config
        config = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
            except Exception:
                pass
        
        db_path = config.get("db_path", str(hermes_home / "agent_memory.db"))
        if db_path.startswith("~"):
            db_path = str(Path(db_path).expanduser())
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        return MemoryStore(
            db_path=db_path,
            auto_embed=config.get("auto_embed", False),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
        )
    except ImportError:
        return None


@click.group()
@click.version_option(version="0.1.0", prog_name="amt-hermes")
def cli():
    """Agent Memory Toolkit - Hermes Plugin CLI.
    
    Manage persistent memories for AI agents using the Agent Memory Toolkit.
    
    Examples:
    
        amt-hermes add "User prefers Python over JavaScript"
        
        amt-hermes search "programming preferences"
        
        amt-hermes list --limit 20
    """
    pass


@cli.command()
@click.argument("content")
@click.option("--tags", "-t", multiple=True, help="Tags for categorization")
@click.option("--confidence", "-c", type=float, default=0.9, help="Confidence level 0.0-1.0")
@click.option("--source", "-s", default="cli", help="Source of this memory")
def add(content: str, tags: tuple, confidence: float, source: str):
    """Add a new memory to the store.
    
    CONTENT is the memory/fact to store.
    
    Examples:
    
        amt-hermes add "User prefers dark mode"
        
        amt-hermes add "Works at TechCorp" -t work -t professional
        
        amt-hermes add "Likes Python" --confidence 0.95 --source user_stated
    """
    store = _get_memory_store()
    if store is None:
        click.echo("Error: Agent Memory Toolkit not installed or configured.", err=True)
        click.echo("Run: pip install agent-memory-toolkit", err=True)
        sys.exit(1)
    
    try:
        memory = store.add(
            content,
            metadata={
                "tags": list(tags) if tags else [],
                "confidence": confidence,
                "source": source,
            }
        )
        click.echo(f"Memory stored successfully.")
        click.echo(f"  ID: {memory.id}")
        click.echo(f"  Content: {content[:50]}{'...' if len(content) > 50 else ''}")
        if tags:
            click.echo(f"  Tags: {', '.join(tags)}")
        store.close()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Maximum results to return")
@click.option("--mode", "-m", type=click.Choice(["auto", "fts", "vector", "hybrid"]), 
              default="auto", help="Search mode")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def search(query: str, limit: int, mode: str, tag: Optional[str], as_json: bool):
    """Search stored memories.
    
    QUERY is what to search for in memories.
    
    Examples:
    
        amt-hermes search "programming preferences"
        
        amt-hermes search "work" --tag professional
        
        amt-hermes search "dark mode" --limit 5 --json
    """
    store = _get_memory_store()
    if store is None:
        click.echo("Error: Agent Memory Toolkit not installed or configured.", err=True)
        sys.exit(1)
    
    try:
        # Choose search method
        if mode == "fts":
            results = store.search_fts(query, limit=limit)
        elif mode == "vector":
            results = store.search_vector(query, limit=limit)
        else:
            results = store.search(query, limit=limit)
        
        # Filter by tag if specified
        if tag and results:
            results = [
                r for r in results
                if tag in (r.memory.metadata.tags or [])
            ]
        
        if not results:
            if as_json:
                click.echo(json.dumps({"results": [], "count": 0}))
            else:
                click.echo("No relevant memories found.")
            store.close()
            return
        
        if as_json:
            items = []
            for r in results:
                item = {
                    "content": r.memory.content,
                    "score": round(r.score, 3),
                    "id": r.memory.id,
                }
                if r.memory.metadata:
                    if hasattr(r.memory.metadata, 'tags'):
                        item["tags"] = r.memory.metadata.tags or []
                items.append(item)
            click.echo(json.dumps({"results": items, "count": len(items)}, indent=2))
        else:
            click.echo(f"Found {len(results)} memories:\n")
            for i, r in enumerate(results, 1):
                click.echo(f"{i}. [{r.score:.2f}] {r.memory.content}")
                if hasattr(r.memory.metadata, 'tags') and r.memory.metadata.tags:
                    click.echo(f"   Tags: {', '.join(r.memory.metadata.tags)}")
        
        store.close()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("list")
@click.option("--limit", "-n", default=20, help="Maximum memories to list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_memories(limit: int, as_json: bool):
    """List recent memories.
    
    Examples:
    
        amt-hermes list
        
        amt-hermes list --limit 50
        
        amt-hermes list --json
    """
    store = _get_memory_store()
    if store is None:
        click.echo("Error: Agent Memory Toolkit not installed or configured.", err=True)
        sys.exit(1)
    
    try:
        memories = list(store.list(limit=limit))
        
        if not memories:
            if as_json:
                click.echo(json.dumps({"memories": [], "count": 0}))
            else:
                click.echo("No memories stored yet.")
            store.close()
            return
        
        if as_json:
            items = []
            for mem in memories:
                item = {
                    "id": mem.id,
                    "content": mem.content,
                }
                if hasattr(mem.metadata, 'tags') and mem.metadata.tags:
                    item["tags"] = mem.metadata.tags
                if hasattr(mem.metadata, 'created_at') and mem.metadata.created_at:
                    item["created_at"] = mem.metadata.created_at.isoformat()
                items.append(item)
            click.echo(json.dumps({"memories": items, "count": len(items)}, indent=2))
        else:
            click.echo(f"Stored memories ({len(memories)}):\n")
            for i, mem in enumerate(memories, 1):
                content_preview = mem.content[:60] + "..." if len(mem.content) > 60 else mem.content
                click.echo(f"{i}. {content_preview}")
                if hasattr(mem.metadata, 'tags') and mem.metadata.tags:
                    click.echo(f"   Tags: {', '.join(mem.metadata.tags)}")
        
        store.close()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def install():
    """Install the Agent Memory Toolkit plugin into Hermes.
    
    Copies plugin files to ~/.hermes/plugins/memory/agent-memory-toolkit/
    and updates config.yaml to enable the plugin.
    """
    hermes_home = get_hermes_home()
    plugins_dir = hermes_home / "plugins" / "memory" / "agent-memory-toolkit"
    source_dir = get_plugin_source_dir()

    click.echo("Installing Agent Memory Toolkit plugin...")
    click.echo(f"  Source: {source_dir}")
    click.echo(f"  Target: {plugins_dir}")

    # Create target directory
    plugins_dir.mkdir(parents=True, exist_ok=True)

    # Copy plugin files
    files_to_copy = ["__init__.py", "plugin.yaml", "context_injection.py", "cli.py"]
    for filename in files_to_copy:
        src = source_dir / filename
        dst = plugins_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            click.echo(f"  Copied: {filename}")
        else:
            click.echo(f"  Warning: {filename} not found")

    # Create symlink to main package (so imports work)
    toolkit_root = source_dir.parent
    package_link = plugins_dir / "agent_memory"
    if not package_link.exists():
        try:
            package_link.symlink_to(toolkit_root)
            click.echo(f"  Linked: agent_memory -> {toolkit_root}")
        except Exception as e:
            click.echo(f"  Warning: Could not create symlink: {e}")

    # Update config.yaml to enable the plugin
    config_path = hermes_home / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            
            # Add to enabled plugins
            if "plugins" not in config:
                config["plugins"] = {}
            if "enabled" not in config["plugins"]:
                config["plugins"]["enabled"] = []
            
            if "agent-memory-toolkit" not in config["plugins"]["enabled"]:
                config["plugins"]["enabled"].append("agent-memory-toolkit")
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                click.echo("  Added to plugins.enabled in config.yaml")
        except ImportError:
            click.echo("  Note: PyYAML not installed, manually add 'agent-memory-toolkit' to plugins.enabled")
        except Exception as e:
            click.echo(f"  Warning: Could not update config.yaml: {e}")

    click.echo("\nInstallation complete!")
    click.echo("\nNext steps:")
    click.echo("  1. Run 'amt-hermes setup' to configure the plugin")
    click.echo("  2. Or set AGENT_MEMORY_DB_PATH environment variable")
    click.echo("  3. Restart Hermes to load the plugin")


@cli.command()
def uninstall():
    """Uninstall the Agent Memory Toolkit plugin from Hermes.
    
    Removes plugin files from ~/.hermes/plugins/memory/agent-memory-toolkit/
    and updates config.yaml.
    """
    hermes_home = get_hermes_home()
    plugins_dir = hermes_home / "plugins" / "memory" / "agent-memory-toolkit"

    if not plugins_dir.exists():
        click.echo("Plugin not installed.")
        sys.exit(1)

    click.echo("Uninstalling Agent Memory Toolkit plugin...")
    click.echo(f"  Removing: {plugins_dir}")

    try:
        shutil.rmtree(plugins_dir)
        click.echo("  Removed plugin directory")
    except Exception as e:
        click.echo(f"  Error: {e}", err=True)
        sys.exit(1)

    # Update config.yaml to disable the plugin
    config_path = hermes_home / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            
            if "plugins" in config and "enabled" in config["plugins"]:
                if "agent-memory-toolkit" in config["plugins"]["enabled"]:
                    config["plugins"]["enabled"].remove("agent-memory-toolkit")
                    with open(config_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False)
                    click.echo("  Removed from plugins.enabled in config.yaml")
        except Exception as e:
            click.echo(f"  Warning: Could not update config.yaml: {e}")

    click.echo("\nUninstallation complete!")


@cli.command()
def setup():
    """Interactive setup wizard for the Agent Memory Toolkit plugin.
    
    Configures database path, embedding settings, extraction mode, and
    security level. Saves configuration to ~/.hermes/agent_memory.json.
    """
    hermes_home = get_hermes_home()
    config_path = hermes_home / "agent_memory.json"

    click.echo("Agent Memory Toolkit Setup")
    click.echo("=" * 40)
    click.echo()

    # Load existing config
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except:
            pass

    config = {}

    # Database path
    default_db = existing.get("db_path", str(hermes_home / "agent_memory.db"))
    db_path = click.prompt("Database path", default=default_db)
    config["db_path"] = db_path

    # Auto-embed
    default_embed = existing.get("auto_embed", False)
    auto_embed = click.confirm("Enable vector embeddings?", default=default_embed)
    config["auto_embed"] = auto_embed

    # Embedding model (only if auto_embed)
    if config["auto_embed"]:
        default_model = existing.get("embedding_model", "all-MiniLM-L6-v2")
        model = click.prompt("Embedding model", default=default_model)
        config["embedding_model"] = model

    # Extraction mode
    click.echo("\nExtraction modes:")
    click.echo("  rule   - Fast pattern-based extraction")
    click.echo("  llm    - Accurate LLM-based extraction (requires API key)")
    click.echo("  hybrid - Combined approach")
    default_mode = existing.get("extraction_mode", "rule")
    mode = click.prompt(
        "Extraction mode",
        default=default_mode,
        type=click.Choice(["rule", "llm", "hybrid"])
    )
    config["extraction_mode"] = mode

    # Security level
    click.echo("\nSecurity levels:")
    click.echo("  minimal  - Basic validation")
    click.echo("  low      - Light checks")
    click.echo("  medium   - Standard validation (recommended)")
    click.echo("  high     - Strict validation")
    click.echo("  paranoid - Maximum security")
    default_security = existing.get("security_level", "medium")
    security = click.prompt(
        "Security level",
        default=default_security,
        type=click.Choice(["minimal", "low", "medium", "high", "paranoid"])
    )
    config["security_level"] = security

    # Save config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))

    click.echo()
    click.echo(f"Configuration saved to: {config_path}")
    click.echo()
    click.echo("Your configuration:")
    for key, value in config.items():
        click.echo(f"  {key}: {value}")


@cli.command()
def status():
    """Show the status of the Agent Memory Toolkit plugin.
    
    Displays installation status, configuration, database info,
    and whether the toolkit is available.
    """
    hermes_home = get_hermes_home()
    plugins_dir = hermes_home / "plugins" / "memory" / "agent-memory-toolkit"
    config_path = hermes_home / "agent_memory.json"

    click.echo("Agent Memory Toolkit Status")
    click.echo("=" * 40)

    # Check if installed
    installed = plugins_dir.exists()
    click.echo(f"Installed: {'Yes' if installed else 'No'}")
    if installed:
        click.echo(f"  Location: {plugins_dir}")

    # Check config
    if config_path.exists():
        click.echo("Configured: Yes")
        try:
            config = json.loads(config_path.read_text())
            click.echo(f"  Config: {config_path}")
            for key, value in config.items():
                click.echo(f"    {key}: {value}")
        except:
            click.echo("  Config file exists but could not be read")
    else:
        click.echo("Configured: No (using defaults)")

    # Check if toolkit is importable
    try:
        from agentmemory import MemoryStore
        click.echo("Toolkit available: Yes")
    except ImportError as e:
        click.echo(f"Toolkit available: No ({e})")

    # Check database
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except:
            pass
    
    db_path = config.get("db_path", str(hermes_home / "agent_memory.db"))
    if db_path.startswith("~"):
        db_path = str(Path(db_path).expanduser())
    
    if Path(db_path).exists():
        size = Path(db_path).stat().st_size
        click.echo(f"Database: {db_path}")
        click.echo(f"  Size: {size / 1024:.1f} KB")
        
        # Try to get memory count
        try:
            store = _get_memory_store()
            if store:
                count = len(list(store.list(limit=10000)))
                click.echo(f"  Memories: {count}")
                store.close()
        except:
            pass
    else:
        click.echo("Database: Not created yet")
        click.echo(f"  Will be created at: {db_path}")


# Legacy main function for backwards compatibility with argparse-based CLI
def main(argv: Optional[list] = None) -> int:
    """Main entry point for the CLI."""
    try:
        cli(argv)
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
