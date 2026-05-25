"""Store management commands (branching, commits, etc.)."""

from __future__ import annotations

import click
import json as json_lib
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..main import Context


def get_context(ctx: click.Context) -> "Context":
    """Get the CLI context object."""
    return ctx.obj


@click.group()
def store() -> None:
    """Store management commands.
    
    Manage branches, commits, and database operations.
    
    \b
    Examples:
        amt store branch               # List branches
        amt store branch feature       # Create branch
        amt store checkout feature     # Switch to branch
        amt store commit "Added data"  # Commit changes
        amt store merge feature        # Merge branch
        amt store log                  # View commit history
    """
    pass


@store.command("branch")
@click.argument("name", required=False)
@click.option("--delete", "-d", "delete_branch", is_flag=True, help="Delete the branch")
@click.option("--list", "-l", "list_branches", is_flag=True, help="List all branches")
@click.pass_context
def branch(ctx: click.Context, name: Optional[str], delete_branch: bool, 
           list_branches: bool) -> None:
    """Manage branches.
    
    Without arguments, lists all branches.
    With a name, creates a new branch.
    
    \b
    Examples:
        amt store branch              # List branches
        amt store branch experiment   # Create 'experiment' branch
        amt store branch -d old       # Delete 'old' branch
    """
    context = get_context(ctx)
    store = context.store
    
    if list_branches or (name is None and not delete_branch):
        # List branches
        branches = store.list_branches()
        current = store.current_branch
        
        if context.json_output:
            data = {
                "current": current,
                "branches": [b.name for b in branches],
            }
            click.echo(json_lib.dumps(data, indent=2))
        else:
            click.echo("Branches:")
            for b in branches:
                marker = "*" if b.name == current else " "
                click.echo(f"  {marker} {b.name}")
    elif delete_branch and name:
        # Delete branch
        if name == "main":
            raise click.ClickException("Cannot delete the main branch")
        store.delete_branch(name)
        click.echo(f"Deleted branch: {name}")
    elif name:
        # Create branch
        store.create_branch(name)
        click.echo(f"Created branch: {name}")


@store.command("checkout")
@click.argument("branch_name")
@click.pass_context
def checkout(ctx: click.Context, branch_name: str) -> None:
    """Switch to a branch.
    
    \b
    Examples:
        amt store checkout main
        amt store checkout experiment
    """
    context = get_context(ctx)
    store = context.store
    
    store.checkout(branch_name)
    click.echo(f"Switched to branch: {branch_name}")


@store.command("commit")
@click.argument("message")
@click.pass_context
def commit(ctx: click.Context, message: str) -> None:
    """Commit changes on the current branch.
    
    \b
    Examples:
        amt store commit "Added user preferences"
        amt store commit "Updated meeting notes"
    """
    context = get_context(ctx)
    store = context.store
    
    commit_obj = store.commit(message)
    
    if context.json_output:
        click.echo(json_lib.dumps({
            "id": commit_obj.id,
            "message": commit_obj.message,
            "branch": store.current_branch,
            "timestamp": commit_obj.timestamp.isoformat() if commit_obj.timestamp else None,
        }, indent=2))
    else:
        click.echo(f"Committed: {commit_obj.id[:8]}")
        click.echo(f"  Message: {message}")
        click.echo(f"  Branch: {store.current_branch}")


@store.command("merge")
@click.argument("source_branch")
@click.option("--no-commit", is_flag=True, help="Don't auto-commit after merge")
@click.pass_context
def merge(ctx: click.Context, source_branch: str, no_commit: bool) -> None:
    """Merge a branch into the current branch.
    
    \b
    Examples:
        amt store merge experiment
        amt store merge feature --no-commit
    """
    context = get_context(ctx)
    store = context.store
    
    current = store.current_branch
    
    try:
        store.merge(source_branch, auto_commit=not no_commit)
        click.echo(f"Merged '{source_branch}' into '{current}'")
    except Exception as e:
        raise click.ClickException(f"Merge failed: {e}")


@store.command("log")
@click.option("--limit", "-n", default=10, help="Number of commits to show")
@click.option("--branch", "-b", default=None, help="Branch to show (default: current)")
@click.pass_context
def log(ctx: click.Context, limit: int, branch: Optional[str]) -> None:
    """Show commit history.
    
    \b
    Examples:
        amt store log
        amt store log --limit 20
        amt store log --branch experiment
    """
    context = get_context(ctx)
    store = context.store
    
    if branch:
        store.checkout(branch)
    
    commits = list(store.log(limit=limit))
    
    if context.json_output:
        data = [
            {
                "id": c.id,
                "message": c.message,
                "timestamp": c.timestamp.isoformat() if c.timestamp else None,
            }
            for c in commits
        ]
        click.echo(json_lib.dumps(data, indent=2))
    else:
        if not commits:
            click.echo("No commits yet.")
            return
        
        click.echo(f"Commit history ({store.current_branch}):")
        click.echo("-" * 50)
        for c in commits:
            ts = c.timestamp.strftime("%Y-%m-%d %H:%M") if c.timestamp else "unknown"
            click.echo(f"  {c.id[:8]}  {ts}  {c.message}")


@store.command("rollback")
@click.argument("commit_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@click.pass_context
def rollback(ctx: click.Context, commit_id: str, force: bool) -> None:
    """Rollback to a previous commit.
    
    \b
    Examples:
        amt store rollback abc123
        amt store rollback abc123 --force
    """
    context = get_context(ctx)
    store = context.store
    
    if not force:
        click.echo(f"Rolling back to commit: {commit_id}")
        if not click.confirm("This will discard changes. Continue?"):
            click.echo("Cancelled.")
            return
    
    store.rollback(commit_id)
    click.echo(f"Rolled back to: {commit_id}")


@store.command("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show store status.
    
    Displays current branch, uncommitted changes, and statistics.
    """
    context = get_context(ctx)
    store = context.store
    
    current_branch = store.current_branch
    memory_count = store.count()
    branches = store.list_branches()
    
    if context.json_output:
        data = {
            "branch": current_branch,
            "memory_count": memory_count,
            "branches": [b.name for b in branches],
            "db_path": str(context.db_path),
        }
        click.echo(json_lib.dumps(data, indent=2))
    else:
        click.echo(f"Database: {context.db_path}")
        click.echo(f"Branch: {current_branch}")
        click.echo(f"Memories: {memory_count}")
        click.echo(f"Branches: {', '.join(b.name for b in branches)}")
