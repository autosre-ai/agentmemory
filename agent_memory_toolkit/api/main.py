"""
Agent Memory Toolkit REST API - Main Entry Point.

This module provides the main FastAPI application and server entry point.
It re-exports from app.py for convention and ease of use.

Usage:
    # Run directly with uvicorn
    uvicorn agent_memory_toolkit.api.main:app --reload
    
    # Or use the CLI
    amt api serve
    
    # Or programmatically
    from agent_memory_toolkit.api.main import app, create_app, run_server
    run_server(port=8000)
"""

from .app import app, create_app, run_server

# Re-export for convenience
__all__ = ["app", "create_app", "run_server"]


if __name__ == "__main__":
    run_server()
