#!/usr/bin/env python3
"""
Run the Agent Memory Toolkit MCP Server.

This script provides a simple entry point for starting the MCP server.
It can be run directly or used as a module.

Usage:
    # Run with default settings (stdio transport)
    python scripts/run_mcp_server.py
    
    # Run with custom database
    python scripts/run_mcp_server.py --db ~/my_memories.db
    
    # Run with SSE transport for debugging
    python scripts/run_mcp_server.py --transport sse --port 8765
    
    # Run with high security
    python scripts/run_mcp_server.py --security-level high

For production use with Claude Desktop or Cursor, prefer the installed CLI:
    amt-mcp serve --db-path ~/agent_memory.db
    
    Or via the main CLI:
    amt mcp serve --db-path ~/agent_memory.db
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(
        description="Agent Memory Toolkit MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Default stdio transport
  %(prog)s --transport sse --port 8765  # SSE transport on port 8765
  %(prog)s --db ~/memories.db           # Custom database path
  %(prog)s --security-level high        # High security mode

MCP Client Configuration:
  Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "agent-memory-toolkit": {
          "command": "python",
          "args": ["path/to/run_mcp_server.py", "--db", "~/agent_memory.db"]
        }
      }
    }
"""
    )
    
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type (default: stdio)"
    )
    parser.add_argument(
        "--db", "--db-path", "-d",
        default="agent_memory.db",
        help="Path to memory database (default: agent_memory.db)"
    )
    parser.add_argument(
        "--host", "-H",
        default="127.0.0.1",
        help="Host for SSE server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="Port for SSE server (default: 8765)"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--security-level", "-s",
        choices=["minimal", "low", "medium", "high", "paranoid"],
        default="medium",
        help="Security level for content validation (default: medium)"
    )
    parser.add_argument(
        "--disable-extraction",
        action="store_true",
        help="Disable memory extraction tools"
    )
    parser.add_argument(
        "--disable-security",
        action="store_true",
        help="Disable security validation tools"
    )
    parser.add_argument(
        "--disable-compression",
        action="store_true",
        help="Disable context compression tools"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if args.transport == "stdio":
        # Log to stderr for stdio transport to not interfere with MCP protocol
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format=log_format,
            stream=sys.stderr
        )
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format=log_format
        )
    
    logger = logging.getLogger(__name__)
    
    # Try to import MCP components
    try:
        from agent_memory_toolkit.mcp import create_mcp_server, MCPConfig
    except ImportError as e:
        logger.error(
            f"Failed to import MCP components: {e}\n"
            "Make sure agent-memory-toolkit is installed with MCP support:\n"
            "  pip install agent-memory-toolkit[mcp]"
        )
        sys.exit(1)
    
    # Expand user path
    db_path = str(Path(args.db).expanduser())
    
    # Create configuration
    config = MCPConfig(
        memory_db=db_path,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        security_level=args.security_level,
        enable_extraction=not args.disable_extraction,
        enable_security=not args.disable_security,
        enable_compression=not args.disable_compression,
    )
    
    # Create and run server
    logger.info("Starting Agent Memory Toolkit MCP Server...")
    logger.info(f"Transport: {args.transport}")
    logger.info(f"Database: {db_path}")
    logger.info(f"Security level: {args.security_level}")
    logger.info(f"Features: extraction={not args.disable_extraction}, "
                f"security={not args.disable_security}, "
                f"compression={not args.disable_compression}")
    
    try:
        mcp_server = create_mcp_server(config)
        
        if args.transport == "stdio":
            mcp_server.run(transport="stdio")
        else:
            logger.info(f"SSE server starting on http://{args.host}:{args.port}")
            mcp_server.run()
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
