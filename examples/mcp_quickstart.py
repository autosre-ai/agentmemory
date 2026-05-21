#!/usr/bin/env python3
"""
MCP Server Quickstart - Agent Memory Toolkit

Demonstrates how to start and use the MCP server programmatically.
For CLI usage, run: amt-mcp serve
"""

import asyncio
import sys
sys.path.insert(0, "..")

from agentmemory.mcp.server import create_mcp_server, MCPConfig


async def demo_mcp_tools():
    """Demonstrate MCP tools programmatically."""
    print("=" * 60)
    print("  MCP SERVER QUICKSTART DEMO")
    print("=" * 60)
    
    # Create server with custom config
    config = MCPConfig(
        name="demo-memory-server",
        memory_db=":memory:",  # In-memory for demo
        enable_extraction=True,
        enable_security=True,
        enable_compression=True,
    )
    
    mcp = create_mcp_server(config)
    
    print("\n1. LISTING AVAILABLE TOOLS")
    print("-" * 40)
    
    # Get available tools
    tools = await mcp.list_tools()
    print(f"   Available tools ({len(tools)}):")
    for tool in tools:
        print(f"     - {tool.name}")
    
    print("\n2. ADDING MEMORIES")
    print("-" * 40)
    
    # Add some memories using the tools
    memories_to_add = [
        "User prefers dark mode and vim keybindings",
        "Project uses Python 3.11 with FastAPI",
        "Deployment target is Kubernetes on AWS",
        "Team standup is at 10am PST daily",
    ]
    
    for content in memories_to_add:
        result = await mcp.call_tool(
            "memory_add",
            {
                "content": content,
                "source": "demo",
                "tags": ["demo", "quickstart"],
            }
        )
        # Parse result (MCP returns list with single TextContent)
        import json
        parsed = json.loads(result[0].text)
        if parsed.get("success"):
            print(f"   ✓ Added: {content[:40]}...")
        else:
            print(f"   ✗ Error: {parsed.get('error')}")
    
    print("\n3. SEARCHING MEMORIES")
    print("-" * 40)
    
    # Search for specific content
    result = await mcp.call_tool(
        "memory_query",
        {"query": "Python OR deployment", "limit": 5}
    )
    import json
    parsed = json.loads(result[0].text)
    
    print(f"   Search results ({parsed.get('count', 0)} matches):")
    for mem in parsed.get("memories", []):
        print(f"     [{mem['score']:.2f}] {mem['content'][:50]}...")
    
    print("\n4. EXTRACTING MEMORIES FROM TEXT")
    print("-" * 40)
    
    sample_text = """
    My name is Alice and I work as a DevOps engineer at CloudCorp.
    I usually start work at 9am and prefer asynchronous communication.
    For deployments, I always run the smoke tests first.
    """
    
    result = await mcp.call_tool(
        "extract_memories",
        {"text": sample_text, "source": "conversation"}
    )
    parsed = json.loads(result[0].text)
    
    if parsed.get("success"):
        print(f"   Extracted {parsed.get('count', 0)} memories:")
        for mem in parsed.get("memories", [])[:5]:
            print(f"     [{mem['domain']}] {mem['key']}: {mem['value']}")
    
    print("\n5. SECURITY VALIDATION")
    print("-" * 40)
    
    # Test security check
    safe_content = "User prefers Python 3.11"
    result = await mcp.call_tool(
        "guard_check",
        {"content": safe_content}
    )
    parsed = json.loads(result[0].text)
    print(f"   Content: {safe_content}")
    print(f"   Is safe: {parsed.get('is_safe')}")
    print(f"   Confidence: {parsed.get('confidence', 0):.2f}")
    
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print("\nTo start the server for Claude Desktop or Cursor:")
    print("  amt-mcp serve --transport stdio")
    print("\nOr configure in Claude Desktop's claude_desktop_config.json:")
    print("""
{
  "mcpServers": {
    "agent-memory-toolkit": {
      "command": "amt-mcp",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
""")


def main():
    """Run the MCP demo."""
    asyncio.run(demo_mcp_tools())


if __name__ == "__main__":
    main()
