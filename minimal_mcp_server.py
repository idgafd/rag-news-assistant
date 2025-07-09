#!/usr/bin/env python3
"""
Minimal MCP Server - Just to test if FastMCP works
"""

from fastmcp import FastMCP
from typing import List, Dict, Any, Optional

# Create MCP server
mcp = FastMCP("Minimal News Search")


@mcp.tool()
def test_tool(message: str) -> str:
    """Simple test tool"""
    print(f"[MCP] test_tool called with: {message}")
    return f"Echo: {message}"


@mcp.tool()
def search_similar_articles(
        prompt: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Mock search function"""
    print(f"[MCP] search_similar_articles called: {prompt}")
    return [
        {
            "payload": {
                "title": f"Mock Article about {prompt}",
                "date": "2024-06-15",
                "text": f"This is a mock article about {prompt}. It contains relevant information for testing."
            }
        }
    ]


@mcp.tool()
def answer_user_query(user_query: str, articles: List[Dict[str, Any]]) -> str:
    """Mock answer function"""
    print(f"[MCP] answer_user_query called: {user_query} with {len(articles)} articles")
    return f"Mock answer for '{user_query}' based on {len(articles)} articles."


if __name__ == "__main__":
    print("🧪 Starting Minimal MCP Server...")
    print("Available tools:")
    print("  - test_tool")
    print("  - search_similar_articles")
    print("  - answer_user_query")
    print("Server starting...")

    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()