"""Entry point for the Computer Use MCP server.

Run with:
    python -m open_llm_vtuber.mcp_servers.computer_use_server
"""

from . import mcp

if __name__ == "__main__":
    mcp.run()
