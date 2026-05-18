"""Entry point for the Terminal MCP server.

Run with:
    python -m open_llm_vtuber.mcp_servers.terminal_server
"""

from . import mcp

if __name__ == "__main__":
    mcp.run()
