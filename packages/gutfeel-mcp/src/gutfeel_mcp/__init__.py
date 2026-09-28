"""The `gutfeel-mcp` command: gut's MCP server, installable on its own.

The server is `gut._mcp`, in the gutfeel package; this package only brings the MCP SDK and Jev's SDK
along with it, so that an MCP client can start it with `uvx gutfeel-mcp` and nothing else.
"""

from __future__ import annotations

from gut._mcp import main

__all__ = ["main"]
