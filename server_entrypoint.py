"""Wrapper entrypoint for running the Alliance Docs MCP server via `fastmcp run`."""

from pathlib import Path
import sys

# Ensure the src/ package directory is importable when FastMCP loads this file directly.
SRC_PATH = Path(__file__).parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from alliance_docs_mcp.server import mcp  # noqa: E402  # re-export server instance

__all__ = ["mcp"]
