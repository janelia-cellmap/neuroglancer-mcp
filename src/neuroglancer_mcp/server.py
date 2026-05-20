"""FastMCP server instance and tool registration.

Tool implementations live in `neuroglancer_mcp.tools.*`. Importing this
module imports them too, which triggers their `@mcp.tool()` decorators
and registers every tool with the server.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("neuroglancer-mcp")

# Importing the tool modules registers their @mcp.tool() functions.
# Imported at module bottom to avoid a circular import (the tool modules
# import `mcp` from here).
from neuroglancer_mcp.tools import (  # noqa: E402, F401
    layers,
    navigation,
    properties,
    segments,
    state,
    styling,
)
