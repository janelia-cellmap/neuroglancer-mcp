"""MCP tools for exposing local filesystem data to the Neuroglancer viewer.

The Neuroglancer JS client (in the browser) can't read `file://` URLs,
so we spawn lightweight HTTP servers that proxy local directories. The
`add_*_layer` tools auto-call into this module when given a path
instead of a URL; these tools let you start/list/stop those servers
explicitly when you need finer control.
"""

from __future__ import annotations

from typing import Any, Optional

from neuroglancer_mcp import local_serve
from neuroglancer_mcp.server import mcp


@mcp.tool()
def serve_local_directory(
    path: str, bind: Optional[str] = None
) -> dict[str, Any]:
    """Expose a local filesystem directory over HTTP for Neuroglancer.

    Use this when you want the agent to display data living on disk
    (e.g. `/groups/cellmap/.../jrc_hela-2.zarr`). The server is started
    in a daemon thread and lives for the rest of the MCP server's
    lifetime; subsequent calls with the same path return the existing
    server's URL (idempotent).

    By default the server binds to the same address as the viewer
    (`NEUROGLANCER_MCP_BIND_ADDRESS`, default `0.0.0.0`) so a
    Neuroglancer view opened from a laptop on the same LAN can fetch
    the data. Pass `bind="127.0.0.1"` to restrict to localhost when the
    directory contains sensitive data and you don't want LAN exposure.

    For the common case of "show me this local dataset", you can also
    just pass the local path directly to `add_image_layer` /
    `add_segmentation_layer` / `add_layer` / `add_layers` — they'll
    spawn the server automatically. Use this tool explicitly when you
    want to share one server across many `add_*` calls or pre-warm a
    directory.

    Args:
        path: Absolute or `~`-relative directory. Symlinks are
            resolved so equivalent paths share one server.
        bind: Bind address. Default: matches the viewer's bind.

    Returns:
        {"path": <abs>, "url": "http://host:port", "port": int,
         "bind": str, "started": bool}.
    """
    return local_serve.ensure_server(path, bind=bind)


@mcp.tool()
def list_served_directories() -> dict[str, Any]:
    """List all local directories the MCP is currently serving over HTTP.

    Returns:
        {"servers": [{path, url, port, bind}, ...]}.
    """
    return {"servers": local_serve.list_servers()}


@mcp.tool()
def stop_serving_directory(path: str) -> dict[str, Any]:
    """Shut down the HTTP server for a specific local directory.

    Idempotent — returns `{stopped: false, reason: ...}` if no server
    was running for that path. Doesn't affect layers already in the
    viewer that point at the now-stopped URL (they'll just fail to
    refresh).

    Args:
        path: The path passed to `serve_local_directory`. Same
            normalization (absolute, ~-expanded, symlink-resolved).

    Returns:
        {"stopped": bool, "path": <abs>, "port": int} on success,
        {"stopped": false, "path": <abs>, "reason": str} otherwise.
    """
    return local_serve.stop_server(path)
