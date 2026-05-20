"""Serve local filesystem paths over HTTP so Neuroglancer's JS client can read them.

The Neuroglancer JS client (running in the user's browser) is what
actually fetches data — the Python viewer just feeds it state. Local
filesystem paths can't be read directly by a browser (`file://` is
cross-origin blocked), so anything we want to display from disk needs
an HTTP server sitting in front of it.

This module spawns `http.server` instances in daemon threads. They
live for the lifetime of the MCP server process; no explicit cleanup
needed. Servers default to binding the same address as the viewer
(`NEUROGLANCER_MCP_BIND_ADDRESS`, which defaults to `0.0.0.0` for LAN
access) so that a Neuroglancer view opened from a different machine on
the same LAN can fetch local data through the workstation's hostname.
"""

from __future__ import annotations

import http.server
import os
import socket
import socketserver
import threading
from functools import partial
from typing import Any, Optional


_registry_lock = threading.Lock()
_servers: dict[str, dict[str, Any]] = {}  # abs_path → {port, httpd, thread, bind}


def _hostname_for_bind(bind: str) -> str:
    """Return the hostname URL clients should use given a bind address.

    Mirrors `neuroglancer.server._get_regular_server_url` so the URL we
    hand back matches what Neuroglancer itself would produce. For
    `0.0.0.0` / `::` we use the FQDN; otherwise the bind address as-is.
    """
    if bind in ("0.0.0.0", "::"):
        return socket.getfqdn()
    return bind


def _default_bind() -> str:
    return os.environ.get("NEUROGLANCER_MCP_BIND_ADDRESS", "0.0.0.0")


def ensure_server(path: str, bind: Optional[str] = None) -> dict[str, Any]:
    """Start (or reuse) an HTTP server rooted at `path`.

    Idempotent: calling again with the same path returns the existing
    server's URL. Path is normalized to an absolute path, with `~`
    expanded and symlinks resolved, so two equivalent paths share one
    server.

    Args:
        path: Local directory to serve.
        bind: Bind address. Defaults to `NEUROGLANCER_MCP_BIND_ADDRESS`
            (which itself defaults to `0.0.0.0`).

    Returns:
        {"path": <abs>, "url": "http://host:port", "port": int,
         "bind": str, "started": bool} — `started` is True if this call
        actually spawned the server, False if it was already running.
    """
    abs_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    if not os.path.isdir(abs_path):
        raise ValueError(f"Not a directory: {abs_path}")
    bind = bind or _default_bind()

    with _registry_lock:
        existing = _servers.get(abs_path)
        if existing is not None:
            hostname = _hostname_for_bind(existing["bind"])
            return {
                "path": abs_path,
                "url": f"http://{hostname}:{existing['port']}",
                "port": existing["port"],
                "bind": existing["bind"],
                "started": False,
            }

        handler = partial(
            http.server.SimpleHTTPRequestHandler, directory=abs_path
        )
        # ThreadingTCPServer so multiple chunk fetches don't serialize.
        # allow_reuse_address avoids "address already in use" between
        # quick restarts during dev.
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        httpd = socketserver.ThreadingTCPServer((bind, 0), handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        _servers[abs_path] = {
            "port": port,
            "httpd": httpd,
            "thread": thread,
            "bind": bind,
        }

    hostname = _hostname_for_bind(bind)
    return {
        "path": abs_path,
        "url": f"http://{hostname}:{port}",
        "port": port,
        "bind": bind,
        "started": True,
    }


def list_servers() -> list[dict[str, Any]]:
    """Snapshot of currently-running data servers."""
    with _registry_lock:
        out: list[dict[str, Any]] = []
        for path, info in _servers.items():
            hostname = _hostname_for_bind(info["bind"])
            out.append(
                {
                    "path": path,
                    "url": f"http://{hostname}:{info['port']}",
                    "port": info["port"],
                    "bind": info["bind"],
                }
            )
        return out


def stop_server(path: str) -> dict[str, Any]:
    """Shut down a previously-started server. No-op if none running."""
    abs_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    with _registry_lock:
        info = _servers.pop(abs_path, None)
    if info is None:
        return {"stopped": False, "path": abs_path, "reason": "no server running for this path"}
    info["httpd"].shutdown()
    info["httpd"].server_close()
    return {"stopped": True, "path": abs_path, "port": info["port"]}


# ---------------------------------------------------------------------------
# Format detection for the auto-serve path in add_*_layer tools
# ---------------------------------------------------------------------------


def detect_format_prefix(path: str) -> str:
    """Return the Neuroglancer URL scheme for a local-filesystem dataset path.

    Looks at the file name and, if needed, the directory contents:
    - `*.zarr` (or contains `.zarr/`) → `zarr://`
    - `*.n5` (or contains `.n5/`) → `n5://`
    - directory containing an `info` file → `precomputed://`

    Raises ValueError if the format can't be determined.
    """
    abs_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    lowered = abs_path.lower()

    if lowered.endswith(".zarr") or ".zarr/" in lowered + "/":
        return "zarr://"
    if lowered.endswith(".n5") or ".n5/" in lowered + "/":
        return "n5://"
    if os.path.isdir(abs_path) and os.path.isfile(os.path.join(abs_path, "info")):
        return "precomputed://"

    raise ValueError(
        f"Could not detect Neuroglancer format for {abs_path}. "
        "Expected a `.zarr`/`.n5` directory, or a precomputed "
        "directory containing an `info` file."
    )


def looks_like_local_path(source: str) -> bool:
    """True if `source` should be treated as a filesystem path, not a URL.

    The MCP layer-add tools accept either kind. Anything containing
    `://` is treated as already-a-URL (precomputed://, zarr://,
    http://, etc.). Everything else — absolute paths, `~/`, relative
    paths — is treated as local.
    """
    return "://" not in source


def resolve_local_source(source: str, bind: Optional[str] = None) -> str:
    """Convert a local-filesystem source path to a Neuroglancer source URL.

    Spawns (or reuses) an HTTP server rooted at the dataset's parent
    directory, then constructs a URL with the right format scheme. The
    dataset stays under its parent so other datasets in the same
    directory share one server.
    """
    abs_path = os.path.realpath(os.path.abspath(os.path.expanduser(source)))
    if not os.path.exists(abs_path):
        raise ValueError(f"Local path does not exist: {abs_path}")
    prefix = detect_format_prefix(abs_path)
    parent = os.path.dirname(abs_path)
    name = os.path.basename(abs_path)
    server = ensure_server(parent, bind=bind)
    return f"{prefix}{server['url']}/{name}"
