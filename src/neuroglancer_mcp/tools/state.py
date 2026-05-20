"""State & introspection tools.

Every tool's docstring is what Claude sees when deciding whether to call
it — be explicit about units, side effects, and return values.
"""

from __future__ import annotations

from typing import Any

import neuroglancer

from neuroglancer_mcp.server import mcp
from neuroglancer_mcp.viewer import get_viewer, get_viewer_url


@mcp.tool()
def get_state() -> dict[str, Any]:
    """Return the full Neuroglancer viewer state as a JSON-compatible dict.

    This is the source of truth for what's currently loaded and visible.
    Call this before making decisions about layers or segments — don't
    assume state from previous tool calls, since the user may have
    interacted with the viewer directly in their browser.

    Returns:
        The viewer state as a dict (layers, position, zoom, dimensions, etc.).
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        return s.to_json()


@mcp.tool()
def get_url() -> dict[str, str]:
    """Return the live viewer URL.

    On first call this is the URL the user should open in their browser
    to see the viewer. On subsequent calls it remains valid for the
    lifetime of the MCP server process — and reflects live updates.

    For a shareable snapshot URL that captures the current state for
    others (no live viewer required on their end), call `share_url`
    instead.

    Returns:
        {"url": "<viewer URL>"} — open this in Chrome or Firefox.
    """
    return {"url": get_viewer_url()}


@mcp.tool()
def share_url() -> dict[str, str]:
    """Return a snapshot URL that encodes the current viewer state.

    Unlike `get_url`, which points at this process's live viewer (only
    reachable while the MCP server is running), `share_url` returns a
    self-contained link on `neuroglancer-demo.appspot.com` with the full
    state encoded in the URL fragment. Send this to collaborators or
    paste into a paper/slide; it will keep working after the MCP server
    exits.

    The snapshot is taken at call time and does not update if the viewer
    state changes afterward — call again to refresh.

    Returns:
        {"url": "https://neuroglancer-demo.appspot.com/#!..."}.
    """
    viewer = get_viewer()
    return {"url": neuroglancer.to_url(viewer.state)}


@mcp.tool()
def load_state(state: dict[str, Any]) -> dict[str, Any]:
    """Replace the entire viewer state with the provided JSON state.

    Useful for restoring a previously-shared Neuroglancer view, or for
    starting from a known template. To preserve specific layers across
    state loads, call `get_state` first and merge manually.

    Args:
        state: A dict matching Neuroglancer's state schema (layers,
            position, dimensions, etc.). Typically obtained from
            `get_state`, from a shared Neuroglancer URL's JSON fragment,
            or from a CAVE state server.

    Returns:
        The resulting state after loading.
    """
    viewer = get_viewer()
    viewer.set_state(state)
    with viewer.txn() as s:
        return s.to_json()


@mcp.tool()
def list_layers() -> list[dict[str, Any]]:
    """List all layers currently in the viewer.

    Returns:
        A list of {name, type, visible, source} dicts, one per layer.
        `type` is e.g. "image", "segmentation", "annotation". `source`
        is the data URL (precomputed://, n5://, zarr://, etc.).
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        out: list[dict[str, Any]] = []
        for layer in s.layers:
            out.append(
                {
                    "name": layer.name,
                    "type": layer.layer.type,
                    "visible": layer.visible,
                    "source": _summarize_source(layer.layer),
                }
            )
        return out


def _summarize_source(layer: Any) -> str | list[str] | None:
    """Extract a string-ish representation of a layer's data source."""
    src = getattr(layer, "source", None)
    if src is None:
        return None
    if isinstance(src, (list, tuple)):
        return [str(s) for s in src]
    return str(src)
