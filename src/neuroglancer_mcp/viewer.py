"""Lazy-initialized Neuroglancer viewer with coordinate helpers.

We delay constructing the viewer until first use so that the MCP server
can start up instantly (Claude Desktop launches it on every chat), and
users who only need URL-mode tools don't pay for a server thread they
won't use.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import neuroglancer


_viewer: Optional[neuroglancer.Viewer] = None
_lock = threading.Lock()


def get_viewer() -> neuroglancer.Viewer:
    """Return the singleton viewer, constructing it on first call.

    The viewer spawns a local web server in a background thread and
    serves a Neuroglancer client at a URL that can be opened in any
    browser. State mutations from MCP tools are reflected live in that
    browser tab.

    The server binds to `0.0.0.0` by default so the viewer URL is
    reachable from other machines on the same LAN (typical workflow:
    MCP runs on a workstation, user opens the viewer from their
    laptop). Override with `NEUROGLANCER_MCP_BIND_ADDRESS=127.0.0.1`
    to restrict to localhost, or to a specific interface IP for
    tighter firewalling. The URL returned by `get_url` reflects the
    bound address.
    """
    global _viewer
    with _lock:
        if _viewer is None:
            bind = os.environ.get("NEUROGLANCER_MCP_BIND_ADDRESS", "0.0.0.0")
            neuroglancer.set_server_bind_address(bind_address=bind)
            _viewer = neuroglancer.Viewer()
        return _viewer


def get_viewer_url() -> str:
    """Return the shareable URL for the live viewer."""
    return str(get_viewer())


def reset_viewer() -> None:
    """Drop the cached viewer. Tests use this between cases; production should not."""
    global _viewer
    with _lock:
        _viewer = None


def nm_to_voxels(
    x_nm: float,
    y_nm: float,
    z_nm: float,
    dimensions: neuroglancer.CoordinateSpace,
) -> list[float]:
    """Convert (x, y, z) in nanometers to voxel coordinates using the viewer's dimensions.

    Neuroglancer's `position` is always in the coordinate space defined by the
    viewer's `dimensions`. For typical EM datasets those are physical units
    (meters or nanometers); for some volumes they're already voxels. We
    normalize on nm input and let dimensions tell us the per-axis scale.
    """
    axes = dimensions.names
    scales = dimensions.scales
    units = dimensions.units

    nm_values = {"x": x_nm, "y": y_nm, "z": z_nm}
    result: list[float] = []
    for axis, scale, unit in zip(axes, scales, units):
        if axis not in nm_values:
            result.append(0.0)
            continue
        if unit == "m":
            scale_nm = scale * 1e9
        elif unit == "nm":
            scale_nm = scale
        elif unit == "":
            scale_nm = 1.0
        else:
            raise ValueError(f"Unsupported unit {unit!r} for axis {axis!r}")
        result.append(nm_values[axis] / scale_nm)
    return result
