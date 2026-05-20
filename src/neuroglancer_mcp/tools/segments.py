"""Segment selection tools.

Show, add, hide, and clear segments in a segmentation layer. These
operate on the `segments` set of a `SegmentationLayer`; they do not
affect coloring or rendering mode (those land in Phase 3).
"""

from __future__ import annotations

from typing import Any

import neuroglancer

from neuroglancer_mcp.server import mcp
from neuroglancer_mcp.viewer import get_viewer


def _get_seg_layer(s: Any, name: str) -> Any:
    for layer in s.layers:
        if layer.name == name:
            if not isinstance(layer.layer, neuroglancer.SegmentationLayer):
                raise ValueError(f"Layer {name!r} is not a segmentation layer")
            return layer.layer
    raise ValueError(f"Segmentation layer {name!r} not found")


@mcp.tool()
def show_segments(layer: str, segment_ids: list[int]) -> dict[str, Any]:
    """Replace the visible segment selection in a segmentation layer.

    Use this when the user says "show me X" — it clears the current
    selection and shows only the specified IDs. For "also show", use
    `add_segments` instead.

    Args:
        layer: Name of the segmentation layer.
        segment_ids: List of segment IDs (integers) to display.

    Returns:
        {"layer": ..., "visible_segments": [...]}.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        seg.segments = set(segment_ids)
        return {"layer": layer, "visible_segments": sorted(seg.segments)}


@mcp.tool()
def add_segments(layer: str, segment_ids: list[int]) -> dict[str, Any]:
    """Add segments to the current selection (keeps existing visible segments).

    Use this when the user says "also show", "add", or "include" — they
    want the current view extended, not replaced.

    Args:
        layer: Segmentation layer name.
        segment_ids: Segment IDs to add to the visible set.

    Returns:
        {"layer": ..., "visible_segments": [...]} — the full resulting set.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        seg.segments = set(seg.segments) | set(segment_ids)
        return {"layer": layer, "visible_segments": sorted(seg.segments)}


@mcp.tool()
def hide_segments(layer: str, segment_ids: list[int]) -> dict[str, Any]:
    """Remove specific segments from the visible selection.

    Args:
        layer: Segmentation layer name.
        segment_ids: Segment IDs to hide. IDs not currently visible are ignored.

    Returns:
        {"layer": ..., "visible_segments": [...]} — segments still visible.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        seg.segments = set(seg.segments) - set(segment_ids)
        return {"layer": layer, "visible_segments": sorted(seg.segments)}


@mcp.tool()
def clear_segments(layer: str) -> dict[str, Any]:
    """Hide all segments in a layer (the layer itself remains).

    Args:
        layer: Segmentation layer name.

    Returns:
        {"layer": ..., "visible_segments": []}.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        seg.segments = set()
        return {"layer": layer, "visible_segments": []}
