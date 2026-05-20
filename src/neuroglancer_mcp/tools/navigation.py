"""Navigation tools: position, zoom, orientation.

Coordinate units are NEVER implicit. Spatial inputs always take a
`units` parameter with no default — the nm↔voxel mix-up is the single
most common Neuroglancer error and the agent must not reproduce it.
"""

from __future__ import annotations

from typing import Any

from neuroglancer_mcp.bounds import apply_center_to_viewer
from neuroglancer_mcp.meshes import estimate_segment_centroid
from neuroglancer_mcp.server import mcp
from neuroglancer_mcp.tools.properties import _layer_source_urls
from neuroglancer_mcp.viewer import get_viewer, nm_to_voxels


@mcp.tool()
def navigate_to(x: float, y: float, z: float, units: str) -> dict[str, Any]:
    """Move the viewer's center to a 3D position.

    Args:
        x: X coordinate.
        y: Y coordinate.
        z: Z coordinate.
        units: Either 'nm' (nanometers, converted via the viewer's
            dimensions) or 'voxels' (raw voxel coordinates in the
            viewer's current coordinate space). REQUIRED — do not guess.

    Returns:
        {"position": [x, y, z]} in voxel coordinates as set on the viewer.
    """
    if units not in {"nm", "voxels"}:
        raise ValueError(f"units must be 'nm' or 'voxels', got {units!r}")

    viewer = get_viewer()
    with viewer.txn() as s:
        if units == "nm":
            pos = nm_to_voxels(x, y, z, s.dimensions)
        else:
            pos = [x, y, z]
        s.position = pos
        return {"position": list(s.position)}


@mcp.tool()
def set_zoom(scale: float) -> dict[str, float]:
    """Set the cross-section view scale (nm/pixel).

    Smaller values zoom in; larger values zoom out. Typical EM datasets
    have native resolutions of 4–32 nm/voxel, so a `scale` of 8 shows
    roughly one voxel per pixel for an 8nm dataset.

    Args:
        scale: Cross-section nm per pixel.

    Returns:
        {"cross_section_scale": <scale>}.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        s.cross_section_scale = scale
        return {"cross_section_scale": s.cross_section_scale}


@mcp.tool()
def get_position() -> dict[str, list[float]]:
    """Return the viewer's current center position in voxels.

    Returns:
        {"position": [x, y, z]} — voxel coordinates in the current
        coordinate space. To get nanometers, multiply by the per-axis
        scales from `get_state()["dimensions"]`.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        return {"position": list(s.position) if s.position is not None else []}


@mcp.tool()
def center_on_layer(layer: str) -> dict[str, Any]:
    """Move the viewer to the volume center of an existing layer.

    Use this when you didn't auto-center on layer add (or you want to
    re-center after navigating elsewhere). Reads the layer's source
    metadata to compute the physical midpoint of the volume —
    supports `precomputed://` and OME-NGFF `zarr://` sources.

    For centering on a specific segment instead of the whole volume,
    use `center_on_segment`.

    Args:
        layer: Layer name (image or segmentation; both work).

    Returns:
        {"layer", "format", "position_nm", "voxel_size_nm",
         "shape_voxels", "position_voxels"} on success, or
        {"layer", "centered": False, "reason": "..."} if metadata
        couldn't be fetched or the source format isn't supported.
    """
    viewer = get_viewer()
    sources = _layer_source_urls(viewer, layer) if False else None  # placeholder
    # We need the raw source url, including the URI scheme. The
    # segmentation-specific helper raises for image layers; do our own
    # lookup that accepts any layer type.
    with viewer.txn() as s:
        match = None
        for layer_obj in s.layers:
            if layer_obj.name == layer:
                match = layer_obj.layer
                break
        if match is None:
            raise ValueError(f"Layer {layer!r} not found")
        src = match.source
        if src is None:
            raise ValueError(f"Layer {layer!r} has no source")
        if isinstance(src, (list, tuple)) or (
            hasattr(src, "__iter__") and not isinstance(src, str)
        ):
            sources_list = [str(getattr(x, "url", x)) for x in src]
        else:
            sources_list = [str(getattr(src, "url", src))]
    primary = next(
        (s for s in sources_list if not s.rstrip("/").endswith("segment_properties")),
        sources_list[0],
    )
    centered = apply_center_to_viewer(viewer, primary)
    if centered is None:
        return {
            "layer": layer,
            "centered": False,
            "reason": "metadata fetch failed or unsupported source format",
        }
    return {"layer": layer, **centered}


@mcp.tool()
def center_on_segment(layer: str, segment_id: int) -> dict[str, Any]:
    """Move the viewer to the approximate centroid of one segment.

    Use this after `show_segments` (or directly with a segment ID you
    already know) when the user wants the camera positioned on a
    specific object — e.g. "show me a Kenyon cell" should result in
    the Kenyon cell visible at viewer center, not just selected in
    some random part of the volume.

    Implementation: reads the segmentation's mesh manifest for this
    segment and uses the coarsest LOD's fragment bounding-box centroid
    as the position. Works for sharded `neuroglancer_multilod_draco`
    meshes (hemibrain, FlyWire, MICrONS, CellMap) and legacy unsharded
    meshes. No mesh vertices are decoded — only the manifest header.

    The manifest's native positions are in the segmentation's voxel
    coordinate space (the precomputed model space), which is what the
    viewer's `position` field expects — no nm round-trip needed. We
    also report the nm equivalent for the user's reference.

    Args:
        layer: Segmentation layer name. Must have a mesh.
        segment_id: The segment ID to center on.

    Returns:
        {"layer", "segment_id", "mesh_type",
         "position": [x, y, z]   (segmentation voxel coords),
         "position_nm": [x, y, z]}.
    """
    viewer = get_viewer()
    sources = _layer_source_urls(viewer, layer)
    if not sources:
        raise ValueError(f"Layer {layer!r} has no sources")
    primary = next(
        (s for s in sources if not s.rstrip("/").endswith("segment_properties")),
        sources[0],
    )
    centroid = estimate_segment_centroid(primary, segment_id)
    pos_voxels = centroid["position"]

    with viewer.txn() as s:
        s.position = list(pos_voxels)
        return {
            "layer": layer,
            "segment_id": segment_id,
            "mesh_type": centroid["mesh_type"],
            "position": [float(v) for v in s.position],
            "position_nm": centroid["position_nm"],
        }


@mcp.tool()
def set_orientation(quaternion: list[float]) -> dict[str, list[float]]:
    """Set the 3D view orientation as a quaternion [x, y, z, w].

    Neuroglancer uses unit quaternions to represent the rotation of the
    3D camera. The identity orientation [0, 0, 0, 1] looks down the
    positive Z axis with X right and Y up. The MCP does not normalize —
    pass a unit quaternion.

    Args:
        quaternion: Four-element list [x, y, z, w] representing the
            rotation. Should be unit-norm.

    Returns:
        {"orientation": [x, y, z, w]}.
    """
    if len(quaternion) != 4:
        raise ValueError(
            f"quaternion must have 4 elements [x, y, z, w], got {len(quaternion)}"
        )

    viewer = get_viewer()
    with viewer.txn() as s:
        s.projection_orientation = list(quaternion)
        return {"orientation": list(s.projection_orientation)}
