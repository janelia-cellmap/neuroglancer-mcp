"""Layer management tools: add image/segmentation, remove, toggle visibility."""

from __future__ import annotations

from typing import Any

import neuroglancer

from neuroglancer_mcp.bounds import apply_center_to_viewer
from neuroglancer_mcp.server import mcp
from neuroglancer_mcp.viewer import get_viewer


@mcp.tool()
def add_image_layer(
    name: str, source: str, center: bool = True
) -> dict[str, Any]:
    """Add an image layer (raw EM, MRI, etc.) to the viewer.

    By default the viewer also navigates to the volume's center, the
    same way Neuroglancer's JS UI does when you drop a layer onto it.
    Pass `center=False` to keep the current position (useful when
    you've already navigated somewhere specific).

    Args:
        name: Display name for the layer.
        source: Data source URL. Common prefixes:
            - precomputed://gs://...   (Google Cloud Storage Precomputed)
            - precomputed://https://...  (HTTP-hosted Precomputed)
            - n5://...
            - zarr://...
            Example: 'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg'
        center: If True (default) and the source format is supported
            (precomputed, OME-NGFF zarr), navigate to the volume's
            physical center after adding. If centering can't happen
            (unsupported format, fetch failure), the layer is still
            added and a note is logged to stderr.

    Returns:
        {"name", "type", "source"} plus an optional `centered_on`
        block when auto-centering succeeded
        ({"format", "position_nm", "voxel_size_nm", "shape_voxels",
        "position_voxels"}).
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        s.layers[name] = neuroglancer.ImageLayer(source=source)
    result: dict[str, Any] = {"name": name, "type": "image", "source": source}
    if center:
        centered = apply_center_to_viewer(viewer, source)
        if centered is not None:
            result["centered_on"] = centered
    return result


@mcp.tool()
def add_segmentation_layer(
    name: str, source: str, center: bool = True
) -> dict[str, Any]:
    """Add a segmentation layer (neurons, organelles, etc.) to the viewer.

    Initially no segments are visible — call `show_segments` to display
    specific objects by ID. By default the viewer also navigates to
    the volume's center; pass `center=False` to keep the current
    position.

    Args:
        name: Display name for the layer.
        source: Data source URL (typically precomputed:// for
            connectomics). Example:
            'precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation'
        center: Auto-navigate to the volume center after adding.
            Default True.

    Returns:
        {"name", "type", "source"} plus optional `centered_on` block
        when auto-centering succeeded.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        s.layers[name] = neuroglancer.SegmentationLayer(source=source)
    result: dict[str, Any] = {
        "name": name,
        "type": "segmentation",
        "source": source,
    }
    if center:
        centered = apply_center_to_viewer(viewer, source)
        if centered is not None:
            result["centered_on"] = centered
    return result


@mcp.tool()
def remove_layer(name: str) -> dict[str, Any]:
    """Remove a layer from the viewer.

    Args:
        name: The layer name to remove.

    Returns:
        {"removed": <name>, "remaining": [<names of layers still present>]}.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        if name in [layer.name for layer in s.layers]:
            del s.layers[name]
        remaining = [layer.name for layer in s.layers]
    return {"removed": name, "remaining": remaining}


@mcp.tool()
def set_layer_visibility(name: str, visible: bool) -> dict[str, Any]:
    """Show or hide a layer without removing it.

    Args:
        name: Layer name.
        visible: True to show, False to hide.

    Returns:
        {"name": ..., "visible": ...}.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        for layer in s.layers:
            if layer.name == name:
                layer.visible = visible
                return {"name": name, "visible": layer.visible}
        raise ValueError(f"Layer {name!r} not found")
