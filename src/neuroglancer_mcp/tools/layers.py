"""Layer management tools: add image/segmentation, remove, toggle visibility."""

from __future__ import annotations

from typing import Any, Optional

import neuroglancer

from neuroglancer_mcp.bounds import apply_center_to_viewer, infer_layer_type
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
def add_layer(
    name: str,
    source: str,
    type: Optional[str] = None,
    center: bool = True,
) -> dict[str, Any]:
    """Add a single layer, inferring image vs segmentation from the dtype.

    Use this when you don't already know whether the source is image
    or segmentation data (e.g., you got a URL from a user / web page
    and don't want to fetch the info file yourself first). The tool
    reads the source's dtype from its metadata and dispatches:

    - `float*` dtypes → image (high confidence)
    - `uint64` → segmentation (high confidence)
    - other integer dtypes → image (low confidence; pass `type=` to
      override if the data is actually a label volume)

    If you already know the type, prefer the explicit
    `add_image_layer` / `add_segmentation_layer` — they skip the
    dtype-probe network round-trip.

    Args:
        name: Display name for the layer.
        source: Data source URL (precomputed://, zarr://, n5://, …).
        type: Either "image" or "segmentation" to override inference.
            Leave None (default) to auto-detect.
        center: Auto-navigate to the volume center after adding. Same
            semantics as `add_image_layer(center=...)`.

    Returns:
        {"name", "type", "source", "inferred": bool,
         "inference": {dtype, confidence, reason} | None,
         "centered_on": {...} | absent}.
    """
    inference: Optional[dict[str, Any]] = None
    if type is None:
        inference = infer_layer_type(source)
        type = inference["type"]
    if type not in {"image", "segmentation"}:
        raise ValueError(
            f"type must be 'image' or 'segmentation', got {type!r}"
        )

    viewer = get_viewer()
    with viewer.txn() as s:
        if type == "image":
            s.layers[name] = neuroglancer.ImageLayer(source=source)
        else:
            s.layers[name] = neuroglancer.SegmentationLayer(source=source)

    result: dict[str, Any] = {
        "name": name,
        "type": type,
        "source": source,
        "inferred": inference is not None,
    }
    if inference is not None:
        result["inference"] = inference
    if center:
        centered = apply_center_to_viewer(viewer, source)
        if centered is not None:
            result["centered_on"] = centered
    return result


@mcp.tool()
def add_layers(
    layers: list[dict[str, Any]],
    center_on: Optional[str] = "first",
) -> dict[str, Any]:
    """Add many layers in a single transaction.

    Use this instead of calling `add_image_layer` / `add_segmentation_layer`
    in a loop when a dataset has many layers (OpenOrganelle volumes
    routinely ship 50–100 layers). One round trip instead of N.

    Each item in `layers` is a dict:
        {"name": str, "source": str,
         "type": "image"|"segmentation" (optional — auto-detected from
                 the source dtype if omitted, same rules as `add_layer`),
         "visible": bool (default True)}

    Auto-centering is applied once based on `center_on`:
        "first" — center on the first layer's bounds (default; matches
            the single-layer `add_image_layer(center=True)` behavior)
        "<name>" — center on the named layer specifically
        None — skip auto-centering entirely (use when you're restoring
            a saved view and already have a position via `load_state`)

    Args:
        layers: List of layer specs.
        center_on: How to choose the layer to center on, or None.

    Returns:
        {"added": [{name, type, source, visible}, ...],
         "centered_on": {layer, ...bounds...} | null,
         "errors": [{layer_index, name, error}, ...]}.
        Errors don't abort the batch; the rest still get added.
    """
    if center_on is not None and center_on != "first":
        if not any(layer.get("name") == center_on for layer in layers):
            raise ValueError(
                f"center_on={center_on!r} not found in this batch's layers"
            )

    viewer = get_viewer()
    added: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    center_source: Optional[str] = None
    center_layer_name: Optional[str] = None

    with viewer.txn() as s:
        for i, spec in enumerate(layers):
            try:
                name = spec["name"]
                source = spec["source"]
                ltype = spec.get("type")
                visible = spec.get("visible", True)
                inference: Optional[dict[str, Any]] = None
                if ltype is None:
                    inference = infer_layer_type(source)
                    ltype = inference["type"]
                if ltype == "image":
                    s.layers[name] = neuroglancer.ImageLayer(source=source)
                elif ltype == "segmentation":
                    s.layers[name] = neuroglancer.SegmentationLayer(source=source)
                else:
                    raise ValueError(
                        f"type must be 'image' or 'segmentation', got {ltype!r}"
                    )
                # Set visibility flag.
                for layer_obj in s.layers:
                    if layer_obj.name == name:
                        layer_obj.visible = visible
                        break
                entry: dict[str, Any] = {
                    "name": name, "type": ltype, "source": source, "visible": visible,
                }
                if inference is not None:
                    entry["inference"] = inference
                added.append(entry)
                if center_source is None and (
                    (center_on == "first" and i == 0)
                    or (center_on not in (None, "first") and name == center_on)
                ):
                    center_source = source
                    center_layer_name = name
            except Exception as e:  # noqa: BLE001
                errors.append(
                    {"layer_index": i, "name": spec.get("name"), "error": str(e)}
                )

    centered = None
    if center_source is not None:
        info = apply_center_to_viewer(viewer, center_source)
        if info is not None:
            centered = {"layer": center_layer_name, **info}

    return {"added": added, "centered_on": centered, "errors": errors}


@mcp.tool()
def set_layers_visibility(updates: dict[str, bool]) -> dict[str, Any]:
    """Bulk-toggle visibility for many layers in one transaction.

    Args:
        updates: Mapping `layer_name → visible` (bool). Names not in
            the viewer are reported in `missing` but don't abort.

    Returns:
        {"updated": [{name, visible}, ...], "missing": [name, ...]}.
    """
    viewer = get_viewer()
    updated: list[dict[str, Any]] = []
    missing: list[str] = []
    with viewer.txn() as s:
        present = {layer.name: layer for layer in s.layers}
        for name, visible in updates.items():
            layer = present.get(name)
            if layer is None:
                missing.append(name)
                continue
            layer.visible = bool(visible)
            updated.append({"name": name, "visible": layer.visible})
    return {"updated": updated, "missing": missing}


@mcp.tool()
def show_only_layers(names: list[str]) -> dict[str, Any]:
    """Show the named layers and hide every other layer.

    The most common "isolate this view" pattern — instead of toggling
    74 layers off and 1 on, just say which ones to keep visible.

    Args:
        names: Layer names to show. Anything else is hidden.

    Returns:
        {"shown": [name, ...], "hidden": [name, ...],
         "missing": [name, ...]}.
    """
    keep = set(names)
    viewer = get_viewer()
    shown: list[str] = []
    hidden: list[str] = []
    present: set[str] = set()
    with viewer.txn() as s:
        for layer in s.layers:
            present.add(layer.name)
            if layer.name in keep:
                layer.visible = True
                shown.append(layer.name)
            else:
                layer.visible = False
                hidden.append(layer.name)
    return {
        "shown": shown,
        "hidden": hidden,
        "missing": [n for n in names if n not in present],
    }


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
