"""Phase 3 — colors and opacity tools.

These mutate the visual properties of layers and segments without
changing what's loaded. They map directly to Neuroglancer state fields
that the JSON viewer state already supports, so the differentiator
isn't capability but ergonomics: "make neuron 1234 bright red" or "50%
transparent" turns into one tool call instead of three menus and
fifteen clicks.
"""

from __future__ import annotations

from typing import Any

import neuroglancer

from neuroglancer_mcp.server import mcp
from neuroglancer_mcp.viewer import get_viewer


def _get_layer(s: Any, name: str) -> Any:
    """Return the underlying layer object (ImageLayer/SegmentationLayer) by name."""
    for layer in s.layers:
        if layer.name == name:
            return layer.layer
    raise ValueError(f"Layer {name!r} not found")


def _get_seg_layer(s: Any, name: str) -> neuroglancer.SegmentationLayer:
    layer = _get_layer(s, name)
    if not isinstance(layer, neuroglancer.SegmentationLayer):
        raise ValueError(f"Layer {name!r} is not a segmentation layer")
    return layer


def _normalize_hex(color: str) -> str:
    """Canonicalize a hex color string.

    Accepts "#rgb", "#rrggbb", "rgb", "rrggbb" (case-insensitive); also
    "" for "clear". Returns "#rrggbb" form (or "" if cleared).
    Neuroglancer's state accepts a few formats; we normalize so the
    return value is predictable.
    """
    if color == "":
        return ""
    c = color.strip().lower()
    if c.startswith("#"):
        c = c[1:]
    if len(c) == 3 and all(ch in "0123456789abcdef" for ch in c):
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6 or not all(ch in "0123456789abcdef" for ch in c):
        raise ValueError(
            f"Color {color!r} is not a hex code "
            "(expected '#rgb', '#rrggbb', or '')"
        )
    return "#" + c


# ---------------------------------------------------------------------------
# Per-segment color overrides
# ---------------------------------------------------------------------------


@mcp.tool()
def set_segment_colors(
    layer: str, color_map: dict[int, str], replace: bool = False
) -> dict[str, Any]:
    """Override the colors of specific segments in a segmentation layer.

    Use this for "make neuron 1234 red" style requests. Existing
    overrides for other segments are preserved by default — pass
    `replace=True` to clear them and apply only `color_map`. Pass an
    empty `color_map` with `replace=True` to clear all overrides.

    Segments without an explicit override are colored by either the
    layer's `default_color` (see `set_layer_color`) or, if none is
    set, Neuroglancer's hash-of-segment-id coloring.

    Args:
        layer: Segmentation layer name.
        color_map: Mapping segment_id (int) → hex color string
            ("#ff0000", "#f00", or "ff0000"). Empty values like ""
            in the map are not supported — to clear a single segment's
            override, omit it and call with `replace=True` if needed.
        replace: If True, clear existing overrides before applying.

    Returns:
        {"layer", "segment_colors": {<id>: "#rrggbb", ...}} — the
        full resulting color map.
    """
    normalized = {int(sid): _normalize_hex(c) for sid, c in color_map.items()}
    if any(v == "" for v in normalized.values()):
        raise ValueError("Empty colors are not allowed in color_map values")

    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        if replace:
            seg.segment_colors = {}
        for sid, color in normalized.items():
            seg.segment_colors[sid] = color
        return {
            "layer": layer,
            "segment_colors": {int(k): v for k, v in seg.segment_colors.items()},
        }


@mcp.tool()
def set_layer_color(layer: str, color: str) -> dict[str, Any]:
    """Set the default color for segments in a segmentation layer.

    Applied to any segment without an explicit override from
    `set_segment_colors`. Pass an empty string ("") to clear the
    default and fall back to Neuroglancer's hash-based per-segment
    coloring. Only segmentation layers — for image-layer tinting,
    use the (Phase 3 group 2) shader tools.

    Args:
        layer: Segmentation layer name.
        color: Hex color ("#ff0000", "#f00", "ff0000"), or "" to clear.

    Returns:
        {"layer", "default_color": "#rrggbb" or None}.
    """
    normalized = _normalize_hex(color)
    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        seg.segment_default_color = normalized if normalized else None
        return {
            "layer": layer,
            "default_color": seg.segment_default_color,
        }


# ---------------------------------------------------------------------------
# Opacity
# ---------------------------------------------------------------------------


@mcp.tool()
def set_layer_opacity(layer: str, alpha: float) -> dict[str, Any]:
    """Set the opacity of a layer (0.0 transparent to 1.0 opaque).

    Mapping:
    - Image layer → sets `opacity` (default 0.5).
    - Segmentation layer → sets `object_alpha`, the 3D mesh opacity
      (default 1.0). This is what users mean by "make the segments
      50% transparent"; the cross-section alphas (`selected_alpha`,
      `not_selected_alpha`) are separate and not changed here.

    Args:
        layer: Layer name.
        alpha: Opacity in [0.0, 1.0].

    Returns:
        {"layer", "type", "alpha"}.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    viewer = get_viewer()
    with viewer.txn() as s:
        layer_obj = _get_layer(s, layer)
        if isinstance(layer_obj, neuroglancer.ImageLayer):
            layer_obj.opacity = alpha
            return {"layer": layer, "type": "image", "alpha": layer_obj.opacity}
        if isinstance(layer_obj, neuroglancer.SegmentationLayer):
            layer_obj.object_alpha = alpha
            return {
                "layer": layer,
                "type": "segmentation",
                "alpha": layer_obj.object_alpha,
            }
        raise ValueError(
            f"Layer {layer!r} is type {type(layer_obj).__name__}; "
            "set_layer_opacity only supports image and segmentation layers"
        )


# ---------------------------------------------------------------------------
# Shaders (image layers)
# ---------------------------------------------------------------------------


def _channel_shader(channel: str) -> str:
    """GLSL that outputs the normalized data value in one RGB channel."""
    slot = {"red": "value, 0.0, 0.0", "green": "0.0, value, 0.0", "blue": "0.0, 0.0, value"}[channel]
    return (
        "void main() {\n"
        "  float value = toNormalized(getDataValue());\n"
        f"  emitRGB(vec3({slot}));\n"
        "}\n"
    )


def _grayscale_shader() -> str:
    return (
        "void main() {\n"
        "  emitGrayscale(toNormalized(getDataValue()));\n"
        "}\n"
    )


def _fire_shader() -> str:
    # Heat colormap: black → red → orange → yellow → white.
    return (
        "void main() {\n"
        "  float v = toNormalized(getDataValue());\n"
        "  float r = clamp(v * 3.0, 0.0, 1.0);\n"
        "  float g = clamp(v * 3.0 - 1.0, 0.0, 1.0);\n"
        "  float b = clamp(v * 3.0 - 2.0, 0.0, 1.0);\n"
        "  emitRGB(vec3(r, g, b));\n"
        "}\n"
    )


def _viridis_shader() -> str:
    # Cubic polynomial approximation to matplotlib's viridis; close
    # enough for figure use without shipping a lookup texture.
    return (
        "void main() {\n"
        "  float t = toNormalized(getDataValue());\n"
        "  float r = 0.267 + t*(-1.030 + t*(3.130 + t*-1.435));\n"
        "  float g = 0.005 + t*(1.404 + t*(-0.534 + t*0.030));\n"
        "  float b = 0.329 + t*(1.385 + t*(-3.061 + t*1.230));\n"
        "  emitRGB(vec3(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0)));\n"
        "}\n"
    )


def _threshold_shader(value: float) -> str:
    return (
        "void main() {\n"
        f"  float v = toNormalized(getDataValue());\n"
        f"  emitGrayscale(v > {value:.6f} ? 1.0 : 0.0);\n"
        "}\n"
    )


def _contrast_shader(lo: float, hi: float) -> str:
    return (
        "void main() {\n"
        "  float v = toNormalized(getDataValue());\n"
        f"  float scaled = clamp((v - {lo:.6f}) / ({hi:.6f} - {lo:.6f}), 0.0, 1.0);\n"
        "  emitGrayscale(scaled);\n"
        "}\n"
    )


IMAGE_PRESETS = ("grayscale", "red", "green", "blue", "fire", "viridis", "threshold")


def _get_image_layer(s: Any, name: str) -> neuroglancer.ImageLayer:
    layer = _get_layer(s, name)
    if not isinstance(layer, neuroglancer.ImageLayer):
        raise ValueError(f"Layer {name!r} is not an image layer")
    return layer


@mcp.tool()
def set_image_shader(layer: str, shader_glsl: str) -> dict[str, Any]:
    """Set a raw GLSL shader on an image layer.

    Neuroglancer shaders are GLSL fragments with a `main()` that calls
    `emitRGB(vec3)`, `emitRGBA(vec4)`, or `emitGrayscale(float)`. The
    normalized data value is available via `toNormalized(getDataValue())`.
    See https://github.com/google/neuroglancer/blob/master/src/sliceview/image_layer_rendering.md
    for the full shader API.

    For named presets ("grayscale", "fire", "viridis", etc.) prefer
    `set_image_preset` — it gives you the same effect without writing
    GLSL by hand.

    Args:
        layer: Image layer name.
        shader_glsl: GLSL source code (the full shader, including
            `void main() { ... }`).

    Returns:
        {"layer", "shader": <the GLSL source you set>}.
    """
    viewer = get_viewer()
    with viewer.txn() as s:
        img = _get_image_layer(s, layer)
        img.shader = shader_glsl
        return {"layer": layer, "shader": img.shader}


@mcp.tool()
def set_image_preset(
    layer: str, preset: str, value: float = 0.5
) -> dict[str, Any]:
    """Apply a named shader preset to an image layer.

    Available presets:
    - "grayscale" — Neuroglancer's default.
    - "red", "green", "blue" — single-channel tints; useful for
      multi-channel overlays.
    - "fire" — black→red→orange→yellow→white heatmap.
    - "viridis" — perceptually-uniform colormap (approximate cubic fit).
    - "threshold" — binary mask above a normalized cutoff. The
      `value` argument is the cutoff in [0,1]; default 0.5.

    Args:
        layer: Image layer name.
        preset: One of the names above.
        value: Used only by "threshold" — normalized cutoff in [0, 1].

    Returns:
        {"layer", "preset", "shader": <the GLSL applied>}.
    """
    if preset not in IMAGE_PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Available: {IMAGE_PRESETS}"
        )

    if preset == "grayscale":
        shader = _grayscale_shader()
    elif preset in {"red", "green", "blue"}:
        shader = _channel_shader(preset)
    elif preset == "fire":
        shader = _fire_shader()
    elif preset == "viridis":
        shader = _viridis_shader()
    elif preset == "threshold":
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold value must be in [0, 1], got {value}")
        shader = _threshold_shader(value)
    else:  # pragma: no cover - guarded by the membership check above
        raise AssertionError(preset)

    viewer = get_viewer()
    with viewer.txn() as s:
        img = _get_image_layer(s, layer)
        img.shader = shader
        return {"layer": layer, "preset": preset, "shader": img.shader}


@mcp.tool()
def set_contrast(layer: str, min: float, max: float) -> dict[str, Any]:
    """Apply a linear-contrast shader to an image layer.

    Maps the normalized data range [min, max] to [0, 1] grayscale,
    clamping outside the window. Useful for "brighten the EM" / "dim
    the membrane channel" requests.

    Args:
        layer: Image layer name.
        min: Lower edge of the window, in normalized data units [0, 1].
        max: Upper edge of the window, in normalized data units [0, 1].
            Must be strictly greater than `min`.

    Returns:
        {"layer", "min", "max", "shader": <the GLSL applied>}.
    """
    if max <= min:
        raise ValueError(f"max ({max}) must be greater than min ({min})")
    shader = _contrast_shader(min, max)
    viewer = get_viewer()
    with viewer.txn() as s:
        img = _get_image_layer(s, layer)
        img.shader = shader
        return {"layer": layer, "min": min, "max": max, "shader": img.shader}


# ---------------------------------------------------------------------------
# Mesh silhouette
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# View arrangement
# ---------------------------------------------------------------------------


_NATIVE_LAYOUTS = (
    "xy",
    "yz",
    "xz",
    "xy-3d",
    "yz-3d",
    "xz-3d",
    "4panel",
    "4panel-alt",
    "3d",
)

_LAYOUT_ALIASES = {
    # Semantic names from CLAUDE.md → Neuroglancer's native layout strings.
    "3d_only": "3d",
    "quad": "4panel",
    "top_xy": "xy-3d",
    # "cross_section_only" doesn't have a native multi-axis equivalent,
    # so we map it to the XY plane (the most common single cross-section).
    "cross_section_only": "xy",
}


@mcp.tool()
def arrange_view(preset: str) -> dict[str, Any]:
    """Switch the viewer panel layout.

    Accepts either a native Neuroglancer layout name or a semantic
    alias:

    Native (mapped 1:1):
        "xy", "yz", "xz" — single cross-section, no 3D
        "xy-3d", "yz-3d", "xz-3d" — one cross-section + 3D
        "4panel", "4panel-alt" — all three cross-sections + 3D
        "3d" — 3D only

    Aliases:
        "3d_only" → "3d"
        "quad" → "4panel"
        "top_xy" → "xy-3d"
        "cross_section_only" → "xy"  (NG has no "all cross-sections, no 3D" layout)

    Args:
        preset: One of the names above.

    Returns:
        {"layout": "<resolved layout name>"}.
    """
    resolved = _LAYOUT_ALIASES.get(preset, preset)
    if resolved not in _NATIVE_LAYOUTS:
        raise ValueError(
            f"Unknown layout {preset!r}. Native: {_NATIVE_LAYOUTS}. "
            f"Aliases: {sorted(_LAYOUT_ALIASES.keys())}"
        )
    viewer = get_viewer()
    with viewer.txn() as s:
        s.layout = resolved
        # `s.layout` is a DataPanelLayout wrapper; we want the bare type string.
        layout_name = getattr(s.layout, "type", None) or str(s.layout)
        return {"layout": layout_name}


@mcp.tool()
def set_background_color(color: str, panel: str = "both") -> dict[str, Any]:
    """Set the background color of cross-section and/or 3D panels.

    The default Neuroglancer background is black; setting white is a
    common figure-making move.

    Args:
        color: Hex color ("#ffffff", "#fff", "ffffff"), or "" to revert
            to the Neuroglancer default.
        panel: "cross_section", "3d", or "both" (default).

    Returns:
        {"cross_section_background_color", "projection_background_color"}
        — the values after the change (None if cleared).
    """
    if panel not in {"cross_section", "3d", "both"}:
        raise ValueError(
            f"panel must be 'cross_section', '3d', or 'both', got {panel!r}"
        )
    normalized = _normalize_hex(color)
    value = normalized if normalized else None

    viewer = get_viewer()
    with viewer.txn() as s:
        if panel in {"cross_section", "both"}:
            s.cross_section_background_color = value
        if panel in {"3d", "both"}:
            s.projection_background_color = value
        return {
            "cross_section_background_color": s.cross_section_background_color,
            "projection_background_color": s.projection_background_color,
        }


# ---------------------------------------------------------------------------
# Segmentation rendering mode + mesh LOD
# ---------------------------------------------------------------------------


@mcp.tool()
def set_segmentation_rendering(layer: str, mode: str) -> dict[str, Any]:
    """Control whether segments render as cross-section voxels, 3D mesh, or both.

    Neuroglancer doesn't have a single "mode" enum — it has independent
    cross-section and mesh alphas. This tool sets them together to
    achieve the three common configurations.

    Modes:
        "voxel" — cross-section voxels visible, 3D mesh hidden.
          (selected_alpha=0.5, object_alpha=0.0)
        "mesh"  — 3D mesh visible, cross-section voxels hidden.
          (selected_alpha=0.0, object_alpha=1.0)
        "both"  — both visible (the default state).
          (selected_alpha=0.5, object_alpha=1.0)

    For fine-grained control, use `set_layer_opacity` (which sets
    object_alpha) and the underlying selected_alpha field directly via
    `load_state` if you need an unusual combination.

    Args:
        layer: Segmentation layer name.
        mode: "voxel", "mesh", or "both".

    Returns:
        {"layer", "mode", "selected_alpha", "object_alpha"}.
    """
    modes = {
        "voxel": (0.5, 0.0),
        "mesh": (0.0, 1.0),
        "both": (0.5, 1.0),
    }
    if mode not in modes:
        raise ValueError(
            f"mode must be one of {sorted(modes)}, got {mode!r}"
        )
    sel, obj = modes[mode]

    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        seg.selected_alpha = sel
        seg.object_alpha = obj
        return {
            "layer": layer,
            "mode": mode,
            "selected_alpha": seg.selected_alpha,
            "object_alpha": seg.object_alpha,
        }


@mcp.tool()
def set_mesh_resolution(layer: str, render_scale: float) -> dict[str, Any]:
    """Set the mesh level-of-detail threshold for a segmentation layer.

    Maps to Neuroglancer's `meshRenderScale`. Lower values fetch finer
    LODs (more detail, more bandwidth, slower); higher values use
    coarser LODs (less detail, faster). Default is 10. Useful tuning
    points: ~5 for high-quality figures, ~20+ for fast exploration.

    Args:
        layer: Segmentation layer name.
        render_scale: Positive number. Lower = finer meshes loaded.

    Returns:
        {"layer", "mesh_render_scale"}.
    """
    if render_scale <= 0:
        raise ValueError(f"render_scale must be > 0, got {render_scale}")
    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        seg.mesh_render_scale = render_scale
        return {"layer": layer, "mesh_render_scale": seg.mesh_render_scale}


# ---------------------------------------------------------------------------
# Mesh silhouette
# ---------------------------------------------------------------------------


@mcp.tool()
def set_mesh_silhouette(layer: str, value: float) -> dict[str, Any]:
    """Set mesh silhouette rendering intensity for a segmentation layer.

    Silhouette rendering darkens the edges of meshes facing away from
    the camera. 0 disables it (default). Typical useful values are
    1.0–5.0 for clean figure-quality outlines; higher values can make
    the mesh look cartoonish.

    Args:
        layer: Segmentation layer name.
        value: Silhouette intensity. 0 disables; values up to ~5 are
            common for publication figures.

    Returns:
        {"layer", "mesh_silhouette_rendering"}.
    """
    if value < 0:
        raise ValueError(f"silhouette value must be >= 0, got {value}")
    viewer = get_viewer()
    with viewer.txn() as s:
        seg = _get_seg_layer(s, layer)
        seg.mesh_silhouette_rendering = value
        return {
            "layer": layer,
            "mesh_silhouette_rendering": seg.mesh_silhouette_rendering,
        }
