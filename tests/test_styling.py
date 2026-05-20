"""Tests for the colors & opacity tools.

Real viewer (singleton reset between cases), no network. Each test
mutates one styling field and reads it back via `get_state` to confirm
the change survived the txn.
"""

from __future__ import annotations

import pytest

from neuroglancer_mcp import viewer as viewer_mod
from neuroglancer_mcp.tools import layers, state, styling


@pytest.fixture(autouse=True)
def fresh_viewer():
    viewer_mod.reset_viewer()
    yield
    viewer_mod.reset_viewer()


def _add_seg(name: str = "seg") -> None:
    layers.add_segmentation_layer(name, "precomputed://gs://example/seg")


def _add_image(name: str = "em") -> None:
    layers.add_image_layer(name, "precomputed://gs://example/em")


# ---------------------------------------------------------------------------
# set_segment_colors
# ---------------------------------------------------------------------------


def test_set_segment_colors_applies_overrides():
    _add_seg()
    result = styling.set_segment_colors("seg", {1: "#ff0000", 2: "#00ff00"})
    assert result["segment_colors"] == {1: "#ff0000", 2: "#00ff00"}


def test_set_segment_colors_normalizes_short_hex():
    _add_seg()
    result = styling.set_segment_colors("seg", {1: "#f00", 2: "0f0"})
    assert result["segment_colors"] == {1: "#ff0000", 2: "#00ff00"}


def test_set_segment_colors_merges_with_existing():
    _add_seg()
    styling.set_segment_colors("seg", {1: "#ff0000"})
    result = styling.set_segment_colors("seg", {2: "#00ff00"})
    assert result["segment_colors"] == {1: "#ff0000", 2: "#00ff00"}


def test_set_segment_colors_replace_clears_prior():
    _add_seg()
    styling.set_segment_colors("seg", {1: "#ff0000", 2: "#00ff00"})
    result = styling.set_segment_colors("seg", {3: "#0000ff"}, replace=True)
    assert result["segment_colors"] == {3: "#0000ff"}


def test_set_segment_colors_empty_with_replace_clears_all():
    _add_seg()
    styling.set_segment_colors("seg", {1: "#ff0000"})
    result = styling.set_segment_colors("seg", {}, replace=True)
    assert result["segment_colors"] == {}


def test_set_segment_colors_rejects_invalid_hex():
    _add_seg()
    with pytest.raises(ValueError, match="hex code"):
        styling.set_segment_colors("seg", {1: "red"})


def test_set_segment_colors_rejects_image_layer():
    _add_image()
    with pytest.raises(ValueError, match="not a segmentation layer"):
        styling.set_segment_colors("em", {1: "#ff0000"})


# ---------------------------------------------------------------------------
# set_layer_color (segmentation default)
# ---------------------------------------------------------------------------


def test_set_layer_color_sets_default():
    _add_seg()
    result = styling.set_layer_color("seg", "#abcdef")
    assert result["default_color"] == "#abcdef"


def test_set_layer_color_empty_clears():
    _add_seg()
    styling.set_layer_color("seg", "#abcdef")
    result = styling.set_layer_color("seg", "")
    assert result["default_color"] is None


# ---------------------------------------------------------------------------
# set_layer_opacity
# ---------------------------------------------------------------------------


def test_opacity_segmentation_sets_object_alpha():
    _add_seg()
    result = styling.set_layer_opacity("seg", 0.25)
    assert result == {"layer": "seg", "type": "segmentation", "alpha": 0.25}
    # Round-trip through get_state to confirm field name.
    snap = state.get_state()
    seg_layer = next(layer for layer in snap["layers"] if layer["name"] == "seg")
    assert seg_layer["objectAlpha"] == 0.25


def test_opacity_image_sets_layer_opacity():
    _add_image()
    result = styling.set_layer_opacity("em", 0.75)
    assert result == {"layer": "em", "type": "image", "alpha": 0.75}
    snap = state.get_state()
    img_layer = next(layer for layer in snap["layers"] if layer["name"] == "em")
    assert img_layer["opacity"] == 0.75


def test_opacity_out_of_range_raises():
    _add_seg()
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        styling.set_layer_opacity("seg", 1.5)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        styling.set_layer_opacity("seg", -0.1)


# ---------------------------------------------------------------------------
# set_mesh_silhouette
# ---------------------------------------------------------------------------


def test_set_mesh_silhouette_round_trip():
    _add_seg()
    result = styling.set_mesh_silhouette("seg", 2.5)
    assert result["mesh_silhouette_rendering"] == 2.5
    snap = state.get_state()
    seg_layer = next(layer for layer in snap["layers"] if layer["name"] == "seg")
    assert seg_layer["meshSilhouetteRendering"] == 2.5


def test_set_mesh_silhouette_zero_disables():
    _add_seg()
    styling.set_mesh_silhouette("seg", 3.0)
    result = styling.set_mesh_silhouette("seg", 0.0)
    assert result["mesh_silhouette_rendering"] == 0.0


def test_set_mesh_silhouette_rejects_negative():
    _add_seg()
    with pytest.raises(ValueError, match=">= 0"):
        styling.set_mesh_silhouette("seg", -1.0)


def test_set_mesh_silhouette_rejects_image_layer():
    _add_image()
    with pytest.raises(ValueError, match="not a segmentation layer"):
        styling.set_mesh_silhouette("em", 1.0)


# ---------------------------------------------------------------------------
# arrange_view & set_background_color
# ---------------------------------------------------------------------------


def test_arrange_view_native_layout():
    result = styling.arrange_view("xy-3d")
    assert result == {"layout": "xy-3d"}
    assert state.get_state()["layout"] == "xy-3d"


def test_arrange_view_alias_resolves():
    assert styling.arrange_view("3d_only")["layout"] == "3d"
    assert styling.arrange_view("quad")["layout"] == "4panel"


def test_arrange_view_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown layout"):
        styling.arrange_view("octopus")


def test_set_background_color_both():
    result = styling.set_background_color("#ffffff")
    assert result["cross_section_background_color"] == "#ffffff"
    assert result["projection_background_color"] == "#ffffff"


def test_set_background_color_only_3d():
    styling.set_background_color("#000000")  # set both first
    result = styling.set_background_color("#ff0000", panel="3d")
    assert result["projection_background_color"] == "#ff0000"
    assert result["cross_section_background_color"] == "#000000"


def test_set_background_color_clear_with_empty():
    styling.set_background_color("#ffffff")
    result = styling.set_background_color("")
    assert result["cross_section_background_color"] is None
    assert result["projection_background_color"] is None


def test_set_background_color_rejects_bad_panel():
    with pytest.raises(ValueError, match="panel must be"):
        styling.set_background_color("#ffffff", panel="frontal")


# ---------------------------------------------------------------------------
# Image shaders
# ---------------------------------------------------------------------------


def test_set_image_shader_round_trip():
    _add_image()
    src = "void main() { emitGrayscale(0.5); }"
    result = styling.set_image_shader("em", src)
    assert result["shader"] == src


def test_set_image_shader_rejects_segmentation_layer():
    _add_seg()
    with pytest.raises(ValueError, match="not an image layer"):
        styling.set_image_shader("seg", "void main() {}")


@pytest.mark.parametrize(
    "preset", ["grayscale", "red", "green", "blue", "fire", "viridis"]
)
def test_set_image_preset_emits_valid_glsl(preset):
    _add_image()
    result = styling.set_image_preset("em", preset)
    assert result["preset"] == preset
    # Every preset's shader should at minimum declare a main() that emits.
    assert "void main()" in result["shader"]
    assert "emit" in result["shader"]


def test_set_image_preset_threshold_embeds_value():
    _add_image()
    result = styling.set_image_preset("em", "threshold", value=0.25)
    assert "0.250000" in result["shader"]


def test_set_image_preset_threshold_out_of_range_raises():
    _add_image()
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        styling.set_image_preset("em", "threshold", value=2.0)


def test_set_image_preset_unknown_raises():
    _add_image()
    with pytest.raises(ValueError, match="Unknown preset"):
        styling.set_image_preset("em", "monochrome")


def test_set_contrast_window_round_trip():
    _add_image()
    result = styling.set_contrast("em", 0.1, 0.9)
    assert result["min"] == 0.1 and result["max"] == 0.9
    # The shader should reference both endpoints.
    assert "0.100000" in result["shader"] and "0.900000" in result["shader"]


def test_set_contrast_rejects_inverted_window():
    _add_image()
    with pytest.raises(ValueError, match="greater than"):
        styling.set_contrast("em", 0.9, 0.1)


# ---------------------------------------------------------------------------
# Segmentation rendering mode + mesh LOD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, sel, obj",
    [("voxel", 0.5, 0.0), ("mesh", 0.0, 1.0), ("both", 0.5, 1.0)],
)
def test_set_segmentation_rendering_modes(mode, sel, obj):
    _add_seg()
    result = styling.set_segmentation_rendering("seg", mode)
    assert result["selected_alpha"] == sel
    assert result["object_alpha"] == obj


def test_set_segmentation_rendering_unknown_raises():
    _add_seg()
    with pytest.raises(ValueError, match="mode must be"):
        styling.set_segmentation_rendering("seg", "wireframe")


def test_set_mesh_resolution_round_trip():
    _add_seg()
    result = styling.set_mesh_resolution("seg", 5.0)
    assert result["mesh_render_scale"] == 5.0


def test_set_mesh_resolution_rejects_non_positive():
    _add_seg()
    with pytest.raises(ValueError, match="> 0"):
        styling.set_mesh_resolution("seg", 0.0)
