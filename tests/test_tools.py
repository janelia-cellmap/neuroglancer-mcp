"""Integration tests for Phase 1 tools.

These spin up a real `neuroglancer.Viewer` (the singleton is reset
before each test) and call the tool functions directly. We do not go
through the MCP transport — that's covered by Claude Desktop in
practice and would slow these tests down without testing useful logic.
"""

from __future__ import annotations

import pytest

from neuroglancer_mcp import viewer as viewer_mod
from neuroglancer_mcp.tools import layers, segments, state


@pytest.fixture(autouse=True)
def fresh_viewer():
    """Drop the cached viewer before and after each test."""
    viewer_mod.reset_viewer()
    yield
    viewer_mod.reset_viewer()


def test_list_layers_empty_initially():
    assert state.list_layers() == []


def test_add_image_layer_appears_in_list():
    result = layers.add_image_layer("em", "precomputed://gs://example/em")
    assert result == {
        "name": "em",
        "type": "image",
        "source": "precomputed://gs://example/em",
    }
    listed = state.list_layers()
    assert [layer["name"] for layer in listed] == ["em"]
    assert listed[0]["type"] == "image"


def test_add_segmentation_layer_and_remove():
    layers.add_segmentation_layer("seg", "precomputed://gs://example/seg")
    assert "seg" in [layer["name"] for layer in state.list_layers()]

    removed = layers.remove_layer("seg")
    assert removed == {"removed": "seg", "remaining": []}
    assert state.list_layers() == []


def test_set_layer_visibility_toggles_flag():
    layers.add_image_layer("em", "precomputed://gs://example/em")
    hidden = layers.set_layer_visibility("em", False)
    assert hidden == {"name": "em", "visible": False}
    shown = layers.set_layer_visibility("em", True)
    assert shown == {"name": "em", "visible": True}


def test_set_layer_visibility_missing_layer_raises():
    with pytest.raises(ValueError, match="not found"):
        layers.set_layer_visibility("nope", True)


def test_segment_show_add_hide_clear_roundtrip():
    layers.add_segmentation_layer("seg", "precomputed://gs://example/seg")

    shown = segments.show_segments("seg", [1, 2, 3])
    assert shown == {"layer": "seg", "visible_segments": [1, 2, 3]}

    added = segments.add_segments("seg", [3, 4, 5])
    assert added == {"layer": "seg", "visible_segments": [1, 2, 3, 4, 5]}

    hidden = segments.hide_segments("seg", [2, 4])
    assert hidden == {"layer": "seg", "visible_segments": [1, 3, 5]}

    cleared = segments.clear_segments("seg")
    assert cleared == {"layer": "seg", "visible_segments": []}


def test_segment_tools_reject_non_segmentation_layer():
    layers.add_image_layer("em", "precomputed://gs://example/em")
    with pytest.raises(ValueError, match="not a segmentation layer"):
        segments.show_segments("em", [1])


def test_segment_tools_reject_missing_layer():
    with pytest.raises(ValueError, match="not found"):
        segments.show_segments("missing", [1])


def test_state_roundtrip_preserves_layers():
    layers.add_image_layer("em", "precomputed://gs://example/em")
    layers.add_segmentation_layer("seg", "precomputed://gs://example/seg")
    snapshot = state.get_state()

    # Drop the viewer and rebuild from the captured state.
    viewer_mod.reset_viewer()
    assert state.list_layers() == []

    state.load_state(snapshot)
    names = sorted(layer["name"] for layer in state.list_layers())
    assert names == ["em", "seg"]


def test_get_url_returns_nonempty_string():
    url = state.get_url()
    assert "url" in url
    assert isinstance(url["url"], str)
    assert url["url"]


def test_share_url_points_at_demo_appspot_and_encodes_state():
    layers.add_image_layer("em", "precomputed://gs://example/em")
    shared = state.share_url()
    assert shared["url"].startswith("https://neuroglancer-demo.appspot.com")
    # The source URL is preserved in the encoded state fragment.
    assert "precomputed" in shared["url"]
