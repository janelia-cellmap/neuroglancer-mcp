"""Bulk-layer tools: add_layers, set_layers_visibility, show_only_layers.

The motivation is OpenOrganelle-style datasets with 50+ layers: doing
N individual `add_image_layer` calls cost N round trips. These tools
collapse common patterns to one call.
"""

from __future__ import annotations

import pytest

from neuroglancer_mcp import bounds as bounds_mod
from neuroglancer_mcp import viewer as viewer_mod
from neuroglancer_mcp.tools import layers, state


@pytest.fixture(autouse=True)
def fresh_viewer():
    viewer_mod.reset_viewer()
    yield
    viewer_mod.reset_viewer()


@pytest.fixture(autouse=True)
def skip_auto_center(monkeypatch):
    """Stub the network-touching center computation by default.

    Tests that exercise centering opt back in by overwriting the stub.
    """
    monkeypatch.setattr(
        bounds_mod,
        "compute_layer_center",
        lambda url: (_ for _ in ()).throw(ValueError("stubbed off")),
    )


# ---------------------------------------------------------------------------
# add_layers
# ---------------------------------------------------------------------------


def test_add_layers_adds_all_in_one_call():
    specs = [
        {"name": "em", "source": "precomputed://gs://example/em", "type": "image"},
        {"name": "seg", "source": "precomputed://gs://example/seg", "type": "segmentation"},
        {"name": "labels", "source": "precomputed://gs://example/labels", "type": "segmentation"},
    ]
    result = layers.add_layers(specs)
    assert [a["name"] for a in result["added"]] == ["em", "seg", "labels"]
    assert result["errors"] == []
    snap = state.list_layers()
    assert [layer["name"] for layer in snap] == ["em", "seg", "labels"]


def test_add_layers_respects_visible_flag():
    specs = [
        {"name": "em", "source": "precomputed://gs://example/em", "type": "image", "visible": False},
        {"name": "seg", "source": "precomputed://gs://example/seg", "type": "segmentation"},
    ]
    layers.add_layers(specs)
    listed = {layer["name"]: layer for layer in state.list_layers()}
    assert listed["em"]["visible"] is False
    assert listed["seg"]["visible"] is True


def test_add_layers_invalid_type_reported_in_errors():
    specs = [
        {"name": "em", "source": "precomputed://gs://example/em", "type": "image"},
        {"name": "weird", "source": "precomputed://gs://example/x", "type": "annotation"},
        {"name": "seg", "source": "precomputed://gs://example/seg", "type": "segmentation"},
    ]
    result = layers.add_layers(specs)
    # The valid ones still went through.
    assert [a["name"] for a in result["added"]] == ["em", "seg"]
    assert len(result["errors"]) == 1
    assert result["errors"][0]["name"] == "weird"


def test_add_layers_center_on_specific_layer(monkeypatch):
    seen = {}

    def fake_compute(url: str):
        seen["url"] = url
        return {
            "format": "precomputed",
            "position_nm": [400.0, 800.0, 1600.0],
            "voxel_size_nm": [8.0, 8.0, 8.0],
            "shape_voxels": [100, 200, 400],
        }

    monkeypatch.setattr(bounds_mod, "compute_layer_center", fake_compute)
    result = layers.add_layers(
        [
            {"name": "em", "source": "precomputed://gs://example/em", "type": "image"},
            {"name": "seg", "source": "precomputed://gs://example/seg", "type": "segmentation"},
        ],
        center_on="seg",
    )
    assert result["centered_on"]["layer"] == "seg"
    # The fetcher saw the segmentation's URL, not the image's.
    assert seen["url"] == "precomputed://gs://example/seg"


def test_add_layers_center_on_none_skips_centering(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(
        bounds_mod,
        "compute_layer_center",
        lambda url: called.update(n=called["n"] + 1) or {},
    )
    layers.add_layers(
        [{"name": "em", "source": "precomputed://gs://example/em", "type": "image"}],
        center_on=None,
    )
    assert called["n"] == 0


def test_add_layers_center_on_unknown_layer_raises():
    with pytest.raises(ValueError, match="not found in this batch"):
        layers.add_layers(
            [{"name": "em", "source": "precomputed://gs://example/em", "type": "image"}],
            center_on="missing",
        )


# ---------------------------------------------------------------------------
# set_layers_visibility
# ---------------------------------------------------------------------------


def test_set_layers_visibility_bulk_toggle():
    layers.add_layers(
        [
            {"name": "em", "source": "precomputed://gs://example/em", "type": "image"},
            {"name": "seg", "source": "precomputed://gs://example/seg", "type": "segmentation"},
            {"name": "extra", "source": "precomputed://gs://example/x", "type": "image"},
        ]
    )
    result = layers.set_layers_visibility({"em": False, "seg": True, "extra": False})
    assert {u["name"]: u["visible"] for u in result["updated"]} == {
        "em": False,
        "seg": True,
        "extra": False,
    }
    assert result["missing"] == []


def test_set_layers_visibility_reports_missing():
    layers.add_layers(
        [{"name": "em", "source": "precomputed://gs://example/em", "type": "image"}]
    )
    result = layers.set_layers_visibility({"em": False, "ghost": True})
    assert result["updated"] == [{"name": "em", "visible": False}]
    assert result["missing"] == ["ghost"]


# ---------------------------------------------------------------------------
# show_only_layers
# ---------------------------------------------------------------------------


def test_show_only_layers_hides_everything_else():
    layers.add_layers(
        [
            {"name": "em", "source": "precomputed://gs://example/em", "type": "image"},
            {"name": "mito", "source": "precomputed://gs://example/mito", "type": "segmentation"},
            {"name": "er", "source": "precomputed://gs://example/er", "type": "segmentation"},
            {"name": "nucleus", "source": "precomputed://gs://example/nuc", "type": "segmentation"},
        ]
    )
    result = layers.show_only_layers(["em", "mito"])
    assert sorted(result["shown"]) == ["em", "mito"]
    assert sorted(result["hidden"]) == ["er", "nucleus"]
    assert result["missing"] == []
    visibility = {layer["name"]: layer["visible"] for layer in state.list_layers()}
    assert visibility == {"em": True, "mito": True, "er": False, "nucleus": False}


def test_show_only_layers_reports_missing_names():
    layers.add_layers(
        [{"name": "em", "source": "precomputed://gs://example/em", "type": "image"}]
    )
    result = layers.show_only_layers(["em", "absent"])
    assert result["shown"] == ["em"]
    assert result["missing"] == ["absent"]
