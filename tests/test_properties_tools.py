"""Tests for the segment_properties MCP tools.

These spin up a real `neuroglancer.Viewer` (so layer-source extraction
is exercised end-to-end) but monkeypatch `_fetch_json` so no network
hit is required. A `network`-marked integration test hits hemibrain;
it's skipped unless `-m network` is passed.
"""

from __future__ import annotations

import pytest

from neuroglancer_mcp import meshes as meshes_mod
from neuroglancer_mcp import segment_properties as sp_mod
from neuroglancer_mcp import viewer as viewer_mod
from neuroglancer_mcp.tools import layers, navigation, properties, segments, state


def _fixture_info() -> dict:
    return {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": ["100", "200", "300"],
            "properties": [
                {
                    "id": "label",
                    "type": "label",
                    "values": ["KC_alpha", "KC_beta", "MBON14"],
                },
                {
                    "id": "size",
                    "type": "number",
                    "data_type": "uint64",
                    "values": [1500, 500, 9000],
                },
                {
                    "id": "tags",
                    "type": "tags",
                    "tags": ["Traced", "KC_class", "MBON_class"],
                    "values": [[0, 1], [1], [0, 2]],
                },
            ],
        },
    }


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Reset viewer + segment_properties cache; stub the JSON fetcher."""
    viewer_mod.reset_viewer()
    sp_mod.clear_cache()
    monkeypatch.setattr(sp_mod, "_fetch_json", lambda url: _fixture_info())
    yield
    viewer_mod.reset_viewer()
    sp_mod.clear_cache()


def _add_seg(name: str = "seg", source: str = "precomputed://gs://example/seg") -> None:
    layers.add_segmentation_layer(name, source)


# ---------------------------------------------------------------------------
# Schema + single-property fetch
# ---------------------------------------------------------------------------


def test_get_segment_properties_schema():
    _add_seg()
    schema = properties.get_segment_properties_schema("seg")
    assert schema["num_segments"] == 3
    types = {p["id"]: p["type"] for p in schema["properties"]}
    assert types == {"label": "label", "size": "number", "tags": "tags"}


def test_get_segment_property_label():
    _add_seg()
    result = properties.get_segment_property("seg", 100, "label")
    assert result == {
        "layer": "seg",
        "segment_id": 100,
        "property_id": "label",
        "value": "KC_alpha",
    }


def test_get_segment_property_tags_returns_names():
    _add_seg()
    result = properties.get_segment_property("seg", 100, "tags")
    assert result["value"] == ["Traced", "KC_class"]


# ---------------------------------------------------------------------------
# Search & filter
# ---------------------------------------------------------------------------


def test_search_segments_by_label_substring():
    _add_seg()
    out = properties.search_segments_by_label("seg", "kc")
    assert out["total_matches"] == 2
    assert out["returned"] == 2
    assert out["truncated"] is False
    assert {m["id"] for m in out["matches"]} == {100, 200}
    assert out["regex"] is False


def test_search_segments_by_label_respects_limit():
    _add_seg()
    out = properties.search_segments_by_label("seg", "kc", limit=1)
    assert out["total_matches"] == 2
    assert out["returned"] == 1
    assert out["truncated"] is True
    assert len(out["matches"]) == 1


def test_search_segments_by_label_regex():
    _add_seg()
    out = properties.search_segments_by_label("seg", r"^MBON\d+$", regex=True)
    assert [m["id"] for m in out["matches"]] == [300]


def test_filter_segments_combined():
    _add_seg()
    out = properties.filter_segments(
        "seg",
        {"label": "KC*", "tags": ["Traced"], "size": {"min": 1000}},
    )
    assert out["total_matches"] == 1
    assert out["returned"] == 1
    assert out["truncated"] is False
    assert out["matching_segment_ids"] == [100]


def test_top_n_segments_by_property():
    _add_seg()
    out = properties.top_n_segments_by_property("seg", "size", 2)
    assert out["results"] == [
        {"id": 300, "value": 9000},
        {"id": 100, "value": 1500},
    ]


# ---------------------------------------------------------------------------
# Convenience: filter + show
# ---------------------------------------------------------------------------


def test_show_segments_by_property_updates_viewer():
    _add_seg()
    out = properties.show_segments_by_property("seg", {"label": "KC*"})
    assert out["num_visible"] == 2
    assert out["visible_segments"] == [100, 200]

    # Verify the viewer's segments set was actually updated.
    listed = state.list_layers()
    assert listed[0]["name"] == "seg"
    # Round-trip through get_state to confirm:
    snapshot = state.get_state()
    seg_layer = next(layer for layer in snapshot["layers"] if layer["name"] == "seg")
    assert sorted(int(s) for s in seg_layer["segments"]) == [100, 200]


def test_show_segments_by_property_overwrites_prior_selection():
    _add_seg()
    segments.show_segments("seg", [42])
    out = properties.show_segments_by_property("seg", {"label": "MBON*"})
    assert out["visible_segments"] == [300]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_schema_on_image_layer_raises():
    layers.add_image_layer("em", "precomputed://gs://example/em")
    with pytest.raises(ValueError, match="not a segmentation layer"):
        properties.get_segment_properties_schema("em")


def test_schema_on_missing_layer_raises():
    with pytest.raises(ValueError, match="not found"):
        properties.get_segment_properties_schema("nope")


# ---------------------------------------------------------------------------
# Caching: a second call does not refetch
# ---------------------------------------------------------------------------


def test_load_segment_properties_is_cached(monkeypatch):
    _add_seg()
    calls = {"n": 0}

    def counting_fetch(url: str):
        calls["n"] += 1
        return _fixture_info()

    sp_mod.clear_cache()
    monkeypatch.setattr(sp_mod, "_fetch_json", counting_fetch)

    properties.get_segment_properties_schema("seg")
    properties.get_segment_properties_schema("seg")
    properties.search_segments_by_label("seg", "kc")

    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Network integration test (skipped by default — `pytest -m network`)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# center_on_segment
# ---------------------------------------------------------------------------


def test_center_on_segment_moves_viewer(monkeypatch):
    _add_seg()

    def fake_centroid(source: str, segment_id: int) -> dict:
        return {
            "position": [100.0, 200.0, 300.0],
            "voxel_size_nm": [8.0, 8.0, 8.0],
            "position_nm": [800.0, 1600.0, 2400.0],
            "mesh_type": "neuroglancer_multilod_draco",
        }

    monkeypatch.setattr(
        "neuroglancer_mcp.tools.navigation.estimate_segment_centroid",
        fake_centroid,
    )
    result = navigation.center_on_segment("seg", 42)
    assert result["mesh_type"] == "neuroglancer_multilod_draco"
    assert result["position_nm"] == [800.0, 1600.0, 2400.0]
    # The viewer's position should now be the voxel coords we returned.
    assert [float(v) for v in result["position"]] == [100.0, 200.0, 300.0]


def test_center_on_segment_rejects_image_layer():
    layers.add_image_layer("em", "precomputed://gs://example/em")
    with pytest.raises(ValueError, match="not a segmentation layer"):
        navigation.center_on_segment("em", 42)


# ---------------------------------------------------------------------------
# Network integration tests (skipped by default — `pytest -m network`)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_hemibrain_center_on_kenyon_cell():
    """End-to-end: real hemibrain murmurhash3-sharded mesh manifest decode.

    Validates the whole pipeline — sharded format hashing, minishard
    index gzip decode, manifest binary parse, centroid math, nm
    conversion — by checking that a known KC's centroid lands inside
    hemibrain's published volume bounds.
    """
    meshes_mod.clear_cache()
    sp_mod.clear_cache()
    layers.add_segmentation_layer(
        "hemibrain",
        "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
    )
    # KC from the earlier search test (1001453586 is "KCa'b'-ap1").
    result = navigation.center_on_segment("hemibrain", 1001453586)
    px, py, pz = result["position_nm"]
    # Hemibrain is roughly 27000 × 32000 × 35000 nm; sanity-check that
    # the centroid falls comfortably inside that box.
    assert 0 < px < 300_000
    assert 0 < py < 400_000
    assert 0 < pz < 400_000
    assert result["mesh_type"] == "neuroglancer_multilod_draco"


@pytest.mark.network
def test_hemibrain_kenyon_cell_search():
    """End-to-end: load the public hemibrain segmentation and search for KCs.

    Hemibrain's segment_properties live at
    `gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation/segment_properties`
    and contain ~25K labeled neurons. We don't undo the autouse fixture's
    monkeypatch here — instead we use a separate, network-enabled
    fixture-less call path by re-importing the real fetcher.
    """
    import urllib.request, json

    sp_mod.clear_cache()
    # Bypass the autouse monkeypatch by calling the real fetcher explicitly.
    real_fetch = lambda url: json.loads(urllib.request.urlopen(url).read())
    sp_mod._fetch_json = real_fetch  # type: ignore[assignment]

    layers.add_segmentation_layer(
        "hemibrain",
        "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
    )
    out = properties.search_segments_by_label("hemibrain", "KC")
    assert out["total_matches"] > 0
