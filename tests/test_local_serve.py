"""Tests for local filesystem serving + auto-detect in layer-add tools.

Real http.server in a temp dir — we fetch the file we serve to confirm
the loop works end-to-end. Format detection runs against tmp_path
contents.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from neuroglancer_mcp import local_serve
from neuroglancer_mcp import viewer as viewer_mod
from neuroglancer_mcp.tools import layers, local_data, state


@pytest.fixture(autouse=True)
def fresh_viewer():
    viewer_mod.reset_viewer()
    yield
    viewer_mod.reset_viewer()


@pytest.fixture(autouse=True)
def stop_all_servers():
    yield
    # Tear down any servers started during the test.
    for entry in list(local_serve.list_servers()):
        local_serve.stop_server(entry["path"])


# ---------------------------------------------------------------------------
# ensure_server / list / stop
# ---------------------------------------------------------------------------


def test_ensure_server_serves_real_file(tmp_path):
    payload = b"hello neuroglancer"
    (tmp_path / "data.bin").write_bytes(payload)

    info = local_serve.ensure_server(str(tmp_path), bind="127.0.0.1")
    assert info["started"] is True
    assert info["port"] > 0

    # Fetch through the spawned server.
    fetched = urllib.request.urlopen(f"{info['url']}/data.bin").read()
    assert fetched == payload


def test_ensure_server_is_idempotent(tmp_path):
    first = local_serve.ensure_server(str(tmp_path), bind="127.0.0.1")
    second = local_serve.ensure_server(str(tmp_path), bind="127.0.0.1")
    assert second["started"] is False
    assert second["port"] == first["port"]


def test_ensure_server_rejects_non_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hi")
    with pytest.raises(ValueError, match="Not a directory"):
        local_serve.ensure_server(str(f), bind="127.0.0.1")


def test_list_and_stop(tmp_path):
    local_serve.ensure_server(str(tmp_path), bind="127.0.0.1")
    listed = local_serve.list_servers()
    assert any(s["path"] == str(tmp_path.resolve()) for s in listed)

    stopped = local_serve.stop_server(str(tmp_path))
    assert stopped["stopped"] is True
    assert all(s["path"] != str(tmp_path.resolve()) for s in local_serve.list_servers())


def test_stop_server_idempotent(tmp_path):
    result = local_serve.stop_server(str(tmp_path))
    assert result["stopped"] is False


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_detect_format_zarr_extension(tmp_path):
    (tmp_path / "data.zarr").mkdir()
    assert local_serve.detect_format_prefix(str(tmp_path / "data.zarr")) == "zarr://"


def test_detect_format_n5_extension(tmp_path):
    (tmp_path / "data.n5").mkdir()
    assert local_serve.detect_format_prefix(str(tmp_path / "data.n5")) == "n5://"


def test_detect_format_precomputed_by_info_file(tmp_path):
    pc = tmp_path / "precomp_layer"
    pc.mkdir()
    (pc / "info").write_text(json.dumps({"@type": "neuroglancer_multiscale_volume"}))
    assert local_serve.detect_format_prefix(str(pc)) == "precomputed://"


def test_detect_format_unknown_raises(tmp_path):
    weird = tmp_path / "weird"
    weird.mkdir()
    with pytest.raises(ValueError, match="Could not detect"):
        local_serve.detect_format_prefix(str(weird))


# ---------------------------------------------------------------------------
# resolve_local_source (end-to-end translate)
# ---------------------------------------------------------------------------


def test_resolve_local_source_builds_url(tmp_path):
    dataset = tmp_path / "foo.zarr"
    dataset.mkdir()
    url = local_serve.resolve_local_source(str(dataset), bind="127.0.0.1")
    assert url.startswith("zarr://http://")
    assert url.endswith("/foo.zarr")


def test_resolve_local_source_passthrough_for_urls():
    assert local_serve.looks_like_local_path("zarr://example.com/x") is False
    assert local_serve.looks_like_local_path("/groups/cellmap/x.zarr") is True
    assert local_serve.looks_like_local_path("~/data.zarr") is True


# ---------------------------------------------------------------------------
# Layer-add tools auto-resolve a local path
# ---------------------------------------------------------------------------


def test_add_image_layer_with_local_path(tmp_path):
    dataset = tmp_path / "em.zarr"
    dataset.mkdir()
    # Stub center computation so we don't try to fetch metadata.
    import neuroglancer_mcp.bounds as bounds_mod
    bounds_mod_compute = bounds_mod.compute_layer_center
    bounds_mod.compute_layer_center = lambda u: (_ for _ in ()).throw(ValueError("stub"))
    try:
        result = layers.add_image_layer("em", str(dataset))
    finally:
        bounds_mod.compute_layer_center = bounds_mod_compute

    assert result["source"].startswith("zarr://http://")
    assert result["source"].endswith("/em.zarr")
    assert result["local"]["original_path"] == str(dataset)


def test_add_segmentation_layer_with_local_n5(tmp_path):
    dataset = tmp_path / "labels.n5"
    dataset.mkdir()
    import neuroglancer_mcp.bounds as bounds_mod
    saved = bounds_mod.compute_layer_center
    bounds_mod.compute_layer_center = lambda u: (_ for _ in ()).throw(ValueError("stub"))
    try:
        result = layers.add_segmentation_layer("seg", str(dataset))
    finally:
        bounds_mod.compute_layer_center = saved
    assert result["source"].startswith("n5://http://")
    assert result["source"].endswith("/labels.n5")


def test_add_layer_local_path_passes_through_inference(tmp_path, monkeypatch):
    dataset = tmp_path / "preds.zarr"
    dataset.mkdir()
    monkeypatch.setattr(
        "neuroglancer_mcp.tools.layers.infer_layer_type",
        lambda u: {
            "type": "image", "dtype": "float32", "confidence": "high",
            "reason": "stub",
        },
    )
    import neuroglancer_mcp.bounds as bounds_mod
    monkeypatch.setattr(
        bounds_mod, "compute_layer_center",
        lambda u: (_ for _ in ()).throw(ValueError("stub")),
    )
    result = layers.add_layer("preds", str(dataset))
    assert result["type"] == "image"
    assert result["inferred"] is True
    assert result["source"].startswith("zarr://http://")
    assert result["local"]["original_path"] == str(dataset)


def test_add_layers_bulk_with_local_paths(tmp_path):
    (tmp_path / "em.zarr").mkdir()
    (tmp_path / "labels.n5").mkdir()
    import neuroglancer_mcp.bounds as bounds_mod
    saved = bounds_mod.compute_layer_center
    bounds_mod.compute_layer_center = lambda u: (_ for _ in ()).throw(ValueError("stub"))
    try:
        result = layers.add_layers(
            [
                {"name": "em", "source": str(tmp_path / "em.zarr"), "type": "image"},
                {"name": "lab", "source": str(tmp_path / "labels.n5"), "type": "segmentation"},
            ],
            center_on=None,
        )
    finally:
        bounds_mod.compute_layer_center = saved
    assert all("local" in entry for entry in result["added"])
    assert any(entry["source"].startswith("zarr://http://") for entry in result["added"])
    assert any(entry["source"].startswith("n5://http://") for entry in result["added"])
    # The two datasets share one HTTP server because they have the same parent.
    assert len(local_serve.list_servers()) == 1


def test_add_image_layer_http_url_unchanged(tmp_path):
    """URLs already-resolved (http://, gs://, etc.) shouldn't trigger serving."""
    import neuroglancer_mcp.bounds as bounds_mod
    saved = bounds_mod.compute_layer_center
    bounds_mod.compute_layer_center = lambda u: (_ for _ in ()).throw(ValueError("stub"))
    try:
        result = layers.add_image_layer(
            "em", "precomputed://gs://example/em"
        )
    finally:
        bounds_mod.compute_layer_center = saved
    assert result["source"] == "precomputed://gs://example/em"
    assert "local" not in result
    assert local_serve.list_servers() == []


# ---------------------------------------------------------------------------
# MCP-tool wrappers
# ---------------------------------------------------------------------------


def test_serve_local_directory_tool(tmp_path):
    result = local_data.serve_local_directory(str(tmp_path), bind="127.0.0.1")
    assert result["started"] is True
    assert result["bind"] == "127.0.0.1"
    listed = local_data.list_served_directories()
    assert len(listed["servers"]) == 1


def test_stop_serving_directory_tool(tmp_path):
    local_data.serve_local_directory(str(tmp_path), bind="127.0.0.1")
    result = local_data.stop_serving_directory(str(tmp_path))
    assert result["stopped"] is True
