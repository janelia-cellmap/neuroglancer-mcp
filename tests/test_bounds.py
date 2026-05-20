"""Tests for volume-center computation and auto-centering on layer add.

Hand-built precomputed `info` and OME-NGFF `.zattrs`/`.zarray` fixtures
exercise the format dispatch and unit conversion. One real-network
test against an OpenOrganelle HeLa-2 zarr validates end-to-end.
"""

from __future__ import annotations

import pytest

from neuroglancer_mcp import bounds as bounds_mod
from neuroglancer_mcp import viewer as viewer_mod
from neuroglancer_mcp.tools import layers, navigation, state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_PRECOMPUTED_INFO = {
    "@type": "neuroglancer_multiscale_volume",
    "type": "image",
    "data_type": "uint8",
    "num_channels": 1,
    "scales": [
        {
            "key": "8_8_8",
            "resolution": [8, 8, 8],
            "size": [1000, 2000, 4000],
            "voxel_offset": [0, 0, 0],
            "chunk_sizes": [[64, 64, 64]],
            "encoding": "raw",
        }
    ],
}


def _zarr_zattrs(translation=(0.0, 0.0, 0.0)) -> dict:
    return {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [5.24, 4.0, 4.0]},
                            {"type": "translation", "translation": list(translation)},
                        ],
                    }
                ],
            }
        ]
    }


_ZARR_ZARRAY = {"shape": [6368, 1600, 12000], "dtype": "|u1"}


@pytest.fixture
def stub_fetcher(monkeypatch):
    """Return a callable that registers a {url_suffix → response} map."""
    routes: dict[str, dict] = {}

    def fake_fetch(url: str):
        for suffix, payload in routes.items():
            if url.endswith(suffix):
                return payload
        raise AssertionError(f"unexpected fetch: {url}")

    monkeypatch.setattr(bounds_mod, "_fetch_json", fake_fetch)
    return routes


# ---------------------------------------------------------------------------
# Format-specific centers
# ---------------------------------------------------------------------------


def test_center_precomputed(stub_fetcher):
    stub_fetcher["/segmentation/info"] = _PRECOMPUTED_INFO
    result = bounds_mod.compute_layer_center(
        "precomputed://gs://example/segmentation"
    )
    # Size 1000 × 2000 × 4000 at 8nm → center (4000, 8000, 16000) nm.
    assert result["format"] == "precomputed"
    assert result["position_nm"] == [4000.0, 8000.0, 16000.0]
    assert result["voxel_size_nm"] == [8.0, 8.0, 8.0]
    assert result["shape_voxels"] == [1000, 2000, 4000]


def test_center_precomputed_honors_voxel_offset(stub_fetcher):
    info = {
        **_PRECOMPUTED_INFO,
        "scales": [{**_PRECOMPUTED_INFO["scales"][0], "voxel_offset": [100, 200, 300]}],
    }
    stub_fetcher["/segmentation/info"] = info
    result = bounds_mod.compute_layer_center(
        "precomputed://gs://example/segmentation"
    )
    # offset shifts center: (100 + 500) * 8 = 4800, etc.
    assert result["position_nm"] == [4800.0, 9600.0, 18400.0]


def test_center_zarr_basic(stub_fetcher):
    stub_fetcher["/data.zarr/.zattrs"] = _zarr_zattrs()
    stub_fetcher["/data.zarr/s0/.zarray"] = _ZARR_ZARRAY
    result = bounds_mod.compute_layer_center("zarr://s3://example/data.zarr")
    # HeLa-2-like volume: shape (z=6368, y=1600, x=12000) at (5.24, 4, 4) nm.
    # x center = 4 * 12000 / 2 = 24000
    # y center = 4 * 1600 / 2 = 3200
    # z center = 5.24 * 6368 / 2 = 16683.84
    assert result["format"] == "zarr"
    assert result["position_nm"][0] == pytest.approx(24000.0)
    assert result["position_nm"][1] == pytest.approx(3200.0)
    assert result["position_nm"][2] == pytest.approx(16684.16)
    assert result["voxel_size_nm"] == [4.0, 4.0, 5.24]
    assert result["shape_voxels"] == [12000, 1600, 6368]


def test_center_zarr_applies_translation(stub_fetcher):
    stub_fetcher["/data.zarr/.zattrs"] = _zarr_zattrs(translation=(10.0, 20.0, 30.0))
    stub_fetcher["/data.zarr/s0/.zarray"] = _ZARR_ZARRAY
    result = bounds_mod.compute_layer_center("zarr://s3://example/data.zarr")
    # Translation per-axis adds to the center (axes are z, y, x).
    assert result["position_nm"][0] == pytest.approx(24000.0 + 30.0)
    assert result["position_nm"][1] == pytest.approx(3200.0 + 20.0)
    assert result["position_nm"][2] == pytest.approx(16684.16 + 10.0)


def test_center_zarr_meter_unit_converts_to_nm(stub_fetcher):
    zattrs = _zarr_zattrs()
    for ax in zattrs["multiscales"][0]["axes"]:
        ax["unit"] = "meter"
    # Scales then in meters: 5.24e-9, 4e-9, 4e-9. Same physical result.
    zattrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0][
        "scale"
    ] = [5.24e-9, 4e-9, 4e-9]
    stub_fetcher["/data.zarr/.zattrs"] = zattrs
    stub_fetcher["/data.zarr/s0/.zarray"] = _ZARR_ZARRAY
    result = bounds_mod.compute_layer_center("zarr://s3://example/data.zarr")
    assert result["position_nm"][0] == pytest.approx(24000.0)


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported source format"):
        bounds_mod.compute_layer_center("n5://example.com/data.n5")


# ---------------------------------------------------------------------------
# apply_center_to_viewer
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_viewer():
    viewer_mod.reset_viewer()
    yield
    viewer_mod.reset_viewer()


def test_apply_center_sets_dimensions_when_empty(monkeypatch):
    def fake_compute(url: str) -> dict:
        return {
            "format": "precomputed",
            "position_nm": [400.0, 800.0, 1600.0],
            "voxel_size_nm": [8.0, 8.0, 8.0],
            "shape_voxels": [100, 200, 400],
        }

    monkeypatch.setattr(bounds_mod, "compute_layer_center", fake_compute)
    info = bounds_mod.apply_center_to_viewer(
        viewer_mod.get_viewer(), "precomputed://gs://x/seg"
    )
    assert info is not None
    # 8 nm dimensions → position 400/8=50, 100, 200
    assert info["position_voxels"] == [50.0, 100.0, 200.0]
    # Dimensions should now be populated with the layer's voxel size.
    snap = state.get_state()
    dims = snap["dimensions"]
    assert dims["x"] == [8e-9, "m"]
    assert dims["y"] == [8e-9, "m"]
    assert dims["z"] == [8e-9, "m"]


def test_apply_center_failure_returns_none(monkeypatch):
    def bad_compute(url: str):
        raise ValueError("nope")

    monkeypatch.setattr(bounds_mod, "compute_layer_center", bad_compute)
    info = bounds_mod.apply_center_to_viewer(
        viewer_mod.get_viewer(), "n5://example/foo"
    )
    assert info is None


# ---------------------------------------------------------------------------
# add_*_layer auto-centers by default
# ---------------------------------------------------------------------------


def test_add_image_layer_autocenters(monkeypatch):
    def fake_compute(url: str) -> dict:
        return {
            "format": "precomputed",
            "position_nm": [800.0, 1600.0, 3200.0],
            "voxel_size_nm": [8.0, 8.0, 8.0],
            "shape_voxels": [200, 400, 800],
        }

    monkeypatch.setattr(bounds_mod, "compute_layer_center", fake_compute)
    result = layers.add_image_layer("em", "precomputed://gs://example/em")
    assert result["centered_on"]["position_nm"] == [800.0, 1600.0, 3200.0]
    snap = state.get_state()
    assert snap["position"] == [100.0, 200.0, 400.0]


def test_add_image_layer_respects_center_false(monkeypatch):
    called = {"n": 0}

    def fake_compute(url: str):
        called["n"] += 1
        raise AssertionError("should not be called")

    monkeypatch.setattr(bounds_mod, "compute_layer_center", fake_compute)
    result = layers.add_image_layer(
        "em", "precomputed://gs://example/em", center=False
    )
    assert "centered_on" not in result
    assert called["n"] == 0


def test_add_image_layer_swallows_center_failure(monkeypatch):
    def boom(url: str):
        raise ValueError("metadata 404")

    monkeypatch.setattr(bounds_mod, "compute_layer_center", boom)
    result = layers.add_image_layer("em", "precomputed://gs://example/em")
    # Layer still added; centered_on absent.
    assert result["name"] == "em"
    assert "centered_on" not in result
    assert [layer["name"] for layer in state.list_layers()] == ["em"]


# ---------------------------------------------------------------------------
# center_on_layer (standalone)
# ---------------------------------------------------------------------------


def test_center_on_layer_uses_existing_layer_source(monkeypatch):
    layers.add_image_layer(
        "em", "precomputed://gs://example/em", center=False
    )

    def fake_compute(url: str) -> dict:
        assert url == "precomputed://gs://example/em"
        return {
            "format": "precomputed",
            "position_nm": [80.0, 160.0, 320.0],
            "voxel_size_nm": [4.0, 4.0, 4.0],
            "shape_voxels": [40, 80, 160],
        }

    monkeypatch.setattr(bounds_mod, "compute_layer_center", fake_compute)
    result = navigation.center_on_layer("em")
    # 80 nm at 4 nm/vox = 20 vox; etc.
    assert result["position_voxels"] == [20.0, 40.0, 80.0]


def test_center_on_layer_missing_layer_raises():
    with pytest.raises(ValueError, match="not found"):
        navigation.center_on_layer("nope")


def test_center_on_layer_reports_unsupported(monkeypatch):
    layers.add_image_layer("em", "n5://example/em.n5", center=False)
    result = navigation.center_on_layer("em")
    assert result["centered"] is False
    assert "unsupported" in result["reason"].lower() or "failed" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Network integration: real OpenOrganelle HeLa-2
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# dtype parsing + layer-type inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("|u1", "uint8"),
        ("<u8", "uint64"),
        (">f4", "float32"),
        ("u2", "uint16"),
        ("|i1", "int8"),
        ("<f8", "float64"),
        ("uint8", "uint8"),     # already canonical, pass through
        ("float32", "float32"),  # already canonical
    ],
)
def test_normalize_dtype(raw, expected):
    assert bounds_mod._normalize_dtype(raw) == expected


def test_fetch_layer_dtype_precomputed(stub_fetcher):
    stub_fetcher["/em/info"] = {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint8",
        "scales": [{"resolution": [8, 8, 8], "size": [10, 10, 10]}],
    }
    assert (
        bounds_mod.fetch_layer_dtype("precomputed://gs://example/em")
        == "uint8"
    )


def test_fetch_layer_dtype_zarr(stub_fetcher):
    stub_fetcher["/data.zarr/.zattrs"] = _zarr_zattrs()
    stub_fetcher["/data.zarr/s0/.zarray"] = {"shape": [1, 1, 1], "dtype": "<f4"}
    assert (
        bounds_mod.fetch_layer_dtype("zarr://s3://example/data.zarr")
        == "float32"
    )


def test_fetch_layer_dtype_unsupported_format_returns_none():
    assert bounds_mod.fetch_layer_dtype("n5://example/data") is None


def test_infer_layer_type_float_is_image(monkeypatch):
    monkeypatch.setattr(bounds_mod, "fetch_layer_dtype", lambda u: "float32")
    out = bounds_mod.infer_layer_type("precomputed://gs://x/em")
    assert out["type"] == "image"
    assert out["confidence"] == "high"
    assert out["dtype"] == "float32"


def test_infer_layer_type_uint64_is_segmentation(monkeypatch):
    monkeypatch.setattr(bounds_mod, "fetch_layer_dtype", lambda u: "uint64")
    out = bounds_mod.infer_layer_type("precomputed://gs://x/seg")
    assert out["type"] == "segmentation"
    assert out["confidence"] == "high"


def test_infer_layer_type_uint8_defaults_to_image_low_confidence(monkeypatch):
    monkeypatch.setattr(bounds_mod, "fetch_layer_dtype", lambda u: "uint8")
    out = bounds_mod.infer_layer_type("precomputed://gs://x/em")
    assert out["type"] == "image"
    assert out["confidence"] == "low"


def test_infer_layer_type_unknown_dtype_low_confidence(monkeypatch):
    monkeypatch.setattr(bounds_mod, "fetch_layer_dtype", lambda u: None)
    out = bounds_mod.infer_layer_type("n5://example/foo")
    assert out["type"] == "image"
    assert out["confidence"] == "low"
    assert out["dtype"] is None


# ---------------------------------------------------------------------------
# add_layer auto-detect (single)
# ---------------------------------------------------------------------------


def test_add_layer_infers_image_from_float(monkeypatch):
    monkeypatch.setattr(
        "neuroglancer_mcp.tools.layers.infer_layer_type",
        lambda u: {
            "type": "image", "dtype": "float32", "confidence": "high",
            "reason": "float dtype",
        },
    )
    monkeypatch.setattr(
        bounds_mod,
        "compute_layer_center",
        lambda u: (_ for _ in ()).throw(ValueError("stub")),
    )
    from neuroglancer_mcp.tools import layers as layers_mod
    result = layers_mod.add_layer("predictions", "zarr://example/preds.zarr")
    assert result["type"] == "image"
    assert result["inferred"] is True
    assert result["inference"]["dtype"] == "float32"


def test_add_layer_infers_segmentation_from_uint64(monkeypatch):
    monkeypatch.setattr(
        "neuroglancer_mcp.tools.layers.infer_layer_type",
        lambda u: {
            "type": "segmentation", "dtype": "uint64", "confidence": "high",
            "reason": "uint64 segment IDs",
        },
    )
    monkeypatch.setattr(
        bounds_mod,
        "compute_layer_center",
        lambda u: (_ for _ in ()).throw(ValueError("stub")),
    )
    from neuroglancer_mcp.tools import layers as layers_mod
    result = layers_mod.add_layer("labels", "precomputed://example/labels")
    assert result["type"] == "segmentation"
    assert result["inferred"] is True
    listed = [layer["name"] for layer in state.list_layers()]
    assert "labels" in listed


def test_add_layer_explicit_type_skips_inference(monkeypatch):
    called = {"n": 0}

    def boom(u):
        called["n"] += 1
        raise AssertionError("inference should be skipped")

    monkeypatch.setattr(bounds_mod, "infer_layer_type", boom)
    monkeypatch.setattr(
        bounds_mod,
        "compute_layer_center",
        lambda u: (_ for _ in ()).throw(ValueError("stub")),
    )
    from neuroglancer_mcp.tools import layers as layers_mod
    result = layers_mod.add_layer("x", "zarr://example/x.zarr", type="image")
    assert called["n"] == 0
    assert result["type"] == "image"
    assert result["inferred"] is False


@pytest.mark.network
def test_hela2_volume_center_matches_published_bounds():
    """Live test: OpenOrganelle jrc_hela-2 FIB-SEM (zarr) auto-center.

    Hits the public S3 bucket. Volume is ~33 × 6.4 × 48 μm at 5.24/4/4 nm
    voxels; center sits at roughly (24000, 3200, 16684) nm.
    """
    result = bounds_mod.compute_layer_center(
        "zarr://https://janelia-cosem-datasets.s3.amazonaws.com/"
        "jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8"
    )
    assert result["format"] == "zarr"
    assert result["position_nm"][0] == pytest.approx(24000.0)
    assert result["position_nm"][1] == pytest.approx(3200.0)
    assert result["position_nm"][2] == pytest.approx(16683.84, rel=1e-3)
