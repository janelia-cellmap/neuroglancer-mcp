"""Compute a layer's physical bounds + center from its source metadata.

Used so that `add_image_layer` / `add_segmentation_layer` can land the
viewer on the data instead of leaving it parked at (0, 0, 0) or
whatever the prior session set. Mirrors what Neuroglancer's JS UI does
automatically when you drop a layer onto the viewer.

Currently supports:
- precomputed: `<base>/info` → scales[0] size/voxel_offset/resolution
- OME-NGFF v0.4 zarr: `<base>/.zattrs` multiscales[0] + the dataset's
  `.zarray` shape, combining scale + translation transforms.

Other formats (n5, zarr v3, raw HTTP shards) return a "not supported"
error; the layer still loads fine, just without auto-centering.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from typing import Any, Optional

import neuroglancer

from neuroglancer_mcp.segment_properties import _normalize_url
from neuroglancer_mcp.viewer import nm_to_voxels


def _fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Unit handling for OME-NGFF axes
# ---------------------------------------------------------------------------


_UNIT_TO_NM = {
    "": 1.0,  # OME-NGFF omits unit for unitless or assumes raw values
    "nanometer": 1.0,
    "micrometer": 1e3,
    "millimeter": 1e6,
    "meter": 1e9,
}


def _to_nm(value: float, unit: Optional[str]) -> float:
    factor = _UNIT_TO_NM.get((unit or "").lower())
    if factor is None:
        raise ValueError(f"Unsupported OME-NGFF unit: {unit!r}")
    return value * factor


# ---------------------------------------------------------------------------
# precomputed
# ---------------------------------------------------------------------------


def _center_precomputed(url: str) -> dict[str, Any]:
    """Volume center for a Neuroglancer Precomputed source.

    Uses scales[0] (the finest). Result is in nm and (x, y, z) order.
    """
    base = _normalize_url(url).rstrip("/")
    info = _fetch_json(base + "/info")
    scales = info.get("scales")
    if not scales:
        raise ValueError(f"Precomputed info at {base} has no scales")
    scale = scales[0]
    res = scale["resolution"]
    size = scale["size"]
    offset = scale.get("voxel_offset", [0, 0, 0])
    if len(res) != 3 or len(size) != 3 or len(offset) != 3:
        raise ValueError(
            f"Precomputed scale[0] dimensions must be 3D, got "
            f"resolution={res}, size={size}, offset={offset}"
        )
    position_nm = [
        res[i] * (offset[i] + size[i] / 2.0) for i in range(3)
    ]
    return {
        "format": "precomputed",
        "position_nm": position_nm,
        "voxel_size_nm": [float(r) for r in res],
        "shape_voxels": [int(s) for s in size],
    }


# ---------------------------------------------------------------------------
# OME-NGFF zarr v0.4
# ---------------------------------------------------------------------------


def _center_zarr(url: str) -> dict[str, Any]:
    """Volume center for an OME-NGFF v0.4 zarr source.

    Reads the multiscales metadata (`.zattrs`) and the finest dataset's
    `.zarray` shape. Combines the dataset's `scale` and `translation`
    coordinateTransformations to get a physical center, converted to
    nm using each axis's `unit`. Returns coords in (x, y, z) order.
    """
    base = _normalize_url(url).rstrip("/")
    zattrs = _fetch_json(base + "/.zattrs")
    multiscales = zattrs.get("multiscales")
    if not multiscales:
        raise ValueError(f"No 'multiscales' in {base}/.zattrs")
    ms = multiscales[0]
    axes = ms.get("axes")
    if not axes:
        raise ValueError(f"No 'axes' in multiscales[0] at {base}")
    if not ms.get("datasets"):
        raise ValueError(f"No 'datasets' in multiscales[0] at {base}")

    ds = ms["datasets"][0]
    path = ds["path"]
    transforms = ds.get("coordinateTransformations", [])
    scale = [1.0] * len(axes)
    translation = [0.0] * len(axes)
    for t in transforms:
        if t["type"] == "scale":
            scale = list(t["scale"])
        elif t["type"] == "translation":
            translation = list(t["translation"])

    zarray = _fetch_json(base + "/" + path + "/.zarray")
    shape = zarray.get("shape")
    if shape is None or len(shape) != len(axes):
        raise ValueError(
            f"shape/axes mismatch at {base}: axes={len(axes)}, shape={shape}"
        )

    # Compute per-axis nm center and voxel size, indexed by axis name.
    center_by_axis: dict[str, float] = {}
    scale_by_axis: dict[str, float] = {}
    shape_by_axis: dict[str, int] = {}
    for i, axis in enumerate(axes):
        name = axis["name"]
        unit = axis.get("unit")
        s_nm = _to_nm(float(scale[i]), unit)
        t_nm = _to_nm(float(translation[i]), unit)
        center_by_axis[name] = t_nm + s_nm * shape[i] / 2.0
        scale_by_axis[name] = s_nm
        shape_by_axis[name] = int(shape[i])

    # Return in Neuroglancer's (x, y, z) order. Missing axes get zero.
    return {
        "format": "zarr",
        "position_nm": [
            center_by_axis.get("x", 0.0),
            center_by_axis.get("y", 0.0),
            center_by_axis.get("z", 0.0),
        ],
        "voxel_size_nm": [
            scale_by_axis.get("x", 1.0),
            scale_by_axis.get("y", 1.0),
            scale_by_axis.get("z", 1.0),
        ],
        "shape_voxels": [
            shape_by_axis.get("x", 0),
            shape_by_axis.get("y", 0),
            shape_by_axis.get("z", 0),
        ],
    }


# ---------------------------------------------------------------------------
# dtype inspection + layer-type inference
# ---------------------------------------------------------------------------


_NUMPY_DTYPE_TO_NAME = {
    "u1": "uint8", "u2": "uint16", "u4": "uint32", "u8": "uint64",
    "i1": "int8", "i2": "int16", "i4": "int32", "i8": "int64",
    "f2": "float16", "f4": "float32", "f8": "float64",
    "b1": "bool",
}


def _normalize_dtype(dtype: str) -> str:
    """Canonicalize numpy dtype strings ('|u1', '<u8') to friendly names.

    Precomputed publishes already-friendly names ("uint8", "float32");
    OME-NGFF zarr publishes numpy-style strings. We accept either.
    """
    if not dtype:
        return dtype
    s = dtype.strip()
    # Strip endianness prefix if present.
    if s and s[0] in "<>|=":
        s = s[1:]
    return _NUMPY_DTYPE_TO_NAME.get(s, dtype)


def fetch_layer_dtype(source_url: str) -> Optional[str]:
    """Return the layer's canonical dtype name, or None if not derivable.

    Reads from the source's metadata; no actual data fetch. Used by
    `infer_layer_type` to decide whether a layer is image or
    segmentation when the caller hasn't said.
    """
    url = source_url.strip()
    try:
        if url.startswith("precomputed://"):
            base = _normalize_url(url).rstrip("/")
            info = _fetch_json(base + "/info")
            return _normalize_dtype(info.get("data_type", ""))
        if url.startswith("zarr://") or url.startswith("zarr2://"):
            base = _normalize_url(url).rstrip("/")
            zattrs = _fetch_json(base + "/.zattrs")
            ms = zattrs.get("multiscales", [])
            if not ms or not ms[0].get("datasets"):
                return None
            path = ms[0]["datasets"][0]["path"]
            zarray = _fetch_json(base + "/" + path + "/.zarray")
            return _normalize_dtype(zarray.get("dtype", ""))
    except Exception as e:  # noqa: BLE001
        print(
            f"[neuroglancer-mcp] dtype probe failed for {source_url}: {e}",
            file=sys.stderr,
        )
    return None


def infer_layer_type(source_url: str) -> dict[str, Any]:
    """Guess 'image' vs 'segmentation' for a data source based on dtype.

    Rules (connectomics conventions):
    - `float*` → image. Floats are virtually never segment IDs.
    - `uint64` → segmentation. Connectomics canonically stores segment
      IDs as uint64 (FlyWire, hemibrain, MICrONS, CellMap).
    - Anything else (uint8/16/32, int*) → image, **low confidence**.
      uint8/uint16 are usually EM raw or 8/16-bit predictions; uint32
      is rare. Override explicitly via the `type` argument when the
      data is actually a small-bitwidth label volume.

    Returns:
        {"type": "image" | "segmentation",
         "dtype": <canonical name or None>,
         "confidence": "high" | "low",
         "reason": <human-readable>}.
    """
    dtype = fetch_layer_dtype(source_url)
    if dtype is None:
        return {
            "type": "image",
            "dtype": None,
            "confidence": "low",
            "reason": "couldn't read dtype from metadata; defaulting to image",
        }
    if dtype.startswith("float"):
        return {
            "type": "image",
            "dtype": dtype,
            "confidence": "high",
            "reason": "float dtype — virtually always image / prediction data",
        }
    if dtype == "uint64":
        return {
            "type": "segmentation",
            "dtype": dtype,
            "confidence": "high",
            "reason": "uint64 dtype — canonical segment-ID width",
        }
    return {
        "type": "image",
        "dtype": dtype,
        "confidence": "low",
        "reason": (
            f"{dtype} could be either image or labels; "
            "defaulting to image. Pass type='segmentation' explicitly "
            "if this is a label volume."
        ),
    }


# ---------------------------------------------------------------------------
# Format dispatch
# ---------------------------------------------------------------------------


def compute_layer_center(source_url: str) -> dict[str, Any]:
    """Compute a layer's physical center from its source metadata.

    Returns:
        {"format": "precomputed" | "zarr",
         "position_nm": [x, y, z],     (physical center in nm)
         "voxel_size_nm": [rx, ry, rz], (nm per voxel of the finest scale)
         "shape_voxels": [sx, sy, sz]}  (volume extent in voxels)

    Raises:
        ValueError: if the source URL format isn't supported, or the
            metadata is malformed.
    """
    url = source_url.strip()
    if url.startswith("precomputed://"):
        return _center_precomputed(url)
    if url.startswith("zarr://") or url.startswith("zarr2://"):
        return _center_zarr(url)
    raise ValueError(
        f"Unsupported source format for auto-centering: {url!r} "
        "(currently support precomputed:// and zarr:// OME-NGFF v0.4)"
    )


# ---------------------------------------------------------------------------
# Apply to a viewer
# ---------------------------------------------------------------------------


def apply_center_to_viewer(
    viewer: neuroglancer.Viewer, source_url: str
) -> Optional[dict[str, Any]]:
    """Compute the layer's center and update the viewer's position.

    If the viewer has no dimensions yet (no browser has connected to
    populate them), this also sets dimensions from the layer's voxel
    size so the position has a defined coordinate space immediately.
    Subsequent layer-adds don't overwrite dimensions — first layer
    wins.

    Returns the centering metadata dict (also includes
    `position_voxels` after the conversion), or None if the source
    format isn't supported / metadata couldn't be fetched. Layer-add
    callers should swallow None and continue: the layer is still added,
    just not auto-centered.
    """
    try:
        info = compute_layer_center(source_url)
    except Exception as e:  # noqa: BLE001 - we deliberately don't propagate
        print(
            f"[neuroglancer-mcp] auto-center skipped for {source_url}: {e}",
            file=sys.stderr,
        )
        return None

    with viewer.txn() as s:
        if not list(s.dimensions.names):
            s.dimensions = neuroglancer.CoordinateSpace(
                names=["x", "y", "z"],
                units=["nm", "nm", "nm"],
                scales=info["voxel_size_nm"],
            )
        x_nm, y_nm, z_nm = info["position_nm"]
        position = nm_to_voxels(x_nm, y_nm, z_nm, s.dimensions)
        s.position = position
        info["position_voxels"] = [float(p) for p in s.position]
    return info
