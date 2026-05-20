"""Unit tests for the mesh manifest decoder.

Pure functions — no viewer, no network. Hand-built binary blobs
exercise the multilod manifest decoder, minishard index decoder,
sharding hash math, and centroid arithmetic.

End-to-end validation of murmurhash3 happens through the hemibrain
network test in test_properties_tools.py (hemibrain's mesh is keyed
by murmurhash3; a successful shard lookup means our hash matches the
reference impl).
"""

from __future__ import annotations

import gzip
import struct

import pytest

from neuroglancer_mcp import meshes
from neuroglancer_mcp.meshes import (
    _centroid_from_multilod_manifest,
    _decode_minishard_index,
    _decode_multilod_manifest,
    _murmurhash3_x86_128_low64,
    _shard_and_minishard,
    _shard_filename,
    estimate_segment_centroid,
)


# ---------------------------------------------------------------------------
# MurmurHash3 x86 128 (low 64 bits)
# ---------------------------------------------------------------------------


def test_murmurhash3_empty_is_zero():
    assert _murmurhash3_x86_128_low64(b"") == 0


def test_murmurhash3_is_deterministic():
    payload = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert _murmurhash3_x86_128_low64(payload) == _murmurhash3_x86_128_low64(payload)


def test_murmurhash3_different_inputs_differ():
    # Tiny spot check — not a uniqueness proof, just a guard against
    # the algorithm collapsing to a constant.
    values = {
        _murmurhash3_x86_128_low64(struct.pack("<Q", i)) for i in range(10)
    }
    assert len(values) == 10


def test_murmurhash3_in_64bit_range():
    h = _murmurhash3_x86_128_low64(struct.pack("<Q", 1234567890))
    assert 0 <= h < (1 << 64)


# ---------------------------------------------------------------------------
# Sharding math
# ---------------------------------------------------------------------------


def test_shard_filename_pads_to_hex_width():
    assert _shard_filename(5, shard_bits=10) == "005.shard"
    assert _shard_filename(0xABCD, shard_bits=16) == "abcd.shard"
    assert _shard_filename(0, shard_bits=1) == "0.shard"


def test_shard_and_minishard_identity_hash():
    sharding = {
        "hash": "identity",
        "preshift_bits": 0,
        "minishard_bits": 4,
        "shard_bits": 4,
    }
    # segment_id = 0xAB = 1010_1011 → minishard = 1011 = 0xB, shard = 1010 = 0xA
    shard, minishard = _shard_and_minishard(0xAB, sharding)
    assert (shard, minishard) == (0xA, 0xB)


def test_shard_and_minishard_with_preshift():
    sharding = {
        "hash": "identity",
        "preshift_bits": 4,
        "minishard_bits": 4,
        "shard_bits": 4,
    }
    # After >> 4, segment 0xAB0 → 0xAB → shard 0xA, minishard 0xB
    shard, minishard = _shard_and_minishard(0xAB0, sharding)
    assert (shard, minishard) == (0xA, 0xB)


# ---------------------------------------------------------------------------
# Minishard index decoder
# ---------------------------------------------------------------------------


def _build_minishard_index(entries: list[tuple[int, int, int]]) -> bytes:
    """Re-encode (chunk_id, offset, size) entries the way the spec stores them.

    chunk_ids are delta-encoded from 0. Offsets are deltas from the
    end of the previous chunk's data (so each entry advances the cursor
    by its delta + the previous size).
    """
    n = len(entries)
    cid_deltas, off_deltas, sizes = [], [], []
    prev_cid = 0
    cursor = 0
    for cid, off, size in entries:
        cid_deltas.append(cid - prev_cid)
        off_deltas.append(off - cursor)
        sizes.append(size)
        cursor = off + size
        prev_cid = cid
    return struct.pack(f"<{3*n}Q", *cid_deltas, *off_deltas, *sizes)


def test_decode_minishard_index_roundtrip():
    entries = [(100, 0, 50), (250, 100, 40), (300, 200, 20)]
    raw = _build_minishard_index(entries)
    decoded = _decode_minishard_index(raw, "raw")
    assert decoded == entries


def test_decode_minishard_index_gzip():
    entries = [(7, 0, 10), (8, 50, 20)]
    raw = _build_minishard_index(entries)
    decoded = _decode_minishard_index(gzip.compress(raw), "gzip")
    assert decoded == entries


def test_decode_minishard_index_unknown_encoding_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        _decode_minishard_index(b"\x00" * 24, "lz4")


# ---------------------------------------------------------------------------
# Multilod manifest decoder
# ---------------------------------------------------------------------------


def _build_multilod_manifest(
    chunk_shape=(100.0, 100.0, 100.0),
    grid_origin=(0.0, 0.0, 0.0),
    lod_scales=(1.0, 2.0),
    vertex_offsets=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    fragments_per_lod=((0, 0, 0),),  # one fragment at LOD 0
) -> bytes:
    """Build a binary manifest matching the precomputed mesh spec."""
    if isinstance(fragments_per_lod[0], tuple) and not isinstance(
        fragments_per_lod[0][0], tuple
    ):
        # Legacy convenience: caller passed a single tuple as the
        # one-LOD positions. Wrap it.
        fragments_per_lod = (fragments_per_lod,)
    num_lods = len(lod_scales)
    out = bytearray()
    out += struct.pack("<3f", *chunk_shape)
    out += struct.pack("<3f", *grid_origin)
    out += struct.pack("<I", num_lods)
    out += struct.pack(f"<{num_lods}f", *lod_scales)
    for vo in vertex_offsets:
        out += struct.pack("<3f", *vo)
    counts = [len(fragments_per_lod[i]) for i in range(num_lods)]
    out += struct.pack(f"<{num_lods}I", *counts)
    for positions in fragments_per_lod:
        # fragment_positions is [3, N] C-order: all xs first, then ys, then zs.
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        out += struct.pack(f"<{len(xs)}I", *xs)
        out += struct.pack(f"<{len(ys)}I", *ys)
        out += struct.pack(f"<{len(zs)}I", *zs)
        # fragment_offsets (sizes), one uint32 per fragment, unused for centroid
        out += struct.pack(f"<{len(positions)}I", *([1] * len(positions)))
    return bytes(out)


def test_decode_multilod_manifest_basic():
    blob = _build_multilod_manifest(
        chunk_shape=(64.0, 64.0, 64.0),
        grid_origin=(10.0, 20.0, 30.0),
        lod_scales=(1.0,),
        vertex_offsets=((0.0, 0.0, 0.0),),
        fragments_per_lod=(((1, 2, 3),),),
    )
    m = _decode_multilod_manifest(blob)
    assert m["chunk_shape"] == (64.0, 64.0, 64.0)
    assert m["grid_origin"] == (10.0, 20.0, 30.0)
    assert m["num_lods"] == 1
    assert m["lod_scales"] == [1.0]
    assert m["num_fragments_per_lod"] == [1]
    assert m["fragment_positions"] == [[(1, 2, 3)]]


def test_decode_multilod_manifest_multiple_lods():
    blob = _build_multilod_manifest(
        chunk_shape=(10.0, 10.0, 10.0),
        grid_origin=(0.0, 0.0, 0.0),
        lod_scales=(1.0, 2.0, 4.0),
        vertex_offsets=((0.0, 0.0, 0.0),) * 3,
        fragments_per_lod=(
            [(0, 0, 0), (1, 0, 0)],  # LOD 0: 2 fragments
            [(0, 0, 0)],              # LOD 1: 1 fragment
            [(0, 0, 0)],              # LOD 2: 1 fragment
        ),
    )
    m = _decode_multilod_manifest(blob)
    assert m["num_fragments_per_lod"] == [2, 1, 1]
    assert m["fragment_positions"][0] == [(0, 0, 0), (1, 0, 0)]


# ---------------------------------------------------------------------------
# Centroid math
# ---------------------------------------------------------------------------


def test_centroid_single_fragment_at_known_position():
    # chunk 100, origin 0, scale 1, one fragment at (1, 2, 3) →
    # center = (1.5*100, 2.5*100, 3.5*100) = (150, 250, 350).
    manifest = {
        "chunk_shape": (100.0, 100.0, 100.0),
        "grid_origin": (0.0, 0.0, 0.0),
        "num_lods": 1,
        "lod_scales": [1.0],
        "vertex_offsets": [(0.0, 0.0, 0.0)],
        "num_fragments_per_lod": [1],
        "fragment_positions": [[(1, 2, 3)]],
    }
    cx, cy, cz = _centroid_from_multilod_manifest(manifest)
    assert (cx, cy, cz) == (150.0, 250.0, 350.0)


def test_centroid_uses_finest_lod_bounding_box():
    # LOD 0 has fragments at the bbox corners; bbox spans grid x=[0,2], y=[0,2], z=[0,1].
    # step = 100 * scale 1 = 100. bbox in model coords: x [0, 200], mid=100; y [0, 200], mid=100; z [0, 100], mid=50.
    manifest = {
        "chunk_shape": (100.0, 100.0, 100.0),
        "grid_origin": (0.0, 0.0, 0.0),
        "num_lods": 2,
        "lod_scales": [1.0, 2.0],
        "vertex_offsets": [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        "num_fragments_per_lod": [4, 1],
        "fragment_positions": [
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            [(99, 99, 99)],  # coarser LOD; ignored because LOD 0 is non-empty
        ],
    }
    cx, cy, cz = _centroid_from_multilod_manifest(manifest)
    assert (cx, cy, cz) == (100.0, 100.0, 50.0)


def test_centroid_skips_empty_finest_lod():
    # LOD 0 is empty; falls back to LOD 1.
    # step = 10 * scale 2 = 20. Fragment at (0,0,0) spans [0, 20]; center = 10.
    manifest = {
        "chunk_shape": (10.0, 10.0, 10.0),
        "grid_origin": (0.0, 0.0, 0.0),
        "num_lods": 2,
        "lod_scales": [1.0, 2.0],
        "vertex_offsets": [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        "num_fragments_per_lod": [0, 1],
        "fragment_positions": [[], [(0, 0, 0)]],
    }
    cx, cy, cz = _centroid_from_multilod_manifest(manifest)
    assert (cx, cy, cz) == (10.0, 10.0, 10.0)


def test_centroid_all_empty_raises():
    manifest = {
        "chunk_shape": (10.0, 10.0, 10.0),
        "grid_origin": (0.0, 0.0, 0.0),
        "num_lods": 1,
        "lod_scales": [1.0],
        "vertex_offsets": [(0.0, 0.0, 0.0)],
        "num_fragments_per_lod": [0],
        "fragment_positions": [[]],
    }
    with pytest.raises(ValueError, match="no fragments"):
        _centroid_from_multilod_manifest(manifest)


# ---------------------------------------------------------------------------
# estimate_segment_centroid dispatch & nm conversion
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_fetchers(monkeypatch):
    """Stub the JSON + byte fetchers and reset the module cache."""
    meshes.clear_cache()
    state: dict[str, object] = {}

    def set_state(**kw):
        state.update(kw)

    def fake_json(url: str):
        if url.endswith("/segmentation/info"):
            return state["seg_info"]
        if url.endswith("/mesh/info"):
            return state["mesh_info"]
        if "/mesh/" in url and url.endswith(":0"):
            return state["legacy_manifest"]
        raise AssertionError(f"unexpected JSON url: {url}")

    monkeypatch.setattr(meshes, "_fetch_json", fake_json)
    # _cached_json wraps _fetch_json; clear cache so our stub is used.
    yield set_state
    meshes.clear_cache()


def test_estimate_centroid_errors_when_no_mesh(patched_fetchers):
    patched_fetchers(
        seg_info={"scales": [{"resolution": [8, 8, 8]}]},
    )
    with pytest.raises(ValueError, match="no mesh subdir"):
        estimate_segment_centroid(
            "precomputed://gs://example/segmentation", 42
        )


def test_estimate_centroid_legacy_path(monkeypatch, patched_fetchers):
    # No mesh transform → identity. Legacy fragment vertices live in
    # the volume's nm space directly; voxel coords = nm / resolution.
    patched_fetchers(
        seg_info={"mesh": "mesh", "scales": [{"resolution": [4, 4, 40]}]},
        mesh_info={"@type": "neuroglancer_legacy_mesh"},
        legacy_manifest={"fragments": ["frag0"]},
    )
    # Two vertices at (40, 80, 400) and (120, 160, 800), average (80, 120, 600) nm.
    n_vertices = 2
    verts = [40.0, 80.0, 400.0, 120.0, 160.0, 800.0]
    blob = struct.pack("<I", n_vertices) + struct.pack(f"<{len(verts)}f", *verts)
    monkeypatch.setattr(meshes, "_fetch_bytes", lambda url, byte_range=None: blob)

    result = estimate_segment_centroid(
        "precomputed://gs://example/segmentation", 42
    )
    assert result["mesh_type"] == "neuroglancer_legacy_mesh"
    assert result["position_nm"] == [80.0, 120.0, 600.0]
    assert result["position"] == [20.0, 30.0, 15.0]
    assert result["voxel_size_nm"] == [4.0, 4.0, 40.0]


def test_estimate_centroid_applies_mesh_transform(monkeypatch, patched_fetchers):
    # Single fragment at grid (1, 1, 1), chunk_shape 10, LOD 0 scale 1 →
    # bbox midpoint in mesh-model = (15, 15, 15). With transform = 16×
    # diagonal, nm = (240, 240, 240). voxel_size_nm = 8 → voxels = (30, 30, 30).
    patched_fetchers(
        seg_info={"mesh": "mesh", "scales": [{"resolution": [8, 8, 8]}]},
        mesh_info={
            "@type": "neuroglancer_multilod_draco",
            "sharding": {
                "hash": "identity",
                "preshift_bits": 0,
                "minishard_bits": 0,
                "shard_bits": 0,
                "minishard_index_encoding": "raw",
                "data_encoding": "raw",
            },
            "transform": [16, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16, 0],
        },
    )
    manifest_blob = _build_multilod_manifest(
        chunk_shape=(10.0, 10.0, 10.0),
        grid_origin=(0.0, 0.0, 0.0),
        lod_scales=(1.0,),
        vertex_offsets=((0.0, 0.0, 0.0),),
        fragments_per_lod=(((1, 1, 1),),),
    )
    monkeypatch.setattr(
        meshes,
        "_fetch_multilod_manifest_blob",
        lambda mesh_url, sharding, segment_id: manifest_blob,
    )

    result = estimate_segment_centroid(
        "precomputed://gs://example/segmentation", 42
    )
    assert result["position_nm"] == [240.0, 240.0, 240.0]
    assert result["position"] == [30.0, 30.0, 30.0]


def test_estimate_centroid_rejects_unknown_mesh_type(patched_fetchers):
    patched_fetchers(
        seg_info={"mesh": "mesh", "scales": [{"resolution": [8, 8, 8]}]},
        mesh_info={"@type": "neuroglancer_weird_format"},
    )
    with pytest.raises(ValueError, match="Unsupported mesh"):
        estimate_segment_centroid(
            "precomputed://gs://example/segmentation", 42
        )
