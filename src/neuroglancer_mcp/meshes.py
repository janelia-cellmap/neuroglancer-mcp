"""Per-segment centroid estimation from precomputed mesh manifests.

The Neuroglancer JS viewer doesn't expose a "jump to segment centroid"
operation — selecting a segment loads its mesh and the 3D camera
auto-fits, but the cross-section panels stay wherever they were. For
an LLM-driven workflow ("show me a Kenyon cell") we want the viewer
to actually move to where the segment lives. This module reads the
mesh manifest for a segment and returns a representative position.

What we DO NOT do here:
- Decode Draco vertex data. We only read the manifest header, which is
  plain little-endian binary. No native Draco decoder needed.
- Compute true vertex-average centroids. We use the bounding-box
  centroid of the coarsest LOD's fragment positions — accurate enough
  to put the viewer "on the segment", which is the actual user need.

Supported mesh formats (covers hemibrain, FlyWire, MICrONS, CellMap,
and most legacy datasets):

- `neuroglancer_multilod_draco` (sharded multiscale). Manifest binary
  layout per the precomputed mesh spec:
  https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md
- `neuroglancer_legacy_mesh` (unsharded single-scale). We fetch the
  per-segment JSON manifest and read the first fragment file's vertex
  header to extract one vertex position as an approximate location.

Sharding hash supports `identity` (hemibrain, MICrONS) and
`murmurhash3_x86_128` (FlyWire); both implemented in pure Python.
"""

from __future__ import annotations

import gzip
import json
import struct
import urllib.request
from typing import Any, Optional

from neuroglancer_mcp.segment_properties import _normalize_url


def _maybe_decompress(blob: bytes, encoding: str) -> bytes:
    """Apply the precomputed sharding encoding ('raw' or 'gzip')."""
    if encoding == "raw":
        return blob
    if encoding == "gzip":
        return gzip.decompress(blob)
    raise ValueError(f"Unsupported precomputed encoding {encoding!r}")


# ---------------------------------------------------------------------------
# HTTP fetcher (monkeypatched by tests)
# ---------------------------------------------------------------------------


def _fetch_bytes(url: str, byte_range: Optional[tuple[int, int]] = None) -> bytes:
    """Fetch raw bytes, optionally restricted to a byte range (inclusive)."""
    req = urllib.request.Request(url)
    if byte_range is not None:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _fetch_json(url: str) -> dict[str, Any]:
    return json.loads(_fetch_bytes(url))


# ---------------------------------------------------------------------------
# MurmurHash3 x86 128-bit (low 64 bits)
# ---------------------------------------------------------------------------

_MASK32 = 0xFFFFFFFF


def _rotl32(x: int, r: int) -> int:
    x &= _MASK32
    return ((x << r) | (x >> (32 - r))) & _MASK32


def _fmix32(h: int) -> int:
    h &= _MASK32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _MASK32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _MASK32
    h ^= h >> 16
    return h


def _murmurhash3_x86_128_low64(data: bytes, seed: int = 0) -> int:
    """Return the low 64 bits of MurmurHash3 x86 128-bit.

    Matches the algorithm in https://github.com/aappleby/smhasher;
    Neuroglancer's sharded format spec calls out this exact variant.
    """
    h1 = h2 = h3 = h4 = seed & _MASK32
    c1, c2, c3, c4 = 0x239B961B, 0xAB0E9789, 0x38B34AE5, 0xA1E38B93
    n = len(data)
    nblocks = n // 16

    # body
    for i in range(nblocks):
        off = i * 16
        k1, k2, k3, k4 = struct.unpack_from("<IIII", data, off)
        k1 = (k1 * c1) & _MASK32; k1 = _rotl32(k1, 15); k1 = (k1 * c2) & _MASK32; h1 ^= k1
        h1 = _rotl32(h1, 19); h1 = (h1 + h2) & _MASK32; h1 = (h1 * 5 + 0x561CCD1B) & _MASK32
        k2 = (k2 * c2) & _MASK32; k2 = _rotl32(k2, 16); k2 = (k2 * c3) & _MASK32; h2 ^= k2
        h2 = _rotl32(h2, 17); h2 = (h2 + h3) & _MASK32; h2 = (h2 * 5 + 0x0BCAA747) & _MASK32
        k3 = (k3 * c3) & _MASK32; k3 = _rotl32(k3, 17); k3 = (k3 * c4) & _MASK32; h3 ^= k3
        h3 = _rotl32(h3, 15); h3 = (h3 + h4) & _MASK32; h3 = (h3 * 5 + 0x96CD1C35) & _MASK32
        k4 = (k4 * c4) & _MASK32; k4 = _rotl32(k4, 18); k4 = (k4 * c1) & _MASK32; h4 ^= k4
        h4 = _rotl32(h4, 13); h4 = (h4 + h1) & _MASK32; h4 = (h4 * 5 + 0x32AC3B17) & _MASK32

    # tail
    tail_off = nblocks * 16
    tail_len = n - tail_off
    k1 = k2 = k3 = k4 = 0
    if tail_len >= 13:
        k4 |= data[tail_off + 12]
        if tail_len >= 14: k4 |= data[tail_off + 13] << 8
        if tail_len >= 15: k4 |= data[tail_off + 14] << 16
        k4 = (k4 * c4) & _MASK32; k4 = _rotl32(k4, 18); k4 = (k4 * c1) & _MASK32; h4 ^= k4
    if tail_len >= 9:
        # bytes 8..11
        b8 = data[tail_off + 8]
        b9 = data[tail_off + 9] if tail_len >= 10 else 0
        b10 = data[tail_off + 10] if tail_len >= 11 else 0
        b11 = data[tail_off + 11] if tail_len >= 12 else 0
        k3 = b8 | (b9 << 8) | (b10 << 16) | (b11 << 24)
        k3 = (k3 * c3) & _MASK32; k3 = _rotl32(k3, 17); k3 = (k3 * c4) & _MASK32; h3 ^= k3
    if tail_len >= 5:
        b4 = data[tail_off + 4]
        b5 = data[tail_off + 5] if tail_len >= 6 else 0
        b6 = data[tail_off + 6] if tail_len >= 7 else 0
        b7 = data[tail_off + 7] if tail_len >= 8 else 0
        k2 = b4 | (b5 << 8) | (b6 << 16) | (b7 << 24)
        k2 = (k2 * c2) & _MASK32; k2 = _rotl32(k2, 16); k2 = (k2 * c3) & _MASK32; h2 ^= k2
    if tail_len >= 1:
        b0 = data[tail_off]
        b1 = data[tail_off + 1] if tail_len >= 2 else 0
        b2 = data[tail_off + 2] if tail_len >= 3 else 0
        b3 = data[tail_off + 3] if tail_len >= 4 else 0
        k1 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        k1 = (k1 * c1) & _MASK32; k1 = _rotl32(k1, 15); k1 = (k1 * c2) & _MASK32; h1 ^= k1

    # finalization
    h1 ^= n; h2 ^= n; h3 ^= n; h4 ^= n
    h1 = (h1 + h2) & _MASK32; h1 = (h1 + h3) & _MASK32; h1 = (h1 + h4) & _MASK32
    h2 = (h2 + h1) & _MASK32; h3 = (h3 + h1) & _MASK32; h4 = (h4 + h1) & _MASK32
    h1 = _fmix32(h1); h2 = _fmix32(h2); h3 = _fmix32(h3); h4 = _fmix32(h4)
    h1 = (h1 + h2) & _MASK32; h1 = (h1 + h3) & _MASK32; h1 = (h1 + h4) & _MASK32
    h2 = (h2 + h1) & _MASK32

    return h1 | (h2 << 32)


# ---------------------------------------------------------------------------
# Sharded mesh reader (neuroglancer_multilod_draco)
# ---------------------------------------------------------------------------


def _hash_segment_id(segment_id: int, sharding: dict[str, Any]) -> int:
    """Apply the sharding spec's hash function to a segment ID."""
    preshift = sharding.get("preshift_bits", 0)
    hash_name = sharding.get("hash", "identity")
    key = segment_id >> preshift
    if hash_name == "identity":
        return key
    if hash_name == "murmurhash3_x86_128":
        return _murmurhash3_x86_128_low64(struct.pack("<Q", key & 0xFFFFFFFFFFFFFFFF))
    raise ValueError(f"Unsupported sharding hash {hash_name!r}")


def _shard_and_minishard(
    segment_id: int, sharding: dict[str, Any]
) -> tuple[int, int]:
    """Return (shard_number, minishard_number) for a segment ID."""
    hashed = _hash_segment_id(segment_id, sharding)
    minishard_bits = sharding["minishard_bits"]
    shard_bits = sharding["shard_bits"]
    minishard = hashed & ((1 << minishard_bits) - 1)
    shard = (hashed >> minishard_bits) & ((1 << shard_bits) - 1)
    return shard, minishard


def _shard_filename(shard: int, shard_bits: int) -> str:
    """Hex-encoded shard number, zero-padded to the spec's hex digit count."""
    # Each hex digit is 4 bits; round up.
    hex_digits = max(1, (shard_bits + 3) // 4)
    return f"{shard:0{hex_digits}x}.shard"


def _decode_minishard_index(
    raw: bytes, encoding: str
) -> list[tuple[int, int, int]]:
    """Decode a minishard index into a list of (chunk_id, offset, size).

    Layout: three uint64 arrays of equal length, concatenated. Each
    array is delta-encoded (each value is the delta from the previous).
    The offsets are deltas from the *end* of the previous chunk's data.
    """
    raw = _maybe_decompress(raw, encoding)
    n = len(raw) // 24
    if n == 0:
        return []
    arr = struct.unpack(f"<{3 * n}Q", raw)
    chunk_ids = arr[:n]
    offsets = arr[n : 2 * n]
    sizes = arr[2 * n : 3 * n]

    out: list[tuple[int, int, int]] = []
    chunk_id = 0
    cursor = 0
    for cid_delta, off_delta, size in zip(chunk_ids, offsets, sizes):
        chunk_id += cid_delta
        cursor += off_delta
        out.append((chunk_id, cursor, size))
        cursor += size
    return out


def _fetch_multilod_manifest_blob(
    mesh_url: str, sharding: dict[str, Any], segment_id: int
) -> bytes:
    """Locate and fetch the binary manifest blob for one segment.

    Walks the sharded format: shard file header → minishard offsets →
    minishard index (decoded) → manifest byte range within the shard.
    Uses HTTP Range requests so we never download the full shard.
    """
    shard_no, minishard_no = _shard_and_minishard(segment_id, sharding)
    shard_url = mesh_url.rstrip("/") + "/" + _shard_filename(shard_no, sharding["shard_bits"])

    # Shard header: 2 uint64 per minishard, giving (start, end) of minishard
    # index within the shard data region (which begins right after the header).
    minishard_bits = sharding["minishard_bits"]
    n_minishards = 1 << minishard_bits
    header_len = n_minishards * 16
    header = _fetch_bytes(shard_url, byte_range=(0, header_len - 1))
    if len(header) < header_len:
        raise ValueError(
            f"Shard header for {shard_url} truncated "
            f"({len(header)} of {header_len} bytes)"
        )
    idx_start, idx_end = struct.unpack_from("<QQ", header, minishard_no * 16)
    if idx_start == idx_end:
        raise KeyError(
            f"Segment {segment_id} not present: minishard {minishard_no} is empty"
        )

    # Minishard index lives at offset (header_len + idx_start), length (idx_end - idx_start).
    idx_off = header_len + idx_start
    idx_len = idx_end - idx_start
    idx_raw = _fetch_bytes(shard_url, byte_range=(idx_off, idx_off + idx_len - 1))

    entries = _decode_minishard_index(
        idx_raw, sharding.get("minishard_index_encoding", "raw")
    )
    data_encoding = sharding.get("data_encoding", "raw")
    for chunk_id, off, size in entries:
        if chunk_id == segment_id:
            data_off = header_len + off
            blob = _fetch_bytes(
                shard_url, byte_range=(data_off, data_off + size - 1)
            )
            return _maybe_decompress(blob, data_encoding)
    raise KeyError(f"Segment {segment_id} not found in minishard {minishard_no}")


def _decode_multilod_manifest(blob: bytes) -> dict[str, Any]:
    """Decode the per-segment binary manifest header.

    Layout (per https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md):
        chunk_shape           : 3 × float32 LE
        grid_origin           : 3 × float32 LE
        num_lods              : uint32 LE
        lod_scales            : num_lods × float32 LE
        vertex_offsets        : num_lods × 3 × float32 LE
        num_fragments_per_lod : num_lods × uint32 LE
        per-LOD i:
            fragment_positions : num_fragments_per_lod[i] × 3 × uint32 LE
            fragment_offsets   : num_fragments_per_lod[i] × uint32 LE
              (despite the name this is the *size in bytes* of each
               fragment's Draco data, not a byte offset)
    """
    chunk_shape = struct.unpack_from("<3f", blob, 0)
    grid_origin = struct.unpack_from("<3f", blob, 12)
    (num_lods,) = struct.unpack_from("<I", blob, 24)
    off = 28

    lod_scales = struct.unpack_from(f"<{num_lods}f", blob, off)
    off += 4 * num_lods
    vertex_offsets = [
        struct.unpack_from("<3f", blob, off + 12 * i) for i in range(num_lods)
    ]
    off += 12 * num_lods
    num_fragments_per_lod = struct.unpack_from(f"<{num_lods}I", blob, off)
    off += 4 * num_lods

    lod_fragment_positions: list[list[tuple[int, int, int]]] = []
    for n_frag in num_fragments_per_lod:
        # fragment_positions is a [3, N] uint32 array in C order — all xs
        # first, then all ys, then all zs. NOT interleaved triplets.
        if n_frag > 0:
            xs = struct.unpack_from(f"<{n_frag}I", blob, off)
            ys = struct.unpack_from(f"<{n_frag}I", blob, off + 4 * n_frag)
            zs = struct.unpack_from(f"<{n_frag}I", blob, off + 8 * n_frag)
            positions = list(zip(xs, ys, zs))
        else:
            positions = []
        off += 12 * n_frag  # fragment_positions
        off += 4 * n_frag   # fragment_offsets (sizes; unused for centroid)
        lod_fragment_positions.append(positions)

    return {
        "chunk_shape": chunk_shape,
        "grid_origin": grid_origin,
        "num_lods": num_lods,
        "lod_scales": list(lod_scales),
        "vertex_offsets": vertex_offsets,
        "num_fragments_per_lod": list(num_fragments_per_lod),
        "fragment_positions": lod_fragment_positions,
    }


def _centroid_from_multilod_manifest(manifest: dict[str, Any]) -> tuple[float, float, float]:
    """Bounding-box center of the finest non-empty LOD, in mesh-model coords.

    We use the bounding box of fragment grid positions (not their
    average) so the result tracks the segment's geometric center
    rather than where the fragments happen to be densest — this
    matches what Neuroglancer itself uses when navigating to a
    segment. The finest LOD gives the tightest bounding box; coarser
    LODs over-cover. If LOD 0 is missing we fall back to the next
    finest non-empty level.
    """
    fragments = manifest["fragment_positions"]
    scales = manifest["lod_scales"]
    chunk = manifest["chunk_shape"]
    origin = manifest["grid_origin"]

    for lod in range(manifest["num_lods"]):
        positions = fragments[lod]
        if not positions:
            continue
        scale = scales[lod]
        step = (chunk[0] * scale, chunk[1] * scale, chunk[2] * scale)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        # Each fragment occupies grid cell [pos, pos+1] in chunk units.
        # Bounding box in mesh-model coords spans [min*step, (max+1)*step].
        cx = origin[0] + (min(xs) + max(xs) + 1) * 0.5 * step[0]
        cy = origin[1] + (min(ys) + max(ys) + 1) * 0.5 * step[1]
        cz = origin[2] + (min(zs) + max(zs) + 1) * 0.5 * step[2]
        return (cx, cy, cz)
    raise ValueError("Manifest has no fragments at any LOD")


def _apply_transform(
    transform: Optional[list[float]], point: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply the mesh's 3×4 affine transform (row-major, length 12).

    Maps mesh-model coords to the volume's physical (nm) coord space.
    Falls back to identity when the mesh has no transform.
    """
    if transform is None:
        return point
    if len(transform) != 12:
        raise ValueError(
            f"Mesh transform must have 12 elements, got {len(transform)}"
        )
    x, y, z = point
    return (
        transform[0] * x + transform[1] * y + transform[2] * z + transform[3],
        transform[4] * x + transform[5] * y + transform[6] * z + transform[7],
        transform[8] * x + transform[9] * y + transform[10] * z + transform[11],
    )


# ---------------------------------------------------------------------------
# Legacy unsharded mesh (neuroglancer_legacy_mesh)
# ---------------------------------------------------------------------------


def _centroid_from_legacy(mesh_url: str, segment_id: int) -> tuple[float, float, float]:
    """Approximate centroid by reading one vertex from the first fragment.

    Legacy mesh manifests are JSON: `{"fragments": ["file1", "file2"]}`.
    Fragment files are binary `<num_vertices uint32><vertices float32×3×N>
    <triangles uint32×3×M>`. We fetch only the first fragment and average
    its vertices — same coordinate space as the mesh (not the segmentation),
    but for legacy precomputed those are the same up to a transform.
    """
    manifest_url = mesh_url.rstrip("/") + f"/{segment_id}:0"
    manifest = _fetch_json(manifest_url)
    fragments = manifest.get("fragments") or []
    if not fragments:
        raise ValueError(f"Segment {segment_id} has no mesh fragments")

    frag_url = mesh_url.rstrip("/") + "/" + fragments[0]
    blob = _fetch_bytes(frag_url)
    if len(blob) < 4:
        raise ValueError(f"Mesh fragment {frag_url} too short")
    (n_vertices,) = struct.unpack_from("<I", blob, 0)
    if n_vertices == 0:
        raise ValueError(f"Mesh fragment {frag_url} has zero vertices")

    verts_end = 4 + 12 * n_vertices
    if len(blob) < verts_end:
        raise ValueError(f"Mesh fragment {frag_url} truncated")
    verts = struct.unpack_from(f"<{3 * n_vertices}f", blob, 4)
    sx = sum(verts[0::3])
    sy = sum(verts[1::3])
    sz = sum(verts[2::3])
    return (sx / n_vertices, sy / n_vertices, sz / n_vertices)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_info_cache: dict[str, dict[str, Any]] = {}


def _cached_json(url: str) -> dict[str, Any]:
    if url not in _info_cache:
        _info_cache[url] = _fetch_json(url)
    return _info_cache[url]


def clear_cache() -> None:
    _info_cache.clear()


def _segmentation_voxel_size_nm(seg_info: dict[str, Any]) -> tuple[float, float, float]:
    """Extract the finest-scale voxel size from a segmentation info file.

    Precomputed info files publish `scales[i].resolution` as nm per
    voxel. We use scale 0 (the finest), which is also the coord space
    the mesh's fragment positions live in.
    """
    scales = seg_info.get("scales")
    if not scales:
        raise ValueError("Segmentation info has no 'scales'")
    res = scales[0].get("resolution")
    if not res or len(res) < 3:
        raise ValueError("Segmentation info scale[0] missing 'resolution'")
    return float(res[0]), float(res[1]), float(res[2])


def estimate_segment_centroid(
    segmentation_source: str, segment_id: int
) -> dict[str, Any]:
    """Return an approximate centroid for one segment.

    Pipeline (matches what Neuroglancer itself does when navigating to
    a segment):

    1. Decode the per-segment mesh manifest. Take the bounding-box
       center of the finest non-empty LOD's fragment positions; this
       lives in *mesh-model* coords.
    2. Apply the mesh's `transform` field (a 3×4 affine) to map
       mesh-model coords → the volume's physical coord space (nm,
       since precomputed `resolution` is published in nm).
    3. Divide by the segmentation's `scales[0].resolution` to get
       segmentation voxel coords.

    The returned `position` (voxel coords) is what you set
    `viewer.position` to — viewer dimensions, once loaded, will be in
    the same space.

    Returns:
        {"position": [x, y, z]            (segmentation voxel coords),
         "position_nm": [x, y, z]         (physical nm),
         "voxel_size_nm": [rx, ry, rz],
         "mesh_type": <str>}.

    Raises:
        ValueError: if the segmentation has no mesh, or the mesh
            format is unsupported.
        KeyError: if the segment ID has no manifest in the mesh.
    """
    base = _normalize_url(segmentation_source).rstrip("/")
    seg_info = _cached_json(base + "/info")
    mesh_field = seg_info.get("mesh")
    if not mesh_field:
        raise ValueError(
            f"Segmentation at {segmentation_source!r} has no mesh subdir"
        )

    mesh_url = base + "/" + mesh_field
    mesh_info = _cached_json(mesh_url + "/info")
    mesh_type = mesh_info.get("@type")
    voxel_size = _segmentation_voxel_size_nm(seg_info)

    if mesh_type == "neuroglancer_multilod_draco":
        sharding = mesh_info.get("sharding")
        if not sharding:
            raise ValueError(
                "multilod_draco mesh has no sharding spec — unsharded "
                "multilod is not supported in v0.1"
            )
        blob = _fetch_multilod_manifest_blob(mesh_url, sharding, segment_id)
        manifest = _decode_multilod_manifest(blob)
        model_xyz = _centroid_from_multilod_manifest(manifest)
    elif mesh_type == "neuroglancer_legacy_mesh":
        model_xyz = _centroid_from_legacy(mesh_url, segment_id)
    else:
        raise ValueError(f"Unsupported mesh @type: {mesh_type!r}")

    transform = mesh_info.get("transform")
    nx, ny, nz = _apply_transform(transform, model_xyz)
    vx = nx / voxel_size[0]
    vy = ny / voxel_size[1]
    vz = nz / voxel_size[2]

    return {
        "position": [vx, vy, vz],
        "position_nm": [nx, ny, nz],
        "voxel_size_nm": list(voxel_size),
        "mesh_type": mesh_type,
    }
