"""Unit tests for coordinate conversion helpers in `viewer.py`.

These don't construct a real Neuroglancer viewer — they exercise the
nm↔voxel math against a hand-built `CoordinateSpace`.
"""

from __future__ import annotations

import neuroglancer
import pytest

from neuroglancer_mcp.viewer import nm_to_voxels


def _space(scales_nm: tuple[float, float, float]) -> neuroglancer.CoordinateSpace:
    """Build an (x, y, z) coordinate space with scales given in nm.

    Neuroglancer's CoordinateSpace stores scales in the supplied units;
    'nm' lets us pass nm scales directly.
    """
    return neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        units=["nm", "nm", "nm"],
        scales=list(scales_nm),
    )


def test_nm_to_voxels_unit_scale():
    space = _space((1.0, 1.0, 1.0))
    assert nm_to_voxels(100, 200, 300, space) == [100.0, 200.0, 300.0]


def test_nm_to_voxels_anisotropic():
    # 8 nm / 8 nm / 40 nm — typical EM dataset.
    space = _space((8.0, 8.0, 40.0))
    assert nm_to_voxels(800, 1600, 4000, space) == [100.0, 200.0, 100.0]


def test_nm_to_voxels_meters():
    # Some sources advertise scales in meters with float values.
    space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        units=["m", "m", "m"],
        scales=[8e-9, 8e-9, 40e-9],
    )
    assert nm_to_voxels(800, 1600, 4000, space) == [100.0, 200.0, 100.0]


def test_nm_to_voxels_rejects_unknown_unit():
    space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"],
        units=["s", "s", "s"],
        scales=[1.0, 1.0, 1.0],
    )
    with pytest.raises(ValueError, match="Unsupported unit"):
        nm_to_voxels(1, 1, 1, space)


def test_nm_to_voxels_non_spatial_axis_zero():
    # 4D space with a channel axis at the front — channel should be 0,
    # the spatial axes should still convert.
    space = neuroglancer.CoordinateSpace(
        names=["c", "x", "y", "z"],
        units=["", "nm", "nm", "nm"],
        scales=[1.0, 8.0, 8.0, 40.0],
    )
    assert nm_to_voxels(800, 1600, 4000, space) == [0.0, 100.0, 200.0, 100.0]
