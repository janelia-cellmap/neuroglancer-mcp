"""Unit tests for the SegmentProperties data model.

No viewer, no network. The fixture covers all four property types
(`label`, `description`, `number`, `tags`) and the queries the MCP
tools dispatch into.
"""

from __future__ import annotations

import pytest

from neuroglancer_mcp.segment_properties import (
    SegmentProperties,
    _normalize_url,
    _resolve_info_url,
)


def _fixture_info() -> dict:
    """A miniature segment_properties/info covering all four types.

    Three segments with deliberately ragged data so filter/search edge
    cases (case-insensitivity, glob wildcards, tag subset, numeric
    ranges) are exercised.
    """
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
                    "id": "description",
                    "type": "description",
                    "values": ["kenyon cell α", "kenyon cell β", "mushroom body output"],
                },
                {
                    "id": "size",
                    "type": "number",
                    "data_type": "uint64",
                    "description": "voxel count",
                    "values": [1500, 500, 9000],
                },
                {
                    "id": "tags",
                    "type": "tags",
                    "tags": ["Traced", "Leaves", "KC_class", "MBON_class"],
                    "tag_descriptions": [
                        "tracing complete",
                        "no untraced branches",
                        "Kenyon cell",
                        "Mushroom body output neuron",
                    ],
                    "values": [[0, 2], [2], [0, 3]],
                },
            ],
        },
    }


@pytest.fixture
def sp() -> SegmentProperties:
    return SegmentProperties.from_info(_fixture_info())


# ---------------------------------------------------------------------------
# from_info / schema
# ---------------------------------------------------------------------------


def test_from_info_parses_ids_as_int(sp: SegmentProperties):
    assert sp.ids == [100, 200, 300]


def test_from_info_rejects_wrong_type():
    with pytest.raises(ValueError, match="Not a segment_properties"):
        SegmentProperties.from_info({"@type": "neuroglancer_skeletons", "inline": {}})


def test_from_info_rejects_non_inline():
    with pytest.raises(ValueError, match="inline"):
        SegmentProperties.from_info({"@type": "neuroglancer_segment_properties"})


def test_schema_shape(sp: SegmentProperties):
    schema = sp.schema()
    assert schema["num_segments"] == 3
    by_id = {p["id"]: p for p in schema["properties"]}
    assert by_id["label"]["type"] == "label"
    assert by_id["label"]["num_unique"] == 3
    assert by_id["size"]["data_type"] == "uint64"
    assert by_id["size"]["description"] == "voxel count"
    assert by_id["tags"]["tags"] == ["Traced", "Leaves", "KC_class", "MBON_class"]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def test_get_label_value(sp: SegmentProperties):
    assert sp.get(100, "label") == "KC_alpha"


def test_get_tags_returns_names_not_indices(sp: SegmentProperties):
    assert sp.get(100, "tags") == ["Traced", "KC_class"]
    assert sp.get(300, "tags") == ["Traced", "MBON_class"]


def test_get_missing_segment_raises(sp: SegmentProperties):
    with pytest.raises(KeyError, match="999"):
        sp.get(999, "label")


def test_get_missing_property_raises(sp: SegmentProperties):
    with pytest.raises(KeyError, match="weight"):
        sp.get(100, "weight")


# ---------------------------------------------------------------------------
# search_by_label
# ---------------------------------------------------------------------------


def test_search_substring_case_insensitive(sp: SegmentProperties):
    matches = sp.search_by_label("kc")
    assert {m["id"] for m in matches} == {100, 200}


def test_search_no_matches(sp: SegmentProperties):
    assert sp.search_by_label("DPM") == []


def test_search_regex(sp: SegmentProperties):
    matches = sp.search_by_label(r"^MBON\d+$", regex=True)
    assert [m["id"] for m in matches] == [300]


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_by_label_glob(sp: SegmentProperties):
    assert sp.filter({"label": "KC*"}) == [100, 200]


def test_filter_by_tags_requires_all_tags(sp: SegmentProperties):
    # Traced AND KC_class — only 100 has both.
    assert sp.filter({"tags": ["Traced", "KC_class"]}) == [100]


def test_filter_by_unknown_tag_raises(sp: SegmentProperties):
    with pytest.raises(KeyError, match="Nonexistent"):
        sp.filter({"tags": ["Nonexistent"]})


def test_filter_by_numeric_range(sp: SegmentProperties):
    assert sp.filter({"size": {"min": 1000}}) == [100, 300]
    assert sp.filter({"size": {"min": 1000, "max": 5000}}) == [100]


def test_filter_by_exact_numeric(sp: SegmentProperties):
    assert sp.filter({"size": 500}) == [200]


def test_filter_combined(sp: SegmentProperties):
    # KC labels + Traced tag + size >= 1000 → only segment 100.
    result = sp.filter(
        {"label": "KC*", "tags": ["Traced"], "size": {"min": 1000}}
    )
    assert result == [100]


# ---------------------------------------------------------------------------
# top_n
# ---------------------------------------------------------------------------


def test_top_n_descending(sp: SegmentProperties):
    assert sp.top_n("size", 2) == [
        {"id": 300, "value": 9000},
        {"id": 100, "value": 1500},
    ]


def test_top_n_ascending(sp: SegmentProperties):
    assert sp.top_n("size", 2, ascending=True) == [
        {"id": 200, "value": 500},
        {"id": 100, "value": 1500},
    ]


def test_top_n_non_numeric_raises(sp: SegmentProperties):
    with pytest.raises(ValueError, match="not 'number'"):
        sp.top_n("label", 2)


def test_top_n_zero_returns_empty(sp: SegmentProperties):
    assert sp.top_n("size", 0) == []


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def test_normalize_url_strips_precomputed_prefix():
    assert (
        _normalize_url("precomputed://gs://bucket/path")
        == "https://storage.googleapis.com/bucket/path"
    )


def test_normalize_url_passes_through_https():
    assert (
        _normalize_url("precomputed://https://example.com/seg")
        == "https://example.com/seg"
    )


def test_resolve_info_url_default_path():
    assert (
        _resolve_info_url(["precomputed://gs://bucket/seg"])
        == "https://storage.googleapis.com/bucket/seg/segment_properties/info"
    )


def test_resolve_info_url_prefers_explicit_segment_properties_source():
    sources = [
        "precomputed://gs://bucket/seg",
        "precomputed://gs://bucket/seg/segment_properties",
    ]
    assert (
        _resolve_info_url(sources)
        == "https://storage.googleapis.com/bucket/seg/segment_properties/info"
    )


def test_resolve_info_url_empty_raises():
    with pytest.raises(ValueError, match="no sources"):
        _resolve_info_url([])
