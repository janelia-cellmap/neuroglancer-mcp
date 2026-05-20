"""segment_properties query tools.

The Python `neuroglancer` package does not expose segment search by
label — you have to know IDs already. These tools fill that gap so an
LLM can say "show all Kenyon cells" instead of asking the user to look
up IDs first.

All six tools resolve the segment_properties URL from the layer's
source(s), fetch and cache the file, then dispatch to the data model
in `neuroglancer_mcp.segment_properties`.
"""

from __future__ import annotations

from typing import Any

import neuroglancer

from neuroglancer_mcp.segment_properties import (
    SegmentProperties,
    load_segment_properties,
)
from neuroglancer_mcp.server import mcp
from neuroglancer_mcp.viewer import get_viewer


def _layer_source_urls(viewer: neuroglancer.Viewer, layer_name: str) -> list[str]:
    """Extract the list of source URL strings from a segmentation layer."""
    with viewer.txn() as s:
        for layer in s.layers:
            if layer.name == layer_name:
                if not isinstance(layer.layer, neuroglancer.SegmentationLayer):
                    raise ValueError(
                        f"Layer {layer_name!r} is not a segmentation layer"
                    )
                src = layer.layer.source
                if src is None:
                    raise ValueError(f"Layer {layer_name!r} has no source")
                if isinstance(src, (list, tuple)) or hasattr(src, "__iter__") and not isinstance(src, str):
                    return [_source_url(x) for x in src]
                return [_source_url(src)]
        raise ValueError(f"Segmentation layer {layer_name!r} not found")


def _source_url(source: Any) -> str:
    """Coerce a neuroglancer LayerDataSource (or plain string) to a URL string."""
    if hasattr(source, "url"):
        return str(source.url)
    return str(source)


def _load(layer_name: str) -> SegmentProperties:
    return load_segment_properties(_layer_source_urls(get_viewer(), layer_name))


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


@mcp.tool()
def get_segment_properties_schema(layer: str) -> dict[str, Any]:
    """Describe the segment_properties available for a segmentation layer.

    Returns the list of properties (id, type, description, data_type,
    num_unique for labels/descriptions, tag vocabulary for tags) plus
    the total number of segments. Call this first when you don't know
    what properties a dataset publishes — it's cheap and tells you
    which other tools (`search_segments_by_label`, `filter_segments`,
    `top_n_segments_by_property`) you can use.

    Args:
        layer: Name of an already-added segmentation layer.

    Returns:
        {"num_segments": int, "properties": [{id, type, ...}, ...]}.
    """
    sp = _load(layer)
    return sp.schema()


@mcp.tool()
def get_segment_property(
    layer: str, segment_id: int, property_id: str
) -> dict[str, Any]:
    """Return one property's value for one segment.

    Useful for sanity checks ("what's the label of segment 5901222347?")
    and for following up on a search result.

    Args:
        layer: Segmentation layer name.
        segment_id: The segment ID to query.
        property_id: The property `id` as returned by
            `get_segment_properties_schema`.

    Returns:
        {"layer", "segment_id", "property_id", "value"}. For `tags`
        properties, `value` is the list of tag names (not the on-disk
        index encoding).
    """
    sp = _load(layer)
    return {
        "layer": layer,
        "segment_id": segment_id,
        "property_id": property_id,
        "value": sp.get(segment_id, property_id),
    }


# ---------------------------------------------------------------------------
# Search & filter
# ---------------------------------------------------------------------------


DEFAULT_LIMIT = 50
"""Cap on rows returned by search/filter tools.

Hemibrain has ~25K labeled neurons and a substring match like "KC" can
hit thousands. Returning all of them blows past the MCP response
budget and forces the client to spill to disk. The default keeps
responses small enough for the model to read inline; callers can raise
`limit` explicitly when they actually want the full set.
"""


@mcp.tool()
def search_segments_by_label(
    layer: str, query: str, regex: bool = False, limit: int = DEFAULT_LIMIT
) -> dict[str, Any]:
    """Find segments whose label matches a query string.

    Case-insensitive substring match by default. Set `regex=True` to
    use Python regex (also case-insensitive). Hemibrain-class datasets
    often have thousands of matches for short queries — results are
    capped at `limit` (default 50) and `total_matches` reports the
    untruncated count so you can refine before re-querying.

    Args:
        layer: Segmentation layer name.
        query: Substring (default) or regex pattern.
        regex: True to interpret `query` as a Python regex.
        limit: Maximum matches to return. Default 50. Pass a larger
            number explicitly when you need the full set; the response
            will be proportionally larger.

    Returns:
        {"layer", "query", "regex", "total_matches", "returned",
         "truncated": bool, "matches": [{id, label}, ...]}.
    """
    sp = _load(layer)
    matches = sp.search_by_label(query, regex=regex)
    truncated = len(matches) > limit
    return {
        "layer": layer,
        "query": query,
        "regex": regex,
        "total_matches": len(matches),
        "returned": min(len(matches), limit),
        "truncated": truncated,
        "matches": matches[:limit],
    }


@mcp.tool()
def filter_segments(
    layer: str, criteria: dict[str, Any], limit: int = DEFAULT_LIMIT
) -> dict[str, Any]:
    """Return segment IDs satisfying multiple criteria at once.

    Combine label patterns, required tags, and numeric ranges. All
    criteria are AND-ed together. Example criteria:

        {"label": "KC*", "tags": ["Traced"], "size": {"min": 1000}}

    means "labels starting with KC, tagged Traced, with size >= 1000".

    Results are capped at `limit` (default 50); `total_matches`
    reports the untruncated count.

    Args:
        layer: Segmentation layer name.
        criteria: Mapping with these recognized keys:
            - "label": glob pattern (fnmatch, case-insensitive) over
              the label property — `*` and `?` wildcards work.
            - "tags": list of tag names; segments must have all of them.
            - any other key: treated as a `property_id`. The value is
              either an exact value or a dict with "min"/"max" for a
              numeric range.
        limit: Maximum IDs to return. Default 50.

    Returns:
        {"layer", "criteria", "total_matches", "returned",
         "truncated": bool, "matching_segment_ids": [int, ...]}.
    """
    sp = _load(layer)
    ids = sp.filter(criteria)
    truncated = len(ids) > limit
    return {
        "layer": layer,
        "criteria": criteria,
        "total_matches": len(ids),
        "returned": min(len(ids), limit),
        "truncated": truncated,
        "matching_segment_ids": ids[:limit],
    }


@mcp.tool()
def top_n_segments_by_property(
    layer: str, property_id: str, n: int, ascending: bool = False
) -> dict[str, Any]:
    """Return the N segments with the largest (or smallest) values of a numeric property.

    Args:
        layer: Segmentation layer name.
        property_id: A `number`-type property ID. Use
            `get_segment_properties_schema` to find available ones.
        n: How many segments to return.
        ascending: False (default) for largest first; True for smallest.

    Returns:
        {"layer", "property_id", "n", "ascending",
         "results": [{"id": int, "value": number}, ...]}.
    """
    sp = _load(layer)
    results = sp.top_n(property_id, n, ascending=ascending)
    return {
        "layer": layer,
        "property_id": property_id,
        "n": n,
        "ascending": ascending,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Convenience: filter + show in one call
# ---------------------------------------------------------------------------


@mcp.tool()
def show_segments_by_property(
    layer: str, criteria: dict[str, Any]
) -> dict[str, Any]:
    """Filter segments by `criteria` and display the matching set.

    Equivalent to calling `filter_segments` followed by `show_segments`,
    but in one round trip. This is the workhorse for "show all the
    Kenyon cells where status is Traced" — a single tool call goes from
    natural-language criteria to a live viewer update.

    Args:
        layer: Segmentation layer name.
        criteria: Same shape as `filter_segments`.

    Returns:
        {"layer", "criteria", "num_visible",
         "visible_segments": [int, ...]}.
    """
    viewer = get_viewer()
    sp = load_segment_properties(_layer_source_urls(viewer, layer))
    ids = sp.filter(criteria)
    with viewer.txn() as s:
        for ng_layer in s.layers:
            if ng_layer.name == layer:
                if not isinstance(ng_layer.layer, neuroglancer.SegmentationLayer):
                    raise ValueError(
                        f"Layer {layer!r} is not a segmentation layer"
                    )
                ng_layer.layer.segments = set(ids)
                break
        else:
            raise ValueError(f"Segmentation layer {layer!r} not found")
    return {
        "layer": layer,
        "criteria": criteria,
        "num_visible": len(ids),
        "visible_segments": sorted(ids),
    }
