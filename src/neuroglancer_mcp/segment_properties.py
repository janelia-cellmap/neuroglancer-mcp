"""Parser, query model, and fetcher for Neuroglancer `segment_properties` files.

The `segment_properties/info` file is a sidecar published alongside a
precomputed segmentation. It maps segment IDs to human-readable labels,
descriptions, numeric properties (volume, length, synapse counts), and
categorical tags. The Python `neuroglancer` package does not expose
search or filter operations over this data — the viewer UI does, but
programmatic access is missing. That's the gap this module fills.

This module is intentionally independent of `neuroglancer_mcp.viewer`
and the MCP layer so the data model can be unit-tested without a viewer
and without network access.
"""

from __future__ import annotations

import fnmatch
import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Property:
    """One column in a segment_properties file.

    `type` is one of "label", "description", "number", "tags". For
    "tags", `values` is a list-of-lists of indices into `tags`. For the
    other types, `values` is a flat list with one entry per segment ID.
    """

    id: str
    type: str
    description: Optional[str] = None
    data_type: Optional[str] = None
    values: Optional[list[Any]] = None
    tags: Optional[list[str]] = None
    tag_descriptions: Optional[list[str]] = None


@dataclass
class SegmentProperties:
    ids: list[int]
    properties: list[Property]
    _id_to_index: dict[int, int] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> "SegmentProperties":
        """Build from a parsed `segment_properties/info` dict."""
        if info.get("@type") not in (None, "neuroglancer_segment_properties"):
            raise ValueError(
                f"Not a segment_properties info file (@type={info.get('@type')!r})"
            )
        inline = info.get("inline")
        if inline is None:
            raise ValueError(
                "Only inline segment_properties are supported in v0.1"
            )
        ids = [int(s) for s in inline["ids"]]
        props: list[Property] = []
        for p in inline.get("properties", []):
            props.append(
                Property(
                    id=p["id"],
                    type=p["type"],
                    description=p.get("description"),
                    data_type=p.get("data_type"),
                    values=p.get("values"),
                    tags=p.get("tags"),
                    tag_descriptions=p.get("tag_descriptions"),
                )
            )
        return cls(ids=ids, properties=props)

    # -- internal lookups --------------------------------------------------

    def _index_of(self, segment_id: int) -> int:
        if self._id_to_index is None:
            self._id_to_index = {sid: i for i, sid in enumerate(self.ids)}
        try:
            return self._id_to_index[segment_id]
        except KeyError as e:
            raise KeyError(
                f"Segment {segment_id} not in this segment_properties"
            ) from e

    def _property(self, property_id: str) -> Property:
        for p in self.properties:
            if p.id == property_id:
                return p
        raise KeyError(f"Property {property_id!r} not in this segment_properties")

    def _property_by_type(self, property_type: str) -> Property:
        for p in self.properties:
            if p.type == property_type:
                return p
        raise KeyError(f"No property of type {property_type!r} in this segment_properties")

    # -- public query API --------------------------------------------------

    def schema(self) -> dict[str, Any]:
        """Return a compact summary suitable for the schema tool.

        Includes per-property metadata (id, type, description, data_type,
        num_unique for label/description, tags list for "tags") plus
        the total number of segments.
        """
        props_out: list[dict[str, Any]] = []
        for p in self.properties:
            entry: dict[str, Any] = {"id": p.id, "type": p.type}
            if p.description is not None:
                entry["description"] = p.description
            if p.data_type is not None:
                entry["data_type"] = p.data_type
            if p.type in ("label", "description") and p.values is not None:
                entry["num_unique"] = len(set(p.values))
            if p.type == "tags" and p.tags is not None:
                entry["tags"] = list(p.tags)
                if p.tag_descriptions is not None:
                    entry["tag_descriptions"] = list(p.tag_descriptions)
            props_out.append(entry)
        return {"num_segments": len(self.ids), "properties": props_out}

    def get(self, segment_id: int, property_id: str) -> Any:
        """Return the value of a single property for a single segment.

        For "tags" properties, returns a list of tag *names* (not the
        index encoding stored on disk).
        """
        p = self._property(property_id)
        i = self._index_of(segment_id)
        if p.values is None:
            raise ValueError(f"Property {property_id!r} has no values")
        if p.type == "tags":
            assert p.tags is not None
            return [p.tags[t] for t in p.values[i]]
        return p.values[i]

    def search_by_label(
        self, query: str, regex: bool = False
    ) -> list[dict[str, Any]]:
        """Return [{id, label}] for every segment whose label matches `query`.

        Case-insensitive substring match by default; `regex=True`
        switches to Python `re.search` (also case-insensitive).
        Internal — the MCP tool caller is responsible for applying any
        `limit` before serializing the result.
        """
        label_prop = self._property_by_type("label")
        assert label_prop.values is not None
        if regex:
            pat = re.compile(query, re.IGNORECASE)
            match = pat.search
        else:
            q = query.lower()

            def match(s: str) -> bool:
                return q in s.lower()

        out: list[dict[str, Any]] = []
        for i, lbl in enumerate(label_prop.values):
            if lbl is not None and match(lbl):
                out.append({"id": self.ids[i], "label": lbl})
        return out

    def filter(self, criteria: dict[str, Any]) -> list[int]:
        """Return segment IDs satisfying ALL of the given criteria.

        Supported criteria keys:

        - `"label"`: a glob pattern (fnmatch) matched case-insensitively
          against the label property. Example: `"KC*"` matches all
          labels starting with "KC".
        - `"tags"`: a list of tag names; segments must have ALL of them.
        - Any other key is treated as a `property_id`. The value is
          either:
            - a number → exact match
            - a string → exact match
            - a dict with `min`/`max` keys → inclusive numeric range.
        """
        n = len(self.ids)
        candidates: set[int] = set(range(n))

        for key, val in criteria.items():
            if key == "label":
                label_prop = self._property_by_type("label")
                pat = val.lower()
                assert label_prop.values is not None
                candidates &= {
                    i
                    for i in candidates
                    if label_prop.values[i] is not None
                    and fnmatch.fnmatch(label_prop.values[i].lower(), pat)
                }
            elif key == "tags":
                tags_prop = self._property_by_type("tags")
                assert tags_prop.tags is not None and tags_prop.values is not None
                required: set[int] = set()
                for t in val:
                    if t not in tags_prop.tags:
                        raise KeyError(
                            f"Tag {t!r} not in segment_properties "
                            f"(known: {tags_prop.tags})"
                        )
                    required.add(tags_prop.tags.index(t))
                candidates &= {
                    i
                    for i in candidates
                    if required.issubset(tags_prop.values[i])
                }
            else:
                p = self._property(key)
                assert p.values is not None
                if isinstance(val, dict):
                    lo = val.get("min")
                    hi = val.get("max")
                    candidates &= {
                        i
                        for i in candidates
                        if (lo is None or p.values[i] >= lo)
                        and (hi is None or p.values[i] <= hi)
                    }
                else:
                    candidates &= {
                        i for i in candidates if p.values[i] == val
                    }

        return [self.ids[i] for i in sorted(candidates)]

    def top_n(
        self, property_id: str, n: int, ascending: bool = False
    ) -> list[dict[str, Any]]:
        """Return the top-N segments by a numeric property.

        Returns a list of `{id, value}` so the caller can show both —
        useful for "the 5 largest neurons by size" with sizes attached.
        """
        p = self._property(property_id)
        if p.type != "number":
            raise ValueError(
                f"Property {property_id!r} is type {p.type!r}, not 'number'"
            )
        assert p.values is not None
        order = sorted(
            range(len(self.ids)),
            key=lambda i: p.values[i],
            reverse=not ascending,
        )
        return [
            {"id": self.ids[i], "value": p.values[i]}
            for i in order[: max(0, n)]
        ]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    """Strip `precomputed://` and translate `gs://` to public HTTPS.

    The Python `neuroglancer` package consumes URLs prefixed with
    `precomputed://` (and friends), but for HTTP fetching we want the
    bare scheme. `gs://bucket/path` becomes
    `https://storage.googleapis.com/bucket/path`, which works for the
    public buckets that host hemibrain, FlyWire, MICrONS, and CellMap.
    Private buckets that need auth aren't supported in v0.1 — for those
    callers should fall back to cloud-volume / cloud-files.
    """
    for prefix in ("precomputed://", "n5://", "zarr://", "graphene://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    if url.startswith("gs://"):
        url = "https://storage.googleapis.com/" + url[len("gs://") :]
    return url


def _resolve_info_url(layer_sources: list[str]) -> str:
    """Pick the segment_properties/info URL from a list of layer sources.

    Precedence:
    1. Any source that already ends in `segment_properties` (the
       co-mounted case where the publisher exposes it as a separate
       source in the layer's source list).
    2. Otherwise, default to `<first source>/segment_properties` —
       the convention for precomputed segmentations.
    """
    if not layer_sources:
        raise ValueError("Layer has no sources")
    for s in layer_sources:
        base = _normalize_url(s).rstrip("/")
        if base.endswith("segment_properties"):
            return base + "/info"
    base = _normalize_url(layer_sources[0]).rstrip("/")
    return base + "/segment_properties/info"


def _fetch_json(url: str) -> dict[str, Any]:
    """HTTP-fetch a JSON document. Replaceable for tests."""
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


_cache: dict[str, SegmentProperties] = {}


def load_segment_properties(layer_sources: list[str]) -> SegmentProperties:
    """Resolve, fetch, parse, and cache the segment_properties for a layer."""
    info_url = _resolve_info_url(layer_sources)
    if info_url not in _cache:
        info = _fetch_json(info_url)
        _cache[info_url] = SegmentProperties.from_info(info)
    return _cache[info_url]


def clear_cache() -> None:
    """Drop all cached segment_properties. Tests use this between cases."""
    _cache.clear()
