# neuroglancer-mcp — Kickoff Plan

> Open this file at the root of an empty `neuroglancer-mcp/` repo. It's the spec Claude Code should work from.

## What this is

An unofficial Model Context Protocol server for Google's [Neuroglancer](https://github.com/google/neuroglancer) viewer. It exposes viewer state manipulation and presentation tools (styling, colors, shaders, layouts) as MCP tools so Claude or any MCP client can drive a Neuroglancer session conversationally.

This is the **presentation layer**, not the analysis layer. It does not compute properties of segments from raw data. It only changes what the viewer shows, how it shows it, and lets you query *native* Neuroglancer segment_properties that the dataset publisher already shipped. Analytical capabilities (computing volumes, running morphology, querying derived metadata) live in a sibling project, `labelscope-mcp`.

Audience: developers using Claude Desktop or Claude Code who want to drive Neuroglancer with natural language. Also: paper-figure-makers who want conversational styling control.

## Prior art

- `napari-mcp` (royerlab) is the equivalent for napari. Cite it in the README.
- `neuroglancer-scripts`, `panel-neuroglancer`, `pydantic-neuroglancer` follow the `<thing>-<integration>` naming convention.
- No `neuroglancer-mcp` exists today as of May 2026. Confirmed via PyPI and GitHub search.

## What's already drafted

A skeleton sits at `~/neuroglancer-mcp-skeleton/` (the version we sketched earlier). It has:

- `pyproject.toml` with `mcp>=1.0.0`, `neuroglancer>=2.40`, `numpy`, `pillow` deps.
- `src/neuroglancer_mcp/server.py` with 16 tools covering state, navigation, layers, segments, screenshot.
- `src/neuroglancer_mcp/viewer.py` with a lazy-init viewer singleton + nm↔voxel converter.
- `src/neuroglancer_mcp/__main__.py` with the stdio entry point.

That skeleton is a good starting point but it's **missing the entire presentation tool surface** (colors, shaders, opacity, view arrangement), which based on our design discussion is the actual product thesis. Phase 2 below adds that.

## Reference code in Tourguide

`server/ng.py` in the Tourguide repo has the `NG_StateTracker` class. Useful primitives to lift or learn from:

- `get_url()` — returns the viewer URL.
- `summarize_state(state)` — extracts a clean state summary.
- `get_available_layers()` — lists layer names and types.
- `get_voxel_size()` — extracts per-axis scales from `dimensions`.

The screenshot loop in `NG_StateTracker.start_screenshot_loop()` is overkill for v0.1 — we only need a single on-demand `screenshot()` tool, not a continuous streaming loop. The streaming is Tourguide's narration feature, not something this MCP exposes.

## Phase 1 — minimal viable

Goal: shippable v0.1 that lets Claude Desktop drive a Neuroglancer viewer.

### Files to create

```
neuroglancer-mcp/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── tests/
│   ├── __init__.py
│   ├── test_viewer.py
│   └── test_tools.py
└── src/neuroglancer_mcp/
    ├── __init__.py
    ├── __main__.py        # entry point → mcp.run() over stdio
    ├── server.py          # FastMCP server, tool definitions
    ├── viewer.py          # singleton viewer + coord helpers
    └── tools/
        ├── __init__.py
        ├── state.py       # get_state, load_state, list_layers, get_url
        ├── navigation.py  # navigate_to, set_zoom, get_position, set_orientation
        ├── layers.py      # add_image_layer, add_segmentation_layer, remove_layer
        ├── segments.py    # show/add/hide/clear_segments, set_segment_colors
        └── capture.py     # screenshot
```

### Tool list (Phase 1)

Same as the skeleton currently has, plus a few I'd add:

**State:**
- `get_state()` → dict
- `get_url()` → {"url": str}
- `load_state(state: dict)` → dict
- `list_layers()` → list[{name, type, visible, source}]

**Navigation:**
- `navigate_to(x, y, z, units: Literal["nm","voxels"])` → {"position": [...]}
- `set_zoom(scale: float)` → {"cross_section_scale": float}
- `get_position()` → {"position": [...]}
- `set_orientation(quaternion: list[float])` → {"orientation": [...]}

**Layers:**
- `add_image_layer(name, source)` → dict
- `add_segmentation_layer(name, source)` → dict
- `remove_layer(name)` → dict
- `set_layer_visibility(name, visible)` → dict

**Segments:**
- `show_segments(layer, segment_ids)` → dict
- `add_segments(layer, segment_ids)` → dict
- `hide_segments(layer, segment_ids)` → dict
- `clear_segments(layer)` → dict

**Capture:**
- `screenshot(size: list[int] | None)` → {"image_base64": str, "mime_type": "image/png"}

### Critical design rules (do not violate)

1. **Units are never implicit.** `navigate_to` REQUIRES the `units` parameter — no default. The nm↔voxel mistake is the single most common Neuroglancer error humans make; don't let the agent reproduce it.
2. **Every tool returns a state summary.** No `None`, no `"ok"`. The agent needs observability after each call.
3. **Lazy viewer init.** Construct the `neuroglancer.Viewer` only on first tool call that actually needs it. The MCP server should start in milliseconds.
4. **Docstrings are the API contract.** Claude reads docstrings to decide which tool to call. Be explicit about units, side effects, and return values. Spend more time on docstrings than on implementation.
5. **No code execution.** This MCP never runs arbitrary Python. That's `labelscope-mcp`'s job.
6. **No analytical computation.** This MCP doesn't compute segment volumes/centroids/morphology from raw data. It only reads what Neuroglancer can already see (state + native segment_properties). Anything requiring scipy/numpy compute against arrays is `labelscope-mcp`'s job.

### Done when

- `pip install -e .` succeeds.
- `neuroglancer-mcp` command starts the server on stdio.
- Adding it to Claude Desktop's config and asking "Load the FlyEM hemibrain dataset and show neuron 5901222347" produces the right viewer state.
- Tests pass for: coord conversion, state round-trip, layer add/remove, segment selection.

## Phase 2 — segment_properties access (the workhorse)

Most major Neuroglancer datasets ship with `segment_properties` — a sidecar that maps segment IDs to human-readable names, descriptions, numeric properties (volume, length, synapse counts), and tags (status, hemilineage, side). Hemibrain has them. FlyWire has them. MICrONS has them. CellMap has them in some sources.

Neuroglancer's UI exposes these via a search box and a filter panel — but the Python API doesn't make it easy to query them programmatically, and there's no current way to add segments by name (see issue #398 in the upstream repo).

This phase fills that gap. It's high-value because it's how biologists actually want to interact: "show all Kenyon cells", not "look up Kenyon cell IDs then call show_segments."

### The segment_properties format

A precomputed segmentation may have a `segment_properties` subdirectory with an `info` file like:

```json
{
  "@type": "neuroglancer_segment_properties",
  "inline": {
    "ids": ["123", "456", "789"],
    "properties": [
      {"id": "label", "type": "label", "values": ["KC", "PEN_a", "MBON14"]},
      {"id": "description", "type": "description", "values": [...]},
      {"id": "size", "type": "number", "data_type": "uint64", "values": [12345, 67890, 11111]},
      {"id": "tags", "type": "tags",
       "tags": ["Traced", "Leaves", "KC_class", "PEN_class"],
       "values": [[0,2], [0,3], [0]]}
    ]
  }
}
```

The four property types are `label`, `description`, `number`, and `tags`. The viewer URL for using these is `precomputed://.../segmentation` with a sibling source `precomputed://.../segmentation/segment_properties` — or the layer source is a list of both.

### Tools to add

**Introspection:**
- `get_segment_properties_schema(layer)` → `{properties: [{id, type, description?, num_unique?}], num_segments}`. Returns what's available without loading all values.
- `get_segment_property(layer, segment_id, property_id)` → the value for one segment.

**Filtering and search:**
- `search_segments_by_label(layer, query, regex=False)` → list of `{id, label}`. Substring match by default, regex if `regex=True`. The most useful tool for "give me all KCs" style queries.
- `filter_segments(layer, criteria: dict)` → list of segment IDs matching criteria. Example: `{"label": "KC*", "tags": ["Traced"], "size": {"min": 1000}}`.
- `top_n_segments_by_property(layer, property_id, n, ascending=False)` → top-N segments by a numeric property. Example: largest 10 neurons by precomputed size.

**Convenience (filter + show in one call):**
- `show_segments_by_property(layer, criteria)` → calls `filter_segments` then `show_segments` with the result. The single tool that handles "show all Kenyon cells where status is Traced."

### Implementation notes

- Use `cloud-volume` to fetch the segment_properties info file. It already handles precomputed sources, GCS auth, caching.
- For large segment_properties (FlyWire has 140K+ entries), consider lazy loading: only parse the property values you're querying against.
- `nglui` from CAVE has some segment_properties helpers worth borrowing from.
- The MCP layer name for the source can be either a single URL (`precomputed://.../segmentation`) or a list. The MCP needs to handle both — segment_properties may be specified as a separate co-mounted source.

### Why this is uniquely valuable

The Python `neuroglancer` package does NOT expose segment search by label. You have to know IDs already. This phase gives Claude a clean way to do what humans do via the UI search box but currently can't do programmatically. That's the actual community contribution beyond "wrapper around an existing API."

### Done when

- All Phase 1 tools still work.
- "Load the hemibrain and show me all the Kenyon cells" works end-to-end.
- "Show the 5 largest neurons by size" works (using hemibrain's precomputed size property).
- `search_segments_by_label` correctly handles substring and regex matches against hemibrain's ~25K neuron labels.

## Phase 3 — presentation/styling (the differentiated thesis)

This is where the MCP becomes uniquely useful beyond "thin wrapper around Python API." Conversational styling control over Neuroglancer is something nobody offers and most users would benefit from.

### Tools to add

**Colors and opacity:**
- `set_segment_colors(layer, color_map: dict[int, str])` — per-segment color overrides. Hex codes.
- `set_layer_color(layer, color)` — default color for segments without explicit overrides.
- `set_layer_opacity(layer, alpha: float)` — 0.0 to 1.0.
- `set_mesh_silhouette(layer, value: float)` — Neuroglancer's silhouette intensity, useful for clean mesh rendering.

**Shaders (image layers):**
- `set_image_shader(layer, shader_glsl: str)` — raw GLSL.
- `set_image_preset(layer, preset: Literal["grayscale","red","green","blue","fire","viridis","threshold"], **params)` — named presets that generate GLSL. The "threshold" preset takes a `value: float` param.
- `set_contrast(layer, min: float, max: float)` — convenience shader that maps `[min,max]` to `[0,1]`.

**View arrangement:**
- `set_cross_section_visibility(visible: bool)` — toggle the 3 cross-section panels.
- `set_3d_visibility(visible: bool)` — toggle the 3D panel.
- `arrange_view(preset: Literal["3d_only","cross_section_only","quad","top_xy"])` — preset panel layouts.
- `set_background_color(color: str)` — for figure-making.

**Segmentation rendering:**
- `set_segmentation_rendering(layer, mode: Literal["voxel","mesh","both"])` — control whether segments render as voxels, meshes, or both.
- `set_mesh_resolution(layer, resolution: int)` — mesh LOD.

### Why this matters

Setting up a publication-quality Neuroglancer view is currently 15 clicks across three menus. Conversational adjustment ("a bit more contrast on the EM", "make neuron 1234 bright red", "hide the cross sections") is a much better workflow for figure-making and educational demos.

For the writeup/application story, lead with this. Phase 1 is table stakes (anyone could build it). Phase 2 is the differentiated thesis.

### Done when

- All Phase 1 and Phase 2 tools still work.
- "Make the membrane layer red and 50% transparent, the mitos solid green, hide everything else" produces the right view.
- README's "tour" section shows three or four worked styling examples.
- One screenshot-or-GIF demo in the README of conversational styling working end-to-end.

## Phase 4 — annotations and meshes (optional, v0.2+)

Defer unless there's demand.

- `add_annotation_layer(name, annotations: list[dict])` — points, lines, ellipsoids, boxes.
- `add_point_annotation(layer, position, units, description)`.
- `mesh_quality(layer, quality: float)` — mesh LOD tuning.

Skip in v0.1 because the annotation schema is rich and deserves its own design pass.

## Community etiquette

Before publishing v0.1 to PyPI:

1. Post to the Neuroglancer Google Group (https://groups.google.com/forum/#!forum/neuroglancer): "FYI, building an MCP server for Neuroglancer. Here's the tool surface I'm proposing. Feedback welcome before I publish."
2. Email Jeremy Maitin-Shepard separately as a courtesy.
3. README's first line: "Unofficial MCP server for Google's Neuroglancer viewer." Don't claim official status.
4. License: BSD 3-Clause (matches Tourguide; permissive).
5. Link prominently to `github.com/google/neuroglancer`.

## What this MCP explicitly does NOT do

- Compute anything. No segment properties, no statistics, no measurements.
- Query metadata. No SQL, no segment_properties.json parsing.
- Run user code. No sandbox, no Python execution.
- Load datasets from CAVE/FlyWire materialization tables. That's a separate MCP.
- Generate URLs without a live viewer. (URL-only mode is a future extension; v0.1 always uses a live viewer.)

Anything not in the tool list above is intentionally out of scope. Keep the surface tight.

## Open questions for first session with Claude Code

1. Should the GLSL preset library live in code (a Python dict of templates) or in a separate YAML/JSON file? Latter is more extensible; former is simpler.
2. Where to put the per-segment color hash function (Neuroglancer's default coloring)? Probably borrow from `neuroglancer` package itself rather than reimplement.
3. Should `screenshot()` block until chunks finish loading (high quality, slow) or return what's currently rendered (fast, possibly blurry)? Default to block-and-wait; expose a `wait_for_chunks: bool = True` parameter.
4. Test strategy: unit tests against a mock viewer? Or integration tests that spin up a real `neuroglancer.Viewer()`? Probably both — unit for coord math, integration for the viewer interactions.
