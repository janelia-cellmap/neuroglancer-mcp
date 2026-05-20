# neuroglancer-mcp

Unofficial [MCP](https://modelcontextprotocol.io) server for Google's [Neuroglancer](https://github.com/google/neuroglancer) volumetric viewer. Lets Claude (or any MCP client) drive a Neuroglancer session: load datasets, navigate, show/hide segments, search by published metadata, style for figures, share snapshot URLs.

This project is not affiliated with Google. Neuroglancer is developed by the Connectomics at Google team.

> **Status:** pre-release, not yet on PyPI. Install from source (below). PyPI publication is gated on stabilizing the tool surface and a courtesy heads-up to the Neuroglancer maintainers.

## Why

Neuroglancer is the standard viewer for petascale connectomics and EM segmentation data (FlyEM hemibrain, MICrONS, FlyWire, H01, CellMap). It has a rich Python API but no agentic interface. This server exposes that API as MCP tools so an LLM can compose viewer operations in response to natural-language requests — e.g., "load the hemibrain and show me three Kenyon cells" becomes a sequence of `add_segmentation_layer` + `show_segments` calls.

## Install from source

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. The lockfile (`uv.lock`) is committed; reproduce the dev environment with:

```bash
git clone https://github.com/janelia-cellmap/neuroglancer-mcp.git
cd neuroglancer-mcp
uv sync                       # creates .venv, installs runtime + dev deps
uv run pytest                 # run the test suite
uv run pytest -m network      # also run live-network tests (hits public GCS buckets)
uv run neuroglancer-mcp       # start the MCP server on stdio
```

There is no PyPI release yet — `pip install neuroglancer-mcp` will not work.

## Use with Claude Desktop / Claude Code

Until PyPI publication, MCP clients launch the server via `uv run` against the cloned project. Substitute `/abs/path/to/neuroglancer-mcp` for your checkout path.

`claude_desktop_config.json` (Claude Desktop):

```json
{
  "mcpServers": {
    "neuroglancer": {
      "command": "uv",
      "args": ["--directory", "/abs/path/to/neuroglancer-mcp", "run", "neuroglancer-mcp"]
    }
  }
}
```

`.mcp.json` at the repo root (Claude Code, project-scoped — already shipped in this repo):

```json
{
  "mcpServers": {
    "neuroglancer": {
      "command": "uv",
      "args": ["run", "neuroglancer-mcp"]
    }
  }
}
```

Reload your MCP client. Ask: *"Load the FlyEM hemibrain dataset and show me a Kenyon cell."*

The first tool call returns a viewer URL. Open it once in your browser; all subsequent calls update that viewer live.

## Tools

**State & introspection**
- `get_state` — current viewer state as JSON
- `get_url` — live viewer URL (only reachable while this MCP server runs)
- `share_url` — snapshot URL on `neuroglancer-demo.appspot.com` that encodes the current state for sharing
- `save_share_url` — write the snapshot URL to a file (HTML or plain text) for when it's too long to inline in chat
- `load_state` — load state from JSON or share URL
- `list_layers` — names, types, visibility

**Navigation**
- `navigate_to` — move to (x, y, z) in nm or voxels
- `set_zoom` — set cross-section scale
- `get_position` — current center position
- `set_orientation` — set the 3D view quaternion
- `center_on_layer` — re-center on a layer's volume midpoint (precomputed + OME-NGFF zarr); useful if you used `center=False` earlier or navigated away
- `center_on_segment` — move the viewer to a single segment's approximate centroid (reads the mesh manifest; supports sharded `multilod_draco` and legacy unsharded meshes)

**Layers**
- `add_image_layer` — add raw EM image layer from precomputed/n5/zarr source; auto-centers on the volume by default (`center=False` to opt out)
- `add_segmentation_layer` — same, for segmentations
- `add_layers` — bulk add (one transaction, one auto-center) for multi-layer datasets like OpenOrganelle volumes
- `remove_layer`, `set_layer_visibility`
- `set_layers_visibility` — bulk-toggle visibility by `{name: bool}` map
- `show_only_layers` — show the named layers, hide everything else (the "isolate this view" shortcut)

**Segments**
- `show_segments` — replace visible segments in a layer
- `add_segments` — add to current selection
- `hide_segments` — remove from selection
- `clear_segments` — hide all

**segment_properties** (search and filter by published metadata)
- `get_segment_properties_schema` — what properties this dataset publishes
- `get_segment_property` — one value for one segment
- `search_segments_by_label` — substring or regex over the label property
- `filter_segments` — combine label glob, tags, and numeric ranges
- `top_n_segments_by_property` — e.g. N largest neurons by precomputed size
- `show_segments_by_property` — filter and display in one round trip

**Styling — colors and opacity**
- `set_segment_colors` — per-segment hex overrides ("make neuron 1234 red")
- `set_layer_color` — default color for a segmentation layer
- `set_layer_opacity` — 0–1 opacity (image opacity, or 3D object_alpha for segs)
- `set_mesh_silhouette` — silhouette-edge intensity for figure-quality meshes

**Styling — view arrangement**
- `arrange_view` — switch among Neuroglancer's nine native layouts (xy, 4panel, 3d, …) or semantic aliases (`quad`, `3d_only`, `top_xy`)
- `set_background_color` — cross-section and/or 3D background (e.g. white for figures)

**Styling — image shaders**
- `set_image_shader` — raw GLSL on an image layer
- `set_image_preset` — `grayscale`, `red`/`green`/`blue`, `fire`, `viridis`, `threshold`
- `set_contrast` — convenience window: maps `[min, max]` to `[0, 1]` grayscale

**Styling — segmentation rendering**
- `set_segmentation_rendering` — `voxel`, `mesh`, or `both`
- `set_mesh_resolution` — mesh LOD threshold (smaller = finer)

## LAN access

The viewer binds to `0.0.0.0` by default, so `get_url` returns a hostname-based URL that other machines on the same LAN can open (typical workflow: MCP runs on a workstation, you open the viewer on your laptop). To restrict to the local machine, set `NEUROGLANCER_MCP_BIND_ADDRESS=127.0.0.1` before launching; you can also pass a specific interface IP for tighter firewalling.

`share_url` is independent of bind address — it always emits a `neuroglancer-demo.appspot.com` snapshot.

Note: the viewer URL contains a random per-session token, so it isn't guessable, but it has no auth on top of that. Don't use the default bind on untrusted networks.

## Design notes

- Every tool returns the resulting state fragment so the agent can verify its action.
- Coordinate units are explicit on every navigation call (nm vs voxels) — the model should never guess.
- The server holds one viewer instance per process. Multi-viewer support is deferred.

## License

BSD 3-Clause. See LICENSE.

## See also

- [neuroglancer](https://github.com/google/neuroglancer) — the upstream viewer
- [napari-mcp](https://github.com/royerlab/napari-mcp) — analogous MCP server for napari
- [cloud-volume](https://github.com/seung-lab/cloud-volume) — Python client for Neuroglancer Precomputed data
