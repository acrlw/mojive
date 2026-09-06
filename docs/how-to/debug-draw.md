# Debug drawing

Debug primitives are retained by stable string identifiers. Reusing an identifier updates its
storage in place, which keeps per-frame diagnostics compact. Widths, point radii, text offsets,
and arrow heads use screen pixels; anchors and segment endpoints use Z-up world coordinates.

Occlusion modes:

- `DEPTH`: normal scene depth testing;
- `ALWAYS`: overlay visible above scene geometry;
- `GHOST`: depth-aware diagnostic overlay.

## Interactive example

```bash
uv run --no-sync python examples/debug_draw.py
```

```python
--8<-- "examples/debug_draw.py"
```

Use finite `duration` values for transient contacts or events. The default retains a primitive
until `erase()`, `Layer.clear()`, or `DebugDraw.clear()` removes it.

Use the singular methods (`line`, `arrow`, `point`, and `frame`) for a small number of primitives
that need independent retained IDs, expiration, or erasure. Use their plural batch forms
(`lines`, `arrows`, `points`, and `frames`) for homogeneous high-cardinality diagnostics. A batch
has one retained ID and replaces all of its records together, avoiding thousands of Python calls,
transport dictionaries, and retained-index entries.

```python
starts = robot_positions
ends = starts + robot_velocities * 0.1
depth.arrows("robot-velocities", starts, ends, (0.2, 0.7, 1.0, 1.0), 2.0)
```

Do not split a batch merely to assign numeric IDs to records. Split only where independent
lifetime or erasure semantics are required.

## UI-style screen arrows

`Layer.arrow_2d` uses the same G3 silhouette as the flat Transform and Joint gizmos.
Coordinates are render-viewport pixels with the origin at the top left. Use an `ALWAYS`
layer; these arrows are screen overlays and have no world depth. `Draw2D.arrow` accepts matching
shape dimensions without the `_px` suffix, in logical window coordinates. Its adapter controls
AA, and `smoothing=None` selects the adapter default. Debug `arrow_2d` has an explicit `antialias`
option and a numeric smoothing value. Existing `arrow` and `arrows` methods retain
their world-anchor API and lightweight GPU line-arrow path.

```python
from mojive.render.debugdraw import Occlusion

overlay = viewer.backend.debug.layer("ui-arrows", Occlusion.ALWAYS)
overlay.arrow_2d(
    "direction", (40, 60), (240, 60), (1, 1, 1, 1), 4,
    head_length_px=18, head_width_px=20,
    corner_radius_px=1.5, smoothing=0.6,
)
```

The head, concave shoulders, and shaft form one continuous outline. Radius zero selects
sharp corners. Positive radius with smoothing zero selects ordinary circular fillets.
`join_radius_px` controls the two shoulders independently. It defaults to half the head
corner radius; set it to zero for sharp shoulders. The UI equivalent is `join_radius`.
The smaller shoulder radius keeps the head more rounded than its connection to the shaft.
Positive smoothing up to one enables G3 joins. `round_tail` controls the rear cap and
`antialias` controls the external fringe. The local outline and triangle fan are cached;
changing position only transforms the cached mesh. Each triangle counts toward the debug
primitive budget, so retain the lighter world-arrow batch for high-cardinality diagnostics.
The local bridge accepts the same options with `op: "arrow_2d"` and `occlusion: "always"`.
Dimensions and endpoints must be finite, dimensions must be nonnegative, and smoothing must
be in 0–1. A zero-length head draws a plain shaft. Zero width or coincident endpoints erase
the retained arrow. Invalid input raises `ValueError` before replacing existing geometry.
The executable example above draws all three corner styles.

For custom UI widgets, follow [the drawing extension guide](ui-drawing.md) for ownership,
coordinates, cache placement, and the matching verification paths.

## 2D physics and geometry diagnostics

`Canvas2D` maps ordinary `(x, y)` coordinates onto a configurable world plane while retaining
the same GPU-batched debug storage. It is a diagnostic canvas, not a sprite or game renderer:
its strengths are grids, lines, arrows, points, labels, circles, rectangles, and polygon
outlines that remain crisp while zooming.

```python
canvas = viewer.canvas2d
grid = canvas.layer("grid", depth=-0.02)
grid.grid("world", (-5, -3, 5, 3), spacing=0.5)

contacts = canvas.layer("contacts")
contacts.circle("body", (0, 0), 0.5, (0.3, 0.8, 1.0, 1.0), 2.5)
contacts.points("manifold", contact_points, (1.0, 0.3, 0.2, 1.0), 6.0)
viewer.sync()  # resolve the docked viewport once
width, height = viewer.viewport_size
viewer.set_camera(canvas.camera((-5, -3, 5, 3), aspect=width / height))
```

Layer names and primitive IDs are stable, so updating a physics frame replaces buffers in
place. Set `layer.visible = False` to hide a logical layer without destroying its retained
contents. Use plural methods for contact manifolds, broad-phase pairs, and other large batches.
The default `ALWAYS` occlusion keeps 2D diagnostics visible; pass `Occlusion.DEPTH` when the
canvas should participate in scene depth.
`canvas.screen_to_canvas(...)` ray-casts a viewport pixel onto the canvas plane for picking and
dragging; `canvas.canvas_to_screen(...)` provides the inverse projection for custom overlays.

Run `make canvas-2d` for visual acceptance. The complete program is
`examples/canvas2d.py` in the repository root.
