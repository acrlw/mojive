# Debug drawing

Debug primitives are retained by stable string identifiers. Reusing an identifier updates its
storage in place, which keeps per-frame diagnostics compact. Widths, point radii, text offsets,
and line-arrow heads use screen pixels; anchors and segment endpoints use Z-up world coordinates.
The solid `arrow_3d` and `arc_arrow_3d` APIs use world units for every dimension.

Occlusion modes:

- `DEPTH`: normal scene depth testing;
- `ALWAYS`: overlay visible above scene geometry;
- `GHOST`: depth-aware diagnostic overlay.

Debug drawing is composited before the selection outline. The outline remains visible
over diagnostics; transform gizmos and viewport UI appear above both. `ALWAYS` bypasses
scene occlusion, not the visual priority of editor selection and interaction feedback.

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

## Solid straight and circular arrows

`arrow_3d` and `arc_arrow_3d` share the 3D gizmo's cylinder, cone, and shoulder geometry.
They use one opaque color, smooth normals, and depth-tested surfaces. The debug pass does
not cast or receive scene shadows; its view-facing lighting makes the shape readable
without shadow-map artifacts. Reusing an ID replaces the complete arrow.

```python
import math
from mojive import Occlusion

velocity = viewer.backend.debug.layer("velocity", Occlusion.ALWAYS)
velocity.arrow_3d("linear", (0, 0, 1), (1, 0, 1), (0, 1, 0, 1), shaft_radius=0.02)
velocity.arc_arrow_3d(
    "yaw", center=(0, 0, 0.6), normal=(0, 0, 1), start_direction=(1, 0, 0),
    sweep=-math.pi, color=(1, 0, 0, 1), radius=0.5, shaft_radius=0.02,
)
```

Both methods accept optional `head_radius`, `head_length`, and `duration`. The default
head radius is 2.8 times the shaft radius; head length is the smaller of six shaft radii
and 40% of total length. `arc_arrow_3d` measures total length along the arc. Its signed
`sweep` is in radians, with positive rotation following the right-hand rule about `normal`.
`start_direction` is projected onto that plane. Coincident straight endpoints or a zero
sweep erase the arrow. Dimensions must be finite and positive; the head must be shorter
than the arrow and its radius cannot be smaller than the shaft radius. For an arc, the
head radius must also be smaller than the arc radius, and the sweep cannot exceed one turn.

Use `Occlusion.DEPTH` for scene occlusion or `Occlusion.ALWAYS` for foreground annotations
that retain their own depth ordering. These opaque arrows require alpha 1 and do not
support `GHOST`. Each mesh triangle counts toward the debug primitive budget; use the
lighter `arrows` batch API for thousands of velocity vectors.

The passive-viewer bridge accepts the same parameters with `op: "arrow_3d"` or
`op: "arc_arrow_3d"`, plus `layer`, `id`, and `occlusion`. Send these compact commands
through `viewer.publish_debug_commands(...)`; geometry is generated in the viewer.

Run `make debug-draw` to inspect both shapes, or capture two camera angles:

```bash
make debug-draw ARGS='--capture-dir output/debug-draw/opengl'
make debug-draw BACKEND=wgpu ARGS='--capture-dir output/debug-draw/wgpu'
```

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

See [Canvas2D](canvas2d.md) for filled shapes, paths, transforms, batches, and offscreen rendering.

The debug bridge also accepts `op: "triangles"` with `vertices` shaped `[N, 3, 3]`,
a stable `id`, and an optional `color` and `duration`. This exposes `Layer.triangles`
for continuous custom meshes, such as a curved arrow whose shaft and head share
an edge. All vertices are world coordinates; use `op: "clear"` with the same layer
and ID to erase the mesh.
