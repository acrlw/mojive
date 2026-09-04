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
uv run python examples/debug_draw.py
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
