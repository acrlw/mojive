# Draw with Canvas2D

`viewer.canvas2d` and `SceneRenderer.canvas2d` expose the same retained drawing API.
Canvas coordinates map to a world plane (XY by default). Canvas2D submits to the existing
DebugDraw GPU pass on OpenGL, WebGPU, and bgfx. It does not use ImGui or allocate a second
renderer. UI controls continue to use ImGui draw lists.

## Author shapes

```python
from mojive import Scene, SceneRenderer
from mojive.canvas2d import Affine2D, PathBuilder2D

scene = Scene()
with SceneRenderer(scene.source, width=640, height=480) as renderer:
    canvas = renderer.canvas2d
    shapes = canvas.layer("shapes")
    shapes.rectangle("panel", (-2, -1), (2, 1), (0.2, 0.5, 0.8, 0.5), filled=True)
    shapes.ellipse("outline", (0, 0), (1.5, 0.7), (1, 1, 1, 1), width_px=2)
    path = PathBuilder2D().move_to(-1, 0).cubic_to(-1, 1, 1, -1, 1, 0).finish()
    with shapes.transformed(Affine2D.translation(0, 0.2)):
        shapes.stroke_path("curve", path, (1, 0.6, 0.2, 1), width=0.08, cap="round")
    renderer.update(scene.frame, camera=canvas.camera((-3, -2, 3, 2), aspect=640/480))
    pixels = renderer.render()
```

Every drawing call creates or replaces one `ident` within its layer. Submit static geometry once;
it remains visible without per-frame Python drawing calls. Updating an ID reuses its storage when
its primitive count is unchanged. `erase(ident)`, `layer.clear()`, `canvas.clear()`, `visible`, and
`duration` use existing debug-layer lifetime semantics. The default duration is retained indefinitely.
A zero-sized triangle batch or empty fill path removes the old ID.

## Interface and units

| Methods | Coordinates and behavior |
|---|---|
| `line`, `lines`, `polyline`, `polygon` | Canvas positions; `width_px` stays constant on screen |
| `rectangle`, `circle`, `ellipse` | Outline by default; `filled=True` creates a filled shape |
| `rounded_rectangle` | Radius and sampling tolerance in canvas units; optional fill |
| `arc` | Elliptical radii; start, signed sweep and rotation in radians |
| `bezier` | Three control points for quadratic, four for cubic; tolerance in canvas units |
| `arrow`, `point`, `points` | Canvas positions, screen-pixel widths/radii |
| `text` | World-anchored label using the existing debug text atlas; screen-pixel offset |
| `grid` | One batched grid in canvas units |
| `triangles` | `[N, 3, 2]` batch; one RGB(A) or `[N, 4]` colors |
| `fill_path` | Concave, multi-contour and holed shapes; nonzero or evenodd fill rule |
| `stroke_path` | Width in **canvas units**, cap `butt/round/square`, join `miter/round/bevel` |
| `transformed(Affine2D)` | Nested context affecting new submissions; restores on exceptions |

`PathBuilder2D` supports `move_to`, `line_to`, `quadratic_to`, `cubic_to`, elliptical `arc`,
`close`, and `finish`. `finish()` returns an immutable snapshot suitable for reuse. Open contours
are implicitly closed when filled. Choose `PathBuilder2D(fill_rule="evenodd")` for nested holes
without winding-direction management.

`Affine2D.translation(...) @ Affine2D.scale(...)` scales first, then translates. Transform contexts
do not move previously submitted objects. Screen widths and label sizes stay legible; local
`stroke_path` widths transform with the shape. For a differently oriented world plane, construct
`Canvas2D(debug_draw, origin=..., x_axis=..., y_axis=...)`. `world`, `canvas_point`,
`canvas_to_screen`, `screen_to_canvas`, and `camera` provide coordinate conversion and framing.

## Preparation, rendering and limits

Path sampling and tessellation are CPU-only `geometry2d` operations. Filled/stroked paths use a
per-canvas cache capped at 128 entries and 8 MiB (including an allowance for authored commands).
Color and placement do not invalidate it. Oversized results are submitted without being cached.
Each compilation has explicit vertex, index and scratch-memory budgets and fails rather than
silently degrading geometry. Build native geometry support with `make cpp-python` or `make setup`.
The existing line/point/text API does not require path compilation.

Use `lines`, `points`, and `triangles` to submit arrays in a single call. DebugDraw packs compatible
primitives into shared, reusable GPU streams. Alpha is applied once to each tessellated fill or
stroke union; overlapping separately authored objects still blend independently. Filled triangles
use the renderer's multisampling setting for edge antialiasing.

This is a retained diagnostic canvas, with the existing debug pass's ordering: fills before
outlines, then points and labels within an occlusion group. It is not an HTML Canvas painter-order
compositor. Layer depth controls world placement and occlusion; layer names do not introduce
isolated alpha groups. Image brushes, clipping stacks, compositing modes and independently styled
font runs are outside this API. Use ImGui for UI panels and image widgets.

## Verify

```bash
make canvas-2d-gallery BACKEND=opengl
make canvas-2d-test BACKEND=opengl
# Repeat with BACKEND=wgpu and BACKEND=bgfx.
make canvas-2d
```

Gallery captures are written to `output/canvas2d/`. Pixel checks cover hole preservation,
transparent fill seams, hidden layers and empty-ID replacement. CPU checks are in
`tests/test_canvas2d.py`; path/stroke numeric checks are in `tests/native/`.
