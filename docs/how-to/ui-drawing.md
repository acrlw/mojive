# Extend UI drawing

Start with the drawing surface and its state owner. UI widgets receive a `Draw2D`; retained
diagnostics use a debug `Layer`. Both consume the shared geometry without importing each other.
For example commands and the pixel-coordinate debug API, see [Debug drawing](debug-draw.md).
For radius, smoothing, and native installation, see [G3 UI corners](ui-corners.md).

## Choose the entry point

| Work | Entry point | Responsibility |
| --- | --- | --- |
| Draw a widget, icon, or viewport overlay | `mojive.ui.draw2d.Draw2D` | Place shapes, choose colors and drawing order |
| Implement ImGui submission | `ImguiDraw2D` in the same module | Native bindings, AA, vertex buffers, font access |
| Add reusable corners, arrows, or stroke caps | `mojive.curves2d` | Pure geometry, sampling, local shape caches |
| Change hollow-origin connectors | `mojive.draglink2d` | Shared implicit field and indexed CPU mesh |
| Publish retained diagnostics | `mojive.render.debugdraw.Layer` | IDs, lifetime, primitive budgets and packing |
| Implement a new GPU primitive | `render/opengl/passes/debug.py` and `render/webgpu/passes/debug.py` | Matching packed layout and shaders |
| Change gizmo interaction or hit regions | `mojive.ui.gizmo` and `mojive.gizmo` | Interaction state and projected handles |
| Change native ImGui colors, radii, or spacing | `mojive.ui.theme` | Standard ImGui style settings |
| Maintain bulk drawing bindings or slider/focus fixes | `tools/build_imgui.py` | Minimal source patch and platform wheel build |

`curves2d` and `draglink2d` import neither ImGui nor a render backend, UI controller, or physics
adapter. Layering tests enforce that boundary. `Draw2D` implementations submit geometry; reusable
geometry belongs in the shared modules. A widget does not need to implement triangle packing or
hold backend objects. Add a specialized helper only when the existing primitive cannot express
its shape. Avoid copying a production glyph into an example or probe.

## Draw a UI overlay

Define placement against the protocol. Construct `ImguiDraw2D` only in an active ImGui frame,
at the panel or viewport drawing boundary, and pass it to the helper.

```python
from mojive.curves2d import smooth_capsule_points
from mojive.ui.draw2d import Draw2D


def draw_direction_badge(draw: Draw2D, origin, scale: float = 1.0, smoothing: float = 0.6):
    x, y = origin
    shell = smooth_capsule_points(x, y, 120 * scale, 32 * scale, smoothing)
    draw.convex_fill(shell, (0.16, 0.18, 0.20, 1.0))
    draw.arrow(
        (x + 20 * scale, y + 16 * scale),
        (x + 100 * scale, y + 16 * scale),
        (0.86, 0.88, 0.90, 1.0),
        2 * scale,
        head_length=7 * scale,
        head_width=8 * scale,
        corner_radius=0.5 * scale,
        smoothing=smoothing,
    )
```

UI positions and widths use logical window coordinates. Apply the widget's scale to authored
dimensions once. Debug `Layer.arrow_2d` uses physical pixels relative to the render viewport.
Convert from a UI point by subtracting the viewport's logical origin and multiplying by the
per-axis framebuffer scale. World-anchor debug arrows and `Canvas2D` use world coordinates;
their stroke widths remain physical pixels. These coordinate systems are not interchangeable.

Radius controls corner size. Smoothing controls how curvature changes within that corner.
For an arrow, radius zero is sharp and positive radius with smoothing zero is circular.
Positive smoothing selects G3 reference transitions. `Draw2D.arrow` has no per-call `antialias`
option; that is owned by the adapter. `Layer.arrow_2d` exposes it explicitly. A custom adapter
can record protocol calls for tests or render them to another surface without creating ImGui.

## Reuse geometry and retain clear ownership

Use `panels.padded_selectable` for standard list rows and native `imgui.menu_item` for menus.
The list helper retains native selection, keyboard navigation, IDs, and highlighting while giving
text explicit insets and equal first/middle/last row bounds. Custom G3 geometry belongs to `Draw2D`;
it does not change global ImGui corners, item bounds, or clipping behavior.

The default ImGui radius is 4.8 logical pixels. Spacing is independent of corner radius:

| Surface | Horizontal inset | Vertical inset |
| --- | --- | --- |
| Panel content | 10 | 10 |
| Native control text | 10 | 4 |
| Custom list rows | 10 | 5 |
| Menu rows | 10 | 4 |
| Viewport content | 0 | 0 |

The default gap between controls is 10 horizontally and 8 vertically. Keep the 4.8 px radius
independent of these spacing values.
Top-level menu labels use a separate compact horizontal gap; popup row padding must not
expand the menu bar. Mutually exclusive segments measure each label and icon, then stack
when their full padded widths cannot fit. Settings toggle groups use measured native table
columns. Checkbox labels share the hit region and support Tab/Space navigation; disabled
controls apply the same alpha to their text, fill, border, and checkmark.

Use `ROW_PADDING_X` and `ROW_PADDING_Y` from `theme` for custom list rows. Rows keep their
full-width hover background while text and disclosure icons sit inside it. Custom text in
one row uses `draw2d.text_line_y`, which centers a shared cap-height reference: descenders
must not move individual words off the baseline. Status keys and telemetry use the same
keycap drawing and `keycap_rounding` proportions.

Use `panels.search_input` for search fields. It reserves trailing icon slots outside the
native text editor, so long text and the caret cannot collide with Search or Clear. Toolbar
actions use standard buttons; `small_button` deliberately removes vertical text padding.

Use native tables for property labels and fields, and `align_text_to_frame_padding` when a
text label sits beside a framed control. Axis badges override horizontal frame padding so their
fixed width can center the letter; the attached numeric field keeps normal text padding.
Responsive vector rows measure their label and minimum field widths, place labels above the
fields when necessary, and stack axes vertically at smaller widths. Let native content height
and the parent scrollbar follow that reflow; do not estimate a fixed child height from font size.
Property sections use one collapsing-header style and the panel surface beneath it.
When a label moves above its fields, give it its own table row without frame alignment.
Putting a nested table below text in the same cell adds unrelated item spacing and makes
compact Transform rows look different from adjacent property tables.

Docked Viewport images fill the content edge. Floating Viewports retain the native one-pixel
resize border outside the image clip, so the image cannot cover resize feedback. Overlay
margins belong to the overlay and do not change either panel layout.
Dock splitters and native resize controls own their complete press before scene routing.
Each window records popup ownership before ImGui starts the frame: Escape and outside-click
dismissal remain UI events even after the popup closes. A held UI press stays with UI until
release. Keyframes uses the wheel for zoom and right-button drag for pan, matching its status
glyphs; channel names wrap within their own clipped column.

Use `smooth_rect_points` for rectangles and `smooth_capsule_points` for fitted capsule ends.
`arrow_points` returns one head-and-shaft outline. `capped_polyline_points` gives a stroke with
independent start/end caps. `box_handle_points` joins a shaft to an oriented square. Arrows and
box handles accept `join_radius` independently of the head's `corner_radius`, defaulting
to half that radius for subtler shoulders. Stroke
caps match the endpoint segments, not an arbitrary globally
smooth centerline. `smooth_ellipse_stroke` handles analytic ellipse offsets where those remain
regular. Curves are sampled within a chord tolerance before rasterization.
`arc_ribbon_points` preserves radial cuts for sampled rotation arcs and caches the local ribbon.
Use `arc_ribbon_mesh` with `indexed_fill` when drawing it; strip and cap indices avoid general
concave triangulation for a regular, non-self-intersecting arc stroke. Endpoint caps use the available arc length, so denser centerline samples
do not trigger repeated curvature fitting. These helpers do not mutate supplied radial vectors.
`arrow_mesh` returns immutable local vertices, triangle indices, and the external contour.
`ImguiDraw2D.arrow` and retained screen arrows share this mesh. Its fan uses a point at the
neck inside the arrow's visibility kernel, so tessellation is linear and cached by dimensions.
Movement reuses both the mesh and its AA preparation. Arbitrary concave contours still require
general triangulation; their first vertex or centroid is not necessarily a valid fan origin.
`Draw2D` and the shared geometry can be imported
without loading ImGui, a window library, or a graphics backend.

Cache immutable local geometry by dimensions, radius, smoothing, and tolerance. Translation,
rotation, color, and hover state should not rebuild a curvature profile. Keep caches bounded;
do not use an unlimited cache keyed by cursor coordinates. Existing helpers already cache the
expensive local shapes. Measure cold construction and changing shapes as well as warm lookups.

Use `convex_fill` for convex contours. A `triangle_fan_fill` contour must be visible from its
first vertex, as with an ordered circular sector. General concave contours use `concave_fill`
or `fringed_concave_fill`. Large or repeatedly drawn shapes with known topology can use
`indexed_fill` with a flat index sequence grouped in threes. Supply the outer contour and hole
contour for external AA; never add fringes along internal triangle edges. The adapter owns
ImGui reservation and vertex-offset handling.

For a moving or rotating mesh, keep vertices and contours in local coordinates and pass
`origin` plus a unit `direction` to `indexed_fill`. Direction is the local X axis in window
coordinates. For example:

```python
draw.indexed_fill(
    vertices, indices, color, outline=outline, hole=hole,
    origin=(200.0, 150.0), direction=(0.6, 0.8),
)
```

The ImGui adapter caches local native vertices and AA contours, then rotates and translates
only the newly submitted range with one native call. This keeps movement out of the geometry
and fringe cache keys. Other adapters implement the same rigid placement as
`x = origin_x + local_x * direction_x - local_y * direction_y` and
`y = origin_y + local_x * direction_y + local_y * direction_x` for every mesh and contour point.
Omitting `origin` preserves the original window-coordinate behavior. Scaling belongs in the
authored dimensions so the AA fringe remains one pixel wide.

The CPU hollow connector belongs to `draglink2d.smooth_drag_link_mesh`. UI code calls
`draw2d.draw_drag_link` for its two colors and placement. Normal Transform and perturbation
dragging publish the corresponding GPU field primitive. Update both shader fields when changing
the implicit blend, then compare CPU reference boundaries and both GPU backends. Do not replace
a one-quad path with a dense per-frame CPU solve solely to share its submission mechanism.

Visual styling does not automatically change hit regions, Session state, or adapter write-back.
Keep those decisions in their current owners. Preview values in UI Feasibility are temporary
and copyable; they are not silently persisted as application settings.

Use `gizmo.screen_path_distance(point, points, closed=False)` for screen-space path picking and
`gizmo.screen_polygon_distance(point, polygon)` for a filled simple polygon. Both return distances
in the coordinates supplied by the caller; keep pixels and logical UI points consistent with the
hit padding. Open paths do not acquire a closing edge. The shared implementation handles collapsed
segments and evaluates distances in one NumPy batch, independent of UI submission or curve sampling.
Transform and joint gizmos use this same implementation.

## Find verification and examples

| Change | Focused evidence |
| --- | --- |
| Reference profile, bounds, sampling, or cache | `python/tests/test_curves2d.py` |
| Implicit connector, hole topology, or tessellation | `python/tests/test_draglink2d.py` |
| Submission, AA, winding, or vertex offsets | `python/tests/test_draw2d.py` |
| Gizmo appearance and interaction | `python/tests/test_gizmo.py`, `python/tests/gpu/test_gizmo.py`, `make gizmo-gallery` |
| Radius/smoothing controls | `python/tests/gpu/test_ui_corner_controls.py`, `make ui-corners-gallery` |
| Retained primitives and bridge | `python/tests/test_debugdraw.py`, `python/tests/gpu/test_debugdraw.py`, `examples/debug_draw.py` |
| Dependency boundaries | `python/tests/test_layering.py` |

Finish with the applicable gates in the [verification matrix](../guides/testing.md#change-mapping).
Use `make g3-benchmark` for CPU shape/submission cases and `make ui-frame-profile` for an actual
viewer. `make g3-benchmark ARGS="--profile"` also saves a call profile after the timed batches;
the profiler does not affect the reported microbenchmarks. Static cache timings are not complete draw costs. Inspect images yourself and keep
temporary captures under `output/`. `make ui-corners`, `make ui-corners-gallery`, and `make g3-ui`
use production geometry rather than independent replicas.

`make interaction-benchmark ARGS="--profile"` measures cold overlapping connectors, batch projection,
transform and joint hit tests, and scene/selected-node bounds at 100, 1,000, and 10,000 instances.
It uses fixed seeds, warmed repeated batches, and a separate untimed call profile. The connector
workload cycles through more shapes than its cache can hold, so it measures rebuilding rather than
cache hits. Raw samples and host versions are saved under `output/interaction-performance.json`.
Run a saved source baseline with the same harness using `PYTHONPATH` to compare implementations;
local CPU timings do not establish whole-frame speedups.
Use `make ui-frame-profile ARGS="--hover-gizmo --gizmo-mode rotate"` to include a scripted
pointer sweep over the selected gizmo. The report records whether hit testing actually ran;
pointer events stay inside the benchmark viewer and do not move the operating system cursor.
