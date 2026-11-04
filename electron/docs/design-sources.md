# Design and viewport provenance

The source of truth remains Mojive's adopted design specification. `scripts/sync-design.mjs`
imports `python/design/resources/design-spec.json`, the semantic SVG mapping in
`examples/ui_design/native/svg/sources.json`, and bundled MiSans resources. It generates
CSS variables, trusted SVG markup, standalone SVG files, and font resources with licenses.
It does not update the native specification. Upstream Lucide/Tabler strokes receive the
adopted 1.5-unit family width; Mojive masters keep their authored fills and explicit widths.
JetBrains Mono Regular is bundled from the official v2.304 distribution with its SIL OFL.

Electron controls currently use ordinary circular corners, with nested radii derived from
their insets. The retained Figma squircle utilities do not change this default.

## Orientation and shared gizmo geometry

`src/viewport/viewcube.ts` ports `python/ui/viewcube.py`, `WorkspaceViewCube`'s 90-point sizing,
and the lollipop geometry in `python/geometry2d/curves.py`. It preserves:

- Orthographic axis layout, depth sorting, separate positive/negative faces and label opacity.
- The native 0.72–0.96 rear-axis fade, equal endpoint radii, origin clearance and hit ownership.
- Seventh-order G3 necks, eight-point curvature integration, short-neck blending, and round-cap fitting.
- Native antialias geometry: lollipop outward fringes, centered circle fringes, inset circle-stroke
  centerlines, and the origin's narrower border fringe. A small WebGL2 pass submits solid triangles
  and alpha gradients; browser path rasterization does not replace these shapes.
- A 4-point click threshold, capture through release/cancellation, nearest endpoint picking,
  projection-toggle origin action, and axis/drag callbacks into the owning camera controller.

`NativeRaster` also serves transform gizmos. Open rings and ticks use a single silhouette from
native `arc_ribbon_mesh(..., round_caps=True, smoothing=0)`, including folded-offset exterior
repair. This avoids accumulating translucent alpha at cap intersections.

`src/viewport/geometry.ts` ports position arrows, centered dimension boxes, affine plane corners,
projection fades and analytical perspective half-ring extrema. Orthographic direction uses the
camera's forward vector; it does not depend on the selected object's lateral position.

## Verification

From the repository root:

```sh
uv run --no-sync python electron/scripts/export-native-geometry.py
node --test electron/tests/viewcube.test.mjs electron/tests/gizmo-geometry.test.mjs
uv run --no-sync python electron/scripts/capture-native-viewcube.py
```

The native exporter obtains fixtures directly from Python production functions. Numerical
checks cover disappearing lollipop necks, tiny arrows, square-handle collapse, affine planes,
fade endpoints, fifteen perspective/orthographic half-ring projections, circular caps and
extreme ellipse offset folds. Shape vertices agree within 1e-8 points; half-ring projection
allows 1e-5 points because the native camera matrices store float32 values.

Run `make -C electron viewcube-review` from the repository root for the complete
visual comparison. The script captures the production Python widget, starts its own Vite
server on an available port, captures the browser fixture with installed Chrome at the native
framebuffer density, checks interactions and writes the raster comparison. It closes the
server and browser on completion or failure. This requires the repository Python runtime
(`uv run --no-sync`), an available native OpenGL display, and installed Chrome. The browser script also checks origin projection toggling, aligned-axis
ownership, held drag, unclaimed backdrop clicks, and pointer cancellation.

`../output/electron/verification/viewcube-native-gallery.png` and `../output/electron/verification/viewcube-web-gallery.png` use the
same 90-point widget, MiSans nominal em size and camera views. Inspect both galleries when changing the geometry.
The native font configuration includes the production 1326/1000 MiSans em conversion.
`../output/electron/verification/viewcube-raster-comparison.json` records the comparison: geometry excluding
dilated bright glyph pixels differs by at most 2/255 channels in four views and 5/255 in the
rear-axis fade view. Chromium and FreeType glyph rasterization remains different; these
captures do not claim bit-identical text output or certify other viewport components.

## Scalar-joint interaction

`src/viewport/joint_interaction.ts` ports native hinge/slide range geometry, adaptive ring
sampling, outward value/limit ticks, external round-tail slide arrows and unchanged larger
picking silhouettes. A collapsed range reveals the native 128-point precision rail after a
0.50-second dwell, with 0.12-second interruption tolerance and 0.30-second reveal grace.
Endpoint ticks set the current joint value to the authored limit on matching pointer release;
they do not edit limits. Rail movement covers the complete range, clamps outside it, supports
Shift snapping, and participates in one undo transaction with cancel restoring the original
value. Joint clamping rebases the drag so reversing away from a limit responds immediately.

`uv run --no-sync python electron/scripts/export-native-joints.py` regenerates the
independent Python projection fixtures. `node --test tests/joint-interactions.test.mjs` from
this folder compares ten native range/rail projections and exercises real browser gestures,
including dwell, endpoint release ownership, cancellation and transaction counts. The geometry
fixtures also cover G3 round-tail arrows and fitted joint-tick caps at very short lengths.

## Active drag guides

`src/viewport/drag_guides.ts` ports the production guide painters' geometry: a translation
ruler anchored to the current displayed pose and frozen start axis, world-space snap value
placement, viewport clipping, center/arrow masking, the native five-to-ten-step tick fade,
constant-pixel radial rotation ticks, screen-space rotation dials, absolute joint-angle sectors,
shortest equivalent multi-turn sectors, radial endpoint cuts, and the hinge's dashed rear axis
while it traverses the projected disk. The sector uses the native triangle fan with an outward fringe only on its complete
outer boundary, including its two radial boundaries; its border remains an independently
fringed ribbon with native limit-endpoint caps.

`uv run --no-sync python electron/scripts/export-native-guides.py` regenerates fixtures
by invoking production `_Guides` methods. `node --test tests/drag-guides.test.mjs` compares
24 translation ruler cases, 16 active rotation/screen/joint cases, and 12 hinge-axis projections
at oblique, near-edge, close-perspective, and orthographic views, with a 2e-5 point tolerance.
These independently verify submitted geometry and cap policies; final integrated raster and
interaction acceptance remains part of the viewport acceptance suite.
