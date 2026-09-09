# G3 UI corners

Mojive's custom UI uses curvature ramps for rounded rectangles, playback and tool capsules,
rounded arrowheads, small rounded Scale handles, Snap, mouse hints, and stroke caps. The geometry lives in
`python/src/mojive/curves2d.py`; ImGui and PNG export consume the same paths. Implicit hollow connectors
live in `python/src/mojive/draglink2d.py`. The [drawing extension guide](ui-drawing.md) maps API entry
points, module responsibilities, coordinates, and verification for new widgets and diagnostics.

## Native ImGui controls

`make setup` also builds and installs Mojive's patched ImGui Bundle. To add it to an existing
checkout environment, run:

```bash
make setup-imgui
uv run --no-sync mojive editor
```

The first build needs `uv`, CMake, a C++17 toolchain, and network access. On macOS this includes
the Xcode command-line tools. `tools/build_imgui.py` downloads the pinned ImGui Bundle
1.92.900 source archive, checks SHA-256, applies the maintained patch, and builds a local wheel.
It pins nanobind 2.9.2 because that source release uses its four-argument ndarray export API.
The source, wheel, and reusable CMake build are under `output/g3-ui/build/`.
Subsequent unchanged builds reuse the wheel for the same Python ABI and host architecture.

Production ImGui controls use ordinary circular corners with a 4.8 px frame radius.
Menus, selectable rows, docking targets, and window decorations retain native drawing.
Custom `Draw2D` shapes select their own G3 profiles independently of ImGui style.
There is no global G3 style field or replacement for native rectangle tessellation.
Use the drawing API below to style custom geometry.

The build recipe has three responsibilities:

- Add `add_indexed_fill` and `add_poly_fringe` to the Python binding. They use public ImDrawList
  buffer writers to avoid a Python-to-C++ call for every vertex or triangle.
- Keep slider grab bounds consistent between input and rendering, including the inset radius.
- Match keyboard focus rings to the control radius, including child windows and the focus outset.

Only the last two require small changes in `imgui_widgets.cpp` and `imgui.cpp`.
`imgui.h`, `imgui_internal.h`, and `imgui_draw.cpp` remain byte-for-byte upstream. Custom curve
math lives only in the shared Python geometry modules. The build restores previously patched
files from the checked archive before applying a changed recipe, so removed code cannot survive
in a cached source tree. `make setup-g3` remains an alias for existing workflows.

Panel padding is fixed at 10 logical pixels and does not depend on the corner radius.
Viewport uses zero content padding in both floating and docked windows. Its scene image
fills the docked content area. Floating windows retain their native one-pixel resize border
outside the image clip; custom overlay margins are owned by the overlays.

Slider grabs are at least as wide as the track's inner height. Vertical sliders apply the
same rule with the axes exchanged. This minimum square footprint fits the frame radius minus
the two-pixel inset without scaling it down again. The effective frame radius clamps to half
the frame's short side; `GrabRounding` can cap the resulting inner radius.
`GrabMinSize` can request a wider grab, and integer sliders still reserve space for one unit.
Short tracks clamp the grab to the available length. `SliderBehaviorT` owns these bounds for
both rendering and input, so radius edits do not shift the grab size or the value mapping.

A plain PyPI installation still supports all custom G3 geometry; native ImGui controls use
upstream circular corners with either installation. Dependency synchronization can replace
the local wheel. Run `make setup-imgui` again after synchronization, and launch with
`uv run --no-sync mojive`. The patch is a source-build recipe,
not a published binary distribution. The wheel is specific to the build host's platform and ABI.

The production defaults use smoothing **0.382** for capsules, playback glyphs and tool-column
icons, and **0.618** for the other custom corner geometry. Capsule outlines use Soft white (the theme's text color at 25% opacity). Playback and tool capsules share their thickness; their lengths follow
the number of controls. Native ImGui frame rounding remains **4.8 logical pixels**.

## Geometry contract

The reference curve is parameterized by arc length. Each corner consists of a cubic smoothstep
curvature ramp, an optional constant-curvature interval, and a reflected ramp. Curvature and
its arc-length derivative match on both sides of each join. At a straight edge they are both
zero. The smoothing fraction changes the ramp length; it is not a radius multiplier.

Rectangle rounding retains the requested edge footprint and clamps to half the short side.
It does not switch curve families when the radius saturates. Capsules have explicit 180-degree
end profiles and preserve their requested bounding box. Short capsules reduce ramp length to
fit; the square limit is a complete circle. True circles need no corner smoothing.

Generic stroke caps shorten the adjoining straight segment slightly to preserve their previous
tip extent. A very short segment reduces the smoothing fraction to keep the caps from overlapping.
This guarantees G3 joins to those straight endpoint segments; it does not turn an arbitrary
input polyline into an analytically smooth centerline.
Transform rotation-ring caps now use the same reference profile and the gizmo's smoothing value.
The sampled ring ribbon retains its endpoint-segment approximation; this is distinct from the
analytic ellipse-offset construction used by the Rotate icon below.

Rotate uses exact ellipse offsets. Its caps map the same reference curve through the ellipse's
arc-length and normal coordinates. This preserves third-order contact with both curved sides
while the offsets remain regular. The three rings keep their cyclic occlusion. Knockout edges
are intentional visibility cuts, not joins between one continuous closed curve. Oversized
knockout masks retain a conservative miter envelope when an ellipse offset would become singular.
View gizmo shafts use seventh-degree Bezier transitions whose first three derivatives match
the straight shaft and circular head. The neck contracts as the head covers its shaft.
Transform and perturbation drag links blend the hollow origin, connector, and solid target
with an implicit union whose blend polynomial matches three derivatives at each limit.
Regular level contours therefore retain G3 contact. OpenGL and WebGPU evaluate the same field
on one existing quad. Endpoint overlap can change the hole topology; a collapsed hole is not
a regular curve and has no G3 guarantee at its instant of disappearance.
The connector blends into the outside of the origin while its inner circle remains intact.
Only an overlapping target can close or reshape that hole.
Flat plane handles and perturbation outlines map G3 reference corners into their local planes.
Regular perspective projection preserves their third-order contact with adjoining straight edges.
Plane handles use a 2 pt corner footprint. The solid gizmo uses a rounded plane mesh built once
and shared by all three axes. Playback and Pause retain matching visible heights; frame-step
glyphs are 12% smaller and Reset is 8% smaller to leave room inside the circular icon guide.
Scale uses the same circular origin as Transform, and its Tool Column glyph has a matching
clear shell around that origin. Shared `arrow_points` geometry rounds both convex head
corners and concave head-to-shaft shoulders. Shoulder radii default to half the head corner
radius and remain independently configurable. `box_handle_points` supplies the same connected
shaft-and-square silhouette for Scale gizmos and Tool Column icons. Its concave joins are
rounded as well as its outer corners. Joint MIN/MAX ticks use the range's stroke width;
their shorter length distinguishes them from the current-value tick.
The viewer draws perturbation axis arrows through the shared UI silhouette path. Headless debug
consumers retain world-arrow primitives. Solid 3D cone/sphere meshes are separate surfaces;
the screen-space corner setting does not promise G3 surface continuity for those meshes.

G3 describes the reference curves. The renderer submits sampled polygons with antialiasing.
The nominal chord tolerance is `0.025` coordinate units; cached icon paths are then scaled with
the UI. Shared paths and transformed ImGui vertices use bounded caches.

Translation reuses local capsule and icon geometry. Short-cap fitting evaluates only the
scalar footprint during its search and samples the final curve once.
The patched binding submits a complete outward antialias fringe in one native call; its
vertices, indices, UVs, and colors match the Python fallback.
The CPU drag-link fallback builds indexed vertical strips around the hollow origin, including
overlapping crescents. This takes linear work in the sampled contour size and avoids concave
ear clipping. An overlapping hole is parameterized by the sum of its distances to the endpoints,
so sampling its boundary does not repeat a root search at every horizontal coordinate.
Long connectors reuse endpoint meshes and stretch only the straight span.
The patched binding submits all indexed vertices and triangles in one call. Only external
contours receive an antialias fringe, including the inward-facing edge of the hole.
Sampled rotation ribbons also use cached strip and cap indices, with linear submission work.
Their end caps consume the available arc length and remove covered samples instead of refitting
curvature to every changing endpoint segment. General concave fills remain available for small
arbitrary contours, but the known rotation-ribbon topology bypasses ear clipping.

These changes remove repeated preparation work without reducing sampling accuracy. G3 paths
still contain more vertices than simple circular corners, and moving silhouettes still need
transformation and rasterization. The CPU benchmark includes warm static shapes, translation,
continuous resizing beyond cache capacity, short-cap fitting, and native submission:

```bash
make g3-benchmark
```

It writes batch timings to `output/g3-perf/cpu.json`. Compare runs on the same machine and
dependency build. Whole-frame timing also includes scene rendering and scheduling noise;
subtracting two frame medians cannot establish that corner drawing costs zero.

## Live corner controls

Open the UI Feasibility corner page with:

```bash
make ui-corners
```

The same page is available under **Probe > Geometry > Corners**. Eight independent floating-point
sliders cover custom capsules, playback icons, tool icons, mouse hints, transform
gizmos, joint gizmos, the view gizmo, and perturbation gizmos. Values span `0.0` through `1.0`
and remain active while switching probe pages. The controls can copy current values or restore
the production default of `0.6`. Probe experiments are not persisted to viewer settings.

The separate **ImGui corner radius** slider spans 0–16 logical pixels and defaults to 4.8.
It scales window, child, popup, frame, tab, and scrollbar radii together, preserving their
relative proportions. Grab radii follow the inset rule above. The same slider is available
from the **Probe** menu on every page.
Copy and reset include both the radius and smoothing controls.

Keyboard focus uses the native circular rectangle path. Its radius follows the outward
expansion of the focus frame after clamping to the control's size. Child panels supply
their own window radius instead of borrowing the frame radius. Scale handles and scalar
joint gizmos draw their colored geometry once, with antialiasing and no contrast outline.

Zero selects the baseline profile. Positive values enable smooth curvature transitions where
rounded joins are defined. Smoothing changes the transition shape; it does not enlarge the corner
footprint. Short
capsules and constrained stroke caps limit their transition length to fit their available space.
Actual circles retain their circular shape. Every geometry cache includes the profile value.

Generate the same preview at zero, the production default, and one with:

```bash
make ui-corners-gallery
```

The screenshots are written under `output/g3-controls/`. The preview uses production drawing
functions and includes native controls, Scale handles, ticks, View necks, and perturbation outlines.
The timeline and hierarchy previews call the production glyph functions. Mirrored polygon fills
normalize their winding before ImGui submission so their antialias fringes face outward.
Local glyph profiles, fill winding, and native vertices use bounded caches; translating a glyph
does not rebuild its curvature profile.

## Verification and captures

```bash
make check
make g3-ui
make ui-frame-profile ARGS='-o output/g3-ui/profile'
```

`make g3-ui` exports glyphs, mouse hints, and actual viewer captures under `output/g3-ui/`.
Inspect the playback and tool-column closeups, mouse shells, Rotate caps, and native settings
controls. Geometry tests verify one-sided curvature limits, dense reference approximation,
exact bounds, mirror symmetry, and scale independence within tolerance. Binding tests compare
indexed fills and antialias fringes with the Python fallback, including buffer rollover.
Native slider and focus checks require `make setup-imgui`.

### Diagnostic glyphs and compact fields

Diagnostic outlines and their internal capsule strokes share the same solid width and antialias
convention. Dots have a minimum visible radius, with separate spacing from the stem. Fixed local
contours, triangle indices and fringe geometry are cached; movement applies a native vertex
translation. Reset arrows reuse ImGui's tessellator once per local shape and scale. The Diagnostics
feasibility page shows 14/20/32/56-point glyphs and normal/hover/pressed slider states.

Search fields use half-height rounding independently of numeric inputs. Unit suffixes reuse the
same joined-frame primitive as Transform axis badges, with only the external corners rounded.

## Small diagnostics and compound fields

Diagnostic glyphs use separate outer and inner stroke widths, capsule-ended stems, and a minimum
visible dot size. The antialias fringe is one framebuffer pixel even at Retina scale. Contours,
triangle indices, and fringe offsets are cached; translating an icon does not resample its curves.
`make ui-diagnostics` renders the shared production glyphs and slider interaction states.

Compound numeric fields suppress the native navigation outline while retaining text selection
and keyboard editing. Only the outer ends are rounded; the number/unit seam stays square.
Non-switchable units use a darker neutral suffix and readable muted text. At narrow widths,
Transform axes and scalar controls wrap and use the complete available row.

`make ui-frame-profile ARGS="--diagnostics --asset actuator_visuals --hover-gizmo"` measures
actual widget calls independently of cProfile. Its budget applies to measured capsule drawing;
interleaved whole-frame differences remain signed because scheduling noise can exceed that cost.
