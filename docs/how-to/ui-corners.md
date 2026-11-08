# Continuous UI corners

The native ImGui workspace uses scoped component radii and **circular corners by default**.
Read the [project corner rules](../guides/ui-design.md#spacing-alignment-and-corners) for
current nested surfaces and the native-window boundary. This page describes the shared
optional G3 geometry, retained feasibility specimens, and base ImGui behavior. Its older
capsule dimensions and base frame radius are not universal workspace defaults.

Mojive's custom UI uses curvature ramps for rounded rectangles, playback and tool capsules,
rounded arrowheads, small rounded Scale handles, Snap, mouse hints, and stroke caps.
`python/geometry2d/curves.py` computes paths for ImGui drawing and PNG export.
`python/geometry2d/drag_link.py` computes implicit hollow connectors. The
[drawing extension guide](ui-drawing.md) describes the functions, coordinate conventions, and
tests for new widgets and diagnostics.

## Electron surfaces

Electron uses Mojive's cubic Bézier construction with **60% corner smoothing** for
rectangles and the native G3 capsule profile with **38.2% capsule smoothing** by default.
`electron/src/smooth-geometry.ts` owns the curve and offset geometry; `smooth-corners.ts` owns
measurement, nesting and paint invalidation. `corners.css` separates SVG paint from content
and input. The SVG layer paints the fill and boundary once; do not combine it with a circular
background clip on the same edge. Shadows and overflowing content retain their DOM owner.
Canvas readouts use the same generator. True circles and native window decorations retain
their existing geometry.

The path is computed from each control's resolved logical width, height, design radii and
smoothing. SVG is the browser drawing format for the computed straight, cubic and circular
segments, not a pre-rendered corner asset. Resizing a control recalculates its path. Browser
rasterization and application zoom occur after this calculation; changing a control's aspect
ratio must not stretch its end profile. DOM elements continue to own layout, text and input.

Focus outlines, borders and inset shadows derive their centerlines from the same surface.
For outline width `w` and CSS outline offset `o`, the centerline's signed outward distance is
`o + w / 2`. Preserve each layer independently and include outward extents in the paint bounds.
Do not replace an external focus outline with an internal border. Decorative rail surfaces
use DOM paint owners too. Native scrollbar thumbs use fixed-size G3 end caps and a variable
straight middle, preserving native hit geometry and dragging.

### Smoothing and curvature

For a right-angle corner, `s` is the fraction of the original 90-degree circular turn replaced
by the two Bézier transitions. The center arc has angle `(1 - s) * 90 degrees`; each transition
turns through `s * 45 degrees`. The design radius `R` remains the center arc radius. Its center
stays fixed as smoothing changes. The edge footprint is `p = R * (1 + s)`.

| Smoothing | Center arc | Each Bézier turn | Edge footprint |
| --- | --- | --- | --- |
| 0% | 90 degrees | No transition | `R` |
| 25% | 67.5 degrees | 11.25 degrees | `1.25 R` |
| 60% | 36 degrees | 27 degrees | `1.6 R` |
| 100% | No arc | 45 degrees | `2 R` |

The incoming transition has control points `P0=(0,0)`, `P1=(a,0)`, `P2=(a+b,0)` and
`P3=(a+b+c,d)` in its local coordinates. The first three points are collinear, giving zero
curvature at the straight-edge join. Solve the final handle from the circular endpoint
curvature instead of assigning a fixed ratio to `a` and `b`:

```text
phi = pi * s / 4
q = tan(phi / 2)
c = R * q * cos(phi)
d = 2 * R * sin(phi / 2)^2
b = 3 * (c^2 + d^2)^(3/2) / (2 * R * d)
a = R * (s + q) - b
```

For numerical stability, evaluate the equivalent `b = 1.5 * R * q / (1 + cos(phi))`.
The outgoing transition is the reflection of the incoming one. At positive smoothing, the
straight/Bézier and Bézier/circle joins have matching position, tangent direction and signed
curvature (G2). At 100%, the two Bézier transitions meet with matching curvature. At 0%, use
the ordinary circular corner explicitly; its straight/circle joins are G1.

G2 does not require monotonically increasing curvature along a transition. At 60%, an
unconstrained corner reaches approximately `1.291973 / R` inside its transition before
returning to `1 / R` at the circle. Offset safety must use the entire curve's maximum
curvature rather than the arc radius alone. The previous third-party `a = 2b` construction
had an endpoint curvature of approximately `1.470438 / R` at 60% and did not meet G2 there.

### Capsules and circles

Capsules use the native Mojive 180-degree end profile from
`python/geometry2d/curves.py::smooth_capsule_points`. They are a separate shape in the same
shared geometry module: `smoothCapsule` constructs the profile, `offsetCapsule` derives its
parallel contours, and `surfacePath` dispatches both capsules and rectangular corner arrays.
The React `Squircle` component declares the authored surface radius; the shared painter
resolves the contour and states. Canvas consumers use the same path entry point.

A capsule keeps its requested outer width and height. Each end consists of an arc-length
curvature ramp, an optional circular interval and the reflected ramp. The normalized ramp
is `kappa(u) = u^2 * (3 - 2u)`; its curvature slope is zero at both ends. The analytic capsule
therefore has G3 joins. Serialize the ramps as adaptively fitted cubic Béziers, retaining
endpoint position, tangent and curvature, and keep the circular interval as an exact SVG arc.
Do not serialize a coarse polyline or substitute four circular quarter-corners.

Smoothing describes the fraction of the 180-degree turn allocated to the two transitions.
At the default 38.2%, the center arc turns 111.24 degrees and each transition turns
34.38 degrees. The cap's
curvature scale is solved from the short dimension; the capsule's half-height design radius
is a bounding-box convention, unlike the fixed center-arc radius of a rectangular G2 corner.
Fit the two complete caps within the long dimension by lowering effective smoothing only
when needed. A square capsule reaches the true-circle limit. Long capsules must not lose
smoothing merely because their corner radii equal half their height.

Recognize capsule surfaces by four equal radii reaching half the shorter dimension, including
large CSS radius values. Explicit round surfaces and percentage-radius circles retain their
circle identity. Capsule borders and evenly inset capsule descendants are normal offsets of
one shared reference; recomputing each inset from its own dimensions changes end clearance.
Circular search-clear buttons and switch thumbs are independent content, not offset copies
of the capsule. Keep them circular and inspect their end padding separately.

Electron `capsuleEndPadding` solves the center using the actual fitted capsule profile. It
matches the angular mean distance to the end contour with the straight-side distance, as in
`python/geometry2d/curves.py::capsule_end_padding`. The continuous angular calculation agrees
with the retained 181-ray native reference within 0.005px over the verification fixtures.
Search icons and clear discs, switch thumbs and chip icons share this placement. CSS receives
`--capsule-end` before descendant geometry is measured, including intrinsically sized chips.
At 38.2% smoothing, a 28px-high shell has a 14.2453px end center; a 20px clear disc therefore has
4px vertical and about 4.2453px trailing box clearance. These unequal box insets compensate
for the G3 end profile. A circle inside a noncircular cap is not an exact parallel contour.

### Nested surfaces

Adjacent surfaces derive their actual contours from one reference curve. The circle-radius
relation remains `inner_radius = outer_radius - inset`; independently rebuilding a smaller
Bézier corner does not preserve the diagonal gap. For outward unit normal `n`, signed offset
`delta`, and reference curvature `kappa`, use:

```text
Q = P + delta * n
kappa_offset = kappa / (1 + delta * kappa)
```

A negative `delta` is an inward offset. Keep lines and circular arcs exact. Fit offset Bézier
transitions adaptively while matching endpoint position, tangent and curvature. Validate
serialized path coordinates with independent normal-intersection measurements. The position
error budget is at most 0.01 CSS pixels and is tightened for small gaps. Subdivision must
preserve continuity at newly introduced joins; a fixed number of sampled points is not an
offset construction.

Prefer the largest outer surface as the reference and derive descendants inward. All corners
of one equal-radius shell share the same resolved profile. Use actual painted border-box
insets after UI zoom, independently of framebuffer density. Resolve the footprint budget for
the complete nesting chain before generating any curve. If any member cannot fit its
transition, lower the shared effective smoothing while preserving design radii and insets.
For example, `R16 -> R8 -> R4` with total inset 12 and a 20-by-20 inner swatch requires
`16 * (1 + s) - 12 <= 10`, so its effective smoothing cannot exceed 37.5%.

An inward offset is regular only when `1 + delta * kappa > 0` along the entire reference.
Check curvature extrema before fitting. If the largest reference cannot produce a regular
chain, select a smaller legal reference and derive the outer members outward; do not hide
cusps, clip off loops or independently substitute a different profile for one child. Square
corners and true circles retain their explicit geometric cases.

Settings > General exposes independent **Corner smoothing** and **Capsule smoothing**
preferences from 0% to 100%. Their defaults are 60% and 38.2%, respectively. Each stored
preference updates its shape family immediately. Effective smoothing may be lower when
available space constrains a corner or nesting chain. Component radii and insets remain owned
by the design rules; the preference cannot change layout or break a nested radius chain.

Run `make -C electron corners-review` for geometry, controls, nested surfaces and Canvas
readouts. Inspect both themes, fractional UI scale, color/checker edges, connected channel
fields, hover/held paint and actual Electron captures. CSS superellipse geometry does not
implement this construction.

## Native ImGui controls

`make setup` also builds and installs Mojive's patched ImGui Bundle. To add it to an existing
checkout environment, run:

```bash
make setup-imgui
uv run --no-sync mojive editor
```

The first build needs `uv`, CMake, a C++17 toolchain, and network access. On macOS this includes
the Xcode command-line tools. `python/tools/build_imgui.py` downloads the pinned ImGui Bundle
1.92.900 source archive, checks SHA-256, applies the maintained patch, and builds a local wheel.
It pins nanobind 2.9.2 because that source release uses its four-argument ndarray export API.
The source, wheel, and reusable CMake build are under `build/imgui/`.
Subsequent unchanged builds reuse the wheel for the same Python ABI and host architecture.

Base ImGui controls use ordinary circular corners with a 4.8 px frame radius before
workspace component scopes apply their own radii.
Base menus, selectable rows, docking targets, and window decorations retain native drawing;
workspace menus use the shared workspace popup presentation.
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

Base ImGui panel padding is 10 logical pixels and does not depend on the corner radius.
Workspace content uses its resolved design specification and semantic group insets.
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

The retained custom G3 defaults use smoothing **0.382** for capsules, playback glyphs and
tool-column icon candidates, and **0.618** for the other custom corner geometry. These capsule
outlines use Soft white (the theme's text color at 25% opacity). Playback and tool capsules
share their thickness; their lengths follow the number of controls. The base ImGui frame
rounding remains **4.8 logical pixels**.

The retained viewport capsule specimens use icon/state/shell radii of **8 / 14 / 20** logical
pixels, a **34**-pixel center step, a **10**-pixel group gap, and a **20**-pixel tool divider. Optical end spacing puts
the first and last control centers **20.3532** pixels from the capsule ends, matching their mean
clearance to the straight sides. `OverlayGeometry.end_padding_ratio` stores the measured ratio
to the shell radius, so production layout scales without fitting the curve during startup.
The Geometry probe's **Copy current values** export includes this ratio for future tuning.
Its Playback zoom and Tool zoom controls only magnify the inspection specimens.

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
Open arcs in Transform rotation and Joint gizmos use ordinary circular round caps. They do
not use the G3 cap profile: screen projection can approach an edge-on degeneracy, and those
small caps do not benefit from the additional construction. Full circles and ticks retain
their separate geometry. This policy is independent of the Rotate UI icon below.

Rotate uses exact ellipse offsets. Its caps map the same reference curve through the ellipse's
arc-length and normal coordinates. This preserves third-order contact with both curved sides
while the offsets remain regular. The three rings keep their cyclic occlusion. Knockout edges
are intentional visibility cuts, not joins between one continuous closed curve. Oversized
knockout masks retain a conservative miter envelope when an ellipse offset would become singular.
View gizmo shafts use seventh-degree Bezier transitions whose first three derivatives match
the straight shaft and circular head. A white origin disk has a diameter of twice the shaft
width. Its antialias fringe extends outward and uses at most a quarter of the surrounding gap,
preserving the solid core and transparent separation at small UI scales.
The rounded tails stop one shaft width away from its edge, leaving a transparent shell without
mask geometry. The dot draws below the endpoints so aligned balls cover it naturally.
Hover brightens and slightly enlarges only the white disk, without a tooltip or full-widget
backdrop. Its padded hit area yields to visible axis balls and is disabled when the origin is
covered. Clicking the origin switches perspective/orthographic projection on release; dragging
orbits the view and never triggers the click action on release.
The neck and its normals approach the same sampled circle as the head covers its shaft;
coincident samples are removed before fill and AA submission.
Back endpoints and their labels fade with zero slope at both ends of the transition.
Transform and perturbation drag links blend the hollow origin, connector, and solid target
with an implicit union whose blend polynomial matches three derivatives at each limit.
Regular level contours therefore retain G3 contact. OpenGL and bgfx evaluate the same field
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
Icon commands are cached by shape, style, and size; screen position and semantic colors
are applied at submission. ImGui translates the complete submitted vertex range, including
the antialias fringe, in one native pass. Circular icon frames share a direct annulus mesh
instead of offsetting, repairing, and triangulating a general closed stroke.
The Rotate icon's interleaved inner rings retain their subtraction boundaries. Style changes
test exact intersections only for overlapping edge bounds and classify boundary midpoints
as a batch; translation never repeats that construction.
Hinge axis guides join the unchanged arrowhead directly to each visible shaft's circular cap;
they do not clip a completed arrow and reconstruct its tail.
Canvas filled rectangles, circles, and ellipses use direct convex fans. Arbitrary polygons
and paths retain the general compiler so concavity, holes, and fill rules remain correct.
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
arbitrary contours. Regular rotation ribbons bypass ear clipping; tight projected offsets can
fold and require exterior repair and triangulation. This repair prunes edge pairs by their
bounds and tests only reflex vertices inside candidate ears. Static caches do not cover a
continuously changing camera; `make joint-gizmo-profile` measures that path through the viewer.

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

This section describes the retained feasibility controls and their shared base painters.
References to production glyphs here mean the `ui/icons/` family used by those specimens.
For the adopted workspace SVG family and live component scopes, use `make ui-design-guide`
and `make ui-native-design` as described in the [UI rules](../guides/ui-design.md).

For a small side-by-side study using current production controls, run:

```bash
make ui-components
make ui-components BACKEND=bgfx ARGS='--ui-scale 2.25'
make ui-components-gallery BACKEND=bgfx
make ui-components BACKEND=bgfx ARGS='--corner-scene inspector'
make ui-components ARGS='--corner-scene material --corner-surface card'
```

The same page is available under **Probe > Components > Corners & controls**. Choose an example
from the selector or use **Previous example / Next example** to browse all nine studies:

| Example | Controls and states to compare |
|---|---|
| Basic controls | Search, numeric fields, projection segments and icon placement |
| Inspector properties | Text, combo, checkbox and exposure reset |
| Search & filters | Search pill, type segments and filter reset |
| Action group | Full-width buttons, focus and disabled actions |
| Transform properties | Production XYZ fields, axis reset and locked editing |
| Material editor | Color picker, roughness/metallic sliders and collapsible surface options |
| Scene browser | Search, selection, visibility, empty results and selection details |
| Joint controls | Production value cards, unit fields, rails, individual/global reset and locking |
| Capture settings | Format segments, resolution, conditional options and an output summary |

These are sample layouts built from Mojive controls, not exact replicas of complete viewer panels.
**A / Default controls** always retains the production control radius on the panel background.
**B / Candidate controls** previews local changes with shared sample values. Controls and previews
scroll independently. Example edits stay in the probe, and capture preview only shows a summary
without writing a file. Use `--corner-scene`
with `transform`, `material`, `scene`, `joints`, or `capture` to open a new study directly.

**Candidate surface > Panel** is the default: sections have no extra fill, border, rounding or
inset. Controls and selection states retain their own backgrounds. **Card** adds experimental
section backgrounds only to B, exposing **Card inset / B** and **Link card radius to inset**.
The link sets the card radius to the control radius plus its inset. This rule applies only to
visible nested surfaces; it does not change panel padding or the reference column. Switching
between modes retains sample edits and card settings. The gallery captures all nine examples
in both modes at 1x and 2.25x, with the surface mode included in each filename.

Both columns call the production search input, value rail, joined numeric/unit field, segmented
control and icon painters. The search pill retains its own half-height radius. **Camera extra Y**
moves only the candidate camera glyph relative to its reviewed production offset on the 24-unit
grid. Hide the guides to compare visual balance. **Reset experiment** restores the trial
parameters and returns to Panel mode. Nothing is saved to viewer settings or production defaults.

The candidate control radius starts at **4** logical pixels. The optional card starts at inset
**2** and linked outer radius **6**. These are probe presets, independent of production defaults.

The **Optical alignment** tab covers all 50 Icon Library production glyphs except
Info/Warning/Error. Open it directly with:

```bash
make ui-optical
make ui-optical BACKEND=bgfx ARGS='--ui-scale 2.25'
make ui-optical-gallery BACKEND=bgfx
```

Use **Family** and **Choose icon**, or step through all glyphs with **Previous icon / Next icon**.
This includes viewport tools and playback, keyframe transport and actions, panel controls,
scene helpers, and mouse hints. Tune X/Y from -4 to +4 on the 24-unit grid (+X right, +Y down). Each glyph
retains its own additional offsets while switching tabs. Zero uses the reviewed production
placement; Auto align measures and corrects that current production silhouette. These experimental
deltas are separate from the Icon Library's absolute offsets. The camera Y adjustment is shared with the corner
study. Compare 16/20/24-pixel glyphs in circular and rounded buttons, English/CJK icon labels,
and a toolbar of the selected family. The labels retain their production text placement while only the glyph
moves. These are context fixtures using production painters, not interactive editor controls.

**Auto-align weighted centroid** corrects each candidate glyph's alpha-weighted ink centroid
(amber dot). **Alignment strength** scales that correction from 0% to 250% (default 100%):
0% uses the production position, 50% moves halfway, 100% puts the centroid at its control
center, and values above 100% overshoot. This scales the displacement without changing the
centroid's weights or the correction direction. It cannot reproduce arbitrary manual X/Y
adjustments; switch auto-align off to tune those independently. The same correction applies
at every preview size, to icon labels, and to the selected family's toolbar glyphs.
The production reference stays unchanged. Auto mode
shows calculated X/Y values and disables manual sliders and reset; switching it off restores
the saved manual offsets. Blur sigma and contour threshold do not affect this alignment.
`--auto-align-centroid` enables the switch at startup. The centroid is also marked on the
sharp enlarged glyph, so guides remain useful with the blur diagnostic hidden.

The enlarged diagnostic shows production geometry and a cached Gaussian-blurred alpha mask.
**Gaussian blur / sigma** adjusts the standard deviation from 0 to 6 grid units (default 3;
0 shows the unblurred mask). `--blur-sigma 4` also selects the initial strength. The probe
convolves a normalized Gaussian kernel along X/Y, truncated at four sigma, on a padded
256-pixel canvas. Display interpolates full 8-bit coverage across a 128-sample grid instead
of drawing enlarged, flat blocks with quantized opacity. Stronger blur spreads and fades ink.

**Circle contour / fraction of peak** selects the enclosing-circle support from 5% to 80% of
peak opacity (default 25%); it affects the measured circle without changing the blurred image.
Optional guides mark control centers, visible bounds, the ink centroid (amber dot), and the
threshold circle with its center (small square). The circle-center trial offset is the negative
of that measured center on the icon grid; it is displayed for comparison and never applied
automatically. A symmetric Gaussian preserves the whole-image centroid without clipping,
while thresholding can change the shape and its enclosing circle. These measurements are not
unique perceptual centers. Judge the actual-size controls with guides hidden before accepting
a correction.
**Link mirrored offsets** applies the same X-reflection/Y-copy rule as the Icon Library to this
study's manual offsets. Edits work from either member of a supported pair; enabling the switch
preserves existing values until edited. **Reset selected** clears that glyph's manual offset,
and its partner's offset while linked. **Copy offsets** copies the active alignment mode,
link setting, and offsets as JSON for review (strength as a multiplier and all 50 effective
offsets in auto mode);
values remain local to the probe session and do not modify the viewer.

Control and Joints scalar samples in the older Panels/Workspace studies also call the production
value rail and numeric/unit controls. Their surrounding sample layouts, other panels, and settings
remain design studies rather than full replicas of the live editor.

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

Native circular rectangle corners must satisfy the same raster precision as circles. Bound
chord error to 0.125 physical pixels, converting through the current framebuffer density
before ImGui starts a frame. Application zoom affects shape size rather than the error budget.
Use adaptive `PathArcTo` instead of a fast lookup call whose fixed sample count cannot satisfy
large radii. Test the resulting vertices, including large radii beyond the lookup cutoff,
and inspect real popup and switch edges at normal/fractional scale. Comparing two equally
coarse renders proves compositing order, not curve quality.

Native temporary non-modal surfaces use the compact menu defaults: an 8-point shell,
4-point content inset and derived 4-point child corners. Text-only hints have no nested
control constraint. Explicit component customization and modal nesting remain independent.
The shadow is an underlay, including coverage beneath the opaque interior. Its fill must
precede the native surface rather than overlaying an independently sampled hollow inner edge.
`make ui-popup-review` compares this production ordering against shadow-first native raster
references in both themes, at normal/fractional scale, with and without borders. Inspect the
production Layers, Timeline and color popups as well as the enlarged edge captures.

Native floating-window borders share the fill's outer contour. Account for ImGui's half-point
`AddRect` shift and inset the stroke centerline and radius by half the complete border width.
Reusing the fill radius after insetting the stroke exposes bright or dark corner slivers.
Keep this correction in both tracked ImGui and the checked Bundle build recipe. Inspect
production color popups in both themes and at fractional scale with `make ui-focused-review`,
then inspect the actual desktop window; framebuffer images do not cover OS compositing.

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

Diagnostic outlines and their internal marks follow the shared
[UI icon design grid](ui-icons.md#output-severity-contract). Every authored dimension scales from
that grid; dots use a documented optical ratio instead of an absolute pixel minimum. Fixed local
contours, triangle indices and fringe geometry are cached; movement applies a native vertex
translation. Reset arrows reuse ImGui's tessellator once per local shape and scale. The Diagnostics
feasibility page shows 14/20/32/56-point glyphs and normal/hover/pressed slider states.

Search fields use half-height rounding independently of numeric inputs. Unit suffixes reuse the
same joined-frame primitive as Transform axis badges, with only the external corners rounded.

## Small diagnostics and compound fields

Diagnostic glyphs use separate outer and inner stroke widths, capsule-ended stems, and one
24-unit design grid. Frame, mark, dot, gap, and safe area scale together; only the antialias fringe
stays one framebuffer pixel wide at Retina scale. Contours, triangle indices, and fringe offsets
are cached; translating an icon does not resample its curves. `make ui-diagnostics` renders the
shared production glyphs, multi-size frame/mark ratios, and slider interaction states.

Compound numeric fields suppress the native navigation outline while retaining text selection
and keyboard editing. Only the outer ends are rounded; the number/unit seam stays square.
Non-switchable units use a darker neutral suffix and readable muted text. At narrow widths,
Transform axes and scalar controls wrap and use the complete available row.

`make ui-frame-profile ARGS="--diagnostics --asset actuator_visuals --hover-gizmo"` measures
actual widget calls independently of cProfile. Its budget applies to measured capsule drawing;
interleaved whole-frame differences remain signed because scheduling noise can exceed that cost.
