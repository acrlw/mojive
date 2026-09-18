# G3 UI corners

Mojive's custom UI uses curvature ramps for rounded rectangles, playback and tool capsules,
rounded arrowheads, small rounded Scale handles, Snap, mouse hints, and stroke caps.
`python/geometry2d/curves.py` computes paths for ImGui drawing and PNG export.
`python/geometry2d/drag_link.py` computes implicit hollow connectors. The
[drawing extension guide](ui-drawing.md) describes the functions, coordinate conventions, and
tests for new widgets and diagnostics.

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

Viewport capsules use icon/state/shell radii of **8 / 14 / 20** logical pixels, a **34**-pixel
center step, a **10**-pixel group gap, and a **20**-pixel tool divider. Optical end spacing puts
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

For a small side-by-side study using current production controls, run:

```bash
make ui-components
make ui-components BACKEND=bgfx ARGS='--ui-scale 2.25'
make ui-components-gallery BACKEND=bgfx
make ui-components BACKEND=bgfx ARGS='--corner-scene inspector'
```

The same page is available under **Probe > Components**. The scene selector offers basic controls,
an Inspector card, a search/filter toolbar, and an action group. All scenes retain current/candidate
columns with shared sample values. The gallery captures every scene at 1x and 2.25x.
Inspector fields, filter reset, and action buttons are interactive without changing the viewer.
Both columns call the production
search input, value rail, joined numeric/unit field, segmented control, and icon painters with
shared sample values. The current column keeps the production frame radius; the candidate uses
a local override. Its nested child surface is a test fixture: **Link outer radius to inset**
sets its radius to the inner control radius plus the shared inset. The search pill retains its
own half-height radius. **Camera extra Y** moves only the candidate camera glyph relative to its
reviewed production offset on the 24-unit grid. Hide the guides
to compare visual balance. **Reset experiment** restores the trial parameters. Nothing is saved
to viewer settings or production defaults.

The corner candidate starts at inner radius **4**, inset **2**, outer radius **6** logical
pixels. This is a probe preset, independent of production defaults.

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
