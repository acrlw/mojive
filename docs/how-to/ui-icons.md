# Design and verify UI icons

Mojive draws compact UI icons from filled vector contours. Treat an icon family as a small type
system: its members share a canvas, proportions, weight, cap style, alignment, and scaling rules.
Changing one contour in isolation usually produces a visible mismatch at Output-row sizes.

This guide applies to panel icons, filter pills, viewport controls, status glyphs, and diagnostic
symbols. For drawing ownership and submission APIs, see [Extend UI drawing](ui-drawing.md). For the
shared corner and cap profiles, see [G3 UI corners](ui-corners.md).

## Author one canonical grid

Choose one square design grid for the family. Use 24 units for new compact UI families unless an
existing family already has a documented grid. Express every visible dimension on that grid:

- outer silhouette and frame diameter;
- frame and internal stroke widths;
- stem length, dot diameter, and the gap between them;
- cap and join style;
- mark bounds, safe area, and alignment axis.

Apply one scale at the drawing boundary:

```python
scale = rendered_size / ICON_GRID
stroke = GRID_STROKE * scale
gap = GRID_GAP * scale
```

Do not clamp one authored dimension with `max(minimum_pixels, value * scale)`. A per-part pixel
floor changes the icon's proportions at the threshold. In particular, a minimum stroke width makes
the stem and dot consume more of a small circle while its safe area continues to shrink.

The antialias fringe is different from authored geometry. `ImguiDraw2D` normally keeps that fringe
one logical pixel wide so an edge remains smooth at each display density. Do not feed the fringe
width back into the icon's frame, mark, or spacing dimensions. A compound mark with subpixel
knockout gaps needs one shared, size-aware fringe for every contour: limit each fringe to a
documented fraction of the rendered gap until the gap can contain the normal one-pixel ramp. This
preserves the authored stroke and gap ratios instead of changing either geometry value at small
sizes.

### Use explicit geometric centers

The Icon library uses a circular 24-unit placement boundary, drawn in orange after the glyph so a
collision cannot hide beneath the visible mark. The circle describes the slot used by layout and
provides one adjustable size envelope for the candidate set. Candidate geometry, including half
of every outline stroke, defaults to 0.75 grid units of circular clearance. This closely matches
Output's reviewed circular severity frame, which measures about 0.72 units, while keeping every
candidate inside the placement circle. The Icon Library header exposes a component-group selector
with `Glyph padding`, which adjusts candidate clearance from 0.50 to 4.00 units. Each actual
consumer keeps an independent value: Viewport tools, Viewport playback, Keyframe
transport, Keyframes actions, Panels, Scene helpers, and Status & input. Family sheets, capsule
specimens, and the whole-UI preview resolve the value for the component they are showing. Viewport
tools default to padding 0.50; the other groups keep their own defaults. Rotate is frame-aligned:
its outer screen-ring centerline coincides with the orange placement circle, while half of its
stroke sits on each side and the three inner rings keep the same visible 1.50-unit width.
Output's mature Info, Warning, and Error painters are also locked; the review control does not
change their geometry, size, or placement. Detail rows report `pad`, the declared anchor at
`0.00, 0.00`, and either the axis-aligned box or minimum enclosing circle as a secondary diagnostic.

Viewport Playback Play is an equilateral G3 triangle. Previous and Next retain their original
90-degree chevrons, while Pause retains its original twin bars. Both More names call one contour:
rotate Previous counterclockwise by 90 degrees, shorten its arms to 94 percent, and keep the source
stroke plus each component's Previous layout scale. Each specimen also draws a neutral square whose
width and height equal the orange circle's diameter, exposing the shared slot center at 14, 24, 56,
and 112 points. Playback and Keyframe Transport Record have a measured visible width of 98 percent
of the corresponding Stop width. Keyframe Transport First and Last make their bar exactly as tall
as the final rounded triangle.

Use circular containment in addition to rectangular canvas containment. A camera body can fit inside
a 24-by-24 square while its stroked corner still crosses a 24-unit circle. Likewise, a centered
axis-aligned box does not prove that every icon looks centered. Placement therefore follows one
declared geometric feature per silhouette:

- Previous, Next, More, Keyframe Transport First/Last, and Tool Snap center their complete visible
  axis-aligned box;
- Playback and Keyframe Transport Reset place the center of the authored circular ring on the slot;
- all remaining candidates center their sampled minimum enclosing circle.

The concept library samples the full visible boundary, including half of each outline stroke,
translates the declared center onto the orange-circle center, and solves the scale that reaches the
requested padding. It counter-scales ordinary construction strokes during that fit, so every main
stroke resolves to 1.50 units on the 24-unit grid; padding changes the glyph's reach without also
making its lines heavier. Solid masses, gaps, dots, and head dimensions continue to follow the
glyph envelope. The reviewed Output severity and original mouse painters retain their established
weights. Align the icon slot to the current font's baseline and cap-height guides, inspect the real
row height at 1x and 2x, and compare related or mirrored icons. If a glyph still looks displaced,
correct its authored geometry or its documented anchor, then rerun every target size. Keep the
before and after capture in the review output.

Apply interaction color at the control boundary rather than baking it into the icon master. Capsule
controls use `viewport.off_foreground` at rest, `hover_foreground` on hover,
`press_foreground` while pressed, and `on_foreground` when selected; the current theme maps the last
three to Primary Bright. Record and Stop are one semantic recording action and always use
`viewport.record` (Danger red), including selected and pressed states. Disabled alpha attenuates the
resolved semantic color after this mapping.

Mouse hints use the original production `draw_mouse_hint_glyph` geometry in the family sheet and in
whole-UI previews. The shell, true knockout around the selected button, control fill, wheel, muted
state, and suffix colors therefore remain identical to Status. `Mouse W` changes the original
hint's aspect ratio and consumed Status layout width together, while height remains tied to the
existing hint control height.

Shaft-and-head arrows use one continuous filled outline. Do not join an independent stroked shaft
to a filled triangle: antialiasing and cap geometry expose the seam at small sizes. Object outlines
also form one intentional contour. A camera integrates its top housing into the body boundary, and
a bulb connects its dome, shoulder, and base before adding interior detail or separated rays.
Rotation and reset also need distinct structures. Mojive's transform rotation mark reuses the
runtime Tool Column geometry: three cyclically occluded half-rings inside a screen ring. Reset uses
one nearly complete circular arrow. Color or a small positional change does not separate two icons
that share the same silhouette.

The Icon Library owns its Move, Rotate, and Scale candidate contours. Keep this experimental drawing
inside `design/tools/ui_icon_concepts.py`; do not change `viewport_widgets.py` while tuning a concept
sheet. A candidate may repeat the reviewed production construction grammar, such as Rotate's three
cyclic half-rings, without sharing the production painter. The feasibility-only `gap / stroke`
control keeps its full 0.25 to 1.00 review range and must not clamp production geometry.

Use the shared G3 curve builders for rounded heads, boxes, and structural corners. In particular,
`arrow_points`, `box_handle_points`, and `smooth_polygon_corners` preserve continuous curvature at
the visible joins. Drawing a square on top of a shaft or letting round-capped cube spokes terminate
on the outer stroke produces protrusions and seams at compact sizes.

Filled triangles and diamonds are contours, not special exceptions. Run their three or four corners
through `smooth_polygon_corners`, including transport Play, skip controls, disclosure arrows, and
keyframe diamonds. A small radius still removes the curvature discontinuity at a nominally sharp
tip; review the 112-point specimen to catch a raw polygon that looks acceptable only at 14 points.
Build open transport chevrons as one filled ribbon and smooth its outer cap, inner join, and tip as a
single contour. Use G3 rectangle paths for Pause, Stop, and skip bars instead of backend rounding.
Build each Keyframe Fit corner the same way, and keep Follow's shaft and head in one arrow mesh with
a round tail.
Snap reuses `_snap_glyph_shape` with `CAPSULE_SMOOTHING`, so its U-turn has the same G3 continuity as
the viewport capsule instead of combining straight stems with a conventional semicircle. Its two
reviewed endpoint blocks cover the stroke caps and use G3-smoothed corners; preserve both blocks
when the production path is shown in a concept sheet.

A magnifier is also one hollow contour. Blend the lens shell and handle field with third-order
contact, tessellate the visible ring while retaining its circular hole, and submit it as one indexed
mesh. Separate circle and line primitives expose a round handle cap inside the lens and cannot form
one continuous neck.

## Keep a family visibly related

Members of one family use the same outer bounds, baseline, frame weight, internal stroke class,
round-cap profile, and semantic color behavior. Derive related symbols instead of tuning them
independently. For example, Mojive's warning mark is the vertical reflection of its information
mark; both therefore keep the same stem, dot, gap, and centerline.

Center the visible silhouette according to the family's declared alignment box. Area-centroid
centering is unsuitable for a stem plus dot: the stem carries much more area, so centering the
filled area moves the visible bounding box toward the dot. For the severity family, the combined mark
bounds and the circular frame share one center.

Mathematical equality may still look unequal after rasterization. A small filled circle loses
visible area to antialiasing around its entire perimeter, while a long stem retains a solid center.
A modest, documented optical overshoot is valid. Keep it as a ratio on the design grid, check it at
the smallest production size, and apply the same correction to mirrored members. Avoid large
per-glyph compensation that creates a different visual language.

Diagonal marks carry more visible area because two strokes overlap and project onto both axes. Preserve the
family's internal stroke width and adjust the diagonal length to balance weight. Do not make the
cross thinner than the information and warning stems merely to reduce its area.

## Use optical sizes deliberately

A proportional master is the default. If actual 1x and 2x captures show that it becomes too light
at a small production size, introduce an explicit compact optical master rather than scattered
pixel floors. An optical master must define the whole icon at that size: stroke weight, dot,
stem, gap, frame inset, and safe area. Keep the outer alignment box stable, state the size range in
which the master applies, and test the transition between masters.

This follows established icon-system practice:

- [Lucide](https://github.com/lucide-icons/lucide/blob/main/docs/guide/packages/icons.md) uses a
  24-by-24 view box with a shared default stroke width and round caps and joins.
- [Material Symbols](https://developers.google.com/fonts/docs/material_symbols) exposes optical size
  separately from weight so stroke weight can adapt when symbol size changes.
- [Apple's icon guidance](https://developer.apple.com/design/human-interface-guidelines/icons)
  requires a custom icon set to keep size, detail, stroke weight, and perspective consistent;
  [SF Symbols](https://developer.apple.com/design/human-interface-guidelines/sf-symbols) supplies
  coordinated weights and scales for the same reason.

These systems do not supply Mojive's exact ratios. They establish the useful separation between a
shared design grid, family weight, and intentional optical variants.

## Output severity contract

`severity_meshes` in `mojive.ui.panels.filters` is the production source for information, warning,
and error glyphs. The current 24-unit master is:

| Dimension | Grid units | Result |
| --- | ---: | --- |
| Canvas | 24.000 | Placement box |
| Circular frame diameter | 22.560 | 94% of the canvas |
| Frame stroke | 1.320 | 5.5% of the canvas |
| Internal stroke | 1.740 | 7.25% of the canvas |
| Dot diameter | 2.053 | 1.18 times the internal stroke |
| Stem height | 6.960 | Shared by information and warning |
| Dot-to-stem gap | 1.740 | One internal stroke |
| Information/warning mark height | 10.753 | 44.8% of the canvas |
| Error mark height | 8.443 | 35.2% of the canvas |

The information and warning marks are exact vertical reflections. Their dot and stem use one
horizontal centerline. At every rendered size, the normalized frame and mark bounds must remain
the same. The measured vertical clearance between the information or warning mark and the inner
edge of the frame is about 4.58 grid units on each side.

## Failure patterns to prevent

| Symptom | Cause | Required correction |
| --- | --- | --- |
| Small mark nearly touches its frame while the large mark has ample space | One or more dimensions use an absolute pixel minimum | Return every dimension to the common grid, or define and verify a complete optical master |
| A glyph becomes larger but remains visibly off-center | Its declared geometric anchor was not moved with the complete master | Translate the declared box, ring, or minimum-circle center first, then apply one complete-master scale |
| `i` or `!` looks vertically displaced despite a centered area centroid | Unequal stem and dot areas skew area-centroid alignment | Center the combined visible bounds or use a documented optical alignment box |
| Warning looks unrelated to information | Stem, gap, or dot was tuned separately | Generate one from the other's reflected geometry |
| Dot disappears even though its diameter equals the stem width | Circular antialiasing removes more visible area | Apply a small grid-relative dot overshoot and inspect the target raster size |
| Error looks heavier, then becomes stylistically thin after correction | Cross stroke width was reduced independently | Keep the shared internal stroke and shorten the diagonals |
| Icons with the same authored stroke look heavier after fitting | The per-icon envelope scale was also applied to its stroke | Counter-scale ordinary construction strokes by the inverse fit so the final main weight remains 1.50 grid units |
| Scale remains displaced after it is enlarged | Its source origin was preserved instead of its declared minimum-circle center | Center the sampled minimum enclosing circle before fitting the complete master |
| Scale endpoints overpower the center hub | Endpoint blocks were sized independently of the shaft and dot | Keep the G3 blocks close to the hub diameter and compare their area with the hub at 112 points |
| Scale shafts merge into the center dot | A copied three-axis sketch omitted the established transparent center shell | Start all three concept shafts outside the dot's circular clearance radius and verify the visible gap |
| Rotate is mistaken for Reset or Refresh | Both use a circular-arrow structure | Use the three-axis Tool Column rotation rings and reserve the single arrow loop for Reset |
| Rotate is clean in a large Pillow export but breaks in the live ImGui UI | Ear clipping runs after narrow concave contours are translated to large screen coordinates and loses precision | Triangulate the contour around its local origin, then translate the completed mesh during submission; verify the actual target-size ImGui capture |
| Rotate has visibly different stair stepping from neighboring small glyphs | A custom subpixel fill fringe replaced the established production AA paths | Use ImGui's antialiased circle stroke for the outer ring and the standard one-pixel filled-contour fringe for the local inner rings |
| A concept experiment changes an established Tool Column icon | Candidate geometry was placed in the production painter | Keep Move, Rotate, and Scale candidate contours in the Icon Library and leave `viewport_widgets.py` untouched |
| A triangle looks rounded only in the thumbnail | The preview hid a raw three-point polygon or an undersized corner profile | Inspect the native 112-point contour and require more than three authored boundary points |
| Playback and Transport More do not match | Separate painters or size adjustments have drifted | Rotate the shared Previous construction and reuse the corresponding Previous layout scale |
| Directional edge marks drift toward their open side | A minimum enclosing circle was used for a silhouette whose opposing terminals define the layout | Center the completed visible box for Previous, Next, More, First, Last, Panel Right, and Panel Down |
| Mouse Wheel has a smaller shell than Left and Right | Each state was independently fitted even though production uses one shell size | Fit all three states from the Left outer-shell master and preserve the adjustable source width |
| Snap looks like a generic U or loses its endpoint blocks | Its stems and semicircle were authored as unrelated primitives, or the probe copied only the centerline | Reuse the production G3 snap path with `CAPSULE_SMOOTHING` and retain both G3 endpoint blocks |
| Snap sits low despite symmetric endpoints | Its open U was centered by a circle unrelated to its visible terminals | Center the complete visible box while preserving the production G3 contour |
| Reset's ring drifts behind its arrowhead | The complete arrow silhouette was centered instead of the circular body | Place the authored ring center on the slot and let the arrowhead extend naturally |
| A compound icon looks low or right despite a zero diagnostic | The wrong geometric feature was declared as its anchor | Choose the visible box, authored ring, or minimum enclosing circle from the actual silhouette |
| Search handle cuts into the lens or exposes a cap | Lens and handle were submitted as separate strokes | Build one hollow G3 union mesh with a continuous outer neck and retained circular hole |
| Search lens grows spikes or leaks white flecks into its hollow center | Near-duplicate outer columns or independently sampled fill and hole-fringe boundaries destabilize antialiasing | Use one monotonic x-grid, replace its nearest samples at the hole endpoints, and derive the hole fringe from the strip's exact inner vertices |
| Sort arrow tip does not meet the visible bottom of its last bar | The arrow endpoint was aligned to the bar centerline | Align the head tip with the lower stroked edge while keeping the round tail on the top centerline |
| Sort arrow has a flat exposed tail | The shared arrow mesh kept its default butt tail | Enable its G3 round-tail contour while retaining the integrated head and shaft |
| Mouse hints change shape or color when Icon Library preview is enabled | A separate concept painter replaced the reviewed Status control | Keep `draw_mouse_hint_glyph`, its true knockout, and its theme color resolver in both modes |
| Cube spokes protrude through the shell | Independently capped lines terminate on top of the outer stroke | Inset the spokes, join their center, and paint the G3 outer contour last |
| One preview looks good but production does not | A copied demo or resized screenshot bypasses production geometry and density | Render the production painter at every target logical size and framebuffer scale |

The severity bug that motivated these rules combined two failures. Absolute stroke floors made the
mark occupy 67.2% of a 12-pixel icon and 61.0% of a 14-pixel icon, compared with 49.5% at 56 pixels.
Area-centroid centering then displaced the visible `i` and `!` bounds. The grid contract now holds
the mark at 44.8% for 12, 14, 20, 32, and 56 pixels.

## Verify geometry and raster output

For every icon-family change:

1. Test normalized bounds at 14, 24, 56, and 112 points. Include frame, mark, stroke, gap, circular
   safe area, declared anchor center, box and minimum-circle diagnostics, centerline, and mirrored
   geometry.
2. Render the production painter at its actual logical sizes. Do not scale down one large capture
   as a substitute.
3. Inspect both 1x and Retina output when raster behavior or visibility is in question.
4. Compare the family together, including inactive pill colors and log-row usage. A glyph that
   works alone can still carry the wrong weight beside its siblings or text.
5. Keep diagnostic guides outside the production painter. Guides may report frame diameter, mark
   height, and safe-area ratios, but must consume the same production meshes.

For Output severity glyphs, run:

```bash
.venv/bin/pytest -q python/tests/test_panels.py -k severity
make ui-diagnostics
.venv/bin/pytest -q -m gpu python/tests/gpu/test_ui_refinement.py \
  -k 'output_toggles or output_filters or output_collapse'
make check
```

`make ui-diagnostics` writes `output/ui-diagnostics.png`. Its labels show
`frame diameter / mark height`; those percentages must remain stable across the displayed sizes.
Inspect the filter pills and log rows in the same capture, because those are the production-size
uses that expose weak dots, crowded safe areas, and mismatched weights.

For the full candidate set, `make ui-icon-concepts` writes 14, 24, 56, and 112-point family pages
under `output/ui-icon-concepts/`, including separate `viewport-playback.png` and
`keyframe-transport.png` sheets, plus `tools-max-gap.png` at the production 1.46-pixel stroke and
the full `gap / stroke = 1.00` setting. `capsules.png` uses the actual playback and viewport-tool
capsules with their icon slots and state circles exposed. Viewport playback and Keyframe transport
each keep their own Previous/Next padding adjustment so
the chevrons' visible height does not exceed that component's Play/Pause. Scale preserves its
original center mask and handle footprint while its candidate shaft resolves to the shared Tool
stroke after DrawList fringe compensation. `playback-layout-4x.png`
checks that the Geometry page sections remain disjoint at its maximum inspection zoom.
It also writes `context-workspace.png`,
`context-panels.png`, `context-hints.png`, `context-hints-hidpi.png`, `context-keyframes.png`, and
`context-redesign.png` with the same candidates placed in real feasibility controls. Pass
`--icon-padding` after `--icon-group` for deterministic non-default captures of that group. In
interactive mode, use the Icon Library's group selector and `Glyph padding` to compare placement and
0.50–4.00 grid-unit clearances. Status `Mouse W` adjusts the original mouse-hint aspect ratio. Use
the always-visible `Icon Library preview` menu-bar switch,
or the matching item under `Probe`, to apply or remove that substitution across every feasibility
page. The Icon Library canvas reserves enough scroll extent for the longest family; verify the final
row is reachable in the normal 1600-by-1000 interactive window.
