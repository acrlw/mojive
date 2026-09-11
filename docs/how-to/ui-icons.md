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

The antialias fringe is different from authored geometry. `ImguiDraw2D` keeps that fringe one
framebuffer pixel wide so an edge remains smooth at each display density. Do not feed the fringe
width back into the icon's frame, mark, or spacing dimensions.

### Separate placement bounds from ink bounds

The Icon library uses a circular 24-unit placement boundary, drawn in orange after the glyph so a
collision cannot hide beneath opaque ink. The circle describes the slot used by layout; it is not a
target that every silhouette should fill. Candidate ink, including half of every outline stroke,
keeps at least 1.5 grid units of radial clearance. Detail rows report radial `pad`, the
axis-aligned `box` center, and the approximate `ink` mass center. A reviewed production icon can
retain its existing optical envelope: Output's circular severity frame has 0.72 units of radial
clearance and conforms to the circular placement boundary without crossing it.

Use radial containment in addition to rectangular canvas containment. A camera body can fit inside
a 24-by-24 square while its stroked corner still crosses a 24-unit circle. Likewise, a centered
axis-aligned box does not prove that the icon looks centered. Point-symmetric and rotationally
symmetric symbols should place their approximate ink center within 0.05 grid units of the placement
origin. Directional symbols keep a centered alignment box and receive a visual check at every
target size; their ink centroid is diagnostic, not a target to force to zero.

The Scale symbol demonstrates why both measurements are required. Its three equal axes are 120
degrees apart, so their shared origin and ink center coincide exactly. The top square makes the
axis-aligned box extend farther upward than downward. Translating that box to zero would move the
actual rotation center below the placement origin and recreate the visible imbalance the metric is
meant to catch.

Shaft-and-head arrows use one continuous filled outline. Do not join an independent stroked shaft
to a filled triangle: antialiasing and cap geometry expose the seam at small sizes. Object outlines
also form one intentional contour. A camera integrates its top housing into the body boundary, and
a bulb connects its dome, shoulder, and base before adding interior detail or separated rays.
Rotation and reset also need distinct structures. Mojive's transform rotation mark reuses the
runtime Tool Column geometry: three cyclically occluded half-rings inside a screen ring. Reset uses
one nearly complete circular arrow. Color or a small positional change does not separate two icons
that share the same silhouette.

Use the shared G3 curve builders for rounded heads, boxes, and structural corners. In particular,
`arrow_points`, `box_handle_points`, and `smooth_polygon_corners` preserve continuous curvature at
the visible joins. Drawing a square on top of a shaft or letting round-capped cube spokes terminate
on the outer stroke produces protrusions and seams at compact sizes.

## Keep a family visibly related

Members of one family use the same outer bounds, baseline, frame weight, internal stroke class,
round-cap profile, and semantic color behavior. Derive related symbols instead of tuning them
independently. For example, Mojive's warning mark is the vertical reflection of its information
mark; both therefore keep the same stem, dot, gap, and centerline.

Center the visible silhouette according to the family's declared alignment box. Area-centroid
centering is unsuitable for a stem plus dot: the stem carries much more area, so centering its ink
mass moves the visible bounding box toward the dot. For the severity family, the combined mark
bounds and the circular frame share one center.

Mathematical equality may still look unequal after rasterization. A small filled circle loses
visible area to antialiasing around its entire perimeter, while a long stem retains a solid center.
A modest, documented optical overshoot is valid. Keep it as a ratio on the design grid, check it at
the smallest production size, and apply the same correction to mirrored members. Avoid large
per-glyph compensation that creates a different visual language.

Diagonal marks carry more ink because two strokes overlap and project onto both axes. Preserve the
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
| `i` or `!` looks vertically displaced despite a centered area centroid | Unequal stem and dot areas skew ink-mass centering | Center the combined visible bounds or use a documented optical alignment box |
| Warning looks unrelated to information | Stem, gap, or dot was tuned separately | Generate one from the other's reflected geometry |
| Dot disappears even though its diameter equals the stem width | Circular antialiasing removes more visible area | Apply a small grid-relative dot overshoot and inspect the target raster size |
| Error looks heavier, then becomes stylistically thin after correction | Cross stroke width was reduced independently | Keep the shared internal stroke and shorten the diagonals |
| Scale looks low although its bounding box is centered | Unequal axis lengths or box-only centering displaced the shared origin | Use equal 120-degree axes and verify the ink center at the placement origin |
| Rotate is mistaken for Reset or Refresh | Both use a circular-arrow structure | Use the three-axis Tool Column rotation rings and reserve the single arrow loop for Reset |
| Cube spokes protrude through the shell | Independently capped lines terminate on top of the outer stroke | Inset the spokes, join their center, and paint the G3 outer contour last |
| One preview looks good but production does not | A copied demo or resized screenshot bypasses production geometry and density | Render the production painter at every target logical size and framebuffer scale |

The severity bug that motivated these rules combined two failures. Absolute stroke floors made the
mark occupy 67.2% of a 12-pixel icon and 61.0% of a 14-pixel icon, compared with 49.5% at 56 pixels.
Area-centroid centering then displaced the visible `i` and `!` bounds. The grid contract now holds
the mark at 44.8% for 12, 14, 20, 32, and 56 pixels.

## Verify geometry and raster output

For every icon-family change:

1. Test normalized bounds at the smallest production size, the design size, and at least one large
   diagnostic size. Include frame, mark, stroke, gap, safe area, centerline, and mirrored geometry.
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
