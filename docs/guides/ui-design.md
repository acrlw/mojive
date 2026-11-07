# UI design rules

This guide defines the visual and interaction contract for Mojive's editor, viewport controls,
design workbench, and interactive design guide. Use it when designing, implementing, or
reviewing UI. The [review checklist](../reference/ui-design-checklist.md) connects common
failure patterns to the constraints and checks that prevent them.

Read this page before changing visible UI or interaction. Then read the relevant
[icon](../how-to/ui-icons.md), [corner](../how-to/ui-corners.md), or
[drawing](../how-to/ui-drawing.md) guide. Required checks remain in the
[verification matrix](testing.md#change-mapping).

## Design intent and authority

Mojive is a desktop scene editor for simulation. The scene and current task receive visual
priority. Frequent actions should be easy to find near their context, with short pointer travel
and predictable keyboard behavior. Use restrained surfaces, readable typography, compact
controls, and deliberate spacing. Avoid decorative cards, repeated explanations, and multiple
selection indicators that compete with the working scene.

Compactness does not mean removing padding. When a layout feels crowded, inspect content
bounds, gaps, information hierarchy, and redundant controls before changing the font size.
When it feels empty, inspect width budgets and grouping before adding technical metadata.

A current user request can intentionally revise a design decision. Otherwise preserve the
adopted behavior and values. Historical experiments explain how the design developed; they
are not instructions to restore an earlier appearance. Existing code establishes what runs
today, but a reported defect in that code is not a design requirement.

| Material | Role |
|---|---|
| This guide and the review checklist | Cross-component rules, rationale, and review criteria |
| `python/design/resources/design-spec.json` | Adopted factory tokens, layout, and component overrides |
| `tokens-schema.json`, `component-catalog.json`, `component-groups.json` in the same directory | Valid fields, defaults, semantic owners, and inheritance |
| `python/ui/workspace/`, `python/design/native/` | Production presenters, shared controls, measurement, painting, and input |
| `make ui-design-guide` | Interactive specimens using production presenters and a local specification copy |
| `make ui-native-design` | Parameter editor attached to the actual production workspace |
| `examples/ui_design/native/svg/` | Adopted SVG masters and provenance; see the icon guide |
| `make ui-design-reference`, `examples/ui_design/web/` | Comparison fixtures and earlier design references |
| `make ui-feasibility` | Retained geometry and G3 icon experiments; not the adopted workspace icon catalog |
| `output/` | Temporary captures and measurements; never a runtime dependency or the only record of a rule |

`make ui-design-native` aliases `make ui-native-design`. Native here names the workspace
implementation; it does not mean that bgfx is required. OpenGL remains the default renderer.
The runnable command catalog and workbench details are in `examples/ui_design/README.md`.

## References and how to use them

The following are references for specific design questions. They do not replace Mojive's
measured geometry, input contracts, or adopted configuration.

| Reference | Project use | Boundary |
|---|---|---|
| [Apple macOS Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/designing-for-macos/) | Desktop input, window resizing, comfortable density, keyboard shortcuts, and personalization | Mojive's radii and spacing are project decisions. Do not claim that an arbitrary value or the nested-radius formula is an Apple requirement. |
| [Refactoring UI](https://refactoringui.com/) | Review hierarchy, spacing, contrast, information density, and unnecessary borders | Apply the principles to a scene editor; adding card backgrounds to every section is not a project requirement. |
| [Lucide design language](https://lucide.dev/contribute/icons/design-principles) | Consistent canvas, simple contours, rounded joins, recognizable meanings, and family review | Mojive deliberately uses a 1.5-unit family stroke; upstream's 2-unit contribution rule does not override it. |
| [Blender viewport gizmos](https://docs.blender.org/manual/en/5.0/editors/3dview/display/gizmo.html) | Reference for scene navigation and spatial affordances | A similar appearance does not authorize changing Mojive's shortcuts or gesture meanings. |
| `examples/ui_design/web/` | The browser prototype that informed the adopted workspace | Native feasibility, docking, and current production behavior must still be checked. |
| [Mojive corner geometry](../how-to/ui-corners.md#geometry-contract) | Mathematical basis for optional curvature-continuous geometry | Circular corners are the workspace default. SVG approximations must be labeled by the continuity actually verified. |

## Typography, color, and density

Factory defaults use bundled MiSans Regular and Semibold for Latin and Chinese. Numeric
fields and keyboard badges have a separate monospace role, with bundled Inconsolata as a
fallback. Preserve saved font choices and the System option. Use the approved resources and
provenance under `python/text/resources/fonts/`; do not copy system fonts into the package
merely because they are installed on the development machine.

Measure and paint with the same resolved font, size, scale, and coordinate precision.
Related text shares a baseline; icon slots align to the font's measured cap-height center.
Do not independently center every word's ink box: changing a caption must not move its
baseline. Missing CJK glyphs, fallback faces, and clipped descenders are acceptance failures.
Apply application UI scale once; framebuffer density is a separate conversion.
Native workspace presenters submit font sizes in resolved layout coordinates, so live scale
changes must not also apply an atlas-relative `FontScaleDpi` multiplier. Classic UI fonts use
that multiplier for their fixed atlas sizes. Verify actual painted text after scale changes,
including fractional scales below 1 on high-density displays. Component text clip rectangles
expand outward to integer pixel bounds (`floor` for minimum, `ceil` for maximum) so backend
scissor truncation never shaves trailing glyphs on right-aligned or fractional-coordinate text.

Diagnostic geometry comparisons use continuous transparency, not periodic pixel rejection.
Fine screen-door patterns can shimmer during camera or model motion and on display scaling.
Keep visual and collision roles legible, preserve opaque scene occlusion, and make picking
stable across adjacent pixels. Verify slow motion as well as still captures, with MSAA both
on and off. The native Both view keeps only the nearest visual surface and nearest collision
surface in separate premultiplied layers, then composites their opacity weights once. Internal
parts, back walls and farther objects of the same role must not build up color or opacity.
Diagnostic collision shells do not cast secondary shadows onto the visual surface.
Depth-tested layers retain instanced batches without transparent sorting. Opacity zero contributes
neither color, identity nor a selected outline.

The following table illustrates the adopted factory tokens. The specification and component
resolver remain the source for actual values; local group overrides can differ. Update these
examples when changing the corresponding factory defaults.

| Role | Factory token/value |
|---|---|
| Body / title / numeric size | `body_font_size=14`, `title_font_size=16`, `mono_font_size=16` |
| Control height / panel insets | `control_height=28`; horizontal `12`, vertical `10` |
| Application menu bar height | `application_height=36`; existing 28-point triggers centered vertically |
| Ordinary control radius / menu and choice row height | `control_radius=4`; `28` |
| Row gap / numeric gap | `row_gap=8`, `numeric_gap=4` |
| Preferred label width / minimum / control gap | `120 / 48 / 12` |
| Main / secondary / disabled text | `#E6E8EB / #A9AFB7 / #8E98A5` |
| Docked / floating / popup surfaces | `#1E2125 / #252A2F / #2D353D` |
| Input surface / subtle border | `#2B2F34 / #FFFFFF0D` |
| Sage accent / amber warning | `#9CBF8D / #E8B04F` |

Use semantic theme and component roles instead of copying these hex values into new callers.
The named Dark and Light palettes live in `design/resources/themes.json`. Settings > General >
Theme uses two labeled sun/moon segments. A separate titlebar action toggles the same
persisted choice, without an extra confirmation or redundant hint. Both apply color roles
through the workspace's design store and preserve geometry, fonts,
and component overrides. Resolve colors before blending surfaces or semantic entity tints;
blending a source dark color first produces an unrecognized ink in the light theme. Test both
themes in independent workspaces, after restart, and with readable text/action contrast.
Backend-neutral painters must receive the cached effective control theme, including nested
viewport chrome. A themed overlay surface does not theme a rail or shortcut badge drawn
inside it. Verify held gizmo feedback and status help as well as idle panels; keep world-axis
and range-endpoint hues independent from the surrounding UI surface.
Readout markers retain those hues but need at least 3:1 contrast on their UI surface. Resolve
their opaque contrast-corrected ink once per color/surface pair; camera-facing world opacity
must not wash out a label marker.
Compact geometry-mode selectors retain readable captions at the normal body size. Add icons
only when the combined content leaves a full text inset on both sides of each segment;
otherwise omit the redundant icons before shrinking text or hiding the mode names.
Menu icons identify recognizable actions or object types, rather than decorating every row.
Separate these action groups from plain-text configuration and information groups. Text-only
groups omit the icon column; check, shortcut and submenu columns remain stable across the popup.

Headings, primary values, labels, metadata, and hints need different emphasis. A source caption
should not dominate its section heading. Avoid large blocks of equally bright text. Preserve
axis and diagnostic meanings when applying hover, active, disabled, and selection colors.
Disabled styling must not multiply alpha twice or make a control unintentionally transparent.
Selection outlines belong below joint helpers, transform gizmos and viewport UI. Verify their
actual overlap, including depth-tested helpers that must still disappear behind scene geometry.
When a selected silhouette reaches the viewport boundary, continue its outline inward along
that boundary. Require continuous selected coverage to the edge so nearby disconnected
silhouettes do not acquire a false closing strip. Inspect all four edges and corners at normal
and high framebuffer density. Follow connected diagonal mask paths within the stroke's
sampling footprint; a row/column-only test leaves notches at slanted joins. At an antialiased
silhouette/border intersection, add the
complementary outside-stroke and inside-band coverage. Taking their maximum leaves a
translucent notch in the closing stroke. Preserve fractional coverage at the inner border-band
edge rather than rounding its width up to whole pixels. Check the full band at fractional
density too.

Use professional, concise English source copy and the shared localization path. The design
guide supports Chinese and English. Translate explanatory labels while retaining source
entity names, API identifiers, units, and engine algorithm choices such as `Euler`.
Use the product's established terms; do not invent another name for an existing action.
Translate complete message templates before interpolation, measurement, and painting. Include
tooltips, disabled-action reasons, empty states, validation feedback, and drag readouts in the
language review. Translate presentation captions without changing command keys, entity or
resource names, source values, or copied diagnostics. Verify English/Chinese switching in the
running workspace as well as the initial language; dictionary coverage alone misses callers
that bypass translation.
Frequent, recognizable actions can use icons with a short delayed tooltip. Unfamiliar actions
still need an understandable label. Do not repeat a fully visible caption in a hover hint;
keep hints for hidden or clipped captions, shortcuts, unavailable-action reasons, and additional
meaning. This applies equally to buttons, switches, and checkboxes. Tool names should stay brief.
Close and search glyphs do not need repeated action-name tooltips. Entity names in trees,
card headings and Inspector titles reveal their full value only when the visible label is
actually clipped. Check the label bounds, not the whole row containing trailing actions.
Keep useful shortcut, unavailable-action and additional-context hints separate from these
name-only hints.
Shortcut hints use the shared 14-point monospace key badges; axis/channel labels retain their
separate 12-point role.
Interaction Guide reuses the status hint descriptors, key badges and canonical mouse glyphs.
Resolve configurable gestures from the current input bindings. Keep action names and useful
context alongside the indicators; wrap complete badges and prose instead of shrinking input
geometry or hiding instructions. Guide indicators describe input, not executable buttons.
Within a guide row, action captions, key legends, pointer glyphs and explanatory text share one
cap-height center. Guide keycaps resolve their font size from the guide body independently of
status-bar preferences. Give single-key and pointer indicators a common input slot, and align
pointer descriptions on one column even when a double-click suffix is present. Do not mix raw
text origins with component-aligned text or add invisible caption gaps to unlabeled hints.

## Spacing, alignment, and corners

Use a 4-point layout grid and 8-point group spacing as the starting scale; dense controls
may use 2-point increments. Choose layout sizes and clearances from that scale before deriving
nested radii. This does not quantize stroke widths, icon paths, font metrics or scaled output
to even pixels. Retain an established component-specific exception only with its owning rule.

Every container owns explicit content insets. Every row or group owns its item gaps.
Text padding, icon-to-text spacing, and clearance on either side of a divider are separate
quantities. Changing one side of a status divider must not move both neighbors. A scrollbar
gutter, selected navigation row, material card, or floating close button needs the same care
as a main panel.

Use proximity to express ownership: label-to-value spacing must be visibly smaller than the
space between field groups, and neighboring entities need the largest separation. Review the
whole repeated column; equal gaps can make a caption appear to belong to the preceding values.

Section heading text and ordinary body labels share one leading content edge. The heading's
hover background keeps its own padding. Manual rows, tables, checkbox grids, and raw text use
that same column, including at fractional scales. Restore the enclosing bounds after a group.
Joint rows with leading icons align the body to the content icon and retain the name's icon gap.
Scrollable tree and inventory children size to whole row strides and defer initial selection
auto-scroll until the child window has resolved at least one row stride of height, keeping
boundary rows unclipped.
Use separators to distinguish groups, without adding a leading rule to every section.
A panel header already separates its first group; that group's heading must not add a second rule.
Read-only identity rows use the shared property grid and baseline, just like editable rows.
About is a modal in the shared dialog shell, with a lead overview, bulleted feature highlights,
and a structured horizontal build-metadata bar (version, source revision, and UI runtime).
It must not open a legacy diagnostic dictionary panel. Size About to its content without
an empty void above the footer, and size Interaction Guide so its initial viewport fits whole
guide rows without slicing a row across the scroll boundary. Left and right panel visibility
controls name their side and restore the last active left panel.
Settings has one rule below its window title; category headings use spacing. Section-level
reset actions (`Reset camera navigation`, `Reset contact style`, `Reset shortcuts`) belong
right-aligned on their owning section heading row with a reset icon rather than occupying an
isolated standalone button row at the bottom of the property table.
Typography contains separate Numeric and Output Messages groups, each with Font family and
Font size rows. Use the group heading to express scope rather than repeating a role prefix in
every field label. Retain independent preferences, collapse state and shared page columns.
Related settings share one property table and its measured label column. Adding a theme
selector must not create a separate full-width group: align its field with Language and
Model updates, and keep its caption beside the control while the row has room.
The same page shares its measured label column across independent sections, including
Geometry display, Both appearance and Contact points. Keep section spacing and collapsing
behavior, but align full-width fields on both sides rather than fitting each table in isolation.
Settings navigation measures complete localized category names with the same resolved font,
inherited icon extent and padding used by its buttons. English category names use title case.
A hover tooltip does not replace a complete caption when the layout has room for it.

Electron rectangular surfaces use Mojive's G2 cubic Bézier corners through the shared SVG
painter in `electron/src/smooth-corners.ts`. Retain authored radii. Derive adjacent nested
contours by normal offsets from a shared reference, checking both transition footprints and
curvature extrema. Capsules use the native G3 180-degree end profile through that same path
entry point; a saturated half-height radius must not disable their smoothing. True circular
controls retain circles. Backgrounds and boundaries use
one paint layer; Canvas readouts use the same generator. See the [corner guide](../how-to/ui-corners.md#electron-surfaces)
for the smoothing definition, constraints and verification. Native ImGui component scopes
retain their adopted drawing profiles.

Adjacent nested rounded surfaces share one resolved radius budget. For smooth Electron
surfaces, this equation describes the circular portion; the complete contour follows the
normal-offset construction in the corner guide:

```text
inner_radius = max(0, outer_radius - actual_inset)
```

Resolve native insets and quantization before deriving the inner radius. At fractional UI
scales, independently rounding the outer radius, inner radius, and padding breaks the
relationship. An explicitly authored parent radius constrains local children without changing
generic control defaults elsewhere. If the inset consumes the radius, retain the spacing and
use a square inner corner.

Menus and choice lists use `8 / 4 / 4` at scale 1 for outer radius / inset / inner
radius. Both use 28-point rows with 2-point within-group gaps; menu dividers retain
their separate 10-point budget. Compact density changes row and shell budgets without
shrinking typography or flattening the larger separation between semantic groups.
Menu rows retain 10-point horizontal content insets in addition to the shell's
4-point inset. Removing an unused icon column must not remove the text's breathing room.
The application menu bar uses a 36-point shell around its existing 28-point controls.
Its first menu starts at the shell's content inset, with no brand glyph or reserved logo slot.
Reduce surplus vertical clearance before shrinking captions or hit targets. Larger authored
controls still expand the shell, and fractional scales reserve the complete native row.
Ordinary framed controls use a 4-point radius at their existing 28-point height.
Floating surfaces continue to derive their child corners from actual insets; these
component defaults do not give every surface the same radius.
An operation notice's Undo action pairs its icon with its caption. Its painted trailing
inset matches the vertical button clearance, and the child radius derives from that actual
inset. The notice text retains its own leading padding. Routine notices use an 8-point
outer corner and 4-point vertical/action inset; the embedded action derives a 4-point
corner. Reserve the 28-point action height even for messages without an action, keeping
the shell 36 points high before and after Undo. The icon-to-caption gap is 8 points;
the caption-to-action gap is 16 points, separating feedback from the available command.
Keep this shell for both committed and pending operation messages; larger authored fonts
or controls expand both states equally.
On light surfaces, filled secondary actions use a subtle tonal step from their host. Keep
the icon and caption readable, with stronger feedback on hover and press; a darker fill must
not make an optional action compete with the message or the primary action. Derive the fill
from semantic field and text inks. Do not impose a text-contrast threshold on the difference
between two background surfaces, darken every input, or change established dark-theme colors.
Use solid accent for the committing action in an action group (Apply, Save, or color Done),
tonal neutral fill for secondary actions (Undo, Cancel, or dialog dismissal), and unfilled
buttons for lightweight utilities. Selected navigation and filter states retain their own
tinted treatment; they are not a substitute for a committing action. Primary hover feedback
must preserve caption contrast rather than washing out its fill. Native extension buttons
and workspace painters share the secondary palette. Viewport tools and Snapping use solid
accent for selection; their hover and held states retain the semantic primary fill and ink.
A generic tinted-selection rule must not replace that fill when the pointer enters. Inspect
real popup footers, pending-edit actions, panel actions and toolbar buttons in idle, hover,
held and disabled states.

Plain help and status tooltips resolve the active control radius rather than an
import-time geometry constant. They own a separate surface, so a caller's nested
corner constraint must not leak into them. Shortcut hints retain the corner budget
derived from their keycap and inset. Form and color popups use an 8-point inset;
modals use 12 points, deriving their outer corners from the controls inside.
Electron uses the shared themed tooltip owner instead of browser `title` bubbles. Its interaction
guide reuses native mouse glyphs and separate keycaps; wrap complete input gestures together.
Hint captions use the normal body role. Electron shortcut hints use a 4-point keycap radius,
6-point inset and derived 10-point shell; larger body text increases the keycap height to preserve
equal clearances. These layout dimensions follow the same 2-point increments as other controls.
Use compact action captions and context notes, resolve modifiers from the host platform, and
list only bindings implemented by that frontend. Hints inside modals retain their dialog owner.
Collapsible section titles need complete rounded hit surfaces and explicit text insets. Keep
caption alignment with adjacent property labels by extending the hover surface into the section
gutter. Color comparison swatches retain clearance on every adjacent edge of their button;
derive the swatch corner from that actual clearance, not from the standalone swatch style.
The Electron color editor uses a 16-point shell, 8-point inset and 8-point controls; its
20-point comparison swatches have 4-point clearance and 4-point corners.
The standalone color trigger uses the same 28-point height, 8-point outer corner,
20-point swatch and equal 4-point leading/top/bottom clearance.
Verify these resolved values in the real workspace as well as component fixtures. Generic
button rules must not override a component's inset or radius when stylesheet order changes.
Scope component geometry under its owner rather than depending on import order.
Color previews paint one rounded surface. Composite transparency over each checker cell
before painting; independently antialiased checker and color layers expose corner fringes.
Inspect dark opaque colors, partial transparency, fractional scale and framebuffer density.
Connected choices and XYZ fields also paint each outer corner once. Their frame provides
layout and corner values; the edge segments provide opaque themed fills and the rounded
silhouette. Do not combine a rounded frame background, rounded child fills and a second
rounded overflow clip. Check a uniform-fill raster against one rounded reference surface
to detect repeated edge coverage that radius measurements alone cannot expose.

Floating menus, choices, color editors and modals have one neutral native border. Their
shadow has a broad, continuous Gaussian-like falloff outside that border, slightly heavier
below the surface. Reuse local native gradient meshes when moving or repainting a window;
do not rebuild blur textures every frame. Avoid narrow linear fringes that read as another
border, stepped outline rings, or an accent halo around an entire popup. Each popup owner
paints its shadow once; the renderer needs no extra blur pass. Electron resolves one subtle
neutral edge and a broad shadow from the active palette; avoid overlapping dark outline rings
or tight secondary shadows that make the edge appear dirty.

Slider thumbs retain their solid accent identity across idle, hover and held states.
Use restrained accent and track changes for feedback; do not switch the thumb to text ink,
white, or an outlined control on press. Keep it above the track and min/max ticks.
Inspect idle, hover, held, endpoint overlap and release in both themes.

Verify the resolved startup specification as well as isolated factory defaults.
Saved specifications and live drafts override the packaged preset; an earlier full
snapshot can retain old geometry or literal dark-theme text colors even when the factory
comparison is correct. Settings navigation uses semantic text/accent inks in both themes.
When adopting a geometry revision into an existing profile, back it up and update
only the agreed fields, preserving fonts, navigation geometry and other authored
values. Do not silently reset a customized profile on application startup.

Numeric and hexadecimal input share the numeric typography role. A color editor's
smaller caption font must not shrink the editable color code. Fit the SV square and
hue/alpha bars to the content width, bounded by the available viewport height;
only an explicitly authored picker height constrains that default. Align a footer
hint to the input text inset, while footer buttons align to the control frames.
Measure native font quantization before deriving field padding so fractional scale
does not introduce a scrollbar and change popup width on alternating frames.
Axis/channel badges have their own `badgeFontSize=12` role and a 20-point strip;
they do not inherit the 16-point numeric role. Keep the label and value centered on the
same row, leaving visible space on both sides of the badge glyph. Computed typography
must survive the production authored-default resolver. Verify emitted native font sizes,
not only the specification or isolated specimen; changing font size alone must not
change the badge reset target.
Electron's compact form revision uses one 28px base height for scalar inputs, numeric channels,
Select triggers and segmented choices. Frames grow to the larger body/numeric font size plus
8px when needed. Channel badges use the same numeric font and size as their values. Derive
badge width from the field height, cap height and character advance; keep font side bearings
and glyph proportions intact. All five Electron segmented consumers share `Segmented`, its
4px outer radius and state palette. Interior fields keep their shared 4px radius; popup
ancestry alone is not a nested-corner constraint.
Electron Inspector density comes from group spacing, not shorter input frames. Use 8px between
property rows and from the heading target to its first row; keep 12px section insets above and
below. Account for the combined margin, padding and row minimum height before adjusting any
one value. Do not accumulate blank space around a shorter control inside a taller row.
RGB and RGBA use the same joined badge-and-value control as XYZ, with channels in one
horizontal row. Keep the badge beside its value in every format; do not turn it into an
above-field caption. Reserve width for all channels, including alpha, rather than reducing
numeric typography. Channel reset restores that channel's original value.
The adopted 12/14/16 type steps are a project convention, not a requirement that
every font or scaled raster size be even. Judge visible glyph size and baseline together.

Where the navigation rail follows a rounded native window edge, its outer radius uses native
window points, independently of application zoom. Derive its inner corner from the actual
equal leading and bottom inset. Fractional authored radii are valid. Inspect the actual desktop
window to assess the compositor edge; an internal framebuffer capture cannot show that edge.
Search fields keep their capsule silhouette. A capsule's leading icon uses its end-circle
placement. Neither is a generic rectangular child whose radius should be overwritten.
Electron search fields use one `SearchField` component for Scene, Project files, Joints,
Cameras, Output, Settings and file-dialog filtering. The component owns the 28px frame, 16px
leading icon, body typography, placeholder ink, 20px circular clear action and inset focus
contour. Panel styles own placement and available width only. Clearing retains input focus.
Search icons, clear discs, switch thumbs and chip icons use the actual capsule profile to
compute optical end placement. The angular mean end clearance matches side clearance; a
circular child and a noncircular G3 shell cannot maintain an exact constant gap at every angle.


G3 smoothing is optional and disabled by default in the adopted workspace. Keep the geometry
contract available without restoring earlier experimental appearances. The nested-radius rule
is the project's visual relationship; arbitrary G3 offsets are not identical mathematical
parallel curves.

## Icons and shared components

Follow the [icon guide](../how-to/ui-icons.md) for authoring and verification. The adopted
workspace family uses a 24-unit canvas and 1.5-unit round stroke, with pinned Lucide and
Mojive-owned SVG masters. Context aliases for the same action use the same master. Different
meanings need distinguishable symbols: Settings and General, Layers and content visualization,
camera objects and projection, state snapshots and image capture, isometric view and box creation.
Mouse hints and Output severity symbols use those same packaged masters. Status pointers use a
22-point canvas without increasing the 28-point bar or changing key/text sizes. Mouse button and
wheel fills use neutral text ink, including the muted status-bar treatment; they do not compete
with primary actions. The shell and dividers use 80% of that ink's alpha: muted outlines must
remain legible beside the filled button or wheel. Keep the canonical stroke geometry and normal
antialiasing; do not compensate for weak contrast with pixel floors, isolated stroke changes or
hard edges. The three status-pointer masters use an explicit 1.0-unit stroke on the 24-unit
grid, keeping their enlarged silhouettes light without fading the ink. Shells and dividers share
the weight; button fills derive their inset from its half-width. Other workspace masters retain
their 1.5-unit stroke. Diagnostic colors retain their severity meanings.

Mouse-button fills end at the inner edges of the shell and divider, not their stroke centerlines.
Keep the whole dark frame visible around the selected region; a brighter fill must not enlarge
the apparent width of the mouse or consume the center divider.

Keep optical placement in the authored grid. Do not refit or recenter each workspace glyph
by its own bounds during drawing. Scale strokes, dots, openings, and gaps together. A fixed
minimum pixel width on one feature is not a valid small-size correction. Inspect expanded
strokes, round caps, junctions, and inner boundaries; a clean bounding box can hide an internal
lobe or a closed gap. Shared mesh compilation handles compatible opaque unions and AA.

Review icons beside their actual neighbors and text, at small production sizes as well as
enlarged sizes. Equal canvases alone do not establish equal visual weight. Review default,
hovered, active, selected, and disabled states as applicable. Scene camera and light helpers
use the same semantic masters while retaining their own size, picking, and selection behavior.

Extend an existing control before writing another painter. The workbench, design guide,
production panels, and acceptance tools must share measurement, paint geometry, hit regions,
focus contours, and native item state. Shared controls contain presentation and gestures;
callers own domain commands.

Icon-and-text buttons measure the same resolved icon canvas, font, gap and padding used for
painting. An icon-size change must not silently remove filter counts. Pose uses independent
spacing roles: 12 points from the joint heading to its content, 4 from an above-row summary
caption to XYZ, 12 between Position and Rotation groups, and 20 between joint entries. Resolve
these through `rowGap`, `summaryLabelGap`, `summaryGroupGap`, and `gap` respectively. Include each
gap in both measurement and painting; inline summary captions have no vertical caption gap.

## Stable fields and component behavior

Settings option groups such as Theme and Geometry view occupy the normal label/control
row when width permits. Use the same property-row owner, control height and centerline
as selects and numeric fields; reflow only when the measured content cannot fit.

| Surface | Required behavior |
|---|---|
| Property rows | One label/control grid; vertically aligned captions; labels shrink or move above controls according to measured budgets. Boolean rows fit their actual caption and switch, independently of XYZ field minima. |
| XYZ and color fields | Joined axis badge and field form one surface. Use stable format budgets and equal columns. Use a complete vector/channel row or a complete stack, not an accidental two-plus-one layout. |
| Numeric text | Value changes do not choose another layout. Reduce display precision or use complete scientific notation within the allocated field; viewing a value never rounds stored data. Editing retains full precision and native caret/scrolling. Units belong to the shared field presentation. |
| Status metrics | Stable text boxes, digit cells, decimal anchors, and unit ends. Keep label/value/unit gaps and divider clearances independently adjustable. Compact large steps/time values, expose the exact value on hover, and copy the exact value on right-click. |
| Projection changes | Menu and view-cube actions share the camera's eased projection transition and target-scale matching. Reversing an unfinished transition starts at its current blend. Programmatic camera restoration can remain immediate. |
| Switches | Right-align in property rows, with shared track/thumb/hit geometry. Hover and press are distinguishable; the light thumb stays readable against the accent track. |
| Search | One production painter resolves icon size, capsule geometry, text padding and the reserved trailing clear slot for every panel. Native text input excludes the icon hit regions. Paint one inset keyboard focus contour; suppress the native input outline when the compound field owns it. Clearing a filter does not delete the underlying data. |
| Exclusive choices | Use shared segmented controls or choices; preserve the selected value and keyboard behavior during reflow. |
| Menus, More, and choices | Measure each popup's own longest title, shortcut, and optional columns. Check, icon, title, shortcut, and submenu arrow have distinct aligned slots. Omit a check column only if the entire menu has no checkable entries; selection changes must not shift columns. |
| Menu activation | Click activates a top-menu group; hover then switches menus. Escape, outside click, or completed action ends that mode. Ordinary inactive hover does not open menus. |
| Desktop menu parity | Native and in-window menus share one action catalog, ordering, separators, checked state and capability gates. Match the native viewer's semantic groups; do not expose unimplemented actions or substitute unrelated ones. Modal input ownership also disables native scene commands. |
| Primitive creation | Audit the complete native shape inventory, canonical icons, dimensions, Z-up orientation and orbit-target placement. An Add entry must create a real typed entity and survive undo/redo, timeline snapshots and save/reopen with its geometry intact. A physics Site is a model element, not a decorative marker. |
| Model loading | Show the candidate name and current stage before expensive work. Report counts only when their denominator is known; compilation has indeterminate progress. Keep the previous scene until the candidate is complete. Cancel terminates candidate work, and failure preserves the installed scene. Asset previews follow the same contract. |
| Tooltips and notices | Use shared surfaces and typography. Keep complete corners and centered content within the clip. File-action buttons align right when wrapped below text. Text-only captions beside action buttons need their own text inset in addition to the shell-to-button clearance. Avoid duplicate status messages or competing tooltip implementations. |
| Operation Undo | Bind the notice action to the exact committed history revision or latest pending-operation checkpoint. Retain earlier drafts when reverting one pending operation; invalidate the token on a later write, clear, apply or incompatible document. Mixed direct/draft writes must not expose a partial undo. Preserve the undo identity when a command interceptor forwards an already published result. Recheck eligibility on click. Translate built-in command and transaction captions at presentation, including Create, Apply and Undo, while preserving user names. |
| Gizmo value feedback | Transform drags, joint precision rails, and range endpoints share `workspace/gizmo_readout.py`: the numeric field’s resolved font, height, padding and radius on the floating surface. Center text by cap height, retain semantic axis dots, and scale viewport-edge clearance with the UI. Precision rails paint the track first, full-height limit ticks centered across it second, and the round thumb last. Inset the painted track by its cap radius so it cannot protrude past the ticks; preserve full-range hit and value mapping. During a gesture, a shorter tick marks the starting value on the rail and a distinct darker segment spans the start and current values in either direction. The change segment has flat caps at the reference boundary so no rounded bulge protrudes beneath the origin tick. The thumb covers the origin tick when they coincide. |
| Floating forms | Reuse `ui/workspace/floating.py` and the shared popup geometry. Closing a chooser preserves the selected joint; double-click focus can dismiss the chooser while retaining the gizmo. |
| Inspector | Properties owns geometry shape/dimensions; Physics owns contact/mass; Material owns appearance. A presentation refactor must not hide or duplicate these capabilities. |
| Output | Align severity glyphs with the first line’s visible text metrics, including localized fallback fonts. Scale the complete glyph with output typography and retain first-line alignment for multiline messages. Preserve selection, context-menu copying, search, severity filters, multiline messages, and clearing. Distinguish empty history from unmatched filters. Avoid a repeated copy button on every row. |

For stable numeric layout, test `51.1 → 59.0`, sign changes, long values, unit promotion,
and narrow fields. Fixed layout does not justify large unused blank regions: budgets remain
compact and adjustable. Test the real Chinese caption and actual font before concluding that
a label or menu needs ellipsis.

## Interaction and continuity

Selection, navigation, editing, and status have distinct owners. Clicking a panel activates its
context; moving the pointer across panels must not continually replace status hints. Orbit and
Pan do not clear selection. A selection hint can remain visible during navigation; an input
owner such as a text field or modal can temporarily take priority.

One pointer gesture has one owner through release or cancellation, even if a modifier is
released first or the pointer leaves the control. Preserve queued event positions. Popup
dismissal consumes its triggering click so it cannot also select a different scene entity.
Electron menu, timeline and help key labels resolve the preload platform: Ctrl/Alt on Windows
and Linux, Command/Option on macOS. The displayed command modifier also owns dispatch.

Binding editors record input directly: keyboard chords, mouse buttons, modifiers and wheel.
Render the result with the same keycaps and mouse glyphs used by interaction hints. Keep preset
selection separate from binding capture; do not require a key dropdown or typed gesture syntax.
Reset is a trailing icon action. Put brief instructions in contextual hints rather than permanent
small-text paragraphs. Section headings, joint pose captions, camera metadata, timeline lane names
and transient errors retain the normal body size; do not reuse compact badge/tick typography for
these readable labels. Capture, cancellation, conflict handling and persistence must agree with
actual dispatch and every displayed shortcut. Camera movement and transform tools
must share the same conflict domain. The native W/A/S/D/Q/E movement map coexists with V/G/R
tools through distinct bindings, rather than allowing one held key to move the view and switch
a tool. Text entry and modal owners suppress viewport movement; focus loss clears held input.
Application/window accelerators remain reserved in both recording and preference restoration;
include native Close, Hide, Minimize and Quit roles when checking platform conflicts.

Physical perturbation belongs to the selected dynamic body, including drags begun over empty
viewport space. Ctrl with the left/right mouse button translates/rotates that body and suppresses
ordinary gizmos and navigation. Translation uses the body's surface point when hit, otherwise a
camera-facing plane at body depth; rotation pivots at the body origin. Release of the button or
modifier ends the force; Escape, focus loss and lost capture cancel it without leaving a held force.
Physics examples must enter through the live backend rather than a kinematic preview loader.

One underlying property has one control within an Inspector. Domain-specific activation replaces
an equivalent generic visibility control; tree actions may remain synchronized alternative entry
points. Preserve separate controls only when their effects are independent.

Physics Environment controls inherit the loaded model's values. Keep timestep and gravity directly
available, with backend-specific solver and contact settings in collapsed groups. A simulation reset
does not restore environment defaults; expose those as distinct actions. Capability descriptors feed
the existing numeric, vector, selection and whole-row Boolean controls rather than engine-specific
widget copies. Actuator inputs reuse joint-card geometry and transaction behavior.
Simulation transport shares the viewport navigation row. Center Previous frame, Play/Pause and
Single step as one group; keep reset and Environment separate from transport, and identify speed
through its control hint. Show simulation time once in the status bar. Narrow layouts retain all
actions and reduce captions before consuming another viewport row. Reserve symmetric space for the
left and right groups so asymmetric captions cannot shift the centered playback control. When the
remaining width cannot fit full-size controls, allow horizontal scrolling instead of overlaps.
Previous frame restores a full dynamic backend state from the bounded 1000-frame history;
resuming replaces the discarded future.
Frame navigation retains current Environment settings; document undo owns parameter changes.
Space toggles simulation playback, and Left/Right step backward/forward while paused. Route toolbar
and keyboard actions through the same guards; held keys do not queue discrete steps or retrigger
Play/Pause. Text entry, dialogs, button activation and focused Timeline editing retain input ownership.
Environment focus uses the complete Frame All operation, including exiting a scene camera and the
same nonlinear camera transition. Tree double-click, Focus selected and Frame All share this route;
a geometry-free Environment entity's transform must not determine its focus bounds.
Asset discovery and project-file navigation have distinct entry symbols. A model preview owns an
independent document, camera and optional simulation runtime; browsing or closing it must not mutate
the open scene, its selection, simulation time or undo history. Opening the model is an explicit action.

Native text input and modal dialogs own their keyboard events. Respect physical Ctrl/Cmd
semantics on macOS and route shortcuts through the existing binding layer.

One completed edit is one undoable transaction. Escape restores the value from gesture start;
Apply/Discard retain their established pending-edit boundary. Clamping at a joint limit does
not end a held drag, and reversing from the limit must respond promptly. Color validation
failure preserves the previous value. Keep failure evidence and atomicity through Session.
See [implementation principles](development.md#implementation-principles) for command ownership.

Timeline gestures remain usable without physics, models, or recorded takes. Gate writes at
their capability boundary. Keep simulation control, recorded-take replay, range selection,
and Loop distinct. Bound zoom/time values, preserve holes after deleted samples, and align
samples, frame cells, range endpoints, and playhead to the same time mapping.
Electron timeline navigation has a zero lower boundary shared by wheel zoom, pan, scrollbars,
manual view fields and Follow. Preserve the pointer's time anchor away from boundaries; at
zero keep the zoom scale and clamp the view origin. Navigation bounds never cap recording.

Camera selection, preview, and editor navigation retain their declared modes. Focusing an
entity uses its current world pose and smooth camera movement; joint focus should leave its
scale legible instead of projecting it onto the axis. Camera Preview drags retain the initial
pointer offset and all viewport margins.

The first revealed window contains a completed frame and resolved docking. Resize rendering
and camera projection follow the current viewport extent; do not wait for a settling timer.
Fixed-resolution capture can retain intentional letterboxing. Scene captures include enabled
scene diagnostics; choosing scene, viewport, or window capture controls the UI layer.
For documentation captures, choose a physical UI scale that remains readable at the page's
display width, then resize dock splits to contain the showcased content. Check all Timeline
tracks after scaling; increasing the UI scale also changes toolbar wrapping and required height.
Inspect viewport tool orientation too: expanding Timeline can reduce viewport height enough to
trigger its compact horizontal layout.

## Independent groups and design editing

A semantic group is an owner such as Scene filters, Settings navigation, Inspector tabs,
viewport playback, Timeline transport, material cards, or status telemetry. Reusing a button
class does not make all its instances one group.

Group-local font family, size, color, icon size, control width/height, and spacing must affect
only that group. Container insets and control text padding remain separate. Resolve measurement
and painting through the same nearest owner and restore the scope afterward. Explicit child
overrides retain their precedence; Reset removes the override and reveals inheritance.
Global controls must be labeled as global.

Alt-click selects the nearest semantic control group without executing its ordinary action.
The inspection gesture owns the button through release. Related parent groups remain
discoverable. Preview and parameter controls should remain visible together while tuning,
using separate scrolling regions where necessary.

The guide edits a local copy. Export preserves all authored parameters, including manual
values temporarily hidden by an automatic comparison. A preview switch does not silently
adopt new defaults. To adopt a reviewed design, update its shared specification or source
master and all affected consumers. Never require a file under `output/` to start the product.

## Verification and maintaining the rules

Use the [verification matrix](testing.md#change-mapping) for required gates. The following
entry points help choose focused evidence; they are not an instruction to run every gallery
for every edit.

| Question | Useful entry point |
|---|---|
| Does the design match the actual workspace? | `make ui-native-capture`; `make ui-design-guide ARGS='--capture'` with the same explicit specification |
| Do headings and body columns align? | `MOJIVE_UI_SCALE=1.2 make ui-design-guide ARGS='--capture-alignment'` |
| Do groups edit and reset independently? | `make ui-native-audit`; `make ui-native-pick-check` |
| Do icons retain their silhouette at multiple sizes? | `make ui-design-reference ARGS='--icons --portable-fonts'` |
| Do floating forms and Inspector tabs retain behavior? | `make ui-floating`; `make ui-inspector-tabs` |
| Do interaction and notices work in production? | `make ui-design-runtime`; `make ui-notice-review` |
| Does resizing follow continuously? | `make viewport-resize`, plus desktop border-drag inspection when changing the OS event loop |
| Is frame or startup work excessive? | `make inspector-profile`; `make startup-profile`; `make ui-design-profile` |

Choose coverage for the affected contract: layout changes need relevant docked/floating
surfaces, narrow/wide layouts, languages, fonts, and fractional scales such as 1.2 and 1.25;
icon changes need actual-size and enlarged views; interaction changes need their input states.
Use both renderer backends where the verification matrix and affected workflow require them.
Compare identical design specifications, font choices, UI scales,
framebuffer density, scene state, and viewport sizes; record these inputs with the evidence.
Factory-default guide captures and a workspace with saved custom settings are not equivalent.

The agent inspects the images and interactions. A passing geometry assertion or a screenshot
of a lookalike implementation is insufficient. For repeated-gesture bugs, exercise held input,
release, cancellation, and subsequent editing. Inspect sibling consumers of a shared fix.

Performance work preserves approved appearance, hit bounds, and transactions. Profile the
reported production entry point, including selection/tab/disclosure changes and recovery to
idle. Separate UI work, renderer work, initialization, and system pressure. Reuse stable
geometry and measurements with explicit invalidation and bounded caches; skip invisible ink
without removing active native items. Do not flatten corners, change spacing, or shrink text
as an unrequested optimization.
Measure animation-frame intervals as well as CPU submission time: short render calls can still
queue expensive cross-context canvas copies. Batch adjacent overlay primitives, flushing at
Canvas2D paint-order boundaries, and copy only their painted device-pixel bounds. Preserve the
original projection, subpixel grid, clipping and alpha composition at integral and fractional
display densities. Prepare viewport-sized GPU backing stores before interaction; changing handle
bounds must not resize them. Measure cold activation separately from steady-state cost. Prepare
mesh raycast bounds during asynchronous scene construction and reuse rigid geometry bounds for
focus, invalidating on position-attribute replacement or upload-version changes. Numeric text
updates invalidate their affected rounded-control family, not
every control in the workspace; inherited theme, scale and font changes still invalidate globally.
For asynchronous physics edits, distinguish the requested gesture value from the last acknowledged
scene frame. Rebase only at a real limit; a delayed frame is not a clamp. Coalesce superseded
continuous targets before backend execution, while preserving capture, commit, cancel and undo
barriers. Pose editing pauses simulation; physical perturbation retains its running-state semantics.
Cancellation reserves its restore position in the command queue before awaiting an asynchronous
baseline. Escape, window blur and unexpected pointer-capture loss restore the same baseline;
clear gesture ownership before releasing capture.
Stable hover must return to idle after dwell feedback settles, including after a drag ends.
At capacity, measurement caches evict cold entries rather than clearing hot layouts along with
them. Repeated scene-row insertion and scrolling must preserve frequently used geometry and its
field-read evidence. Bulk cache churn can promote short-lived layout objects into older GC
generations even when entity and memory counts stay bounded.
Measure actual submitted-frame cadence separately from animation callbacks and worker latency.
An idle, on-demand viewport reports idle rather than a misleading low frame rate. Stress repeated
model replacement and multiple full-resolution robots in every geometry mode, including immediate
orbit after release, cancellation, recording and Undo/Redo. Depth-only and selection-mask passes
must not rebuild light shadow maps; preserve the shadow flags across these passes and retain
independent role-specific shadows in color passes. Require pixel comparisons with moving casters.
Pose-only recording history stores pose, physics and timeline state without copying immutable
mesh assets. Promote to a document transaction before an authoring edit, preserving the original
pose baseline and asynchronous capture ordering.
Share identical immutable field-read evidence within a frozen style instead of duplicating it
for every row position. Interning stays bounded and must preserve argument-dependent reads;
do not share mutable layouts or evidence across incompatible specifications.

When a correction establishes or revises a reusable rule, update its owning guide and the
relevant checklist entry in the same change. Describe the generic failure pattern, governing
invariant, affected consumers, and reproducible verification. Use repository code, tests, or
public references to explain the rule; exclude private conversations, personal identifiers,
and machine-specific incident reports. Remove or label obsolete guidance so it cannot be
mistaken for a current default.

Keep temporary measurements and captures under `output/` and reusable instructions in tracked
files. The repository skill `skills/mojive-ui-design/SKILL.md` routes contributors through this
workflow; it does not own another copy of the specification. A new rule does not create an
approval gate: use the project's existing visual review and baseline policy.

Electron status bars show task status, interaction hints and concise time/frame-rate feedback.
Rendering submission timings, draw-call counts and triangle counts belong to diagnostics.
Timeline and Output panel controls belong together in the activity navigation.
Camera cards use primary ink for the name and secondary ink for projection/resolution. Wrap
metadata from its actual available width instead of a fixed card-width threshold. Joint and
actuator metadata follows the same hierarchy without shrinking normal body text.
