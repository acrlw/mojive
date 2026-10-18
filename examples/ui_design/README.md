# UI design study

The native design is now the default production Mojive workspace. `mojive view`,
`mojive editor`, and public viewer builders use `python/ui/workspace/`, with
Session commands and the existing ViewerApp frame pipeline retaining scene and
application ownership. `make ui-native-design` opens the native parameter editor
alongside that production UI. `make ui-design-native` is a compatibility alias.

The standalone source in this directory remains a comparison fixture for icon,
typography and prototype-workbench experiments. Use `make ui-design-reference`
for those galleries; the production runtime imports no example or output code.

Run from the repository root after `make setup`:

```sh
make ui-native-design
make ui-native-design BACKEND=bgfx
make ui-native-design ARGS='--part statusbar'
```

Alt-click a production component, including the bottom status bar, to select its
parameters. Edits update its painted geometry and picking bounds. The header
switches between separate and embedded editor windows.
The inspection gesture owns the left mouse button until release, including when
Alt is released first. Buttons retain their normal actions on ordinary clicks.
Alt-click the runtime status text or dot to select the status and operation parent group.
Its margins and padding position the status, recording actions and hints together. The hint
subgroup retains its own item spacing; telemetry remains in the right-hand metrics group.
Close the window to end the interactive run. bgfx requires the
[native runtime and shaders](../../docs/how-to/native-viewer.md).
On an existing checkout, install the new development dependency without replacing
the locally built ImGui wheel: `uv pip install 'fonttools>=4.0'`.
Python dependencies and their resolved versions are declared in `pyproject.toml`
and `uv.lock`.

`make scene-entities BACKEND=opengl` captures the production camera and light scene
markers, including selected camera frustums and point, directional, spot and area light
helpers. Use `BACKEND=bgfx` for the matching native captures. Both use the adopted UI
camera and bulb silhouettes at a 24-point viewport size. Add
`ARGS='--diagnostic-only --overview-only -o output/scene-diagnostics'` to inspect the
explicit Camera and Light render flags with editor helpers hidden. The tool uses a
hidden window and writes `overview.png` before the selected-entity captures.

## Shared design guide

Browse the workspace specification and its production panels with:

```sh
make ui-design-guide
make ui-design-guide ARGS='--language en'
make ui-design-guide ARGS='--page "Group laboratory"'
make ui-design-guide ARGS='--defaults'
make ui-design-guide ARGS='--spec /absolute/path/design-spec.json'
make ui-design-guide BACKEND=bgfx
make ui-design-guide ARGS='--capture --output output/ui_design/guide/opengl'
MOJIVE_UI_SCALE=1.2 make ui-design-guide ARGS='--capture-alignment --output output/ui_design/alignment/opengl'
MOJIVE_UI_SCALE=1.2 make ui-design-guide ARGS='--capture-review --output output/ui_design/review/guide'
MOJIVE_UI_SCALE=1.2 make ui-native-capture ARGS='--asset joint_gizmo --review --output output/ui_design/review/workspace'
```

Interactive startup copies the same saved or live specification used by the native
workbench, including component and group overrides. Typography preferences are copied
as well. Editing a guide specimen changes its local copy; it never overwrites the running
workspace's preferences or design draft. `--defaults` uses the shared factory specification.
`--spec` previews an explicit file. Captures use factory defaults unless `--spec` is supplied,
so repeated visual checks do not depend on local workstation settings.
`--capture-alignment` exercises the production Inspector, grouped panels, all six Settings
categories, Pose, status readouts and menus. It compares section text origins with the owned
body column and saves actual geometry with the images. Use `--spec` to check the current
saved or live specification and `BACKEND=bgfx` for the other renderer. This focused capture
exits after verification and does not change the saved specification.
`--capture-review` focuses on status phrases, scene counts, hover help, geometry friction,
and the compact color editor. The matching production `ui-native-capture --review` scenario
uses `joint_gizmo` and also records timeline samples, loop endpoints and the playhead through
real Session commands. It saves full captures, panel crops and measured assertions. Production
captures use the current workspace specification; set `MOJIVE_DESIGN_SPEC` to an explicit file
to compare factory and saved profiles. Run each target separately with `BACKEND=bgfx` for native
renderer coverage.
Version 1 files saved before the floating and popup surface colors were added load with
defaults for those two tokens. Reading preserves all existing overrides and does not rewrite
the file; other missing tokens or invalid values still fail validation.

The guide defaults to Simplified Chinese; its header switches between Chinese and English.
Page names, explanations, group editors and native choices switch together. Resource names,
font names, units and API identifiers keep their original spelling. The native window displays
a completed first page when opened.

`DesignRuntime` applies the same fonts, colors, dimensions and interaction theme in both
entry points. The guide contains no second component appearance preset. Factory defaults
use portable MiSans Regular and Semibold for Latin and Chinese, with a separate numeric face.
The original font files, checksums, attribution and Xiaomi license ship under
`python/text/resources/fonts/`. Existing saved font choices remain authoritative. The palette
page and exported manifest show the active workspace colors and tokens. The default body
size is 14 points, heading size 16, control height 28, and panel insets 12 horizontally and 10
vertically. The guide's explanatory headings and navigation have their own page layout.

The Scene, Inspector, Pose, Control, Assets, Cameras, Sensors, Layers, Timeline, Output,
Statistics, Curves and Settings pages host the production panel presenters in an isolated
example session backed by the bundled `joint_types` MuJoCo scene. `make setup` installs
the required MuJoCo package; the guide writes its modified sample XML into its output
directory. Viewport, menu and status specimens share the native controls, layout,
font resolution, painters and hit regions. The session provides local selection and control
interactions without modifying the user's open scene. Native panel headers, toolbars, lists,
property sections and menus retain their own semantic group scopes.

Settings uses one separator below the window title. Category headings are separated from
content by spacing, without an additional rule. Section separators occur between content
groups; the first section has no leading separator. These rules are shared by the native
Settings window and guide specimens. Menu hosts reserve their measured native bar height,
and standalone navigation specimens expand to their full row height; clipping is owned by
the containing scroll region.

**Viewer layout** independently adjusts content insets for guide navigation buttons,
header actions, parameter buttons, parameter fields and content panes. Navigation buttons
default to 12-point horizontal and 6-point vertical insets; pane content uses 24 points.
These controls do not write sample-group overrides. **Group laboratory** edits a declared
local owner's width, height, fonts, icon colors and content insets beside an unchanged
neighboring group. The comparison and parameter regions scroll independently; reserved
scrollbar gutters keep sample widths stable. Reset restores that group's initial snapshot.

**Icons** inventories the adopted workspace family and retained input symbols. The family
has 120 SVG entries on one 24-unit grid with a 1.5-unit round stroke. Authored domain and
transport glyphs are kept under `native/svg/mojive/`; upstream sources and licenses are
recorded in `native/svg/sources.json`. Context aliases use the same compiled master, including
viewport and timeline actions. Mature mouse silhouettes and diagnostic symbols retain
their reviewed painters. The earlier G3 icon exploration remains available in
`make ui-feasibility` as a separate candidate library.

Family filters and name search select icon specimens; captions share the glyph's horizontal
center. The detail plate shows 14/16/20/24/32-point sizes and default, hovered, active and
disabled states. Optical placement is authored on the canonical grid, not recomputed from
each glyph's bounding box. `design-system.json` records the complete inventory and owner.

Frequent actions use icons with short delayed hints. Output retains its diagnostic icons,
count chips, search, selection, copying, context menus and clearing. Status values retain
localized labels, tabular digit cells, fixed decimal and unit anchors, independent text
insets and separate divider clearances. Mouse-button hints preserve their custom silhouettes;
keyboard modifiers use a dedicated monospaced badge style. Inspector numeric fields use the
production responsive XYZ layout, so numeric magnitude does not change the chosen layout.

Adjacent nested corners share the same resolved inset: outer radius equals inner radius plus
inset, with the inner radius clamped to zero for a square offset. Factory menus use 6-point
insets around 6-point row corners; choices and form/color popups use 8-point insets and
14-point outer corners. Floating Timeline and Settings use a 14 / 8 / 4 corner chain with 6 / 4 insets.
Settings title actions add a native-quantized 4-point inset to the host content inset; the close
plate has equal top/right clearance and a radius derived from the host's outer corner.
Search fields retain their capsule silhouette independently of their popup container.
Fractional scales resolve native insets first and derive the nested radii afterward. The
navigation rail's outer target uses native window points even when authored, independently of
UI zoom. Its default is 16; its contained button radius follows the actual equal left/bottom inset.
The Camera Preview uses 14 / 6 corners
with equal 8-point edge clearances. Manual dragging retains the pointer-to-panel offset and
respects all four viewport margins. Explicit parent radii constrain local child corners.
Their painters and popup lifetimes are shared. Switches retain
a neutral light thumb on the accent track through hover and press. G3 is disabled by default;
its optional smoothing values remain 0.618 for rounded rectangles and 0.382 for capsules.
Filter glyphs center on the capsule's leading end circle; explicit group padding can override
that placement, and the icon-to-count gap is adjustable. Nested radii and insets remain an
optical design relationship; arbitrary G3 offsets are not identical mathematical curves.

Guide text describes component roles, geometry and behavior in formal Chinese and English.
New specimens must use the production controls and preserve independent groups, explicit
text bounds, stable value slots, separate padding and gaps, and proportional icon scaling.
`--capture` records pages, group samples, popup hover states, icon families, Switch hover,
output search, playback and exact-value hints through the same Make target.

**Export draft** writes `proposed-spec.json`, `design-system.json` and the guide's separate
`viewer-layout.json` under `output/ui_design/guide/`. Grayscale changes the preview only;
the exported specification retains its authored colors. To inspect an exported local edit
in the complete production application, load that file explicitly:

```sh
make ui-native-design ARGS='--spec output/ui_design/guide/proposed-spec.json'
```

References include [Apple's Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/),
[Refactoring UI](https://refactoringui.com/),
[LovroPodobnik's skill](https://github.com/LovroPodobnik/refactoring-ui-skill) and
[s0xDk's skill](https://github.com/s0xDk/refactoring-ui-skill). Their hierarchy and spacing
principles inform the shared native controls.

## Component appearance

Use **部件 → 文字 / 图标** to change a component class or a semantic group. Text
has independent font family, size and color; icons have independent size and
color. The activity rail, viewport camera toolbar, transform toolbox, viewport
transport, timeline toolbar, panel headings and tree rows each own their appearance.
Alt-click a control to select its nearest independent semantic group, such as
settings navigation or a viewport toolbar. **关联分组** keeps its broader groups
available; shared classes are listed in a separate collapsed section. Editing a generic
class affects every instance of that class. Group edits override generic button
defaults; a dedicated child class can retain its own explicit override. Scene and
pose filters have independent groups, with shared panel-filter defaults available
through the generic class.

Panel-owned variants separate headings, content, navigation, lists and action
groups across all built-in panels, including Help and Info. The six Settings
categories have separate page and section styles. `component-groups.json` names
each owner and its reused control definitions; production drawing enters that
owner's scope and restores it afterward. Local overrides inherit shared defaults
without modifying another panel. Native docking tabs have their own
**停靠面板标签导航** group, separate from Inspector tabs and the collapsed bottom bar.

Alt-click a button to select its group's control style, such as **设置导航栏 · 文字按钮**,
**场景 · 场景树 · 树节点按钮**, or **检查器 · 标签导航 · 标签页按钮**. These styles own the
buttons' width mode, width, height, text padding, icon spacing and appearance.
Container padding and row spacing remain in the enclosing group. Control
measurement and painting resolve the same nearest group, including segmented
buttons in Scene, Cameras, Settings and Plot. The workbench lists these local styles
separately from the collapsed **全局控件类（影响所有分组）** section; choosing a shared
class there explicitly changes its defaults across groups.

In **资产 · 材质卡片**, adjust the color swatch size, swatch-to-caption gap and
four content insets. The default insets are 12 points horizontally and 8 vertically.
**资产 · 材质列表** separately owns list padding, card row spacing and the gap above
Apply material. Card and list spacing do not change other panels.

**视口播放条** and **变换工具浮栏** each expose a **分隔线** section for thickness,
endpoint inset, clearance on both sides and color. Zero thickness hides the line
and restores ordinary item spacing. Tool-column measurement reserves the same
divider space in vertical, horizontal and wrapped layouts. Scene filter counts
use their group's font for both measurement and painting, including monospaced fonts.

In **设置导航栏 → 导航栏宽度**, set the complete navigation column width directly.
Zero uses the automatic width or **设置整体布局 → 导航栏默认宽度**. Local navigation
widths apply to both the side column and the narrow window's top navigation.
**设置整体布局 → 设置窗口宽度** instead resizes the complete floating Settings
window, preserving its height. Zero restores the manual width from before the
override; available viewport space limits both window and navigation widths.

**共享** provides application defaults and **模块** narrows defaults to a panel family.
Resetting a component field removes its override. `inherit` preserves the inherited
font or semantic color, including active, warning and disabled states. Explicit
colors accept `#RRGGBB` or `#RRGGBBAA` and can be entered as text or chosen with
the adjacent color swatch. Font choices are the system sans family, bundled MiSans
and the current numeric monospace family. Portable MiSans contains both Latin and Chinese;
system fonts retain their configured Chinese fallback.
Numeric and output font preferences supply defaults when no class or group size
has been specified. Font families are installed once per window and cached;
changing a color or font size does not rebuild the atlas.

Status telemetry reserves stable text boxes, including the separate physics and
render rates. Its labels and units have separate anchors, and digits use equal
cells within the selected font. Metrics default to 4 points on each divider side; the step/time
readout has no keycap background and reserves a budget for its selected mode.
Steps promote through `k`, `M`, `B` and larger units without displaying `1000k`.
Hover reveals the exact step/time value, right-click copies it, and left-click
switches the mode. Timestep and rate units compact independently.
In **状态栏运行指标组 → 数字排版**, adjust the label-to-value and value-to-unit
gaps separately. Timestep, physics rate, step count, elapsed time and render rate
each have an independent value-width control in logical pixels; `0` selects a
font-based budget. Timestep defaults to one decimal place, physics rate reserves
two integer digits, and step count and render rate use a `999.9` budget. Narrower
widths reduce display precision or promote the unit before showing an ellipsis;
a time value that cannot fit displays a single ellipsis. Hovering and copying
retain the complete value. Value changes preserve decimal and unit anchors;
changing these settings updates the layout. In **状态栏运行指标组 → 悬停提示**,
adjust horizontal padding, vertical padding and row spacing independently. Their
defaults are 10, 8 and 6 logical pixels; they apply to metric tooltips only.
Alt-click an individual readout to select **状态栏 · 渲染后端 / 仿真步数 / 仿真时间 /
仿真步长 / 物理频率 / 渲染帧率**. Each owns its font and color, text or numeric
width, text insets and separate left/right divider gaps. Backend names such as
OpenGL use the same independent controls. **文字右内距** measures from the unit's
visible end to its item border; **右分隔线间距** measures from that border to the
divider. The next item's **左分隔线间距** controls the other side independently.
These gaps start at the visible line edges; line thickness occupies its own space.
The common **分隔线单侧默认间距** defaults to 4 logical pixels per side. The first
left gap and last right gap instead control the group's outer edges, defaulting
to zero. Units align at their right edge: SI prefixes reserve space before Hz or
FPS, leaving no hidden reservation after them. Decimal and unit-end anchors stay
fixed while values change.
Inspector vector breakpoints depend on available width, style and
format budgets. Changing values does not switch between inline, label-above and
stacked layouts. Numeric display can reduce precision or use scientific notation
inside its existing box; browsing a field does not round its stored value.

Clicking a top menu activates the menu group. Hovering another visible top menu
then switches the popup; Escape, an outside click or a completed action ends that
mode. Ordinary hover over an inactive menu bar does not open a popup.

## Capture and verification

```sh
# Capture production panels, menus and the RGBA color editor.
make ui-native-capture
make ui-native-capture BACKEND=bgfx
# Capture reference typography and icon specimens.
make ui-design-reference ARGS=--menus
make ui-design-reference ARGS=--icons
make ui-design-reference ARGS=--typography
make ui-design-reference ARGS=--appearance
make ui-design-reference ARGS='--portable-fonts --capture-only'
make ui-design-native-check
# Verify scene interactions, 2D handles and settings; capture representative states.
make ui-design-gizmos
make ui-design-gizmos BACKEND=bgfx
# Verify scaled panels, scene selection, numeric edits, undo and document actions.
make ui-design-runtime
make ui-design-runtime BACKEND=bgfx
# Compare warmed default/custom appearance frames and capture both workspaces.
make ui-design-profile
make ui-design-profile BACKEND=bgfx
# Audit every catalog field through native editor input and production draw data.
make ui-native-audit
make ui-native-audit BACKEND=bgfx UI_AUDIT_SCALE=2
# Verify selection and parent groups on every visible production instance.
make ui-native-pick-check
make ui-native-pick-check BACKEND=bgfx UI_AUDIT_SCALE=2
```

Parameter audits write per-field results, restoration checks and failure captures to
`output/design-parameter-audit/<backend>-<scale>x/`. Picking results are stored in
`output/design-picking-audit/<backend>-<scale>x/`. These commands create isolated
scene, settings and specification files. `ARGS='--resume'` resumes an interrupted
parameter audit; `ARGS='--cases switch.columnGap,transport.marginLeft'` selects
specific checks. `--retry` accepts comma-separated result statuses.

The audit opens production panels, menus and dialogs, edits and resets each field
through the native input queue, and compares submitted geometry and draw vertices.
Explicit child overrides are temporarily reset when testing parent inheritance;
the report records those supporting conditions and restores the original preset.
Conditional fields retain their actual applicability, such as icon gaps on rows
that have icons and vertical spacing on groups that wrap to multiple rows.

Screenshots, reports, saved scenes and layout files go to
`output/ui_design/native/<backend>/`. Direct gallery invocation uses
`python -m mojive.tools.native_workspace --capture` and defaults to
`output/native-workspace/`. `make ui-native-capture ARGS='--output output/my_review'`
selects another output directory. Production sources and icon data live under `python/ui/workspace/`;
no file in `output/` is required at startup.

## Production module boundaries

| Module | Responsibility |
|---|---|
| `ui/workspace/design.py` | Shared field sizes, spacing and rail geometry |
| `ui/workspace/widgets.py` | Basic controls and per-frame hit/edit state |
| `ui/workspace/numeric_fields.py` | Numeric gestures, responsive groups and value rails |
| `ui/workspace/widget_style.py`, `widget_icons.py` | Font/theme setup and icon drawing |
| `ui/workspace/property_style.py` | Section headings, labels and switch rows shared by panels |
| `ui/panels/settings.py` | Settings navigation, draft state and presentation hooks |
| `ui/panels/settings_pages/` | Domain settings and writes through `PanelContext` callbacks |
| `ui/workspace/settings_panel.py` | Native presentation of the shared settings pages |
| `ui/workspace/entities.py`, `navigation.py`, `pose_panel.py` | Inspector projection, browsing and joint editing respectively |

The `UI` instance owns control state; numeric and icon modules do not create a second
control registry. Settings pages reuse the same actions in the classic and native
presentations. Preview values remain separate from persistent preferences, and
Escape restores the value at gesture activation.

Pose submits visible rows plus the active drag owner. Session owns the joint index;
navigation caches only browsing results and invalidates them when metadata changes.
Free and ball joints in the Pose list have compact XYZ summaries and independent
disclosure controls. Selecting a target expands its editors; explicit disclosure state
remains local to that window. Summary columns use a fixed font budget and reflow with
the panel width, while expanded fields retain their original precision and transactions.
Inspector fields remain directly editable, and hinge and slide controls retain their
existing layout. Read-only adapters still support browsing and disclosure.
Inspector keeps load-time transform and RGBA defaults for Restore and builds full
presentation data only for selected or pinned entities.

## Shared panel presentation

Camera sections share the Inspector heading, label and row-spacing controls. The
slider, numeric field and unit button use one native compound control; angle units
switch between degrees and radians without changing the camera pose. Narrow rows
place the label above the control and stack the slider when necessary. The capture
gallery includes camera parameters, radian units and the model-camera source.

Assets, Sensors, Plot and all six Settings pages reuse the same native property
presentation. Compound numeric fields, color pickers, section headings and
responsive label columns are shared; the registered panels retain asset commands,
sensor samples, telemetry history and settings ownership. Control retains actuator
units and equality-constraint switches. Output and Statistics use their adopted
list spacing. The gallery captures every Settings page, including scrolled content,
model asset inventory, material controls and both Plot sources.

Model asset imports share one menu, with texture and height-field parameters in
Import options. Output distinguishes an empty history from unmatched filters and
offers Clear filters without deleting messages. Status time supports compact seconds
and clock formats, with fixed numeric budgets and exact-value hints.

Docked content, floating windows and temporary popups have separate surface colors.
Global and module palettes retain their overrides; Panel, Menu and Select popup can
override their own surface color. Settings use headings and spacing between sections,
with a single rule below the window header. Rendering settings also provide a
Low contrast editor ground option: infinite planes use a neutral display surface,
while scene materials, textures, collision colors and physics state remain unchanged.

The sidebar shows one active view with a title and actions on the same row. More
contains view actions and controls for choosing header buttons. Scene and Settings
share four geometry modes: Default, Visual, Collision and Both. Default retains the
scene source's visibility groups; explicit modes filter geometry by its visual or
collision role. The selector displays the current renderer state and uses icons
without labels in narrow panels. Scene also includes icon-and-count type filters, semantic icons and
an expandable hierarchy. Selected rows use one background; hover does not add a
second inner rectangle. Single-click selects; double-click focuses the camera.
Pose lists the actual joints and supports the same selection and focus actions.
Hinge and slide joints expose scalar values in degrees and meters. Ball joints
expose numerical XYZ Euler rotation in degrees relative to the joint frame. Free
joints expose position in meters and XYZ Euler rotation in degrees in the world
frame. The Pose view and Inspector Joint tab use the same fields and viewport
handles. Each completed numeric drag or text edit commits one undo operation;
Escape cancels the gesture.
Output, Timeline and Statistics share the bottom tab group. Output retains search,
severity filters, selection, copy, file actions, multiline messages and optional
auto-scroll. The status bar shows simulation time or steps, backend, frame rate,
recording controls and contextual pointer hints.

The appearance review uses 28-point controls, compact joined axis badges and value
rails. Transform and RGBA fields support dragging, Ctrl-click or double-click text
entry, and Escape cancellation. Axis badges highlight on hover and press; clicking
resets that component. A gesture commits one undo operation; scale edits remain
pending until Apply. Inspector uses one entity header with pin and close actions,
and shares the editor's camera, light, model and physics fields. These fields use
the same native sections, numeric fields, axis badges and switches as Body. Property
tables reserve 28-point fields and at least eight points between adjacent rows;
scalar fields, dropdowns and vectors in a section share one control column. Labels
use up to 120 points with a 12-point gap and shrink before controls wrap. Long
labels use an ellipsis and retain their complete text on hover. Labels move above
the controls when fewer than 48 points remain after reserving the numeric fields.
All vectors in a transform group use one layout. Switches use the shared track, thumb and hit-region dimensions, centered with the row's label.
The active thumb uses a neutral light ink against the accent track.
Body, camera, light, physics and joint vectors use the same grid.
XYZ groups use three equal columns when common numeric precision fits, and one
column at compact widths. Color channels use one complete RGB/RGBA row or one channel
per row; partial rows are avoided. Units appear in field labels. Numeric values keep
their authored decimal precision, including trailing zeros. Plain inputs reserve eight
points of horizontal padding; axis fields use their independently adjustable three-point
insets. The grid measures a static numeric domain before choosing its column count.
Values exceeding that budget use complete scientific notation, then an ellipsis when
even the exponent cannot fit, with the full value on hover. Text editing retains the
native caret, horizontal scrolling and original precision.
Color popups provide Hex, RGBA 0–255 and RGBA 0–1 entry, alpha, Copy and Paste.
The color editor owns one adjustable inset around its content and footer. Settings
navigation similarly insets its selected rows inside the navigation background.
Invalid entry retains the previous color and displays its error. Done commits one
undo operation; Cancel or Escape restores the color from before the popup opened.
Joint type, Qpos address and degrees of freedom share a compact metadata row.
Advanced Joint properties start collapsed and remain available through their section
header. Object and joint names retain their source spelling in both languages.
Camera Preview displays the selected camera's live image inside the viewport.
Its header supports dragging, Pin, Lock and Close, and its placement leaves the
viewport controls accessible. The image retains a 16:9 aspect ratio with six-point
side and bottom insets inside a rounded frame. WASD moves the editor camera
horizontally and Q/E moves it vertically while the pointer is over the viewport. G and R select Move
and Rotate; B switches world/body frame. Shift enables temporary snapping.
Settings uses one
compact header and a category sidebar at standard widths, with top navigation in
narrow windows. Header menus and viewport tools move into More when space is
limited. The review captures both languages and checks category and editable-field
bounds under `appearance/`. It also captures `hinge_body` Physics and
`hinge_limited` Joint at compact, standard and wide Inspector sizes, checking
equal inertial-vector widths and consistent label placement.

The default workspace and guide use the bundled MiSans Regular and Semibold faces for
Latin and Chinese. Numeric fields and status keycaps use the separately resolved monospace
face, with bundled Inconsolata as a fallback. The System font option remains available;
font discovery does not download or distribute system fonts. `--portable-fonts` explicitly
selects the bundled faces for standalone reference captures. `--typography` captures
ordinary controls and a mixed Latin/Chinese sheet. Current body and numeric defaults are
14 points, and panel titles are 16 points. Each loaded face's measured cap-height center
defines its native input baseline. Static numeric values and editable text share the line origin.

Native icons use the logical canvas sizes below. The 24-unit masters preserve authored
proportions, optical placement and one 1.5-unit round stroke. Viewport tools, playback,
scene rows and panel actions use the same compiled drawing path. Domain symbols use
a tilted rounded object with a short rotation arrow for Pose, a mechanical pivot for hinge joints, opposing arrows for slide
joints and rotation arcs for ball joints. Runtime statistics use a performance gauge.
Sensors use a sensing point between reception arcs; Curves retains the distinct waveform.
Snap uses a filled horseshoe with separated poles. Bodies use a framed center mark;
state snapshots use a card with a diamond. Content visualization uses a body, reference plane and normal
arrow, independently of the layer-stack symbol. Projection frames have rounded inner and outer
corners.
Default uses a preset flag, Visual a palette, Collision two contacting body outlines,
and Both overlapping circles. Chinese
labels use “叠加” for Both with a tooltip describing simultaneous visual and collision geometry.

General symbols derive from pinned Lucide masters. Mojive-owned SVGs define the domain,
projection and transport forms. Source assets, licenses and mappings are retained in
[native/svg](native/svg/sources.json). Compilation applies the common family stroke;
geometry is authored on the canonical grid instead of fitted independently at runtime.
Output filters and records call the same reviewed severity painter and palette.
Entity names use muted semantic hues; selection uses one background.
`--icons` also captures `icon-roles.png`.

| Role | Canonical canvas (pt) | Default slot (pt) | Use |
|---|---:|---:|---|
| Activity | 24 | 36 | Sidebar navigation |
| Title | 22 | 30 | Inspector entity and Settings title |
| Toolbar | 18 | 28 | Viewport and Timeline tools |
| Subtle | 16 | 28 | Inspector and preview actions, viewport playback |
| Inline | 14 | 28 | Labeled controls, hierarchy, joint names and track labels |
| Status | 12 | 20 | Compact status indicators |

`native/icons-source.json` contains the 24-unit SVG masters;
`python/ui/workspace/icons.json` contains their compiled runtime contours. After changing a
master, run `make ui-design-icons` and the geometry checks. The small malformed
rotation fixture in `tests/fixtures/ui_design/` preserves the previous compiler
failure without retaining historical prototype directories.

The native reference uses the `joint_types` MuJoCo workspace with an additional
authored camera. Its flat 2D handles include translation arrows, rotation rings,
dimension handles, joint ranges, endpoint controls and numeric drag feedback.
Selection, double-click focus, navigation, camera/light helpers, Inspector fields,
materials and undo/redo use the same scene state. Use the Scene Geom filter to
select a geometry and edit its dimensions. Joint-driven bodies use the Joint tab
or viewport handles for motion. Pending dimension edits have Apply and Discard
buttons in the viewport.

The native transport operates real simulation playback, stepping and take
recording; the Control panel writes actuator values. Timeline uses the editor's
model keyframes, recorded takes and complete scene snapshots. It supports capture,
selection, loading, keyframe editing, range playback, scrubbing, zoom, pan and
playhead follow. Use Add Keyframe in the model lane to capture a pose; the initial
workspace has no model keyframes. Recorded takes can be exported as video.
The toolbar measures transport, time, Record, take selection, Snapshot, Video,
range Start and End, Loop, all three Follow modes and Fit. These common actions
share one row when the panel is wide enough and wrap into groups as it narrows.
Timeline Settings contains Pause at last frame, recording start delay and end
hold in a compact form with left-aligned labels and a pause switch. Labeled icons
use the 13-point body-text size; standalone tools retain their larger canvases.
The Pose entry uses a tilted object outline with a separate short rotation arrow. Recorded takes show connected
sample coverage with one name; frame boundaries
appear at readable zoom levels. Deleting frames leaves their original time cells
empty at every zoom level. Selection uses the retained cells and preserves these
gaps. Saved-file receipts fit their content within a compact viewport overlay,
and Camera Preview titles have padding inside the draggable header. The three
lanes use solid diamond key markers and distinct snapshot markers. Selected keyframe name,
time and actions share a compact detail bar below the tracks.
Entity fields are projected from Session. File menu save/load uses the workspace
scene format and retains model keyframes.
Wireframe displays unlit mesh edges and retains solid object picking. The interactive
reference uses the window's VSync pacing without an additional per-frame sleep.

Open Settings with the header gear, Window menu, F9 or Ctrl+Comma. Its General page
switches between English and Simplified Chinese. Camera, Interaction, Rendering,
Recording and MuJoCo Visuals pages use shared settings actions within the new
design's navigation. The reference shortcuts are V for selection, G/R for
translation/rotation, B for body/world frame, Shift for temporary snapping and T for
Timeline. WASD and Q/E move the editor camera while the pointer is over the viewport;
active text inputs retain their keyboard events. Interaction captures go to
`output/ui_design/native/<backend>/interactions/`.

## Browser reference

```sh
make ui-design
make ui-design-check
```

Open `http://127.0.0.1:8768/examples/ui_design/web/`. Set `UI_DESIGN_PORT` to use
another port. The server binds to localhost and serves the repository root so
the import map can reach the tracked Three.js modules in `3rdparty/three.js/`.
No npm install or CDN is required. A browser with WebGL is required for the
viewport; Node.js is required only for `ui-design-check`.

The web reference supports scene selection, Inspector edits, pose keys, material
changes, undo/redo and local browser storage. Its panels are responsive browser
layouts; the native reference exercises real ImGui docking. `ui-design-check`
runs the native CPU checks and browser document-model tests. Browser source is
under `web/`; historical variants and generated comparison images are excluded.
