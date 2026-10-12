# UI design study

The native design is now the default production Mojive workspace. `mojive view`,
`mojive editor`, and public viewer builders use `python/ui/workspace/`, with
Session commands and the existing ViewerApp frame pipeline retaining scene and
application ownership. `make ui-design-native` opens or captures that production UI.

The standalone source in this directory remains a comparison fixture for icon,
typography and prototype-workbench experiments. Use `make ui-design-reference`
for those galleries; the production runtime imports no example or output code.

Run from the repository root after `make setup`:

```sh
make ui-design-native ARGS=--interactive
make ui-design-native BACKEND=bgfx ARGS=--interactive
make ui-design-native ARGS='--interactive --panel Cameras'
```

Close the window to end the interactive run. bgfx requires the
[native runtime and shaders](../../docs/how-to/native-viewer.md).
On an existing checkout, install the new development dependency without replacing
the locally built ImGui wheel: `uv pip install 'fonttools>=4.0'`.
Python dependencies and their resolved versions are declared in `pyproject.toml`
and `uv.lock`.

## Capture and verification

```sh
# Capture production panels, menus and the RGBA color editor.
make ui-design-native
make ui-design-native BACKEND=bgfx
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
```

Screenshots, reports, saved scenes and layout files go to
`output/ui_design/native/<backend>/`. Direct module invocation also defaults to
`output/ui_design/native/`. `ARGS='--output output/my_review'` selects another
output directory. Production sources and icon data live under `python/ui/workspace/`;
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

The sidebar shows one active view with a title and actions on the same row. More
contains view actions and controls for choosing header buttons. Scene offers three
geometry modes, Visual, Collision and Both, in one exclusive-choice control and
starts in Visual. It also includes icon-and-count type filters, semantic icons and
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
All vectors in a transform group use one layout. Switches use a 32 × 18-point
track inside a 36 × 28-point hit region, centered with the row's label.
Body, camera, light, physics and joint vectors use the same grid.
XYZ groups use three equal columns when common numeric precision fits, and one
column at compact widths. RGBA groups use four, two or one equal columns.
Units appear in field labels. Numeric values reserve eight points of horizontal
padding. The grid measures common decimal values before choosing its column count;
values exceeding that space use an ellipsis with the full value on hover. Text editing
retains the native caret and horizontal scrolling; display truncation preserves
the stored value and edit precision.
Color popups provide Hex, RGBA 0–255 and RGBA 0–1 entry, alpha, Copy and Paste.
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

The macOS reference uses installed SF Regular, SF Semibold and SF Mono Regular
named instances. Other platforms use the installed Noto or Source Han Sans SC
family for both Latin and Chinese when available, with matching Regular and Bold
faces. Noto collections select the Simplified Chinese face. A synthesized heading
weight is used when a separate Bold face is unavailable. Numeric fields retain
Mojive's monospace discovery policy, with bundled Inconsolata as its fallback.
`--portable-fonts` uses bundled Roboto and the same weight-matched Chinese fallback
on every platform. Font discovery performs no downloads. macOS system fonts are
not distributed with this example. `--typography` captures both ordinary size
specimens and a mixed Latin/Chinese sheet. Body text, property labels and section
headings use a 13-point em size, with 15-point entity titles; FreeType height conversion preserves that size
for both Latin and Chinese. Numeric fields use the existing 13-point monospace
metrics.
Each loaded face's measured cap-height center defines its native input baseline.
Static numeric values and their editable text use the same line origin.

Native icons use these logical canvas sizes. The 24-unit masters share one scale
and anchor per role, retaining their original proportions and stroke weight.
Compact action icons preserve their natural size beside larger entity symbols.
Pose uses a standing figure. Hinge, slide, ball and free joints use rotation,
horizontal motion, 3D rotation and 3D motion symbols. Visual uses a palette,
Collision a wireframe box, and Both overlapping circles. Chinese labels use
“叠加” for Both with a tooltip describing simultaneous visual and collision geometry.
Perspective and Orthographic use converging and parallel projection planes and
retain their labels in one exclusive-choice control.
Geometry modes and joint motion icons use original SVG files from Lucide 1.55.0;
projection icons use Tabler 3.49.0. Source assets, licenses and mappings are retained
in [native/svg](native/svg/sources.json). Compilation normalizes the whole family's
stroke to 1.75 units on the 24-unit grid without changing path geometry.
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
`native/icons.json` contains their compiled runtime contours. After changing a
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
The Pose entry uses Lucide's connected waypoint nodes. Recorded takes show connected
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
