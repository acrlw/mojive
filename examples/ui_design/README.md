# UI design study

This standalone reference explores native docking, typography, menus and scene
authoring with Mojive's OpenGL and bgfx renderers. It keeps its own panel layout,
controls and visual style while using the production ViewerApp, Session and input
pipeline for scene operations.

Run from the repository root after `make setup`:

```sh
make ui-design-native ARGS=--interactive
make ui-design-native BACKEND=bgfx ARGS=--interactive
```

Close the window to end the interactive run. bgfx requires the
[native runtime and shaders](../../docs/how-to/native-viewer.md).
On an existing checkout, install the new development dependency without replacing
the locally built ImGui wheel: `uv pip install 'fonttools>=4.0'`.
Python dependencies and their resolved versions are declared in `pyproject.toml`
and `uv.lock`.

## Capture and verification

```sh
# Capture docking, floating, redocking, grouping and fresh-context layout restore.
make ui-design-native
make ui-design-native BACKEND=bgfx
# Capture expanded native menus and validate their geometry and actions.
make ui-design-native ARGS=--menus
# Render icons at their actual target sizes and typography specimens.
make ui-design-native ARGS=--icons
make ui-design-native ARGS=--typography
# Capture editor panels and every settings page in English, Chinese and a narrow window.
make ui-design-native ARGS=--appearance
# Exercise the portable font configuration on macOS too.
make ui-design-native ARGS='--portable-fonts --capture-only'
make ui-design-native-check
# Verify scene interactions, 2D handles and settings; capture representative states.
make ui-design-gizmos
make ui-design-gizmos BACKEND=bgfx
# Verify timeline, workspace and document actions at 2x scale in a 1440x900 window.
make ui-design-runtime
make ui-design-runtime BACKEND=bgfx
```

Screenshots, reports, saved scenes and layout files go to
`output/ui_design/native/<backend>/`. Direct module invocation also defaults to
`output/ui_design/native/`. `ARGS='--output output/my_review'` selects another
output directory. Sources and runtime data remain under `examples/ui_design/`;
no file in `output/` is required at startup.

The appearance review uses 28-point controls, aligned transform fields and value
rails with separate numeric entries. Settings uses a category sidebar at standard
widths and switches to top navigation in narrow windows. The review captures both
languages and checks category and editable-field bounds under `appearance/`.

The macOS reference uses installed SF Regular, SF Semibold and SF Mono Regular
named instances. Other platforms use Roboto from ImGui Bundle and Mojive's shared
monospace discovery policy, with bundled Inconsolata when no independent source
is available. Font discovery performs no downloads. `--portable-fonts` selects
that configuration on every platform. Font metrics differ, so screenshots from
different font configurations are not pixel-identical. macOS system fonts are
not distributed with this example.

`native/icons-source.json` contains the 24-unit SVG masters;
`native/icons.json` contains their compiled runtime contours. After changing a
master, run `make ui-design-icons` and the geometry checks. The small malformed
rotation fixture in `tests/fixtures/ui_design/` preserves the previous compiler
failure without retaining historical prototype directories.

The native reference uses the `joint_types` MuJoCo workspace with an additional
authored camera. Its flat 2D handles include translation arrows, rotation rings,
dimension handles, joint ranges, endpoint controls and numeric drag feedback.
Selection, double-click focus, navigation, camera/light helpers, Inspector fields,
materials and undo/redo use the same scene state. Select a geometry through the
Scene list's Geom filter to edit dimensions. Joint-driven bodies use the Joint tab
or viewport handles for motion. Pending dimension edits have Apply and Discard
buttons in the viewport.

The native transport operates real simulation playback, stepping and take
recording; the Control panel writes actuator values. The reference Timeline keeps
its pose-preset design and applies those values to the workspace's joints. Its
preset keys come from `native/document.json`; entity fields are projected from
Session. File menu save/load uses the workspace scene format.

Open Settings with the header gear, Window menu, F9 or Ctrl+Comma. Its General page
switches between English and Simplified Chinese. Camera, Interaction, Rendering,
Recording and MuJoCo Visuals pages use shared settings actions within the new
design's navigation. The reference shortcuts are V for selection, W/E/R for
translation/rotation/dimensions, B for body/world frame, S for snapping and T for
Timeline. Fly navigation uses I/K/J/L/U/O. Interaction captures go to
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
