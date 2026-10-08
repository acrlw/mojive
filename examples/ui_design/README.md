# UI design study

This standalone reference explores native docking, typography, menus and scene
authoring with Mojive's OpenGL and bgfx renderers. It has its own document and
layout; it does not replace the production editor or its Session.

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
# Exercise the portable font configuration on macOS too.
make ui-design-native ARGS='--portable-fonts --capture-only'
make ui-design-native-check
```

Screenshots, reports, saved scenes and layout files go to
`output/ui_design/native/<backend>/`. Direct module invocation also defaults to
`output/ui_design/native/`. `ARGS='--output output/my_review'` selects another
output directory. Sources and runtime data remain under `examples/ui_design/`;
no file in `output/` is required at startup.

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

The scene supports selection, visibility, camera navigation, transform gestures,
Inspector editing, material changes, local JSON save/load and undo/redo. Joint
motion and playback are previews. Physics execution, actuator write-back and
simulation recording require a physics adapter and are unavailable here.
