# Mojive UI design tools

`render_ui_feasibility.py` is Mojive's interactive UI concept workbench. Use it when component
geometry, responsive layout, or an interaction concept needs exploration. Routine fixes can use
the existing production UI and focused acceptance targets directly. The agent reviews generated
images and links useful results under `output/` for the user to view; user approval is required
only when explicitly requested. See the [verification guide](../docs/guides/testing.md).

The probe creates Mojive's real `Window` and ImGui context. It starts from the production theme and
reuses the production Draw2D paths for playback, tool, mouse-hint, projection, status, and object
gizmo graphics. Panel specimens use lightweight sample data so they remain safe and fast to edit.
Pillow only writes the OpenGL framebuffer to PNG.

## Interactive workflow

Start the probe with:

```bash
make ui-feasibility
```

Use the `Probe` menu to switch between:

- `Workspace`: the current default arrangement—Hierarchy, Viewport, Output, the
  Control/Joints/Camera dock, Inspector, and the persistent status bar.
- `Panels`: responsive specimens for Control, Joints, Camera, Inspector, Hierarchy, Assets, Stats,
  and Sensors.
- `Geometry`: focused construction and state studies for playback, tools, hints, transform gizmos,
  joint gizmos, helpers, status, settings, panels, and lower workspaces.

The Playback, Tools, and Hints tabs include a `Live component experiment` panel. Changes update the
specimen immediately. `Copy current values` copies the relevant `OverlayGeometry` fields for review;
`Reset production defaults` restores values from `python/src/mojive/ui/viewport_widgets.py`. Probe changes
remain local until they are deliberately implemented in production.

The default Workspace follows production behavior: regular tool hints live in the status bar,
Type-value appears only in the delayed handle-hover state, and joint limit labels are hidden until
their ticks are hovered. Optional scene-surface hints and construction overlays can still be enabled
from the `Probe` menu for experiments.

## Scale and layout checks

All vector paths, strokes, controls, and text follow `ui_scale`. At scales above 2×, the capture
window expands like the production UI runtime. Workspace, panel, and geometry pages also keep a
minimum logical canvas and expose scrollbars instead of compressing unrelated components until text
overlaps.

Examples:

```bash
make ui-feasibility ARGS="--ui-scale 1.5 --page workspace"
make ui-feasibility ARGS="--ui-scale 4 --page geometry --geometry-tab helpers"
```

`Esc` closes an open value editor first, then closes the probe. `Ctrl+C` in the terminal also exits.
Interactive mode is paced at 30 FPS by default; pass `--fps 60` when motion needs closer inspection.

## Deterministic captures

Generate the full acceptance gallery with:

```bash
make ui-gallery
```

The target captures every Geometry tab, the current Workspace, 4× Workspace/Panels/joint-helper
checks, and the production UI runtime gallery. Generated files stay under `output/`.

Individual captures are useful while iterating:

```bash
.venv/bin/python design/tools/render_ui_feasibility.py --page workspace -o output/ui-workspace.png
.venv/bin/python design/tools/render_ui_feasibility.py --page panels -o output/ui-panels.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab playback -o output/ui-playback.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab tools -o output/ui-tools.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab hints -o output/ui-hints.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab gizmos -o output/ui-gizmos.png
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab helpers -o output/ui-helpers.png
```

The feasibility probe validates UI composition and interaction concepts. Use `make ui-runtime` for
the production UI with real panel state, and the relevant GPU acceptance targets for 3D depth,
occlusion, picking, and renderer output.

## Vector icon sources

`icons/ui-icons.svg` contains the design-source paths. Compile supported SVG primitives into
deterministic Draw2D data with:

```bash
.venv/bin/python design/tools/compile_svg_icons.py
```

The generated module is written under `output/`; Mojive does not parse SVG or rasterize icons in the
frame loop.

`index.html` and `style.css` remain a browser-readable design history. They are not the production
theme or the current feature inventory.

## Diagnostic icons and value controls

Open the production severity paths and responsive rails in the workbench:

```bash
make ui-feasibility ARGS="--page geometry --geometry-tab diagnostics"
make ui-diagnostics
```

The levels use the local palette: Info `#8AB7C0` from
`kimi-design/mojive-ui-redesign.html`, Warning `#C9A15C` and Danger `#D06744` from
`design/index.html` section 5.1. The warning triangle uses the shared curvature-continuous
corner generator at smoothing 0.618. The gallery calls the same glyph and value-control
functions as Output, Control and Joints; the workspace Output specimen also uses the
production panel. Compact severity capsules precede Search and Clear and wrap on narrow panels.

The Diagnostics page also compares normal, hover and pressed slider colors, plus joined rad/deg
fields at wide and narrow widths. The unit button converts only presentation and input; source
state remains in radians. Control rows use the production renderer and mapped right-click reset,
without a separate reset button. The Panels/Workspace Keyframes specimen uses a real local
Session with independent transient scene snapshots, rather than a painted timeline mockup.
