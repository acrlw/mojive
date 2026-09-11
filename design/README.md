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
- `Geometry`: focused construction and state studies for playback, tools, icon concepts, hints,
  transform gizmos, joint gizmos, helpers, status, settings, panels, and lower workspaces.

The Playback, Tools, and Hints tabs include a `Live component experiment` panel. Changes update the
specimen immediately. `Copy current values` copies the relevant `OverlayGeometry` fields for review;
`Reset production defaults` restores values from `python/src/mojive/ui/viewport_widgets.py`. Probe changes
remain local until they are deliberately implemented in production.

The `Icon library` tab is a concept-only review surface. Its Overview and family tabs compare
Viewport tools, Viewport playback, Keyframe transport, Keyframes actions, Panels, Scene helpers,
and Status & input candidates on one
24-unit grid at 14, 24, 56, and 112 pt. The orange circle is the complete 24-unit circular placement
boundary, matching Diagnostics. The header keeps an independent `Glyph padding` value for each
actual component group; the Status group also exposes mouse width. Family sheets,
capsule specimens, and the whole-UI preview resolve the values for the component they show. Each
detail row reports circular padding, its declared placement anchor, and one secondary diagnostic.
Previous, Next, More, Keyframe Transport First/Last, Panel Right/Down, and Tool Snap use their
complete visible box; Playback and Keyframe Transport Reset use the center of their circular ring;
remaining candidates use their sampled minimum enclosing circle. Rotate stays frame-aligned, and
the reviewed Output Info, Warning, and Error icons stay on their production painter and ignore the
padding control. Playback Play remains equilateral; Previous and Next use their original 90-degree
chevrons, Pause keeps its original twin bars, and each More is a 90-degree counterclockwise rotation
of a slightly shorter Previous at the same fitted scale. Playback and Keyframe Transport Record are
both two percent smaller than Stop. First and Last use bars exactly as tall as their rounded
triangles. Status mouse hints continue to use the original production painter, and all three states
share the Left candidate's fitted outer-shell scale.
A neutral square with the orange circle's diameter makes the shared slot center visible at every
review size.

Use the always-visible `Icon Library preview` menu-bar switch, or `Probe > Preview Icon Library`,
to substitute the candidates into the current Workspace, Panels, Geometry, and Redesign layouts.
This keeps the real control sizes, baselines, hover states, and panel spacing, so an icon can be
judged where it will be used. The command-line equivalent is `--preview-icon-library`. Generate the
family sheets and representative context captures with:

```bash
make ui-icon-concepts
```

The default Workspace follows production behavior: regular tool hints live in the status bar,
Type-value appears only in the delayed handle-hover state, and joint limit labels are hidden until
their ticks are hovered. Optional scene-surface hints and construction overlays can still be enabled
from the `Probe` menu for experiments.

## Scale and layout checks

All vector paths, strokes, controls, and text follow `ui_scale`. At scales above 1×, the capture
window grows so one image still shows the whole concept: the geometry canvas, the live experiment
controls, and the workspace docks. Tab rows wrap instead of running past the panel edge, matching
how production panels reflow their own rows. Workspace, panel, and geometry pages also keep a
minimum logical canvas and expose scrollbars instead of compressing unrelated components until text
overlaps.

Examples:

```bash
make ui-feasibility ARGS="--ui-scale 1.5 --page workspace"
make ui-feasibility ARGS="--ui-scale 2.5 --page geometry --geometry-tab diagnostics"
make ui-feasibility ARGS="--ui-scale 4 --page geometry --geometry-tab helpers"
```

`Esc` closes an open value editor first, then closes the probe. `Ctrl+C` in the terminal also exits.
Interactive mode is paced at 30 FPS by default; pass `--fps 60` when motion needs closer inspection.

On a machine with no GPU device nodes, Mesa's surfaceless EGL platform runs the same captures
through llvmpipe:

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 MOJIVE_GL=egl \
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab diagnostics \
  -o output/ui-diagnostics.png
```

`output/diagnostics/probe_render.sh` wraps that environment, and
`output/diagnostics/severity_preview.py` rasterizes the severity glyph meshes alone at 8× so their
small-size geometry can be reviewed without any GL context.

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
.venv/bin/python design/tools/render_ui_feasibility.py --page geometry --geometry-tab icons --icon-group viewport-tools -o output/ui-icon-concepts/viewport-tools.png
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

Every severity glyph is one box with one stroke weight: circles reach 0.94× the requested size, and
the warning triangle spans the same box while keeping its sides equal, so all three frames carry the
same optical weight. Info and warning draw one shared exclamation mark: the dot takes half of the
mark height budget `0.30`, the gap `0.36` and the stem `0.42`, and the dot is the same width in both
frames. The triangle only shortens the stem when its taper cannot clear the full mark, and the mark
sits centered on the triangle's own box center, exactly like the circle frames center theirs.

Stroke, dot, gap and stem each carry an absolute floor, because Output draws the glyph at the font
size and a purely proportional mark fuses into a blob below roughly 16 px. The error cross keeps a
measured margin to the ring, because a diagonal arm reaches further than its bounding box suggests.

The recording-options chevron shares the playback capsule with filled symbols. Enlarging it means
lengthening its two capsule arms, not thickening them: `draw_expand_glyph` multiplies both the arm
stroke and the path by its glyph scale and then scales the path back out, so a scale pair cannot grow
the envelope at a constant stroke. `RECORDING_OPTIONS_ENVELOPE_SCALE` lengthens the arms instead,
which holds the arm at the reset arrow's 1.46 px while the drawn chevron grows from 9.64 x 5.63 to
13.64 x 7.63 px and moves from 4.02 px to 1.78 px inside the state ring (the reset arrow reaches
0.47 px). `make ui-feasibility ARGS="--page geometry --geometry-tab playback"` shows the balance.

The Diagnostics page also compares normal, hover and pressed slider colors, plus joined rad/deg
fields at wide and narrow widths. The unit button converts only presentation and input; source
state remains in radians. Control rows use the production renderer and mapped right-click reset,
without a separate reset button. The Panels/Workspace Keyframes specimen uses a real local
Session with independent transient scene snapshots, rather than a painted timeline mockup.
