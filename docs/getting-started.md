# Getting started

## Install from a source checkout

Source builds require Python 3.11 or newer, uv, CMake, Ninja and a C++20 toolchain:

```bash
git clone --recurse-submodules https://github.com/acrlw/mojive.git
cd mojive
make setup
```

Optional dependencies are grouped into the following extras:

- `mujoco` loads MJCF/URDF, runs simulation, and enables the compatible `Renderer` API;
- `dev` installs pytest and Ruff; and
- `docs` installs the documentation build tools.

`make setup` installs the development and backend dependencies, builds the
[custom ImGui bindings](how-to/ui-corners.md), and registers the native compiler with the
editable Python package. Use `uv run --no-sync` after setup to preserve the locally built
ImGui wheel and native registration. If you reinstall the editable package, run
`make native-editable` before opening the viewer. Set `MOJIVE_NATIVE_BUILD` to load the native
extension from a different build directory.

## Open the editor

To open an empty workspace:

```bash
uv run --no-sync mojive editor
```

Use **File > Add Model...** or drop MJCF/URDF files into the viewport. The first dropped file opens
as the primary model; additional files are added to the composition. Pause simulation before
editing model topology or root transforms.

You can also open a model directly:

```bash
uv run --no-sync mojive editor assets/joint_types.xml
```

The editor saves `.mojive.json` workspaces. These retain model references, root transforms,
resource directories, edited model XML, and Mojive-authored entities. **File > Save As** can export
a portable MJCF/XML model instead.

## Open the viewer

The bundled `joint_types` scene includes joints and actuators:

```bash
uv run --no-sync mojive view joint_types
```

Open **Joints**, **Inspector** and **Keyframes** from the Window menu. Drag their tabs to dock
them. The arrow in Output collapses the panel. Select a body in Hierarchy to inspect its
properties; press **G** or **R** to enable the position or rotation tool. Available controls
depend on the selected entity and the adapter's capabilities.

Use `view` for one model or bundled scene:

```bash
uv run --no-sync mojive view test_scene
uv run --no-sync mojive view path/to/model.xml --paused
uv run --no-sync mojive view path/to/model.urdf --paused
```

Bundled asset names do not need an extension. List them with:

```bash
uv run --no-sync mojive assets --quick
```

The viewer starts paused by default. Pass `--play` to start simulation immediately.

## Choose a render backend

OpenGL is the default for fast scene and editor validation; bgfx provides native rendering.
The same scene works with either renderer:

```bash
MOJIVE_RENDERER=opengl uv run --no-sync mojive view joint_types
make native-viewer SCENE=joint_types
```

The native target builds bgfx and its shaders, then starts the viewer with that renderer. See
[native backend setup](how-to/native-viewer.md) for scripts and installed packages.
`MOJIVE_RENDERER` selects rendering; `--adapter` selects scene/physics integration.
The compatible `-b/--backend` option also means adapter, not renderer.

To check that a renderer can create a window and render frames:

```bash
uv run --no-sync mojive doctor joint_types
```

For PowerShell launch commands and UI scaling, see
[Windows](how-to/windows.md).

## Configure the UI

Open **Edit > Settings...**, **Window > Settings**, or press `F9`. Settings is a dockable,
non-modal panel. It controls interaction, shortcuts, render flags, visual groups, debug views,
labels, frames, UI language, and helper visibility.

Camera preview is disabled by default. To display a camera's image in the viewport, select the
camera and enable **preview** in Inspector.

Mojive uses the display's reported scale by default. The following environment variables override
the scale, language, and Chinese font for one process:

```bash
MOJIVE_UI_SCALE=2 uv run --no-sync mojive editor
MOJIVE_LANGUAGE=zh_CN uv run --no-sync mojive editor
MOJIVE_CJK_FONT=/path/to/font.otf uv run --no-sync mojive editor
```

See the [configuration reference](reference/configuration.md) for all variables and persistence
paths.

## Render from Python

```python
import mujoco

from mojive import Renderer

model = mujoco.MjModel.from_xml_path("assets/test_scene.xml")
data = mujoco.MjData(model)

with Renderer(model, width=640, height=480) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=-1)
    rgb = renderer.render()
```

See the [MuJoCo rendering tutorial](tutorials/mujoco-rendering.md) for metric depth,
segmentation, multiple cameras, and renderer selection.

## Run a headless check

These commands inspect, audit, or render a model without opening an interactive viewer:

```bash
uv run --no-sync mojive inspect test_scene
uv run --no-sync mojive audit test_scene --strict
uv run --no-sync python examples/mujoco_render.py assets/test_scene.xml \
  --output output/examples/render
```

Repository contributors should run `make check`. The [testing guide](guides/testing.md) maps changes to focused targets.

## Next steps

- [Editor and MJCF](guides/editor-and-mjcf.md)
- [Examples and tutorials](guides/examples.md)
- [Programmatic scenes](tutorials/programmatic-scene.md)
- [Custom adapters](how-to/custom-adapter.md)
- [Remote viewing and replay](tutorials/remote-viewing.md)
- [API map](api/index.md)
