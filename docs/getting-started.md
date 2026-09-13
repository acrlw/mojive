# Getting started

## Install from a source checkout

Source builds require Python 3.11 or newer, uv, CMake and a C++20 toolchain:

```bash
git clone --recurse-submodules https://github.com/acrlw/mojive.git
cd mojive
make setup
```

The extras are independent:

- `mujoco` loads MJCF/URDF, runs simulation, and enables the compatible `Renderer` API;
- `wgpu` enables the Metal/Vulkan/DX12 render backend;
- `dev` installs pytest and Ruff; and
- `docs` installs the strict documentation build.

`make setup` installs the development and backend dependencies, builds the
[custom ImGui bindings](how-to/ui-corners.md), and registers the native compiler with the
editable Python package. Use `uv run --no-sync` after setup to preserve the locally built
ImGui wheel and native registration. If you replace the editable install, run
`make native-editable` before opening the viewer. `MOJIVE_NATIVE_BUILD` can still select
an explicit alternative build for development and acceptance.

## Open the editor

Start with an empty workspace:

```bash
uv run --no-sync mojive editor
```

Use **File > Add Model...** or drop MJCF/URDF files into the viewport. The first dropped file opens
as the primary model; additional files are added to the composition. Pause simulation before
editing model topology or root transforms.

You can also open a model directly:

```bash
uv run --no-sync mojive editor assets/test_scene.xml
```

The editor saves `.mojive.json` workspaces. These retain model references, root transforms,
resource directories, edited model XML, and Mojive-authored entities. **File > Save As** can export
a portable MJCF/XML model instead.

## Open the viewer

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

OpenGL is the default. Select wgpu with an environment variable:

```bash
MOJIVE_BACKEND=wgpu uv run --no-sync mojive editor
MOJIVE_BACKEND=wgpu uv run --no-sync mojive view test_scene
```

Do not use `--backend wgpu`: `-b/--backend` selects the scene/physics adapter, while
`MOJIVE_BACKEND` selects rendering. The [CLI reference](reference/cli.md) documents both layers.

Validate a window and rendering path with:

```bash
uv run --no-sync mojive doctor test_scene
MOJIVE_BACKEND=wgpu uv run --no-sync mojive doctor test_scene
```

## Configure the UI

Open **Edit > Settings...**, **Window > Settings**, or press `F9`. Settings is a dockable,
non-modal panel. It controls interaction, shortcuts, render flags, visual groups, debug views,
labels, frames, UI language, and helper visibility.

Camera preview is disabled by default. Select a camera and enable **preview** in Inspector when a
live inset is useful.

Mojive normally follows the display content scale. These process overrides are useful for visual
acceptance or a misreported desktop scale:

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

Continue with the [MuJoCo rendering tutorial](tutorials/mujoco-rendering.md) for metric depth,
segmentation, multiple cameras, and wgpu.

## Run a headless check

These commands exercise useful non-interactive paths:

```bash
uv run --no-sync mojive inspect test_scene
uv run --no-sync mojive audit test_scene --strict
uv run python examples/mujoco_render.py assets/test_scene.xml \
  --output output/examples/render
```

Repository contributors should run `make check`. Rendering changes additionally run `make gpu`;
the [testing guide](guides/testing.md) maps changes to focused targets.

## Next steps

- [Editor and MJCF](guides/editor-and-mjcf.md)
- [Examples and tutorials](guides/examples.md)
- [Programmatic scenes](tutorials/programmatic-scene.md)
- [Custom adapters](how-to/custom-adapter.md)
- [Remote viewing and replay](tutorials/remote-viewing.md)
- [API map](api/index.md)
