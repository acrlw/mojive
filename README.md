# Mojive

![Mojive with a jointed scene, joint controls, Inspector and Keyframes](docs/images/readme/hero.png)

**A backend-neutral 3D viewer, editor, and renderer for robotics and simulation.**

[Quick start](#quick-start) · [User guide](docs/getting-started.md) ·
[Python API](docs/api/index.md) · [Examples](examples/README.md)

Mojive stands for **Mo**del/**J**oint **I**nteractive **V**iewer & **E**ditor.
It supports MJCF and URDF models, scenes created in Python, custom simulations, and remote streams.
Rendering uses OpenGL, WebGPU (`wgpu`), or bgfx. The choice of renderer does not depend on the
physics adapter.

- **Scene editing:** model composition, geometry, materials, lights, cameras, Undo/Redo, and MJCF export.
- **Simulation:** joint and actuator controls, viewport manipulation, physical perturbation,
  sensors, contacts, tendons, and constraints.
- **Timeline:** model keyframes, scene snapshots, recorded takes, and video export.
- **Rendering:** RGB, metric depth, object IDs, and segmentation from Python or the CLI.
- **Integration:** custom adapters, passive viewing, remote publishing and replay, and local RPC.
- **Diagnostics:** 3D DebugDraw primitives and 2D paths, shapes, text, and images through Canvas2D.

| Joint tools | Rendering |
| :---: | :---: |
| ![Joint controls and a viewport rotation gizmo](docs/images/readme/joint-authoring.png) | ![Materials, lighting, shadows, transparency and debug overlays](docs/images/readme/rendering.png) |

## Quick start

Source builds require Python 3.11+, [uv](https://docs.astral.sh/uv/), CMake, Ninja, and a C++20
compiler. The default interactive renderer requires OpenGL 3.3 core.

```bash
git clone --recurse-submodules https://github.com/acrlw/mojive.git
cd mojive
make setup
uv run --no-sync mojive view joint_types
```

The viewer starts paused. Select a body in **Hierarchy** to inspect its properties in
**Inspector** or adjust its joints in **Joints**. Panels can be opened from the Window menu
and docked by dragging their tabs.

```bash
uv run --no-sync mojive editor                       # Empty workspace
uv run --no-sync mojive view path/to/model.xml        # MJCF
uv run --no-sync mojive view path/to/model.urdf        # URDF
uv run --no-sync mojive assets --quick                # Bundled scenes
```

`make setup` builds the modified ImGui bindings and native geometry extension. Subsequent commands
use `--no-sync` to avoid replacing these local builds. After reinstalling the editable package,
run `make native-editable` to register the extension again.
See [installation](docs/getting-started.md) for dependency details.

## Render backends

OpenGL is the default. To use another renderer:

```bash
MOJIVE_RENDERER=wgpu uv run --no-sync mojive view joint_types
make native-viewer SCENE=joint_types
```

`make native-viewer` builds bgfx and its shaders, then starts the viewer with that renderer.
See [native backend setup](docs/how-to/native-viewer.md) for `MOJIVE_RENDERER=bgfx` in scripts.
`--adapter` selects the scene/physics adapter; `-b/--backend` is its compatibility alias.

On Linux, offscreen OpenGL rendering uses EGL and requires a GPU driver and EGL installation.
Set `MOJIVE_GL=egl` to disable fallback to a hidden GLFW window, which requires a desktop session.
See [backend requirements](docs/reference/configuration.md#render-backend-requirements).

## Editor

The editor saves `.mojive.json` workspaces containing model references, edits, resource paths,
and entities created in Mojive. Use **File > Add Model** or drop MJCF/URDF files to add models.
**File > Save As > MuJoCo XML / MJCF** exports the composition and its required assets.

Inspector displays editable position, rotation, dimensions, material, light, and camera properties.
For geometry created in Mojive, **Apply** multiplies the dimensions by the edited scale and resets
the scale to `(1, 1, 1)`. MuJoCo model entities display read-only identity scale. Camera properties
include an optional preview. See the [editor guide](docs/guides/editor-and-mjcf.md) for topology
editing, keyframes, and export.

### Common controls

| Input | Action |
|---|---|
| `Space` | Play or pause |
| `Backspace` | Previous frame; hold to rewind |
| `G` / `R` | Position / rotation tool |
| `T` | Switch body/world frame |
| `Shift` while dragging | Snap |
| `Ctrl` + left/right drag | Translation/rotation perturbation |
| `F` | Frame the scene |
| `F9` | Open Settings |

Shortcuts are configurable in Settings. The dimensions tool has no default key binding.
UI scale and language default to the system settings; override them with `MOJIVE_UI_SCALE` and
`MOJIVE_LANGUAGE` (for example, `zh_CN`). See [configuration](docs/reference/configuration.md).

## Python

Create and render a scene without a physics engine or editor window:

```python
from mojive import Scene, SceneRenderer

scene = Scene()
scene.box(name="workpiece", position=(0, 0, 0.5), color=(0.2, 0.6, 0.9, 1))

with SceneRenderer(scene.source, width=640, height=480, renderer="opengl") as renderer:
    renderer.update(scene.frame)
    rgb = renderer.render()
```

For an interactive scene, use `build_scene(scene)` as a context manager and call `viewer.run()`.
The [programmatic scene guide](docs/tutorials/programmatic-scene.md) describes entity creation and
updates. [DebugDraw](docs/how-to/debug-draw.md) and [Canvas2D](docs/how-to/canvas2d.md) provide
diagnostic drawing functions.

`Renderer` provides `update_scene()` and `render()` methods similar to `mujoco.Renderer`:

```python
import mujoco
from mojive import Renderer

model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

with Renderer(model, width=640, height=480) as renderer:
    renderer.update_scene(data)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth_m = renderer.render()
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render()
```

It accepts free, named, fixed, and `MjvCamera` cameras, `MjvOption`, and reusable output arrays.
See [MuJoCo rendering](docs/tutorials/mujoco-rendering.md) for details.
[`launch_passive(model, data)`](docs/tutorials/passive-viewing.md) displays an existing simulation;
the calling script controls physics stepping.

To connect another simulation, implement `SceneAdapterBase`. Its `scene_source()` method returns
meshes, materials, and object identities; `frame()` returns the current state. `AdapterCaps`
declares which operations the adapter supports.
See the [adapter guide](docs/how-to/custom-adapter.md) and [runnable examples](examples/README.md).

## CLI and automation

Capture, record, or inspect a model:

```bash
uv run --no-sync mojive capture joint_types -o output/scene.png
uv run --no-sync mojive record joint_types -o output/scene.mp4 --frames 300
uv run --no-sync mojive inspect joint_types --json
```

Publish a simulation and attach a viewer from another terminal:

```bash
uv run --no-sync mojive serve joint_types --record-snapshot output/session.fvs
uv run --no-sync mojive attach
```

After stopping the publisher, run `uv run --no-sync mojive replay output/session.fvs --loop`
and attach again.
The default address is `127.0.0.1:47650`; `--host` and `--port` select another address.
Streams and `.fvs` files use pickle and require trusted peers and files.
See [remote viewing and replay](docs/tutorials/remote-viewing.md).

To control the same viewer through local RPC, run these in separate terminals:

```bash
uv run --no-sync mojive view joint_types --rpc-socket output/mojive.sock
uv run --no-sync mojive control get_state --socket output/mojive.sock --json
```

`rpc-serve` runs a separate headless scene-control service. The
[RPC guide](docs/how-to/rpc-control.md) covers discovery, edits, capture, and `RpcClient`;
the [CLI reference](docs/reference/cli.md) lists all commands.

## Development

The `python/` directory contains the Python package, installed as `mojive`. Native C++ code is in
`cpp/`, and dependencies are in `3rdparty/`. Adapters convert simulation data into scene data.
`Session` stores editor state and processes commands from the UI. Renderers allocate GPU
resources and produce images.

```bash
make check             # Formatting, lint, CPU and integration tests
make gpu               # OpenGL rendering tests
make docs-check        # Documentation checks and strict site build
make readme-media      # Refresh the README screenshots
make help              # Examples, backend checks and performance targets
```

Generated captures, recordings, reports, and the documentation site are written to `output/`.

- [Architecture](docs/concepts/architecture.md) and [renderer contracts](docs/concepts/rendering.md)
- [Development guide](docs/guides/development.md): module layout, native builds, and documentation setup
- [Verification matrix](docs/guides/testing.md#change-mapping): required checks by change type

[MIT license](LICENSE).
