# Mojive

![Mojive with a jointed scene, joint controls, Inspector and Keyframes](docs/images/readme/hero.png)

**A backend-neutral 3D viewer, editor, and renderer for robotics and simulation.**

[Quick start](#quick-start) · [User guide](docs/getting-started.md) ·
[Python API](docs/api/index.md) · [Examples](examples/README.md)

Mojive stands for **Mo**del/**J**oint **I**nteractive **V**iewer & **E**ditor.
It opens MJCF and URDF models, authored Python scenes, custom simulations, and remote streams.
OpenGL, WebGPU (`wgpu`), and native bgfx are available independently of the physics adapter.

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

The viewer starts paused. Select a body in Hierarchy, adjust its joints in **Joints**, and use
**Inspector** for properties. Open panels from the Window menu and drag their tabs to dock them.

```bash
uv run --no-sync mojive editor                       # Empty workspace
uv run --no-sync mojive view path/to/model.xml        # MJCF
uv run --no-sync mojive view path/to/model.urdf        # URDF
uv run --no-sync mojive assets --quick                # Bundled scenes
```

`make setup` builds the customized ImGui bindings and native geometry extension. Use `--no-sync`
afterward to preserve those builds; after replacing the editable install, run `make native-editable`.
See [installation](docs/getting-started.md) for dependency details.

## Render backends

OpenGL is the default. To use another renderer:

```bash
MOJIVE_RENDERER=wgpu uv run --no-sync mojive view joint_types
make native-viewer SCENE=joint_types
```

`make native-viewer` builds bgfx and its shaders, then opens the same Python UI.
See [native backend setup](docs/how-to/native-viewer.md) for `MOJIVE_RENDERER=bgfx` in scripts.
`--adapter` selects the scene/physics adapter; `-b/--backend` is its compatibility alias.

On Linux, offscreen OpenGL rendering can use EGL without a desktop display. It still needs a
working GPU driver and EGL installation. Set `MOJIVE_GL=egl` to require that path; the hidden GLFW
fallback needs a desktop session. See [backend requirements](docs/reference/configuration.md#render-backend-requirements).

## Editor

Save compositions as `.mojive.json` workspaces. They retain model references, edits, resource
paths, and Mojive-authored entities. Use **File > Add Model** or drop MJCF/URDF files to compose
models; **File > Save As > MuJoCo XML / MJCF** exports the model and its required assets.

Inspector edits position, rotation, dimensions, materials, lights, and cameras. Scale is editable
for authored geometry and baked into its dimensions on Apply; MuJoCo model entities show read-only
identity scale. Camera entities also offer an optional live preview. The
[editor guide](docs/guides/editor-and-mjcf.md) covers supported edits, topology, keyframes, and export.

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
UI scale and language follow the desktop; override them with `MOJIVE_UI_SCALE` and
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
The [programmatic scene guide](docs/tutorials/programmatic-scene.md) covers entities and updates;
[DebugDraw](docs/how-to/debug-draw.md) and [Canvas2D](docs/how-to/canvas2d.md) cover custom diagnostics.

`Renderer` follows the `mujoco.Renderer` update-and-render loop:

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
See [MuJoCo rendering](docs/tutorials/mujoco-rendering.md) for details. To display an existing
simulation loop while keeping physics stepping in your code, use
[`launch_passive(model, data)`](docs/tutorials/passive-viewing.md).

Custom simulations implement `SceneAdapterBase`: `scene_source()` supplies stable structure,
`frame()` supplies changing state, and `AdapterCaps` declares supported operations.
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

Python sources live under `python/` and install as `mojive`; native code lives under `cpp/`,
and dependencies under `3rdparty/`. Adapters supply scene data, `Session` owns editor state and
command routing, and renderers own GPU resources and images. The UI submits typed commands.

```bash
make check             # Formatting, lint, CPU and integration tests
make gpu               # OpenGL rendering tests
make docs-check        # Documentation checks and strict site build
make readme-media      # Refresh the README screenshots
make help              # Examples, backend checks and performance targets
```

Captures, recordings, reports, and the built documentation site go to `output/`.

- [Architecture](docs/concepts/architecture.md) and [renderer contracts](docs/concepts/rendering.md)
- [Development guide](docs/guides/development.md): module layout, native builds, and documentation setup
- [Verification matrix](docs/guides/testing.md#change-mapping): required checks by change type

[MIT license](LICENSE).
