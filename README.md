# Mojive

Mojive is an interactive 3D viewer, editor, and renderer for robotics and simulation. MuJoCo
models, programmatic scenes, custom adapters, remote publishers, and recorded snapshots all use
the same scene contract, rendering pipeline, and UI.

Mojive currently provides:

- an editor for composing MJCF and URDF models with authored geometry, materials, cameras, lights,
  and environment settings;
- interactive selection, transform and joint gizmos, physical perturbation, debug drawing,
  diagnostics, capture, and video recording;
- a `mujoco.Renderer`-style offscreen API for RGB, metric depth, and segmentation output;
- OpenGL and wgpu render backends; and
- live remote viewing, snapshot replay, and local RPC automation.

The project is under active development. File formats are versioned, but pre-1.0 APIs and editor
workflows may still change.

## Install

Mojive requires Python 3.11 or newer. The default interactive backend needs an OpenGL 3.3 core
profile. The wgpu extra uses Metal, Vulkan, or DX12. MuJoCo is optional unless you load MJCF/URDF
or use the compatible `Renderer` API.

From a source checkout:

```bash
git clone https://github.com/acrlw/mojive.git
cd mojive
uv sync --python 3.11 --extra mujoco --extra wgpu
```

For development, install the test and documentation dependencies as well:

```bash
make setup
```

A standard virtual environment also works:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[mujoco,wgpu]"
```

## Start the application

Open an empty workspace:

```bash
uv run mojive editor
```

Open a bundled scene or a model file directly:

```bash
uv run mojive view test_scene
uv run mojive view path/to/model.xml --paused
uv run mojive view path/to/model.urdf --paused
```

Use the wgpu renderer instead of OpenGL:

```bash
MOJIVE_BACKEND=wgpu uv run mojive editor
MOJIVE_BACKEND=wgpu uv run mojive view test_scene
```

`MOJIVE_BACKEND` selects the **render backend** (`opengl` or `wgpu`). The CLI option
`--backend` selects the **scene/physics adapter** (`mujoco`, `toy`, or another registered adapter).
These are independent choices. Run these discovery commands to inspect the current installation:

```bash
uv run mojive backends
uv run mojive assets --quick
uv run mojive --help
```

The editor opens and saves `.mojive.json` workspaces. A workspace can combine multiple MJCF or
URDF models with Mojive-authored entities and resource directories. Use **File > Save As** and
choose MJCF/XML to export a portable MuJoCo model. See the
[editor and MJCF guide](docs/guides/editor-and-mjcf.md) for topology editing, assets, keyframes,
resource repair, and export behavior.

## Basic interaction

- `G`: position gizmo
- `R`: rotation gizmo
- `T`: switch body/world frame
- `Shift` while dragging: snap to the configured position or rotation step
- `Ctrl` + left/right drag: MuJoCo translation/rotation perturbation
- `F`: frame the scene
- `F9`: open the dockable Settings panel

Camera preview is off by default. Select a camera and enable **preview** in Inspector when needed.
The preview can then be pinned to its current view or locked to the camera entity.

The UI follows the display content scale. Override it only when the desktop reports the wrong
value:

```bash
MOJIVE_UI_SCALE=1.5 uv run mojive editor
MOJIVE_LANGUAGE=zh_CN uv run mojive editor
```

All supported runtime settings and persistence paths are listed in the
[configuration reference](docs/reference/configuration.md).

## Programmatic rendering

`mojive.Renderer` follows the common `mujoco.Renderer` workflow while adding explicit backend
selection through `MOJIVE_BACKEND`:

```python
import mujoco

from mojive import Renderer

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

with Renderer(model, width=640, height=480) as renderer:
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=-1)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    depth_m = renderer.render()

    renderer.enable_segmentation_rendering()
    object_id_and_type = renderer.render()
```

The API supports free, fixed, named, and `MjvCamera` cameras, `MjvOption`, caller-owned output
arrays, multiple contexts, and deterministic release. See the
[MuJoCo rendering tutorial](docs/tutorials/mujoco-rendering.md) and
[rendering API reference](docs/api/rendering.md).

## Programmatic scenes and adapters

Use `Scene` for geometry, cameras, lights, and materials created in Python:

```python
from mojive import Light, Scene, build_scene

scene = Scene()
ball = scene.sphere(name="ball", position=(0.0, 0.0, 0.5))
key = scene.add_light("key", Light())
viewer = build_scene(scene)

for frame in range(300):
    ball.set_pose((frame * 0.01, 0.0, 0.5))
    viewer.sync()

key.remove()
viewer.release()
```

Custom simulations implement `SceneAdapterBase`: `scene_source()` publishes stable structure,
`frame()` publishes dynamic data, and `AdapterCaps` declares optional commands. The
[custom adapter guide](docs/how-to/custom-adapter.md) and [`examples/`](examples/README.md) cover
the supported integration paths.

## Remote viewing, replay, and control

Run a headless publisher and attach one or more independent viewers:

```bash
uv run mojive serve deformables --host 127.0.0.1 --port 47650
uv run mojive attach --host 127.0.0.1 --port 47650
```

Record the published stream with `--record-snapshot output/session.fvs`, then start replay and
attach in separate terminals:

```bash
uv run mojive replay output/session.fvs --loop
uv run mojive attach
```

Local automation uses a versioned AF_UNIX service:

```bash
MOJIVE_BACKEND=wgpu uv run mojive rpc-serve test_scene --socket output/mojive.sock
uv run mojive control get_state --socket output/mojive.sock --json
uv run mojive control capture \
  --socket output/mojive.sock \
  --params '{"mode":"depth","output":"output/depth.npy"}'
```

The wgpu selection keeps capture portable on macOS, where an OpenGL context cannot be created in
the RPC request worker. Linux may use the default OpenGL renderer for the same service.

See the [CLI reference](docs/reference/cli.md),
[remote viewing tutorial](docs/tutorials/remote-viewing.md), and
[RPC guide](docs/how-to/rpc-control.md) for the complete workflows.

## Development and verification

Use the smallest relevant target while iterating, then run the required repository gate:

```bash
make check
```

Rendering changes additionally require:

```bash
make gpu
```

Useful focused targets include `make renderer-api`, `make gpu-wgpu`, `make mujoco-audit`,
`make adapter-conformance`, `make docs-check`, `make gizmo-gallery`, and `make showcase`. Run
`make help` for the maintained target catalog. Generated captures, recordings, reports, and the
documentation site are written under `output/`.

## Architecture

```text
src/mojive/
├── types.py, math3d.py, commands.py   shared contracts
├── session.py                         application state and command routing
├── scene.py                           programmatic scenes
├── adapters/                          MuJoCo, static, toy, and remote sources
├── render/
│   ├── scene.py                       renderer scene representation
│   ├── backend.py                     rendering backend protocol
│   ├── opengl/                        OpenGL passes and shaders
│   └── webgpu/                        wgpu passes and WGSL shaders
└── ui/                                window, panels, gestures, and gizmos
```

Python matrices are row-major with translation in `matrix[:3, 3]`. World coordinates use Z-up.
Renderer code consumes shared scene contracts; physics-specific state remains in adapters;
`Session` owns selection, overrides, history, and command routing.

## Documentation

- [User guide](docs/index.md)
- [Getting started](docs/getting-started.md)
- [CLI reference](docs/reference/cli.md)
- [Configuration reference](docs/reference/configuration.md)
- [Examples](examples/README.md)
- [Architecture](docs/concepts/architecture.md)
- [Renderer design](docs/RENDERER.md)
- [API map](docs/api/index.md)
- [Testing guide](docs/guides/testing.md)

Current priorities are tracked in [`plan/STATUS.md`](plan/STATUS.md). Completed implementation
plans are retained in [`plan/`](plan/README.md) as historical engineering records.

## License

Mojive is available under the [MIT License](LICENSE).
