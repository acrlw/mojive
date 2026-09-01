# Examples

Run these programs from the repository root after `uv sync --extra mujoco --extra wgpu` or
`make setup`. Commands below use `uv run` so they work without activating `.venv`.

| Example | Window | Result |
|---|---:|---|
| `programmatic_scene.py` | yes | authored scene without a physics backend |
| `debug_draw.py` | yes | retained diagnostics and world-space labels |
| `custom_adapter.py` | yes | small independent simulation adapter |
| `mujoco_render.py` | no | RGB PNG, metric depth NPY, and segmentation NPY |
| `multi_camera_render.py` | no | one PNG for the free view and each fixed camera |
| `mujoco_control.py` | no | qpos editing and deterministic stepping through `Session` |
| `compose_scene.py` | no | combined `.mojive.json` workspace or portable MJCF |
| `remote_publish.py` | no | live latest-state publisher for attached viewers |
| `record_replay.py` | no | versioned `.fvs` snapshot recording |
| `control_client.py` | no | persistent local RPC automation |

All generated files in these examples are placed under the ignored `output/examples/` directory.

## Interactive scenes

```bash
uv run python examples/programmatic_scene.py
uv run python examples/custom_adapter.py
uv run python examples/debug_draw.py
```

Close the application window to end each program. The source for all three is embedded in the
corresponding user-guide page and is checked during `make docs-check`.

## MuJoCo rendering and control

```bash
uv run python examples/mujoco_render.py assets/test_scene.xml \
  --output output/examples/render

uv run python examples/mujoco_render.py assets/test_scene.xml \
  --backend wgpu \
  --output output/examples/render-wgpu

uv run python examples/mujoco_control.py assets/slider_crank.xml --steps 120

uv run python examples/multi_camera_render.py assets/showcase.xml \
  --output output/examples/cameras
```

`mujoco_render.py` creates `rgb.png`, `depth.npy`, and `segmentation.npy`. The depth array contains
metric camera distance. The segmentation array stores `(object ID, object type)` pairs.

`multi_camera_render.py` writes `free.png` and one file for every model camera. `showcase.xml` is
used here because it contains named fixed cameras.

## Scene composition and MJCF export

Save an editable workspace:

```bash
uv run python examples/compose_scene.py \
  assets/test_scene.xml assets/test_scene.urdf \
  --spacing 2.5 \
  --output output/examples/workcell.mojive.json
```

Save the same composition as portable MJCF:

```bash
uv run python examples/compose_scene.py \
  assets/test_scene.xml assets/test_scene.urdf \
  --spacing 2.5 \
  --output output/examples/workcell.xml
```

The MJCF path compiles the result for validation and copies file-backed resources into a sibling
asset directory. The `.mojive.json` path retains model references, authored entities, and resource
roots for later editing.

## Remote viewing

Start the publisher and attach one or more viewers in separate terminals:

```bash
uv run python examples/remote_publish.py --host 127.0.0.1 --port 47650 --hz 30
```

```bash
uv run mojive attach --host 127.0.0.1 --port 47650 --title effect
uv run mojive attach --host 127.0.0.1 --port 47650 \
  --title normals --debug-view normal
```

Structure changes use reliable delivery. Dynamic frames keep only the latest state, so a slow
viewer resumes from the current frame instead of accumulating latency.

Create a finite recording, then start replay and attach in separate terminals:

```bash
uv run python examples/record_replay.py \
  --output output/examples/orbit.fvs --frames 300 --fps 60
uv run mojive replay output/examples/orbit.fvs --loop
uv run mojive attach
```

## Local control

Start the service:

```bash
MOJIVE_BACKEND=wgpu uv run mojive rpc-serve assets/test_scene.xml \
  --socket output/mojive.sock
```

Run the persistent Python client from another terminal:

```bash
uv run python examples/control_client.py \
  --socket output/mojive.sock \
  --steps 120 \
  --capture output/examples/rpc.png
```

The AF_UNIX service is for trusted local automation. See the
[RPC control guide](../docs/how-to/rpc-control.md) for the equivalent one-shot CLI. The wgpu
renderer is used here because macOS does not permit the OpenGL capture context on an RPC worker
thread; Linux can use the default OpenGL path.
