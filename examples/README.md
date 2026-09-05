# Examples

Run these programs from the repository root after `uv sync --extra mujoco --extra wgpu` or
`make setup`. Commands below use `uv run` so they work without activating `.venv`.

| Example | Window | Result |
|---|---:|---|
| `offscreen_scene.py` | no | authored RGB, depth, and object-ID output without MuJoCo |
| `programmatic_scene.py` | yes | authored scene without a physics backend |
| `debug_draw.py` | yes | retained diagnostics and world-space labels |
| `canvas2d.py` | yes | layered 2D physics and geometry diagnostics |
| `custom_adapter.py` | yes | small independent simulation adapter |
| `mujoco_render.py` | no | RGB PNG, metric depth NPY, and segmentation NPY |
| `mujoco_video.py` | no | streamed MP4 rollout with optional Pillow label/timestamp |
| `multi_camera_render.py` | no | one PNG for the free view and each fixed camera |
| `mujoco_control.py` | no | qpos editing and deterministic stepping through `Session` |
| `compose_scene.py` | no | combined `.mojive.json` workspace or portable MJCF |
| `remote_publish.py` | no | live latest-state publisher for attached viewers |
| `record_replay.py` | no | versioned `.fvs` snapshot recording |
| `control_client.py` | no | persistent local RPC automation |
| `agent_inspection.py` | no | RPC discovery, transactional editing, document lifecycle, and image verification |

All generated files in these examples are placed under the ignored `output/` directory.

## Interactive scenes

```bash
uv run python examples/programmatic_scene.py
uv run python examples/custom_adapter.py
uv run python examples/debug_draw.py
uv run python examples/canvas2d.py
```

Close the application window to end each program. These sources are cross-referenced from the
user guide and checked during `make docs-check`.

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

Record a rollout at video FPS without changing the model's physical timestep:

```bash
uv run python examples/mujoco_video.py assets/test_scene.xml \
  --frames 90 --fps 30 --label "Policy A" --output output/examples/rollout.mp4
```

Omit `--label` for raw RGB. MP4 defaults to player-compatible `yuv420p`; use
`--pixel-format yuv444p` for full chroma resolution. Odd dimensions are edge-padded for `yuv420p`,
not resized. The [rendering tutorial](../docs/tutorials/mujoco-rendering.md#record-a-rollout)
embeds the complete example. `make rollout-video` runs the same example for visual acceptance.
On a Linux desktop, `MOJIVE_GL=glfw` selects hidden-window rendering if EGL initialization fails;
display-free servers still need working EGL. See [configuration](../docs/reference/configuration.md).

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

`agent_inspection.py` runs a self-contained authored scene through RPC discovery, object
inspection, visibility, atomic edits, Undo/Redo, save/reopen, stale-document rejection, and
RGB/object-ID verification. Run `make agent-control` and review `output/agent-control/`.
`make agent-viewer` also verifies the presented viewport and window. Neither mode requires a
physics engine or separately running service.

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
