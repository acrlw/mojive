# Examples

Run these programs from the repository root after `make setup`. Commands use
`uv run --no-sync` to preserve the locally built ImGui wheel and native extension registration.

| Example | Window | Result |
|---|---:|---|
| `offscreen_scene.py` | no | authored RGB, depth, and object-ID output without MuJoCo |
| `mesh_lod.py` | no | shared native meshes; LOD preparation only with `--mesh-lod` |
| `programmatic_scene.py` | yes | authored scene without a physics backend |
| `debug_draw.py` | yes | retained diagnostics and world-space labels |
| `canvas2d.py` | yes | layered 2D physics and geometry diagnostics |
| `custom_adapter.py` | yes | small independent simulation adapter |
| `mujoco_render.py` | no | RGB PNG, metric depth NPY, and segmentation NPY |
| `mujoco_viewer.py` | yes | MuJoCo-style viewer handle, callback, camera/options and user geoms |
| `viewer_managed.py` | yes | Minimal managed viewer: Mojive owns physics stepping |
| `viewer_passive.py` | yes | Minimal passive viewer: the caller steps and synchronizes physics |
| `mujoco_video.py` | no | streamed MP4 rollout with optional Pillow label/timestamp |
| `multi_camera_render.py` | no | one PNG for the free view and each fixed camera |
| `mujoco_control.py` | no | qpos editing and deterministic stepping through `Session` |
| `passive_viewer.py` | yes | caller-owned physics, independent display rate, and in-memory captures |
| `compose_scene.py` | no | combined `.mojive.json` workspace or portable MJCF |
| `remote_publish.py` | no | live latest-state publisher for attached viewers |
| `record_replay.py` | no | versioned `.fvs` snapshot recording |
| `rollout_preview.py` | no | bounded rollout windows, selected worlds, and manual viewer synchronization |
| `control_client.py` | no | persistent local RPC automation |
| `agent_inspection.py` | no | RPC discovery, transactional editing, document lifecycle, and image verification |

All generated files in these examples are placed under the ignored `output/` directory.

Run `make passive-viewer` for a three-second 1000 Hz physics / 60 FPS display example.
Use `ARGS='--physics-hz 500 --display-fps 30 --hidden'` for an automated run.
The guarded script entry point is required by the passive viewer's spawned display process.

## Interactive scenes

```bash
uv run --no-sync python examples/programmatic_scene.py
uv run --no-sync python examples/custom_adapter.py
uv run --no-sync python examples/debug_draw.py
uv run --no-sync python examples/canvas2d.py
```

Close the application window to end each program. These sources are cross-referenced from the
user guide and checked during `make docs-check`.

## Optional mesh detail

Commands selecting bgfx require the [native runtime and shaders](../docs/how-to/native-viewer.md).

Compare original geometry and opt-in display LOD without external models:

```bash
make mesh-lod-example ARGS='--output output/examples/mesh-exact'
make mesh-lod-example ARGS='--mesh-lod --output output/examples/mesh-lod'
```

Both runs save `rgb.png`. The default run never starts LOD preparation. The enabled run
renders 120 frames, allowing background preparation to progress; this is not a readiness
deadline. Small scenes usually do not need LOD. For interactive use, toggle **Mesh LOD** in
Settings > Rendering when using bgfx. See [adaptive mesh detail](../docs/concepts/rendering.md#adaptive-mesh-detail)
for preparation costs, sharing, and exact sensor output.

## MuJoCo rendering and control

```bash
uv run --no-sync python examples/mujoco_render.py assets/test_scene.xml \
  --output output/examples/render

uv run --no-sync python examples/mujoco_render.py assets/test_scene.xml \
  --renderer bgfx \
  --output output/examples/render-bgfx

uv run --no-sync python examples/mujoco_control.py assets/slider_crank.xml --steps 120

uv run --no-sync python examples/multi_camera_render.py assets/showcase.xml \
  --output output/examples/cameras
```

`mujoco_render.py` creates `rgb.png`, `depth.npy`, and `segmentation.npy`. The depth array contains
metric camera distance. The segmentation array stores `(object ID, object type)` pairs.

`multi_camera_render.py` writes `free.png` and one file for every model camera. `showcase.xml` is
used here because it contains named fixed cameras.

Record a rollout at video FPS without changing the model's physical timestep:

```bash
uv run --no-sync python examples/mujoco_video.py assets/test_scene.xml \
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
uv run --no-sync python examples/compose_scene.py \
  assets/test_scene.xml assets/test_scene.urdf \
  --spacing 2.5 \
  --output output/examples/workcell.mojive.json
```

Save the same composition as portable MJCF:

```bash
uv run --no-sync python examples/compose_scene.py \
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
uv run --no-sync python examples/remote_publish.py --host 127.0.0.1 --port 47650 --hz 30
```

```bash
uv run --no-sync mojive attach --host 127.0.0.1 --port 47650 --title effect
uv run --no-sync mojive attach --host 127.0.0.1 --port 47650 \
  --title normals --debug-view normal
```

Structure changes use reliable delivery. Dynamic frames keep only the latest state, so a slow
viewer resumes from the current frame instead of accumulating latency.

Create a finite recording, then start replay and attach in separate terminals:

```bash
uv run --no-sync python examples/record_replay.py \
  --output output/examples/orbit.fvs --frames 300 --fps 60
uv run --no-sync mojive replay output/examples/orbit.fvs --loop
uv run --no-sync mojive attach
```

## Local control

`agent_inspection.py` runs a self-contained authored scene through RPC discovery, object
inspection, visibility, atomic edits, Undo/Redo, save/reopen, stale-document rejection, and
RGB/object-ID verification. Run `make agent-control` and review `output/agent-control/`.
`make agent-viewer` also verifies the presented viewport and window. Neither mode requires a
physics engine or separately running service.

Start the service:

```bash
MOJIVE_RENDERER=bgfx uv run --no-sync mojive rpc-serve assets/test_scene.xml \
  --socket output/mojive.sock
```

Run the persistent Python client from another terminal:

```bash
uv run --no-sync python examples/control_client.py \
  --socket output/mojive.sock \
  --steps 120 \
  --capture output/examples/rpc.png
```

The AF_UNIX service is for trusted local automation. See the
[RPC control guide](../docs/how-to/rpc-control.md) for the equivalent one-shot CLI. The bgfx command
requires the [native runtime and shaders](../docs/how-to/native-viewer.md). macOS does not permit
OpenGL context creation on an RPC worker thread; Linux can use the default OpenGL path.
