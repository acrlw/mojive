# Examples

Run examples from the repository root after `make setup` or an editable installation.

| Example | Workflow |
|---|---|
| `programmatic_scene.py` | Build an interactive Mojive scene without a physics backend |
| `debug_draw.py` | Draw retained diagnostics and world-space labels |
| `custom_adapter.py` | Integrate a small independent simulation backend |
| `mujoco_render.py` | Render MuJoCo RGB, depth, and segmentation images offscreen |
| `multi_camera_render.py` | Render the free view and every fixed MuJoCo camera |
| `mujoco_control.py` | Change qpos and advance MuJoCo through the command/session API |
| `compose_scene.py` | Combine MJCF or URDF models and export a workspace or MJCF |
| `remote_publish.py` | Publish a live programmatic scene for independent viewers |
| `record_replay.py` | Write a remote snapshot recording for replay |
| `control_client.py` | Automate a running local RPC service |

## Programmatic scene

```bash
.venv/bin/python examples/programmatic_scene.py
```

## MuJoCo rendering and control

```bash
.venv/bin/python examples/mujoco_render.py assets/test_scene.xml --output output/examples
.venv/bin/python examples/mujoco_render.py assets/test_scene.xml --backend wgpu
.venv/bin/python examples/mujoco_control.py assets/slider_crank.xml --steps 120
.venv/bin/python examples/multi_camera_render.py assets/humanoid.xml \
  --output output/examples/cameras
```

## Custom adapters and debug draw

```bash
.venv/bin/python examples/custom_adapter.py
.venv/bin/python examples/debug_draw.py
```

## Scene composition and MJCF export

```bash
.venv/bin/python examples/compose_scene.py \
  assets/test_scene.xml assets/test_scene.urdf \
  --output output/examples/workcell.mojive.json

.venv/bin/python examples/compose_scene.py \
  assets/test_scene.xml assets/test_scene.urdf \
  --output output/examples/workcell.xml
```

MJCF export validates the model, copies file-backed assets into a sibling asset directory, and
writes relative paths. Mojive-only light and texture types that MJCF cannot preserve produce an
explicit error.

## Remote viewing

Start the publisher and attach one or more viewers in separate terminals:

```bash
.venv/bin/python examples/remote_publish.py
.venv/bin/mojive attach --title effect
.venv/bin/mojive attach --title normals --debug-view normal
```

Scene structure uses reliable delivery. Dynamic frames keep the latest state so a slow viewer
resumes from the current frame instead of accumulating latency.

Record and replay the same packet format without a live publisher:

```bash
.venv/bin/python examples/record_replay.py --output output/examples/orbit.fvs
.venv/bin/mojive replay output/examples/orbit.fvs
```

## Local control

```bash
.venv/bin/mojive rpc-serve assets/test_scene.xml --socket output/mojive.sock
.venv/bin/python examples/control_client.py --socket output/mojive.sock --steps 120
```
