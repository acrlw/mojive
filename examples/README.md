# Examples

Run examples from the repository root after `make setup` or an editable installation.

| Example | Workflow |
|---|---|
| `programmatic_scene.py` | Build an interactive Forge scene without a physics backend |
| `mujoco_render.py` | Render MuJoCo RGB, depth, and segmentation images offscreen |
| `mujoco_control.py` | Change qpos and advance MuJoCo through the command/session API |
| `compose_scene.py` | Combine MJCF or URDF models and export a workspace or MJCF |
| `remote_publish.py` | Publish a live programmatic scene for independent viewers |

## Programmatic scene

```bash
.venv/bin/python examples/programmatic_scene.py
```

## MuJoCo rendering and control

```bash
.venv/bin/python examples/mujoco_render.py assets/test_scene.xml --output output/examples
.venv/bin/python examples/mujoco_render.py assets/test_scene.xml --backend wgpu
.venv/bin/python examples/mujoco_control.py assets/slider_crank.xml --steps 120
```

## Scene composition and MJCF export

```bash
.venv/bin/python examples/compose_scene.py \
  assets/test_scene.xml assets/test_scene.urdf \
  --output output/examples/workcell.forge.json

.venv/bin/python examples/compose_scene.py \
  assets/test_scene.xml assets/test_scene.urdf \
  --output output/examples/workcell.xml
```

MJCF export validates the model, copies file-backed assets into a sibling asset directory, and
writes relative paths. Forge-only light and texture types that MJCF cannot preserve produce an
explicit error.

## Remote viewing

Start the publisher and attach one or more viewers in separate terminals:

```bash
.venv/bin/python examples/remote_publish.py
.venv/bin/forge-viewer attach --title effect
.venv/bin/forge-viewer attach --title normals --debug-view normal
```

Scene structure uses reliable delivery. Dynamic frames keep the latest state so a slow viewer
resumes from the current frame instead of accumulating latency.
