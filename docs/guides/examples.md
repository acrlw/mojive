# Examples

The repository `examples/` directory contains small programs that use the public interfaces. Run
them from an editable checkout after `make setup`.

| Example | Interfaces | Result |
|---|---|---|
| `programmatic_scene.py` | `Scene`, `build_scene` | interactive scene without physics |
| `mujoco_render.py` | `Renderer` | RGB PNG, metric depth, and segmentation arrays |
| `mujoco_control.py` | `Session`, `FrameNeeds`, commands | qpos editing and deterministic stepping |
| `compose_scene.py` | `WorkspaceAdapter`, `MuJoCoAdapter` | composed `.forge.json` or portable MJCF |
| `remote_publish.py` | `SnapshotPublisher`, remote commands | live scene for independent viewers |

## Programmatic scene

```bash
.venv/bin/python examples/programmatic_scene.py
```

## MuJoCo rendering

```bash
.venv/bin/python examples/mujoco_render.py assets/test_scene.xml --output output/examples
.venv/bin/python examples/mujoco_render.py assets/test_scene.xml --backend wgpu
```

## MuJoCo state and control

```bash
.venv/bin/python examples/mujoco_control.py assets/slider_crank.xml --steps 120
```

Pass `--qpos-index` and `--qpos` to change one generalized position before stepping.

## Model composition

```bash
.venv/bin/python examples/compose_scene.py \
  assets/test_scene.xml assets/test_scene.urdf \
  --output output/examples/workcell.forge.json
```

Use an `.xml` output path to export portable MJCF with a sibling asset directory.

## Remote viewing

Run the publisher and viewer in separate terminals:

```bash
.venv/bin/python examples/remote_publish.py
.venv/bin/forge-viewer attach --title effect
```

Additional viewers may use independent cameras and render settings. Dynamic frames retain the
latest state, while scene structure and commands use reliable delivery.
