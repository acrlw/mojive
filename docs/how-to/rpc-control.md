# Local RPC control

The local control service exposes typed simulation, state, selection, camera, and capture
operations over an AF_UNIX socket. `RpcClient` keeps one connection open across sequential calls
and reconnects after a transport failure.

## Start a service

```bash
MOJIVE_BACKEND=wgpu uv run mojive rpc-serve assets/test_scene.xml \
  --socket output/mojive.sock
```

wgpu is the portable choice for RPC capture on macOS because OpenGL contexts there must be created
on the process main thread, while requests run on server workers. Linux can use the default OpenGL
path. Non-rendering methods work with either selection.

To control the same interactive viewer that a user sees, attach the service to the viewer instead
of starting a second headless session. Socket workers queue requests and the viewer executes them
on its UI thread at the start of a frame:

```python
from mojive import build

viewer = build("assets/test_scene.xml")
viewer.start_rpc("output/mojive.sock")
try:
    viewer.run()
finally:
    viewer.release()
```

The standalone `rpc-serve` command starts a real-time scheduler after `resume`. An attached
service uses the viewer's existing frame scheduler and never advances the same session twice.

## Run the Python client

```bash
uv run python examples/control_client.py \
  --socket output/mojive.sock \
  --steps 120 \
  --capture output/examples/rpc.png
```

```python
--8<-- "examples/control_client.py"
```

The command-line client provides the same protocol for scripts and shell automation:

```bash
uv run mojive control get_state --socket output/mojive.sock --json
```

Pass method parameters as one JSON object:

```bash
uv run mojive control step --params '{"count":10}' --json
uv run mojive control set_qpos --params '{"index":0,"value":0.25}' --json
uv run mojive control capture \
  --params '{"mode":"depth","width":640,"height":480,"output":"output/depth.npy"}' \
  --json
```

## Methods

| Method | Parameters | Result |
|---|---|---|
| `hello`, `get_capabilities` | none | protocol, method, adapter, and attachment capabilities |
| `load` | `path` | load another model and reset capture resources |
| `reload` | none | reload the current model |
| `pause`, `resume`, `reset` | none | simulation control |
| `step` | optional `count`, `ctrl`, `observe` | apply controls, advance fixed steps, and optionally return state |
| `set_speed` | `factor` | set simulation playback speed |
| `set_keyframe` | `keyframe_id` | load one model keyframe |
| `set_qpos` | `index` + `value`, or complete `values` | update generalized position |
| `set_qvel`, `set_ctrl` | complete `values`; `set_ctrl` also accepts `index` + `value` | update velocity or actuator controls |
| `set_mocap` | optional `position`, `quaternion` | update mocap state while paused |
| `set_state` | complete state vectors and optional `time` | restore validated physics fields while paused |
| `get_state` | none | physics, selection, camera, and asset state |
| `get_scene`, `get_bounds` | none | scene metadata or world bounds |
| `set_camera` | `camera_id`, or camera-bookmark fields | select or update the capture camera |
| `load_camera_bookmark` | `name`, optional `directory` | load a saved camera bookmark |
| `set_visual_group` | `category`, `group`, `visible` | change one MuJoCo visual group |
| `set_render_flag` | `name`, `enabled` | change one `mjtRndFlag` |
| `set_shadow_quality` | `quality`, optional `persist` | set the attached viewer's shadow preset |
| `set_visualization_flag` | `name`, `enabled` | change one `mjtVisFlag` |
| `capture` | optional `mode`, `width`, `height`, `output` | save RGB PNG or depth/segmentation NPY |
| `list_objects` | none | list selectable scene nodes |
| `select_object` | `object_id` | update selection |
| `select_node` | `node_id` | update hierarchy selection directly |
| `inspect_object` | `object_id` or `node_id` | return one node and its children |
| `set_visible` | `node_id`, `visible` | update hierarchy-node visibility |
| `get_viewer_settings` | none | attached viewer interactions, selection style, and panels |
| `reset_layout` | none | discard stale docking coordinates and rebuild the default layout |
| `set_interactions` | partial interaction mapping | update attached viewer input ownership |
| `set_selection_style` | partial selection-style mapping | update attached viewer selection presentation |
| `get_panels`, `set_panel` | none; or `id` plus `open`/`enabled` | inspect or change attached viewer panels |

Attached-viewer settings are instance-local by default. Pass `"persist": true` to
`set_interactions`, `set_selection_style`, or `set_shadow_quality` only when the remote caller
intentionally wants to update the user's desktop preferences.

Capture mode defaults to `rgb`; width and height default to 640×480. Depth contains metric camera
distance. Segmentation contains `(object ID, object type)` pairs.

The socket is local and the protocol is intended for trusted processes on the same machine. Use
the remote snapshot transport when a renderer runs on another host.
