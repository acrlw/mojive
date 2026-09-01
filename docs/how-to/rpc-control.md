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
| `load` | `path` | load another model and reset capture resources |
| `reload` | none | reload the current model |
| `pause`, `resume`, `reset` | none | simulation control |
| `step` | optional `count` | advance one or more fixed steps |
| `set_keyframe` | `keyframe_id` | load one model keyframe |
| `set_qpos` | `index` + `value`, or complete `values` | update generalized position |
| `get_state` | none | physics, selection, camera, and asset state |
| `set_camera` | `camera_id`, or camera-bookmark fields | select or update the capture camera |
| `load_camera_bookmark` | `name`, optional `directory` | load a saved camera bookmark |
| `set_visual_group` | `category`, `group`, `visible` | change one MuJoCo visual group |
| `set_render_flag` | `name`, `enabled` | change one `mjtRndFlag` |
| `set_visualization_flag` | `name`, `enabled` | change one `mjtVisFlag` |
| `capture` | optional `mode`, `width`, `height`, `output` | save RGB PNG or depth/segmentation NPY |
| `list_objects` | none | list selectable scene nodes |
| `select_object` | `object_id` | update selection |
| `inspect_object` | `object_id` or `node_id` | return one node and its children |

Capture mode defaults to `rgb`; width and height default to 640×480. Depth contains metric camera
distance. Segmentation contains `(object ID, object type)` pairs.

The socket is local and the protocol is intended for trusted processes on the same machine. Use
the remote snapshot transport when a renderer runs on another host.
