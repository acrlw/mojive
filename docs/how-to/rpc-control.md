# Local RPC control

The local control service exposes typed simulation, state, selection, camera, and capture
operations over an AF_UNIX socket. `RpcClient` keeps one connection open across sequential calls
and reconnects after a transport failure.

## Start a service

Start an editor with an attached endpoint, or attach one to an existing viewer in Python:

```bash
uv run mojive editor --rpc-socket output/mojive.sock
uv run mojive view assets/test_scene.xml --rpc-socket output/model.sock
```

For a separate standalone simulation:

```bash
MOJIVE_RENDERER=wgpu uv run mojive rpc-serve assets/test_scene.xml \
  --socket output/mojive.sock
```

wgpu is the portable choice for RPC capture on macOS because OpenGL contexts there must be created
on the process main thread. A headless service uses one dedicated graphics worker for capture;
socket workers serialize commands through the service. Linux can use the default OpenGL path.
Non-rendering methods work with either selection.

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

The standalone service starts paused and starts a real-time scheduler after `resume`. An attached
service uses the viewer's existing frame scheduler and never advances the same session twice.

Use a different socket path for each running service. Startup rejects regular files, symlinks, and
active sockets; it reclaims a stale socket only after the operating system refuses a connection.
Shutdown removes only the socket entry owned by that service.

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

## Deadlines and recovery

`RpcClient(timeout=5.0)` applies a deadline to each complete call. Requests include an optional
`deadline` field in host `time.monotonic()` seconds; the local AF_UNIX service shares this clock.
`hello` advertises `deadline_clock: "monotonic"`. Protocol version 1 clients that omit the field
retain their original behavior without a server execution deadline.

An expired request that has not started returns `deadline_exceeded` and does not mutate the scene.
Queued viewer commands are cancelled when their deadline expires, so resuming the UI does not
execute abandoned steps. If execution already started, `completion_unknown` means it may still
finish. A client-side `timeout` or connection failure can likewise leave the outcome unknown.
Inspect the current state before retrying a mutation such as `step`; the client never retries a
command automatically. The next call reconnects after a transport failure.

## Discover operations

`hello` lists recognized `methods`, currently `available_methods`, adapter capabilities, and
whether a viewer is attached. Query the schema for the operation you need:

```bash
uv run mojive control describe_operations --params '{"name":"add_scene_object"}' --json
uv run mojive control describe_operations --params '{"scope":"viewport","available_only":true}' --json
```

Each description includes JSON Schema Draft 2020-12 `input_schema` and `output_schema`, defaults,
`scope`, `mutates`, `transactional`, requirements, and current `available`/`unavailable_reason`.
Availability reflects the adapter, pause state, history, and viewer attachment. Refresh it after
state changes. Input validation rejects missing, unknown, incorrectly typed, and non-finite
parameters before dispatch. Python clients can call `client.describe_operations(name="edit_scene")`.
The catalog in `operations.py` drives both discovery and dispatch. Result schemas describe
scene inspection, physics array shapes/values, camera bookmarks, viewer settings, and discovery
records. Native remote authoring commands consume and check `expected_document` before command
construction, retaining their existing `CommandResult` error format.

| Scope | Common operations |
|---|---|
| Service | `hello`, `get_capabilities`, `describe_operations` |
| Scene queries | `get_scene`, `get_state`, `get_bounds`, `list_objects`, `inspect_object` |
| Selection | `select_object`, `select_node`, `set_visible`, `set_visual_group` |
| Simulation | `pause`, `resume`, `step`, `reset`, `set_speed`, `set_keyframe`, `set_qpos`, `set_qvel`, `set_ctrl`, `set_mocap`, `set_state` |
| Documents | `load`, `reload`, `new_scene`, `open_scene`, `save_scene` |
| Authoring | `add_scene_object`, `add_scene_camera`, `add_scene_light`, `set_pose`, `set_scene_camera`, `set_geometry_color`, `set_geometry_size`, `rename_scene_entity`, `duplicate_scene_entity`, `remove_scene_entity` |
| History | `edit_scene`, `undo`, `redo` |
| Capture | `get_capture_settings`, `set_capture_camera`, `capture`, `set_render_flag`, `set_visualization_flag`, `load_camera_bookmark` |
| Viewport | `get_viewport_camera`, `set_viewport_camera`, `capture_viewport`, `get_viewer_settings`, `get_panels`, `set_panel`, `set_interactions`, `set_selection_style`, `set_shadow_quality`, `reset_layout` |

`set_camera` remains a version-1 alias for `set_capture_camera`. Camera and light removal also
have explicit `remove_scene_camera`/`remove_scene_light` operations. Consult discovery for their
parameter shapes rather than translating an object ID into a camera or light ID.

## Edit a document

`get_scene`, `get_state`, `inspect_object`, and command results include a `document` token with
an opaque `id` and authored-history `revision`. New, open, load, and reload establish a fresh
identity. Save and Undo/Redo retain it. The revision identifies an authored history state; it is
not a simulation frame counter or a revision of viewport/visibility settings.

Pass the most recently observed token as `expected_document` on a scene mutation to reject
stale references. A mismatch returns `stale_document`, including expected and actual tokens,
before changing state. Omitting the revision checks only document identity. Legacy clients may
omit the precondition entirely. `structure_generation` describes render structure and does not
replace document identity.

Creation results retain legacy `entity_id` and add a named `object_id`, `camera_id`, or `light_id`.
Scene, object-list, bounds, and inspection queries refresh the composed Session before reading,
so they also observe updates made through a caller-owned scene provider.
`get_scene.cameras` maps camera IDs to selectable object IDs. `inspect_object` returns position
and a 3×3 rotation in the world frame. Geometry color/size operations use the `node_id` returned
by `inspect_object.geometries`; the selected parent can have a different ID. Camera angles are
radians; legacy orbit yaw/pitch are degrees.

`source_editable` identifies a compiled adapter node with an editable model-source element.
Mojive scene authoring has separate capabilities and can remain available when this field is
false. Consult operation discovery for current support and command results for the chosen target.

`inspect_object.geometries` lists the current render instances in the node's subtree, including
hidden geometry. Each entry provides `instance_index`, geometry `node_id`, selection `object_id`,
`mesh` (shape/index), `size`, instance `rgba`, `material_index`, and complete material parameters.
These values come from the composed Session, so color overrides, Undo/Redo, and reopened edits
are visible immediately. `material_index: -1` identifies the default material. Instance/material
indices are scoped to `structure_generation`; refresh inspection after structure changes.

`size` is the render-size vector: boxes use half extents, spheres use radii. `dimensions`
provides a conventional label and values, such as full box width/depth/height, or is null for
unsupported shape families. Composite shapes
can produce multiple instances with the same geometry node ID. Instance color and material RGBA
remain separate values. `visible` is the local hierarchy flag; `hierarchy_visible` accounts for
hidden ancestors and does not claim that an object is inside the camera or unoccluded.

```python
scene = client.call("get_scene")
created = client.call("add_scene_object", {
    "shape": "box", "name": "crate", "expected_document": scene["document"],
})
node = client.call("inspect_object", {"object_id": created["object_id"]})
client.call("edit_scene", {
    "label": "Arrange crate", "expected_document": node["document"],
    "operations": [
        {"method": "rename_scene_entity", "params": {"object_id": node["object_id"], "name": "cargo"}},
        {"method": "set_pose", "params": {
            "node_id": node["node_id"], "position": [1, 0, 0.5],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }},
    ],
})
client.call("save_scene", {"path": "output/cargo.mojive.json"})
```

`edit_scene` validates all nested operations before starting, commits one undo record, and
cancels the whole edit on failure. Only operations marked `transactional` are accepted. Parameters
are literal values: inspect a newly created entity before referencing its ID in another request.
Use `undo`/`redo` for authored history; visibility is a Session display override and is outside
these transactions. Simulation commands and document replacement are also outside transactions.

With `--json`, CLI failures print `{"error":{"code":...,"message":...,"details":...}}` to stdout
and exit with status 2. Success prints the operation result. Python raises `RpcError` with the
same `code` and optional `details`. No client retry is implicit.

Attached-viewer settings are instance-local by default. Pass `"persist": true` to
`set_interactions`, `set_selection_style`, or `set_shadow_quality` only when the remote caller
intentionally wants to update the user's desktop preferences.

Capture mode defaults to `rgb`; width and height default to 640×480. Depth contains metric camera
distance. `object_id` returns a uint32 NPY image whose nonzero values match selection IDs from
`list_objects` and `inspect_object`. `segmentation` returns int32 `(semantic ID, semantic type)`
pairs supplied by the adapter. MuJoCo supplies native geometry/site/flex/skin IDs and `mjtObj`
types; unknown semantics and the background use `(-1, -1)`. These semantic IDs are distinct from
selection object IDs.

Captures consume the current Session scene, including authored geometry, visibility, materials,
camera overrides, and dynamic meshes. They work with static, workspace, remote, and physics
adapters through the same scene contracts. Capture refreshes the frame with zero elapsed wall
time; external publishers may deliver a newer frame during this refresh.

The result includes `orientation: "top_left"`, `scope: "session_scene"`, `structure_generation`,
`step`, `time`, and `document`, alongside the saved path, shape, and dtype. `hello` advertises `capture_modes`
and `capture_scope`. Use `describe_operations` for per-operation availability and schemas.

Capture is an offscreen scene image with its own rendering settings. It excludes application
panels, selection decoration, viewport overlays, and debug Bridge commands. `set_render_flag`
and `set_visualization_flag` affect subsequent RPC captures. They accept the corresponding
`RenderFlag` names and legacy `mjRND_`/`mjVIS_` aliases for supported features. The attached
viewer's shadow preset remains controlled by `set_shadow_quality`. To capture the actual
application surface, call `capture_viewport` with `surface: "viewport"` or `surface: "window"`.
It waits until the viewer has presented a frame and saved the image, then returns the artifact
metadata. It requires a running viewer loop. The default surface is `viewport`.

`set_capture_camera` changes only offscreen capture; `set_viewport_camera` changes the visible
viewer. Both retain camera roll and physical intrinsics. A free viewport camera preserves its
exact view across frames and resize until an orbit gesture or framing command takes control.
The viewport supplies its own aspect ratio. `get_capture_settings` reports the capture camera,
explicit `render_flag_overrides`, debug view, opacity, and image modes. Selecting a scene camera
follows its current pose; removing it returns capture to a free camera at the last observed pose.
Replacing the document resets capture camera selection.

Send JSON booleans for `visible` and flag `enabled`, and positive JSON integers for capture
dimensions. These fields reject string coercions and return `invalid_params` before mutation.

The socket is local and the protocol is intended for trusted processes on the same machine. Use
the remote snapshot transport when a renderer runs on another host.
