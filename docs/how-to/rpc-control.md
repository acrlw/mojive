# Local RPC control

The local control service exposes typed simulation, state, selection, camera, and capture
operations over an AF_UNIX socket. `RpcClient` keeps one connection open across sequential calls
and reconnects after a transport failure.

## Start a service

Start an editor with an attached endpoint, or attach one to an existing viewer in Python:

```bash
uv run --no-sync mojive editor --rpc-socket
uv run --no-sync mojive view assets/test_scene.xml --rpc-socket
```

For a separate standalone simulation:

```bash
MOJIVE_RENDERER=wgpu uv run --no-sync mojive rpc-serve assets/test_scene.xml
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
viewer.start_rpc()
try:
    viewer.run()
finally:
    viewer.release()
```

The standalone service starts paused and starts a real-time scheduler after `resume`. An attached
service uses the viewer's existing frame scheduler and never advances the same session twice.
For a caller-owned policy loop, use [passive viewing](../tutorials/passive-viewing.md).

Use a different socket path for each running service. Startup rejects regular files, symlinks, and
active sockets; it reclaims a stale socket only after the operating system refuses a connection.
Shutdown removes only the socket entry owned by that service.
The default is `$XDG_RUNTIME_DIR/mojive/control.sock`, or `mojive-<uid>/control.sock` under the
system temporary directory when `XDG_RUNTIME_DIR` is unset. These examples are alternatives;
they share the default endpoint. Pass the same explicit path to both viewer and client when
running multiple services. Clearing `output/` does not remove the default socket.

## Run the Python client

```bash
uv run --no-sync python examples/control_client.py \
  --steps 120 \
  --capture output/examples/rpc.png
```

```python
--8<-- "examples/control_client.py"
```

The command-line client provides the same protocol for scripts and shell automation:

```bash
uv run --no-sync mojive control get_state --json
```

Pass method parameters as one JSON object:

```bash
uv run --no-sync mojive control step --params '{"count":10}' --json
uv run --no-sync mojive control set_qpos --params '{"index":0,"value":0.25}' --json
uv run --no-sync mojive control capture \
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

An EOF cancels requests that have not started, even when no deadline was supplied. Keep both
directions of the socket open until a response arrives; write-half-close is treated as disconnect.
Shutdown closes active sockets and drains unstarted viewer requests. Cancellation cannot undo
an already-running operation. An asynchronous operation retains its admission slot until actual
completion, including after its client has received `completion_unknown`.

## Work budgets and diagnostics

`mojive.control.rpc.RpcLimits` provides the same immutable configuration for attached and
standalone servers. All byte limits include the newline delimiter.

| Field | Default | Meaning |
| --- | --- | --- |
| `max_request_bytes` | 16 MiB | Maximum encoded request per connection |
| `max_response_bytes` | 256 MiB | Maximum encoded response |
| `max_connections` | 16 | Concurrent accepted sockets, including idle clients |
| `max_inflight_requests` | 128 | Accepted operations not yet completed or cancelled |
| `max_pending_requests` | 128 | Attached-viewer queue capacity |
| `max_pending_bytes` | 64 MiB | Sum of encoded sizes still in that queue |
| `requests_per_pump` | 8 | Maximum queued items processed per viewer frame |
| `pump_budget_ms` | 2.0 | Elapsed-time budget, checked between handlers |

The pump advances at least one queued item, including cancelled items. It cannot preempt a
handler: a single expensive operation can exceed the time budget. Synchronous model compilation
is still such an operation; the queue budget does not make compilation asynchronous. Queue
limits apply to the attached viewer; headless dispatch serializes through its application lock.
The unfinished-request limit also covers headless requests waiting for that lock.

Overloaded connections or queues return `busy` without starting the request. Oversized requests
return `request_too_large` before JSON parsing and close that connection; the remainder is never
parsed as another request. Rejections before reading a request ID carry `id: null`. `RpcClient`
recognizes those transport errors and closes the socket, without resending the request.

Oversized responses return `response_too_large` before response bytes are sent. The operation may
already have completed, so inspect state before repeating a mutation. If even the error envelope
exceeds the configured response limit, the connection closes. These are **wire-size budgets**,
not bounds on decoded JSON memory or the temporary string produced by the JSON encoder. Use
shared-memory capture for large repeated images instead of raising every message budget.

Set server limits explicitly in Python:

```python
from mojive.control.rpc import RpcLimits

viewer.start_rpc(limits=RpcLimits(max_connections=8, pump_budget_ms=1.5))
```

For CLI use, put a JSON object such as `{"max_connections": 8, "pump_budget_ms": 1.5}` in a file.
Omitted fields keep their defaults; unknown fields, invalid counts and non-finite values fail
before creating the viewer or connecting the client.

```bash
uv run --no-sync mojive editor --rpc-socket --rpc-limits output/rpc-limits.json
uv run --no-sync mojive rpc-serve test_scene --limits-file output/rpc-limits.json
uv run --no-sync mojive control get_rpc_stats --json
```

`control --limits-file PATH` configures the client's outgoing-request and incoming-response
limits; it does not change the server. Python clients accept `RpcClient(..., limits=RpcLimits(...))`.
Existing attached servers retain their limits; passing different limits to `start_rpc` fails.
Stop that endpoint explicitly before replacing its configuration.

`get_rpc_stats` is a read-only operation in the catalog. It returns the effective limits, current
queue/connection/inflight counts, rejection and cancellation counters, and the last 256 samples
of each timing with mean, p50, p95, p99 and maximum. Counters are lifetime values; percentiles are
rolling-window values. `queue_wait_ms` covers admission to handler start; `handler_ms` covers
dispatch through actual completion, including asynchronous work and headless lock wait;
`pump_ms` covers synchronous viewer service work. Cancelled queued items have no handler sample.
Expiry and cancellation counters can overlap. A stats request observes itself before completion.

To compare input dispatch under a real socket workload:

```bash
make rpc-benchmark BACKEND=opengl ARGS="--clients 16 --requests 30 --repeats 3"
```

The benchmark alternates a count-only policy with count-and-time budgets on the same transport.
It preserves raw request/frame samples, an injected-key dispatch measurement and window captures
under `output/rpc-benchmark/`. It does not measure OS input latency or GPU completion. Run it
separately from tests or other benchmarks; compare request completion and throughput alongside
input tail latency. This is a pump-policy comparison, not a comparison of whole historical servers.

## Discover operations

`hello` lists recognized `methods`, currently `available_methods`, adapter capabilities, and
whether a viewer is attached. Query the schema for the operation you need:

```bash
uv run --no-sync mojive control describe_operations --params '{"name":"add_scene_object"}' --json
uv run --no-sync mojive control describe_operations --params '{"scope":"viewport","available_only":true}' --json
```

Each description includes JSON Schema Draft 2020-12 `input_schema` and `output_schema`, defaults,
`scope`, `mutates`, `transactional`, `writes_document`, requirements, and current
`available`/`unavailable_reason`.
Availability reflects the adapter, pause state, history, and viewer attachment. Refresh it after
state changes. Input validation rejects missing, unknown, incorrectly typed, and non-finite
parameters before dispatch. Python clients can call `client.describe_operations(name="edit_scene")`.
The catalog in `control/operations.py` drives both discovery and dispatch. Result schemas describe
scene inspection, physics array shapes/values, camera bookmarks, viewer settings, and discovery
records. Native remote authoring commands consume and check `expected_document` before command
construction, retaining their existing `CommandResult` error format.

| Scope | Common operations |
|---|---|
| Service | `hello`, `get_capabilities`, `describe_operations`, `get_rpc_stats` |
| Scene queries | `get_scene`, `get_state`, `get_bounds`, `list_objects`, `inspect_object` |
| Selection | `select_object`, `select_node`, `set_visible`, `set_visual_group` |
| Simulation | `pause`, `resume`, `step`, `reset`, `set_speed`, `set_keyframe`, `set_qpos`, `set_qvel`, `set_ctrl`, `set_mocap`, `set_state` |
| Documents | `load`, `reload`, `new_scene`, `open_scene`, `save_scene` |
| Authoring | `add_scene_object`, `add_scene_camera`, `add_scene_light`, `set_pose`, `set_scale`, `set_scene_camera`, `set_geometry_color`, `set_geometry_size`, `rename_scene_entity`, `duplicate_scene_entity`, `remove_scene_entity` |
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

`inspect_object.scalable` resolves the chosen target through Session and the adapter's
`write_scale` capability. A scalable object and its geometry child address the same local
geometry. Articulated subtrees and unsupported targets are rejected. `set_scale` requires a
paused session and three positive finite XYZ factors. It multiplies the current local render
dimensions, preserves position and rotation, and immediately bakes the result. A subsequent
inspection returns `scale: [1, 1, 1]` and the updated `geometries[].size`; repeating a factor of
two scales the current dimensions by two again. Overflow and underflow fail without a partial
edit. Scale participates in `edit_scene`, Undo/Redo, document preconditions, and scene saving.

Read-only inspection uses the currently composed Session: an active editor preview can expose
unbaked `scale` values and preview dimensions. These previews are not saved until applied.
Operations marked `writes_document` reject an active UI draft or unfinished edit transaction with
`pending_edits`. This includes document replacement, saving and Undo/Redo. Apply or discard the
draft in the viewer before editing remotely; RPC never implicitly commits or clears it.
Read-only queries, selection and display visibility remain available.

```python
node = client.call("inspect_object", {"object_id": object_id})
client.call("set_scale", {
    "node_id": node["node_id"], "scale": [2, 1, 0.5],
    "expected_document": node["document"],
})
```

Use `mojive operations set_scale --json` to read the installed schema, then pass the same
parameters through `mojive control set_scale --params-file output/scale.json --json`.

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

`edit_scene` validates all nested operations and rechecks the document before starting. It uses
the same adapter rebuild group and rollback executor as UI Apply, retaining request order and
one undo record. MuJoCo geometry declaration edits compile the final specification once;
pose changes that subsequent world-space commands depend on still update physics immediately.
Successful nested results carry the final committed document revision.

A failure cancels the whole edit and reports the original error. When recovery also fails,
`rollback_failed` includes `details.rollback_error` and retains the recovery checkpoint;
inspect and resolve that failure before submitting further edits. Command failures include
their input `index` and `method`; group finalization failures have no single failing input.
Only operations marked `transactional` are accepted. Parameters
are literal values: inspect a newly created entity before referencing its ID in another request.
Use `undo`/`redo` for authored history; visibility is a Session display override and is outside
these transactions. Simulation commands and document replacement are also outside transactions.

With `--json`, CLI failures print `{"error":{"code":...,"message":...,"details":...}}` to stdout
and exit with status 2. Success prints the operation result. Python raises `RpcError` with the
same `code` and optional `details`. No client retry is implicit.

For shell automation, `mojive operations edit_scene --json` reads the installed schema without
starting a service. Check `control describe_operations` for the target service's live contract
and availability. Save a multi-operation request as UTF-8 JSON, then call
`mojive control edit_scene --params-file output/edit.json --json`. Use `--params-file -` for stdin.
Inline `--params` remains supported. Malformed JSON and non-finite numbers fail before connecting.

Malformed response envelopes return `invalid_response` and close the connection. A subsequent
explicit call reconnects. A connection or response failure after submission can leave a mutation's
outcome unknown; read the current document before deciding whether another edit is needed.

Attached-viewer settings are instance-local by default. Pass `"persist": true` to
`set_interactions`, `set_selection_style`, or `set_shadow_quality` only when the remote caller
intentionally wants to update the user's desktop preferences.

Capture mode defaults to `rgb`; width and height default to 640×480. Depth contains metric camera
distance. `object_id` returns a uint32 NPY image whose nonzero values match selection IDs from
`list_objects` and `inspect_object`. `segmentation` returns int32 `(semantic ID, semantic type)`
pairs supplied by the adapter. MuJoCo supplies native geometry/site/flex/skin IDs and `mjtObj`
types; unknown semantics and the background use `(-1, -1)`. These semantic IDs are distinct from
selection object IDs.

## Measurements and vector actions

`set_ctrl` with `values` and `step` with `ctrl` use one validated `SetCtrlVector` command.
The vector length must equal the flat control dimension; all values must be finite. MuJoCo
applies each actuator's control limits before committing the vector. Invalid vectors leave
every control unchanged. Native integrations can submit `commands.SetCtrlVector(values)`
or call `adapter.set_ctrl_vector(values)` directly.

`get_state` includes `observations` when the adapter supports them. This record contains
`sensordata`, sensor names/types and their `data_adr`/`dim` slices, `actuator_force`, and
contacts. Contact records retain native `geom_ids`, `body_indices`, `flex_ids`, position,
distance, dimension, and the contact frame. Each six-component wrench is force followed by
torque at the contact point, acting on the second geometry. `wrench` uses contact coordinates;
`world_wrench` uses world coordinates. Negative geometry/body indices identify flex contacts.
Body indices can be matched against scene nodes; they are distinct from selection object IDs.

Observations are owned copies of the current native solver outputs. Reading does not step or
recompute physics. MuJoCo's own sensor computation stages therefore determine their sampling
time; call `mj_forward` yourself if you explicitly need recomputation after a manual state edit.
An adapter without this capability returns null. `get_state(observations=False)` omits the
measurement work. `step(..., observe=True)` includes the post-step state and measurements in
the same response. The existing `physics` snapshot representation remains separate.

## In-memory capture

The default capture transport writes a file. Set `"transport": "base64"` to return image bytes
in the JSON response without creating a file. `encoding` accepts `raw` (default, uncompressed
array bytes), `npy` (NumPy container), or `png` (RGB only). `shape`, NumPy `dtype` including byte
order, and top-left orientation describe the image. An in-memory request cannot specify an
`output` path. Base64 increases wire size; it removes disk I/O rather than providing zero-copy
transport. `capture_viewport` supports the same transport for viewport/window RGB.

For repeated local capture, allocate one `SharedImage` and use `capture_into`. It returns frame
metadata; pixel data is written directly into the caller's buffer without JSON image encoding
or a client-side image copy. RGB, metric depth, object IDs, and segmentation are supported.

```python
import numpy as np
from mojive import SharedImage

with SharedImage((480, 640, 3), np.uint8) as rgb:
    for _ in range(100):
        metadata = client.capture_into(rgb)
        image = rgb.array  # Zero-copy view, overwritten by the next capture into rgb.
        print(metadata["step"], image.mean())

with SharedImage((480, 640), np.float32) as depth:
    client.capture_into(depth, mode="depth")
```

Raw RPC clients send `{"transport": "shared_memory", "buffer": {"name": ..., "shape": ...,
"dtype": ...}}` with the usual capture dimensions and mode. The response echoes `buffer` with
`transport`, shape, dtype, orientation, and frame metadata. No pixel bytes travel over the socket.
Shared memory requires the same host and shared-memory namespace. Base64 remains available
when processes cannot map the same allocation.

The caller owns allocation and cleanup. Read only after the successful response and finish
consuming the view before the next write to the same buffer. Use `image.copy()` to retain a
frame, or separate buffers for concurrent consumers. If a request times out, discard that buffer:
the server may still be completing its write. Readers never unlink another process's allocation.
Closing the owner unlinks the POSIX name; existing local NumPy views retain their mapping until
released. Windows frees the allocation when its last mapping closes.

For presented RGB, use `client.capture_into(buffer, surface="viewport")` or `surface="window"`.
The buffer must match that surface's current pixel dimensions; reallocate after a resize.
GPU readback, orientation changes, and format conversion still have a cost. Shared memory
removes IPC pixel copies; it does not imply a zero-copy GPU-to-policy path.

```python
from mojive.capture import decode_image

rgb = client.capture_array(width=640, height=480)
depth = client.capture_array(mode="depth", encoding="npy")
payload = client.call("capture", {
    "mode": "rgb", "transport": "base64", "encoding": "png",
})
image = decode_image(payload)
print(payload["step"], payload["time"])
```

The regular Python `Viewer.capture_array(surface="scene")` returns an owned RGB array directly.
The MuJoCo `Renderer` and generic `SceneRenderer` also return arrays for RGB, depth, and IDs.

## Scene and presented capture settings

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
