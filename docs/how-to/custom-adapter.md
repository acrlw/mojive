# Custom scene adapter

A scene adapter connects stable structure and dynamic state to the viewer. This boundary supports
custom physics engines, procedural tools, replay sources, and remote publishers.

Implement these members first:

- `structure_revision`: increments after topology or resource changes;
- `scene_source()`: meshes, materials, hierarchy, cameras, and lights;
- `frame(needs)`: current transforms and requested dynamic diagnostics;
- `step()` and `reset()`: simulation control when `caps.simulation` is enabled.

Declare optional behavior through `AdapterCaps`. The session uses those capabilities to enable UI
and command paths.

## Minimal simulation adapter

```bash
uv run python examples/custom_adapter.py
```

```python
--8<-- "examples/custom_adapter.py"
```

Derive from `SceneAdapterBase` for the full editor: it supplies defaults for unsupported
operations. `SceneAdapter` describes that complete editor interface; `SceneProvider` describes
only `structure_revision`, `scene_source()`, and `frame(needs)`. Read-only consumers such as
`SceneRenderer.update_from()` accept the smaller protocol without simulation or authoring stubs.
Capabilities describe supported write-back; do not advertise a capability without implementing
its operations. Frames may reuse arrays until the next frame request; consumers retaining them
must copy the required data.

Register an external adapter factory in the process that will use it:

```python
from mojive import build, check_adapter, register_adapter
from my_engine import MyAdapter

register_adapter("my-engine", MyAdapter, label="My engine")
adapter = MyAdapter()
try:
    report = check_adapter(adapter)
finally:
    adapter.release()

with build("model.custom", adapter_name="my-engine", renderer="opengl") as viewer:
    viewer.run()
```

The factory takes no arguments; a closure can supply engine configuration. `make_adapter()`
loads an optional asset and releases the newly created adapter if loading fails. Registration
rejects duplicate or built-in names. `unregister_adapter(name)` removes a custom registration;
existing adapter instances remain caller-owned. Registrations are process-local, with no
implicit package imports or discovery across processes. An application CLI can register its
factories before calling `mojive.cli.main()`.

Use `check_adapter(adapter)` for a custom instance, or
`make adapter-conformance ADAPTER=toy` for a built-in adapter. Importing shared contracts,
`Scene`, or `SceneProvider` does not initialize physics, UI, or graphics packages.

Legacy metadata names retain their constructor compatibility: `JointInfo.body` is a body index,
`qpos_adr` and `qvel_adr` are starting addresses in their state arrays, and `ActuatorInfo.joint`
is a joint index. These lookup values are distinct from selection object IDs. New extension
names should use explicit `*_index`, `*_address`, and `object_id` terminology.

`ActuatorInfo.target_node_id` optionally identifies the scene node selected/focused by the Control
panel. Resolve it against the adapter's current `nodes()` structure and refresh it on rebuild;
this is neither an object ID nor a physics index. The default `-1` means no target is exposed.
Legacy joint metadata remains a fallback. MuJoCo provides joint, body, site, and slider-crank
site targets, falling back to the owning body when a visual group hides the leaf node.

## Capability and version contracts

Declare `AdapterCaps.model_formats` explicitly, for example `(".urdf",)`, together with
`asset_loading=True`. Extensions are lowercase and include the leading dot. The file menu,
file dialogs, drop handling and queued model imports offer only these formats. An empty tuple
exposes no model import entry. Mojive workspace files belong to `scene_files`, independently
of physics model formats. Do not infer format support from a backend's name.

`backend_version` is informational. `features` advertises exact extension contracts as
`(("engine.feature", 1), ...)`; `caps.supports("engine.feature", 1)` checks an exact revision,
not a minimum engine version. Boolean capabilities retain their revision-1 contracts. Unknown
features and revisions are unavailable. Determine support from the loaded engine instance;
newer engine releases do not automatically acquire features through a version comparison.

Session validates command requirements before physics fences and edit-history capture.
Rejected commands do not mutate the adapter; a rejected edit invalidates the active transaction.
UI callers suppress unsupported gestures before submission, including perturbation, so hovering
or dragging does not repeatedly produce unsupported-operation log messages. Programmatic callers
still receive a failed `CommandResult` for an unsupported request.

Generic pose, selection, scene appearance, and authoring commands retain existing Python names.
MJCF source replacement requires both `topology_editing` and `mujoco.mjcf` revision 1;
topology support alone does not imply XML editing. The existing structured component contract
requires `model.components` revision 1 and `topology_editing`; model-keyframe editing requires
`model.keyframe_edit` revision 1, `keyframes`, and `topology_editing`. These extensions refer to
the documented `ModelComponentInfo` / keyframe method contracts, not arbitrary engine data.
Adapters must implement those contracts completely before advertising them. New incompatible
payloads need a new extension revision or a distinct namespaced operation.

RPC discovery includes operation `version`, `method_versions`, `available_methods`, and
availability reasons. Clients should discover once per connection and refresh after document
or adapter changes. Requests may send `operation_version`; a mismatch is rejected before
execution. An omitted revision means 1. `RpcClient.call()` omits its default revision-1 field
for compatibility with existing revision-1 servers; callers can pass an explicit supported
revision after discovery. No client guesses an unknown operation's arguments or retries a
mutation after a version error.

## Independent display worlds

`mojive.adapters.WorldInstances(source, frame, offsets)` repeats rigid instances while sharing
mesh, texture, and material resources. Static geometry appears once. Its `set_poses()` accepts
positions `(worlds, moving_instances, 3)` and rotations `(worlds, moving_instances, 3, 3)` in
Z-up coordinates, adds display offsets, and reuses frame buffers. `pose_indices` identifies the
input template frame slots; `moving_instances` identifies the source instance order.

The adapter owns no simulation and never introduces cross-world contacts. A Python application
or remote publisher supplies complete batches and owns synchronization; do not modify arrays
concurrently with `frame()` consumption. Selection identifies a world; segmentation is
`(world_index, template_instance_index)`, with `-1` for shared static geometry. Deforming meshes,
site-driven poses, tendons, and flexible surfaces are rejected explicitly by this rigid adapter.


`write_ctrl` explicitly enables actuator control writes. Simulation metadata or a displayed
`ctrl` array does not imply write access; adapters implementing `set_ctrl` and
`set_ctrl_vector` must declare this capability. Read-only actuator rows remain visible but
cannot submit edits. MuJoCo and the workspace adapter already declare it.

An externally owned clock cannot accept local `SetSpeed` changes. Remote clock controls are
available only when all pause, resume, step, and reset contracts are negotiated; receiving
simulation frames remains independent of these write permissions.
