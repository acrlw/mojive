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
uv run --no-sync python examples/custom_adapter.py
```

```python
--8<-- "examples/custom_adapter.py"
```

Derive from `SceneAdapterBase` for the full editor: it supplies defaults for unsupported
operations. `SceneAdapter` describes that complete editor interface; `SceneProvider` describes
only `structure_revision`, `scene_source()`, and `frame(needs)`. Read-only consumers such as
`SceneRenderer.update_from()` accept the smaller protocol without simulation or editing stubs.
Capabilities describe supported write-back; do not advertise a capability without implementing
its operations. Frames may reuse arrays until the next frame request; consumers retaining them
must copy the required data.

### Choose the contract your consumer actually needs

The full editor protocol is a compatibility facade composed from the following structural
protocols in `mojive.adapters`. They document dependencies; inheriting them does not enable
features. Existing adapters can continue inheriting only `SceneAdapterBase` and overriding
the operations they advertise.

| Contract | Responsibility |
|---|---|
| `SceneProvider` | revision, stable structure and requested frames |
| `SceneInspection` | provider plus object metadata, cameras and timing |
| `SimulationControl` | stepping, controls, state snapshots and physical perturbation |
| `SceneDocuments` | compatibility group for document I/O and edit snapshots |
| `SceneEditing` | scene object and appearance writes |
| `ModelEditing` | compatibility group for model composition, topology, properties and assets |
| `SceneRuntime` | provider plus preparation, hierarchy, camera hint and lifetime |
| `SimulationAccess` | optional simulation observations, control and state write-back |
| `SceneAppearance` | scene appearance, camera views and visual groups |
| `SceneAuthoring` | backend-neutral object, light and camera authoring |
| `ScenePersistence` | document I/O, resource roots and restorable edit state |
| `ModelComposition` | attached model identities and world placement |
| `ModelTopology` | model-source topology and declaration edits |
| `ModelProperties` | authored body, joint, site and geometry properties |
| `ModelAssets` | model-local resources and material binding |
| `KeyframeCatalog` | preset metadata without playback or editing |
| `KeyframePlayback` | metadata and loading a preset |
| `KeyframeEditing` | compatibility group for preset playback, properties and writes |
| `ModelKeyframes` | compatibility aggregate of playback and editing |

A read-only consumer should accept `SceneProvider`. A model browser can accept
`ModelComposition`; it does not need a physics engine or the whole editor interface.
Session and Workspace coordinate multiple responsibilities and still use `SceneAdapter`.
An `isinstance(adapter, SceneAdapter)` check only establishes structural membership: inherited
unsupported methods satisfy that check. Use `caps.supports(...)` to determine availability.
Implementing keyframe playback does not require advertising `model.keyframe_edit`. The Session
metadata index consumes `KeyframeCatalog`; loading and editing controls check their capabilities
independently. Basic timeline selection and navigation remain available without either write capability.

Document operations use `caps.supports("scene_new")`, `"scene_open"`, and `"scene_save"`.
Each flag defaults to `None`, which inherits legacy `scene_files`; an explicit Boolean overrides
that fallback. Partial implementations should set the individual flags and leave `scene_files`
false. File support is independent from `scene_authoring` and `edit_history`. Declare
`edit_history=True` only when capture and restore preserve the entire editable primary state;
Workspace trusts this declaration rather than cloning a document to probe availability.

`check_scene_provider(provider)`, available from `mojive`, validates the minimal stream independently
of UI, simulation and authoring. `check_adapter(adapter)` additionally validates editor metadata
and reports advertised operations that are missing or still use unsupported base defaults.
Neither check invokes editing writes. Empty inventories are valid, and a successful check does
not prove that a custom implementation preserves atomicity; exercise its writes and rollback
with representative engine state.

A non-simulating inspection adapter can pass `check_adapter()` without simulation-control
stubs, provided it does not advertise unavailable writes. `SceneDocuments` shares the persistence
contract, and `ModelEditing` groups the model interfaces above.

Session-owned changes to adapter values are exposed as `session.scene_overrides`, whose type is
`mojive.session.SceneOverrides`. `authored_overlay` and `AuthoredSceneOverlay` remain compatible
names. The legacy capability field `scene_authoring` means support for scene object creation,
removal and editing; it does not imply simulation or model topology editing.

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

Workspace document replacement uses the optional `DocumentCheckpoint` protocol in
`mojive.adapters`: `capture_document_state()` and `restore_document_state(state)`.
Declare `document_checkpoints=True` when both hooks can restore a replaced document.
Workspace opening additionally requires the primary adapter to support `scene_new`; Workspace
saving requires a primary model-composition catalog. A primary
standalone `scene_save` does not suffice: Workspace JSON cannot preserve an arbitrary primary
scene or another Workspace's independent authored entities. Those compositions remain usable
in memory but cannot be saved through the outer Workspace. Unsupported operations fail before writes.
This extension is separate from the full compatibility facade. Checkpoints include backing file paths and ownership;
ordinary edit snapshots intentionally retain the current save destination during Undo.
Workspace restores the primary state, authored entities and resource roots if loading fails. Stored MJCF sources require
the `mujoco.mjcf` contract; a rejected source write fails the load instead of silently omitting
the edited source. The previously open document path is retained on failure.

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

Generic pose, selection, scene appearance, and editing commands retain existing Python names.
MJCF source replacement requires both `topology_editing` and `mujoco.mjcf` revision 1;
topology support alone does not imply XML editing. The existing structured component contract
requires `model.components` revision 1 and `topology_editing`; model-keyframe editing requires
`model.keyframe_edit` revision 1, `keyframes`, and `topology_editing`. These extensions refer to
the documented `ModelComponentInfo` / keyframe method contracts, not arbitrary engine data.
Adapters must implement those contracts completely before advertising them. New incompatible
payloads need a new extension revision or a distinct namespaced operation.

### Physical grab points

`perturb=True` retains the four-argument `apply_perturb` method. Adapters that can
apply force at a selected point additionally advertise `("physics.perturb_point", 1)`
and implement `PointPerturbation.apply_perturb_at_point`. The target position and
rotation describe the body pose in world coordinates; `local_position` identifies
the grab point in the body frame and remains fixed throughout the gesture.
`clear_perturb` ends either form of perturbation. Workspace forwards the extension
only when its primary adapter advertises it.

The viewer sends the grab point to supporting adapters and keeps the original call
for other adapters. An explicit `Perturb(local_position=...)` command fails before
physics mutation when the extension is unavailable. MuJoCo uses the selected point
for effective mass, damping and moment arm, with the model's native viewer stiffness
parameters. A zero-mobility pivot uses MuJoCo's `localmass=1` convention.

### Local geometry scale

Advertise `AdapterCaps.write_scale` and mark each supported geometry `SceneNode.scalable`.
A single-geometry object's parent can also be scalable; Session resolves it to that geometry,
so both Inspector selections share one pending value. Implement `set_scale(node_id, factors)`
to atomically bake positive local XYZ factors into source dimensions, preserve world position
and rotation, and advance `structure_revision`. Return `False` without mutations for unsupported
targets. This contract scales geometry in its local frame, not articulated subtrees or shear.
Only mark shapes that can preserve the requested nonuniform scaling.

`SetScale` submitted directly applies its factors once. The UI coalesces it in `ModelEditDraft`,
previews the render sizes, and applies one transaction on confirmation. The committed transform
has identity scale; saved scenes and future rebuilds use the baked dimensions. `SceneObject.scale`
provides the equivalent operation for programmatic scenes.

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
