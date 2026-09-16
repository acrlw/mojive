# Architecture

## Application boundary

Mojive provides scene inspection, rendering, simulation integration, state/control APIs, and
image transport. Applications built on these interfaces own controller mappings, teleoperation
policies, inverse kinematics, episode/reward tracking, custom plots, trajectory dataset formats,
and Gymnasium integration. These are application responsibilities, not missing viewer modules.

## Data flow

An adapter publishes a `SceneSource` when stable structure changes and a `SceneFrame` for dynamic
state. `Session` owns selection, editor history, overrides, simulation control, and command
routing. A rendering backend consumes the source and frame. The UI reads session protocols and
submits typed commands.

```text
model / physics / remote stream
             │
          adapter
             │ SceneSource + SceneFrame
             ▼
          Session ◀──── typed commands
             │
        render backend
             │
          viewport
```

## Ownership

| Component | Owns |
|---|---|
| `SceneSource` | meshes, materials, entity identities, stable render metadata |
| `SceneFrame` | poses, controls, contacts, tendons, sensors, dynamic meshes |
| `Session` | selection, history, editor state, command routing |
| Mojive `Scene` | authored geometry, cameras, lights, materials, environment |
| Physics adapter | simulation state, topology, capability-specific write-back |
| Render backend | GPU resources, render passes, output images |

`WorkspaceAdapter` combines one physics/model adapter with a Mojive-authored scene. Model topology
and dynamic state remain in the primary adapter. Editor entities remain backend-neutral.

Interactive viewers use a concurrent MuJoCo simulation driver by default. Native stepping owns
one `MjData`; the UI reads a completed snapshot from a three-slot pool. Publishing replaces only
pending snapshots, never a slot still used by the reader. Window events, ImGui, scene conversion,
and GPU submission stay on the main thread. Adapters without a simulation driver retain the
serial path, and externally clocked adapters never acquire a second physics owner.

`Session.submit()` fences the active native batch before commands mutate model or physics state
or capture edit history. Camera and selection commands only touch Session state and bypass this
fence. During a fenced edit, authoritative state is temporarily bound to the adapter; resuming
publishes an immutable initial snapshot before stepping again. Model replacement invalidates the
old buffers. Pause, replay, reset, step-back, and shutdown use the same ownership boundary.
Control and perturbation updates preserve the running clock; resuming from pause, changing speed,
or restoring a different simulation time establishes a new clock origin.

Publication batches target at most eight milliseconds of wall time, subject to the model's
timestep, and are limited to eight steps and approximately eight milliseconds of measured native
work. One expensive physics step cannot be preempted. This is cooperative scheduling, not a
hard real-time guarantee. Worker
failures pause the session and report an error; an explicit Play can then resume synchronously.
Displayed-state history and recorded takes sample completed frames, not every integrator step.

`ViewerConfig(threaded_physics=False)` selects the serial viewer path. Direct `Session` instances
remain synchronous unless enabled with `set_threaded_physics(True)`. Ticks without `wall_dt`
fence background work and execute explicit synchronous steps. Offline rendering and caller-owned
MuJoCo workflows retain their existing state ownership.

`SceneProvider` is the minimal read-only stream contract: a structure revision, a scene source,
and requested frames. `SceneRenderer` can consume this contract directly without `Session` or UI.
The full `SceneAdapter` interface adds editor operations; inheriting `SceneAdapterBase` supplies
defaults for unavailable capabilities. Factory registration and renderer selection live outside
these contracts.

`SceneAdapter` combines cohesive structural contracts for runtime, simulation, appearance,
authoring, persistence, model composition, topology, properties, assets and keyframes.
Consumers can depend on one group; capability checks remain the runtime authority.
`check_scene_provider` validates the minimal stream, while `check_adapter` adds editor metadata
and advertised-operation diagnostics. A protocol declaration does not imply write support.

Timeline coordinates, marker queries and decimation live in `interaction/timeline.py` without
ImGui or Session dependencies. The timeline controller owns disposable indexes keyed by model/content revision
and caches only the latest view. Range searches and marker hit tests use binary search; dense
draw decimation skips occupied pixel buckets. `interaction/timeline_edit.py` owns transient gesture state and edit decisions;
`ui/keyframe_editor/controller.py` coordinates that state with Session commands, while toolbar,
track and property modules handle rendering and input sampling. Drag previews overlay one marker on a cached view.
`session/keyframes.py` owns a content-revisioned catalog with stable per-model views and ID lookup,
so appearance changes do not invalidate preset metadata. Session routes committed edits, history
and authoritative scene/take data. Shared transform refusal policy lives in `ui/edit_policy.py`; gizmos do not import
Inspector panels.

Interactive video uses `capture/video_queue.py` to hand owned RGB buffers to one encoder worker.
Capture and graphics calls remain on the viewer thread. Two reusable buffers bound queue memory;
full buffers apply backpressure, and finalization drains accepted frames and reports failures.
The first worker write must succeed before an automatic simulation start. UI stop requests enter
`finalizing` while the worker drains; public stop and viewer release wait for completion. Worker
failures are checked even while paused. A stalled encoder is killed after 30 seconds so its pipe
writer can exit and report failure. The synchronous `VideoRecorder` API and offline take export
retain their ordered execution model.

`control/operations.py` defines public operations once: schemas, capability requirements, command
construction, and descriptions. `control/schema.py` provides reusable value contracts and cached
validation. `ControlApplication` coordinates operations against one Session. `control/rpc/`
separates protocol envelopes, client and server transport, deadlines, bounded diagnostics,
and viewer-thread scheduling. `RpcLimits` defines admission and per-frame work budgets. Importing the
RPC client does not initialize graphics, load UI modules, or import the schema validator.
Native remote authoring messages share the catalog's validation and typed command construction;
the UI continues to submit typed Session commands. `scene/queries.py` provides world-pose queries
without importing UI controllers.

Capture camera state is independent from the viewport camera recorded by Session. The UI
owns gestures and presentation; an asynchronous viewport capture finishes after presentation.
Local RPC translates requests into application operations. `SessionCapture` refreshes the
Session's composed source and frame, and owns a cached `SceneRenderer`. A standalone service uses
one graphics worker; an attached viewer captures on its UI thread. Socket workers never own or
migrate graphics resources. Capture does not reconstruct a physics adapter or read raw model
state. Adapters publish semantic segmentation pairs; composition retains them alongside selection
object IDs. These identities have different meanings and remain separate.

`history.py` owns retained Undo/Redo records and shared-resource memory accounting. `bounds.py`
owns instance geometry bounds and batched scene framing. Scene snapshots share immutable mesh and
texture storage while copying mutable authoring state. Undo/Redo and reload restore the same
`Scene` owner; new/open document operations establish a new document owner.
History also snapshots Session-owned appearance overrides. Workspace advertises edit history only
when its primary adapter supplies a restorable edit snapshot; otherwise Undo/Redo and atomic edits
are unavailable. A successful history operation therefore restores both model and authored state.

Model recompilation can renumber hierarchy entries and rendered object IDs. Session preserves
selection of editable model elements by model, type and `SceneNode.source_name` (or `name` when
no separate source name is supplied), requiring an unambiguous match. This keeps composed display
names separate from model-local authoring names;
LINK/ROBOT are two presentations of the same body. An exact history restoration also restores
the selected hierarchy entry, including joints with no rendered object ID. External integrations
must refresh retained indices after structure changes.

## Composition and export

`.mojive.json` is the editor document. It records model paths, model root transforms, resource
directories, edited model XML, and Mojive entities. MJCF export composes model specs and authored
entities into one formatted XML document, compiles it for validation, and writes it to disk.

## Coordinates

World coordinates use Z-up. Python matrices are row-major and store translation in
`matrix[:3, 3]`. Each renderer applies its upload convention at the GPU boundary. Render target
metadata carries vertical image orientation.

## Dependency boundaries

Shared types, commands, math functions, and adapter interfaces are defined in `types.py`,
`commands.py`, `math3d.py`, and `adapters/base.py`. Render modules import these shared definitions.
Physics-specific code is implemented in adapters. UI modules depend
on session state and protocols. `tests/test_layering.py` enforces these boundaries.

For overlays, `geometry2d/curves.py` computes paths and stroke profiles; `geometry2d/drag_link.py`
computes the implicit hollow-connector field and mesh. Neither module imports UI or backend code.
`ui/paint_protocol.py` defines the immediate UI contract; `ui/imgui_draw.py` adapts it
to ImGui. `render/canvas.py` provides the retained Canvas2D API over DebugDraw.
`geometry2d/` performs CPU path sampling and triangulation; the shared debug pass renders its output.
Widgets determine placement and handle input. Retained debug layers manage identifiers, lifetime,
budgets, and stream packing. The [UI drawing guide](../how-to/ui-drawing.md) describes these
interfaces and includes examples.

### Deferred viewer document operations

Viewer RPC load, reload, open, save and MJCF-source replacement use the same exclusive worker as
UI model loading. Session ownership stays with that worker until its UI-thread completion;
subsequent RPC requests remain queued and check their preconditions when execution starts.
The client receives the original command result after completion. Headless services and direct
Python document methods retain synchronous semantics. Native compilation is not preempted once
started. Closing the viewer completes queued requests with an error and joins started work;
a write that already committed returns its actual result.
