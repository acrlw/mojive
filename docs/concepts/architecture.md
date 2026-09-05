# Architecture

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

`SceneProvider` is the minimal read-only stream contract: a structure revision, a scene source,
and requested frames. `SceneRenderer` can consume this contract directly without `Session` or UI.
The full `SceneAdapter` interface adds editor operations; inheriting `SceneAdapterBase` supplies
defaults for unavailable capabilities. Factory registration and renderer selection live outside
these contracts.

`operations.py` defines public operations once: schemas, capability requirements, command
construction, and descriptions. `control_schema.py` provides reusable value contracts and cached
validation. `ControlApplication` coordinates operations against one Session. `control_rpc.py`
owns protocol envelopes, socket lifetime, deadlines, and viewer-thread queuing. Importing the
RPC client does not initialize graphics, load UI modules, or import the schema validator.
Native remote authoring messages share the catalog's validation and typed command construction;
the UI continues to submit typed Session commands. `scene_queries.py` provides world-pose queries
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

## Composition and export

`.mojive.json` is the editor document. It records model paths, model root transforms, resource
directories, edited model XML, and Mojive entities. MJCF export composes model specs and authored
entities into one formatted XML document, compiles it for validation, and writes it to disk.

## Coordinates

World coordinates use Z-up. Python matrices are row-major and store translation in
`matrix[:3, 3]`. Each renderer applies its upload convention at the GPU boundary. Render target
metadata carries vertical image orientation.

## Dependency boundaries

Shared contracts live in `types.py`, `commands.py`, `math3d.py`, and `adapters/base.py`. Render
code imports shared contracts. Adapter integrations own physics-specific code. UI modules depend
on session state and protocols. `tests/test_layering.py` enforces these boundaries.
