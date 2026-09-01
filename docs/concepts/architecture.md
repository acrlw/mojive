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
