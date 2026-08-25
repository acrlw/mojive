# API map

The public package exports the common scene, rendering, adapter, remote, and recording types from
`forge_viewer`. Advanced integrations may import the owning module directly.

| Module | Purpose | Primary interfaces |
|---|---|---|
| `forge_viewer.types` | Backend-neutral value types | `CameraView`, `Light`, `Material`, `MeshData` |
| `forge_viewer.adapters.base` | Adapter contracts | `SceneSource`, `SceneFrame`, `SceneAdapterBase` |
| `forge_viewer.commands` | Typed application operations | `Command`, `Query`, scene and simulation commands |
| `forge_viewer.scene` | Programmatic authored scenes | `Scene`, `SceneObject`, `SceneLight` |
| `forge_viewer.session` | Application state and routing | `Session`, `PerturbState` |
| `forge_viewer.renderer` | MuJoCo-compatible offscreen rendering | `Renderer` |
| `forge_viewer.render.debugdraw` | Debug primitives and layers | `DebugDraw`, `Layer`, `Occlusion` |
| `forge_viewer.remote` | Live structure, frame, and command transport | `SnapshotPublisher`, `RemoteSceneAdapter` |
| `forge_viewer.recording` | Video and snapshot streams | `VideoRecorder`, `SnapshotWriter` |

## Integration paths

| Goal | Start with | Example |
|---|---|---|
| Build a scene in Python | `Scene`, `build_scene` | `examples/programmatic_scene.py` |
| Render MuJoCo arrays | `Renderer` | `examples/mujoco_render.py` |
| Control MuJoCo state | `Session`, typed commands | `examples/mujoco_control.py` |
| Compose MJCF and URDF | `WorkspaceAdapter` | `examples/compose_scene.py` |
| Publish a live scene | `SnapshotPublisher` | `examples/remote_publish.py` |

The [examples guide](../guides/examples.md) explains how to run each workflow. The module pages
list signatures, types, and public members generated from the source documentation.

## Stability

Names exported from `forge_viewer.__all__` form the supported public surface. Adapter
implementations and render-pass modules expose extension points with a narrower compatibility
scope. Shader resources and UI internals are implementation details.
