# API map

The public package exports the common scene, rendering, adapter, remote, and recording types from
`mojive`. Advanced integrations may import the owning module directly.

| Module | Purpose | Primary interfaces |
|---|---|---|
| `mojive.types` | Backend-neutral value types | `CameraView`, `Light`, `Material`, `MeshData` |
| `mojive.adapters.base` | Adapter contracts | `SceneSource`, `SceneFrame`, `SceneAdapterBase` |
| `mojive.commands` | Typed application operations | `Command`, `Query`, scene and simulation commands |
| `mojive.scene` | Programmatic authored scenes | `Scene`, `SceneObject`, `SceneLight` |
| `mojive.session` | Application state and routing | `Session`, `PerturbState` |
| `mojive.renderer` | MuJoCo-compatible offscreen rendering | `Renderer` |
| `mojive.render.debugdraw` | Debug primitives and layers | `DebugDraw`, `Layer`, `Occlusion` |
| `mojive.remote` | Live structure, frame, and command transport | `SnapshotPublisher`, `RemoteSceneAdapter` |
| `mojive.recording` | Video and snapshot streams | `VideoRecorder`, `SnapshotWriter` |
| `mojive.control_rpc` | Local process control | `ControlServer`, `ControlService`, `RpcClient` |

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

Names exported from `mojive.__all__` form the supported public surface. Adapter
implementations and render-pass modules expose extension points with a narrower compatibility
scope. Shader resources and UI internals are implementation details.
