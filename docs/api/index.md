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
| `mojive.scene_renderer` | Backend-neutral offscreen rendering | `SceneRenderer` |
| `mojive.renderer` | MuJoCo-compatible offscreen rendering | `Renderer` |
| `mojive.render.debugdraw` | Debug primitives and layers | `DebugDraw`, `Layer`, `Occlusion` |
| `mojive.drawing.curves` | Shared sampled paths and stroke profiles | `smooth_rect_points`, `smooth_capsule_points`, `arrow_points`, `capped_polyline_points` |
| `mojive.drawing.drag_link` | Hollow-connector geometry | `drag_link_field`, `smooth_drag_link_mesh` |
| `mojive.ui.draw2d` | UI overlay extension protocol | `Draw2D`, `ImguiDraw2D` |
| `mojive.canvas2d` | Layered 2D physics and geometry diagnostics | `Canvas2D`, `CanvasLayer2D` |
| `mojive.capture` | Interactive screenshot and recording contracts | `CaptureSurface`, `RecordingInfo` |
| `mojive.remote` | Live structure, frame, and command transport | `SnapshotPublisher`, `RemoteSceneAdapter` |
| `mojive.recording` | Video and snapshot streams | `VideoRecorder`, `SnapshotWriter` |
| `mojive.control_rpc` | Local process control | `ControlServer`, `ControlService`, `RpcClient` |
| `mojive.control` | Session application operations | `ControlApplication` |
| `mojive.operations` | Operation discovery and validation | `OPERATIONS`, `Operation` |

## Integration paths

| Goal | Start with | Example |
|---|---|---|
| Build a scene in Python | `Scene`, `build_scene` | `examples/programmatic_scene.py` |
| Render authored or custom scene frames | `SceneRenderer` | `examples/offscreen_scene.py` |
| Render MuJoCo arrays | `Renderer` | `examples/mujoco_render.py` |
| Record a video or annotate RGB | `VideoRecorder` | `examples/mujoco_video.py` |
| Debug a 2D physics or geometry algorithm | `Canvas2D` | `examples/canvas2d.py` |
| Control MuJoCo state | `Session`, typed commands | `examples/mujoco_control.py` |
| Compose MJCF and URDF | `WorkspaceAdapter` | `examples/compose_scene.py` |
| Publish a live scene | `SnapshotPublisher` | `examples/remote_publish.py` |

The [examples guide](../guides/examples.md) explains how to run each workflow. The module pages
list signatures, types, and public members generated from the source documentation.
For custom widgets, icons, or screen diagnostics, use the [drawing extension guide](../how-to/ui-drawing.md)
to choose the surface and its state owner before changing a renderer or UI controller.

## Stability

Names exported from `mojive.__all__` form the supported public surface. Adapter
implementations and render-pass modules expose extension points with a narrower compatibility
scope. Shader resources and UI internals are implementation details.
