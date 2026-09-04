# Tutorials and examples

The `examples/` programs are executable API documentation. Tutorial pages embed their source
directly, so the guide and runnable code stay synchronized.

The [runnable examples README](https://github.com/acrlw/mojive/tree/main/examples) contains exact
commands, required assets, and generated outputs. Commands use `uv run`, so activating `.venv` is
not required.

## Start here

| Goal | Tutorial | Primary interfaces |
|---|---|---|
| Create geometry, cameras, and lights | [Programmatic scene](../tutorials/programmatic-scene.md) | `Scene`, `build_scene` |
| Render MuJoCo arrays | [MuJoCo rendering](../tutorials/mujoco-rendering.md) | `Renderer` |
| Record a rollout or add video subtitles | [Rollout video](../tutorials/mujoco-rendering.md#record-a-rollout) | `VideoRecorder`, Pillow |
| Publish or record dynamic frames | [Remote viewing and replay](../tutorials/remote-viewing.md) | `SnapshotPublisher`, `SnapshotWriter` |
| Integrate another physics engine | [Custom scene adapter](../how-to/custom-adapter.md) | `SceneAdapterBase`, `SceneSource`, `SceneFrame` |
| Add diagnostics and labels | [Debug drawing](../how-to/debug-draw.md) | `DebugDraw`, `Layer`, `Occlusion` |
| Debug 2D physics and geometry | [2D diagnostic canvas](../how-to/debug-draw.md#2d-physics-and-geometry-diagnostics) | `Canvas2D`, `CanvasLayer2D` |
| Automate a running process | [Local RPC control](../how-to/rpc-control.md) | `RpcClient`, `ControlService` |

## Example catalog

| Example | Result |
|---|---|
| `programmatic_scene.py` | interactive scene without a physics backend |
| `debug_draw.py` | retained lines, arrows, points, frames, and labels |
| `canvas2d.py` | layered 2D physics and geometry diagnostics |
| `custom_adapter.py` | backend-neutral simulation integration |
| `mujoco_render.py` | RGB, metric depth, and segmentation output |
| `mujoco_video.py` | streamed MP4 rollout with optional RGB label/timestamp |
| `multi_camera_render.py` | one PNG per MuJoCo camera |
| `mujoco_control.py` | qpos editing and deterministic stepping |
| `compose_scene.py` | combined MJCF/URDF workspace or portable MJCF |
| `remote_publish.py` | live latest-state publisher for independent viewers |
| `record_replay.py` | `.fvs` snapshot recording |
| `control_client.py` | persistent local RPC automation |
