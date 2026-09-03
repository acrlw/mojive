# MuJoCo rendering

`Renderer` matches the shape and control flow of `mujoco.Renderer` while sending scene data through
the selected Mojive render backend. One renderer instance owns reusable targets for RGB, metric
depth, and segmentation output.

## RGB, depth, and segmentation

```bash
uv run python examples/mujoco_render.py assets/test_scene.xml \
  --output output/examples/render
```

This example exposes `--backend opengl|wgpu` for convenience. Library code selects the same backend
with `MOJIVE_BACKEND`. On Linux, offscreen OpenGL creates an EGL context by default.
If initialization fails, see [context troubleshooting](../reference/configuration.md#render-backend-requirements):
`MOJIVE_GL=glfw` is an alternative on a desktop, not on a display-free server.

```python
--8<-- "examples/mujoco_render.py"
```

## Multiple model cameras

The same renderer can update and render each fixed camera without rebuilding model structure:

```bash
uv run python examples/multi_camera_render.py assets/showcase.xml \
  --output output/examples/cameras
```

```python
--8<-- "examples/multi_camera_render.py"
```

`update_scene()` consumes the current `MjData`, camera selection, and optional `MjvOption`.
`render()` copies the selected output into a new array or a caller-provided `out` array.

## Record a rollout

`VideoRecorder` is exported directly from `mojive`. Use one recorder around your render loop
and call `video.append(renderer.render())` at each video sample; frames are streamed, not stored
as a growing Python list. No custom FFmpeg pipe is needed in application code.

```bash
uv run python examples/mujoco_video.py assets/test_scene.xml \
  --frames 90 --fps 30 --output output/examples/rollout.mp4

# Optional RGB-only subtitle with the actual simulation time:
uv run python examples/mujoco_video.py assets/test_scene.xml \
  --label "Policy A" --output output/examples/annotated-rollout.mp4
```

The example samples the initial state at time zero, then the first physical state at or after
each `index / fps` target. It never changes `model.opt.timestep`. For example, 500 physics steps
per second and 30 video frames per second require multiple simulation steps between most frames,
not one recorded frame per physics step. Sampling is quantized to the physical timestep; video
playback duration is `frames / fps`. Policy/controller updates belong inside the physics loop.

```python
--8<-- "examples/mujoco_video.py"
```

The default MP4 uses `yuv420p`; `--pixel-format yuv444p` opts into full chroma resolution at the
cost of player compatibility. Odd dimensions are padded, not stretched. See
[video encoding settings](../reference/configuration.md#video-encoding).

Pillow annotation is deliberately an example-level RGB operation. Keep raw RGB if it is also
used for observations, and never annotate metric depth or segmentation IDs. For labels attached
to objects in world space, use [DebugDraw text](../how-to/debug-draw.md).

## Compose several model files

Python model composition already uses `WorkspaceAdapter.add_scene_model(path, position, rotation)`.
See [compose_scene.py](https://github.com/acrlw/mojive/blob/main/examples/compose_scene.py) for an
executable recipe that assigns a root transform to each MJCF/URDF model and saves a
`.mojive.json` workspace or portable MJCF. No hand-written `<include>` file is required.

```bash
uv run python examples/compose_scene.py assets/test_scene.xml assets/test_scene.urdf \
  --spacing 2.5 --output output/examples/workcell.xml
uv run python examples/mujoco_video.py output/examples/workcell.xml \
  --output output/examples/workcell.mp4
```

This assembles and compiles one physical model. It does **not** accept independent running
`MjModel`/`MjData` pairs as render-only instances, and is not an RL batch renderer. `Renderer`
still binds one compiled model and accepts data from that model. For independent policy runs,
render each state with its own compatible renderer and concatenate the RGB images for a
side-by-side video; that is image composition, not a shared 3D world.
