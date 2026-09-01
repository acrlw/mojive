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
