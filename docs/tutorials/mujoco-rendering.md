# MuJoCo rendering

`Renderer` matches the shape and control flow of `mujoco.Renderer` while sending scene data through
the OpenGL render pipeline. One renderer instance owns reusable targets for RGB, metric depth, and
segmentation output.

## RGB, depth, and segmentation

```bash
.venv/bin/python examples/mujoco_render.py assets/test_scene.xml \
  --output output/examples/render
```

Select WebGPU with `--backend wgpu`. On Linux, the OpenGL backend creates an EGL context for
offscreen rendering by default.

```python
--8<-- "examples/mujoco_render.py"
```

## Multiple model cameras

The same renderer can update and render each fixed camera without rebuilding model structure:

```bash
.venv/bin/python examples/multi_camera_render.py assets/humanoid.xml \
  --output output/examples/cameras
```

```python
--8<-- "examples/multi_camera_render.py"
```

`update_scene()` consumes the current `MjData`, camera selection, and optional `MjvOption`.
`render()` copies the selected output into a new array or a caller-provided `out` array.
