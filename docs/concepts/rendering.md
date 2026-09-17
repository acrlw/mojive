# Rendering

## Public entry points

`SceneRenderer` renders backend-neutral scene data. `Renderer` provides the MuJoCo-compatible
`update_scene(data)` interface. Both accept `renderer="opengl"` or `"bgfx"`.
Explicit selection overrides `MOJIVE_RENDERER`; `MOJIVE_BACKEND` is a legacy fallback.
Interactive builders use the same selection policy. Scene adapters are selected independently.

```python
from mojive import RenderProduct, Scene, SceneRenderer

scene = Scene()
scene.box(color=(0.1, 0.6, 0.9, 1.0))
with SceneRenderer(scene.source, width=640, height=480) as renderer:
    renderer.update(scene.frame)
    rgb = renderer.render()
    depth = renderer.render(product=RenderProduct.METRIC_DEPTH)
    object_ids = renderer.render(product=RenderProduct.OBJECT_ID)
```

`set_scene(source)` replaces stable structure; `update(frame, camera=...)` supplies dynamic data.
`update_from(provider)` follows a `SceneProvider` structure revision and requests poses and
deformables by default. The caller owns stepping and provider lifetime. Keep graphics operations
on the creating thread and release resources with a context manager.

## Image contracts

All public image arrays use a top-left origin.

| Product | Shape | Type | Background |
|---|---|---|---|
| RGB | `(height, width, 3)` | `uint8` | Rendered background |
| Metric depth | `(height, width)` | `float32` | Camera far plane |
| Object ID | `(height, width)` | `uint32` | `0` |
| Segmentation | `(height, width, 2)` | `int32` | `(-1, -1)` |

Metric depth is measured along the camera's forward axis in world units. Segmentation pairs come
from adapter metadata; they are distinct from selection object IDs. `out=` accepts a writable
array with the exact shape and dtype, including strided destinations. `resize()` updates target
dimensions and camera aspect.

`render_async()` returns a `concurrent.futures.Future`. Retain exclusive access to a supplied
`out` array until completion. OpenGL currently returns a completed future; native bgfx uses
bounded asynchronous readback and cancels outstanding results on resize or scene replacement,
retaining staging storage until the GPU finishes. Handle future errors and backpressure explicitly;
asynchronous output is not
unbounded storage or a multi-world rendering API.

## Pipeline and ownership

The shared pass plan schedules only products requested by the caller. A viewport needs color
and object IDs; an RGB capture does not automatically need picking. Depth and segmentation use
typed export targets.

```text
SceneSource + SceneFrame
         ↓
scene preparation and revision tracking
         ↓
requested shadow / reflection / opaque / identity / export passes
         ↓
skybox / tendons / transparency / debug / outline / gizmo
         ↓
presentation or image readback
```

`python/render/` owns shared contracts and preparation. `opengl/` and `native/` own
backend implementations; `cpp/src/` contains native resources and GPU submission. Render modules
consume scene contracts, without depending on UI panels or a physics implementation.

Opaque instances are grouped by shared mesh and texture bindings. Color and shading values stay
in per-instance data; different logical materials need not create separate draws. Transparent
instances retain camera-dependent ordering. Pose, visual and identity revisions allow independent
buffer updates, and unchanged shadow/reflection products are reused. Custom sources with revision
zero use conservative change detection. Do not bypass invalidation to improve a benchmark.

Adapters use `FrameNeeds` to avoid extracting invisible contacts, sensors, tendons or other
optional diagnostics. Normal UI rendering samples the scene texture on the GPU; full-frame
readback is requested by capture, recording and explicit image consumers.

## Backend differences

OpenGL is the default for fast application and interaction validation. bgfx is the native
rendering path; both backends implement the same public scene and image contracts.

| Backend | Implementation | Setup |
|---|---|---|
| OpenGL | Python/ModernGL, OpenGL 3.3+ core | Default after `make setup` |
| bgfx | Private C++ runtime, platform graphics API | [Native backend build](../how-to/native-viewer.md) |

OpenGL can use shared multisampled color/ID attachments when supported, or separate targets when
integer MSAA is unavailable. On drivers limited to one-pixel native lines, wide lines expand to
triangles. Native bgfx uses typed export targets for data products. Backend-specific details
stay behind the shared image contracts.

Mojive-authored scenes shade in linear light and encode for display after tone mapping.
MuJoCo sources select `mujoco-classic` shading to preserve their display-domain lighting model.
Textures, transparency, selection and reflection must follow the selected color pipeline.
Directional and local shadows, dynamic meshes, planar reflections and physics diagnostics are
controlled by shared render flags; consult the [CLI reference](../reference/cli.md) for names.

World coordinates are Z-up. Python matrices are row-major with translation at `matrix[:3, 3]`.
Upload-boundary conversion and render-target orientation belong to the backend. Callers do not
transpose scene matrices or infer image orientation from the graphics API.

## UI and diagnostic drawing

ImGui draw lists render panel controls and their custom glyphs. `geometry2d` supplies reusable
CPU paths and tessellation. [Canvas2D](../how-to/canvas2d.md) exposes retained shapes in a world
plane through the existing DebugDraw GPU pass. It shares buffers, layers and occlusion behavior
with [3D diagnostics](../how-to/debug-draw.md); there is no separate Draw2D renderer to initialize.

Debug layers use stable IDs for replacement and removal. Screen-width primitives and world-size
geometry have different units; retain the chosen width model throughout an object's lifetime.
Overlay drawing must preserve the ID attachment used for picking and segmentation.

## Verification and profiling

Required checks are listed in the [verification matrix](../guides/testing.md#change-mapping).
Render examples and comparisons with `make scene-renderer`, `make showcase`, `make parity`,
`make calibrate`, or `make native-parity`. Use `make renderer-benchmark` for timing measurements.
Save reports in `output/`.

Compare identical scenes, poses, cameras, resolution, sampling and output products. Report cold
startup separately from warmed frames, and CPU submission separately from completed GPU output
and readback. Run benchmarks without competing GPU jobs. GPU pass timers can be approximate on
tile-based hardware; absent timing data is not zero cost. Historical machine timings are not
portable performance guarantees.
