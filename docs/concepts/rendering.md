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

## Adaptive mesh detail

Native bgfx supports opt-in `RenderFlag.MESH_LOD` (`mesh_lod`). Enable it with
`renderer.set_flag("mesh_lod", True)`, the Settings render flags, or CLI `--enable-render mesh_lod`.
Agents can use these public Python/CLI paths. The existing RPC `set_render_flag` controls capture
flags, not viewport flags, and does not expose LOD. OpenGL does not advertise this capability.
It remains off by default. Opted-out scenes do not create LOD jobs, build level lists, or
run detail selection in color or shadow passes. Small scenes usually need no LOD; keep
original geometry unless repeated detailed meshes are a measured rendering bottleneck.

The renderer generates compact shared levels with the pinned meshoptimizer library on one CPU
worker. GPU uploads happen on the render owner, with a preparation time budget checked between
meshes. The initial frames use available geometry while preparation completes; cold startup can
still be expensive. This budget is not a hard frame deadline: one mesh upload or one
meshoptimizer call cannot be interrupted. Preparation also consumes CPU alongside the app.
Rigid instances share their levels. Vertex-deformed meshes, selected objects and wireframe
views retain original geometry.

Disabling the flag restores original geometry immediately, without waiting for a running CPU
job. When the last enabled scene stops using a mesh, its queued work is canceled and prepared
CPU/GPU levels are released. An active simplification stops between meshoptimizer calls;
completed worker threads are reaped on subsequent rendering. Enabled peer scenes retain their
shared levels. Re-enabling after the last user opts out prepares levels again instead of
retaining a hidden cache. No disk cache or asset rewriting is involved.

`make mesh-lod-example` runs the [self-contained example](../guides/examples.md) with LOD off;
add `ARGS='--mesh-lod'` to opt in. This is a display option, not mandatory preprocessing.

Each camera chooses detail from projected simplification error and target resolution, with
hysteresis to reduce changes around a threshold. Perspective, orthographic and asymmetric
projections are supported. Shadows use their own light projection and map resolution. Existing
camera and light frustum culling still use original bounds; an offscreen shadow caster is not
discarded merely because the main camera cannot see it. Color, picking, segmentation and depth
use the displayed geometry, so disable this flag when exact mesh-based sensor data is required.
The error estimate guides detail selection; it is not a strict image-difference guarantee.

Native frame statistics expose submitted `color_triangles` (including reflections),
`shadow_triangles`, `data_triangles`, `lod_instances`, `lod_meshes_ready`, and
`lod_meshes_pending`. LOD instances count non-shadow draw submissions, not unique objects.
Cached passes submit zero triangles. These differ from the full source
triangle count. Use `make mesh-lod` for lifecycle/product acceptance and a comparison capture.
For the shared G1 kinematic replay workload:

```bash
make g1-worlds G1_MODEL=/path/to/unitree_g1/scene.xml G1_WORLDS=4096 ARGS='--mesh-lod --camera detail'
make g1-worlds-benchmark G1_MODEL=/path/to/unitree_g1/scene.xml ARGS='--worker bgfx --count 4096 --mesh-lod --camera detail --output output/g1-lod'
```

The benchmark waits for pending LOD preparation before warmup and records startup separately.
This workload replays independent poses; it does not simulate 4096 physics worlds.
The viewer supports normal orbit, pan and zoom; **Camera > Presets > frame all** shows the whole
grid. Add `--rpc-socket /tmp/mojive-g1.sock` to the interactive command to let an agent inspect,
move the viewport camera and capture that same window through the public RPC API.

## Visibility and shadow work

Native bgfx culls each color, reflection and data view using transformed original mesh bounds.
Directional shadows also reject casters whose light-ray extrusion cannot reach the visible
receiver volume. That volume includes reflected cameras and is bounded by scene geometry;
color-only overlay surfaces conservatively disable the geometry bound. Screen-edge filtering
retains a margin for the shadow sampling footprint. Point and spot lights retain light-frustum
culling. This runs independently of the optional mesh LOD feature.

The shadow cache records its caster coverage. Narrowing the view can reuse a wider map, while
revealing previously excluded casters redraws it. Offscreen objects that cast visible shadows
remain included. `make shadow-visibility` checks this against complete shadow maps and saves a
comparison under `output/shadow-visibility/` (culled, complete, shadows disabled).

Culling reduces draw submission and uploads; it does not unload shared meshes or suspend
simulation and pose updates. The G1 replay still updates every world. Its 4096 instances share
one mesh resource set, so loading separate robot assets on camera movement would not address
that CPU cost. Headless benchmark timings include synchronous image readback; the interactive
`--duration 12 --capture` mode separately records animation plus viewer synchronization.

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
