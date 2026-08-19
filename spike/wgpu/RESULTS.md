# wgpu-py backend spike — results

Date: 2026-08-19. Worktree `forge-viewer-wgpu-spike` @ eda41ce, venv: Python 3.11.15,
wgpu 0.32.0, rendercanvas 2.7.2, imgui-bundle 1.92.900, RTX 5090 D (Vulkan), X11.

Question: how painful would a wgpu-py `RenderBackend` be?

## What was built (spike/wgpu/, ~910 LOC incl. drivers)

`backend.py` (~450 LOC): a `WgpuSceneBackend` consuming the same `RenderScene`
contract as `ForgeBackend` (via `SceneSourceBuilder`). One WGSL module, MRT
geometry pass (rgba8 + r32float linear depth + r32uint object id + depth24plus),
instancing via storage buffer, aligned `queue.read_texture` readbacks, plus a
single-target surface pipeline for canvas rendering.

## A — offscreen parity (spike_offscreen.py, reference_render.py, compare.py)

parity_scene.xml, identical camera, forge/EGL reference (moderngl):

| metric | result |
|---|---|
| silhouette IoU | 1.0000 |
| per-pixel object id match | 1.0000 |
| depth p95 abs err | 0.0033 m |
| rgb | lighting model intentionally not ported (no shadows/textures/tonemap) |

Geometry, pose, depth, and pick-id paths are pixel-equivalent. The analytic
infinite plane (bucket 0 flagged `infinite_planes`) is the only scene element
forge renders differently by construction.

## B — robustness and readback cost (spike_stress.py)

| probe | result |
|---|---|
| device request x50 | 6.6 ms each, no crash |
| backend create/render/destroy x200 | 2.5 ms/cycle |
| RSS over 1000 create/destroy cycles | plateaus at ~223 MB once GC runs (lazy GPUObject collection, not a leak) |
| 200 renders + concurrent read_texture thread | no error, no crash |

Readback vs forge/EGL baseline (/tmp/egl_bench, median):

| resolution | forge EGL read_rgb | wgpu read_rgb | wgpu e2e (render+3 readbacks) |
|---|---|---|---|
| 640x480 | 1.28 ms | 0.19 ms | 0.77 ms |
| 1920x1080 | 6.67 ms | 0.70 ms (~12 GB/s) | 2.30 ms |

`queue.read_texture` is ~10x faster than the moderngl readback path and scales
linearly with bytes; the readback bottleneck measured in the EGL benchmark is a
moderngl/GL implementation property, not inherent to offscreen rendering.

## C — canvas + imgui (spike_gui.py)

- `wgpu.gui` no longer exists in wgpu 0.32; the canvas layer is the separate
  `rendercanvas` package (glfw/qt/offscreen), already a wgpu dependency.
- Offscreen canvas: 640x480 logical at pixel_ratio 2.0 -> 1280x960 physical,
  correct HiDPI handling; `canvas.draw()` returns pixels.
- Real glfw window on DISPLAY :1: ratio 2.0, 27 fps with
  `update_mode="continuous"`, 56 fps with `"fastest"` (compositor/vsync
  territory; scheduler config matters, not a GPU limit).
- imgui: `wgpu.utils.imgui.ImguiRenderer` targets imgui-bundle 1.92 (new
  ImTextureRef path) but has an untested leftover: `draw_data.cmd_lists_count`
  was removed in imgui 1.92 -> AttributeError on first frame. One-line fix
  (`cmd_lists.size()`), patched in the spike venv; overlay then renders
  correctly (see output/spike/wgpu_gui_offscreen.png). Upstreamable.
- Cosmetic: with an `rgba8unorm-srgb` surface the shader must output linear
  color (hardware does the sRGB encode); the spike shader applies its own
  gamma, so the canvas image looks washed out. Spike artifact, not a wgpu issue.

## Porting ledger for a real backend

Free/cheap: buffers, instancing, MRT, depth, readback, offscreen headless
(no EGL/X needed at all), HiDPI canvas, compute shaders available for
outline/picking rework.

Direct ports (WGSL is close to GLSL 330 for these): opaque, id, skybox,
tendon, debug, gizmo, outline, present/tonemap, transparent (existing
CPU-side sort transfers as-is).

Real work:
- shadow (CSM, 442 LOC + shadow_sample.glsl) — the biggest single pass.
- reflect (planar reflection pass + clip plane; WebGPU has no clip planes,
  use fragment discard or an oblique projection matrix).
- wireframe uses a geometry shader (wireframe.geom); WebGPU has no geometry
  stage — rewrite as line-expanded triangles or a fragment barycentric trick.
- MSAA needs per-attachment resolve targets (mechanical, but each MRT target
  pays it).
- materials/textures/skybox sampling (skipped in the spike) — texture upload
  and sampler plumbing, cube maps included.
- `Renderer` API needs metric depth + seg readback: validated above.
- imgui glue is effectively yours to maintain (one-line patch today, but the
  project pins imgui-bundle moving fast).

Estimate to reach parity with the current forge backend: ~3 person-weeks
(2-3 d offscreen core, 5-8 d main-viewer passes, 4-6 d debug/gizmo/pick/labels,
3-5 d cross-platform validation). The spike removed the two biggest unknowns
(parity of the core pass, readback economics); remaining risk is concentrated
in cross-platform validation (only Linux/NVIDIA was tested here) and in
wgpu-py/rendercanvas/imgui glue maturity, which is real but bounded and
patchable in-repo.

## Verdict

Not "很麻烦". The protocol slice that matters (RenderScene in, rgb/depth/id
out) was running at pixel parity within a day of spike code. The work is a
long tail of pass ports, not an architecture fight. wgpu-py 0.32.0 on this
machine crashed nowhere; the imgui 1.92 leftover confirms the "lightly
exercised integration layer" concern, but the failure mode is Python-level
and fixable, not a native ABI crash.
