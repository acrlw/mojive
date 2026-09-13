# wgpu Backend Completion Record

Status: completed historical record. Current user-facing behavior is documented in
[`docs/WGPU_BACKEND_REPORT.md`](../docs/WGPU_BACKEND_REPORT.md).

This document records the implementation milestones for the wgpu backend. The backend is
integrated and supports the public Renderer API, the interactive viewer, render passes,
diagnostic overlays, and tooling. Current validation is summarized in
`docs/WGPU_BACKEND_REPORT.md`.

## Principles

1. **Parallel pass structure.** `render/webgpu/` mirrors `render/opengl/` structure: a
   `passes/` package (one module per pass, same names), `shaders/*.wgsl` mirroring the
   GLSL files, the same store class conventions (`sync/get/release`), the same
   PASS_ORDER. Backend-specific limits are exposed through `BackendCaps.notes`.
2. **Shared renderer contracts.** `render/scene.py`, `builder.py`, `debugdraw.py`,
   `mesh.py`, `text.py`, `overlay.py`, top-level `gizmo.py`, and `picking.py` are shared.
   Both backends consume the same glyph atlas and debug-overlay publishers.
3. **WGSL idioms for GL idioms.** WGSL has no preprocessor: `DEBUG_VIEW` / `WIREFRAME` /
   `USE_SHADOW` variants use pipeline-creation `constants` (WGSL `override`). No
   geometry shader: wireframe barycentrics come from a lazily-built vertex attribute.
   No `gl_ClipDistance`: reflection clipping is a fragment discard on a plane equation
   in the frame uniforms. No MSAA resolve for uint/depth: the existing single-sample
   export MRT pass remains the readback path. `gl_VertexID` → `@builtin(vertex_index)`,
   `flat`/`noperspective` → `@interpolate(flat/linear)` — both used by debug draw.
4. **Capability reporting.** `BackendCaps` and `_SUPPORTED_FLAGS` reflect implemented,
   tested behavior.
5. **Milestone verification.** Each milestone includes focused WGPU tests, the default
   CPU suite, OpenGL GPU regressions, formatting, and lint checks.

## Milestones

### M1 — Backend-parameterized test infrastructure — done (42847a1)
- `tests/gpu/conftest.py`: `make_backend(...)` factory honoring `MOJIVE_BACKEND`;
  `backend_name` fixture; GL-internals test files (test_opengl_core, test_id_outline,
  test_debugdraw_gpu, and the GL-specific parts of test_pipeline) get an explicit
  opengl-only skip guard.
- `tools/_harness.py`: `OffscreenHarness` backend switch (mirrors
  `renderer._select_backend`).
- Makefile: `gpu-wgpu` target — per-file loop over the backend-neutral subset with
  `MOJIVE_BACKEND=wgpu`.
- Acceptance: `make gpu-wgpu` runs, wgpu-unsupported cases skip on caps; `make gpu`
  output identical to before.

### M2 — Cubemaps: skybox, IBL image light, horizon haze — done (e490784)
- `textures.py`: cubemap store (6-face upload, CPU-generated mip chain — WebGPU has no
  auto-mipmap; IBL roughness LOD depends on it), `black_cube` fallback.
- `passes/skybox.py` + `shaders/skybox.wgsl`: fullscreen triangle, inverse view-proj
  ray, Z-up cubemap swizzle, far-plane depth; haze ring geometry (port `haze.vert`).
- Scene shader: IBL sampling (`textureSampleLevel`, diffuse at max mip, specular at
  roughness·maxMip, 5000-reference normalization).
- Flags: SKYBOX; caps notes updated.
- Acceptance: `test_horizon_haze.py` and the skybox/image-light cases of
  `test_shading.py` pass under wgpu.

### M3 — Shadows (largest single shader port) — done (f95f5ee)
- `cascades.py` port with WebGPU z∈[0,1] ortho (4096² atlas, 3 tiles, texel snap).
- `passes/shadow.py` + `shadow.wgsl` (depth-only), `spot_dist.wgsl` (R16Float distance,
  MIN blend, 2D-array layers — per-layer views are native in WebGPU, simpler than GL's
  `glFramebufferTextureLayer`).
- `shadow_sample.wgsl`: cascade select/fallback, slope-scaled bias, 3×3 PCF, atlas
  clamp, spot perspective PCF, point cube-face mapping, area-light 7×7 kernel.
- `lighting.py`: shadow-aware light scheduling (1 directional + 8 local).
- Flags: SHADOW; cast_shadow semantics; transparent does not cast.
- Acceptance: `test_shadows.py` (14) under wgpu.

### M4 — Planar reflections — done (bb3b25d)
- `passes/reflect.py`: plane detection/dedup (max 4), mirrored view matrix, front-face
  flip pipeline variant, clip-plane discard in the scene fragment, negative-reflectance
  channel encoding (layer/top-face), RGBA16F targets + shared depth, transparent
  re-draw inside reflection.
- Flags: REFLECTION.
- Acceptance: `test_reflection.py` (9) under wgpu.

### M5 — Debug views, selection outline, present modes — done (24ab3ad)
- WIREFRAME: lazy barycentric vertex buffer in `MeshStore` + pipeline constant variant
  (no geometry shader).
- OVERDRAW: additive, no-depth pipeline variant. ALBEDO/NORMAL/DEPTH already exist.
- SEGMENT/IDCOLOR: present pass reading the export id texture (port `present.frag`).
- `passes/outline.py` + `outline.wgsl`: selection mask (single-sample — WebGPU cannot
  resolve uint MSAA; the dilation shader already does its own AA) + port of the 62-line
  morphology shader with edge-continuity fix.
- Acceptance: debug-view cases of `test_shading.py`; new backend-neutral outline
  behavior tests (mask continuity under occlusion, one outline per link, viewport-edge
  behavior) run under both backends.

### M6 — Tendons — done (111c440)
- `passes/tendon.py`: `_publish_tendons` packing, capsule shaft/cap instances (3 per
  segment) from the shared built-in meshes, own `InstanceStore`, material/transparent
  buckets with back-to-front order. Reuses the scene pipeline wholesale.
- Flags: TENDON.
- Acceptance: tendon cases of `test_pipeline.py` under wgpu.

### M7 — Debug draw + text + overlays (largest CPU+GPU chunk) — done (ce270bd)
- Extract shared `DebugDraw` publisher out of `OpenGLBackend` (labels, frames, bvh,
  contacts, joints, COM, inertia, actuators, rangefinder, constraint, flex).
- `passes/debug.py` + the six WGSL families (`debug_line/point/solid/sector/stroke/
  drag_link`, all `@builtin(vertex_index)` expansion), occlusion modes
  DEPTH/ALWAYS/GHOST (two-pass).
- Split `render/opengl/text.py`: atlas building → shared; new wgpu glyph draw
  (`debug_text.wgsl`); `configure_text` on `WgpuBackend`.
- Flags: JOINT/COM/INERTIA/ACTUATOR/CONTACT*/BVH/CAMERA/LIGHT/RANGEFINDER/CONSTRAINT/
  FLEX*; `set_label_mode`/`set_frame_mode`/`set_bvh_depth` become real.
- Acceptance: backend-neutral debug-draw behavior tests (occlusion modes, pixel-constant
  line width, glyph quads, 10k-line frame) + label/frame/bvh cases of `test_pipeline.py`
  under wgpu.

### M8 — Gizmo — done (1364aeb)
- `passes/gizmo.py` + `gizmo.wgsl`: 7 built-in handle meshes, screen-constant scale
  (`px_scale·clip.w`), depth pinned to the near plane (WebGPU z≈0), `u_mask_radius`
  center hole; `set_gizmo` stores and `caps.gizmo=True`.
- Acceptance: backend-neutral gizmo render tests (handle visible, screen-size
  invariance, mask hole); existing CPU hit-tests unchanged.

### M9 — Interactive viewer surface path (most integration risk) — done (808eb50)
- `ui/window_wgpu.py`: GLFW creates a `GLFW_CLIENT_API=GLFW_NO_API` window and
  rendercanvas supplies the native surface description; imgui renders through
  `wgpu.utils.imgui.ImguiWgpuBackend`. The known imgui-bundle 1.92 incompatibility
  (`cmd_lists_count`) is handled by a version-guarded subclass in the project.
- `ViewportImage`: carries a backend payload; `_draw_viewport` binds the wgpu texture
  through the imgui renderer instead of `ImTextureRef(gl_id)`.
- `composition._compose`: honor `MOJIVE_BACKEND`; `doctor`/`backends` reporting.
- HiDPI: the GL and wgpu windows share point-to-pixel conversion and explicit UI scaling.
- Acceptance: `MOJIVE_BACKEND=wgpu make viewer` opens the full UI; window-stack
  tests that are backend-neutral (open, read_frame, edit→pixels) pass under wgpu;
  `make doctor` reports the wgpu path accurately.

### M10 — Finalization — done
- MSAA flag semantics aligned with opengl (samples fixed at construction — verify opengl
  behavior first, then match).
- `id_msaa`/caps/notes sweep; README, `docs/RENDERER.md`, and `plan/ROADMAP.md` updates.
- Full verification matrix: default suite, per-file GPU loop on both backends,
  `renderer-api` + `renderer-api-wgpu`, gallery vs `mujoco.Renderer` on both backends.

Outcome: the initial caps sweep recorded `id_msaa=False`, construction-time MSAA, and CPU-only
frame timing as honest backend differences. M11 supersedes the latter two limits while the
single-sampled export MRT remains required by WebGPU.

### M11 — Runtime timing and MSAA reconfiguration — done

- Request `timestamp-query` as an optional shared-device feature and collect shadow,
  reflection, outline, scene, export, and present durations through a three-buffer asynchronous
  readback queue. Unsupported adapters retain CPU frame timing.
- Make the existing MSAA render flag switch between 1× and the configured 4× count by rebuilding
  render targets and sample-count-dependent pipelines at runtime.
- Keep native present-mode selection, public surface release, and removal of the imgui 1.92
  adapter on the upstream-dependent list. wgpu-py 0.32 exposes none of the first two APIs and its
  imgui renderer still reads `cmd_lists_count`.

## Explicitly out of scope (recorded in caps notes)

- GL state guarding, native GL entry points, and GLSL hot reload.
- `imgui-bundle` upstream fix submission (tracked separately).
