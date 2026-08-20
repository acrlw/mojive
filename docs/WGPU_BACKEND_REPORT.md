# wgpu Backend Completion Report

- Date: 2026-08-20
- Branch: `wgpu-py` (commits `512b637`..`f80417a` + texture filtering fix)
- Scope: complete wgpu-py render backend for forge-viewer
- Status: done — capability parity with the OpenGL (forge) backend, all checks green

## Conclusion

`FORGE_VIEWER_BACKEND=wgpu` now covers the full forge-viewer capability set: the
offscreen Renderer API, shadows, planar reflections, skybox/IBL, all debug views,
selection outline, tendons, debug draw with text overlays, gizmo, and the
interactive viewer window. Planning and milestone detail live in
`docs/WGPU_BACKEND_PLAN.md`; this file is the closing summary.

## How to run the tests

- `make gpu-wgpu` — per-file GPU suite against the wgpu backend (14 files, list in
  the Makefile variable `GPU_WGPU_FILES`); `make gpu` is the forge counterpart.
- `make renderer-api-wgpu` / `make renderer-api` — Renderer API contract suites plus
  the comparison gallery against `mujoco.Renderer`.
- `FORGE_VIEWER_BACKEND=wgpu make viewer` — interactive viewer on wgpu.
- The default `pytest -q` suite (495 tests) is backend-neutral and always runs.

## Capability matrix

| Area | Status | Notes |
|---|---|---|
| Offscreen Renderer API (rgb/depth/segmentation) | done | gallery MAE 1.44 vs `mujoco.Renderer` |
| Cameras (free/fixed/named, intrinsics, ortho) | done | |
| Shadows | done | CSM (4096 atlas, 3 cascades) + spot/point distance maps, PCF |
| Planar reflections | done | clip-plane discard replaces `gl_ClipDistance`, max 4 planes |
| Skybox / IBL image light / horizon haze | done | CPU mip chains (WebGPU has no mipmap generation) |
| Debug views (8) | done | WIREFRAME via lazy barycentric expansion (no geometry stage) |
| Selection outline | done | single-sample r8 mask + dilation composite |
| Tendons | done | 3 capsule instances per segment, scene pipeline |
| Debug draw + text overlays | done | 6 primitive families, occlusion DEPTH/ALWAYS/GHOST |
| Labels / frames / BVH | done | shared `render/overlay.py` publisher with forge |
| Gizmo | done | screen-constant size, near-plane depth pin |
| Render flags (38) / label modes (14) / frame modes (7) | done | identical sets on both backends |
| Interactive viewer | done | GLFW NO_API window + rendercanvas present + vendored imgui fix |

## Verification results

| Check | Result |
|---|---|
| `make gpu-wgpu` (14 files) | all green (gizmo suite now 7 cases: axis direction/sign vs CPU projection, foreshortened-stub placement, body-frame following) |
| `pytest -q` | 501 passed (view cube gained a 6-case nearest-ball-on-top pin) |
| forge per-file GPU regression | failure signature identical to `main` (pre-existing: forge_core 2, id_outline 3, pipeline 1, model_loading 2 errors, ui_interaction 1+39) |
| `make renderer-api-wgpu` | 6+9 passed, gallery `passed: true` (seg agreement 0.99990) |
| Viewer smoke, both backends | main loop runs clean, no tracebacks |
| `ruff check .` + `ruff format --check` | clean |
| Caps comparison vs forge | identical except the honest gaps below |
| Gizmo / view-cube alignment audit (2026-08-20) | forge-faithful: full-window captures at 3 cameras and offscreen ground-truth renders at 5 cameras × 2 modes match forge within AA noise; the view cube is pixel-identical across backends (shared imgui code), its near-ball-on-top order is the correct painter sort, and a faint foreshortened axis handle is the deliberate shared `axis_handle_alpha` fade, not a backend bug |

Milestone feedback renders (wgpu vs forge side by side) live in `output/wgpu/`
(`m3_*.png` … `m9_*.png`, audit captures `audit_viewcube_*` / `audit_gizmo_*` /
`audit_truth_*` / `audit_zoom_*`, all gitignored).

## Deliberate differences from forge (reported in `caps.notes`)

- No per-pass/GPU timer queries (WebGPU timestamp is an optional feature).
- `id_msaa=False`: a single-sample export MRT pass re-rasterizes depth+id instead of
  resolving MSAA integer attachments (not expressible in WebGPU).
- MSAA sample count is fixed at construction on both backends; the flag is stored
  but has no draw-time effect.
- WGSL translations for GL-only constructs: pipeline variants instead of defines,
  fragment-discard clip plane, barycentric vertex streams, z∈[0,1] projection.

## Follow-ups

### Correctness / merging (do first)

- `main@4aa187d` lost the local `state_guard.py` EGL patch in a pull and currently
  crashes the forge gallery on headless EGL (`ctx.screen` is None); the committed
  fix is `512b637` on this branch — land it on `main`.
- Cross-platform validation: run `make gpu-wgpu` + viewer smoke on macOS (Metal)
  and Windows (DX12); this is the remaining gap behind the "mujoco.Renderer
  replacement" claim.
- Merge decision for the `wgpu-py` branch, after cross-platform validation; forge
  stays the default backend.

### Performance (deferred until a profile justifies them)

- Gate the export MRT pass on actual depth/segmentation/pick demand (mode-switch
  semantics need care; Renderer-API users usually read depth anyway).
- Async double-buffered readback for the MP4 recording path.
- Shadow layer-draw consolidation (bounded: WebGPU has no layered rendering).

Rejected with reasons: eliminating the viewer frame copy (breaks `read_frame`/
recording — swapchain textures are not readable), instance-buffer dirty tracking
(poses change every frame under simulation, so nothing is ever clean).

### Upstream (wgpu-py)

- Upstream the vendored imgui-bundle 1.92 fix in `ui/window_wgpu.py`
  (`cmd_lists_count`); it self-disables once fixed.
- Track `timestamp-query` support (API signature present, marked unused) — unblock
  per-pass/GPU timing once wired.
- Track present-mode/vsync control and a public surface-release API (we call the
  private `_release()` to avoid an X11 segfault at GC, pygfx#642). wgpu-py 0.32
  hardcodes the present-mode preference `immediate > mailbox > fifo`, so
  `WgpuWindow` emulates vsync with an app-side frame pacer (`_pace_frame`);
  upstream present-mode control would let the driver do the pacing.

### Optional polish

- `id_msaa`: shader-side resolve of a 4x uint mask would remove the 1-px picking
  difference at MSAA edges; modest payoff for the complexity.
- GPU-side mip generation, only if float/HDR texture inputs ever appear (CPU box
  filter covers u8 today).

### Environment note

- The forge-path gallery segmentation agreement (0.99702) misses the 0.999 gate on
  this machine; identical value reproduced on `main`, so it is a pre-existing
  driver/MSAA environment artifact, not a branch regression.
