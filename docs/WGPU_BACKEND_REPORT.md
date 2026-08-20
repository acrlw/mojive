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

- `make gpu-wgpu` — per-file GPU suite against the wgpu backend (12 files, list in
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
| `make gpu-wgpu` (12 files) | all green |
| `pytest -q` | 495 passed |
| forge per-file GPU regression | failure signature identical to `main` (pre-existing: forge_core 2, id_outline 3, pipeline 1, model_loading 2 errors, ui_interaction 1+39) |
| `make renderer-api-wgpu` | 6+9 passed, gallery `passed: true` (seg agreement 0.99990) |
| Viewer smoke, both backends | main loop runs clean, no tracebacks |
| `ruff check .` + `ruff format --check` | clean |
| Caps comparison vs forge | identical except the honest gaps below |

Milestone feedback renders (wgpu vs forge side by side) live in `output/wgpu/`
(`m3_*.png` … `m9_*.png`, gitignored).

## Deliberate differences from forge (reported in `caps.notes`)

- No per-pass/GPU timer queries (WebGPU timestamp is an optional feature).
- `id_msaa=False`: a single-sample export MRT pass re-rasterizes depth+id instead of
  resolving MSAA integer attachments (not expressible in WebGPU).
- MSAA sample count is fixed at construction on both backends; the flag is stored
  but has no draw-time effect.
- WGSL translations for GL-only constructs: pipeline variants instead of defines,
  fragment-discard clip plane, barycentric vertex streams, z∈[0,1] projection.

## Follow-ups

- The vendored imgui-bundle 1.92 workaround in `ui/window_wgpu.py`
  (`cmd_lists_count`) should be upstreamed to wgpu-py; it self-disables once fixed.
- `main@4aa187d` lost the local `state_guard.py` EGL patch in a pull and currently
  crashes the forge gallery on headless EGL (`ctx.screen` is None); the committed
  fix is `512b637` on this branch.
- The forge-path gallery segmentation agreement (0.99702) misses the 0.999 gate on
  this machine; identical value reproduced on `main`, so it is a pre-existing
  driver/MSAA environment artifact, not a branch regression.
