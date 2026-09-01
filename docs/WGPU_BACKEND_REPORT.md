# wgpu Backend Report

Status reviewed: 2026-09-02

## Status

The wgpu backend implements the public Renderer API and the interactive viewer. It consumes
the same scene contracts as OpenGL and runs the same UI, session, adapter, debug-draw, overlay,
and gizmo layers.

Validated platforms:

- macOS with Metal
- Linux with Vulkan

OpenGL stays the default backend.

## Usage

```bash
uv sync --extra wgpu
MOJIVE_BACKEND=wgpu uv run mojive view test_scene
MOJIVE_BACKEND=wgpu uv run mojive editor
```

`MOJIVE_BACKEND` selects the renderer. A command's `-b/--backend` option still selects the
scene/physics adapter and should normally remain `mujoco` for MJCF/URDF.

Run the acceptance suites with:

```bash
make gpu-wgpu
make renderer-api-wgpu
```

## Coverage

| Area | Implementation |
|---|---|
| Renderer API | RGB, metric depth, segmentation, free and fixed cameras |
| Scene rendering | Opaque, transparent, additive, textures, materials, fog, haze |
| Lighting | Directional, point, spot, headlight, image light, skybox |
| Shadows | Cascaded directional atlas and local-light distance maps |
| Reflections | Planar reflection passes with clip-plane discard |
| Diagnostics | Debug views, labels, frames, BVH, constraints, contacts, tendons |
| Tooling | Selection outline, debug draw, text overlay, native gizmo |
| Viewer | GLFW NO_API surface, imgui composition, capture and readback |
| Runtime diagnostics | Non-blocking GPU pass timestamps; live 1×/4× MSAA toggle |

## Backend limits

- GPU pass timestamps are requested as an optional device feature and read back asynchronously.
  Unsupported adapters continue with CPU frame timing only.
- Object ID and depth export use a single-sample pass because WebGPU does not resolve
  multisampled integer or depth attachments.
- The MSAA render flag switches between 1× and the configured 4× sample count by rebuilding
  multisampled targets and sample-count-dependent pipelines.
- Window pacing implements the viewer's vsync setting while wgpu-py lacks present-mode
  selection.
- Surface shutdown uses wgpu-py's private release hook until a public API is available.
- The version-guarded imgui 1.92 adapter remains necessary because wgpu-py 0.32 still reads
  the removed `cmd_lists_count` property.

These limits are also exposed through `backend.caps.notes`.

## Architecture

`render/webgpu/` owns WebGPU resources and passes. Shared contracts remain in
`render/backend.py`, `render/scene.py`, `render/builder.py`, `render/debugdraw.py`, and
`render/overlay.py`. `ui/window_wgpu.py` implements the existing window contract and keeps
one ImGui context per window. Offscreen rendering creates a device without a native window.

WGSL uses pipeline variants for shader configuration, explicit barycentric attributes for
wireframe rendering, fragment clipping for reflection planes, and WebGPU's zero-to-one depth
range. Texture mip chains are generated on the CPU.
