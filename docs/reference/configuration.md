# Configuration reference

The Settings panel is the preferred interface for persistent user preferences. Open it from
**Edit > Settings...**, **Window > Settings**, or `F9`. It is a dockable, non-modal panel; opening
it does not block viewport interaction.

## Environment variables

| Variable | Values | Purpose |
|---|---|---|
| `MOJIVE_BACKEND` | `opengl`, `wgpu` | Select the render backend for interactive and offscreen rendering. |
| `MOJIVE_GL` | `auto`, `native`, `glfw`, `egl` | Select OpenGL context creation. Offscreen `auto` tries EGL on Linux, then hidden GLFW only when a desktop display and the main thread are available. |
| `MOJIVE_UI_SCALE` | positive number | Override the logical UI scale when desktop scale detection is wrong. |
| `MOJIVE_LANGUAGE` | `en`, `zh_CN` | Override the UI language for the process. |
| `MOJIVE_CJK_FONT` | font path | Use a specific CJK fallback font. |
| `MOJIVE_SETTINGS` | JSON path | Override the persistent application settings file. |
| `MOJIVE_CONFIG_DIR` | directory path | Override the directory containing `imgui.ini`. |
| `MOJIVE_IMGUI_INI` | file path | Override the ImGui layout file directly. |
| `MOJIVE_OPEN_SETTINGS` | `1` | Open the Settings panel at startup; primarily used by visual acceptance. |

`MOJIVE_BACKEND` controls rendering, not the scene/physics adapter. Model-oriented CLI commands
use `-b/--backend` for the adapter. See the [CLI reference](cli.md#render-backend-and-scene-adapter).

`FORGE_VIEWER_BACKEND` and the value `forge` remain accepted only as migration compatibility for
older local launch scripts. New scripts should use `MOJIVE_BACKEND=opengl`.

## Persistent files

Application preferences default to these locations:

| Platform | Settings file | Layout file |
|---|---|---|
| macOS | `~/Library/Application Support/mojive/settings.json` | `~/Library/Application Support/mojive/imgui.ini` |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/mojive/settings.json` | `${XDG_CONFIG_HOME:-~/.config}/mojive/imgui.ini` |
| Windows | `%APPDATA%/mojive/settings.json` | `%APPDATA%/mojive/imgui.ini` |

`MOJIVE_SETTINGS` replaces the settings-file path. `MOJIVE_IMGUI_INI` replaces the layout-file
path. `MOJIVE_CONFIG_DIR` affects the layout file only.

Generated captures, recordings, reports, visual-acceptance images, and the built documentation
site belong under the repository's ignored `output/` directory.

## Render backend requirements

OpenGL is the default. Interactive windows require a desktop OpenGL 3.3 core profile. On Linux,
interactive GLFW normally chooses the native context API (EGL on Wayland, GLX on X11); set
`MOJIVE_GL=egl` to force and verify EGL for the interactive window.

Offscreen `Renderer` has a separate context policy:

| Selection | Behavior |
|---|---|
| `auto` or unset, Linux | Try EGL first. If creation fails, try hidden GLFW only with `DISPLAY` or `WAYLAND_DISPLAY` set and on the main thread. Warn when fallback succeeds. |
| `auto` or unset, macOS/Windows | Use a hidden native GLFW window. |
| `egl` | Require EGL; never silently switch to GLFW. |
| `glfw` or `native` | Require a hidden native GLFW window; do not try EGL first. |

A hidden window is offscreen, not display-free. On a Linux desktop where EGL fails, try:

```bash
MOJIVE_GL=glfw uv run python examples/mujoco_render.py assets/test_scene.xml
```

On a server without X11/Wayland, keep EGL and check the GPU driver, EGL vendor libraries, and
GPU/device access inside the container. Setting a dummy `DISPLAY` does not create a display
server. Worker threads do not automatically fall back to GLFW. Require EGL for reproducible
headless jobs and benchmarks:

```bash
MOJIVE_GL=egl uv run python examples/mujoco_video.py assets/test_scene.xml
```

Context errors retain the original initialization failure and every attempted backend. A
successful GLFW fallback may select a different GPU from EGL, so the runtime emits a warning.
The presence of a display variable permits an attempt; it does not guarantee a working display.

The wgpu backend requires the `wgpu` optional dependency and a compatible Metal, Vulkan, or DX12
adapter:

```bash
uv sync --extra wgpu
MOJIVE_BACKEND=wgpu uv run mojive doctor test_scene
```

Use `mojive backends` for adapter availability and `mojive probe` for OpenGL capability details.

## Video encoding

`VideoRecorder` streams frames through FFmpeg. `imageio-ffmpeg` is a normal Mojive dependency
and supplies executable discovery. If the executable is missing, reinstall that dependency or
set `IMAGEIO_FFMPEG_EXE` to a working FFmpeg binary with the requested encoder.

MP4 defaults to H.264 (`libx264`) and `yuv420p` for player compatibility. Use
`VideoRecorder(..., pixel_format="yuv444p")` for full chroma resolution, or `codec=` to select
another encoder supported by your FFmpeg. Odd input dimensions in `yuv420p` are edge-padded by
one pixel on the right/bottom as needed, never scaled. The recorder warns and exposes
`encoded_size`; `size` remains the required input dimensions. Choose even dimensions for exact
output dimensions with the compatible default. Not every encoder supports every pixel format.

Always close the recorder (prefer a `with` block). Empty recordings, write failures, encoder
errors, and finalization timeouts raise errors instead of reporting success. On failure, a
partial file may remain for diagnosis. See the [rollout video recipe](../tutorials/mujoco-rendering.md#record-a-rollout).

## UI scale and fonts

Without an override, Mojive combines GLFW content scale and framebuffer scale so fonts, ImGui
controls, overlays, gizmos, labels, and hit regions retain a consistent physical size. Use a fixed
scale only to diagnose a desktop-reporting problem or to run visual acceptance:

```bash
MOJIVE_UI_SCALE=2 uv run mojive editor
```

The UI atlas uses JetBrains Mono and a CJK fallback. Mojive searches for an installed Noto Sans SC
font and may place a checksum-verified copy in the application cache. Set `MOJIVE_CJK_FONT` when a
specific local font is required.
