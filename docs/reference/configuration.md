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

The Settings panel writes user choices to the JSON settings file and restores them at the next
launch. The file is intentionally backend-neutral, so switching between OpenGL and WebGPU keeps
the same preference.

| Preference | Values | Settings panel |
|---|---|---|
| `shadow_quality` | `performance`, `balanced`, `high` | Rendering > Shadows > Shadow quality |

`balanced` is the default. `performance` reduces receiver filtering work, while `high` increases
near-cascade density and filtering quality for close inspection. Changing the preset invalidates
the shadow-map cache once; subsequent static frames reuse the rebuilt maps.

## Programmatic viewer configuration

Embedding code can define interaction ownership before the viewer starts. The options are
independent: disabling scene picking does not disable focus or camera motion, disabling fly keys
does not disable orbit, and hiding the gizmo does not clear the logical selection.

```python
from pathlib import Path

from mojive import (
    CameraInputConfig,
    InteractionConfig,
    InputClaim,
    LayoutConfig,
    PanelConfig,
    SelectionInputConfig,
    SelectionStyle,
    ViewerConfig,
    ViewportOverlayConfig,
    build,
)

config = ViewerConfig(
    interactions=InteractionConfig(
        camera=CameraInputConfig(fly=False),
        selection=SelectionInputConfig(pick=False, clear_with_escape=False),
        panel_shortcuts=False,
    ),
    selection=SelectionStyle(highlight=True, outline=True, gizmo=False, frame=True),
    panels={
        "hierarchy": PanelConfig(enabled=False),
        "inspector": PanelConfig(open=False),
    },
    # Keep this product's docking state separate from the Mojive editor.
    layout=LayoutConfig(path=".policy-eval-imgui.ini"),
    viewport_overlays=ViewportOverlayConfig(
        playback_scale=0.85,
        tool_scale=0.75,
        movable=True,
    ),
    shadow_quality="high",
)

viewer = build(Path("robot.xml"), config=config)

def policy_input(input):
    if input.viewport_focused and input.key_down("w"):
        move_policy_target_forward()
    return InputClaim(keys=frozenset({"w", "a", "s", "d"}))

viewer.set_input_handler(policy_input)
viewer.run()
```

`InputClaim` is per-frame ownership, so an application may claim WASD only while a policy-control
mode is active. Broad ImGui keyboard capture from a focused panel does not block the callback;
only active text editing, modal UI, and native dialogs do. Clicking the viewport returns focus,
while `pick_on_focus=False` can make that first click focus-only. Actions may also be assigned to
`None` in the Settings shortcut editor. At
runtime, use `viewer.configure_interactions(...)`, `viewer.configure_selection(...)`,
`viewer.configure_viewport_overlays(...)`, and the
stable panel IDs through `viewer.panels.open("hierarchy")`, `close`, `enable`, or `disable`.
Explicit `ViewerConfig` values apply to that viewer instance. Changes made in Settings are stored
as desktop preferences for later viewers created without an explicit config. Runtime
`configure_*` calls are also instance-local unless passed `persist=True`.

Use `LayoutConfig(persistence=False)` for a deterministic default layout on every launch, or
`LayoutConfig(reset=True)` to discard stale/off-screen docking coordinates once and then keep the
new layout. A custom `path` isolates each embedding application from Mojive's editor layout.

Library code configures shadow quality explicitly per renderer and does not read or update the
application settings file:

```python
from mojive import Renderer, ShadowQuality

with Renderer(model, shadow_quality=ShadowQuality.HIGH) as renderer:
    renderer.update_scene(data)
    image = renderer.render()
    renderer.set_shadow_quality(ShadowQuality.PERFORMANCE)
```

## Interactive capture and recording

Interactive captures distinguish the raw scene from composed UI. `SCENE` is the default and
does not incur a full-window framebuffer readback. `VIEWPORT` includes Mojive's viewport tools
and overlays but crops panels and the menu bar; `WINDOW` captures the complete ImGui window.

```python
from mojive import CaptureSurface

viewer.capture("output/raw.png")
viewer.capture("output/tutorial.png", surface=CaptureSurface.VIEWPORT)

viewer.start_recording("output/walkthrough.mp4", surface=CaptureSurface.WINDOW, fps=30)
for _ in range(120):
    viewer.sync()
viewer.pause_recording()
# Update the scene without appending video frames, then continue the same file.
viewer.resume_recording()
for _ in range(120):
    viewer.sync()
viewer.stop_recording()
```

`viewer.record(...)` remains the deterministic fixed-frame, UI-free rollout API. The
`start_recording` lifecycle is for user-driven or automated editor demonstrations. The current
phase, surface, output path, frame count, and duration are available through `viewer.recording`.
Changing the window or viewport dimensions while recording a UI surface stops that recording
with an explicit error instead of silently stretching or cropping frames.

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
