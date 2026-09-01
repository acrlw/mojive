# Configuration reference

The Settings panel is the preferred interface for persistent user preferences. Open it from
**Edit > Settings...**, **Window > Settings**, or `F9`. It is a dockable, non-modal panel; opening
it does not block viewport interaction.

## Environment variables

| Variable | Values | Purpose |
|---|---|---|
| `MOJIVE_BACKEND` | `opengl`, `wgpu` | Select the render backend for interactive and offscreen rendering. |
| `MOJIVE_GL` | `auto`, `native`, `glfw`, `egl` | Select OpenGL context creation. EGL is supported on Linux; offscreen OpenGL uses EGL there by default. |
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
`MOJIVE_GL=egl` to force and verify EGL. Offscreen `Renderer` uses EGL by default on Linux and a
hidden GLFW context when `MOJIVE_GL=native` is selected.

The wgpu backend requires the `wgpu` optional dependency and a compatible Metal, Vulkan, or DX12
adapter:

```bash
uv sync --extra wgpu
MOJIVE_BACKEND=wgpu uv run mojive doctor test_scene
```

Use `mojive backends` for adapter availability and `mojive probe` for OpenGL capability details.

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
