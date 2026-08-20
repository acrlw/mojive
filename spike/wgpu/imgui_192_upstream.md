# Upstream material: wgpu-py `utils.imgui` vs imgui-bundle 1.92

Draft for an issue/PR against https://github.com/pygfx/wgpu (not submitted yet).

## Bug

`wgpu.utils.imgui.imgui_backend.ImguiWgpuBackend.render` reads
`draw_data.cmd_lists_count` (imgui_backend.py:448 in wgpu 0.32.0):

```python
if fb_width <= 0 or fb_height <= 0 or draw_data.cmd_lists_count == 0:
    return
```

`ImDrawData.cmd_lists_count` was removed in imgui-bundle 1.92 (Dear ImGui 1.92
moved draw data to `CmdLists`; the Python binding dropped the scalar). With
imgui-bundle >= 1.92 installed, the first frame raises:

```
AttributeError: 'imgui.ImDrawData' object has no attribute 'cmd_lists_count'
```

## Environment

- wgpu 0.32.0, imgui-bundle 1.92.900, Python 3.11, Linux (also platform-independent)

## Repro

```python
from imgui_bundle import imgui

# imgui-bundle 1.92.900: False — the attribute no longer exists
print(hasattr(imgui.ImDrawData, "cmd_lists_count"))
```

Any `ImguiWgpuBackend.render(...)` call with a live frame then raises the
`AttributeError` above on its first draw-data guard.

## Fix

One line, keeping compatibility with older imgui-bundle:

```python
cmd_lists = draw_data.cmd_lists
cmd_lists_count = len(cmd_lists) if not hasattr(draw_data, "cmd_lists_count") else draw_data.cmd_lists_count
```

or simply replace the condition with `draw_data.cmd_lists.size() == 0` when the
minimum supported imgui-bundle is 1.92.

## Downstream workaround (for reference)

forge-viewer carries a vendored subclass `ui/window_wgpu.py::_WgpuImguiBackend`
that overrides `render` only when a behavior probe detects the incompatibility
(`"cmd_lists_count" in render.__code__.co_names and not hasattr(imgui.ImDrawData,
"cmd_lists_count")`). It returns to `super().render` automatically once upstream
is fixed, so the upstream fix needs no coordination on our side.
