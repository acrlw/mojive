"""Window and UI renderer composition shared by viewers and design tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mojive.render.selection import render_backend_name

if TYPE_CHECKING:
    from mojive.ui.window import Window, WindowConfig


def create_window(config: WindowConfig, renderer: str | None = None) -> Window:
    """Create the selected window without allocating any scene renderer resources."""
    renderer = render_backend_name(renderer)
    if renderer == "bgfx":
        from mojive.render.native.device import acquire_device
        from mojive.ui.window_native import NativeWindow

        return NativeWindow(config, device_factory=acquire_device)
    if renderer == "wgpu":
        from mojive.ui.window_wgpu import WgpuWindow

        return WgpuWindow(config)
    from mojive.ui.window import Window

    return Window(config)
