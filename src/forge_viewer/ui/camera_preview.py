"""Secondary scene rendering for the selected camera preview."""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ..adapters.base import NodeKind, SceneFrame, SceneSource
from ..types import CameraView, ViewportImage


class CameraPreview:
    def __init__(self) -> None:
        self._backend: Any | None = None
        self._image: ViewportImage | None = None
        self._source_generation = -1
        self._position: tuple[float, float] | None = None

    def update(
        self,
        main_backend: Any,
        source: SceneSource | None,
        source_generation: int,
        frame: SceneFrame,
        camera: CameraView | None,
        size: tuple[int, int],
    ) -> None:
        if source is None or camera is None:
            self._image = None
            return
        backend = self._ensure_backend(main_backend, size)
        backend.resize(*size)
        if source_generation != self._source_generation:
            backend.set_scene(source)
            self._source_generation = source_generation
        for flag in main_backend.render_options():
            backend.set_flag(flag, main_backend.get_flag(flag))
        backend.set_camera(camera.with_aspect(size[0] / max(size[1], 1)))
        backend.highlight(0)
        backend.update(frame)
        self._image = backend.render()

    def draw(
        self,
        window: Any,
        viewport: tuple[float, float, float, float],
        camera_name: str,
    ) -> None:
        image = self._image
        if image is None:
            return
        x, y, width, height = viewport
        scale = window.style_scale
        panel_width = min(340.0 * scale, max(180.0, width * 0.48))
        image_height = panel_width * 9.0 / 16.0
        header_height = 25.0 * scale
        panel_height = header_height + image_height
        margin = 12.0 * scale
        if self._position is None:
            self._position = (x + width - panel_width - margin, y + height - panel_height - margin)
        px = min(max(self._position[0], x + margin), x + width - panel_width - margin)
        py = min(max(self._position[1], y + margin), y + height - panel_height - margin)
        self._position = (px, py)

        imgui.set_cursor_screen_pos(imgui.ImVec2(px, py))
        child_flags = imgui.ChildFlags_.borders.value
        window_flags = (
            imgui.WindowFlags_.no_scrollbar.value | imgui.WindowFlags_.no_scroll_with_mouse.value
        )
        if not imgui.begin_child(
            "##camera_preview", imgui.ImVec2(panel_width, panel_height), child_flags, window_flags
        ):
            imgui.end_child()
            return
        title = f"Camera · {camera_name}"
        imgui.button(title, imgui.ImVec2(-1.0, header_height))
        if imgui.is_item_active() and imgui.is_mouse_dragging(0):
            delta = imgui.get_io().mouse_delta
            self._position = (px + float(delta.x), py + float(delta.y))
        available = imgui.get_content_region_avail()
        uv0 = imgui.ImVec2(0.0, 1.0) if image.flip_y else imgui.ImVec2(0.0, 0.0)
        uv1 = imgui.ImVec2(1.0, 0.0) if image.flip_y else imgui.ImVec2(1.0, 1.0)
        imgui.image(window.viewport_texture_ref(image), available, uv0, uv1)
        imgui.end_child()

    @staticmethod
    def selected_camera(session) -> tuple[str, CameraView | None]:
        node = session.selected_node
        if node is None or node.kind is not NodeKind.CAMERA or node.camera_index < 0:
            return "", None
        return node.name, session.camera_view(node.camera_index)

    def release(self) -> None:
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        self._image = None

    def _ensure_backend(self, main_backend: Any, size: tuple[int, int]) -> Any:
        if self._backend is not None:
            return self._backend
        self._backend = main_backend.create_peer(*size)
        return self._backend
