"""Secondary scene rendering for the selected camera preview."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from imgui_bundle import imgui

from ..adapters.base import NodeType, SceneFrame, SceneSource
from ..types import CameraView, ViewportImage


class CameraPreview:
    def __init__(self) -> None:
        self._backend: Any | None = None
        self._image: ViewportImage | None = None
        self._source_generation = -1
        self._position: tuple[float, float] | None = None
        self._pinned = False
        self._locked = False
        self._camera_name = ""
        self._camera_id = -1
        self._camera: CameraView | None = None

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
        backend.set_debug_view(main_backend.get_debug_view())
        backend.set_label_mode(main_backend.get_label_mode())
        backend.set_frame_mode(main_backend.get_frame_mode())
        backend.set_bvh_depth(main_backend.get_bvh_depth())
        backend.set_camera(camera.with_aspect(size[0] / max(size[1], 1)))
        backend.highlight(0)
        backend.update(frame)
        self._image = backend.render()

    def draw(
        self,
        window: Any,
        viewport: tuple[float, float, float, float],
        camera_name: str,
        translate: Any = None,
    ) -> None:
        image = self._image
        if image is None:
            return
        translate = translate or str
        x, y, width, height = viewport
        scale = window.style_scale
        header_height = 25.0 * scale
        margin = 12.0 * scale
        panel_width = min(
            340.0 * scale,
            max(180.0 * scale, width * 0.48),
            max(1.0, width - 2.0 * margin),
            max(1.0, height - header_height - 2.0 * margin) * 16.0 / 9.0,
        )
        image_height = panel_width * 9.0 / 16.0
        panel_height = header_height + image_height
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
        pin_label = translate("Pinned" if self._pinned else "Pin")
        lock_label = translate("Locked" if self._locked else "Lock")
        pin_width = imgui.calc_text_size(pin_label).x + 18.0 * scale
        lock_width = imgui.calc_text_size(lock_label).x + 18.0 * scale
        spacing = imgui.get_style().item_spacing.x
        title_width = max(
            1.0,
            imgui.get_content_region_avail().x - pin_width - lock_width - 2.0 * spacing,
        )
        title = f"{translate('Camera')} · {camera_name}"
        imgui.button(title, imgui.ImVec2(title_width, header_height))
        if (
            not (self._pinned or self._locked)
            and imgui.is_item_active()
            and imgui.is_mouse_dragging(0)
        ):
            delta = imgui.get_io().mouse_delta
            self._position = (px + float(delta.x), py + float(delta.y))
        imgui.same_line()
        if imgui.button(pin_label, imgui.ImVec2(pin_width, header_height)):
            self.set_pinned(not self._pinned)
        imgui.same_line()
        if imgui.button(lock_label, imgui.ImVec2(lock_width, header_height)):
            self.set_locked(not self._locked)
        available = imgui.get_content_region_avail()
        uv0 = imgui.ImVec2(0.0, 1.0) if image.flip_y else imgui.ImVec2(0.0, 0.0)
        uv1 = imgui.ImVec2(1.0, 0.0) if image.flip_y else imgui.ImVec2(1.0, 1.0)
        imgui.image(window.viewport_texture_ref(image), available, uv0, uv1)
        imgui.end_child()

    @property
    def pinned(self) -> bool:
        return self._pinned

    @property
    def locked(self) -> bool:
        return self._locked

    def set_pinned(self, pinned: bool) -> None:
        self._pinned = bool(pinned and self._camera is not None)
        if self._pinned:
            self._locked = False

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked and self._camera_id >= 0)
        if self._locked:
            self._pinned = False

    def selected_camera(self, session) -> tuple[str, CameraView | None]:
        if self._pinned:
            return self._camera_name, self._camera
        if self._locked:
            camera = session.camera_view(self._camera_id)
            if camera is None:
                self._locked = False
                return "", None
            self._camera = _copy_camera(camera)
            return self._camera_name, self._camera
        node = session.selected_node
        if node is None or node.type is not NodeType.CAMERA or node.camera_index < 0:
            return "", None
        if node.camera_index >= len(session.cameras):
            return "", None
        camera_id = session.cameras[node.camera_index].camera_id
        camera = session.camera_view(camera_id)
        if camera is None:
            return "", None
        self._camera_name = node.name
        self._camera_id = camera_id
        self._camera = _copy_camera(camera)
        return self._camera_name, self._camera

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


def _copy_camera(camera: CameraView) -> CameraView:
    return replace(
        camera,
        eye=np.asarray(camera.eye).copy(),
        target=np.asarray(camera.target).copy(),
        up=np.asarray(camera.up).copy(),
        focal_length=np.asarray(camera.focal_length).copy(),
        sensor_size=np.asarray(camera.sensor_size).copy(),
        principal_offset=np.asarray(camera.principal_offset).copy(),
    )
