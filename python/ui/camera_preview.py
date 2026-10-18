"""Secondary scene rendering for the selected camera preview."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, floor, trunc
from typing import Any

import numpy as np
from imgui_bundle import imgui

from mojive.adapters.base import NodeType, SceneFrame, SceneSource
from mojive.render.backend import RenderRequest
from mojive.types import CameraView, ViewportImage
from mojive.ui.preview_geometry import preview_image_fit, preview_origin, preview_surface


class CameraPreview:
    def __init__(self) -> None:
        # A selected camera must not automatically cover the viewport.  The
        # preview is an explicit inspection aid, controlled from Inspector.
        self._enabled = False
        self._backend: Any | None = None
        self._image: ViewportImage | None = None
        self._source_generation = -1
        self._position: tuple[float, float] | None = None
        self._drag_offset: tuple[float, float] | None = None
        self._pinned = False
        self._locked = False
        self._camera_name = ""
        self._camera_id = -1
        self._camera: CameraView | None = None
        self._bounds: tuple[float, float, float, float] | None = None

    def update(
        self,
        main_backend: Any,
        source: SceneSource | None,
        source_generation: int,
        frame: SceneFrame,
        camera: CameraView | None,
        size: tuple[int, int],
    ) -> None:
        if not self._enabled or source is None or camera is None:
            self._image = None
            return
        backend = self._ensure_backend(main_backend, size)
        backend.resize(*size)
        if source_generation != self._source_generation:
            backend.set_scene(source)
            self._source_generation = source_generation
        for flag in main_backend.render_options():
            backend.set_flag(flag, main_backend.get_flag(flag))
        backend.set_background(main_backend.get_background())
        backend.set_geometry_style(main_backend.get_geometry_style())
        backend.set_contact_style(main_backend.get_contact_style())
        backend.set_shadow_quality(main_backend.get_shadow_quality())
        backend.set_debug_view(main_backend.get_debug_view())
        backend.set_label_mode(main_backend.get_label_mode())
        backend.set_frame_mode(main_backend.get_frame_mode())
        backend.set_bvh_depth(main_backend.get_bvh_depth())
        backend.set_camera(camera.with_aspect(size[0] / max(size[1], 1)))
        backend.highlight(0)
        backend.update(frame)
        # Preview has no picking or selection outline. The backend still adds
        # identity work when the selected debug view actually consumes it.
        self._image = backend.render(request=RenderRequest.color())

    def draw(
        self,
        window: Any,
        viewport: tuple[float, float, float, float],
        camera_name: str,
        translate: Any = None,
        *,
        draw_header: Any = None,
        margin_pt: float = 12.0,
        header_height_pt: float = 25.0,
        image_padding_pt: float = 0.0,
        maximum_width_pt: float = 340.0,
        image_height_pt: float | None = None,
        image_fit: str = "stretch",
        body_margin_pt: tuple[float, float, float, float] = (0, 0, 0, 0),
        row_gap_pt: float = 0.0,
        draw_frame: Any = None,
        draw_image: Any = None,
        record_image: Any = None,
        native_layout: bool = False,
    ) -> None:
        if not self._enabled:
            self._bounds = None
            return
        image = self._image
        if image is None:
            self._bounds = None
            return
        self.update_drag_position()
        t = translate or str
        _x, _y, width, height = viewport
        scale = window.style_scale
        header_height = header_height_pt * scale
        margin = margin_pt * scale
        left, top, right, bottom = body_margin_pt
        available_width, available_height = max(0, width - 2 * margin), max(0, height - 2 * margin)
        if native_layout:
            native_left, native_top = ceil(_x + margin), ceil(_y + margin)
            native_right, native_bottom = floor(_x + width - margin), floor(_y + height - margin)
            available_width = max(0, native_right - native_left)
            available_height = max(0, native_bottom - native_top)
        surface = preview_surface(
            available_width,
            available_height,
            maximum_width=min(
                maximum_width_pt * scale,
                max(180 * scale, width * 0.48) if draw_header is None else width,
            ),
            header_height=header_height,
            image_aspect=image.aspect,
            image_height=None if image_height_pt is None else image_height_pt * scale,
            insets=(
                (image_padding_pt + left) * scale,
                top * scale,
                (image_padding_pt + right) * scale,
                (image_padding_pt + bottom) * scale,
            ),
            row_gap=row_gap_pt * scale,
        )
        if surface is None:
            self._bounds = None
            return
        if native_layout:
            # BeginChild truncates its frame. Absorb the rounding remainder in
            # the header so the camera image retains its aspect and edge insets.
            panel_width = floor(surface.panel_width)
            ix, _iy, _iw, ih = surface.image_rect
            iw = panel_width - ix - (image_padding_pt + right) * scale
            if image_height_pt is None:
                ih = iw / max(image.aspect, 1e-9)
            body_height = (
                row_gap_pt * scale + top * scale + ih + (image_padding_pt + bottom) * scale
            )
            panel_height = ceil(header_height + body_height)
            header_height = panel_height - body_height
            surface = replace(
                surface,
                panel_width=panel_width,
                panel_height=panel_height,
                image_rect=(ix, header_height + row_gap_pt * scale + top * scale, iw, ih),
            )
        panel_width, panel_height = surface.panel_width, surface.panel_height
        px, py = preview_origin(
            viewport, (panel_width, panel_height), desired=self._position, margin=margin
        )
        if native_layout:
            px = min(max(trunc(px), native_left), native_right - panel_width)
            py = min(max(trunc(py), native_top), native_bottom - panel_height)
        self._bounds = (px, py, px + panel_width, py + panel_height)

        imgui.set_cursor_screen_pos(imgui.ImVec2(px, py))
        child_flags = imgui.ChildFlags_.borders.value if draw_frame is None else 0
        window_flags = (
            imgui.WindowFlags_.no_scrollbar.value | imgui.WindowFlags_.no_scroll_with_mouse.value
        )
        if draw_header is not None:
            imgui.push_style_var(imgui.StyleVar_.window_padding, (0.0, 0.0))
            imgui.push_style_var(imgui.StyleVar_.item_spacing, (0.0, 0.0))
        if draw_frame is not None:
            draw_frame(px, py, panel_width, panel_height)
        if not imgui.begin_child(
            "##camera_preview", imgui.ImVec2(panel_width, panel_height), child_flags, window_flags
        ):
            imgui.end_child()
            if draw_header is not None:
                imgui.pop_style_var(2)
            return
        if draw_header is not None:
            self._capture_drag(draw_header(camera_name, header_height), px, py)
        else:
            self._draw_header(camera_name, t, header_height, scale, px, py)
        ix, iy, iw, ih = surface.image_rect
        frame = (px + ix, py + iy, iw, ih)
        destination, uv0, uv1 = preview_image_fit(
            frame, image.aspect, fit=image_fit, flip_y=image.flip_y
        )
        dx, dy, dw, dh = destination
        imgui.set_cursor_screen_pos((dx, dy))
        texture = window.viewport_texture_ref(image)
        if draw_image is None:
            imgui.image(texture, imgui.ImVec2(dw, dh), uv0, uv1)
        else:
            draw_image(texture, destination, uv0, uv1)
        if record_image is not None:
            record_image(frame, destination, uv0, uv1)
        imgui.end_child()
        if draw_header is not None:
            imgui.pop_style_var(2)

    def update_drag_position(self) -> None:
        """Apply a captured header's absolute pointer anchor before surface layout."""
        if self._drag_offset is None:
            return
        io = imgui.get_io()
        if self._pinned or not io.mouse_down[0]:
            self._drag_offset = None
            return
        self._position = (
            float(io.mouse_pos.x) + self._drag_offset[0],
            float(io.mouse_pos.y) + self._drag_offset[1],
        )

    def _capture_drag(self, active, x, y):
        if active and not self._pinned and self._drag_offset is None:
            mouse = imgui.get_io().mouse_pos
            self._drag_offset = (x - float(mouse.x), y - float(mouse.y))
            self._position = (x, y)

    def _draw_header(self, camera_name, t, header_height, scale, px, py):
        pin_label = t("Pinned" if self._pinned else "Pin")
        lock_label = t("Locked" if self._locked else "Lock")
        pin_width = imgui.calc_text_size(pin_label).x + 18.0 * scale
        lock_width = imgui.calc_text_size(lock_label).x + 18.0 * scale
        spacing = imgui.get_style().item_spacing.x
        title_width = max(
            1.0,
            imgui.get_content_region_avail().x - pin_width - lock_width - 2.0 * spacing,
        )
        title = f"{t('Camera')} · {camera_name}"
        imgui.button(title, imgui.ImVec2(title_width, header_height))
        self._capture_drag(imgui.is_item_active(), px, py)
        imgui.same_line()
        if imgui.button(pin_label, imgui.ImVec2(pin_width, header_height)):
            self.set_pinned(not self._pinned)
        imgui.set_item_tooltip(
            t("Freeze the preview viewpoint and panel position. Scene motion continues.")
        )
        imgui.same_line()
        if imgui.button(lock_label, imgui.ImVec2(lock_width, header_height)):
            self.set_locked(not self._locked)
        imgui.set_item_tooltip(
            t("Follow this camera even when selection changes. The camera can still move.")
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if self._enabled:
            return
        self._image = None
        self._bounds = None
        self._drag_offset = None
        self._pinned = False
        self._locked = False

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        return self._bounds

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
            self._drag_offset = None

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked and self._camera_id >= 0)
        if self._locked:
            self._pinned = False

    def selected_camera(self, session) -> tuple[str, CameraView | None]:
        if not self._enabled:
            return "", None
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
        self._source_generation = -1
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
