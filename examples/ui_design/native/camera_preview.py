"""Camera preview presentation in the native viewport's unoccupied area."""

from imgui_bundle import imgui

from mojive.ui.workspace.widgets import MUTED, PANEL

HEADER_HEIGHT = 30.0
IMAGE_PADDING = 6.0
TITLE_PADDING = 8.0


def preview_region(viewport, obstacles, scale, position=None):
    """Use the production viewport and manual-anchor bounds for this legacy header."""
    from mojive.ui.workspace.camera_preview import preview_region as native_region

    return native_region(
        viewport,
        obstacles,
        scale,
        position,
        header_height=HEADER_HEIGHT,
        image_padding=IMAGE_PADDING,
    )


class ReferenceCameraPreview:
    """Use the shared preview camera and render target with native header actions."""

    def __init__(self, study):
        self.study = study
        self._region_key = self._region = None

    def draw(self, preview_name):
        study, app = self.study, self.study.preview
        preview = app.camera_preview
        preview.update_drag_position()
        study.rects.pop("camera_preview", None)
        if not app.viewport_layers.viewport_ui:
            preview.draw(study.window, (0, 0, 0, 0), preview_name)
            return
        obstacles = list(study.rects.get("overlays", ()))
        for name in ("_status_notice_bounds", "_status_path_bounds"):
            bounds = getattr(app, name, None)
            if bounds is not None:
                a, b, c, d = bounds
                obstacles.append((a, b, c - a, d - b))
        key = (app._viewport_rect, tuple(obstacles), study.s, preview._position)
        if key != self._region_key:
            self._region = preview_region(app._viewport_rect, obstacles, study.s, preview._position)
            self._region_key = key
        imgui.push_style_var(imgui.StyleVar_.child_rounding, 8 * study.s)
        imgui.push_style_color(imgui.Col_.child_bg, tuple(channel / 255 for channel in PANEL))
        try:
            preview.draw(
                study.window,
                self._region or (0, 0, 0, 0),
                preview_name,
                study.ui.translate,
                draw_header=self._header,
                margin_pt=0,
                header_height_pt=HEADER_HEIGHT,
                image_padding_pt=IMAGE_PADDING,
            )
        finally:
            imgui.pop_style_color()
            imgui.pop_style_var()
        if preview.bounds is not None and preview.enabled:
            a, b, c, d = preview.bounds
            rect = (a, b, c - a, d - b)
            study.rects["camera_preview"] = rect
            study.rects["overlays"].append(rect)
            study.input_overlays.append(rect)

    def _header(self, camera_name, height):
        study, u = self.study, self.study.ui
        preview, s = study.preview.camera_preview, study.s
        pos, available = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
        x, y, width = pos.x, pos.y, available.x
        compact = width < 152 * s
        slot, gap, inset = 24 * s, 2 * s, 6 * s
        actions_width = (2 if compact else 3) * (slot + gap) - gap
        title_width = max(0, width - actions_width - 2 * inset - gap)
        button_y = y + (height - slot) / 2
        action_x = x + width - inset - actions_width
        dragging = False
        if title_width > 0:
            u.button(
                "camera-preview-title",
                x + inset,
                button_y,
                title_width / s,
                24,
                tooltip=camera_name,
            )
            dragging = imgui.is_item_active()
            u.text(
                x + inset + TITLE_PADDING * s,
                y + (height - 14 * s) / 2,
                camera_name,
                MUTED,
                max_width=max(0, title_width - 2 * TITLE_PADDING * s),
            )
        popup = "Camera preview actions"
        if compact:
            if u.button(
                "camera-preview-more",
                action_x,
                button_y,
                24,
                24,
                icon="more",
                icon_role="subtle",
                tooltip="Camera Preview",
            ):
                imgui.open_popup(popup)
            with study.menus.style():
                if imgui.begin_popup(popup):
                    for key, label, active, callback in (
                        ("pin", "Pin", preview.pinned, preview.set_pinned),
                        ("lock", "Lock", preview.locked, preview.set_locked),
                    ):
                        clicked, _ = imgui.menu_item(u.translate(label), "", active)
                        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                        u.hits["camera-preview-" + key] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
                        if clicked:
                            callback(not active)
                    imgui.end_popup()
        else:
            for index, key, label, active, callback in (
                (0, "pin", "Pin", preview.pinned, preview.set_pinned),
                (1, "lock", "Lock", preview.locked, preview.set_locked),
            ):
                if u.button(
                    "camera-preview-" + key,
                    action_x + index * (slot + gap),
                    button_y,
                    24,
                    24,
                    icon=key,
                    icon_role="subtle",
                    active=active,
                    tooltip=label,
                ):
                    callback(not active)
        if u.button(
            "camera-preview-close",
            x + width - inset - slot,
            button_y,
            24,
            24,
            icon="close",
            icon_role="subtle",
            tooltip="Close Camera Preview",
        ):
            preview.set_enabled(False)
        imgui.set_cursor_screen_pos((x, y + height))
        return dragging
