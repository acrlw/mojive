"""Camera preview presentation in the native viewport's unoccupied area."""

from imgui_bundle import imgui

from .design import DESIGN
from .widgets import MUTED, PANEL

HEADER_HEIGHT = 30.0
IMAGE_PADDING = 6.0
TITLE_PADDING = 8.0


def preview_region(viewport, obstacles, scale, position=None):
    """Fit a 16:9 preview below its header without covering viewport controls."""
    x, y, width, height = viewport
    margin, gap = 8 * scale, 6 * scale
    left, top = x + margin, y + margin
    right = x + width - DESIGN.camera_right * scale
    bottom = y + height - DESIGN.camera_bottom * scale
    if right <= left or bottom <= top:
        return None
    blocked = []
    edges = {left, right}
    for ox, oy, ow, oh in obstacles:
        a, b = max(left, ox - gap), min(right, ox + ow + gap)
        c, d = max(top, oy - gap), min(bottom, oy + oh + gap)
        if a < b and c < d:
            blocked.append((a, c, b, d))
            edges.update((a, b))
    edges = sorted(edges)
    best, best_score = None, None
    header, padding = HEADER_HEIGHT * scale, IMAGE_PADDING * scale
    for i, a in enumerate(edges[:-1]):
        for b in edges[i + 1 :]:
            intervals = sorted((c, d) for ox, c, ex, d in blocked if ox < b and ex > a)
            cursor = top
            for c, d in (*intervals, (bottom, bottom)):
                if c > cursor:
                    panel_width = min(
                        340 * scale,
                        b - a,
                        (c - cursor - header - padding) * 16 / 9 + 2 * padding,
                    )
                    if panel_width >= 56 * scale:
                        panel_height = header + (panel_width - 2 * padding) * 9 / 16 + padding
                        px, py = position or (right, bottom)
                        px = min(max(px, a), b - panel_width)
                        py = min(max(py, cursor), c - panel_height)
                        score = (panel_width, px + panel_width, py + panel_height)
                        if best_score is None or score > best_score:
                            best, best_score = (a, cursor, b - a, c - cursor), score
                cursor = max(cursor, d)
    return best


class WorkspaceCameraPreview:
    """Use the shared preview camera and render target with native header actions."""

    def __init__(self, workspace):
        self.workspace = workspace
        self._region_key = self._region = None

    def draw(self, preview_name):
        workspace, app = self.workspace, self.workspace.app
        preview = app.camera_preview
        workspace.rects.pop("camera_preview", None)
        if not app.viewport_layers.viewport_ui:
            preview.draw(workspace.window, (0, 0, 0, 0), preview_name)
            return
        obstacles = list(workspace.rects.get("overlays", ()))
        for name in ("_status_notice_bounds", "_status_path_bounds"):
            bounds = getattr(app, name, None)
            if bounds is not None:
                a, b, c, d = bounds
                obstacles.append((a, b, c - a, d - b))
        key = (app._viewport_rect, tuple(obstacles), workspace.s)
        if key != self._region_key:
            self._region = preview_region(
                app._viewport_rect, obstacles, workspace.s, preview._position
            )
            self._region_key = key
        imgui.push_style_var(imgui.StyleVar_.child_rounding, DESIGN.child_radius * workspace.s)
        imgui.push_style_color(imgui.Col_.child_bg, tuple(channel / 255 for channel in PANEL))
        try:
            preview.draw(
                workspace.window,
                self._region or (0, 0, 0, 0),
                preview_name,
                workspace.ui.translate,
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
            workspace.rects["camera_preview"] = rect
            workspace.rects["overlays"].append(rect)
            workspace.input_overlays.append(rect)

    def _header(self, camera_name, height):
        workspace, u = self.workspace, self.workspace.ui
        preview, s = workspace.app.camera_preview, workspace.s
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
            dragging = imgui.is_item_active() and imgui.is_mouse_dragging(0)
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
            with workspace.menus.style():
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
