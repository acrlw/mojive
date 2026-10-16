"""Camera preview presentation in the native viewport's unoccupied area."""

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.preview_geometry import preview_surface
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.widgets import MUTED, PANEL

HEADER_HEIGHT = 30.0
IMAGE_PADDING = 6.0


def preview_region(
    viewport,
    obstacles,
    scale,
    position=None,
    *,
    maximum_width=340,
    header_height=HEADER_HEIGHT,
    body_height=None,
    body_margin=(0, 0, 0, 0),
    margins=(0, 0, 0, 0),
    right_inset=0,
    bottom_inset=0,
    row_gap=0,
    aspect=16 / 9,
):
    """Choose the largest unobstructed budget for the measured preview surface."""
    x, y, width, height = viewport
    margin, gap = 8 * scale, 6 * scale
    left, top = x + margin + margins[0], y + margin + margins[1]
    right = x + width - right_inset * scale - margins[2]
    bottom = y + height - bottom_inset * scale - margins[3]
    if right <= left or bottom <= top:
        return None
    blocked, edges = [], {left, right}
    for ox, oy, ow, oh in obstacles:
        a, b = max(left, ox - gap), min(right, ox + ow + gap)
        c, d = max(top, oy - gap), min(bottom, oy + oh + gap)
        if a < b and c < d:
            blocked.append((a, c, b, d))
            edges.update((a, b))
    edges, best, best_score = sorted(edges), None, None
    insets = (
        IMAGE_PADDING * scale + body_margin[0],
        body_margin[1],
        IMAGE_PADDING * scale + body_margin[2],
        IMAGE_PADDING * scale + body_margin[3],
    )
    for i, a in enumerate(edges[:-1]):
        for b in edges[i + 1 :]:
            intervals = sorted((c, d) for ox, c, ex, d in blocked if ox < b and ex > a)
            cursor = top
            for c, d in (*intervals, (bottom, bottom)):
                if c > cursor:
                    surface = preview_surface(
                        b - a,
                        c - cursor,
                        maximum_width=maximum_width * scale,
                        header_height=header_height * scale,
                        image_height=body_height,
                        insets=insets,
                        row_gap=row_gap * scale,
                        image_aspect=aspect,
                    )
                    if surface is not None and surface.panel_width >= 56 * scale:
                        pw, ph = surface.panel_width, surface.panel_height
                        px, py = position or (right, bottom)
                        px, py = min(max(px, a), b - pw), min(max(py, cursor), c - ph)
                        score = (
                            (surface.image_rect[3], pw, px + pw, py + ph)
                            if body_height is not None
                            else (pw, px + pw, py + ph)
                        )
                        if best_score is None or score > best_score:
                            best, best_score = (a, cursor, b - a, c - cursor), score
                cursor = max(cursor, d)
    return best


class WorkspaceCameraPreview:
    """Use the shared camera and render target with measured native presentation."""

    def __init__(self, workspace):
        self.workspace = workspace
        self._region_key = self._region = None

    def _header_height(self):
        ui, scale = self.workspace.ui, self.workspace.s
        style = ui.component_style("camera_header")
        padding, margin = style.edges("padding", scale, (6, 3, 6, 3)), style.edges("margin", scale)
        font, icon = style.get("fontSize", 14), style.get("iconSize", 14)
        actions = max(
            control_size(ui, "camera-preview-" + key, 24, 24, icon=key, icon_role="subtle")[1]
            for key in ("pin", "lock", "close", "more")
        )
        return (
            max(actions, style.line_height(font * 1.4) * scale, icon * scale)
            + padding[1]
            + padding[3]
            + margin[1]
            + margin[3]
        ) / scale

    def draw(self, preview_name):
        workspace, app, ui = self.workspace, self.workspace.app, self.workspace.ui
        preview, scale = app.camera_preview, workspace.s
        workspace.rects.pop("camera_preview", None)
        if not app.viewport_layers.viewport_ui:
            preview.draw(workspace.window, (0, 0, 0, 0), preview_name)
            return
        style, body = ui.component_style("camera_preview"), ui.component_style("camera_body")
        margins, body_margin = style.edges("margin", scale), body.edges("margin", scale)
        header_height = self._header_height()
        body_height = (
            body.get("height", 163) * scale if body.get("heightMode", "auto") == "fixed" else None
        )
        obstacles = list(workspace.rects.get("overlays", ()))
        for name in ("_status_notice_bounds", "_status_path_bounds"):
            bounds = getattr(app, name, None)
            if bounds is not None:
                a, b, c, d = bounds
                obstacles.append((a, b, c - a, d - b))
        options = {
            "maximum_width": style.get("width", 340),
            "header_height": header_height,
            "body_height": body_height,
            "body_margin": body_margin,
            "margins": margins,
            "right_inset": style.get("right", 0),
            "bottom_inset": style.get("bottom", 0),
            "row_gap": style.get("rowGap", 0),
            "aspect": preview._image.aspect if preview._image is not None else 16 / 9,
        }
        key = (
            app._viewport_rect,
            tuple(obstacles),
            scale,
            preview._position,
            tuple(options.items()),
        )
        if key != self._region_key:
            self._region = preview_region(
                app._viewport_rect, obstacles, scale, preview._position, **options
            )
            self._region_key = key
        imgui.push_style_color(imgui.Col_.child_bg, (0, 0, 0, 0))
        try:
            preview.draw(
                workspace.window,
                self._region or (0, 0, 0, 0),
                preview_name,
                ui.translate,
                draw_header=self._header,
                margin_pt=0,
                header_height_pt=header_height,
                image_padding_pt=IMAGE_PADDING,
                maximum_width_pt=options["maximum_width"],
                image_height_pt=None if body_height is None else body_height / scale,
                body_margin_pt=tuple(v / scale for v in body_margin),
                row_gap_pt=options["row_gap"],
                image_fit=body.get("fit", "contain"),
                draw_frame=lambda x, y, w, h: ui.rect(x, y, w, h, PANEL, style.get("radius", 8)),
                record_image=self._record_image,
            )
        finally:
            imgui.pop_style_color()
        if preview.bounds is not None and preview.enabled:
            a, b, c, d = preview.bounds
            rect = (a, b, c - a, d - b)
            workspace.rects["camera_preview"] = rect
            workspace.rects["overlays"].append(rect)
            workspace.input_overlays.append(rect)
            if ui.design is not None:
                ui.design.record(
                    "camera-preview",
                    ("camera_preview",),
                    rect,
                    "container",
                    outer=[
                        a - margins[0],
                        b - margins[1],
                        c - a + margins[0] + margins[2],
                        d - b + margins[1] + margins[3],
                    ],
                    margin=list(margins),
                )

    def _record_image(self, frame, destination, uv0, uv1):
        ui = self.workspace.ui
        if ui.design is not None:
            margin = ui.component_style("camera_body").edges("margin", ui.s)
            x, y, w, h = frame
            ui.design.record(
                "camera-preview-image",
                ("camera_body",),
                frame,
                "image",
                content=list(destination),
                outer=[
                    x - margin[0],
                    y - margin[1],
                    w + margin[0] + margin[2],
                    h + margin[1] + margin[3],
                ],
                margin=list(margin),
                uv0=list(uv0),
                uv1=list(uv1),
            )

    def _header(self, camera_name, height):
        workspace, ui, scale = self.workspace, self.workspace.ui, self.workspace.s
        preview = workspace.app.camera_preview
        pos, width = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail().x
        box = ui.component_box(
            "camera_header",
            pos.x,
            pos.y,
            width,
            height,
            key="camera-preview-header",
            padding=(6 * scale, 3 * scale, 6 * scale, 3 * scale),
        )
        style, content = ui.component_style("camera_header"), box.content
        gap = style.get("columnGap", style.get("gap", 6)) * scale
        keys = ("pin", "lock", "close")
        sizes = {
            key: control_size(ui, "camera-preview-" + key, 24, 24, icon=key, icon_role="subtle")
            for key in (*keys, "more")
        }
        icon = style.get("iconSize", 14) * scale
        compact = content.width < sum(sizes[k][0] for k in keys) + 3 * gap + icon + 32 * scale
        keys = ("more", "close") if compact else keys
        actions_width = sum(sizes[k][0] for k in keys) + gap * (len(keys) - 1)
        title_width = max(0, content.width - actions_width - gap)
        dragging = False
        if title_width > 0:
            title = Rect(content.x, content.y, title_width, content.height)
            imgui.set_cursor_screen_pos((title.x, title.y))
            imgui.invisible_button("##camera-preview-title", (title.width, title.height))
            ui.hits["camera-preview-title"] = title.tuple()
            dragging = imgui.is_item_active() and imgui.is_mouse_dragging(0)
            glyph = style.align_rect(title, icon, icon, horizontal="start")
            ui.icon("camera", glyph.x, glyph.y, icon / scale, MUTED)
            caption = title.inset(icon + gap)
            ui.component_text("camera_header", caption, camera_name, MUTED, translate=False)
            if (
                ui.measure(camera_name, style.get("fontSize", 14), translate=False) > caption.width
                and imgui.is_item_hovered()
            ):
                ui.tooltip(camera_name, translate=False)
        action_x = content.right - actions_width
        popup = "Camera preview actions"
        with ui.component_scope("camera_header"):
            for key in keys:
                aw, ah = sizes[key]
                placement = style.align_rect(
                    Rect(action_x, content.y, aw, content.height), aw, ah, horizontal="start"
                )
                fw, fh = control_frame_size(ui, "camera-preview-" + key, aw, ah, icon=key)
                active = (
                    preview.pinned if key == "pin" else preview.locked if key == "lock" else False
                )
                clicked = ui.button(
                    "camera-preview-" + key,
                    placement.x,
                    placement.y,
                    fw,
                    fh,
                    icon=key,
                    icon_role="subtle",
                    width_limit=fw * scale,
                    active=active,
                    tooltip={
                        "pin": "Pin",
                        "lock": "Lock",
                        "more": "Camera Preview",
                        "close": "Close Camera Preview",
                    }[key],
                )
                if clicked:
                    if key == "more":
                        imgui.open_popup(popup)
                    elif key == "close":
                        preview.set_enabled(False)
                    else:
                        (preview.set_pinned if key == "pin" else preview.set_locked)(not active)
                action_x += aw + gap
        with workspace.menus.local_popup(popup) as opened:
            if opened:
                for key, label, active, callback in (
                    ("pin", "Pin", preview.pinned, preview.set_pinned),
                    ("lock", "Lock", preview.locked, preview.set_locked),
                ):
                    clicked, _ = workspace.menus.local_item(
                        label, checked=active, key="camera-preview-" + key, icon=key
                    )
                    if clicked:
                        callback(not active)
        return dragging
