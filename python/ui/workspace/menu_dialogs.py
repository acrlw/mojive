"""Native interaction guide dialog; document dialogs remain owned by ViewerApp."""

from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.ui.input_bindings import InputAction
from mojive.ui.workspace.component_layout import control_frame_size, control_size
from mojive.ui.workspace.design import BODY_SIZE
from mojive.ui.workspace.form_popup import form_corners
from mojive.ui.workspace.widgets import DIM, MUTED, TEXT


def dialogs(menu):
    if menu.dialog_pending:
        menu.dialog = menu.dialog_pending
        menu.dialog_pending = None
        imgui.open_popup("##workspace-dialog")
    corners = form_corners(
        menu.ui, "dialog", control_part="button", inner_limit=max(0, 26 * menu.s / 2 - 1)
    )
    with (
        menu.style(controls=True),
        menu.ui.component_scope("dialog"),
        menu.ui.corner_scope(corners.inner_radius, ("button", "icon_button")),
    ):
        vp = imgui.get_main_viewport()
        u, scale = menu.ui, menu.s
        style = u.component_style("dialog")
        margin = style.edges("margin", scale)
        width = min(
            style.get("width", 500) * scale,
            style.get("maxWidth", 900) * scale,
            vp.size.x - 32 * scale - margin[0] - margin[2],
        )
        height = min(430 * scale, vp.size.y - 32 * scale - margin[1] - margin[3])
        imgui.set_next_window_size((width, height))
        imgui.set_next_window_pos(
            (
                vp.pos.x + vp.size.x / 2 + (margin[0] - margin[2]) / 2,
                vp.pos.y + vp.size.y / 2 + (margin[1] - margin[3]) / 2,
            ),
            imgui.Cond_.appearing,
            (0.5, 0.5),
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_rounding, corners.radius)
        opened, _ = imgui.begin_popup_modal(
            "##workspace-dialog",
            None,
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_saved_settings,
        )
        imgui.pop_style_var(2)
        if not opened:
            return
        from mojive.ui.workspace.window_chrome import draw_floating_shadow

        draw_floating_shadow(menu.ui)
        menu.record_window("Dialog", ("Dialog",))
        s = scale
        origin = imgui.get_window_pos()
        frame = u.component_box("dialog", origin.x, origin.y, width, height, key="guide-dialog")
        header_style = u.component_style("dialog_header")
        body_style = u.component_style("dialog_body")
        footer_style = u.component_style("dialog_footer")
        row_gap = style.get("rowGap", 0) * s
        header_pad = header_style.edges("padding", s, tuple(p / s for p in corners.padding))
        header_margin = header_style.edges("margin", s)
        generic_edited = u.design is not None and any(
            u.design.runtime.edited(part, "fontSize") for part in ("dialog", "dialog_header")
        )
        title_field = (
            "fontSize" if generic_edited and not header_style.explicit("titleSize") else "titleSize"
        )
        subtitle_field = (
            "fontSize"
            if generic_edited and not header_style.explicit("subtitleSize")
            else "subtitleSize"
        )
        title_size = header_style.get(title_field, 20)
        subtitle_size = header_style.get(subtitle_field, BODY_SIZE)
        header_gap = header_style.get("gap", 8) * s
        close_gap = header_style.get("columnGap", header_style.get("gap", 8)) * s
        close_size = control_size(u, "close-guide", 26, 26, icon="close")
        title_height = max(
            28 * s,
            close_size[1],
            max(title_size, header_style.line_height(title_size * 1.4)) * s,
        )
        header_height = (
            title_height
            + subtitle_size * s
            + header_gap
            + header_pad[1]
            + header_pad[3]
            + header_margin[1]
            + header_margin[3]
        )
        header = u.component_box(
            "dialog_header",
            frame.content.x,
            frame.content.y,
            frame.content.width,
            header_height,
            key="guide-dialog-header",
            padding=header_pad,
            observe=False,
        )
        close_width = min(header.content.width, close_size[0])
        title = Rect(
            header.content.x,
            header.content.y,
            max(0, header.content.width - close_width - close_gap),
            title_height,
        )
        u.component_text(
            "dialog_header",
            title,
            menu.dialog,
            TEXT,
            size=title_size,
            font_field=title_field,
            bold=True,
        )
        close = header_style.align_rect(
            Rect(header.content.x, title.y, header.content.width, title.height),
            close_width,
            close_size[1],
            horizontal="end",
        )
        u.record_component_box(
            "dialog_header",
            header,
            key="guide-dialog-header",
            titleRect=list(title.tuple()),
            closeRect=list(close.tuple()),
            fieldRegions={
                "gap": [header.content.x, title.bottom, header.content.width, header_gap],
                "columnGap": [title.right, title.y, close.x - title.right, title.height],
            },
        )
        close_frame = control_frame_size(u, "close-guide", close_width, close_size[1], icon="close")
        if u.button(
            "close-guide",
            close.x,
            close.y,
            *close_frame,
            icon="close",
            tooltip="Close dialog",
            width_limit=close_frame[0] * s,
        ) or imgui.is_key_pressed(imgui.Key.escape):
            imgui.close_current_popup()
        u.component_text(
            "dialog_header",
            Rect(
                header.content.x, title.bottom + header_gap, header.content.width, subtitle_size * s
            ),
            "Mojive",
            DIM,
            size=subtitle_size,
            font_field=subtitle_field,
            translate=False,
        )
        footer_pad = footer_style.edges("padding", s, tuple(p / s for p in corners.padding))
        footer_margin = footer_style.edges("margin", s)
        with u.component_scope("dialog_footer"):
            done_style = u.component_style("button")
            done_padding = done_style.edges("padding", s, (6, 0, 6, 0))
            done_font = done_style.get("fontSize", BODY_SIZE)
            done_width = max(
                72 * s,
                u.measure("Done", done_font, font_resolved=True)
                + done_padding[0]
                + done_padding[2],
            )
            done_size = control_size(u, "done-guide", done_width / s, 28, label="Done")
        footer_height = (
            done_size[1] + footer_pad[1] + footer_pad[3] + footer_margin[1] + footer_margin[3]
        )
        footer = u.component_box(
            "dialog_footer",
            frame.content.x,
            frame.content.bottom - footer_height,
            frame.content.width,
            footer_height,
            key="guide-dialog-footer",
            padding=footer_pad,
        )
        body = u.component_box(
            "dialog_body",
            frame.content.x,
            header.outer.bottom + row_gap,
            frame.content.width,
            max(s, footer.outer.y - row_gap - header.outer.bottom),
            key="guide-dialog-body",
            padding=(corners.padding[0], 0, corners.padding[2], 0),
            observe=False,
        )
        p = imgui.ImVec2(body.content.x, body.content.y)
        w = body.content.width
        imgui.set_cursor_screen_pos((p.x, p.y))
        imgui.push_style_color(imgui.Col_.child_bg, (0, 0, 0, 0))
        body_font = body_style.get("fontSize", style.get("fontSize", BODY_SIZE))
        body_line = max(body_font, body_style.line_height(body_font * 1.4)) * s
        body_gap = (body_style.get("rowGap") if u.design is not None else 12) * s
        imgui.push_font(u.resolve_font(body_style), body_font * s)
        for slot, fallback in ((imgui.Col_.text, TEXT), (imgui.Col_.text_disabled, DIM)):
            fallback = u.ink(fallback, body_style)
            tint = u.style_color(body_style, "textColor", fallback)
            imgui.push_style_color(slot, tuple(channel / 255 for channel in tint))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (8 * s, body_gap))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.begin_child("Guide content", (w, body.content.height))
        imgui.pop_style_var()
        body_rows = []
        if menu.dialog == "Interaction guide":
            bindings = menu.workspace.app.input_bindings
            for title, value in [
                ("Orbit", "Drag an empty area"),
                ("Pan", "Middle mouse drag"),
                ("Zoom", "Mouse wheel"),
                (
                    "Fly camera",
                    " / ".join(
                        " ".join(bindings.label(action) for action in group)
                        for group in (
                            (
                                InputAction.FLY_FORWARD,
                                InputAction.FLY_LEFT,
                                InputAction.FLY_BACK,
                                InputAction.FLY_RIGHT,
                            ),
                            (InputAction.FLY_UP, InputAction.FLY_DOWN),
                        )
                    ),
                ),
                (
                    "Transform tools",
                    "V / "
                    + " / ".join(
                        bindings.label(action)
                        for action in (InputAction.GIZMO_TRANSLATE, InputAction.GIZMO_ROTATE)
                    ),
                ),
                ("World / body frame", bindings.label(InputAction.GIZMO_SPACE)),
                ("Snap", bindings.label(InputAction.SNAP)),
                (
                    "Frame all / focus",
                    bindings.label(InputAction.FRAME_SCENE) + " / double-click",
                ),
                (
                    "Settings",
                    "F9 / " + ("Cmd" if imgui.get_io().config_mac_osx_behaviors else "Ctrl") + ",",
                ),
                ("Precise gizmo input", "Double-click a handle"),
                (
                    "Undo / redo",
                    ("Cmd" if imgui.get_io().config_mac_osx_behaviors else "Ctrl") + "+Z / Shift+Z",
                ),
                ("Simulation playback", "Space"),
                ("Timeline", "T"),
                ("Dock panels", "Drag a panel tab"),
            ]:
                q = imgui.get_cursor_screen_pos()
                column = w * 0.47
                if (
                    u.measure(title, body_font, style=body_style) + 12 * s > column
                    or u.measure(value, body_font, style=body_style) > w - column
                ):
                    imgui.push_text_wrap_pos(0)
                    fallback = u.ink(MUTED, body_style)
                    tint = u.style_color(body_style, "textColor", fallback)
                    imgui.text_colored(tuple(v / 255 for v in tint), u.translate(title))
                    imgui.text_wrapped(u.translate(value))
                    imgui.pop_text_wrap_pos()
                    imgui.dummy((0, body_gap))
                else:
                    u.component_text("dialog_body", Rect(q.x, q.y, column, body_line), title, MUTED)
                    u.component_text(
                        "dialog_body",
                        Rect(q.x + column, q.y, max(0, w - column), body_line),
                        value,
                        TEXT,
                    )
                    imgui.dummy((w, body_line))
                body_rows.append([q.x, q.y, w, imgui.get_item_rect_max().y - q.y])
        else:
            from mojive.build_info import current_build

            version, commit = current_build()
            for text in (
                "Backend-neutral 3D scene editing and simulation tools.",
                "Scene authoring, materials, cameras and lights.",
                "Joint posing, transform gizmos and simulation playback.",
                "OpenGL and bgfx rendering; Python APIs and remote scene sources.",
            ):
                imgui.text_wrapped(u.translate(text))
            imgui.spacing()
            imgui.text(u.translate("Version") + ": " + version)
            imgui.text(u.translate("Commit") + ": " + (commit or u.translate("Unavailable")))
            imgui.text_disabled(u.translate("Dear ImGui") + " " + imgui.get_version())
        u.record_component_box("dialog_body", body, key="guide-dialog-body", fields=body_rows)
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_font()
        imgui.pop_style_color(3)
        u.line(footer.frame.x, footer.frame.y, footer.frame.width)
        button_width = min(footer.content.width, done_size[0])
        button = footer_style.align_rect(
            footer.content, button_width, done_size[1], horizontal=footer_style.get("align", "end")
        )
        with u.component_scope("dialog_footer"):
            button_frame = control_frame_size(
                u, "done-guide", button.width, button.height, label="Done"
            )
            if u.button(
                "done-guide",
                button.x,
                button.y,
                *button_frame,
                label="Done",
                filled=True,
                width_limit=button_frame[0] * s,
            ):
                imgui.close_current_popup()
        imgui.end_popup()
