"""Quiet native help tags with the same keycaps as the status bar."""

from dataclasses import dataclass
from math import ceil

from imgui_bundle import imgui

from mojive.design.native.geometry import independent_corner_scope, part_scope
from mojive.ui.workspace.design import DESIGN
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.widget_style import MUTED

TOOLTIP_FONT_SIZE = 12


@dataclass(frozen=True, slots=True)
class ShortcutHint:
    """Keep the localized action separate from its actual configured binding."""

    text: str
    shortcut: str


def draw_tooltip(ui, hint, *, translate=True):
    from mojive.ui.workspace.status_components import StatusBadge

    text = hint.text if isinstance(hint, ShortcutHint) else hint
    shortcut = hint.shortcut if isinstance(hint, ShortcutHint) else ""
    shortcut = "" if shortcut == "Unbound" else shortcut
    text = ui.translate(text) if translate else text
    scale = ui.s
    # A help tag owns its typography; it must not inherit a heading, numeric
    # field or icon-only control's active font from the caller's native stack.
    with independent_corner_scope(), part_scope(inherit=False):
        imgui.push_font(ui.body, TOOLTIP_FONT_SIZE * scale)
        imgui.push_style_color(imgui.Col_.text, tuple(v / 255 for v in ui.ink(MUTED)))
        try:
            badge_rect = None
            if shortcut:
                badge = StatusBadge(ui, keycap=True)
                width, height = badge.size(shortcut)
                with part_scope("status_hints"):
                    badge_style = ui.component_style("badge")
                    margins = badge_style.edges("margin", scale)
                    badge_radius = badge_style.get("radius", 3) * scale
                frame_width = ceil(max(0, width - margins[0] - margins[2]))
                frame_height = ceil(max(0, height - margins[1] - margins[3]))
                width = frame_width + margins[0] + margins[2]
                height = frame_height + margins[1] + margins[3]
                padding = (native_unit(6 * scale),) * 2
                gap = native_unit(8 * scale)
                text_size = imgui.calc_text_size(text, wrap_width=260 * scale)
                text_width, text_height = ceil(text_size.x), ceil(text_size.y)
                row_width = text_width + gap + frame_width
                row_height = max(text_height, frame_height)
                badge_radius = max(0, min(badge_radius, frame_width / 2 - 1, frame_height / 2 - 1))
                rounding = badge_radius + padding[0]
            else:
                padding = (8 * scale, 6 * scale)
                rounding = ui.component_style("button").get("radius", DESIGN.control_radius) * scale
            imgui.push_style_var(imgui.StyleVar_.window_padding, padding)
            imgui.push_style_var(imgui.StyleVar_.window_min_size, (0, 0))
            imgui.push_style_var(imgui.StyleVar_.window_rounding, rounding)
            try:
                pending = imgui.get_current_context().next_window_data
                previous_size = pending.has_flags, tuple(pending.size_val), pending.size_cond
                if shortcut:
                    imgui.set_next_window_size(
                        (row_width + 2 * padding[0], row_height + 2 * padding[1])
                    )
                if not imgui.begin_tooltip():
                    if shortcut:
                        pending.has_flags, pending.size_val, pending.size_cond = previous_size
                    return
                try:
                    window = imgui.internal.get_current_window()
                    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + 260 * scale)
                    try:
                        if shortcut:
                            start = imgui.get_cursor_screen_pos()
                            try:
                                imgui.set_cursor_screen_pos(
                                    (start.x, start.y + (row_height - text_height) / 2)
                                )
                                imgui.text_unformatted(text)
                                badge_rect = (
                                    start.x + text_width + gap,
                                    start.y,
                                    frame_width,
                                    frame_height,
                                )
                                with ui.corner_scope(
                                    max(0, window.window_rounding - padding[0]), "badge"
                                ):
                                    badge.draw(
                                        badge_rect[0] - margins[0],
                                        badge_rect[1] - margins[1] + height / 2,
                                        shortcut,
                                        ui.control_theme(),
                                        muted=True,
                                        width=width,
                                        height=height,
                                    )
                            finally:
                                imgui.set_cursor_screen_pos(start)
                                imgui.dummy((row_width, row_height))
                        else:
                            imgui.text_unformatted(text)
                    finally:
                        imgui.pop_text_wrap_pos()
                    if ui.design is not None:
                        ui.design.record(
                            "tooltip",
                            (),
                            (*window.pos, *window.size),
                            "tooltip",
                            text=text,
                            shortcut=shortcut,
                            fontSize=TOOLTIP_FONT_SIZE,
                            nativeRadius=window.window_rounding,
                            nativePadding=list(window.window_padding),
                            badgeRect=badge_rect,
                            badgeRadius=max(0, window.window_rounding - padding[0])
                            if shortcut
                            else None,
                            contentSize=(row_width, row_height) if shortcut else None,
                        )
                finally:
                    imgui.end_tooltip()
            finally:
                imgui.pop_style_var(3)
        finally:
            imgui.pop_style_color()
            imgui.pop_font()
