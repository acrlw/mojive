"""Shared native popup styling and content budgets with explicit lifetimes."""

from contextlib import contextmanager
from contextvars import ContextVar

from imgui_bundle import imgui

from mojive.design.native.geometry import Box, Rect, budget_margins
from mojive.ui.design_palette import rgba8
from mojive.ui.workspace.design import BODY_SIZE, CONTROL_RADIUS, DESIGN

MENU_BG = (39, 45, 53, 255)
POPUP_ROWS = ContextVar("popup_rows", default=None)


def popup_box(ui, outer, *, part="menu"):
    """Use one outer budget for live popups and reference plates."""
    style = ui.component_style(part)
    margin = budget_margins(style.edges("margin", ui.s), outer.width, outer.height)
    padding = style.edges("padding", ui.s, (12, 6, 12, 6))
    frame = outer.inset(*margin)
    return Box(outer, frame, frame.inset(*padding), padding, margin)


def prepare_menu_popup(ui, *, minimum_width=0, auto_minimum_width=0, part="menu"):
    """Constrain authored popup dimensions inside its real viewport budget."""
    style, scale = ui.component_style(part), ui.s
    fixed_width = style.get("widthMode", "auto") == "fixed"
    fixed_height = style.get("heightMode", "auto") == "fixed"
    if not (fixed_width or fixed_height or auto_minimum_width):
        return
    viewport = imgui.get_main_viewport().size
    maximum_width, maximum_height = max(1, viewport.x - 16 * scale), max(1, viewport.y - 16 * scale)
    width = min(
        maximum_width,
        max(minimum_width, style.get("width", 240) * scale if fixed_width else auto_minimum_width),
    )
    height = min(maximum_height, style.get("height", 300) * scale) if fixed_height else 0
    imgui.set_next_window_size_constraints(
        (width, height),
        (
            width if fixed_width or auto_minimum_width else maximum_width,
            height if fixed_height else maximum_height,
        ),
    )


def menu_choice_width(ui, choices, selected, *, icon=None):
    """Predict choices independently of a popup's previously allocated width."""
    from mojive.ui.workspace.menu_item import Item
    from mojive.ui.workspace.menu_rows import menu_row_size

    with ui.component_scope("select_popup"):
        style = ui.component_style("select_popup")
        padding, margin = style.edges("padding", ui.s, (12, 6, 12, 6)), style.edges("margin", ui.s)
        widths = (
            menu_row_size(
                ui, Item(str(choice), checked=index == selected, icon=icon), part="select_option"
            )[0]
            for index, choice in enumerate(choices)
        )
        return max(widths, default=0) + padding[0] + padding[2] + margin[0] + margin[2]


@contextmanager
def menu_popup_style(ui, controls=False, *, part="menu"):
    scale = ui.s
    style = ui.component_style(part)
    item = ui.component_style(
        "select_option" if part == "select_popup" else "input_number" if controls else "menu_item"
    )
    font_size = style.get("fontSize", BODY_SIZE)
    item_font = item.get("fontSize", font_size)
    height = (
        item.get("height", 30)
        if part == "select_popup"
        else item.get("height", DESIGN.menu_item_height)
    )
    if not item.explicit("height"):
        height = max(height, item.line_height(item_font * 1.4))
    native_row_padding = max(0, height - item_font)
    padding = style.edges("padding", fallback=(16, 14, 16, 14) if controls else (12, 6, 12, 6))
    item_padding = item.edges("padding", fallback=(9, 0, 9, 0))
    values = [
        (imgui.StyleVar_.window_padding, (padding[0], padding[1]) if controls else (0, 0)),
        (imgui.StyleVar_.window_min_size, (0, 0)),
        (imgui.StyleVar_.item_spacing, (12, native_row_padding)),
        (
            imgui.StyleVar_.frame_padding,
            (item_padding[0], max(item_padding[1], (height - item_font) / 2)),
        ),
    ]
    for key, value in values:
        imgui.push_style_var(key, tuple(v * scale for v in value))
    imgui.push_style_var(imgui.StyleVar_.popup_rounding, style.get("radius", 8) * scale)
    imgui.push_style_var(
        imgui.StyleVar_.menu_item_rounding, item.get("radius", CONTROL_RADIUS) * scale
    )
    imgui.push_style_var(imgui.StyleVar_.scrollbar_size, 6 * scale)
    colors = [
        (imgui.Col_.popup_bg, MENU_BG),
        (imgui.Col_.border, (76, 86, 99, 150)),
        (imgui.Col_.header, rgba8("primary", 24)),
        (imgui.Col_.header_hovered, (255, 255, 255, 13)),
        (imgui.Col_.header_active, (255, 255, 255, 18)),
    ]
    for key, value in colors:
        imgui.push_style_color(key, imgui.ImVec4(*(v / 255 for v in value)))
    imgui.push_font(ui.body, font_size * scale)
    with ui.component_scope(part):
        try:
            yield
        finally:
            imgui.pop_font()
            imgui.pop_style_color(len(colors))
            imgui.pop_style_var(len(values) + 3)


@contextmanager
def menu_popup_content(ui, key, *, part="menu"):
    """Budget content after Begin; the caller closes the native popup afterward."""
    scale = ui.s
    pos, size = imgui.get_window_pos(), imgui.get_window_size()
    style = ui.component_style(part)
    outer = Rect(pos.x, pos.y, size.x, size.y)
    box = popup_box(ui, outer, part=part)
    content = box.content
    window = imgui.get_current_context().current_window
    content = Rect(
        content.x,
        content.y,
        max(0, min(content.right, outer.right - window.scrollbar_sizes.x) - content.x),
        max(0, min(content.bottom, outer.bottom - window.scrollbar_sizes.y) - content.y),
    )
    box = Box(outer, box.frame, content, box.padding, box.margin)
    minimum_trigger = ui.hits.get(key, (0, 0, 0, 0))[2]
    rows = []
    rows_token = POPUP_ROWS.set((window.id_, rows))
    imgui.set_cursor_screen_pos((box.content.x, box.content.y - imgui.get_scroll_y()))
    imgui.push_clip_rect(
        (box.content.x, box.content.y), (box.content.right, box.content.bottom), True
    )
    try:
        with ui.component_scope(part):
            yield box
    finally:
        imgui.pop_clip_rect()
        ui.record_component_box(
            part,
            box,
            key="local-menu-" + key,
            fields=[row.tuple() for row, _gap in rows],
            windowRect=list(outer.tuple()),
            requestedWidth=style.get("width", 240) * scale,
            minimumTrigger=minimum_trigger,
            widthLimitedByTrigger=style.get("widthMode", "auto") == "fixed"
            and style.get("width", 240) * scale < minimum_trigger,
        )
        POPUP_ROWS.reset(rows_token)
        cursor = imgui.get_cursor_screen_pos()
        bottom = cursor.y
        if rows:
            last, gap = rows[-1]
            if abs(bottom - last.bottom - gap) < 0.5:
                bottom = last.bottom
        imgui.set_cursor_screen_pos((box.content.x, bottom + box.padding[3] + box.margin[3]))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
        imgui.dummy((box.content.width + box.padding[2] + box.margin[2], 0))
        imgui.pop_style_var()
