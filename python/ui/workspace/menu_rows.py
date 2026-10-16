"""Native menu-row geometry shared by commands, local actions and choice lists."""

from dataclasses import dataclass

from imgui_bundle import imgui

from mojive.design.native.geometry import Box, Rect
from mojive.ui.design_palette import rgba8
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.interaction_paint import ItemPaint
from mojive.ui.workspace.design import BODY_SIZE, CONTROL_RADIUS, DESIGN
from mojive.ui.workspace.widgets import ACCENT, DIM, MUTED, TEXT, color

DISABLED = (126, 136, 149, 255)


@dataclass(frozen=True)
class MenuRowResult:
    activated: bool
    box: Box
    native_rect: tuple[float, float, float, float]
    label_rect: Rect
    tail_x: float
    visible: bool
    minimum_width: float


def menu_row_size(ui, row, *, part="menu_item"):
    """Measure the same glyph, text, padding and margins used by the painter."""
    style, scale = ui.component_style(part), ui.s
    choice = part == "select_option"
    padding = style.edges("padding", scale, (12, 0, 12, 0))
    margins = style.edges("margin", scale)
    font, icon = style.get("fontSize", BODY_SIZE), style.get("iconSize", 16) * scale
    gap = style.get("gap", 12) * scale
    tail_gap = (
        style.get("checkGap", 12)
        if choice
        else ui.metric("menu_item_content", "columnGap", ui.metric("menu_item_content", "gap", 16))
    ) * scale
    label_width = ui.measure(row.label, font, font_resolved=True)
    shortcut_width = (
        ui.measure(row.shortcut, font, translate=False, font_resolved=True) if row.shortcut else 0
    )
    width = (icon + gap if row.icon or not choice else 0) + label_width + shortcut_width
    width += tail_gap if row.shortcut or choice else 0
    width += icon if row.checked or row.submenu or choice else 0
    width += padding[0] + padding[2]
    if style.get("widthMode", "auto") == "fixed":
        width = style.get("width", width / scale) * scale
    height = max(
        (style.get("height", 30) if choice else style.get("height", DESIGN.menu_item_height))
        * scale,
        max(style.line_height(font * 1.4) * scale, icon) + padding[1] + padding[3],
    )
    return width + margins[0] + margins[2], height + margins[1] + margins[3]


def paint_menu_row(
    ui, row, key, *, content_right=None, open_submenu=None, minimum_height=0, part="menu_item"
):
    """Paint one row; the caller retains popup lifetime and command ownership."""
    style, scale = ui.component_style(part), ui.s
    choice = part == "select_option"
    enabled = row.enabled and not bool(
        imgui.get_current_context().current_item_flags & imgui.ItemFlags_.disabled
    )
    origin = imgui.get_cursor_screen_pos()
    width = max(
        0,
        (
            content_right
            if content_right is not None
            else origin.x + imgui.get_content_region_avail().x
        )
        - origin.x,
    )
    minimum_width, height = menu_row_size(ui, row, part=part)
    margins = style.edges("margin", scale)
    height = max(height, minimum_height + margins[1] + margins[3])
    box = ui.component_box(
        part,
        origin.x,
        origin.y,
        width,
        height,
        key=key,
        padding=(12 * scale, 0, 12 * scale, 0),
        observe=False,
        minimum_height=minimum_height,
    )
    font = style.get("fontSize", BODY_SIZE)
    window = imgui.get_current_context().current_window
    draw = imgui.get_window_draw_list()
    imgui.set_cursor_screen_pos((box.frame.x, box.frame.y))
    imgui.push_font(ui.body, font * scale)
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
    for slot in (
        imgui.Col_.text,
        imgui.Col_.text_disabled,
        imgui.Col_.header,
        imgui.Col_.header_hovered,
        imgui.Col_.header_active,
    ):
        imgui.push_style_color(slot, (0, 0, 0, 0))
    try:
        if open_submenu is not None:
            vertical_space = max(0, box.frame.height - font * scale)
            imgui.set_cursor_screen_pos((box.frame.x, box.frame.y + vertical_space / 2))
            imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, vertical_space))
            imgui.push_style_var(
                imgui.StyleVar_.frame_padding, (0, max(0, (box.frame.height - font * scale) / 2))
            )
            try:
                activated = open_submenu(row.label + "###" + key)
            finally:
                imgui.pop_style_var(2)
        else:
            flags = imgui.SelectableFlags_.disabled if not enabled else 0
            activated, _ = imgui.selectable(
                row.label + "###" + key,
                row.checked,
                flags,
                (max(1, box.frame.width), max(1, box.frame.height)),
            )
        visible = imgui.is_item_visible()
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    finally:
        imgui.pop_style_color(5)
        imgui.pop_style_var()
        imgui.pop_font()
    metadata = {"windowId": window.id_, "drawListSource": "window"}
    ui.hits[key] = box.frame.tuple()
    paint = ItemPaint.capture(imgui)
    enabled = enabled and not paint.disabled
    highlighted = enabled and (paint.hovered or paint.active or (choice and row.checked))
    if visible and highlighted:
        tint = (
            rgba8("field_active" if paint.active else "field_hovered")
            if choice and (paint.hovered or paint.active)
            else rgba8("primary", 24)
            if choice
            else (255, 255, 255, 18 if paint.active else 13)
        )
        smoothing = None
        if ui.design is not None:
            tint = ui.design.runtime.ink(tint)
            if ui.design.runtime.spec["native"]["g3"]:
                smoothing = ui.design.runtime.spec["native"]["g3_smoothing"]
        radius = style.get("radius", CONTROL_RADIUS) * scale
        if smoothing is None:
            draw.add_rect_filled(
                (box.frame.x, box.frame.y), (box.frame.right, box.frame.bottom), color(tint), radius
            )
        else:
            ImguiDraw2D(draw).rect_filled(
                (box.frame.x, box.frame.y),
                (box.frame.right, box.frame.bottom),
                tuple(c / 255 for c in tint),
                rounding=radius,
                smoothing=smoothing,
            )
    foreground = TEXT if enabled else DISABLED
    extent = style.get("iconSize", 16) * scale
    gap = style.get("gap", 12) * scale
    tail_gap = (
        style.get("checkGap", 12)
        if choice
        else ui.metric("menu_item_content", "columnGap", ui.metric("menu_item_content", "gap", 16))
    ) * scale
    shortcut_width = (
        ui.measure(row.shortcut, font, translate=False, font_resolved=True) if row.shortcut else 0
    )
    tail_width = extent if row.submenu or row.checked or choice else 0
    prefix = extent + gap if row.icon or not choice else 0
    label_width = max(
        0,
        box.content.width
        - prefix
        - shortcut_width
        - (tail_gap if row.shortcut or choice else 0)
        - tail_width,
    )
    label_rect = Rect(box.content.x + prefix, box.content.y, label_width, box.content.height)
    ui.record_component_box(
        part,
        box,
        key=key,
        draw=draw,
        background=visible and highlighted,
        visible=visible,
        hovered=hovered,
        labelRect=list(label_rect.tuple()),
        iconRect=[box.content.x, box.content.y, extent, box.content.height] if row.icon else None,
        label=row.label,
        clippedLabel=ui.measure(row.label, font, font_resolved=True) > label_rect.width + 0.5,
        fieldRegions={
            "gap": [box.content.x + extent, box.content.y, gap, box.content.height]
            if prefix
            else [],
            **(
                {"checkGap": [label_rect.right, box.content.y, tail_gap, box.content.height]}
                if choice
                else {}
            ),
        },
        **metadata,
    )
    tail_x = box.content.right - tail_width
    if visible:
        if ui.design is not None and not choice:
            ui.design.record(
                key + "/content",
                ("menu_item_content",),
                (
                    label_rect.x,
                    box.content.y,
                    max(0, box.content.right - label_rect.x),
                    box.content.height,
                ),
                "group",
                draw=draw,
                labelRect=list(label_rect.tuple()),
                label=row.label,
                clippedLabel=ui.measure(row.label, font, font_resolved=True)
                > label_rect.width + 0.5,
                shortcutRect=[
                    box.content.right - tail_width - shortcut_width - (gap if tail_width else 0),
                    box.content.y,
                    shortcut_width,
                    box.content.height,
                ]
                if row.shortcut
                else None,
                **metadata,
            )
        if row.icon:
            glyph = style.align_rect(box.content, extent, extent, horizontal="start")
            ui.icon(
                row.icon,
                glyph.x,
                glyph.y,
                extent / scale,
                MUTED if enabled else DISABLED,
                draw=draw,
            )
        ui.component_text(part, label_rect, row.label, foreground, draw=draw)
        if row.shortcut:
            tail_x -= shortcut_width + (gap if tail_width else 0)
            ui.component_text(
                part,
                Rect(tail_x, box.content.y, shortcut_width, box.content.height),
                row.shortcut,
                DIM if enabled else DISABLED,
                draw=draw,
                translate=False,
            )
        if row.submenu or row.checked:
            glyph = style.align_rect(
                Rect(box.content.right - extent, box.content.y, extent, box.content.height),
                extent,
                extent,
            )
            ui.icon(
                "right" if row.submenu else "check",
                glyph.x,
                glyph.y,
                extent / scale,
                (DIM if row.submenu else ACCENT) if enabled else DISABLED,
                draw=draw,
            )
    if hovered:
        if row.reason and not enabled:
            ui.tooltip(row.reason)
        elif ui.measure(row.label, font, font_resolved=True) > label_rect.width:
            ui.tooltip(row.label)
    return MenuRowResult(
        bool(activated),
        box,
        (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y),
        label_rect,
        tail_x,
        visible,
        minimum_width,
    )


def advance_menu_row(ui, result, *, part="menu"):
    """Publish the occupied row and its shared following gap to native layout."""
    outer = result.box.outer
    style = ui.component_style(part)
    gap = (
        style.get("rowGap", 6)
        if part == "select_popup"
        else ui.metric(part, "rowGap", DESIGN.menu_row_gap)
    )
    from mojive.ui.workspace.popup_style import POPUP_ROWS

    current = POPUP_ROWS.get()
    if current is not None and current[0] == imgui.get_current_context().current_window.id_:
        current[1].append((outer, gap * ui.s))
    y = outer.bottom + gap * ui.s
    imgui.set_cursor_screen_pos((outer.x, outer.bottom))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
    width = outer.width
    if ui.component_style(part).get("widthMode", "auto") == "auto":
        width = max(width, result.minimum_width)
    imgui.dummy((width, 0))
    imgui.pop_style_var()
    imgui.set_cursor_screen_pos((outer.x, y))
