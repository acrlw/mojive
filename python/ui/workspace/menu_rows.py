"""Native menu-row geometry shared by commands, local actions and choice lists."""

from dataclasses import dataclass
from math import ceil

from imgui_bundle import imgui

from mojive.design.native.geometry import Box, Rect
from mojive.ui.design_palette import rgba8
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.interaction_paint import ItemPaint
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.menu_geometry import (
    glyph_rect,
    native_box,
    native_edges,
    native_epsilon,
    native_unit,
    row_radius,
    text_budget,
)
from mojive.ui.workspace.widgets import ACCENT, DIM, MUTED, TEXT, color

DISABLED = rgba8("disabled")


@dataclass(frozen=True)
class MenuColumns:
    """One popup's capabilities and measured text columns in window coordinates."""

    check: bool
    icon: bool
    arrow: bool
    label_width: float
    shortcut_width: float


def menu_columns(ui, rows, *, part="menu_item"):
    """Measure the longest title and shortcut independently across the group."""
    rows = tuple(row for row in rows if row is not None)
    style = ui.component_style(part)
    font = style.get("fontSize", BODY_SIZE)
    return MenuColumns(
        part == "select_option" or any(row.checkable or row.checked for row in rows),
        any(row.icon for row in rows),
        any(row.submenu for row in rows),
        max(
            (ui.measure(row.label, font, font_resolved=True, style=style) for row in rows),
            default=0,
        ),
        max(
            (
                ui.measure(row.shortcut, font, translate=False, font_resolved=True, style=style)
                for row in rows
                if row.shortcut
            ),
            default=0,
        ),
    )


@dataclass(frozen=True)
class MenuRowResult:
    activated: bool
    box: Box
    native_rect: tuple[float, float, float, float]
    label_rect: Rect
    tail_x: float
    visible: bool
    minimum_width: float
    observation: dict


def menu_row_size(ui, row, *, part="menu_item", columns=None):
    """Measure the same glyph, text, padding and margins used by the painter."""
    style, scale = ui.component_style(part), ui.s
    choice = part == "select_option"
    padding = native_edges(style, "padding", scale, (12, 0, 12, 0) if choice else (6, 0, 6, 0))
    margins = native_edges(style, "margin", scale)
    font, icon = style.get("fontSize", BODY_SIZE), native_unit(style.get("iconSize", 16) * scale)
    gap = native_unit(style.get("gap", 12 if choice else 6) * scale)
    tail_gap = (
        style.get("checkGap", 12)
        if choice
        else ui.metric("menu_item_content", "columnGap", ui.metric("menu_item_content", "gap", 12))
    ) * scale
    tail_gap = native_unit(tail_gap)
    label_width = ui.measure(row.label, font, font_resolved=True, style=style)
    shortcut_width = (
        ui.measure(row.shortcut, font, translate=False, font_resolved=True, style=style)
        if row.shortcut
        else 0
    )
    has_check = columns.check if columns is not None else choice or row.checkable or row.checked
    has_icon = columns.icon if columns is not None else bool(row.icon)
    has_arrow = columns.arrow if columns is not None else bool(row.submenu)
    if columns is not None:
        label_width, shortcut_width = columns.label_width, columns.shortcut_width
    width = (icon + (tail_gap if choice else gap) if has_check else 0) + (
        icon + gap if has_icon else 0
    )
    width += label_width + shortcut_width + ((gap if choice else tail_gap) if shortcut_width else 0)
    width += icon + gap if has_arrow else 0
    width += padding[0] + padding[2]
    if style.get("widthMode", "auto") == "fixed":
        width = native_unit(style.get("width", width / scale) * scale)
    height = native_unit(
        max(
            (style.get("height", 30) if choice else style.get("height", DESIGN.menu_item_height))
            * scale,
            max(style.line_height(font * 1.4) * scale, icon) + padding[1] + padding[3],
        )
    )
    if getattr(style, "origins", {}).get("radius") == "parent_corner":
        # Keep an authored popup's derived child curve drawable at its actual
        # radius instead of letting a short row silently clamp the paint.
        height = max(height, ceil(2 * row_radius(style, scale) + 2))
    return width + margins[0] + margins[2], height + margins[1] + margins[3]


def paint_menu_row(
    ui,
    row,
    key,
    *,
    content_right=None,
    open_submenu=None,
    minimum_height=0,
    part="menu_item",
    columns=None,
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
    minimum_width, height = menu_row_size(ui, row, part=part, columns=columns)
    margins = native_edges(style, "margin", scale)
    height = max(height, native_unit(minimum_height) + margins[1] + margins[3])
    box = native_box(
        ui.component_box(
            part,
            origin.x,
            origin.y,
            width,
            height,
            key=key,
            padding=((12 if choice else 6) * scale, 0, (12 if choice else 6) * scale, 0),
            observe=False,
            minimum_height=minimum_height,
        )
    )
    font = style.get("fontSize", BODY_SIZE)
    window = imgui.get_current_context().current_window
    draw = imgui.get_window_draw_list()
    imgui.set_cursor_screen_pos((box.frame.x, box.frame.y))
    imgui.push_font(ui.resolve_font(style), font * scale)
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
        hint_hovered = imgui.is_item_hovered(
            imgui.HoveredFlags_.delay_normal | imgui.HoveredFlags_.allow_when_disabled
        )
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
            tint = ui.ink(tint, style)
            if ui.design.runtime.spec["native"]["g3"]:
                smoothing = ui.design.runtime.spec["native"]["g3_smoothing"]
        radius = row_radius(style, scale, box.frame.width, box.frame.height)
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
    glyph_extent = style.get("iconSize", 16) * scale
    extent = native_unit(glyph_extent)
    gap = native_unit(style.get("gap", 12 if choice else 6) * scale)
    tail_gap = (
        style.get("checkGap", 12)
        if choice
        else ui.metric("menu_item_content", "columnGap", ui.metric("menu_item_content", "gap", 12))
    ) * scale
    tail_gap = native_unit(tail_gap)
    shortcut_width = (
        ui.measure(row.shortcut, font, translate=False, font_resolved=True, style=style)
        if row.shortcut
        else 0
    )
    label_measure = ui.measure(row.label, font, font_resolved=True, style=style)
    has_check = columns.check if columns is not None else choice or row.checkable or row.checked
    has_icon = columns.icon if columns is not None else bool(row.icon)
    has_arrow = columns.arrow if columns is not None else bool(row.submenu)
    check_gap = tail_gap if choice else gap
    budget = (extent + check_gap if has_check else 0) + ((extent + gap) if has_icon else 0)
    budget += extent + gap if has_arrow else 0
    factor = min(1, box.content.width / budget) if budget else 1
    extent, gap, check_gap = extent * factor, gap * factor, check_gap * factor
    glyph_extent *= factor
    check_x = box.content.x
    icon_x = check_x + (extent + check_gap if has_check else 0)
    prefix = (extent + check_gap if has_check else 0) + (extent + gap if has_icon else 0)
    tail_width = extent + gap if has_arrow else 0
    shortcut_end = box.content.right - tail_width
    shortcut_column = columns.shortcut_width if columns is not None else shortcut_width
    shortcut_column = min(shortcut_column, max(0, shortcut_end - box.content.x - prefix))
    text_gap = gap if choice else tail_gap
    label_width = max(
        0,
        box.content.width
        - prefix
        - shortcut_column
        - (text_gap if shortcut_column else 0)
        - tail_width,
    )
    epsilon = native_epsilon(box.content.x, box.content.right, box.frame.y, box.frame.bottom)
    label_width = text_budget(
        label_width, columns.label_width if columns is not None else label_measure, epsilon
    )
    label_rect = Rect(box.content.x + prefix, box.content.y, label_width, box.content.height)
    check_rect = Rect(check_x, box.content.y, extent if has_check else 0, box.content.height)
    icon_rect = Rect(icon_x, box.content.y, extent if has_icon else 0, box.content.height)
    shortcut_slot = Rect(
        shortcut_end - shortcut_column, box.content.y, shortcut_column, box.content.height
    )
    shortcut_column = text_budget(shortcut_column, shortcut_width, epsilon)
    shortcut_extent = min(shortcut_width, shortcut_column)
    shortcut_rect = Rect(
        shortcut_end - shortcut_extent, box.content.y, shortcut_extent, box.content.height
    )
    arrow_rect = Rect(
        box.content.right - (extent if has_arrow else 0),
        box.content.y,
        extent if has_arrow else 0,
        box.content.height,
    )
    ui.record_component_box(
        part,
        box,
        key=key,
        draw=draw,
        background=visible and highlighted,
        visible=visible,
        hovered=hovered,
        backgroundRect=list(box.frame.tuple()) if visible and highlighted else None,
        backgroundRadius=row_radius(style, scale, box.frame.width, box.frame.height),
        backgroundCornerFlags="all",
        labelRect=list(label_rect.tuple()),
        checkRect=list(check_rect.tuple()),
        iconSlotRect=list(icon_rect.tuple()),
        iconRect=list(icon_rect.tuple()) if row.icon else None,
        shortcutRect=list(shortcut_rect.tuple()) if row.shortcut else None,
        shortcutSlotRect=list(shortcut_slot.tuple()),
        shortcutAnchor=shortcut_end,
        arrowSlotRect=list(arrow_rect.tuple()),
        arrowRect=list(arrow_rect.tuple()) if row.submenu else None,
        label=row.label,
        labelMeasuredWidth=label_measure,
        shortcutMeasuredWidth=shortcut_width,
        nativeEpsilon=epsilon,
        clippedLabel=label_measure > label_rect.width,
        clippedShortcut=shortcut_width > shortcut_rect.width,
        columnCapabilities={"check": has_check, "icon": has_icon, "arrow": has_arrow},
        columnWidths={"label": label_rect.width, "shortcut": shortcut_column},
        fieldRegions={
            "gap": ([icon_rect.right, box.content.y, gap, box.content.height] if has_icon else [])
            if choice
            else [
                *(
                    [[check_rect.right, box.content.y, check_gap, box.content.height]]
                    if has_check
                    else []
                ),
                *([[icon_rect.right, box.content.y, gap, box.content.height]] if has_icon else []),
                *(
                    [[arrow_rect.x - gap, box.content.y, gap, box.content.height]]
                    if has_arrow
                    else []
                ),
            ],
            **(
                {"checkGap": [check_rect.right, box.content.y, check_gap, box.content.height]}
                if choice
                else {}
            ),
        },
        **metadata,
    )
    tail_x = shortcut_rect.x if row.shortcut else arrow_rect.x
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
                clippedLabel=label_measure > label_rect.width,
                shortcutRect=list(shortcut_rect.tuple()) if row.shortcut else None,
                fieldRegions={
                    field: [label_rect.right, box.content.y, text_gap, box.content.height]
                    if shortcut_column
                    else []
                    for field in ("gap", "columnGap")
                },
                **metadata,
            )
        if row.icon:
            glyph = glyph_rect(icon_rect, glyph_extent)
            ui.icon(
                row.icon,
                glyph.x,
                glyph.y,
                glyph_extent / scale,
                MUTED if enabled else DISABLED,
                draw=draw,
                style=style,
            )
        ui.component_text(part, label_rect, row.label, foreground, draw=draw)
        if row.shortcut:
            ui.component_text(
                part,
                shortcut_rect,
                row.shortcut,
                DIM if enabled else DISABLED,
                draw=draw,
                translate=False,
                align="end",
            )
        if row.checked:
            glyph = glyph_rect(check_rect, glyph_extent)
            ui.icon(
                "check",
                glyph.x,
                glyph.y,
                glyph_extent / scale,
                ACCENT if enabled else DISABLED,
                draw=draw,
                style=style,
            )
        if row.submenu:
            glyph = glyph_rect(arrow_rect, glyph_extent)
            ui.icon(
                "right",
                glyph.x,
                glyph.y,
                glyph_extent / scale,
                DIM if enabled else DISABLED,
                draw=draw,
                style=style,
            )
    if hint_hovered:
        if row.reason and not enabled:
            ui.tooltip(row.reason)
        elif label_measure > label_rect.width or shortcut_width > shortcut_rect.width:
            ui.tooltip(
                ui.translate(row.label) + ("  " + row.shortcut if row.shortcut else ""),
                translate=False,
            )
    return MenuRowResult(
        bool(activated),
        box,
        (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y),
        label_rect,
        tail_x,
        visible,
        minimum_width,
        {
            "key": key,
            "label": row.label,
            "icon": row.icon,
            "enabled": enabled,
            "checked": bool(row.checked),
            "checkable": row.checkable,
            "rect": [lo.x, lo.y, hi.x - lo.x, hi.y - lo.y],
            "frame_rect": list(box.frame.tuple()),
            "label_rect": list(label_rect.tuple()),
            "tail_x": tail_x,
            "visible": visible,
            "hovered": hovered,
            "clippedLabel": label_measure > label_rect.width,
            "clippedShortcut": shortcut_width > shortcut_rect.width,
            "labelMeasuredWidth": label_measure,
            "shortcutMeasuredWidth": shortcut_width,
            "shortcutRect": list(shortcut_rect.tuple()) if row.shortcut else None,
            "shortcutSlotRect": list(shortcut_slot.tuple()),
            "checkRect": list(check_rect.tuple()),
            "iconSlotRect": list(icon_rect.tuple()),
            "arrowSlotRect": list(arrow_rect.tuple()),
            "backgroundRadius": row_radius(style, scale, box.frame.width, box.frame.height),
            "nativeEpsilon": epsilon,
            "backgroundRect": list(box.frame.tuple()) if visible and highlighted else None,
        },
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
    gap = native_unit(gap * ui.s)
    if current is not None and current[0] == imgui.get_current_context().current_window.id_:
        current[1].append((outer, gap))
    y = outer.bottom + gap
    imgui.set_cursor_screen_pos((outer.x, outer.bottom))
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
    width = outer.width
    if ui.component_style(part).get("widthMode", "auto") == "auto":
        width = max(width, result.minimum_width)
    imgui.dummy((width, 0))
    imgui.pop_style_var()
    imgui.set_cursor_screen_pos((outer.x, y))
