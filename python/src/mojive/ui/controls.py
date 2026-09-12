"""Reusable ImGui controls, independent of panels and application state.

Controls keep native item IDs, input and keyboard navigation. Callers own domain
state and translations; geometry and interaction belong here.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache

from imgui_bundle import imgui

from mojive.drawing.curves import smooth_capsule_points

from .compound_fields import draw_focus_frame
from .draw2d import Draw2D, ImguiDraw2D, fit_text
from .input_bindings import DEFAULT_INPUT_BINDINGS
from .pointer_bindings import PointerAction
from .theme import THEME, Theme
from .viewport_widgets import draw_projection_label

IconLabelDrawer = Callable[
    [
        Draw2D,
        tuple[float, float],
        tuple[float, float],
        tuple[float, float, float, float],
        float,
        str,
        str,
    ],
    None,
]


def _production_control_icon(draw, center, size: float, kind: str, color) -> None:
    # Import lazily because the icon library reuses viewport's low-level rotate,
    # snap, and mouse geometry while controls are imported by viewport panels.
    from .icons import draw_control_icon

    draw_control_icon(draw, center, size, kind, color)


def action_menu_popup(str_id, items, *, on_item=None):
    """Draw equally sized action buttons in one popup; None separates action groups."""
    if not imgui.is_popup_open(str_id):
        return None
    scale = imgui.get_font_size() / 14.0
    width = max(
        (
            110.0 * scale,
            *(imgui.calc_text_size(item[1]).x + 24.0 * scale for item in items if item is not None),
        )
    )
    for variable, value in (
        (imgui.StyleVar_.window_padding, (10 * scale, 10 * scale)),
        (imgui.StyleVar_.item_spacing, (0, 6 * scale)),
        (imgui.StyleVar_.frame_padding, (12 * scale, 6 * scale)),
        (imgui.StyleVar_.popup_border_size, scale),
        (imgui.StyleVar_.popup_rounding, 8 * scale),
        (imgui.StyleVar_.frame_rounding, 6 * scale),
    ):
        imgui.push_style_var(variable, value)
    imgui.set_next_window_content_size((width, 0))
    action = None
    if imgui.begin_popup(str_id):
        for item in items:
            if item is None:
                imgui.separator()
                continue
            key, label, enabled = item
            imgui.begin_disabled(not enabled)
            clicked = imgui.button(f"{label}##{key}", (width, 0))
            if on_item is not None:
                on_item(key)
            if clicked:
                action = key
                imgui.close_current_popup()
            imgui.end_disabled()
        imgui.end_popup()
    imgui.pop_style_var(6)
    return action


def search_input(
    str_id: str,
    value: str,
    *,
    hint: str = "",
    search_tooltip: str = "Search",
    clear_tooltip: str = "Clear search",
    icon_drawer=None,
) -> tuple[bool, str]:
    """Draw a live filter with a leading search glyph and a trailing clear button."""

    icon_drawer = icon_drawer or _production_control_icon
    style = imgui.get_style()
    width, height = float(imgui.calc_item_width()), float(imgui.get_frame_height())
    start = imgui.get_cursor_screen_pos()
    lo = (start.x, start.y)
    hi = (start.x + width, start.y + height)
    padding = float(style.frame_padding.x) * 0.5
    slot_width = min(height, width * 0.25)
    input_width = max(1.0, width - 2.0 * (slot_width + padding))
    input_id = imgui.get_id(str_id)
    draw_list = imgui.get_window_draw_list()
    hovered = imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(lo, hi)
    active = imgui.internal.get_active_id() == input_id
    background = (
        imgui.Col_.frame_bg_active
        if active
        else (imgui.Col_.frame_bg_hovered if hovered else imgui.Col_.frame_bg)
    )
    draw_list.add_rect_filled(lo, hi, imgui.get_color_u32(background), height * 0.5)
    if style.frame_border_size > 0.0:
        draw_list.add_rect(
            lo,
            hi,
            imgui.get_color_u32(imgui.Col_.border),
            height * 0.5,
            thickness=style.frame_border_size,
        )

    # The native editor owns a separate text region. Long text and the caret
    # cannot pass beneath either glyph, and the whole group owns one frame.
    imgui.begin_group()
    imgui.set_cursor_screen_pos((lo[0] + slot_width + padding, lo[1]))
    imgui.set_next_item_width(input_width)
    for color in (
        imgui.Col_.frame_bg,
        imgui.Col_.frame_bg_hovered,
        imgui.Col_.frame_bg_active,
        imgui.Col_.nav_cursor,
    ):
        imgui.push_style_color(color, imgui.ImVec4(0, 0, 0, 0))
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)
    if hint:
        changed, value = imgui.input_text_with_hint(str_id, hint, value)
    else:
        changed, value = imgui.input_text(str_id, value)
    imgui.pop_style_var()
    imgui.pop_style_color(4)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, height * 0.5)
    imgui.internal.render_nav_cursor(imgui.internal.ImRect(lo, hi), input_id)
    imgui.pop_style_var()

    draw = ImguiDraw2D()
    color = imgui.get_style_color_vec4(imgui.Col_.text_disabled)
    identifier = str_id.partition("##")[2] or str_id
    center_y = (lo[1] + hi[1]) * 0.5
    search_x = lo[0] + padding + slot_width * 0.5
    focus_requested = False
    icon_drawer(draw, (search_x, center_y), height * 0.72, "search", color)
    if imgui.is_window_hovered() and imgui.is_mouse_hovering_rect(
        lo, (lo[0] + slot_width + padding, hi[1])
    ):
        imgui.set_tooltip(search_tooltip)
        focus_requested = imgui.is_mouse_clicked(imgui.MouseButton_.left)
    if value:
        center_x = hi[0] - padding - slot_width * 0.5
        imgui.set_cursor_screen_pos((center_x - slot_width * 0.5, lo[1]))
        clicked = clear_button(
            f"##clear_{identifier}",
            (slot_width, height),
            clear_tooltip,
            icon_drawer=icon_drawer,
        )
        if clicked:
            value, changed = "", True
            focus_requested = True
    if focus_requested:
        imgui.internal.activate_item_by_id(input_id)
        imgui.get_current_context().nav_next_activate_flags = (
            imgui.internal.ActivateFlags_.prefer_input
        )
    imgui.set_cursor_screen_pos(lo)
    imgui.dummy((width, height))
    imgui.end_group()
    return changed, value


def sort_order_glyph(
    rect: tuple[float, float, float, float],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Return a compact list-and-arrow sorting glyph inside ``rect``."""

    x0, y0, x1, y1 = rect
    width, height = x1 - x0, y1 - y0
    left = x0 + width * 0.24
    arrow_x = x0 + width * 0.72
    ys = (y0 + height * 0.32, y0 + height * 0.50, y0 + height * 0.68)
    bars = tuple(
        ((left, y), (left + width * length, y))
        for y, length in zip(ys, (0.28, 0.21, 0.14), strict=True)
    )
    arrow_top = y0 + height * 0.28
    arrow_bottom = y0 + height * 0.70
    wing = width * 0.09
    return (
        *bars,
        ((arrow_x, arrow_top), (arrow_x, arrow_bottom)),
        ((arrow_x - wing, arrow_bottom - wing), (arrow_x, arrow_bottom)),
        ((arrow_x + wing, arrow_bottom - wing), (arrow_x, arrow_bottom)),
    )


def sort_order_tooltip(
    by_name: bool,
    state_order: str,
    translate=lambda value: value,
) -> str:
    """Describe only the active order; the button itself communicates switching."""

    current = translate("Name") if by_name else state_order
    return f"{translate('Order')}: {current}"


def sort_order_button(
    str_id: str,
    by_name: bool,
    *,
    state_order: str,
    translate=lambda value: value,
    bindings=None,
    icon_drawer=None,
) -> tuple[bool, bool]:
    """Draw a compact state/name order toggle beside a search field."""

    icon_drawer = icon_drawer or _production_control_icon
    size = imgui.get_frame_height()
    left_clicked = imgui.button(f"##sort_order_{str_id}", imgui.ImVec2(size, 0.0))
    hovered = imgui.is_item_hovered()
    bindings = bindings or DEFAULT_INPUT_BINDINGS
    alternate = hovered and bindings.pointer_match(
        PointerAction.SORT_ALTERNATE, bindings.pointer_frame(), press=True
    )
    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    color_value = imgui.get_style_color_vec4(imgui.Col_.check_mark if by_name else imgui.Col_.text)
    color = color_value
    draw = ImguiDraw2D()
    icon_drawer(
        draw,
        ((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5),
        size * 0.72,
        "sort",
        color,
    )
    imgui.set_item_tooltip(sort_order_tooltip(by_name, state_order, translate))
    changed = bool(left_clicked or alternate)
    return changed, (not by_name if changed else by_name)


def searchable_ordered_list_header(
    str_id: str,
    value: str,
    by_name: bool,
    *,
    hint: str,
    search_tooltip: str,
    clear_tooltip: str,
    state_order: str,
    translate=lambda text: text,
    bindings=None,
    icon_drawer=None,
) -> tuple[bool, str, bool, bool]:
    """Draw one responsive search field followed by its list-order button."""

    spacing = imgui.get_style().item_spacing.x
    button_width = imgui.get_frame_height()
    available = imgui.get_content_region_avail().x
    imgui.set_next_item_width(max(1.0, available - spacing - button_width))
    search_changed, value = search_input(
        str_id,
        value,
        hint=hint,
        search_tooltip=search_tooltip,
        clear_tooltip=clear_tooltip,
        icon_drawer=icon_drawer,
    )
    imgui.same_line()
    sort_changed, by_name = sort_order_button(
        str_id,
        by_name,
        state_order=state_order,
        translate=translate,
        bindings=bindings,
        icon_drawer=icon_drawer,
    )
    return search_changed, value, sort_changed, by_name


def padded_selectable(label: str, selected: bool = False, flags=0, size=None):
    """Keep native selection/navigation with padded text and equal row bounds."""
    style = imgui.get_style()
    width = float(size.x) if size is not None and size.x > 0 else imgui.get_content_region_avail().x
    height = float(size.y) if size is not None and size.y > 0 else imgui.get_frame_height()
    padding = min(float(style.frame_padding.x), max(0.0, (width - 1.0) * 0.5))
    origin = imgui.get_cursor_screen_pos()
    imgui.set_cursor_screen_pos((origin.x + padding, origin.y))
    # Native Selectable expands its background by half ItemSpacing. Supply
    # the horizontal inset explicitly and avoid clipped half-rows at child edges.
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(padding * 2.0, 0.0))
    imgui.push_style_var(
        imgui.StyleVar_.selectable_text_align, imgui.ImVec2(style.selectable_text_align.x, 0.5)
    )
    result = imgui.selectable(
        label, selected, flags, imgui.ImVec2(max(1.0, width - padding * 2.0), height)
    )
    imgui.pop_style_var(2)
    return result


def themed_checkbox(
    label: str,
    value: bool,
    theme: Theme = THEME,
) -> tuple[bool, bool]:
    """Draw a neutral checkbox without the platform/default blue selected fill."""

    visible_label = label.partition("##")[0]
    style = imgui.get_style()
    size = imgui.get_frame_height()
    gap = float(style.item_inner_spacing.x)
    text_width = max(1.0, imgui.get_content_region_avail().x - size - gap)
    text_size = imgui.calc_text_size(visible_label, wrap_width=text_width)
    width = size + gap + min(text_size.x, text_width) if visible_label else size
    height = max(size, text_size.y + style.frame_padding.y * 2.0) if visible_label else size
    clicked = imgui.invisible_button(
        label, imgui.ImVec2(width, height), imgui.ButtonFlags_.enable_nav
    )
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    item_lo, item_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    lo = imgui.ImVec2(item_lo.x, item_lo.y + (height - size) * 0.5)
    hi = imgui.ImVec2(lo.x + size, lo.y + size)
    surface = (
        theme.bg_frame_active
        if active or value
        else theme.bg_frame_hovered
        if hovered
        else theme.bg_frame
    )
    draw = ImguiDraw2D()
    rounding = min(imgui.get_style().frame_rounding, size * 0.24)
    alpha = style.alpha
    draw.rect_filled(
        (lo.x, lo.y), (hi.x, hi.y), (*surface[:3], surface[3] * alpha), rounding=rounding
    )
    draw.rect(
        (lo.x, lo.y),
        (hi.x, hi.y),
        (*theme.border[:3], theme.border[3] * alpha),
        1.0,
        rounding=rounding,
    )
    if value:
        draw.polyline(
            (
                (lo.x + size * 0.22, lo.y + size * 0.52),
                (lo.x + size * 0.43, lo.y + size * 0.72),
                (lo.x + size * 0.80, lo.y + size * 0.29),
            ),
            (*theme.primary_bright[:3], theme.primary_bright[3] * alpha),
            max(1.5, size * 0.11),
        )
    if visible_label:
        imgui.get_window_draw_list().add_text(
            imgui.get_font(),
            imgui.get_font_size(),
            (lo.x + size + gap, (item_lo.y + item_hi.y - text_size.y) * 0.5),
            imgui.get_color_u32(imgui.Col_.text),
            visible_label,
            wrap_width=text_width,
        )
    return clicked, not value if clicked else value


@lru_cache(maxsize=256)
def _segment_width(font, font_size, labels, icons, padding, glyph_scale, icon_only_width):
    """Measure stable labels once per font, language and UI scale."""
    return max(
        icon_only_width
        if icon and icon_only_width
        else imgui.calc_text_size(label).x + padding + (20.0 * glyph_scale if icon else 0.0)
        for label, icon in zip(labels, icons, strict=True)
    )


def segmented_control_width(
    labels: tuple[str, ...], *, icons: tuple[str, ...] | None = None, show_labels: bool = True
) -> float:
    """Return the horizontal width using the control's actual font and padding."""
    if not labels:
        return 0.0
    icons = tuple(icons[i] if icons and i < len(icons) else "" for i in range(len(labels)))
    return len(labels) * _segment_width(
        imgui.get_font(),
        imgui.get_font_size(),
        labels,
        icons,
        2 * imgui.get_style().frame_padding.x,
        max(0.65, imgui.get_frame_height() / 24.0),
        0.0 if show_labels else imgui.get_frame_height(),
    )


def segmented_control(
    str_id: str,
    labels: tuple[str, ...],
    selected: int,
    *,
    width: float = 0.0,
    theme: Theme = THEME,
    icons: tuple[str, ...] | None = None,
    icon_label_drawer: IconLabelDrawer | None = None,
    show_labels: bool = True,
) -> int:
    """Choose one of N equal segments in one rounded frame.

    Only outer corners are rounded. Options form a contiguous vertical group
    when their labels cannot fit horizontally. Stable native button IDs retain
    keyboard navigation; measurement is cached independently of hover/selection.
    Hidden icon captions become per-item tooltips and use square segments.
    """
    if not labels:
        return 0
    style = imgui.get_style()
    available = max(1.0, float(width if width > 0 else imgui.get_content_region_avail().x))
    glyph_scale = max(0.65, imgui.get_frame_height() / 24.0)
    icons = tuple(icons[i] if icons and i < len(icons) else "" for i in range(len(labels)))
    inline = segmented_control_width(labels, icons=icons, show_labels=show_labels) <= available
    selected = min(max(0, int(selected)), len(labels) - 1)
    result = selected
    draw = ImguiDraw2D()
    dl = imgui.get_window_draw_list()
    splitter = imgui.ImDrawListSplitter()
    splitter.split(dl, 2)
    origin = imgui.get_cursor_screen_pos()
    height = imgui.get_frame_height()
    dl.add_rect_filled(
        origin,
        (origin.x + available, origin.y + height * (1 if inline else len(labels))),
        imgui.get_color_u32(imgui.ImVec4(*theme.bg_frame)),
        style.frame_rounding,
    )
    imgui.begin_group()
    imgui.push_style_var(imgui.StyleVar_.item_spacing, (0, 0))
    imgui.push_style_var(imgui.StyleVar_.button_text_align, (0.5, 0.5))
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)
    for slot in (imgui.Col_.button, imgui.Col_.button_hovered, imgui.Col_.button_active):
        imgui.push_style_color(slot, (0, 0, 0, 0))
    for index, (label, icon) in enumerate(zip(labels, icons, strict=True)):
        # Derive each edge from the group origin so rounding cannot accumulate.
        start = round(available * index / len(labels)) if inline else 0
        end = (
            round(available * (index + 1) / len(labels))
            if inline and index < len(labels) - 1
            else available
        )
        imgui.set_cursor_screen_pos(
            (origin.x + start, origin.y + (0 if inline else index * height))
        )
        item_width = end - start
        is_selected = index == selected
        color = theme.primary_bright if is_selected else theme.text
        imgui.push_style_color(imgui.Col_.text, color)
        splitter.set_current_channel(dl, 1)
        button_label = f"##{str_id}-{index}" if icon else f"{label}##{str_id}-{index}"
        imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
        clicked = imgui.button(button_label, (item_width, 0))
        imgui.pop_style_color(2)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        fill = (
            theme.bg_frame_active
            if is_selected or imgui.is_item_active()
            else theme.bg_frame_hovered
            if imgui.is_item_hovered()
            else theme.bg_frame
        )
        if len(labels) == 1:
            corners = imgui.ImDrawFlags_.round_corners_all
        elif index == 0:
            corners = (
                imgui.ImDrawFlags_.round_corners_left
                if inline
                else imgui.ImDrawFlags_.round_corners_top
            )
        elif index == len(labels) - 1:
            corners = (
                imgui.ImDrawFlags_.round_corners_right
                if inline
                else imgui.ImDrawFlags_.round_corners_bottom
            )
        else:
            corners = imgui.ImDrawFlags_.round_corners_none
        splitter.set_current_channel(dl, 0)
        if fill != theme.bg_frame:
            dl.add_rect_filled(
                lo,
                hi,
                imgui.get_color_u32(imgui.ImVec4(*fill)),
                style.frame_rounding,
                corners.value,
            )
        splitter.set_current_channel(dl, 1)
        draw_focus_frame(lo, hi, rounding=style.frame_rounding, corners=corners)
        if icon:
            splitter.set_current_channel(dl, 1)
            dl.push_clip_rect(lo, hi, True)
            (icon_label_drawer or draw_projection_label)(
                draw,
                (lo.x, lo.y),
                (hi.x, hi.y),
                (*color[:3], color[3] * style.alpha),
                glyph_scale,
                icon,
                label if show_labels else "",
            )
            dl.pop_clip_rect()
            if not show_labels:
                imgui.set_item_tooltip(label)
        if clicked:
            result = index
    splitter.merge(dl)
    imgui.pop_style_color(3)
    imgui.pop_style_var(3)
    imgui.end_group()
    return result


def button_width(label: str, minimum: float = 0.0) -> float:
    text = label.partition("##")[0]
    padding = 2.0 * imgui.get_style().frame_padding.x
    return max(float(minimum), imgui.calc_text_size(text).x + padding)


def button_row_layout(
    widths: tuple[float, ...], available: float, spacing: float
) -> tuple[bool, ...]:
    same_line: list[bool] = []
    used = 0.0
    for width in widths:
        inline = bool(same_line) and used + spacing + width <= available
        if inline:
            used += spacing + width
        else:
            used = width
        same_line.append(inline)
    return tuple(same_line)


def clear_button(
    str_id: str,
    size: tuple[float, float],
    tooltip: str = "",
    *,
    icon_drawer=None,
) -> bool:
    """Draw a keyboard-accessible, round-stroke clear action without a frame."""
    icon_drawer = icon_drawer or _production_control_icon
    clicked = imgui.invisible_button(str_id, size, imgui.ButtonFlags_.enable_nav)
    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    hovered = imgui.is_item_hovered()
    color = imgui.get_style_color_vec4(imgui.Col_.text if hovered else imgui.Col_.text_disabled)
    color = (color.x, color.y, color.z, color.w * imgui.get_style().alpha)
    x, y = (lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5
    draw = ImguiDraw2D()
    icon_drawer(draw, (x, y), size[1] * 0.72, "clear", color)
    if hovered:
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand)
        if tooltip:
            imgui.set_tooltip(tooltip)
    return clicked


@contextmanager
def clearable_combo(
    str_id: str,
    preview: str,
    *,
    has_value: bool,
    clear_tooltip: str = "Clear",
    tooltip: str = "",
):
    """Yield (open, cleared) for a native combo with an inset clear action.

    Submit the clear item first so it owns input in its slot, before the larger
    combo can claim the pointer. Draw it last, above the native frame, and reserve
    its space in the preview so long names never paint beneath the cross.
    """
    width, height = max(1.0, imgui.calc_item_width()), imgui.get_frame_height()
    start = imgui.get_cursor_screen_pos()
    padding = imgui.get_style().frame_padding.x
    slot = min(height, max(0.0, (width - height) * 0.5)) if has_value else 0.0
    draw_list = imgui.get_window_draw_list()
    splitter = imgui.ImDrawListSplitter()
    splitter.split(draw_list, 2)
    imgui.begin_group()
    cleared = clear_hovered = False
    if slot > 0:
        splitter.set_current_channel(draw_list, 1)
        imgui.set_cursor_screen_pos((start.x + width - height - slot, start.y))
        identifier = str_id.partition("##")[2] or str_id
        cleared = clear_button(f"##clear_{identifier}", (slot, height), clear_tooltip)
        clear_hovered = imgui.is_item_hovered()
        imgui.set_cursor_screen_pos(start)
    splitter.set_current_channel(draw_list, 0)
    shown = fit_text(ImguiDraw2D(), preview, max(1.0, width - height - slot - 2 * padding))
    imgui.set_next_item_width(width)
    opened = imgui.begin_combo(str_id, shown)
    splitter.merge(draw_list)
    if not opened and tooltip and not clear_hovered:
        imgui.set_item_tooltip(tooltip)
    try:
        yield opened, cleared
    finally:
        if opened:
            imgui.end_combo()
        imgui.end_group()


@lru_cache(maxsize=64)
def _property_label_width(font, font_size, labels):
    return max(imgui.calc_text_size(label).x for label in labels)


def begin_property_table(str_id: str, *, labels: tuple[str, ...] = ()) -> bool:
    """Begin the shared left-label/right-control property grid."""
    flags = imgui.TableFlags_.sizing_stretch_prop | imgui.TableFlags_.no_pad_outer_x
    if not imgui.begin_table(str_id, 2, flags):
        return False
    if labels:
        width = min(
            _property_label_width(imgui.get_font(), imgui.get_font_size(), labels),
            imgui.get_content_region_avail().x * 0.4,
        )
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed, width)
    else:
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_stretch, 0.26)
    imgui.table_setup_column("control", imgui.TableColumnFlags_.width_stretch, 0.74)
    return True


def property_row(label: str, *, tooltip: str = "", wrap: bool = False) -> None:
    """Advance a property grid, fitting its label inside the left column."""
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.align_text_to_frame_padding()
    if wrap:
        imgui.push_text_wrap_pos(0.0)
        imgui.text_disabled(label)
        imgui.pop_text_wrap_pos()
    else:
        shown = fit_text(ImguiDraw2D(), label, imgui.get_content_region_avail().x)
        imgui.text_disabled(shown)
        if not tooltip and shown != label:
            tooltip = label
    if tooltip:
        imgui.set_item_tooltip(tooltip)
    imgui.table_next_column()
    imgui.set_next_item_width(-1.0)


@lru_cache(maxsize=64)
def _pill_outline(width: float, height: float):
    return smooth_capsule_points(0, 0, width, height, smoothing=0)


def pill_label(label: str, *, width: float, height: float, background, color) -> None:
    """Draw a passive, centered capsule badge with cached local geometry."""
    pos = imgui.get_cursor_screen_pos()
    draw = ImguiDraw2D()
    alpha = imgui.get_style().alpha
    draw.fringed_concave_fill(
        _pill_outline(width, height),
        (*background[:3], background[3] * alpha),
        origin=(pos.x, pos.y),
    )
    text = imgui.calc_text_size(label)
    draw.text(
        (pos.x + (width - text.x) * 0.5, pos.y + (height - text.y) * 0.5),
        (*color[:3], color[3] * alpha),
        label,
    )
    imgui.dummy((width, height))
