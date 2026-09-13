"""Reusable layout, property rows and parameter sliders for UI review."""

from __future__ import annotations

from imgui_bundle import imgui

from mojive.ui.icons import (
    ICON_MAX_PADDING,
    ICON_MAX_STROKE,
    ICON_MIN_CLEARANCE,
    ICON_MIN_STROKE,
    icon_family,
)
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.panels import button_row_layout
from mojive.ui.text_layout import text_line_y
from mojive.ui.viewport_widgets import draw_projection_label

from ..ui_capsule_geometry import capsule_layout
from .fixtures import _ICON_REVIEW_SIZES, CONCEPT_THEME, GEOMETRY_CANVAS_SIZE
from .state import ProbeState


def _probe_window_size(width: int, height: int, ui_scale: float) -> tuple[int, int]:
    """Keep the whole concept canvas visible while inspecting extreme UI scales.

    The probe is the design reference, so a capture that clips its right-hand panels
    misrepresents the design. Growing the window with the UI scale keeps every specimen
    laid out at its real proportions instead of hiding it behind a scrollbar. Both factors
    stay at 1.0 for the default scale, which already fits the base window.
    """

    scale = float(ui_scale)
    # The geometry canvas and its right-hand experiment controls together need about 1.25x
    # the base width, and the tallest canvas about 0.95x the base height.
    growth = scale * 1.25 if scale > 1.0 else 1.0
    vertical = max(1.0, scale * 0.95)
    return round(width * growth), round(height * vertical)


def _virtual_canvas_size(
    available,
    scale: float,
    logical_size: tuple[float, float],
) -> tuple[float, float]:
    """Preserve component proportions and let extreme scales scroll."""

    return (
        max(float(available.x), logical_size[0] * scale),
        max(float(available.y), logical_size[1] * scale),
    )


def _wrapped_tabs(
    tabs: tuple[tuple[str, str, str], ...],
    active: str,
    available: float,
    *,
    initial: str | None = None,
    gap: float | None = None,
) -> str:
    """Draw one tab row that wraps instead of clipping at large UI scales.

    Native tab bars extend past the panel edge once the labels grow, which hides the
    rightmost entries and overlaps the neighbouring content. The probe keeps the tab
    grammar but reflows it, matching how the production panels wrap their own rows.
    """

    style = imgui.get_style()
    spacing = style.item_spacing.x if gap is None else float(gap)
    width = max(1.0, float(available))
    default_focus = bool(initial) and active != initial
    if default_focus:
        active = str(initial)
    padding = 10.0
    # Every label needs its own ImGui ID, since the probe shows overlapping tab sets.
    entries = tuple(
        (label, f"{scope}-{index}-{slug}") for index, (label, scope, slug) in enumerate(tabs)
    )
    widths = tuple(
        min(width, imgui.calc_text_size(label).x + 2.0 * style.frame_padding.x + padding)
        for label, _ in entries
    )
    inline = button_row_layout(widths, width, spacing)
    height = style.frame_padding.y * 2.0 + imgui.get_text_line_height()
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    imgui.begin_group()
    for index, (label, identifier) in enumerate(entries):
        if index and inline[index]:
            imgui.same_line(0.0, spacing)
        selected = label == active
        pressed = imgui.invisible_button(
            f"##{identifier}", imgui.ImVec2(widths[index], height), imgui.ButtonFlags_.none
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        hovered = imgui.is_item_hovered()
        # A passive label keeps one click meaning "select", never "toggle off".
        color = CONCEPT_THEME.text if selected or hovered else CONCEPT_THEME.text_disabled
        text = imgui.calc_text_size(label)
        draw.text(
            (
                (lo.x + hi.x - text.x) * 0.5,
                (lo.y + hi.y) * 0.5 - text.y * 0.5 - style.frame_padding.y * 0.5,
            ),
            color,
            label,
        )
        if selected:
            draw.line(
                (lo.x, hi.y - 1.0),
                (hi.x, hi.y - 1.0),
                CONCEPT_THEME.primary,
                max(1.0, style.frame_padding.y),
            )
        if default_focus and selected:
            imgui.set_item_default_focus()
        if pressed:
            active = label
    imgui.end_group()
    return active


def _flags(*values) -> int:
    result = 0
    for value in values:
        result |= int(value.value if hasattr(value, "value") else value)
    return result


def _table_next_control_row() -> None:
    """Start a property row with one explicit framed-control height."""

    style = imgui.get_style()
    row_height = imgui.get_frame_height() + style.cell_padding.y * 2.0
    imgui.table_next_row(imgui.TableRowFlags_.none.value, row_height)


def _table_text(value: str, *, disabled: bool = False) -> None:
    """Center text on the same visual axis as a framed control in its row."""

    cursor = imgui.get_cursor_screen_pos()
    text_height = imgui.calc_text_size(value).y
    offset_y = max(0.0, (imgui.get_frame_height() - text_height) * 0.5)
    imgui.set_cursor_screen_pos(imgui.ImVec2(cursor.x, cursor.y + offset_y))
    if disabled:
        imgui.text_disabled(value)
    else:
        imgui.text(value)


def _property_label(label: str) -> None:
    _table_next_control_row()
    imgui.table_next_column()
    width = imgui.calc_text_size(label).x
    available = imgui.get_content_region_avail().x
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, available - width))
    _table_text(label, disabled=True)
    imgui.table_next_column()
    imgui.set_next_item_width(-1.0)


def _probe_checkbox(label: str, value: bool) -> tuple[bool, bool]:
    """Draw the accepted neutral checkbox instead of ImGui's blue checked fill."""

    style = imgui.get_style()
    visible_label = label.split("##", 1)[0]
    box_size = float(imgui.get_frame_height())
    label_width = imgui.calc_text_size(visible_label).x if visible_label else 0.0
    total_width = box_size
    if visible_label:
        total_width += float(style.item_inner_spacing.x) + float(label_width)

    clicked = imgui.invisible_button(label, imgui.ImVec2(total_width, box_size))
    hovered = imgui.is_item_hovered()
    active = imgui.is_item_active()
    if clicked:
        value = not value

    lo = imgui.get_item_rect_min()
    draw = ImguiDraw2D(imgui.get_window_draw_list())
    opacity = float(style.alpha)

    def faded(color):
        return (*color[:3], color[3] * opacity)

    background = (
        CONCEPT_THEME.bg_frame_active
        if value or active
        else CONCEPT_THEME.bg_frame_hovered
        if hovered
        else CONCEPT_THEME.bg_frame
    )
    draw.rect_filled(
        (lo.x, lo.y),
        (lo.x + box_size, lo.y + box_size),
        faded(background),
        rounding=min(float(style.frame_rounding), box_size * 0.25),
    )
    draw.rect(
        (lo.x, lo.y),
        (lo.x + box_size, lo.y + box_size),
        faded(CONCEPT_THEME.border),
        1.0,
        rounding=min(float(style.frame_rounding), box_size * 0.25),
    )
    if value:
        check = (
            (lo.x + box_size * 0.22, lo.y + box_size * 0.53),
            (lo.x + box_size * 0.43, lo.y + box_size * 0.73),
            (lo.x + box_size * 0.79, lo.y + box_size * 0.29),
        )
        draw.polyline(
            check,
            faded(CONCEPT_THEME.primary_bright),
            max(1.5, box_size * 0.12),
        )
    if visible_label:
        text_height = imgui.calc_text_size(visible_label).y
        text_color = CONCEPT_THEME.text_disabled if opacity < 0.999 else CONCEPT_THEME.text
        draw.text(
            (
                lo.x + box_size + float(style.item_inner_spacing.x),
                lo.y + (box_size - text_height) * 0.5,
            ),
            faded(text_color),
            visible_label,
        )
    return clicked, value


def _draw_segmented(
    item_id: str,
    labels: tuple[str, ...],
    selected: int,
    *,
    width: float = 82.0,
    icons: tuple[str, ...] | None = None,
    icon_drawer=None,
) -> int:
    result = selected
    draw = ImguiDraw2D()
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(1.0, 0.0))
    for index, label in enumerate(labels):
        if index:
            imgui.same_line()
        if index == selected:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*CONCEPT_THEME.bg_frame_active))
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*CONCEPT_THEME.primary_bright))
        icon = icons[index] if icons is not None and index < len(icons) else ""
        button_label = f"##{item_id}-{index}" if icon else f"{label}##{item_id}-{index}"
        if imgui.button(button_label, imgui.ImVec2(width, 0.0)):
            result = index
        item_min = imgui.get_item_rect_min()
        item_max = imgui.get_item_rect_max()
        if index == selected:
            imgui.pop_style_color(2)
        if icon:
            color = CONCEPT_THEME.primary_bright if index == selected else CONCEPT_THEME.text
            if icon_drawer is not None:
                height = item_max.y - item_min.y
                icon_size = height * 0.64
                center = (item_min.x + height * 0.5, (item_min.y + item_max.y) * 0.5)
                icon_drawer(draw, center, icon_size, icon, color)
                draw.text(
                    (item_min.x + height, text_line_y(draw, center[1])),
                    color,
                    label,
                )
            else:
                glyph_scale = max(0.65, imgui.get_frame_height() / 24.0)
                draw_projection_label(
                    draw,
                    (item_min.x, item_min.y),
                    (item_max.x, item_max.y),
                    color,
                    glyph_scale,
                    icon,
                    label,
                )
    imgui.pop_style_var()
    return result


def _dimension_line(
    draw: ImguiDraw2D,
    start,
    end,
    label: str,
    scale: float,
    *,
    vertical: bool = False,
) -> None:
    color = (*CONCEPT_THEME.text_disabled[:3], 0.9)
    draw.line(start, end, color, 1.0 * scale)
    if vertical:
        draw.line(
            (start[0] - 5.0 * scale, start[1]),
            (start[0] + 5.0 * scale, start[1]),
            color,
            1.0 * scale,
        )
        draw.line(
            (end[0] - 5.0 * scale, end[1]),
            (end[0] + 5.0 * scale, end[1]),
            color,
            1.0 * scale,
        )
        draw.text(
            (start[0] + 9.0 * scale, (start[1] + end[1]) * 0.5 - 7.0 * scale),
            color,
            label,
        )
    else:
        draw.line(
            (start[0], start[1] - 5.0 * scale),
            (start[0], start[1] + 5.0 * scale),
            color,
            1.0 * scale,
        )
        draw.line(
            (end[0], end[1] - 5.0 * scale),
            (end[0], end[1] + 5.0 * scale),
            color,
            1.0 * scale,
        )
        width, _ = draw.text_size(label)
        draw.text(
            ((start[0] + end[0] - width) * 0.5, start[1] + 8.0 * scale),
            color,
            label,
        )


def _even_slider(item_id: str, value: int, minimum: int, maximum: int) -> int:
    changed, candidate = imgui.slider_int(item_id, value, minimum, maximum)
    if not changed:
        return value
    snapped = int(round(candidate / 2.0) * 2)
    return max(minimum, min(maximum, snapped))


def _stepped_slider(
    item_id: str,
    value: float,
    minimum: float,
    maximum: float,
    display_format: str,
    *,
    step: float = 0.05,
) -> tuple[bool, float]:
    """Keep interactive geometry controls on stable, cacheable values."""

    changed, candidate = imgui.slider_float(
        item_id,
        value,
        minimum,
        maximum,
        display_format,
        imgui.SliderFlags_.always_clamp.value,
    )
    if not changed:
        return False, value
    snapped = minimum + round((candidate - minimum) / step) * step
    return True, round(max(minimum, min(maximum, snapped)), 6)


def _deferred_icon_group_slider(
    item_id: str,
    state: ProbeState,
    group: str,
    *,
    kind: str,
) -> float:
    """Edit a whole group without rebuilding every glyph during the drag."""

    if kind == "padding":
        drafts = state.icon_padding_draft_by_group
        value = drafts.get(group, state.icon_padding_for(group))
        minimum, maximum, display_format = ICON_MIN_CLEARANCE, ICON_MAX_PADDING, "%.2f u"
    else:
        drafts = state.icon_stroke_draft_by_group
        value = drafts.get(group, state.icon_stroke_for(group))
        minimum, maximum, display_format = ICON_MIN_STROKE, ICON_MAX_STROKE, "%.2f u"
    changed, candidate = _stepped_slider(
        item_id,
        value,
        minimum,
        maximum,
        display_format,
    )
    if changed:
        drafts[group] = candidate
    if imgui.is_item_deactivated_after_edit():
        candidate = drafts.pop(group, candidate)
        if kind == "padding":
            state.set_icon_padding_for(group, candidate)
        else:
            state.set_icon_stroke_for(group, candidate)
    return candidate


def _icon_library_canvas_size(family: str) -> tuple[float, float]:
    if family in {"Overview", "UI context"}:
        return GEOMETRY_CANVAS_SIZE
    if family == "Capsules":
        return GEOMETRY_CANVAS_SIZE[0], 1260.0
    rows = len(icon_family(family))
    required_height = 260.0 + (max(_ICON_REVIEW_SIZES) + 20.0) * rows
    if family == "Keyframe follow":
        required_height += 110.0
    return GEOMETRY_CANVAS_SIZE[0], max(GEOMETRY_CANVAS_SIZE[1], required_height)


def _geometry_canvas_size(active_tab: str, state: ProbeState) -> tuple[float, float]:
    """Reserve the logical extent required by zoomed geometry specimens."""

    width, height = GEOMETRY_CANVAS_SIZE
    if active_tab == "Playback":
        shell_radius = state.overlay_icon_radius + 2.0 * state.overlay_radial_step
        inspection = state.construction_playback_scale
        _centers, length = capsule_layout(6, (3, 4), state)
        width = max(width, 54.0 + length * inspection + 54.0 + 360.0 + 24.0)
        construction_bottom = 80.0 + 4.0 * shell_radius * inspection + 62.0
        height = max(height, construction_bottom + 1010.0)
    elif active_tab == "Tools":
        _centers, length = capsule_layout(5, (3,), state)
        height = max(height, 120.0 + length * state.construction_tool_scale)
    return width, height
