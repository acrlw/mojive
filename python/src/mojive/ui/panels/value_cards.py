"""Compact, virtualizable property rows with native input and shared rail geometry."""

from __future__ import annotations

import math
from contextlib import contextmanager

from imgui_bundle import imgui

from ..compound_fields import borderless_numeric_input, draw_joined_field_frame
from ..draw2d import ImguiDraw2D, fit_text
from ..pointer_bindings import PointerAction
from ..viewport_widgets import draw_reset_glyph
from . import copyable_name_item, pointer_pressed, value_edit


def interval_text(bounds: tuple[float, float], unit: str = "") -> str:
    """Format real limits without implying that an unbounded control has limits."""
    return f"[{bounds[0]:.3g}, {bounds[1]:.3g}]" + (f" {unit}" if unit else "")


def _stacked(width: float, scale: float) -> bool:
    return width < 164.0 * scale


def card_height(scale: float, width: float) -> float:
    """Use a single padded row except when the rail and entry cannot fit."""
    inner = width - 2 * min(8 * scale, width * 0.08)
    if inner >= 240 * scale:
        return imgui.get_frame_height() + 8 * scale
    height = imgui.get_text_line_height() + imgui.get_frame_height() + 16 * scale
    if _stacked(inner, scale):
        height += imgui.get_frame_height() + imgui.get_style().item_spacing.y
    return height


def card_heading(ctx, origin, width, name, detail):
    """Reserve a smaller trailing range and align it to the name's text baseline."""
    draw = ImguiDraw2D()
    font_size = imgui.get_font_size()
    baseline = imgui.get_font_baked().ascent
    gap = 10.0 * ctx.style_scale
    imgui.push_font(None, font_size * 0.85)
    detail_width = min(imgui.calc_text_size(detail).x, max(0, (width - gap) * 0.48))
    shown_detail = fit_text(draw, detail, detail_width, middle=True)
    detail_y = origin[1] + baseline - imgui.get_font_baked().ascent
    imgui.pop_font()
    name_width = max(1.0, width - detail_width - gap)
    shown_name = fit_text(draw, name, name_width)
    draw.text(origin, ctx.theme.text, shown_name)
    detail_x = origin[0] + min(draw.text_size(shown_name)[0], name_width) + gap
    imgui.push_font(None, font_size * 0.85)
    draw.text((detail_x, detail_y), ctx.theme.text_disabled, shown_detail)
    imgui.pop_font()
    return shown_name != name or shown_detail != detail


@contextmanager
def value_card(ctx, item_id: str, name: str, detail: str, *, selected: bool = False):
    """Yield heading selection/focus actions and a responsive value control slot.

    Height depends only on panel width so ListClipper can skip rows reliably.
    """
    origin = imgui.get_cursor_screen_pos()
    width = max(1.0, float(imgui.get_content_region_avail().x))
    scale = ctx.style_scale
    padding = min(8 * scale, width * 0.08)
    line = imgui.get_text_line_height()
    height = card_height(scale, width)
    inline = width - 2 * padding >= 240 * scale
    inner = width - 2 * padding
    gap = 8 * scale
    label_width = min(150 * scale, inner * 0.26) if inline else inner
    control_x = origin.x + padding + (label_width + gap if inline else 0.0)
    control_y = origin.y + (4 * scale if inline else line + 12 * scale)
    draw = ImguiDraw2D()
    hi = (origin.x + width, origin.y + height)
    if selected:
        draw.rect_filled(
            (origin.x, origin.y),
            hi,
            ctx.theme.bg_header,
            rounding=imgui.get_style().frame_rounding,
        )
    imgui.set_cursor_screen_pos((origin.x + padding, origin.y + 4 * scale))
    imgui.invisible_button(
        item_id, imgui.ImVec2(label_width, imgui.get_frame_height() if inline else line)
    )
    clicked = imgui.is_item_hovered() and pointer_pressed(ctx, PointerAction.SELECT)
    focused = imgui.is_item_hovered() and pointer_pressed(ctx, PointerAction.PANEL_FOCUS)
    copyable_name_item(ctx, name, label_width)
    text_y = origin.y + 4 * scale + (imgui.get_style().frame_padding.y if inline else 0)
    imgui.push_clip_rect(
        (origin.x + padding, origin.y), (origin.x + padding + label_width, hi[1]), True
    )
    shown = fit_text(draw, name, label_width)
    draw.text((origin.x + padding, text_y), ctx.theme.text, shown)
    imgui.pop_clip_rect()
    if shown != name or detail:
        imgui.set_item_tooltip(name + ("\n" + detail if detail else ""))
    imgui.set_cursor_screen_pos((control_x, control_y))
    imgui.set_next_item_width(max(1.0, origin.x + width - padding - control_x))
    try:
        yield clicked, focused
    finally:
        imgui.set_cursor_screen_pos(origin)
        imgui.dummy(imgui.ImVec2(width, height))


def value_rail(
    ctx,
    label,
    value,
    bounds,
    *,
    initial,
    fmt="%.4f",
    show_reset=True,
    unit="",
    angular_degrees=False,
    toggle_unit=None,
):
    """Thin bounded rail plus numeric entry; unbounded values use only numeric entry.

    ImGui owns dragging, keyboard navigation and text editing. Pointer shortcuts
    share the existing mapping; the fill always starts at the minimum bound.
    """
    factor = 180.0 / math.pi if unit == "rad" and angular_degrees else 1.0
    value *= factor
    initial = None if initial is None else initial * factor
    bounds = None if bounds is None else (bounds[0] * factor, bounds[1] * factor)
    shown_unit = "deg" if unit == "rad" and angular_degrees else unit
    width = max(1.0, imgui.calc_item_width())
    scale = ctx.style_scale
    style = imgui.get_style()

    def color(value):
        return (*value[:3], value[3] * style.alpha)

    gap = style.item_spacing.x
    reset_width = imgui.get_frame_height() if show_reset else 0.0
    reset_space = reset_width + gap if show_reset else 0.0
    stacked = _stacked(width, scale)
    unit_width = 34.0 * scale if unit else 0.0
    number_width = min(68.0 * scale + unit_width, max(1.0, width - reset_space))
    result = None
    imgui.push_id(label)
    if bounds is not None:
        lo, hi = bounds
        rail_width = width if stacked else max(1.0, width - number_width - reset_space - gap)
        for slot in (
            imgui.Col_.frame_bg,
            imgui.Col_.frame_bg_hovered,
            imgui.Col_.frame_bg_active,
            imgui.Col_.slider_grab,
            imgui.Col_.slider_grab_active,
        ):
            imgui.push_style_color(slot, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_var(imgui.StyleVar_.grab_min_size, 10 * scale)
        imgui.set_next_item_width(rail_width)
        changed, current = imgui.slider_float(label, value, lo, hi, "", imgui.SliderFlags_.no_input)
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        imgui.pop_style_var()
        imgui.pop_style_color(5)
        result = value_edit(
            label, value, changed, current, initial=initial, fmt=fmt, bindings=ctx.input_bindings
        )
        value = result.value
        draw = ImguiDraw2D()
        # Match SliderBehavior's two-point padding and half-grab endpoints.
        radius = min(5 * scale, max(0, (b.x - a.x - 4) * 0.5))
        left, right, cy = a.x + 2 + radius, b.x - 2 - radius, (a.y + b.y) * 0.5
        fraction = min(1.0, max(0.0, (value - lo) / (hi - lo)))
        position = left + (right - left) * fraction
        hovered, pressed = imgui.is_item_hovered(), imgui.is_item_active()
        draw_value_rail(
            draw,
            (left, cy),
            (right, cy),
            position,
            radius,
            ctx.theme,
            scale,
            hovered=hovered,
            pressed=pressed,
            alpha=style.alpha,
        )
        if not stacked:
            imgui.same_line()
    entry_width = max(1.0, width - reset_space) if stacked or bounds is None else number_width
    numeric = _value_entry(
        ctx,
        label,
        value,
        bounds,
        initial,
        fmt,
        entry_width,
        shown_unit,
        toggle_unit if unit == "rad" else None,
    )
    activated = numeric.activated or (result is not None and result.activated)
    if result is None or numeric.changed:
        result = numeric
    result.activated = activated
    if show_reset:
        imgui.same_line()
        imgui.begin_disabled(initial is None)
        if imgui.button(f"{label}-restore", imgui.ImVec2(reset_width, reset_width)):
            result.changed, result.value = True, float(initial)
        imgui.set_item_tooltip(ctx.tr("Restore initial value"))
        a, b = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw_reset_glyph(
            ImguiDraw2D(),
            ((a.x + b.x) * 0.5, (a.y + b.y) * 0.5),
            color(ctx.theme.text),
            0.65 * scale,
        )
        imgui.end_disabled()
    imgui.pop_id()
    result.value /= factor
    return result


def _value_entry(ctx, label, value, bounds, initial, fmt, width, unit, toggle_unit):
    """Join a native numeric editor to a trailing unit badge without a seam."""
    style = imgui.get_style()
    unit_width = min(34.0 * ctx.style_scale, width * 0.45) if unit else 0
    draw_list = imgui.get_window_draw_list()
    splitter = imgui.ImDrawListSplitter()
    if unit:
        splitter.split(draw_list, 2)
        splitter.set_current_channel(draw_list, 1)
        for slot in (imgui.Col_.frame_bg, imgui.Col_.frame_bg_hovered, imgui.Col_.frame_bg_active):
            imgui.push_style_color(slot, (0, 0, 0, 0))
    imgui.set_next_item_width(max(1.0, width - unit_width))
    lo, hi = bounds if bounds is not None else (0.0, 0.0)
    with borderless_numeric_input():
        changed, current = imgui.drag_float(
            f"{label}-value",
            value,
            0.01,
            lo,
            hi,
            fmt,
            imgui.SliderFlags_.always_clamp if bounds is not None else 0,
        )
    field_lo, field_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    hovered, active = imgui.is_item_hovered(), imgui.is_item_active()
    edit = value_edit(
        label, value, changed, current, initial=initial, fmt=fmt, bindings=ctx.input_bindings
    )
    if unit:
        imgui.pop_style_color(3)
        imgui.same_line(0, 0)
        for slot in (imgui.Col_.button, imgui.Col_.button_hovered, imgui.Col_.button_active):
            imgui.push_style_color(slot, (0, 0, 0, 0))
        imgui.push_style_var(imgui.StyleVar_.button_text_align, (0.5, 0.5))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, (0, style.frame_padding.y))
        imgui.push_style_color(
            imgui.Col_.text, ctx.theme.bg_popup if toggle_unit else ctx.theme.text
        )
        imgui.begin_disabled(toggle_unit is None)
        if (
            imgui.button(f"{unit}##{label}-unit", (unit_width, imgui.get_frame_height()))
            and toggle_unit
        ):
            toggle_unit()
        imgui.end_disabled()
        badge_lo, badge_hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        badge_hover, badge_active = imgui.is_item_hovered(), imgui.is_item_active()
        imgui.pop_style_color(4)
        imgui.pop_style_var(2)
        if toggle_unit:
            imgui.set_item_tooltip(
                ctx.tr("Switch to degrees" if unit == "rad" else "Switch to radians")
            )
        splitter.set_current_channel(draw_list, 0)
        draw_joined_field_frame(
            draw_list,
            badge_lo,
            badge_hi,
            field_lo,
            field_hi,
            badge_color=(
                ctx.theme.primary_dim
                if badge_active
                else ctx.theme.primary_bright
                if badge_hover
                else ctx.theme.primary
            )
            if toggle_unit
            else ctx.theme.bg_child,
            field_color=ctx.theme.bg_frame_active
            if active
            else ctx.theme.bg_frame_hovered
            if hovered
            else ctx.theme.bg_frame,
            rounding=style.frame_rounding,
            badge_opacity=style.alpha,
            field_opacity=style.alpha,
        )
        splitter.merge(draw_list)
    return edit


def draw_value_rail(
    draw, left, right, position, radius, theme, scale, *, hovered=False, pressed=False, alpha=1.0
):
    """Paint a left-to-value rail with explicit normal, hover and pressed feedback."""

    def color(value):
        return (*value[:3], value[3] * alpha)

    fill = theme.primary_bright if pressed else theme.primary if hovered else theme.primary_dim
    knob = theme.text if pressed else theme.primary_bright if hovered else theme.primary
    track = (
        theme.bg_frame_active if pressed else theme.bg_frame_hovered if hovered else theme.bg_frame
    )
    draw.line(left, right, color(track), 3 * scale, cap="round")
    if position > left[0]:
        draw.line(left, (position, left[1]), color(fill), 3 * scale, cap="round")
    draw.circle_filled((position, left[1]), radius, color(knob), segments=24)
