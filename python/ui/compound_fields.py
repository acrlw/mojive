"""Shared drawing primitives for compound ImGui fields."""

from __future__ import annotations

from contextlib import contextmanager

from imgui_bundle import imgui


def draw_joined_field_frame(
    draw_list,
    badge_lo,
    badge_hi,
    field_lo,
    field_hi,
    *,
    badge_color,
    field_color,
    rounding: float,
    badge_opacity: float = 1.0,
    field_opacity: float = 1.0,
) -> None:
    """Draw one outer rounded frame with a square badge/field join.

    The field surface is the complete outer silhouette.  The badge is then
    painted over its own side with only the external corners rounded.  Keeping a
    continuous surface below the color boundary prevents either antialiased
    edge from exposing the panel background at fractional framebuffer scales.
    """

    def packed(color, opacity: float) -> int:
        alpha = min(1.0, max(0.0, float(opacity))) * float(color[3])
        return imgui.color_convert_float4_to_u32(
            imgui.ImVec4(float(color[0]), float(color[1]), float(color[2]), alpha)
        )

    y0 = min(float(badge_lo.y), float(field_lo.y))
    y1 = max(float(badge_hi.y), float(field_hi.y))
    radius = min(max(0.0, float(rounding)), max(0.0, (y1 - y0) * 0.5))
    outer_lo = imgui.ImVec2(min(float(badge_lo.x), float(field_lo.x)), y0)
    outer_hi = imgui.ImVec2(max(float(badge_hi.x), float(field_hi.x)), y1)
    draw_list.add_rect_filled(
        outer_lo,
        outer_hi,
        packed(field_color, field_opacity),
        radius,
        imgui.ImDrawFlags_.round_corners_all.value,
    )
    draw_list.add_rect_filled(
        imgui.ImVec2(float(badge_lo.x), y0),
        imgui.ImVec2(float(badge_hi.x), y1),
        packed(badge_color, badge_opacity),
        radius,
        (
            imgui.ImDrawFlags_.round_corners_left
            if badge_lo.x < field_lo.x
            else imgui.ImDrawFlags_.round_corners_right
        ).value,
    )


@contextmanager
def borderless_numeric_input():
    """Keep a compound field's editor inside its shared outer silhouette."""
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)
    imgui.push_style_color(imgui.Col_.nav_cursor, (0, 0, 0, 0))
    try:
        yield
    finally:
        imgui.pop_style_color()
        imgui.pop_style_var()


def draw_focus_frame(lo, hi, *, rounding: float, corners=None, item_id=None, color=None) -> None:
    """Inset keyboard focus inside the same rounded/square contour as its field."""
    context = imgui.get_current_context()
    item_id = imgui.get_item_id() if item_id is None else item_id
    if not context.nav_cursor_visible or context.nav_id != item_id:
        return
    if corners is None:
        corners = imgui.ImDrawFlags_.round_corners_all
    thickness = max(1.0, imgui.get_font_size() / 13.0)
    inset = thickness * 0.5 + 0.5
    radius = max(0.0, min(rounding, (hi.y - lo.y) * 0.5) - inset)
    dl = imgui.get_window_draw_list()
    dl.path_rect((lo.x + inset, lo.y + inset), (hi.x - inset, hi.y - inset), radius, corners)
    packed = imgui.get_color_u32(imgui.Col_.nav_cursor if color is None else imgui.ImVec4(*color))
    dl.path_stroke(packed, thickness, imgui.ImDrawFlags_.closed)
