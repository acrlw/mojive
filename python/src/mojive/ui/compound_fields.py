"""Shared drawing primitives for compound ImGui fields."""

from __future__ import annotations

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
    painted over its left side with only its left corners rounded.  Keeping a
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
    outer_lo = imgui.ImVec2(float(badge_lo.x), y0)
    outer_hi = imgui.ImVec2(float(field_hi.x), y1)
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
        imgui.ImDrawFlags_.round_corners_left.value,
    )
