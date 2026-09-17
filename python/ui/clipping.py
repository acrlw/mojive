"""Map UI clip rectangles into framebuffer coordinates for every backend."""

import math


def scissor_rect_for_target(
    clip_rect: tuple[float, float, float, float],
    display_pos: tuple[float, float],
    framebuffer_scale: tuple[float, float],
    draw_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Map an ImGui clip rectangle into the current render target.

    A native resize may land after ImGui generated draw data but before the
    command encoder is finished. In that frame, draw-space and target-space
    dimensions differ; keep every scissor rectangle inside the attachment
    that is actually being encoded.
    """

    draw_w, draw_h = draw_size
    target_w, target_h = target_size
    if min(draw_w, draw_h, target_w, target_h) <= 0:
        return None
    clip_x0 = (float(clip_rect[0]) - float(display_pos[0])) * float(framebuffer_scale[0])
    clip_y0 = (float(clip_rect[1]) - float(display_pos[1])) * float(framebuffer_scale[1])
    clip_x1 = (float(clip_rect[2]) - float(display_pos[0])) * float(framebuffer_scale[0])
    clip_y1 = (float(clip_rect[3]) - float(display_pos[1])) * float(framebuffer_scale[1])
    target_scale_x = target_w / draw_w
    target_scale_y = target_h / draw_h
    x0 = max(0, min(target_w, math.floor(clip_x0 * target_scale_x)))
    y0 = max(0, min(target_h, math.floor(clip_y0 * target_scale_y)))
    x1 = max(0, min(target_w, math.ceil(clip_x1 * target_scale_x)))
    y1 = max(0, min(target_h, math.ceil(clip_y1 * target_scale_y)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0
