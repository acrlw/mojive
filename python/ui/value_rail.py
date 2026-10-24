"""Shared rail painting for numeric controls and viewport precision overlays."""

from mojive.geometry2d.curves import arc_ribbon_mesh


def _draw_rail_segment(draw, left, right, ink, width, *, round_start, round_end):
    if right[0] <= left[0]:
        return
    vertices, indices = arc_ribbon_mesh(
        (left, right),
        None,
        None,
        width,
        round_start=round_start,
        round_end=round_end,
        smoothing=0.0,
    )
    draw.indexed_fill(vertices, indices, ink, outline=vertices)


def draw_value_rail(
    draw,
    left,
    right,
    position,
    radius,
    theme,
    scale,
    *,
    hovered=False,
    pressed=False,
    alpha=1.0,
    disabled=False,
    track_size=3,
    limit_colors=None,
    reference_position=None,
):
    """Paint a subdued rail and separate round handle, with opaque disabled colors."""

    def color(value):
        opacity = value[3] * alpha
        if disabled:
            # Resolve disabled dimming once against the panel surface. The
            # opaque handle then covers the rail instead of blending it twice.
            rgb = (
                b + (c - b) * opacity for c, b in zip(value[:3], theme.bg_window[:3], strict=True)
            )
            return (*rgb, 1.0)
        return (*value[:3], opacity)

    fill = theme.primary_bright if pressed else theme.primary if hovered else theme.primary_dim
    knob = theme.text if pressed else theme.primary_bright if hovered else theme.primary
    track = (
        theme.bg_frame_active if pressed else theme.bg_frame_hovered if hovered else theme.bg_frame
    )
    # Limit markers cover the track caps; the thumb remains the topmost layer.
    # Insets affect painting only, never the full-range numeric mapping.
    inset = track_size * scale * 0.5 if limit_colors is not None else 0.0
    track_left = (left[0] + inset, left[1])
    track_right = (right[0] - inset, right[1])
    if reference_position is None:
        fill_end = min(position, track_right[0])
        draw.line(track_left, track_right, color(track), track_size * scale, cap="round")
        if fill_end > track_left[0]:
            draw.line(track_left, (fill_end, left[1]), color(fill), track_size * scale, cap="round")
    else:
        delta_left = min(track_right[0], max(track_left[0], min(reference_position, position)))
        delta_right = max(track_left[0], min(track_right[0], max(reference_position, position)))
        # Adjacent colors share one stroke contour and AA policy. Layering a
        # native flat-capped line over a rounded fill exposes the brighter AA edge.
        for start, end, ink in (
            (track_left[0], delta_left, fill),
            (delta_left, delta_right, theme.primary_dim),
            (delta_right, track_right[0], track),
        ):
            _draw_rail_segment(
                draw,
                (start, left[1]),
                (end, left[1]),
                color(ink),
                track_size * scale,
                round_start=start == track_left[0],
                round_end=end == track_right[0],
            )
        # The origin is a rail tick, shorter than the thumb so zero movement
        # leaves only the current handle visible. It never floats off the rail.
        origin_half = radius - 2.0 * scale
        draw.line(
            (reference_position, left[1] - origin_half),
            (reference_position, left[1] + origin_half),
            color(theme.primary_dim),
            2.0 * scale,
            cap="round",
        )
    if limit_colors is not None:
        tick_half = radius + 2.0 * scale
        for point, ink in zip((left, right), limit_colors, strict=True):
            draw.line(
                (point[0], point[1] - tick_half),
                (point[0], point[1] + tick_half),
                color(ink),
                3.0 * scale,
                cap="round",
            )
    draw.circle_filled((position, left[1]), radius, color(knob), segments=24)
