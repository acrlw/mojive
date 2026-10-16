"""Camera-preview surface measurement and source UV mapping."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreviewSurface:
    panel_width: float
    panel_height: float
    image_rect: tuple[float, float, float, float]


def preview_origin(viewport, panel_size, *, desired=None, margin=0):
    """Clamp display placement while retaining an optional user anchor unchanged."""
    x, y, width, height = viewport
    panel_width, panel_height = panel_size
    right, bottom = x + width - panel_width - margin, y + height - panel_height - margin
    px, py = (right, bottom) if desired is None else desired
    return min(max(px, x + margin), right), min(max(py, y + margin), bottom)


def preview_surface(
    available_width,
    available_height,
    *,
    maximum_width,
    header_height,
    image_aspect=16 / 9,
    image_height=None,
    insets=(0, 0, 0, 0),
    row_gap=0,
):
    """Measure one physical preview without allocating or modifying a render target."""
    left, top, right, bottom = insets
    body_budget = available_height - header_height - row_gap - top - bottom
    if body_budget <= 0 or available_width <= left + right:
        return None
    width = min(maximum_width, available_width)
    if image_height is None:
        width = min(width, body_budget * image_aspect + left + right)
        body_height = max(0, width - left - right) / max(image_aspect, 1e-9)
    else:
        body_height = min(body_budget, max(0, image_height))
    body_width = width - left - right
    if body_width <= 0 or body_height <= 0:
        return None
    return PreviewSurface(
        width,
        header_height + row_gap + top + body_height + bottom,
        (left, header_height + row_gap + top, body_width, body_height),
    )


def preview_image_fit(frame, aspect, *, fit="contain", flip_y=False):
    """Return a destination rectangle and UV crop for contain, cover or stretch."""
    x, y, width, height = frame
    if fit not in ("contain", "cover", "stretch"):
        raise ValueError(f"Unknown preview fit: {fit!r}")
    u0 = v0 = 0.0
    u1 = v1 = 1.0
    frame_aspect = width / max(height, 1e-9)
    if fit == "contain":
        if frame_aspect > aspect:
            fitted_width = height * aspect
            x += (width - fitted_width) / 2
            width = fitted_width
        else:
            fitted_height = width / max(aspect, 1e-9)
            y += (height - fitted_height) / 2
            height = fitted_height
    elif fit == "cover":
        if frame_aspect > aspect:
            shown = aspect / frame_aspect
            v0, v1 = (1 - shown) / 2, (1 + shown) / 2
        else:
            shown = frame_aspect / max(aspect, 1e-9)
            u0, u1 = (1 - shown) / 2, (1 + shown) / 2
    if flip_y:
        v0, v1 = 1 - v0, 1 - v1
    return (x, y, width, height), (u0, v0), (u1, v1)
