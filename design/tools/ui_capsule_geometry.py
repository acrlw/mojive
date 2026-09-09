"""Measured spacing and curve-phase inspection for the existing G3 capsule."""

import math
from functools import lru_cache

import numpy as np

from mojive.curves2d import (
    CURVE_TOLERANCE,
    _cap_geometry,
    _fit_cap_smoothing,
    smooth_capsule_points,
)
from mojive.ui.theme import THEME
from mojive.ui.viewport_widgets import CAPSULE_SMOOTHING as CAPSULE_SMOOTHING
from mojive.ui.viewport_widgets import CAPSULE_SURFACE_ALPHA

CAPSULE_OUTLINE_LABELS = ("Neutral gray", "Soft white")


def capsule_outline_color(geometry):
    return THEME.border if geometry.capsule_outline == "Neutral gray" else (*THEME.text[:3], 0.25)


def end_radii(path, center, samples=181):
    """Intersect outward semicircle rays with the actual sampled shell."""
    points = np.asarray(path)
    edges = np.roll(points, -1, axis=0) - points
    offsets = points - center
    angles = np.linspace(math.pi / 2, 3 * math.pi / 2, samples)
    rays = np.column_stack((np.cos(angles), np.sin(angles)))

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    denominator = cross(rays[:, None], edges[None])
    denominator = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
    distance = cross(offsets[None], edges[None]) / denominator
    along = cross(offsets[None], rays[:, None]) / denominator
    return np.where((distance > 0) & (along >= 0) & (along <= 1), distance, np.inf).min(axis=1)


@lru_cache(maxsize=128)
def optical_end_padding(radius, smoothing):
    """Equalize mean end clearance and straight-side clearance without changing the cap."""
    if smoothing == 0:
        return radius
    path = smooth_capsule_points(0, 0, radius * 8, radius * 2, smoothing, tolerance=radius / 26000)
    lo, hi = radius, radius * 2
    for _ in range(24):
        middle = (lo + hi) / 2
        if end_radii(path, (middle, radius)).mean() < radius:
            lo = middle
        else:
            hi = middle
    return (lo + hi) / 2


def end_padding(geometry):
    radius = geometry.overlay_icon_radius + 2 * geometry.overlay_radial_step
    return (
        optical_end_padding(radius, geometry.capsule_smoothing)
        if geometry.optical_capsule_spacing
        else radius
    )


@lru_cache(maxsize=128)
def spacing_metrics(radius, state_radius, smoothing, optical):
    padding = optical_end_padding(radius, smoothing) if optical else radius
    path = smooth_capsule_points(0, 0, radius * 8, radius * 2, smoothing, tolerance=radius / 26000)
    gaps = end_radii(path, (padding, radius), samples=721) - state_radius - 0.7
    return float(gaps.min()), float(gaps.mean()), float(gaps.max()), radius - state_radius - 0.7


def capsule_layout(count, breaks, geometry):
    padding = end_padding(geometry)
    centers = tuple(
        padding
        + i * geometry.overlay_center_step
        + sum(b <= i for b in breaks) * geometry.tool_group_gap
        for i in range(count)
    )
    return centers, centers[-1] + padding


@lru_cache(maxsize=128)
def g3_capsule_spans(width, height, smoothing):
    """Identify curvature ramps using the actual cap's fitted profile and tangent joins."""
    long, short = max(width, height), min(width, height)
    q = _fit_cap_smoothing(long / (2 * short), smoothing)
    if q == 0:
        return ()
    cap, depth = _cap_geometry(short, q, CURVE_TOLERANCE)
    edges = np.diff(cap, axis=0)
    tangents = np.arctan2(edges[:, 1], edges[:, 0])
    ramp = (tangents < q * math.pi / 2) | (tangents > math.pi - q * math.pi / 2)
    starts = np.flatnonzero(ramp & ~np.r_[False, ramp[:-1]])
    ends = np.flatnonzero(ramp & ~np.r_[ramp[1:], False]) + 1
    spans = []
    for points in (cap + np.array((long - depth, short / 2)), -cap + np.array((depth, short / 2))):
        if height > width:
            points = np.column_stack((short - points[:, 1], points[:, 0]))
        spans.extend(
            tuple(map(tuple, points[a : b + 1].tolist())) for a, b in zip(starts, ends, strict=True)
        )
    return tuple(spans)


def draw_capsule_shell(draw, x, y, width, height, scale, geometry):
    shell = smooth_capsule_points(x, y, width, height, geometry.capsule_smoothing)
    draw.convex_fill(shell, (*THEME.bg_child[:3], CAPSULE_SURFACE_ALPHA))
    draw.polyline(shell, capsule_outline_color(geometry), 1.4 * scale, closed=True)
    if geometry.highlight_g3:
        for span in g3_capsule_spans(width, height, geometry.capsule_smoothing):
            draw.polyline(tuple((x + px, y + py) for px, py in span), THEME.warning, 1.4 * scale)
