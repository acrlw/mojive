"""Implicit hollow-origin drag links shared by UI meshes and GPU field references.

This module owns field evaluation and cached strip tessellation only. Placement,
colors, interaction, and GPU submission belong to its callers.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from mojive.drawing.curves import CORNER_SMOOTHING, CURVE_TOLERANCE


def smooth_union(a, b, blend: float):
    """Blend two implicit fields with third-order contact at both blend limits."""
    if blend <= 0.0:
        return np.minimum(a, b)
    h = np.maximum(1.0 - np.abs(np.asarray(a) - b) / blend, 0.0)
    h2 = h * h
    return np.minimum(a, b) - blend / 32.0 * h2 * h2 * (10.0 + h * (h - 6.0))


def drag_link_field(x, y, distance: float, radius: float, width: float, smoothing: float):
    """Evaluate the same hollow-origin drag-link field as both GPU backends."""
    half = width * 0.5
    radial = np.hypot(x, y)
    outer = radial - radius - half
    hole = radius - half - radial
    along = np.clip(x, min(radius, distance), max(radius, distance))
    link = np.hypot(x - along, y) - half
    target = np.hypot(x - distance, y) - radius
    blend = radius * smoothing
    return smooth_union(np.maximum(smooth_union(outer, link, blend), hole), target, blend)


def _drag_hole_curve(distance, radius, width, smoothing, level, tolerance):
    """Parameterize the empty crescent by the sum of distances to both centers.

    On its boundary, r1-r0 = half_width + 2*level + blend*P((c-r0-r1)/blend).
    Choosing s=r0+r1 makes x and y explicit; only two scalar polynomial roots
    locate the axis intercept and the rightmost point. No per-sample root search
    or optimization is needed for the overlapping hole.
    """
    blend = radius * smoothing
    half = width * 0.5 + 2.0 * level
    c = 2.0 * radius - width * 0.5
    if radius <= width * 0.5 + level or distance <= half + blend * 5.0 / 16.0 + 1e-9:
        return np.empty((0, 2)), 0

    def profile(t):
        t2 = t * t
        return (5.0 + t2 * (15.0 + t2 * (t2 - 5.0))) / 16.0

    offset = distance - half
    if 0.0 < offset < blend:
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = (lo + hi) * 0.5
            if profile(mid) * blend < offset:
                lo = mid
            else:
                hi = mid
        offset = blend * (lo + hi) * 0.5
    lo, hi = max(distance, c - offset), c + offset

    def difference(parameter):
        t = np.clip((c - parameter) / max(blend, 1e-12), -1.0, 1.0)
        return half + np.where(
            np.abs(c - parameter) >= blend, np.abs(c - parameter), blend * profile(t)
        )

    def slope(parameter):
        t = min(1.0, max(-1.0, (c - parameter) / max(blend, 1e-12)))
        derivative = t * (30.0 - 20.0 * t * t + 6.0 * t**4) / 16.0
        diff = half + (abs(c - parameter) if abs(c - parameter) >= blend else blend * profile(t))
        return parameter * derivative - diff

    peak = min(hi, max(lo, c))
    if blend > 0.0:
        a, b = lo, hi
        for _ in range(28):
            mid = (a + b) * 0.5
            if slope(mid) > 0.0:
                a = mid
            else:
                b = mid
        peak = (a + b) * 0.5

    def points(parameters):
        diff = difference(parameters)
        x = (distance * distance - parameters * diff) / (2.0 * distance)
        r0 = (parameters - diff) * 0.5
        y = np.sqrt(np.maximum(0.0, r0 * r0 - x * x))
        return np.column_stack((x, y))

    count = max(17, math.ceil(math.pi * math.sqrt(radius / tolerance)))
    unit = (1.0 - np.cos(np.linspace(0.0, math.pi, count))) * 0.5
    parameters = np.unique(
        np.concatenate(
            (lo + unit * (hi - lo), np.clip((lo, hi, peak, c - blend, c + blend), lo, hi))
        )
    )
    curve = points(parameters)
    for _ in range(8):
        mid = (parameters[:-1] + parameters[1:]) * 0.5
        middle = points(mid)
        quarter = points(parameters[:-1] * 0.75 + parameters[1:] * 0.25)
        third = points(parameters[:-1] * 0.25 + parameters[1:] * 0.75)
        chord = curve[1:] - curve[:-1]
        chord_length = np.maximum(np.linalg.norm(chord, axis=1), 1e-12)
        error = np.zeros(len(mid))
        for sample in (middle, quarter, third):
            offset_xy = sample - curve[:-1]
            error = np.maximum(
                error,
                np.abs(chord[:, 0] * offset_xy[:, 1] - chord[:, 1] * offset_xy[:, 0])
                / chord_length,
            )
        refine = error > tolerance * 0.25
        if not np.any(refine):
            break
        merged = np.concatenate((parameters, mid[refine]))
        order = np.argsort(merged)
        parameters, curve = merged[order], np.vstack((curve, middle[refine]))[order]
    curve[0, 1] = curve[-1, 1] = 0.0
    return curve, int(np.argmax(curve[:, 0]))


@lru_cache(maxsize=128)
def smooth_drag_link_mesh(
    distance: float,
    radius: float,
    width: float,
    smoothing: float = CORNER_SMOOTHING,
    level: float = 0.0,
    tolerance: float = CURVE_TOLERANCE,
):
    """Tessellate an implicit drag link as linear strips, including its actual hollow center.

    Samples cluster around the two endpoint circles. A long straight connector adds no
    samples. Root searches are vectorized; strip indices avoid concave ear clipping.
    """

    reach = radius + width * 0.5 + 2.0 * radius * smoothing + level
    reference = 4.0 * reach
    if distance > 3.0 * reach and distance != reference:
        vertices, indices, outer, hole = smooth_drag_link_mesh(
            reference, radius, width, smoothing, level, tolerance
        )
        shift = distance - reference
        midpoint = reference * 0.5

        def stretch(points):
            return tuple((x + shift if x > midpoint else x, y) for x, y in points)

        return stretch(vertices), indices, stretch(outer), hole

    blend = radius * smoothing

    def union(a, b):
        if blend <= 0.0 or abs(a - b) >= blend:
            return min(a, b)
        t2 = ((a - b) / blend) ** 2
        return 0.5 * (a + b - blend * (5.0 + t2 * (15.0 + t2 * (-5.0 + t2))) / 16.0)

    def scalar_field(x, y=0.0):
        radial = math.hypot(x, y)
        outer = radial - radius - width * 0.5
        hole = radius - width * 0.5 - radial
        along = min(max(x, min(radius, distance)), max(radius, distance))
        link = math.hypot(x - along, y) - width * 0.5
        target = math.hypot(x - distance, y) - radius
        return union(max(union(outer, link), hole), target) - level

    def root(lo, hi, lo_inside, fn=scalar_field):
        for _ in range(24):
            mid = (lo + hi) * 0.5
            if (fn(mid) <= 0.0) == lo_inside:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    left = root(-reach, -radius, False)
    right = root(max(radius, distance), distance + reach + radius, True)
    hole_curve, peak_index = _drag_hole_curve(distance, radius, width, smoothing, level, tolerance)
    has_hole = len(hole_curve) > 0
    hole = (hole_curve[-1, 0], hole_curve[0, 0], hole_curve[peak_index, 0]) if has_hole else None
    hole_upper, hole_lower = hole_curve[peak_index:][::-1], hole_curve[: peak_index + 1]
    unit = (1.0 - np.cos(np.linspace(0.0, math.pi, 49))) * 0.5
    xs = np.unique(
        np.clip(
            np.concatenate(
                (
                    left + unit * (radius + reach - left),
                    distance - reach + unit * (right - distance + reach),
                    np.asarray((left, right)),
                    hole_curve[:, 0] if has_hole else np.empty(0),
                )
            ),
            left,
            right,
        )
    )
    max_y = reach

    def heights(x):
        seed = np.sqrt(np.maximum(0.0, radius * radius - x * x))

        # X is constant during all vertical root searches. Reuse squared
        # distances instead of clipping and preparing the same arrays each step.
        def vertical_field(xx):
            xx2 = xx * xx
            along2 = (xx - np.clip(xx, min(radius, distance), max(radius, distance))) ** 2
            target2 = (xx - distance) ** 2

            def evaluate(y):
                y2 = y * y
                radial = np.sqrt(xx2 + y2)
                outer = radial - radius - width * 0.5
                hole = radius - width * 0.5 - radial
                link = np.sqrt(along2 + y2) - width * 0.5
                target = np.sqrt(target2 + y2) - radius
                return (
                    smooth_union(np.maximum(smooth_union(outer, link, blend), hole), target, blend)
                    - level
                )

            return evaluate

        evaluate = vertical_field(x)
        lo, hi = seed.copy(), np.full_like(x, max_y)
        for _ in range(16):
            mid = (lo + hi) * 0.5
            inside = evaluate(mid) <= 0.0
            lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
        outer = (lo + hi) * 0.5
        inner_upper = np.zeros_like(x)
        inner_lower = np.zeros_like(x)
        if has_hole:
            active = (x >= hole[0] - 1e-7) & (x <= hole[2] + 1e-7)
            xx = x[active]
            inner_upper[active] = np.interp(xx, hole_upper[:, 0], hole_upper[:, 1])
            inner_lower[active] = np.interp(xx, hole_lower[:, 0], hole_lower[:, 1], left=0.0)
        outer[(x <= left + 1e-7) | (x >= right - 1e-7)] = 0.0
        return np.column_stack((outer, inner_upper, inner_lower))

    ys = heights(xs)
    for _ in range(10):
        mid = (xs[:-1] + xs[1:]) * 0.5
        middle_y = heights(mid)
        dy = ys[1:] - ys[:-1]
        dx = np.diff(xs)[:, None]
        error = np.abs(middle_y - (ys[:-1] + ys[1:]) * 0.5) * dx / np.sqrt(dx * dx + dy * dy)
        refine = np.any(error > tolerance * 0.5, axis=1)
        if hole:
            # Interior strip divisions outside the hole are not visible contours.
            outside = (mid < hole[0]) | (mid > hole[2])
            refine[outside] = error[outside, 0] > tolerance * 0.5
        if not np.any(refine):
            break
        merged_x = np.concatenate((xs, mid[refine]))
        order = np.argsort(merged_x)
        xs = merged_x[order]
        ys = np.vstack((ys, middle_y[refine]))[order]
    vertices = np.empty((len(xs), 6, 2))
    vertices[:, :, 0] = xs[:, None]
    vertices[:, :, 1] = np.column_stack(
        (-ys[:, 0], -ys[:, 1], -ys[:, 2], ys[:, 2], ys[:, 1], ys[:, 0])
    )
    strip = np.array((0, 1, 7, 0, 7, 6))
    indices = np.arange(len(xs) - 1)[:, None] * 6 + np.concatenate((strip, strip + 2, strip + 4))
    flat = vertices.reshape(-1, 2)
    triangles = indices.reshape(-1, 3)
    edges = flat[triangles[:, 1:]] - flat[triangles[:, :1]]
    area = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
    triangles = triangles[np.abs(area) > 1e-9]
    outer = np.vstack((vertices[:, 0], vertices[-2:0:-1, 5]))
    inner = np.empty((0, 2))
    if has_hole:
        inner = np.vstack((hole_curve, hole_curve[-2:0:-1] * (1.0, -1.0)))
        distinct = np.linalg.norm(inner - np.roll(inner, 1, axis=0), axis=1) > 1e-5
        inner = inner[distinct]
    # Collapsed strip divisions share one vertex. Compact within each column
    # in linear time before caching or transforming the mesh.
    distinct = np.ones(len(flat), dtype=bool)
    distinct[1:] = np.linalg.norm(np.diff(flat, axis=0), axis=1) > 1e-9
    remap = np.cumsum(distinct) - 1
    flat, triangles = flat[distinct], remap[triangles]
    return (
        tuple(map(tuple, flat.tolist())),
        tuple(triangles.ravel().tolist()),
        tuple(map(tuple, outer.tolist())),
        tuple(map(tuple, inner.tolist())),
    )
