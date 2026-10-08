"""Compile the browser's SVG masters to shared-grid contours; no runtime dependency."""

import json
import math
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path

ROOT = Path(__file__).parent


class Contours(BasePen):
    def __init__(self):
        super().__init__(None)
        self.paths = []
        self.points = []

    def _moveTo(self, p):
        self._endPath()
        self.points = [p]

    def _lineTo(self, p):
        self.points.append(p)

    def _curveToOne(self, a, b, c):
        o = self.points[-1]
        for step in range(1, 17):
            t = step / 16
            self.points.append(
                tuple(
                    (1 - t) ** 3 * o[j]
                    + 3 * (1 - t) ** 2 * t * a[j]
                    + 3 * (1 - t) * t * t * b[j]
                    + t**3 * c[j]
                    for j in range(2)
                )
            )

    def _qCurveToOne(self, a, b):
        o = self.points[-1]
        for step in range(1, 17):
            t = step / 16
            self.points.append(
                tuple((1 - t) ** 2 * o[j] + 2 * (1 - t) * t * a[j] + t * t * b[j] for j in range(2))
            )

    def _closePath(self):
        if self.points:
            self.paths.append((self.points, True))
            self.points = []

    def _endPath(self):
        if self.points:
            self.paths.append((self.points, False))
            self.points = []


def dashed_paths(points, closed, pattern):
    if not pattern or any(length <= 0 for length in pattern):
        raise ValueError("Dash lengths must be positive")
    if len(pattern) % 2:
        pattern = pattern * 2
    vertices = [*points, points[0]] if closed else points
    result, active = [], []
    index, remaining = 0, pattern[0]
    for start, end in pairwise(vertices):
        length = math.dist(start, end)
        if length < 1e-9:
            continue
        distance = 0
        while distance < length - 1e-9:
            step = min(remaining, length - distance)
            a, b = distance / length, (distance + step) / length
            p = tuple(x + (y - x) * a for x, y in zip(start, end, strict=True))
            q = tuple(x + (y - x) * b for x, y in zip(start, end, strict=True))
            if index % 2 == 0:
                if not active:
                    active.append(p)
                active.append(q)
            distance += step
            remaining -= step
            if remaining < 1e-9:
                if active:
                    result.append(active)
                    active = []
                index = (index + 1) % len(pattern)
                remaining = pattern[index]
    if active:
        result.append(active)
    return result


def compile_svg(svg):
    root = ET.fromstring(svg)
    view_box = [float(v) for v in root.attrib["viewBox"].split()]
    vx, vy, vw, vh = view_box
    if vw <= 0 or vh <= 0:
        raise ValueError("Icon viewBox must have positive dimensions")
    # SVG's default xMidYMid meet: one scale, with centered letterboxing.
    scale = 24 / max(vw, vh)
    dx, dy = (24 - vw * scale) / 2 - vx * scale, (24 - vh * scale) / 2 - vy * scale
    contours = []
    bounds = [math.inf, math.inf, -math.inf, -math.inf]
    for e in root:
        a = root.attrib | e.attrib
        if "transform" in a:
            raise ValueError("Bake transforms into the shared master before compiling")
        opacity = float(root.get("opacity", 1)) * float(e.get("opacity", 1))
        if e.tag == "path":
            pen = Contours()
            parse_path(a["d"], pen)
            pen._endPath()
            paths = pen.paths
        elif e.tag in ("circle", "ellipse"):
            cx, cy = (float(a[k]) for k in ["cx", "cy"])
            rx = float(a.get("rx", a.get("r", 0)))
            ry = float(a.get("ry", a.get("r", 0)))
            paths = [
                (
                    [
                        (
                            cx + rx * math.cos(i * math.tau / 48),
                            cy + ry * math.sin(i * math.tau / 48),
                        )
                        for i in range(48)
                    ],
                    True,
                )
            ]
        elif e.tag == "rect":
            x, y, w, h = (float(a[k]) for k in ["x", "y", "width", "height"])
            r = float(a.get("rx", 0))
            r = min(r, w / 2, h / 2)
            pts = []
            for cx, cy, start in [
                (x + w - r, y + r, -90),
                (x + w - r, y + h - r, 0),
                (x + r, y + h - r, 90),
                (x + r, y + r, 180),
            ]:
                pts.extend(
                    (
                        cx + r * math.cos(math.radians(start + i * 90 / 8)),
                        cy + r * math.sin(math.radians(start + i * 90 / 8)),
                    )
                    for i in range(9)
                )
            paths = [(pts, True)]
        else:
            raise ValueError(e.tag)
        for points, closed in paths:
            # SVG paths may repeat their closing vertex; capsule corners may coincide.
            # ImGui's polyline normals require nonzero edges, including the closing edge.
            clean = []
            for point in points:
                if not clean or math.dist(clean[-1], point) > 1e-6:
                    clean.append(point)
            if closed and len(clean) > 1 and math.dist(clean[0], clean[-1]) <= 1e-6:
                clean.pop()
            points = clean
            stroke = a.get("stroke", "none")
            width = float(a.get("stroke-width", 1)) if stroke != "none" else 0
            half = width / 2
            for px, py in points:
                bounds = [
                    min(bounds[0], px - half),
                    min(bounds[1], py - half),
                    max(bounds[2], px + half),
                    max(bounds[3], py + half),
                ]
            normalized = [(round(x * scale + dx, 6), round(y * scale + dy, 6)) for x, y in points]
            contours.append(
                {
                    "points": normalized,
                    "closed": closed,
                    "fill": a.get("fill", "black"),
                    "stroke": stroke,
                    "width": width * scale,
                    "opacity": opacity,
                    "fill_opacity": float(a.get("fill-opacity", 1)),
                    "stroke_opacity": float(a.get("stroke-opacity", 1)),
                    "dash": [float(v) * scale for v in a.get("stroke-dasharray", "").split()],
                }
            )
            if contours[-1]["dash"]:
                contours[-1]["stroke_paths"] = dashed_paths(
                    normalized, closed, contours[-1]["dash"]
                )
    if not contours or any(not math.isfinite(v) for v in bounds):
        raise ValueError("Icon must contain finite visible geometry")
    if bounds[0] < vx or bounds[1] < vy or bounds[2] > vx + vw or bounds[3] > vy + vh:
        raise ValueError(f"Stroked icon bounds {bounds} exceed its viewBox {view_box}")
    return {
        "viewBox": view_box,
        "bounds": [
            bounds[0] * scale + dx,
            bounds[1] * scale + dy,
            bounds[2] * scale + dx,
            bounds[3] * scale + dy,
        ],
        "contours": contours,
    }


def main():
    result = {}
    for name, svg in json.loads((ROOT / "icons-source.json").read_text()).items():
        try:
            result[name] = compile_svg(svg)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    (ROOT / "icons.json").write_text(json.dumps(result, separators=(",", ":")))
    print("Compiled and checked", len(result), "shared SVG masters")


if __name__ == "__main__":
    main()
