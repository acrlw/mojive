"""Compile the browser's SVG masters to shared-grid contours; no runtime dependency."""

import hashlib
import json
import math
import re
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path

ROOT = Path(__file__).parent
COMPILED_ICONS = ROOT.parents[2] / "python/ui/workspace/icons.json"
SCENE_ICONS = ROOT.parents[2] / "python/interaction/resources/scene_icons.json"
COMPILED_MESHES = COMPILED_ICONS.with_name("icon_meshes.json")


def library_masters():
    """Preserve upstream SVG geometry and normalize the whole family's stroke."""
    directory = ROOT / "svg"
    manifest = json.loads((directory / "sources.json").read_text())
    sources = {}
    for name, path in manifest["icons"].items():
        root = ET.fromstring((directory / path).read_text())
        for element in root.iter():
            element.tag = element.tag.rsplit("}", 1)[-1]
        root.set("stroke-width", str(manifest["style"]["stroke"]))
        sources[name] = ET.tostring(root, encoding="unicode")
    return sources


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
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    view_box = [float(v) for v in root.attrib["viewBox"].split()]
    vx, vy, vw, vh = view_box
    if vw <= 0 or vh <= 0:
        raise ValueError("Icon viewBox must have positive dimensions")
    # SVG's default xMidYMid meet: one scale, with centered letterboxing.
    scale = 24 / max(vw, vh)
    dx, dy = (24 - vw * scale) / 2 - vx * scale, (24 - vh * scale) / 2 - vy * scale
    contours = []
    bounds = [math.inf, math.inf, -math.inf, -math.inf]
    for element_index, e in enumerate(root):
        a = root.attrib | e.attrib
        if "transform" in a:
            raise ValueError("Bake transforms into the shared master before compiling")
        opacity = float(root.get("opacity", 1)) * float(e.get("opacity", 1))
        if e.tag == "path":
            pen = Contours()
            parse_path(a["d"], pen)
            pen._endPath()
            paths = pen.paths
        elif e.tag == "line":
            paths = [([(float(a["x1"]), float(a["y1"])), (float(a["x2"]), float(a["y2"]))], False)]
        elif e.tag in ("polyline", "polygon"):
            coordinates = [float(value) for value in re.split(r"[\s,]+", a["points"].strip())]
            if len(coordinates) % 2:
                raise ValueError("SVG points must contain complete coordinate pairs")
            paths = [
                (list(zip(coordinates[::2], coordinates[1::2], strict=True)), e.tag == "polygon")
            ]
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
            cap = a.get("stroke-linecap", "butt")
            join = a.get("stroke-linejoin", "miter")
            miter_limit = float(a.get("stroke-miterlimit", 4))
            if cap not in ("butt", "round", "square"):
                raise ValueError(f"Unsupported SVG stroke cap: {cap}")
            if join not in ("miter", "round", "bevel"):
                raise ValueError(f"Unsupported SVG stroke join: {join}")
            if not math.isfinite(miter_limit) or miter_limit < 1:
                raise ValueError("SVG stroke miter limit must be finite and at least one")
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
                    "cap": cap,
                    "join": join,
                    "miter_limit": miter_limit,
                    "opacity": opacity,
                    "fill_opacity": float(a.get("fill-opacity", 1)),
                    "stroke_opacity": float(a.get("stroke-opacity", 1)),
                    "dash": [float(v) * scale for v in a.get("stroke-dasharray", "").split()],
                }
            )
            if len(paths) > 1 and a.get("fill", "black") != "none":
                # Subpaths in one SVG element share a fill rule. Separate fills
                # would close apertures in rings, bearings and outlined handles.
                rule = a.get("fill-rule", "nonzero")
                if rule not in ("nonzero", "evenodd"):
                    raise ValueError(f"Unsupported SVG fill rule: {rule}")
                contours[-1].update(fill_group=element_index, fill_rule=rule)
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
        # Bounds use the same canonical precision as contour coordinates so
        # platform trigonometric roundoff cannot invalidate packaged resources.
        "bounds": [
            round(bounds[0] * scale + dx, 6),
            round(bounds[1] * scale + dy, 6),
            round(bounds[2] * scale + dx, 6),
            round(bounds[3] * scale + dy, 6),
        ],
        "contours": contours,
    }


def main():
    result = {}
    sources = json.loads((ROOT / "icons-source.json").read_text())
    # One authored stroke applies to all outline masters, including local domain
    # glyphs. Scale at submission; never impose a separate minimum pixel width.
    manifest = json.loads((ROOT / "svg/sources.json").read_text())
    for name, svg in sources.items():
        root = ET.fromstring(svg)
        root.set("stroke-width", str(manifest["style"]["stroke"]))
        sources[name] = ET.tostring(root, encoding="unicode")
    sources.update(library_masters())
    for name, svg in sources.items():
        try:
            result[name] = compile_svg(svg)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
    source_payload = json.dumps(sources, separators=(",", ":"))
    compiled = json.dumps(result, separators=(",", ":"))
    # Fixed union geometry belongs to resource generation. Only the AA fringe
    # depends on the runtime size; defer neither startup nor first-use clicks.
    from mojive.geometry2d.icon_mesh import compile_icon_layers

    mesh_payload = json.dumps(
        {
            "version": 1,
            "source_sha256": hashlib.sha256(compiled.encode()).hexdigest(),
            "icons": {
                name: compile_icon_layers(glyph["contours"]) for name, glyph in result.items()
            },
        },
        separators=(",", ":"),
    )
    scene_payload = json.dumps(
        {name: result[name] for name in ("camera", "light")}, separators=(",", ":")
    )
    # Validation/compilation failures must leave the previous resource set intact.
    (ROOT / "icons-source.json").write_text(source_payload)
    COMPILED_ICONS.write_text(compiled)
    COMPILED_MESHES.write_text(mesh_payload)
    SCENE_ICONS.parent.mkdir(parents=True, exist_ok=True)
    SCENE_ICONS.write_text(scene_payload)
    print("Compiled and checked", len(result), "shared SVG masters")


if __name__ == "__main__":
    main()
