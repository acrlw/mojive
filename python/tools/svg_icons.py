"""Export production icons as outlined SVG assets and prepare them for native review."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from mojive.geometry2d.compiler import Contour2D
from mojive.geometry2d.curves import CORNER_SMOOTHING, circular_stroke_mesh, smooth_rect_points
from mojive.geometry2d.mesh import tessellate_contours
from mojive.geometry2d.polygons import signed_polygon_area
from mojive.ui.icon_draw import ImguiIconDraw
from mojive.ui.icons import (
    ICON_FAMILIES,
    ICON_GRID,
    REVIEW_LOCKED_ICONS,
    STROKE_SCALE_LOCKED_ICONS,
    draw_icon,
    production_icon_offset,
    production_icon_style,
)
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.severity_icons import (
    SEVERITY_FRAME_MIN_SEGMENTS,
    SEVERITY_FRAME_SEGMENT_DENSITY,
    severity_frame,
)

DEFAULT_DIRECTORY = Path("output/svg-icons")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ICON_NAMES = tuple(name for _, family in ICON_FAMILIES for _, name in family)
_FOREGROUND = (1.0, 1.0, 1.0, 1.0)
_ACCENT = (0.0, 1.0, 0.0, 1.0)
_TOKEN = re.compile(r"[MLZ]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("SVG coordinates must be finite")
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


def _closed_points(points):
    result = tuple((float(x), float(y)) for x, y in points)
    if len(result) > 1 and result[0] == result[-1]:
        result = result[:-1]
    return result


class _SvgDraw:
    """Record the production painter without creating a window or ImGui context.

    Geometry-only primitive methods are shared with the production adapters;
    this object deliberately is not an ImguiDraw2D submission adapter.
    """

    corner_smoothing = CORNER_SMOOTHING
    line = ImguiDraw2D.line
    arrow = ImguiDraw2D.arrow
    rect = ImguiIconDraw.rect

    def __init__(self, *, native_strokes=False, device_fringe=False):
        self.paths = []
        self.native_strokes = native_strokes
        self.device_fringe = device_fringe

    def _fill(self, contours, color, *, aa="fringe", fringe=1.0):
        contours = tuple(_closed_points(points) for points in contours if len(points) >= 3)
        if not contours:
            return
        role = "accent" if tuple(color[:3]) == _ACCENT[:3] else "foreground"
        if self.device_fringe and aa == "fringe":
            aa = "device-fringe"
        elif self.native_strokes and aa == "fringe":
            aa = "native-fringe"
        pieces = []
        for points in contours:
            pieces.append("M " + " L ".join(f"{_number(x)} {_number(y)}" for x, y in points) + " Z")
        self.paths.append(
            (
                "path",
                {
                    "d": " ".join(pieces),
                    "fill": "#9bbf88" if role == "accent" else "currentColor",
                    "fill-rule": "evenodd",
                    "fill-opacity": _number(color[3]),
                    "data-color-role": role,
                    "data-aa": aa,
                    "data-fringe-width": _number(fringe),
                },
            )
        )

    def polyline(self, points, color, width, *, closed=False, cap="butt", smoothing=None):
        if self.native_strokes and (closed or cap == "butt"):
            data = "M " + " L ".join(f"{_number(x)} {_number(y)}" for x, y in points)
            if closed:
                data += " Z"
            self.paths.append(
                (
                    "path",
                    {
                        "d": data,
                        "fill": "none",
                        "stroke": "currentColor",
                        "stroke-width": _number(width),
                        "stroke-opacity": _number(color[3]),
                        "stroke-linecap": "butt",
                        "stroke-linejoin": "miter",
                        "data-color-role": "foreground",
                    },
                )
            )
        else:
            painter = ImguiDraw2D if self.native_strokes else ImguiIconDraw
            painter.polyline(
                self, points, color, width, closed=closed, cap=cap, smoothing=smoothing
            )

    def circle(self, center, radius, color, width=1.0, *, segments=0):
        if self.native_strokes:
            self.paths.append(
                (
                    "circle",
                    {
                        "cx": _number(center[0]),
                        "cy": _number(center[1]),
                        "r": _number(radius),
                        "fill": "none",
                        "stroke": "currentColor",
                        "stroke-width": _number(width),
                        "stroke-opacity": _number(color[3]),
                        "data-color-role": "foreground",
                        "data-segments": "24",
                        "data-segment-density": "4",
                    },
                )
            )
        else:
            ImguiIconDraw.circle(self, center, radius, color, width, segments=segments)

    def convex_fill(self, points, color):
        self._fill((points,), color, aa="native")

    def fringed_concave_fill(self, points, color, *, origin=None):
        if origin is not None:
            points = tuple((x + origin[0], y + origin[1]) for x, y in points)
        self._fill((points,), color)

    def indexed_fill(
        self,
        points,
        indices,
        color,
        *,
        outline=(),
        hole=(),
        origin=None,
        direction=(1.0, 0.0),
        fringe_width=1.0,
    ):
        if not len(indices):
            return
        if not len(outline):
            raise ValueError("SVG export requires the mesh's exterior boundary")
        contours = (outline, hole) if len(hole) else (outline,)
        if origin is not None:
            dx, dy = direction
            contours = tuple(
                tuple(
                    (origin[0] + x * dx - y * dy, origin[1] + x * dy + y * dx) for x, y in contour
                )
                for contour in contours
            )
        if self.device_fringe and len(hole):
            # This is the shared severity frame. Preserve its authored circle
            # rather than freezing the 24-unit mesh's polygon sampling in SVG.
            points = contours[0]
            cx = (min(x for x, _ in points) + max(x for x, _ in points)) / 2
            cy = (min(y for _, y in points) + max(y for _, y in points)) / 2
            radius, width = severity_frame(ICON_GRID)
            self.paths.append(
                (
                    "circle",
                    {
                        "cx": _number(cx),
                        "cy": _number(cy),
                        "r": _number(radius),
                        "fill": "none",
                        "stroke": "currentColor",
                        "stroke-width": _number(width),
                        "stroke-opacity": _number(color[3]),
                        "data-color-role": "foreground",
                        "data-aa": "device-fringe",
                        "data-segments": str(SEVERITY_FRAME_MIN_SEGMENTS),
                        "data-segment-density": _number(SEVERITY_FRAME_SEGMENT_DENSITY),
                        "data-segment-basis": "size",
                    },
                )
            )
            return
        self._fill(contours, color, fringe=fringe_width)

    def circle_filled(self, center, radius, color, *, segments=0):
        if self.native_strokes:
            self.paths.append(
                (
                    "circle",
                    {
                        "cx": _number(center[0]),
                        "cy": _number(center[1]),
                        "r": _number(radius),
                        "fill": "currentColor",
                        "fill-opacity": _number(color[3]),
                        "data-color-role": "foreground",
                        "data-segments": "16",
                        "data-segment-density": "4",
                    },
                )
            )
            return
        count = max(12, segments or 64)
        points = tuple(
            (
                center[0] + radius * math.cos(i * math.tau / count),
                center[1] + radius * math.sin(i * math.tau / count),
            )
            for i in range(count)
        )
        self.convex_fill(points, color)

    def rect_filled(self, lo, hi, color, *, rounding=0.0, smoothing=None):
        self.convex_fill(
            smooth_rect_points(
                *lo,
                *hi,
                rounding,
                smoothing=self.corner_smoothing if smoothing is None else smoothing,
            ),
            color,
        )


def export_icon(name: str) -> str:
    """Export the current production shape, including its optical translation."""
    if name not in ICON_NAMES:
        raise ValueError(f"Unknown production icon: {name}")
    draw = _SvgDraw(
        native_strokes=name in STROKE_SCALE_LOCKED_ICONS,
        device_fringe=name in REVIEW_LOCKED_ICONS,
    )
    draw_icon(
        draw, (ICON_GRID / 2, ICON_GRID / 2), ICON_GRID, name, _FOREGROUND, accent_color=_ACCENT
    )
    fringe = (
        1.0 if name in STROKE_SCALE_LOCKED_ICONS else production_icon_style(name).stroke_width / 2
    )
    root = ET.Element(
        "svg",
        {
            "xmlns": SVG_NAMESPACE,
            "viewBox": "0 0 24 24",
            "width": "24",
            "height": "24",
            "preserveAspectRatio": "xMidYMid meet",
            "color": "#29323a",
            "data-icon": name,
            "data-fringe-width": _number(fringe),
        },
    )
    ET.SubElement(root, "title").text = name
    for tag, attributes in draw.paths:
        ET.SubElement(root, tag, attributes)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


@dataclass(frozen=True)
class SvgShape:
    contours: tuple
    role: str
    opacity: float
    aa: str
    fringe_width: float
    primitive: str = "fill"
    stroke_width: float = 0.0
    closed: bool = True
    circle: tuple = ()
    segments: int = 0
    segment_density: float = 0.0
    segment_basis: str = "radius"


@dataclass(frozen=True)
class SvgIcon:
    shapes: tuple[SvgShape, ...]
    fringe_width: float
    _content_hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Stable frame submissions must not hash every contour coordinate again.
        object.__setattr__(self, "_content_hash", hash((self.shapes, self.fringe_width)))

    def __hash__(self):
        return self._content_hash

    @property
    def bounds(self):
        bounds = []
        for shape in self.shapes:
            half = shape.stroke_width / 2
            if shape.circle:
                x, y, radius = shape.circle
                extent = radius + half
                bounds.append((x - extent, y - extent, x + extent, y + extent))
            else:
                points = [point for contour in shape.contours for point in contour]
                lo, hi = np.min(points, axis=0) - half, np.max(points, axis=0) + half
                bounds.append((*lo, *hi))
        return (*np.min(np.array(bounds)[:, :2], axis=0), *np.max(np.array(bounds)[:, 2:], axis=0))


def _parse_contours(data, *, allow_open=False):
    """Read the absolute M/L/Z contract, rejecting other SVG syntax."""
    if _TOKEN.sub("", data).strip(" ,\t\r\n"):
        raise ValueError("The SVG study supports absolute M/L/Z paths only")
    tokens = _TOKEN.findall(data)
    contours, points = [], []
    index = 0
    while index < len(tokens):
        command = tokens[index]
        index += 1
        if command == "Z":
            if len(points) < 3:
                raise ValueError("Filled SVG paths need at least three points")
            contours.append(_closed_points(points))
            points = []
            continue
        if command not in {"M", "L"} or index + 1 >= len(tokens):
            raise ValueError("Expected an explicit M/L coordinate pair or Z")
        if (command == "M") == bool(points):
            raise ValueError("Each closed contour must start with M and finish with Z")
        point = (float(tokens[index]), float(tokens[index + 1]))
        if not all(map(math.isfinite, point)):
            raise ValueError("SVG coordinates must be finite")
        points.append(point)
        index += 2
    if points and allow_open and len(points) >= 2:
        contours.append(tuple(points))
        points = []
    if points or not contours:
        raise ValueError("SVG contours must be closed")
    return tuple(contours)


def parse_svg(source: str) -> SvgIcon:
    """Load the study's path/circle subset, with uniform viewBox normalization.

    This is an explicit asset contract, not a general-purpose SVG implementation.
    Unsupported elements/attributes fail instead of producing a partial icon.
    """
    root = ET.fromstring(source)
    if root.tag != f"{{{SVG_NAMESPACE}}}svg":
        raise ValueError("Expected an SVG root in the SVG namespace")
    root_attributes = {
        "viewBox",
        "width",
        "height",
        "preserveAspectRatio",
        "color",
        "data-icon",
        "data-fringe-width",
    }
    if set(root.attrib) - root_attributes:
        raise ValueError("Unsupported SVG root attributes")
    x, y, width, height = map(float, root.attrib["viewBox"].replace(",", " ").split())
    if not all(map(math.isfinite, (x, y, width, height))) or min(width, height) <= 0:
        raise ValueError("SVG viewBox must be finite and non-empty")
    if root.get("preserveAspectRatio", "xMidYMid meet") != "xMidYMid meet":
        raise ValueError("SVG study requires preserveAspectRatio='xMidYMid meet'")
    scale = ICON_GRID / max(width, height)
    offset = ((ICON_GRID - width * scale) / 2, (ICON_GRID - height * scale) / 2)
    shapes = []
    allowed = {
        "d",
        "fill",
        "fill-rule",
        "fill-opacity",
        "data-color-role",
        "data-aa",
        "data-fringe-width",
        "stroke",
        "stroke-width",
        "stroke-opacity",
        "stroke-linecap",
        "stroke-linejoin",
        "cx",
        "cy",
        "r",
        "data-segments",
        "data-segment-density",
        "data-segment-basis",
    }
    for element in root:
        if element.tag in {f"{{{SVG_NAMESPACE}}}title", f"{{{SVG_NAMESPACE}}}desc"}:
            continue
        if (
            element.tag not in {f"{{{SVG_NAMESPACE}}}path", f"{{{SVG_NAMESPACE}}}circle"}
            or len(element)
            or set(element.attrib) - allowed
        ):
            raise ValueError("SVG study accepts flat paths and circles; bake transforms first")
        stroke = element.get("fill") == "none"
        paint = element.get("stroke") if stroke else element.get("fill", "currentColor")
        if paint not in {"currentColor", "#9bbf88"}:
            raise ValueError("SVG study supports foreground/accent theme colors only")
        if not stroke and "stroke" in element.attrib:
            raise ValueError("SVG study requires separate stroke and fill elements")
        if (
            element.get("stroke-linecap", "butt") != "butt"
            or element.get("stroke-linejoin", "miter") != "miter"
        ):
            raise ValueError("SVG study requires baked caps/joins except native butt/miter strokes")
        if element.get("fill-rule", "evenodd") != "evenodd":
            raise ValueError("SVG study requires evenodd filled paths")
        role = element.get("data-color-role", "foreground")
        aa = element.get("data-aa", "fringe")
        opacity = float(element.get("stroke-opacity" if stroke else "fill-opacity", 1))
        stroke_width = float(element.get("stroke-width", 1)) * scale if stroke else 0.0
        if not math.isfinite(stroke_width) or stroke_width < 0:
            raise ValueError("SVG stroke width must be finite and nonnegative")
        fringe = float(element.get("data-fringe-width", 1))
        if role not in {"foreground", "accent"} or aa not in {
            "fringe",
            "native",
            "native-fringe",
            "device-fringe",
        }:
            raise ValueError("Unsupported SVG paint role or antialias mode")
        if (
            not math.isfinite(opacity)
            or not 0 <= opacity <= 1
            or not math.isfinite(fringe)
            or fringe < 0
        ):
            raise ValueError("Invalid SVG opacity or fringe width")
        if element.tag.endswith("}circle"):
            cx, cy, radius = (float(element.get(key, 0)) for key in ("cx", "cy", "r"))
            segments = int(element.get("data-segments", 0))
            density = float(element.get("data-segment-density", 0))
            basis = element.get("data-segment-basis", "radius")
            if basis not in {"radius", "size"}:
                raise ValueError("Unsupported SVG circle segment basis")
            if (
                not all(map(math.isfinite, (cx, cy, radius, density)))
                or radius <= 0
                or segments < 0
                or density < 0
            ):
                raise ValueError("SVG circle requires a finite positive radius and valid segments")
            circle = ((cx - x) * scale + offset[0], (cy - y) * scale + offset[1], radius * scale)
            shapes.append(
                SvgShape(
                    (),
                    role,
                    opacity,
                    aa,
                    fringe * scale,
                    "circle-stroke" if stroke else "circle-fill",
                    stroke_width,
                    True,
                    circle,
                    segments,
                    density,
                    basis,
                )
            )
            continue
        closed = element.attrib["d"].rstrip().endswith("Z")
        contours = tuple(
            tuple(
                ((px - x) * scale + offset[0], (py - y) * scale + offset[1]) for px, py in contour
            )
            for contour in _parse_contours(element.attrib["d"], allow_open=stroke)
        )
        if stroke and len(contours) != 1:
            raise ValueError("SVG study requires one contour per stroke element")
        shapes.append(
            SvgShape(
                contours,
                role,
                opacity,
                aa,
                fringe * scale,
                "stroke" if stroke else "fill",
                stroke_width,
                closed,
            )
        )
    fringe = float(root.get("data-fringe-width", 1)) * scale
    if not shapes or not math.isfinite(fringe) or fringe < 0:
        raise ValueError("SVG must contain filled paths and a valid fringe width")
    return SvgIcon(tuple(shapes), fringe)


def load_icons(directory: Path) -> dict[str, SvgIcon]:
    """Read a complete production asset set atomically for reload in the probe."""
    icons = {}
    for name in ICON_NAMES:
        file = directory / f"{name}.svg"
        try:
            icon = parse_svg(file.read_text())
            prepare_svg(icon, ICON_GRID)
        except (ValueError, KeyError, ET.ParseError) as error:
            raise ValueError(f"{file.name}: {error}") from error
        icons[name] = icon
    return icons


def _boundary_loops(mesh):
    edges = {int(a): int(b) for a, b in mesh.boundary_edges}
    if len(edges) != len(mesh.boundary_edges):
        raise ValueError("SVG study requires non-touching contour boundaries")
    loops = []
    while edges:
        start = next(iter(edges))
        current, loop = start, []
        while current in edges:
            loop.append(tuple(mesh.positions[current]))
            current = edges.pop(current)
        if current != start:
            raise ValueError("SVG tessellation produced an open boundary")
        inside = signed_polygon_area(loop) < 0
        loops.append((tuple(reversed(loop)) if inside else tuple(loop), inside))
    return tuple(loops)


@lru_cache(maxsize=256)
def prepare_svg(icon: SvgIcon, size: float):
    """Cache local native geometry; position and interaction colors are late-bound."""
    if not math.isfinite(size) or size <= 0:
        raise ValueError("SVG icon size must be positive and finite")
    prepared = []
    unit = size / ICON_GRID
    for shape in icon.shapes:
        if shape.circle:
            x, y, radius = shape.circle
            circle = ((x - 12) * unit, (y - 12) * unit, radius * unit)
            if shape.primitive == "circle-stroke" and shape.aa == "device-fringe":
                basis = size if shape.segment_basis == "size" else circle[2]
                segments = max(shape.segments, math.ceil(basis * shape.segment_density))
                vertices, indices, outline, hole = circular_stroke_mesh(
                    circle[2], shape.stroke_width * unit, segments
                )
                prepared.append((shape, vertices, indices, (outline, hole), circle))
            else:
                prepared.append((shape, (), (), (), circle))
            continue
        contours = tuple(
            tuple(((x - 12) * unit, (y - 12) * unit) for x, y in path) for path in shape.contours
        )
        if shape.primitive == "stroke":
            prepared.append((shape, (), (), (), contours))
            continue
        mesh = tessellate_contours(
            (Contour2D(np.asarray(path), True) for path in contours), fill_rule="evenodd"
        )
        prepared.append(
            (
                shape,
                tuple(map(tuple, mesh.positions)),
                tuple(mesh.indices),
                _boundary_loops(mesh),
                contours,
            )
        )
    return tuple(prepared)


def draw_svg_icon(draw: ImguiDraw2D, center, size, icon: SvgIcon, color, *, accent_color=None):
    """Draw only loaded SVG data through the same ImGui adapter as production."""
    prepared = prepare_svg(icon, float(size))
    first = len(draw._dl.vtx_buffer)
    painter = ImguiIconDraw(draw, min(1.0, icon.fringe_width * size / ICON_GRID))
    density = draw._imgui.get_io().display_framebuffer_scale
    device_scale = max(1.0, density.x, density.y)
    for shape, positions, indices, loops, contours in prepared:
        tint = accent_color if shape.role == "accent" and accent_color is not None else color
        tint = (*tint[:3], tint[3] * shape.opacity)
        if shape.circle:
            x, y, radius = contours
            segments = max(shape.segments, round(radius * shape.segment_density))
            if shape.primitive == "circle-stroke":
                if shape.aa == "device-fringe":
                    outline, hole = loops
                    draw.indexed_fill(
                        positions,
                        indices,
                        tint,
                        outline=outline,
                        hole=hole,
                        origin=(x, y),
                        fringe_width=shape.fringe_width / device_scale,
                    )
                else:
                    draw.circle(
                        (x, y),
                        radius,
                        tint,
                        width=shape.stroke_width * size / ICON_GRID,
                        segments=segments,
                    )
            else:
                draw.circle_filled((x, y), radius, tint, segments=segments)
        elif shape.primitive == "stroke":
            for points in contours:
                draw.polyline(
                    points, tint, shape.stroke_width * size / ICON_GRID, closed=shape.closed
                )
        elif shape.aa == "native" and len(contours) == 1:
            draw.concave_fill(contours[0], tint)
        else:
            shape_painter = draw if shape.aa in {"native-fringe", "device-fringe"} else painter
            shape_painter.indexed_fill(positions, indices, tint)
            rgba = shape_painter._u32(tint)
            if shape.aa == "device-fringe":
                fringe = shape.fringe_width / device_scale
            elif shape.aa == "native-fringe":
                fringe = shape.fringe_width
            else:
                fringe = shape.fringe_width * size / ICON_GRID
            for outline, inside in loops:
                shape_painter._write_anti_alias_fringe(outline, rgba, inside=inside, width=fringe)
    draw._imgui.internal.shade_verts_transform_pos(
        draw._dl, first, len(draw._dl.vtx_buffer), (0.0, 0.0), 1.0, 0.0, draw._vec(center)
    )


def export_icons(directory: Path) -> dict:
    """Write standalone editable assets and a browser-native SVG contact sheet."""
    assets = {name: export_icon(name) for name in ICON_NAMES}
    icons = {name: parse_svg(source) for name, source in assets.items()}
    manifest = {
        "version": 1,
        "grid": ICON_GRID,
        "format": "svg-mlz-circles",
        "optical_offset_baked": True,
        "icons": [
            {
                "id": name,
                "label": label,
                "family": family,
                "file": f"{name}.svg",
                "bounds": list(icons[name].bounds),
                "paths": len(icons[name].shapes),
                "production_style": asdict(production_icon_style(name)),
                "optical_offset": production_icon_offset(name),
            }
            for family, entries in ICON_FAMILIES
            for label, name in entries
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in assets.items():
        (directory / f"{name}.svg").write_text(source)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cards = "\n".join(
        f'<article data-family="{html.escape(item["family"])}" data-name="{item["id"]}">'
        f'<div class="slot">{assets[item["id"]]}</div><strong>{html.escape(item["label"])}</strong>'
        f'<small>{item["id"]}</small><a href="{item["file"]}" download>Download SVG</a></article>'
        for item in manifest["icons"]
    )
    template = Path(__file__).with_name("svg_icons_gallery.html").read_text()
    (directory / "index.html").write_text(template.replace("<!-- ICON_CARDS -->", cards))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    manifest = export_icons(args.output)
    print(f"Exported {len(manifest['icons'])} SVG icons to {args.output.resolve()}")


if __name__ == "__main__":
    main()
