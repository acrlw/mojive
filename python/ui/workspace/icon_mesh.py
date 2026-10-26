"""Prepare packaged canonical meshes without runtime stroke union or native draw lists."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from imgui_bundle import imgui

from mojive.geometry2d.icon_mesh import compile_icon_layers
from mojive.ui.icon_draw import ImguiIconDraw
from mojive.ui.imgui_draw import ImguiDraw2D, _concave_indices


def _freeze_icon_arrays(value):
    """Let cyclic GC stop scanning immutable master and mesh coordinate arrays."""
    if isinstance(value, list):
        return tuple(_freeze_icon_arrays(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_icon_arrays(item) for key, item in value.items()}
    return value


@lru_cache(maxsize=1)
def _packaged_layers():
    root = Path(__file__).parent
    source = (root / "icons.json").read_bytes()
    compiled = json.loads((root / "icon_meshes.json").read_text())
    if (
        compiled["version"] != 1
        or compiled["source_sha256"] != hashlib.sha256(source).hexdigest()
        or set(compiled["icons"]) != set(json.loads(source))
    ):
        raise ValueError(
            "Packaged icon meshes do not match their masters; run make ui-design-icons"
        )
    return _freeze_icon_arrays(compiled["icons"])


class _MeshRecorder:
    def __init__(self, scale):
        self.flags = imgui.ImDrawListFlags_.anti_aliased_fill.value
        self.operations = []
        self.scale = scale

    def vertices(self, points):
        return [imgui.ImVec2(point.x * self.scale, point.y * self.scale) for point in points]

    def add_indexed_fill(self, points, indices, color):
        self.operations.append((False, self.vertices(points), indices))

    def add_poly_fringe(self, inner, outer, color):
        self.operations.append((True, self.vertices(inner), self.vertices(outer)))

    def add_concave_poly_filled(self, points, color):
        path = tuple((point.x, point.y) for point in points)
        self.add_indexed_fill(points, _concave_indices(path), color)


def prepared_icon(name, size):
    """Build temporary scaled vertices and AA fringe for the retained native mesh."""
    from mojive.ui.workspace.widget_icons import GLYPHS, ICON_GRID, ICON_STROKE

    if size <= 0:
        return ()
    layers = _packaged_layers().get(name)
    if layers is None:
        # Explicit in-memory glyphs used by geometry tools have no packaged mesh.
        layers = compile_icon_layers(GLYPHS[name]["contours"])
    scale = size / ICON_GRID
    recorder = _MeshRecorder(scale)
    stroke = min(
        (contour["width"] for contour in GLYPHS[name]["contours"] if contour["stroke"] != "none"),
        default=ICON_STROKE,
    )
    painter = ImguiIconDraw(ImguiDraw2D(recorder), min(1.0 / scale, stroke / 2))
    groups = []
    for layer in layers:
        recorder.operations = []
        if "fill" in layer:
            painter.fringed_concave_fill(tuple(map(tuple, layer["fill"])), (1.0, 1.0, 1.0, 1.0))
        else:
            mesh = layer.get("mesh", layer.get("stroke"))
            painter.indexed_fill(
                tuple(map(tuple, mesh["vertices"])), tuple(mesh["indices"]), (1.0, 1.0, 1.0, 1.0)
            )
            for boundary, inside in mesh["boundaries"]:
                painter._write_anti_alias_fringe(
                    tuple(map(tuple, boundary)), 0xFFFFFFFF, inside=inside
                )
        groups.append((layer["background"], layer["opacity"], tuple(recorder.operations)))
    return tuple(groups)


@lru_cache(maxsize=512)
def native_icon(name, size, antialias=True):
    """Retain validated native buffers; each layer needs only placement and ink per frame."""
    groups = []
    for background, opacity, operations in prepared_icon(name, size):
        points, indices, opaque = [], [], []
        for fringe, vertices, elements in operations:
            if fringe and not antialias:
                continue
            base = len(points)
            if fringe:
                for inner, outer in zip(vertices, elements, strict=True):
                    points.extend((inner, outer))
                    opaque.extend((True, False))
                for i in range(len(vertices)):
                    a, b = base + i * 2, base + (i + 1) % len(vertices) * 2
                    indices.extend((a, b, b + 1, a, b + 1, a + 1))
            else:
                points.extend(vertices)
                opaque.extend([True] * len(vertices))
                indices.extend(base + i for i in elements)
        groups.append((background, opacity, imgui.DrawMesh(points, indices, opaque)))
    return tuple(groups)


@lru_cache(maxsize=512)
def _scaled_layers(name, size):
    """Draw2D/export coordinates use the same package, centered on the icon slot."""
    unit = size / 24

    def points(values):
        return tuple(((x - 12) * unit, (y - 12) * unit) for x, y in values)

    result = []
    for layer in _packaged_layers()[name]:
        if "fill" in layer:
            result.append((layer, points(layer["fill"]), (), ()))
        else:
            mesh = layer.get("mesh", layer.get("stroke"))
            boundaries = tuple((points(edge), inside) for edge, inside in mesh["boundaries"])
            result.append((layer, points(mesh["vertices"]), tuple(mesh["indices"]), boundaries))
    return tuple(result)


def draw_packaged_icon(draw, center, size, name, color, *, stroke_color=None):
    """Share adopted geometry with legacy Draw2D controls and vector exporters."""
    if size <= 0:
        return

    native = getattr(draw, "native_draw_list", None)
    if isinstance(native, imgui.ImDrawList):
        from mojive.ui.imgui_draw import _cached_color

        antialias = bool(native.flags & imgui.ImDrawListFlags_.anti_aliased_fill.value)
        for layer, (_background, opacity, mesh) in zip(
            _packaged_layers()[name], native_icon(name, size, antialias), strict=True
        ):
            ink = stroke_color if stroke_color is not None and "stroke" in layer else color
            rgba = _cached_color((ink[0], ink[1], ink[2], ink[3] * opacity))
            mesh.draw(native, (center[0] - size / 2, center[1] - size / 2), rgba)
        return

    def placed(points):
        return tuple((x + center[0], y + center[1]) for x, y in points)

    for layer, vertices, indices, boundaries in _scaled_layers(name, float(size)):
        ink = stroke_color if stroke_color is not None and "stroke" in layer else color
        tint = (ink[0], ink[1], ink[2], ink[3] * layer["opacity"])
        if "fill" in layer:
            draw.fringed_concave_fill(placed(vertices), tint)
        else:
            draw.indexed_fill(
                placed(vertices),
                indices,
                tint,
                boundaries=tuple((placed(edge), inside) for edge, inside in boundaries),
                fringe_width=min(1.0, size / 32),
            )
