"""Prepare packaged canonical meshes without runtime stroke union or native draw lists."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from imgui_bundle import imgui

from mojive.geometry2d.icon_mesh import compile_icon_layers
from mojive.ui.icon_draw import ImguiIconDraw
from mojive.ui.imgui_draw import ImguiDraw2D, _concave_indices


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
    return compiled["icons"]


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


@lru_cache(maxsize=512)
def prepared_icon(name, size):
    """Scale the authored mesh and prepare only its size-dependent AA fringe."""
    from mojive.ui.workspace.widget_icons import GLYPHS, ICON_GRID, ICON_STROKE

    if size <= 0:
        return ()
    layers = _packaged_layers().get(name)
    if layers is None:
        # Explicit in-memory glyphs used by geometry tools have no packaged mesh.
        layers = compile_icon_layers(GLYPHS[name]["contours"])
    scale = size / ICON_GRID
    recorder = _MeshRecorder(scale)
    painter = ImguiIconDraw(ImguiDraw2D(recorder), min(1.0 / scale, ICON_STROKE / 2))
    groups = []
    for layer in layers:
        recorder.operations = []
        if "fill" in layer:
            painter.fringed_concave_fill(tuple(map(tuple, layer["fill"])), (1.0, 1.0, 1.0, 1.0))
        else:
            mesh = layer["stroke"]
            painter.indexed_fill(
                tuple(map(tuple, mesh["vertices"])), tuple(mesh["indices"]), (1.0, 1.0, 1.0, 1.0)
            )
            for boundary, inside in mesh["boundaries"]:
                painter._write_anti_alias_fringe(
                    tuple(map(tuple, boundary)), 0xFFFFFFFF, inside=inside
                )
        groups.append((layer["background"], layer["opacity"], tuple(recorder.operations)))
    return tuple(groups)
