from __future__ import annotations

import ast
import colorsys
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forge_viewer.adapters.base import NodeKind
from forge_viewer.ui import theme

MAX_NODE_SATURATION = 0.40


MAX_NODE_CHROMA = 40.0


MIN_NODE_LSTAR_GAP = 6.0


MIN_DANGER_PRIMARY_LUMA_GAP = 30.0


MAX_AXIS_LSTAR_SPREAD = 1.5


def _hsl_saturation(color) -> float:
    _h, _l, s = colorsys.rgb_to_hls(*color[:3])
    return s


def _hue_deg(color) -> float:
    h, _l, _s = colorsys.rgb_to_hls(*color[:3])
    return h * 360.0


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _xyz(color):
    r, g, b = (_linear(c) for c in color[:3])
    return (
        0.4124 * r + 0.3576 * g + 0.1805 * b,
        0.2126 * r + 0.7152 * g + 0.0722 * b,
        0.0193 * r + 0.1192 * g + 0.9505 * b,
    )


def _f(t: float) -> float:
    return t ** (1.0 / 3.0) if t > 216.0 / 24389.0 else (841.0 / 108.0) * t + 4.0 / 29.0


def _lstar(color) -> float:
    _x, y, _z = _xyz(color)
    return 116.0 * _f(y) - 16.0


def _chroma(color) -> float:
    x, y, z = _xyz(color)
    fx, fy, fz = _f(x / 0.95047), _f(y), _f(z / 1.08883)
    return math.hypot(500.0 * (fx - fy), 200.0 * (fy - fz))


def _luma601(color) -> float:
    r, g, b = color[:3]
    return 255.0 * (0.299 * r + 0.587 * g + 0.114 * b)


def test_independent_color_math_agrees_with_theme():

    assert _lstar((1.0, 1.0, 1.0, 1.0)) == pytest.approx(100.0, abs=0.01)
    assert _lstar((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0, abs=0.01)
    assert _lstar((1.0, 0.0, 0.0, 1.0)) == pytest.approx(53.24, abs=0.05)
    assert _lstar((0.0, 1.0, 0.0, 1.0)) == pytest.approx(87.73, abs=0.05)
    assert _lstar((0.0, 0.0, 1.0, 1.0)) == pytest.approx(32.30, abs=0.05)
    assert _hsl_saturation((1.0, 0.0, 0.0, 1.0)) == pytest.approx(1.0)
    assert _luma601((1.0, 1.0, 1.0, 1.0)) == pytest.approx(255.0)


def test_every_node_kind_has_a_color():

    missing = [k for k in NodeKind if k not in theme.NODE_COLORS]
    assert not missing
    assert len(theme.NODE_COLORS) == len(list(NodeKind))


@pytest.mark.parametrize("kind", list(NodeKind))
def test_node_colors_are_desaturated(kind: NodeKind):

    color = theme.NODE_COLORS[kind]
    sat = _hsl_saturation(color)
    chroma = _chroma(color)
    assert sat < MAX_NODE_SATURATION
    assert chroma < MAX_NODE_CHROMA


def test_node_colors_are_separated_by_lightness():

    items = [(k, _lstar(c)) for k, c in theme.NODE_COLORS.items()]
    worst = min((abs(a[1] - b[1]), a[0], b[0]) for i, a in enumerate(items) for b in items[i + 1 :])
    gap, _ka, _kb = worst
    assert gap >= MIN_NODE_LSTAR_GAP


def test_node_colors_are_not_the_axis_colors():

    axis = set(theme.AXIS_COLORS.values())
    clash = [k for k, c in theme.NODE_COLORS.items() if c in axis]
    assert not clash


def test_primary_is_desaturated():

    sat = _hsl_saturation(theme.PRIMARY)
    hue = _hue_deg(theme.PRIMARY)
    assert sat < 0.25
    assert 70.0 <= hue <= 100.0


def test_danger_is_orange_red_not_pure_red():

    hue = _hue_deg(theme.DANGER)
    assert 8.0 <= hue <= 30.0


def test_danger_primary_luma_gap():

    danger = _luma601(theme.DANGER)
    primary = _luma601(theme.PRIMARY)
    gap = primary - danger
    assert gap >= MIN_DANGER_PRIMARY_LUMA_GAP

    assert danger == pytest.approx(130.0, abs=6.0)
    assert primary == pytest.approx(165.0, abs=6.0)


def test_axis_colors_are_luminance_balanced():

    ls = {k: _lstar(c) for k, c in theme.AXIS_COLORS.items()}
    assert set(ls) == {"x", "y", "z"}
    spread = max(ls.values()) - min(ls.values())
    assert spread <= MAX_AXIS_LSTAR_SPREAD


def test_native_gizmo_uses_theme_axis_colors():

    from forge_viewer.gizmo import AXIS_COLORS as GIZMO_COLORS

    expected = [theme.AXIS_COLORS[k] for k in theme.AXIS_ORDER]
    for actual, want in zip(GIZMO_COLORS, expected, strict=True):
        assert tuple(actual) == pytest.approx(want)


def test_derived_colors_are_axis_colors_not_lookalikes():

    axis = list(theme.AXIS_COLORS.values())
    assert theme.PERTURB_COMMANDED in axis
    assert theme.PERTURB_ACTUAL in axis
    assert theme.PERTURB_COMMANDED != theme.PERTURB_ACTUAL


def test_apply_does_not_need_imgui_imported_at_module_level():

    tree = ast.parse(Path(theme.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "imgui" not in imported and "imgui_bundle" not in imported, imported

    calls: list[tuple[int, tuple]] = []

    class _Vec2:
        def __init__(self, *a):
            self.a = a

    class _Vec4:
        def __init__(self, *a):
            self.a = a

    class _Style:
        def set_color_(self, idx, col):
            calls.append((idx, col.a))

        def scale_all_sizes(self, s):
            calls.append((-1, (s,)))

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    class _Col:
        def __getattr__(self, name):
            return abs(hash(name)) % 60

    class _FakeImgui:
        Col_ = _Col()
        ImVec2 = _Vec2
        ImVec4 = _Vec4
        _style = _Style()

        @staticmethod
        def get_style():
            return _FakeImgui._style

    theme.apply(_FakeImgui, ui_scale=2.0)
    assert calls
    assert (-1, (2.0,)) in calls
