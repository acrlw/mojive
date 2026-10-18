"""Settings row paint stays inside its actual navigation surface."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace.menu_geometry import native_unit
from mojive.ui.workspace.settings_layout import draw_navigation
from mojive.ui.workspace.settings_panel import CATEGORIES
from mojive.ui.workspace.widgets import UI
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("scale", (1.0, 1.2, 1.25, 2.25))
@pytest.mark.parametrize("narrow", (False, True))
@pytest.mark.parametrize("host", ("Settings", "Native specimen/Settings"))
@pytest.mark.parametrize("state", ("default", "hover", "selected"))
@pytest.mark.parametrize("authored", (False, True))
def test_navigation_paint_and_hits_preserve_real_insets(
    tmp_path, monkeypatch, scale, narrow, host, state, authored
):
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        io.display_size = (1000 * scale, 800 * scale)
        ui = UI(
            SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
            portable_fonts=True,
        )
        runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
        runtime.observe_components = True
        ui.design = DesignPresentation(ui, runtime)
        if authored:
            spec = deepcopy(runtime.spec)
            spec["parts"]["settings_nav"] = {
                "fontSize": 14,
                "iconSize": 20,
                "marginRight": 8,
                "paddingLeft": 6,
                "paddingRight": 6,
            }
            spec["parts"]["settings_nav_button"] = {"height": 36}
            runtime.update_spec(spec)
        panel = SimpleNamespace(
            workspace=SimpleNamespace(ui=ui, s=scale),
            _category="General" if state == "selected" else "Recording",
        )
        painted = []
        original = ui.rect

        def record_rectangle(x, y, width, height, *args, **kwargs):
            painted.append((x, y, width, height))
            return original(x, y, width, height, *args, **kwargs)

        monkeypatch.setattr(ui, "rect", record_rectangle)
        for frame in range(5):
            if state == "hover" and frame >= 2:
                x, y, width, height = ui.hits["settings-General"]
                io.add_mouse_pos_event(x + width / 2, y + height / 2)
            else:
                io.add_mouse_pos_event(-100, -100)
            ui.hits.clear()
            painted.clear()
            runtime.observed.clear()
            ui.design_primitives.observed.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin(host, flags=imgui.WindowFlags_.no_title_bar)
            with group_scope("Settings"), ui.design.family_scope("settings"):
                draw_navigation(
                    panel,
                    SimpleNamespace(tr=str),
                    Rect(20 * scale, 20 * scale, (420 if narrow else 700) * scale, 600 * scale),
                    narrow,
                )
            imgui.end()
            imgui.render()
        container = next(r for r in runtime.observed if r["key"] == "settings-navigation")
        x, y, width, height = container["hit"]
        cx, cy, cw, ch = container["content"]
        assert (cx - x, cy - y, x + width - cx - cw, y + height - cy - ch) == pytest.approx(
            (native_unit((6 if authored else 4) * scale),) * 4
        )
        assert painted[0] == pytest.approx(container["hit"])
        rows = [r for r in ui.design_primitives.observed if "clippedLabel" in r]
        assert len(rows) == 6
        for row in rows:
            rx, ry, rw, rh = row["hit"]
            assert cx <= rx < rx + rw <= cx + cw + 1
            assert cy <= ry < ry + rh <= cy + ch + 1
            assert not row["clippedLabel"]
        general = next(row for row in rows if row["key"] == "settings-General")
        assert general["active"] == (state == "selected")
        assert general["hovered"] == (state == "hover")
        matching = [r for r in painted[1:] if r == pytest.approx(general["hit"])]
        assert len(matching) == int(state != "default")
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("appearance", (1.0, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("narrow", (False, True))
@pytest.mark.parametrize(
    "padding,button_radius,radius",
    ((6, 6, None), (18, 6, None), (18, 15, None), (18, 15, 0), (18, 15, 21)),
)
def test_navigation_radius_matches_its_actual_inset_and_bounded_child_corner(
    appearance, narrow, padding, button_radius, radius
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    navigation = {"paddingX": padding, "paddingY": padding}
    if radius is not None:
        navigation["radius"] = radius
    update({"settings_nav": navigation, "settings_nav_button": {"radius": button_radius}})
    panel = SimpleNamespace(workspace=SimpleNamespace(ui=ui, s=ui.s), _category="General")

    def draw():
        runtime.begin_frame()
        with group_scope("Settings"), ui.design.family_scope("settings"):
            draw_navigation(
                panel,
                SimpleNamespace(tr=str),
                Rect(20 * ui.s, 20 * ui.s, (420 if narrow else 700) * ui.s, 600 * ui.s),
                narrow,
            )

    for _ in range(4):
        frame(draw)
    container = next(row for row in runtime.observed if row["key"] == "settings-navigation")
    first = ui.hits["settings-General"]
    bounded_inner = min(button_radius * ui.s, max(0, first[2] / 2 - 1), max(0, first[3] / 2 - 1))
    expected_radius = (
        bounded_inner + native_unit(padding * ui.s) if radius is None else radius * ui.s
    )
    assert container["radius"] == pytest.approx(expected_radius)
    assert container["innerRadius"] == pytest.approx(
        max(0, expected_radius - container["padding"][0])
    )
    assert container["padding"] == [native_unit(padding * ui.s)] * 4
    x, y, width, height = container["hit"]
    for category in CATEGORIES:
        rx, ry, rw, rh = ui.hits["settings-" + category]
        assert x <= rx < rx + rw <= x + width + 1
        assert y <= ry < ry + rh <= y + height + 1


@pytest.mark.parametrize("appearance", (1.0, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("radius", (0, 6, 12))
def test_floating_parent_constrains_navigation_and_its_actual_row_paint(
    appearance, monkeypatch, radius
):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"settings_nav_button": {"radius": 9}})
    panel = SimpleNamespace(workspace=SimpleNamespace(ui=ui, s=ui.s), _category="General")
    paints = []
    original = ui.rect

    def rectangle(x, y, width, height, *args, **options):
        paints.append(((x, y, width, height), options.get("r")))
        return original(x, y, width, height, *args, **options)

    monkeypatch.setattr(ui, "rect", rectangle)

    def draw():
        runtime.begin_frame()
        paints.clear()
        with ui.corner_scope(radius * ui.s, "settings_nav"):
            draw_navigation(
                panel,
                SimpleNamespace(tr=str),
                Rect(20 * ui.s, 20 * ui.s, 700 * ui.s, 600 * ui.s),
                False,
            )

    for _ in range(3):
        frame(draw)
    container = next(row for row in runtime.observed if row["key"] == "settings-navigation")
    inner = max(0, radius * ui.s - native_unit(4 * ui.s))
    assert container["parentCornerConstrained"]
    assert container["radius"] == pytest.approx(radius * ui.s)
    assert container["innerRadius"] == pytest.approx(inner)
    first = ui.hits["settings-General"]
    assert any(
        bounds == pytest.approx(first) and painted * ui.s == pytest.approx(inner)
        for bounds, painted in paints
        if painted is not None
    )
    assert ui.component_style("settings_nav_button").get("radius") == 9
