"""Settings title actions follow the real native host's top/right corner."""

from copy import deepcopy
from math import sqrt
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.geometry2d.curves import smooth_rect_points
from mojive.ui.workspace import panel_host, settings_layout
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit
from mojive.ui.workspace.settings_panel import WorkspaceSettings
from mojive.ui.workspace.widgets import color
from tests.test_component_appearance import appearance as appearance
from tests.test_settings_corner_chain import _paint_covers

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


@pytest.fixture
def header(appearance, monkeypatch):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        open={"Settings": True},
        widths={},
        focus_next=None,
        bottom_focus=None,
        app=SimpleNamespace(apply_document_edits=lambda: None),
        scene_panel=None,
        inspector=None,
        control_panel=None,
        assets_panel=None,
        pose_panel=None,
        camera_panel=None,
        layers_panel=None,
        timeline=None,
    )
    workspace.settings = WorkspaceSettings(workspace)
    workspace.app.panels = SimpleNamespace(
        get=lambda _name: SimpleNamespace(finish_frame=lambda _ctx: None)
    )
    monkeypatch.setattr(settings_layout, "_draw_page", lambda *_args: None)
    painted, icons = [], []
    rectangle, icon = ui.rect, ui.icon

    def rect(x, y, width, height, tint, r=None, **options):
        resolved = tint if options.get("color_resolved") else ui.ink(tint)
        painted.append(
            (
                (x, y, width, height),
                (r or 0) * ui.s,
                color(resolved),
                imgui.internal.get_current_window().id_,
            )
        )
        return rectangle(x, y, width, height, tint, r, **options)

    def glyph(name, x, y, size=16, *args, **options):
        if name == "settings":
            icons.append((x, y, size * ui.s, size * ui.s))
        return icon(name, x, y, size, *args, **options)

    monkeypatch.setattr(ui, "rect", rect)
    monkeypatch.setattr(ui, "icon", glyph)

    def draw(*, guide=False):
        runtime.begin_frame()
        ui.design.begin_frame()
        painted.clear()
        icons.clear()
        if guide:
            from examples.ui_design.native.guide_workspace import GuideWorkspace

            from mojive.design.native.geometry import Rect

            sample = SimpleNamespace(
                guide=SimpleNamespace(ui=ui, s=ui.s, rect=lambda *r: Rect(*(v * ui.s for v in r))),
                app=SimpleNamespace(
                    workspace=workspace,
                    _draw_rename_popup=lambda: None,
                    model_edits=SimpleNamespace(active=False),
                ),
            )
            with GuideWorkspace.panel_region(sample, "Settings", 40, 40, 820, 620) as content:
                panel_host.draw_content(workspace, "Settings", ui.value_context, content)
            return next(
                window
                for window in imgui.get_current_context().windows
                if "Native specimen/Settings" in window.name
                and "Panel content" not in window.name
                and "Reference settings" not in window.name
            )
        imgui.set_next_window_pos((40 * ui.s, 40 * ui.s))
        imgui.set_next_window_size((820 * ui.s, 620 * ui.s))
        panel_host.panel(workspace, "Settings", ui.value_context)
        return imgui.internal.find_window_by_name("Settings")

    return ui, runtime, update, frame, draw, painted, icons


def _diagonal(rect, radius, runtime):
    if not runtime.spec["native"]["g3"] or radius == 0:
        return radius * (1 - sqrt(0.5))
    x, y, width, height = rect
    points = smooth_rect_points(
        x,
        y,
        x + width,
        y + height,
        radius,
        smoothing=runtime.spec["native"]["g3_smoothing"],
    )
    candidates = []
    for a, b in zip(points, (*points[1:], points[0]), strict=True):
        start, end = a[0] + a[1] - x - width - y, b[0] + b[1] - x - width - y
        if start * end <= 0 and start != end:
            t = start / (start - end)
            inset = x + width - a[0] - t * (b[0] - a[0])
            if 0 <= inset <= radius:
                candidates.append(inset)
    return min(candidates)


def _assert_hover_mesh(ui, runtime, painted):
    close = ui.hits["settings-close"]
    bounds, radius, ink, window_id = next(
        paint for paint in painted if paint[0] == pytest.approx(close)
    )
    window = next(w for w in imgui.get_current_context().windows if w.id_ == window_id)
    x, y, width, height = bounds
    assert radius <= min(width, height) / 2 - 1 + native_epsilon(*bounds)
    assert _paint_covers(window.draw_list, (x + width / 2, y + height / 2), ink)
    diagonal = _diagonal(bounds, radius, runtime)
    for right, bottom in ((False, False), (True, False), (False, True), (True, True)):
        for offset, covered in ((diagonal - 0.8, False), (diagonal + 0.8, True)):
            point = (
                x + width - offset if right else x + offset,
                y + height - offset if bottom else y + offset,
            )
            assert _paint_covers(window.draw_list, point, ink) is covered
    return radius


@pytest.mark.parametrize("g3", (False, True))
def test_real_settings_close_hover_has_equal_native_corner_offsets(header, g3):
    ui, runtime, _update, frame, draw, painted, icons = header
    spec = deepcopy(runtime.spec)
    spec["native"]["g3"] = g3
    runtime.update_spec(spec)
    for _ in range(3):
        host = frame(draw)
    x, y, width, height = ui.hits["settings-close"]
    imgui.get_io().add_mouse_pos_event(x + width / 2, y + height / 2)
    for _ in range(3):
        host = frame(draw)
    expected = native_unit(6 * ui.s) + native_unit(4 * ui.s)
    close = ui.hits["settings-close"]
    epsilon = native_epsilon(*host.pos, *host.size, *close)
    assert (close[1] - host.pos.y, host.pos.x + host.size.x - close[0] - close[2]) == pytest.approx(
        (expected,) * 2, abs=epsilon
    )
    assert icons[0][0] - host.pos.x == pytest.approx(expected, abs=epsilon)
    radius = _assert_hover_mesh(ui, runtime, painted)
    assert radius + expected == pytest.approx(host.window_rounding)
    # Header growth must leave the independently inset navigation bottom intact.
    nav = next(row for row in runtime.observed if row["key"] == "settings-navigation")
    assert host.pos.y + host.size.y - nav["hit"][1] - nav["hit"][3] == pytest.approx(
        native_unit(6 * ui.s), abs=epsilon
    )


@pytest.mark.parametrize("profile", ("tall-asymmetric", "small-close", "square-host"))
def test_authored_settings_header_keeps_the_actual_close_frame_concentric(header, profile):
    ui, runtime, update, frame, draw, painted, _icons = header
    if profile == "tall-asymmetric":
        update(
            {
                "panel": {"radius": 24},
                "panel_header": {
                    "height": 72,
                    "paddingLeft": 1,
                    "paddingTop": 0.5,
                    "paddingRight": 2,
                    "paddingBottom": 3,
                    "marginTop": 0.75,
                    "marginRight": 1,
                },
                "icon_button": {
                    "width": 24,
                    "height": 20,
                    "marginLeft": 1.25,
                    "marginTop": 0.5,
                    "marginRight": 1.5,
                    "marginBottom": 2,
                },
            }
        )
    elif profile == "small-close":
        update({"icon_button": {"width": 8, "height": 8, "iconSize": 6}})
    else:
        update({"panel": {"radius": 0}})
    for _ in range(3):
        host = frame(draw)
    x, y, width, height = ui.hits["settings-close"]
    imgui.get_io().add_mouse_pos_event(x + width / 2, y + height / 2)
    for _ in range(3):
        host = frame(draw)
    close = ui.hits["settings-close"]
    top, right = close[1] - host.pos.y, host.pos.x + host.size.x - close[0] - close[2]
    epsilon = native_epsilon(*host.pos, *host.size, *close)
    assert top == pytest.approx(right, abs=epsilon)
    assert top >= max(host.window_padding) + native_unit(4 * ui.s)
    radius = _assert_hover_mesh(ui, runtime, painted)
    assert radius == pytest.approx(max(0, host.window_rounding - top), abs=epsilon)
    record = next(row for row in runtime.observed if row["key"] == "settings-header")
    assert record["closeRect"] == pytest.approx(close)
    assert record["closeRadius"] == pytest.approx(radius)
    assert record["hit"][1] + record["hit"][3] >= close[1] + close[3]
    if profile == "tall-asymmetric":
        assert record["hit"][3] >= 72 * ui.s
        assert close[2:] == pytest.approx((24 * ui.s, 20 * ui.s))
    elif profile == "small-close":
        assert close[2:] == pytest.approx((8 * ui.s,) * 2)
        assert radius < 4 * ui.s


@pytest.mark.parametrize("radius_override", (None, 0, 24))
def test_guide_settings_region_matches_the_production_header_and_navigation(
    header, radius_override
):
    ui, runtime, update, frame, draw, painted, icons = header
    if radius_override is not None:
        update({"panel": {"radius": radius_override}})
    for _ in range(3):
        production = frame(draw)
    header_record = next(row for row in runtime.observed if row["key"] == "settings-header")
    nav_record = next(row for row in runtime.observed if row["key"] == "settings-navigation")
    expected = (
        production.window_rounding,
        tuple(production.window_padding),
        header_record["cornerInset"],
        header_record["closeRadius"],
        nav_record["radius"],
        nav_record["innerRadius"],
    )
    for _ in range(3):
        host = frame(lambda: draw(guide=True))
    x, y, width, height = ui.hits["settings-close"]
    imgui.get_io().add_mouse_pos_event(x + width / 2, y + height / 2)
    for _ in range(3):
        host = frame(lambda: draw(guide=True))
    close = ui.hits["settings-close"]
    top, right = close[1] - host.pos.y, host.pos.x + host.size.x - close[0] - close[2]
    epsilon = native_epsilon(*host.pos, *host.size, *close)
    assert top == pytest.approx(right, abs=epsilon)
    assert icons[0][0] - host.pos.x == pytest.approx(top, abs=epsilon)
    radius = _assert_hover_mesh(ui, runtime, painted)
    assert radius == pytest.approx(max(0, host.window_rounding - top), abs=epsilon)
    navigation = next(row for row in runtime.observed if row["key"] == "settings-navigation")
    assert host.window_rounding == pytest.approx(expected[0])
    assert tuple(host.window_padding) == expected[1]
    assert top == pytest.approx(expected[2], abs=epsilon)
    assert radius == pytest.approx(expected[3])
    assert navigation["radius"] == pytest.approx(expected[4])
    assert navigation["innerRadius"] == pytest.approx(expected[5])
    if radius_override is None:
        host_radius = 6 * ui.s + native_unit(8 * ui.s)
        inset = native_unit(6 * ui.s) + native_unit(4 * ui.s)
        assert host.window_rounding == pytest.approx(host_radius)
        assert top == pytest.approx(inset, abs=epsilon)
        assert radius == pytest.approx(host_radius - inset)
