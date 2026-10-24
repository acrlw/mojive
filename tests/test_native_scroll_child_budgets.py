"""Owned scroll children budget fields against their own native paintable edge."""

from math import floor
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.geometry import Rect
from mojive.ui.workspace import inspector as inspector_module
from mojive.ui.workspace import inspector_heading, settings_layout
from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.navigation import Navigation
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1.2, 1.25, 2.25), indirect=True)


def _probe_field(ui, observations, *, width=None):
    window = imgui.internal.get_current_window()
    p = imgui.get_cursor_screen_pos()
    available = imgui.get_content_region_avail().x
    width = available if width is None else width
    ui.input("probe-field", p.x, p.y, width, 0.0)
    field = ui.hits["probe-field-frame"]
    observations.append(
        (
            field[0] + field[2],
            floor(window.inner_rect.max.x - window.window_border_size / 2),
            available,
            window.inner_clip_rect.max.x,
        )
    )
    imgui.dummy((0, 600 * ui.s))


@pytest.mark.parametrize(
    "kind", ("inspector", "settings", "scene", "navigation", "navigation_narrow")
)
@pytest.mark.parametrize("ancestor_clip", (False, True))
@pytest.mark.parametrize("width", (240, 241))
def test_owned_children_keep_fields_inside_own_clip_without_ancestor_reflow(
    appearance, monkeypatch, kind, ancestor_clip, width
):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    observations = []
    workspace = SimpleNamespace(ui=ui, s=ui.s)
    if kind == "inspector":
        entry = {"identity": "entity", "node_id": 1}
        workspace.app = SimpleNamespace(entity=entry, initial_entities=())
        workspace.pinned = None
        workspace._panel_context = SimpleNamespace()
        workspace.inspector_tab = "Properties"
        monkeypatch.setattr(inspector_heading, "draw_heading", lambda *_args: 40 * ui.s)
        monkeypatch.setattr(
            inspector_module,
            "draw_component",
            lambda *_args: _probe_field(ui, observations) or True,
        )

        def draw():
            inspector_module.inspector(workspace, 30 * ui.s, 30 * ui.s, width * ui.s, 240 * ui.s)

        owner = "Inspector"
    elif kind == "settings":
        panel = SimpleNamespace(
            workspace=workspace, _category="General", _control_padding=lambda: None
        )
        for category in (
            "_general",
            "_camera_navigation",
            "_interaction",
            "_rendering",
            "_recording",
            "_mujoco_visuals",
        ):
            setattr(panel, category, lambda _ctx: _probe_field(ui, observations))

        def draw():
            settings_layout._draw_page(
                panel,
                SimpleNamespace(tr=str),
                Rect(30 * ui.s, 30 * ui.s, width * ui.s, 240 * ui.s),
                (),
            )

        owner = "Settings/General"
    elif kind == "scene":
        workspace.query = workspace.filter = ""
        navigation = Navigation.__new__(Navigation)
        navigation.workspace = workspace
        navigation._scene_filter_token = navigation._scroll_scene_node = None
        monkeypatch.setattr(navigation, "_scene_controls", lambda _ctx, _x, y, _w: y)
        monkeypatch.setattr(
            navigation, "_scene_row_geometry", lambda: (28 * ui.s, (), (0, 0, 0, 0))
        )
        monkeypatch.setattr(navigation, "scene_rows", lambda _ctx: [(None, 0, True)] * 30)
        monkeypatch.setattr(
            navigation,
            "_draw_scene_row",
            lambda _ctx, _node, _depth, _leaf, budget, _geometry: _probe_field(
                ui, observations, width=budget
            ),
        )
        monkeypatch.setattr(
            "mojive.ui.workspace.navigation.publish_focus_item_hint", lambda _ctx: None
        )

        def draw():
            navigation.draw_scene(None, 30 * ui.s, 30 * ui.s, width * ui.s, 240 * ui.s)

        owner = "Scene"
    else:
        panel = SimpleNamespace(workspace=workspace, _category="General")

        def draw():
            settings_layout.draw_navigation(
                panel,
                SimpleNamespace(tr=str),
                Rect(
                    30 * ui.s,
                    30 * ui.s,
                    (width + (400 if kind == "navigation" else 0)) * ui.s,
                    (600 if kind == "navigation_narrow" else 120) * ui.s,
                ),
                kind == "navigation_narrow",
            )
            window = next(
                w
                for w in imgui.get_current_context().windows
                if "Reference settings navigation" in w.name
            )
            own_right = floor(window.inner_rect.max.x - window.window_border_size / 2)
            right = max(
                ui.hits["settings-" + c][0] + ui.hits["settings-" + c][2]
                for c in (
                    "General",
                    "Camera",
                    "Interaction",
                    "Rendering",
                    "Recording",
                    "MuJoCo Visuals",
                )
            )
            observations.append(
                (
                    right,
                    own_right,
                    window.work_rect.max.x - window.work_rect.min.x,
                    window.inner_clip_rect.max.x,
                )
            )

        owner = "Settings"

    def paint():
        runtime.begin_frame()
        if ancestor_clip:
            imgui.push_clip_rect((0, 0), (80 * ui.s, 700 * ui.s), True)
        with group_scope(owner):
            draw()
        if ancestor_clip:
            imgui.pop_clip_rect()

    for _ in range(5):
        frame(paint)
    if kind == "navigation":
        for row in ui.design.primitives.observed:
            if row["key"].startswith("settings-"):
                assert not row.get("clippedLabel", False)
    for right, own_right, _budget, clip_right in observations:
        assert right <= own_right + native_epsilon(right, own_right), (kind, ui.s, right, own_right)
        if ancestor_clip:
            assert own_right > clip_right


@pytest.mark.parametrize("authored", ("default", "radius", "asymmetric", "row-margin"))
def test_navigation_corners_follow_actual_insets_without_losing_first_row_margins(
    appearance, authored
):
    from mojive.ui.workspace.menu_geometry import native_unit

    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    if authored == "radius":
        update({"settings_nav": {"radius": 23}})
    elif authored == "asymmetric":
        update(
            {
                "settings_nav": {
                    "paddingLeft": 7,
                    "paddingTop": 9,
                    "paddingRight": 13,
                    "paddingBottom": 11,
                }
            }
        )
    elif authored == "row-margin":
        update({"settings_nav_button": {"marginLeft": 4, "marginTop": 2}})
    panel = SimpleNamespace(workspace=SimpleNamespace(ui=ui, s=ui.s), _category="General")

    def draw():
        runtime.begin_frame()
        with group_scope("Settings"):
            settings_layout.draw_navigation(
                panel,
                SimpleNamespace(tr=str),
                Rect(30 * ui.s, 30 * ui.s, 640 * ui.s, 600 * ui.s),
                False,
            )

    for _ in range(4):
        frame(draw)
    container = next(row for row in runtime.observed if row["key"] == "settings-navigation")
    first = ui.hits["settings-General"]
    inset = native_unit((13 if authored == "asymmetric" else 4) * ui.s)
    assert container["padding"] == [inset] * 4
    row_margin = tuple(value * ui.s for value in (4, 2)) if authored == "row-margin" else (0, 0)
    assert first[0] - container["hit"][0] == pytest.approx(inset + row_margin[0])
    assert first[1] - container["hit"][1] == pytest.approx(inset + row_margin[1])
    outer_radius = 23 * ui.s if authored == "radius" else 4 * ui.s + inset
    assert container["radius"] == pytest.approx(outer_radius)
    assert container["innerRadius"] == pytest.approx(max(0, outer_radius - inset))
    assert first[0] + first[2] <= container["nativePaintableRight"]
