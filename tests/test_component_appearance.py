"""Semantic type and icon styles reach production painters without global churn."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect, part_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.theme import THEME
from mojive.ui.viewport_widgets import draw_status
from mojive.ui.workspace import widgets
from mojive.ui.workspace.status_components import StatusGroups
from mojive.ui.workspace.typography import TextRole, Typography
from mojive.ui.workspace.viewport_controls import toolbox
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.0))
def appearance(request, tmp_path):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600 * request.param, 700 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)))
    ui.prepare_fonts("system", ("portable",))
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    ui.design = DesignPresentation(ui, runtime)

    def update(parts):
        spec = deepcopy(runtime.spec)
        spec["parts"].update(parts)
        runtime.update_spec(spec)

    def frame(paint):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Appearance", flags=imgui.WindowFlags_.no_title_bar)
        try:
            return paint()
        finally:
            imgui.end()
            imgui.render()

    frame(lambda: None)
    try:
        yield ui, runtime, update, frame
    finally:
        imgui.destroy_context(context)


def test_semantic_icon_groups_override_generic_presets_and_stay_independent(
    appearance, monkeypatch
):
    ui, _runtime, update, frame = appearance
    update(
        {
            "rail": {"iconSize": 30, "iconColor": "#ff0000"},
            "toolbox": {"iconSize": 26, "iconColor": "#00ff00"},
            "viewport_toolbar": {"iconSize": 14, "iconColor": "#0000ff"},
        }
    )
    painted = []
    original = widgets.draw_icon

    def capture(draw, name, x, y, size, tint, scale):
        painted.append((size, tint))
        return original(draw, name, x, y, size, tint, scale)

    monkeypatch.setattr(widgets, "draw_icon", capture)

    def paint():
        for index, part in enumerate(("rail", "toolbox", "viewport_toolbar")):
            with part_scope(part):
                ui.button(part, (40 + index * 60) * ui.s, 40 * ui.s, 40, 40, icon="camera")
        ui.button("unscoped", 240 * ui.s, 40 * ui.s, 40, 40, icon="camera")

    frame(paint)
    assert [size for size, _ in painted] == [30, 26, 14, 18]
    assert [tint[:3] for _, tint in painted[:3]] == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def test_actual_viewport_toolbox_uses_its_group_while_specific_leaf_wins(appearance, monkeypatch):
    ui, _runtime, update, frame = appearance
    update(
        {"toolbox": {"iconSize": 28}, "tree_row": {"iconSize": 26}, "tree_label": {"iconSize": 12}}
    )
    sizes = []
    monkeypatch.setattr(
        widgets, "draw_icon", lambda draw, name, x, y, size, tint, scale: sizes.append(size)
    )
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=SimpleNamespace(tool="select", input_bindings=ui.value_context.input_bindings),
        snap=False,
        space="World",
    )

    def paint():
        toolbox(
            workspace, (40 * ui.s, 40 * ui.s, 500 * ui.s, 400 * ui.s), 80 * ui.s, False, False, []
        )
        with part_scope("tree_row"):
            assert ui.component_style("tree_label").get("iconSize", 16) == 12

    frame(paint)
    assert sizes == [28] * 6


def test_actual_scene_and_pose_filters_keep_independent_type_icons_and_spacing(
    appearance, monkeypatch
):
    from mojive.ui.workspace import geometry_view as geometry_module
    from mojive.ui.workspace import navigation as navigation_module

    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update(
        {
            "panel_tools": {"fontSize": 13, "iconSize": 14},
            "scene_tools": {
                "fontSize": 22,
                "fontFamily": "portable",
                "iconSize": 26,
                "textColor": "#ff0000",
                "rowGap": 24,
            },
            "pose_tools": {
                "fontSize": 18,
                "fontFamily": "system",
                "iconSize": 16,
                "textColor": "#00ff00",
                "rowGap": 18,
            },
        }
    )
    navigation = navigation_module.Navigation.__new__(navigation_module.Navigation)
    navigation.workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        search_visible=True,
        query="Scene filter",
        filter="All",
        menus=SimpleNamespace(local_popup=lambda _key: nullcontext(False)),
    )
    navigation.joints = SimpleNamespace(_search="Joint filter")
    navigation.pose_search_visible = True
    navigation._pose_copy_pending = False
    navigation.scene_filters = lambda _ctx: (("all", 0),)
    ctx = SimpleNamespace(tr=str, theme=THEME, backend=None)
    monkeypatch.setattr(
        geometry_module, "geometry_view", lambda _backend: geometry_module.GeometryView.VISUAL
    )
    measurements = []
    original = geometry_module.segmented_control_width

    def measure(*args, **kwargs):
        style = ui.component_style("segment")
        assert imgui.get_font().font_id == ui.resolve_font(style).font_id
        assert imgui.get_font_size() == style.get("fontSize", 14) * ui.s
        width = original(*args, **kwargs)
        measurements.append(width)
        return width

    monkeypatch.setattr(geometry_module, "segmented_control_width", measure)

    def paint():
        ui.design.begin_frame()
        runtime.observed.clear()
        navigation._scene_controls(ctx, 20 * ui.s, 40 * ui.s, 700 * ui.s)
        bottom = navigation._pose_controls(ctx, 800 * ui.s, 40 * ui.s, 400 * ui.s)
        records = {item["key"]: item for item in ui.design.primitives.observed}
        groups = {item["key"]: item for item in runtime.observed}
        assert groups["scene-panel-tools"]["parts"] == ("panel_tools", "scene_tools")
        assert groups["pose-panel-tools"]["parts"] == ("panel_tools", "pose_tools")
        assert groups["scene-panel-tools"]["fieldRegions"]["rowGap"][3] == 24 * ui.s
        pose = records["##pose-search"]
        assert bottom - (pose["hit"][1] + pose["hit"][3]) == 18 * ui.s
        return records

    records = frame(paint)
    for key, role, font, size, icon, tint in (
        ("##search", "scene_tools", "portable", 22, 26, "#ff0000"),
        ("##pose-search", "pose_tools", "system", 18, 16, "#00ff00"),
    ):
        record = records[key]
        for field, value in (
            ("fontFamily", font),
            ("fontSize", size),
            ("iconSize", icon),
            ("textColor", tint),
        ):
            assert record["fieldValues"][field] == value
            assert record["styleOrigins"][field] == role
    pose_before = records["##pose-search"]
    update({"scene_tools": {"fontSize": 26, "iconSize": 30, "rowGap": 24}})
    edited = frame(paint)
    assert edited["##search"]["fieldValues"]["fontSize"] == 26
    assert edited["##search"]["fieldValues"]["iconSize"] == 30
    assert edited["##pose-search"]["fieldValues"] == pose_before["fieldValues"]
    assert edited["##pose-search"]["hit"] == pose_before["hit"]
    assert len(measurements) == 2 and measurements[1] > measurements[0]


@pytest.mark.parametrize("part", ("scene_tools", "pose_tools"))
def test_filter_references_dispatch_to_real_search_and_segment_controls(
    appearance, part, monkeypatch
):
    from mojive.design.native.specimens import ComponentSpecimens
    from mojive.ui.workspace.geometry_view import GEOMETRY_ICONS

    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({part: {"fontSize": 22, "fontFamily": "portable", "rowGap": 18}})
    specimens = ComponentSpecimens(runtime)
    icons = []
    original = widgets.draw_icon

    def observe(draw, name, *args, **kwargs):
        icons.append(name)
        return original(draw, name, *args, **kwargs)

    monkeypatch.setattr(widgets, "draw_icon", observe)

    def paint():
        ui.design.begin_frame()
        width, height = specimens.measure(ui, part, 600 * ui.s)
        result = specimens.paint(ui, part, Rect(20 * ui.s, 40 * ui.s, width, height))
        assert result["part"] == part
        assert len(result["children"]) == (2 if part == "scene_tools" else 1)
        assert all(
            left >= 20 * ui.s
            and top >= 40 * ui.s
            and left + child_width <= 20 * ui.s + width
            and top + child_height <= 40 * ui.s + height
            for left, top, child_width, child_height in result["children"]
        ), (width, height, result["children"])
        records = {item["key"]: item for item in ui.design.primitives.observed}
        search = records["reference-" + part]
        assert search["kind"] == "search"
        assert search["parts"] == ("input_text", "search")
        assert search["fieldValues"]["fontSize"] == 22
        assert search["fieldValues"]["fontFamily"] == "portable"
        assert search["styleOrigins"]["fontSize"] == part

    frame(paint)
    if part == "scene_tools":
        assert set(GEOMETRY_ICONS) <= set(icons)


def test_compact_joint_tab_uses_joint_identity_instead_of_pose_editor(appearance, monkeypatch):
    from mojive.ui.workspace.inspector_heading import draw_tabs

    ui, _runtime, _update, frame = appearance
    painted = []
    original = widgets.draw_icon

    def observe(draw, name, *args, **kwargs):
        painted.append(name)
        return original(draw, name, *args, **kwargs)

    monkeypatch.setattr(widgets, "draw_icon", observe)
    workspace = SimpleNamespace(ui=ui, s=ui.s, inspector_tab="Joint")
    frame(lambda: draw_tabs(workspace, 20 * ui.s, 40 * ui.s, 140 * ui.s))
    assert "joint" in painted
    assert "pose" not in painted


def test_settings_navigation_sample_uses_general_category_identity(appearance, monkeypatch):
    from mojive.design.native.specimens import ComponentSpecimens

    ui, runtime, _update, frame = appearance
    specimens, painted = ComponentSpecimens(runtime), []
    original = widgets.draw_icon

    def observe(draw, name, *args, **kwargs):
        painted.append(name)
        return original(draw, name, *args, **kwargs)

    monkeypatch.setattr(widgets, "draw_icon", observe)
    frame(
        lambda: specimens.paint(
            ui, "settings_nav", Rect(20 * ui.s, 40 * ui.s, 250 * ui.s, 300 * ui.s)
        )
    )
    assert "general" in painted
    assert "settings" not in painted


def test_component_text_measurement_and_paint_share_family_size_and_ink(appearance, monkeypatch):
    ui, _runtime, update, frame = appearance
    update(
        {"viewport_toolbar": {"fontFamily": "portable", "fontSize": 22, "textColor": "#e0204080"}}
    )
    painted = []
    original = ImguiDraw2D.text

    def capture(draw, position, color, value, **options):
        painted.append((imgui.get_font().font_id, imgui.get_font_size(), color, value))
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", capture)

    def paint():
        with part_scope("viewport_toolbar"):
            style = ui.component_style("button")
            assert style.get("fontSize", 12) == 22
            measured = ui.measure("Viewport", 22, font_resolved=True, style=style)
            imgui.push_font(ui._font_families["portable"][1][0], 22 * ui.s)
            expected = imgui.calc_text_size("Viewport").x
            imgui.pop_font()
            assert measured == expected
            ui.button("caption", 40 * ui.s, 40 * ui.s, 180, 40, label="Viewport")
        ui.component_text(
            "menu_item", Rect(40 * ui.s, 140 * ui.s, 200 * ui.s, 40 * ui.s), "Menu", widgets.TEXT
        )

    frame(paint)
    assert painted[0][:2] == (ui._font_families["portable"][1][0].font_id, 22 * ui.s)
    assert painted[0][2] == pytest.approx((224 / 255, 32 / 255, 64 / 255, 128 / 255))
    assert painted[1][0] == ui.body.font_id
    assert painted[1][1] != painted[0][1]


def test_numeric_preferences_are_defaults_below_explicit_component_and_group_styles(appearance):
    ui, _runtime, update, frame = appearance
    ui.typography = Typography(numeric=TextRole("mono", 18), output=TextRole("mono", 20))
    assert frame(lambda: ui.component_style("input_number").get("fontSize", 14)) == 18
    update({"vector_row": {"fontSize": 22}})

    def group():
        with part_scope("vector_row"):
            assert ui.component_style("input_number").get("fontSize", 14) == 22
            assert ui.metric("input_number", "fontSize", 14) == 22

    frame(group)
    update({"input_number": {"fontSize": 16}})
    assert frame(lambda: ui.component_style("input_number").get("fontSize", 14)) == 16


def test_family_switching_reuses_installed_faces_and_keeps_atlas_warm(appearance, monkeypatch):
    ui, _runtime, _update, _frame = appearance
    count = len(imgui.get_io().fonts.fonts)
    calls = []
    monkeypatch.setattr(widgets, "install_fonts", lambda *args, **kwargs: calls.append(args))
    for _ in range(5):
        ui.prepare_fonts("portable", ("system", "portable", "mono", "inherit"))
        ui.prepare_fonts("system", ("portable",))
    assert not calls
    assert len(imgui.get_io().fonts.fonts) == count


def test_status_groups_use_independent_fonts_for_measuring_and_painting(appearance, monkeypatch):
    ui, _runtime, update, frame = appearance
    update(
        {
            "status_context": {"fontFamily": "portable", "fontSize": 18, "textColor": "#ff0000"},
            "status_metrics": {"fontFamily": "mono", "fontSize": 12, "textColor": "#0000ff"},
        }
    )
    painted = []
    original = ImguiDraw2D.text

    def capture(draw, position, color, value, **options):
        painted.append((value, imgui.get_font().font_id, imgui.get_font_size(), color))
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", capture)

    def paint():
        with part_scope("statusbar"):
            draw_status(
                ImguiDraw2D(),
                (0, 100 * ui.s),
                1500 * ui.s,
                40 * ui.s,
                THEME,
                ui.s,
                selected="",
                state="paused",
                sim_time=1,
                step=1,
                metric_mode="steps",
                backend="OpenGL",
                dt=0.002,
                fps=60,
                presentation=StatusGroups(ui),
            )

    frame(paint)
    state = next(item for item in painted if item[0] == "Paused")
    fps = next(item for item in painted if "FPS" in item[0])
    assert state[1:3] == (ui._font_families["portable"][1][0].font_id, 18 * ui.s)
    assert state[3] == (1, 0, 0, 1)
    assert fps[1:3] == (ui.mono.font_id, 12 * ui.s)
    assert fps[3] == (0, 0, 1, 1)


def test_status_readout_typography_and_padding_stay_inside_their_columns(appearance, monkeypatch):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update(
        {
            "status_metrics": {
                "fontSize": 13,
                "textColor": "#80a080",
                "paddingX": 7,
                "labelGap": 1,
            },
            "status_backend": {
                "fontSize": 20,
                "fontFamily": "portable",
                "textColor": "#ff0000",
                "paddingRight": 4,
                "rightGap": 13,
            },
            "status_physics": {
                "fontSize": 15,
                "textColor": "#0000ff",
                "paddingLeft": 3,
                "rightGap": 6,
                "leftGap": 11,
            },
        }
    )
    painted = []
    original = ImguiDraw2D.text

    def capture(draw, position, color, value, **options):
        painted.append((value, imgui.get_font().font_id, imgui.get_font_size(), color))
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", capture)

    def paint():
        runtime.observed.clear()
        with part_scope("statusbar"):
            draw_status(
                ImguiDraw2D(),
                (0, 100 * ui.s),
                1500 * ui.s,
                48 * ui.s,
                THEME,
                ui.s,
                selected="",
                state="paused",
                sim_time=1,
                step=1,
                metric_mode="steps",
                backend="OpenGL",
                dt=0.002,
                fps=59,
                show_physics=True,
                physics_hz=51,
                presentation=StatusGroups(ui),
            )

    frame(paint)
    backend = next(row for row in painted if row[0] == "OpenGL")
    physics = next(row for row in painted if row[0] == "Physics")
    render = next(row for row in painted if row[0] == "Render")
    assert backend[1:] == (ui._font_families["portable"][1][0].font_id, 20 * ui.s, (1, 0, 0, 1))
    assert physics[2:] == (15 * ui.s, (0, 0, 1, 1))
    assert render[2] == 13 * ui.s
    rows = {row["parts"][0]: row for row in runtime.observed if "valueRect" in row}
    assert rows["status_backend"]["padding"] == [2 * ui.s, 0, 4 * ui.s, 0]
    assert rows["status_physics"]["padding"] == [3 * ui.s, 0, 2 * ui.s, 0]
    assert rows["status_steps"]["padding"] == [2 * ui.s, 0, 2 * ui.s, 0]
    assert rows["status_backend"]["rightGap"] == 13 * ui.s
    assert rows["status_physics"]["rightGap"] == 6 * ui.s
    assert rows["status_physics"]["leftGap"] == 11 * ui.s
    for name in ("status_physics", "status_render"):
        row = rows[name]
        assert row["unitRight"] + row["trailingReserve"] == pytest.approx(
            row["hit"][0] + row["hit"][2] - row["padding"][2]
        )
        assert row["unitX"] - row["valueTextRect"][0] - row["valueTextRect"][2] == (
            pytest.approx(row["unitGap"])
        )


def test_status_custom_widths_keep_actual_glyphs_in_their_own_readouts(appearance, monkeypatch):
    from mojive.ui.viewport_widgets.status_metrics import _StatusNumbers

    ui, _runtime, update, frame = appearance
    update(
        {
            "status_backend": {"fontFamily": "portable", "valueWidth": 72, "rightGap": 14},
            "status_steps": {"valueWidth": 26, "rightGap": 4},
            "status_delta": {"valueWidth": 24, "rightGap": 5, "unitGap": 1},
            "status_physics": {"valueWidth": 16, "paddingRight": 2, "rightGap": 7},
            "status_render": {"valueWidth": 34, "leftGap": 11, "rightGap": 12},
        }
    )
    current = []
    paint_numbers, paint_text = _StatusNumbers.paint, ImguiDraw2D.text

    def numbers(painter, field, *args):
        current.append(field)
        try:
            return paint_numbers(painter, field, *args)
        finally:
            current.pop()

    def text(draw, position, color, value, **options):
        if current:
            field = current[-1]
            left, right = position[0], position[0] + imgui.calc_text_size(value).x
            assert field.x <= left <= right <= field.x + field.width + 1e-5, (
                field.key,
                value,
                left,
                right,
                field,
            )
            if value.isdigit() or value in (".", "…", "—"):
                assert field.value_x <= left <= right <= field.value_x + field.value_width + 1e-5, (
                    field.key,
                    value,
                    left,
                    right,
                    field,
                )
        return paint_text(draw, position, color, value, **options)

    monkeypatch.setattr(_StatusNumbers, "paint", numbers)
    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def paint():
        with part_scope("statusbar"):
            draw_status(
                ImguiDraw2D(),
                (0, 100 * ui.s),
                1500 * ui.s,
                48 * ui.s,
                THEME,
                ui.s,
                selected="",
                state="paused",
                sim_time=0,
                step=0,
                metric_mode="steps",
                backend="OpenGL",
                dt=0.002,
                fps=59.0,
                show_physics=True,
                physics_hz=0,
                presentation=StatusGroups(ui),
            )

    frame(paint)


@pytest.mark.parametrize(
    "part",
    (
        "status_backend",
        "status_steps",
        "status_time",
        "status_delta",
        "status_physics",
        "status_render",
    ),
)
def test_narrow_readout_reference_shows_the_selected_item(appearance, part):
    from mojive.design.native.specimens import ComponentSpecimens

    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    specimens = ComponentSpecimens(runtime)

    def paint():
        width, height = specimens.measure(ui, part, 150 * ui.s)
        specimens.paint(ui, part, Rect(30 * ui.s, 30 * ui.s, width, height))

    frame(paint)
    assert len(specimens.last[part]["children"]) == 1
    assert any(row["key"] == "application-" + part for row in runtime.observed)


def test_warmed_status_fonts_reuse_advances_until_the_font_size_changes(appearance, monkeypatch):
    ui, _runtime, update, frame = appearance
    update(
        {
            "status_metrics": {"fontSize": 12},
            "status_backend": {"fontFamily": "portable", "fontSize": 18},
        }
    )
    widths = []

    def measure():
        groups = StatusGroups(ui)
        for part in ("status_metrics", "status_backend"):
            groups.text_budget("999.9", part=part)
            painter = groups.number_painter(ui.text_draw(ImguiDraw2D(), part))
            widths.append(painter.width("59.1"))

    frame(measure)
    calls = []
    original = imgui.calc_text_size

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(imgui, "calc_text_size", counted)
    for _ in range(3):
        frame(measure)
    assert not calls
    before = widths[-1]
    update({"status_backend": {"fontFamily": "portable", "fontSize": 22}})
    frame(measure)
    assert calls and widths[-1] > before


def test_batched_group_font_preserves_explicit_child_typography(appearance, monkeypatch):
    from mojive.design.groups import group_scope

    ui, _runtime, update, frame = appearance
    update(
        {
            "scene_tree_row": {"fontFamily": "portable", "fontSize": 13},
            "tree_label": {"fontFamily": "mono", "fontSize": 19},
        }
    )
    painted = []
    original = ImguiDraw2D.text

    def capture(draw, position, color, value, **options):
        painted.append((value, imgui.get_font().font_id, imgui.get_font_size()))
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", capture)

    def paint():
        font, size = imgui.get_font().font_id, imgui.get_font_size()
        with group_scope("Scene"), ui.component_scope("tree_row"), ui.component_font("tree_row"):
            assert imgui.get_font_size() == 13 * ui.s
            ui.button("entity-example", 30 * ui.s, 30 * ui.s, 200, 40, label="hinge_body")
        assert (imgui.get_font().font_id, imgui.get_font_size()) == (font, size)

    frame(paint)
    assert painted[0] == ("hinge_body", ui.mono.font_id, 19 * ui.s)


def test_popup_typography_is_independent_of_its_trigger_group(appearance):
    from mojive.ui.workspace.popup_style import menu_choice_width, menu_popup_style

    ui, _runtime, update, frame = appearance
    update({"viewport_toolbar": {"fontFamily": "mono", "fontSize": 30, "iconSize": 30}})

    def paint():
        baseline = menu_choice_width(ui, ("Short", "Longer option"), 0)
        with part_scope("viewport_toolbar"):
            assert ui.component_style("button").get("fontSize", 12) == 30
            assert menu_choice_width(ui, ("Short", "Longer option"), 0) == baseline
            with menu_popup_style(ui):
                style = ui.component_style("menu_item")
                assert style.get("fontSize", 14) == 14
                assert ui.resolve_font(style).font_id == ui.body.font_id
            assert ui.component_style("button").get("fontSize", 12) == 30

    frame(paint)


def test_actual_orientation_labels_use_the_navigation_widget_type_style(appearance, monkeypatch):
    from mojive.types import CameraView
    from mojive.ui.workspace.viewcube import WorkspaceViewCube

    ui, _runtime, update, frame = appearance
    update({"nav_widget": {"fontFamily": "portable", "fontSize": 16, "textColor": "#ffe010"}})
    cube = WorkspaceViewCube(1.2)
    cube.design_ui = ui
    painted = []
    original = ImguiDraw2D.centered_label

    def capture(draw, value, position, color, max_width):
        painted.append((value, imgui.get_font().font_id, imgui.get_font_size(), color))
        return original(draw, value, position, color, max_width)

    monkeypatch.setattr(ImguiDraw2D, "centered_label", capture)

    def paint():
        cube.update(CameraView(), (0, 0, 800 * ui.s, 600 * ui.s), (-1, -1), ui.s)
        cube.draw(ImguiDraw2D())

    frame(paint)
    assert {item[0] for item in painted} == {"X", "Y", "Z"}
    assert all(item[1] == ui._font_families["portable"][1][0].font_id for item in painted)
    assert all(item[2] == 16 * ui.s for item in painted)
    assert all(item[3][:3] == pytest.approx((1, 224 / 255, 16 / 255)) for item in painted)


@pytest.mark.parametrize("part", ("status_context", "statusbar"))
def test_status_icon_settings_reach_the_state_dot_and_expand_its_box(appearance, monkeypatch, part):
    ui, _runtime, update, frame = appearance
    update({part: {"iconSize": 30, "iconColor": "#ff00ff"}})
    dots = []
    monkeypatch.setattr(
        ImguiDraw2D,
        "circle_filled",
        lambda draw, center, radius, color, **options: dots.append((radius, color)),
    )

    def paint():
        with part_scope("statusbar"):
            groups = StatusGroups(ui)
            assert groups.minimum_height(20 * ui.s) >= 30 * ui.s
            draw_status(
                ImguiDraw2D(),
                (0, 100 * ui.s),
                1500 * ui.s,
                50 * ui.s,
                THEME,
                ui.s,
                selected="",
                state="paused",
                sim_time=1,
                step=1,
                metric_mode="steps",
                backend="OpenGL",
                dt=0.002,
                fps=60,
                presentation=groups,
            )

    frame(paint)
    assert dots == [(15 * ui.s, (1, 0, 1, 1))]


def test_timeline_lane_icons_use_their_exact_label_style(appearance, monkeypatch):
    from mojive.ui.workspace.timeline_track import NativeTimelineTrack

    ui, _runtime, update, frame = appearance
    update({"lane_label": {"iconSize": 20, "iconColor": "#ff00ff"}})
    icons = []
    monkeypatch.setattr(
        widgets,
        "draw_icon",
        lambda draw, name, x, y, size, tint, scale: icons.append((name, size, tint)),
    )
    track = NativeTimelineTrack(SimpleNamespace(ui=ui, s=ui.s))
    for name in ("_ruler", "_range", "_take", "_keys", "_snapshots", "_playhead"):
        monkeypatch.setattr(track, name, lambda *args: None)
    layout = SimpleNamespace(
        lo=(20 * ui.s, 40 * ui.s),
        hi=(900 * ui.s, 250 * ui.s),
        channel_width=150 * ui.s,
        channel_right=170 * ui.s,
        track_left=182 * ui.s,
        ruler_bottom=80 * ui.s,
        row_height=40 * ui.s,
        lane_gap=2 * ui.s,
        width=880 * ui.s,
        height=210 * ui.s,
    )
    frame(
        lambda: track._paint(
            SimpleNamespace(edit_lane="model"),
            SimpleNamespace(style_scale=ui.s, theme=THEME),
            layout,
            None,
            None,
            SimpleNamespace(snapshot_id=-1, keyframe_id=-1),
            {},
            (),
        )
    )
    assert icons == [(name, 20, (255, 0, 255, 255)) for name in ("diamond", "record", "snapshot")]


@pytest.mark.parametrize("dialog", ("About", "Interaction guide"))
def test_native_dialog_wrapped_body_uses_its_exact_font_and_ink(appearance, monkeypatch, dialog):
    from mojive.ui.workspace.menu_dialogs import dialogs
    from mojive.ui.workspace.popup_style import menu_popup_style

    ui, _runtime, update, frame = appearance
    update({"dialog_body": {"fontFamily": "portable", "fontSize": 18, "textColor": "#ff00ff"}})
    ui.translate = lambda value: (
        "Drag an empty area to orbit the current camera around the selected scene object"
        if value == "Drag an empty area"
        else value
    )
    wrapped = []
    original = imgui.text_wrapped

    def text(value):
        tint = imgui.get_style().color_(imgui.Col_.text)
        wrapped.append(
            (imgui.get_font().font_id, imgui.get_font_size(), (tint.x, tint.y, tint.z, tint.w))
        )
        return original(value)

    monkeypatch.setattr(imgui, "text_wrapped", text)
    menu = SimpleNamespace(
        ui=ui,
        s=ui.s,
        dialog_pending=dialog,
        dialog=None,
        style=lambda controls=False: menu_popup_style(ui, controls, part="dialog"),
        shadow_and_record=lambda *args: None,
        workspace=SimpleNamespace(
            app=SimpleNamespace(
                localizer=SimpleNamespace(text=str),
                input_bindings=ui.value_context.input_bindings,
            )
        ),
    )
    frame(lambda: dialogs(menu))
    assert wrapped
    assert all(
        item[:2] == (ui._font_families["portable"][1][0].font_id, 18 * ui.s) for item in wrapped
    )
    assert all(item[2] == (1, 0, 1, 1) for item in wrapped)


def test_status_mouse_icons_scale_uniformly_and_measure_without_resizing_the_text(
    appearance, monkeypatch
):
    from mojive.ui.viewport_widgets import hints
    from mojive.ui.viewport_widgets.model import ToolHint

    ui, _runtime, update, frame = appearance
    shapes, captions, bounds = [], [], []
    original_shape, original_text = hints._paint_mouse_shape, ImguiDraw2D.text

    def shape(
        draw, x, y, width, height, button, stroke, pixel_size, geometry, smoothing, shell, fill
    ):
        shapes.append((width, height, stroke, shell, fill))
        return original_shape(
            draw, x, y, width, height, button, stroke, pixel_size, geometry, smoothing, shell, fill
        )

    def text(draw, position, color, value, **options):
        if value == "Push":
            captions.append((imgui.get_font().font_id, imgui.get_font_size(), color))
        return original_text(draw, position, color, value, **options)

    monkeypatch.setattr(hints, "_paint_mouse_shape", shape)
    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def paint():
        groups = StatusGroups(ui)
        record = groups.record

        def capture(part, rect, fields=()):
            if part == "status_hints":
                bounds.append(rect)
            return record(part, rect, fields)

        groups.record = capture
        draw_status(
            ImguiDraw2D(),
            (0, 100 * ui.s),
            1500 * ui.s,
            70 * ui.s,
            THEME,
            ui.s,
            selected="",
            state="paused",
            sim_time=1,
            step=1,
            metric_mode="steps",
            backend="OpenGL",
            dt=0.002,
            fps=60,
            tool_hints=(ToolHint("mouse", "left", "Push", suffix="×2"),),
            presentation=groups,
        )

    with part_scope("statusbar"):
        frame(paint)
    update({"status_hints": {"iconSize": 36, "iconColor": "#ff00ff"}})
    with part_scope("statusbar"):
        frame(paint)
    assert shapes[0][1] == hints.OVERLAY_GEOMETRY.hint_control_height * ui.s
    assert shapes[1][:3] == pytest.approx(tuple(value * 2 for value in shapes[0][:3]))
    assert shapes[1][3:] == ((1, 0, 1, 1), (1, 0, 1, 1))
    assert captions[1] == captions[0]
    assert bounds[1][2] - bounds[0][2] == pytest.approx(shapes[0][0] + 5 * ui.s)
    assert bounds[1][3] >= 36 * ui.s


def test_component_text_drawer_reuses_the_current_font_without_stack_churn(appearance, monkeypatch):
    ui, _runtime, update, frame = appearance
    calls = []
    original = imgui.push_font

    def push(font, size):
        calls.append((font.font_id, size))
        return original(font, size)

    def exercise():
        painter = ui.text_draw(ImguiDraw2D(), "status_context")
        painter.text_size("H")
        painter.text_ink_bounds("H")
        painter.text((40 * ui.s, 50 * ui.s), (1, 1, 1, 1), "H")
        painter.centered_label("X", (200 * ui.s, 100 * ui.s), (1, 1, 1, 1), 100 * ui.s)

    def paint():
        original(ui.body, 14 * ui.s)
        try:
            monkeypatch.setattr(imgui, "push_font", push)
            exercise()
            assert not calls
            update({"status_context": {"fontFamily": "mono", "fontSize": 18}})
            exercise()
            assert calls == [(ui.mono.font_id, 18 * ui.s)] * 4
            assert imgui.get_font().font_id == ui.body.font_id
            assert imgui.get_font_size() == 14 * ui.s
        finally:
            imgui.pop_font()

    frame(paint)


def test_segment_buttons_have_owned_dimensions_and_text_padding(appearance):
    from mojive.design.groups import group_scope

    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update(
        {
            "scene_segment_button": {
                "widthMode": "fixed",
                "width": 104,
                "height": 40,
                "paddingLeft": 11,
                "paddingRight": 13,
                "paddingTop": 3,
                "paddingBottom": 5,
            },
            "settings_general_segment_button": {
                "widthMode": "fixed",
                "width": 88,
                "height": 32,
                "paddingLeft": 7,
                "paddingRight": 9,
            },
        }
    )

    def paint():
        ui.design.begin_frame()
        for owner, key, y in (("Scene", "scene", 40), ("Settings/General", "settings", 120)):
            with group_scope(owner):
                ui.segmented(
                    key, 20 * ui.s, y * ui.s, 600, ("One", "Two", "Three"), 0, layout="content"
                )
        return list(ui.design.primitives.observed)

    records = frame(paint)
    for key, part, width, height, left, right in (
        ("scene", "scene_segment_button", 104, 40, 11, 13),
        ("settings", "settings_general_segment_button", 88, 32, 7, 9),
    ):
        group = [row for row in records if row["key"].startswith(key + "-")]
        assert len(group) == 3
        for row in group:
            assert row["parts"][-1] == part
            assert row["hit"][2:] == pytest.approx((width * ui.s, height * ui.s))
            assert row["content"][0] - row["hit"][0] == pytest.approx(left * ui.s)
            assert row["hit"][0] + row["hit"][2] - row["content"][0] - row["content"][
                2
            ] == pytest.approx(right * ui.s)


def test_segment_reflow_accounts_for_group_icon_size_and_text_insets(appearance):
    from mojive.design.groups import group_scope

    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update(
        {
            "scene_segment_button": {
                "iconSize": 40,
                "fontSize": 18,
                "gap": 12,
                "paddingLeft": 20,
                "paddingRight": 16,
                "paddingTop": 5,
                "paddingBottom": 7,
            }
        }
    )

    def paint():
        ui.design.begin_frame()
        with group_scope("Scene"):
            assert ui.segment_height() == pytest.approx(52 * ui.s)
            ui.segmented(
                "reflow",
                20 * ui.s,
                40 * ui.s,
                220,
                ("One", "Two", "Three"),
                0,
                icons=("camera", "camera", "camera"),
                layout="content",
            )
        return list(ui.design.primitives.observed)

    records = frame(paint)
    assert len(records) == 3
    for index, row in enumerate(records):
        assert row["parts"][-1] == "scene_segment_button"
        assert row["hit"] == pytest.approx(
            (20 * ui.s, (40 + index * 52) * ui.s, 220 * ui.s, 52 * ui.s)
        )
        assert row["content"][3] == pytest.approx(40 * ui.s)


def test_custom_control_observations_use_the_same_owned_style_as_painting(appearance):
    from mojive.design.groups import group_scope

    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"scene_type_filters_filter": {"height": 36, "paddingLeft": 11}})

    def paint():
        runtime.observed.clear()
        with group_scope("Scene"), part_scope("type_filters"):
            style = ui.component_style("filter")
            box = ui.component_box(
                "filter", 20 * ui.s, 40 * ui.s, 160 * ui.s, 60 * ui.s, key="pill"
            )
            assert runtime.observed[-1]["parts"] == style.part_ids
            assert runtime.observed[-1]["parts"][-1] == "scene_type_filters_filter"
            assert box.frame.height == pytest.approx(36 * ui.s)
            assert box.padding[0] == pytest.approx(11 * ui.s)

    frame(paint)


@pytest.mark.parametrize("size", (10, 12, 18))
def test_scene_filter_counts_remain_visible_after_switching_to_mono(appearance, monkeypatch, size):
    from mojive.design.groups import group_scope
    from mojive.ui.workspace import geometry_view as geometry_module
    from mojive.ui.workspace import navigation as navigation_module

    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"scene_type_filters_filter": {"fontFamily": "mono", "fontSize": size}})
    navigation = navigation_module.Navigation.__new__(navigation_module.Navigation)
    navigation.workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        search_visible=False,
        query="",
        filter="All",
    )
    navigation.scene_filters = lambda _ctx: (("all", 52), ("link", 11), ("geom", 20), ("camera", 1))
    ctx = SimpleNamespace(tr=str, backend=None, theme=THEME)
    monkeypatch.setattr(
        geometry_module, "geometry_view", lambda _backend: geometry_module.GeometryView.VISUAL
    )
    painted = []
    original = ImguiDraw2D.text

    def text(draw, pos, ink, value, **kwargs):
        if imgui.get_font().font_id == ui.mono.font_id:
            painted.append(value)
        return original(draw, pos, ink, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def paint():
        with group_scope("Scene"):
            navigation._scene_controls(ctx, 20 * ui.s, 40 * ui.s, 700 * ui.s)

    frame(paint)
    assert all(str(count) in painted for count in (52, 11, 20, 1)), painted


def test_material_cards_and_list_own_their_padding_and_spacing(appearance):
    from mojive.design.groups import group_scope
    from mojive.ui.workspace.material_library import draw_material_library

    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    global_material = deepcopy(runtime.spec["parts"].get("material"))
    workspace = SimpleNamespace(ui=ui, s=ui.s, material=0, app=SimpleNamespace(entity=None))

    def paint():
        runtime.observed.clear()
        with group_scope("Assets"):
            draw_material_library(workspace, 20 * ui.s, 40 * ui.s, 500 * ui.s)
        return {row["key"]: row for row in runtime.observed}

    def insets(row):
        x, y, width, height = row["hit"]
        cx, cy, cw, ch = row["content"]
        return cx - x, cy - y, x + width - cx - cw, y + height - cy - ch

    baseline = frame(paint)
    for index in range(4):
        card = baseline[f"material-card-{index}"]
        assert insets(card) == pytest.approx(tuple(v * ui.s for v in (12, 8, 12, 8)))
        assert card["fields"][0][0] - card["hit"][0] == pytest.approx(12 * ui.s)
        assert card["fields"][1][0] - Rect(*card["fields"][0]).right == pytest.approx(12 * ui.s)
    update(
        {
            "assets_material": {
                "paddingLeft": 20,
                "paddingRight": 24,
                "paddingTop": 10,
                "paddingBottom": 12,
                "columnGap": 18,
                "swatchSize": 28,
            },
            "assets_material_list": {
                "paddingLeft": 10,
                "paddingRight": 14,
                "paddingTop": 6,
                "paddingBottom": 8,
                "rowGap": 14,
                "actionGap": 20,
            },
        }
    )
    changed = frame(paint)
    listing = changed["material-library"]
    assert listing["parts"][-1] == "assets_material_list"
    assert insets(listing) == pytest.approx(tuple(v * ui.s for v in (10, 6, 14, 8)))
    for index in range(4):
        card = changed[f"material-card-{index}"]
        assert card["parts"][-1] == "assets_material"
        assert insets(card) == pytest.approx(tuple(v * ui.s for v in (20, 10, 24, 12)))
        assert card["fields"][0][2:] == pytest.approx((28 * ui.s, 28 * ui.s))
        assert card["fields"][1][0] - Rect(*card["fields"][0]).right == pytest.approx(18 * ui.s)
        assert card["hit"][0] == listing["content"][0]
        assert card["hit"][2] == listing["content"][2]
        if index:
            previous = changed[f"material-card-{index - 1}"]["hit"]
            assert card["hit"][1] - previous[1] - previous[3] == pytest.approx(14 * ui.s)
    assert ui.hits["apply-material"][1] - listing["outer"][1] - listing["outer"][
        3
    ] == pytest.approx(20 * ui.s)
    assert runtime.spec["parts"].get("material") == global_material


@pytest.mark.parametrize("horizontal,wrapped", ((False, False), (True, False), (True, True)))
def test_tool_column_divider_reserves_space_in_every_orientation(appearance, horizontal, wrapped):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        snap=False,
        space="World",
        app=SimpleNamespace(tool="select", input_bindings=ui.value_context.input_bindings),
    )

    def paint():
        runtime.observed.clear()
        toolbox(
            workspace,
            (20 * ui.s, 40 * ui.s, 900 * ui.s, 600 * ui.s),
            60 * ui.s,
            horizontal,
            wrapped,
            [],
        )
        return next(row for row in runtime.observed if row["key"] == "viewport-toolbox")

    def assert_divider(row, thickness, gap):
        before, after = (Rect(*row["fields"][n]) for n in (3, 4))
        divider = Rect(*row["separators"][0])
        if horizontal:
            assert divider.width == pytest.approx(thickness * ui.s)
            assert divider.x - before.right == pytest.approx(gap * ui.s)
            assert after.x - divider.right == pytest.approx(gap * ui.s)
        else:
            assert divider.height == pytest.approx(thickness * ui.s)
            assert divider.y - before.bottom == pytest.approx(gap * ui.s)
            assert after.y - divider.bottom == pytest.approx(gap * ui.s)
        frame = Rect(*row["hit"])
        assert frame.x <= divider.x <= divider.right <= frame.right
        assert frame.y <= divider.y <= divider.bottom <= frame.bottom

    assert_divider(frame(paint), 1, 6)
    with part_scope("transport"):
        transport_before = dict(ui.component_style("transport").values)
    update(
        {
            "toolbox": {
                "separatorThickness": 2,
                "separatorInset": 3,
                "separatorGap": 10,
                "separatorColor": "#ee2211ff",
            }
        }
    )
    assert_divider(frame(paint), 2, 10)
    with part_scope("transport"):
        assert dict(ui.component_style("transport").values) == transport_before
    update({"toolbox": {"separatorThickness": 0}})
    hidden = frame(paint)
    assert hidden["separators"] == []
    before, after = (Rect(*hidden["fields"][n]) for n in (3, 4))
    assert (after.x - before.right if horizontal else after.y - before.bottom) == pytest.approx(
        4 * ui.s
    )


def test_viewport_camera_group_keeps_one_icon_canvas_with_and_without_caption(
    appearance, monkeypatch
):
    from mojive.ui.workspace.viewport_controls import camera_controls

    ui, _runtime, update, frame = appearance
    painted = {}
    original = widgets.draw_icon

    def capture(draw, name, x, y, size, tint, scale):
        painted[name] = size
        return original(draw, name, x, y, size, tint, scale)

    monkeypatch.setattr(widgets, "draw_icon", capture)
    workspace = SimpleNamespace(
        ui=ui,
        s=ui.s,
        app=SimpleNamespace(camera=SimpleNamespace(orthographic=False)),
        menus=SimpleNamespace(popup=lambda *args: None),
        activate=lambda name: None,
    )
    for width in (800, 220):
        frame(
            lambda width=width: camera_controls(
                workspace, (0, 20 * ui.s, width * ui.s, 400 * ui.s), []
            )
        )
        assert painted["perspective"] == painted["shading"] == 18
        if width == 800:
            assert painted["layers"] == painted["perspective"]
    update({"viewport_toolbar": {"iconSize": 20}})
    frame(lambda: camera_controls(workspace, (0, 20 * ui.s, 800 * ui.s, 400 * ui.s), []))
    assert painted["perspective"] == painted["shading"] == painted["layers"] == 20


def test_scene_and_content_settings_share_four_icon_choices_and_backend_state(
    appearance, monkeypatch
):
    from mojive.render.geometry import GeometryView
    from mojive.ui.workspace import geometry_view as selector
    from mojive.ui.workspace.settings_panel import WorkspaceSettings

    ui, _runtime, _update, frame = appearance
    backend = SimpleNamespace(view=GeometryView.DEFAULT)
    ctx = SimpleNamespace(backend=backend, tr=str)
    monkeypatch.setattr(selector, "geometry_view", lambda owner: owner.view)
    monkeypatch.setattr(
        selector, "set_geometry_view", lambda owner, view: setattr(owner, "view", view)
    )
    calls = []
    original = ui.segmented

    def draw(*args, **kwargs):
        calls.append((args[0], args[4], args[5], kwargs["icons"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(ui, "segmented", draw)
    panel = WorkspaceSettings(SimpleNamespace(ui=ui, s=ui.s))

    def paint():
        selector.geometry_selector(ui, ctx, 20 * ui.s, 30 * ui.s, 360 * ui.s, key="scene-geometry")
        imgui.set_cursor_screen_pos((20 * ui.s, 100 * ui.s))
        imgui.set_next_item_width(360 * ui.s)
        panel._geometry_view(ctx)

    frame(paint)
    assert (
        calls[-2][1:]
        == calls[-1][1:]
        == (
            ("Default", "Visual", "Collision", "Both"),
            0,
            ("geometry-default", "visual", "collision", "both"),
        )
    )
    assert "settings-geometry-default" in ui.hits
    x, y, width, height = ui.hits["settings-geometry-collision"]
    io = imgui.get_io()
    io.add_mouse_pos_event(x + width / 2, y + height / 2)
    frame(paint)
    io.add_mouse_button_event(0, True)
    frame(paint)
    io.add_mouse_button_event(0, False)
    frame(paint)
    frame(paint)
    assert backend.view is GeometryView.COLLISION
    assert calls[-2][2] == calls[-1][2] == 2
    x, y, width, height = ui.hits["settings-geometry-default"]
    io.add_mouse_pos_event(x + width / 2, y + height / 2)
    frame(paint)
    io.add_mouse_button_event(0, True)
    frame(paint)
    io.add_mouse_button_event(0, False)
    frame(paint)
    frame(paint)
    assert backend.view is GeometryView.DEFAULT
    assert calls[-2][2] == calls[-1][2] == 0
