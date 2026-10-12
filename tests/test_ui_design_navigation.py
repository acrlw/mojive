"""Scene and joint navigation through real Session state and CPU ImGui input."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from examples.ui_design.native.docking_study import Study
from examples.ui_design.native.navigation import (
    GEOMETRY_ICONS,
    GEOMETRY_VIEWS,
    NODE_ICONS,
    Navigation,
)
from examples.ui_design.native.reference_menus import Menus
from examples.ui_design.native.scene_preview import Preview, ReferencePanels
from examples.ui_design.native.widgets import UI
from imgui_bundle import imgui

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds, NodeType
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.render.backend import NullBackend, RenderFlag
from mojive.render.geometry import GeometryView, geometry_view, set_geometry_flags
from mojive.scene.assets import resolve
from mojive.session import Session
from mojive.ui.gizmo import ObjectGizmo
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext, PanelManager, state_vector_text
from mojive.ui.theme import THEME


class GeometryBackend(NullBackend):
    def __init__(self):
        super().__init__()
        self.caps = replace(
            self.caps,
            render_flags=frozenset((RenderFlag.VISUAL_GEOMETRY, RenderFlag.COLLISION_GEOMETRY)),
        )
        self.set_geometry_view(GeometryView.VISUAL)

    def set_geometry_view(self, view):
        return set_geometry_flags(self._flags, view)


def _headless_preview(session):
    app = Preview.__new__(Preview)
    app.session = session
    app.theme = THEME
    app.entities, app._entity_cache = [], {}
    app.document = {"pose": {"hinge": 0, "ball": 0, "slide": 0}}
    app.panels = PanelManager(builtin_ids=("hierarchy", "joints"))
    app._model_camera_id = -1
    app._pending_node_focus_id = app._pending_joint_focus_id = None
    return app


def test_static_scene_projection_preserves_real_node_types_and_selection():
    scene = Scene()
    scene.box(name="body")
    session = Session(StaticSceneAdapter(scene))
    try:
        app = _headless_preview(session)
        app.refresh_document()
        assert {entry["node_id"] for entry in app.entities} == {
            node.node_id for node in session.nodes
        }
        for node in session.nodes:
            assert session.submit(cmd.SelectNode(node.node_id)).ok
            entry = app.entity
            assert entry["type"] == node.type.value
            assert entry["icon"] == NODE_ICONS[node.type.value]
    finally:
        session.release()


@pytest.fixture(params=(1.0, 2.25))
def navigation(request, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (320 * request.param, 760 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    if "language" in request.node.callspec.params:
        report = SimpleNamespace(cjk_path=None)
        if request.node.callspec.params["language"] == "zh_CN":
            from mojive.text.sources import resolve_font_sources

            sources = resolve_font_sources(allow_download=False)
            if len(sources) < 2:
                pytest.skip("CJK font is unavailable")
            report.cjk_path, report.cjk_index = sources[1].path, sources[1].index
        ui = UI(SimpleNamespace(style_scale=request.param, font_report=report))
        imgui.get_style().window_min_size = (32 * ui.s, 32 * ui.s)
    else:
        ui = UI.__new__(UI)
        ui.s = request.param
        ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    ui.translate = str
    ui.hits, ui.numeric_edits = {}, {}
    session = Session(WorkspaceAdapter(MuJoCoAdapter(resolve("joint_types")), Scene()))
    session.submit(cmd.Pause())
    app = _headless_preview(session)
    app.model_edits = SimpleNamespace(_checkpoint=None)
    backend = GeometryBackend()
    ctx = PanelContext(
        session,
        backend,
        style_scale=ui.s,
        focus_node=app.request_node_focus,
        focus_joint=app.request_joint_focus,
        gizmo=ObjectGizmo(mode="translate", enabled=False),
    )
    ui.value_context = ctx
    study = Study.__new__(Study)
    study._numeric_edit = None
    study.s, study.ui, study.preview = ui.s, ui, app
    study.query, study.filter, study.search_visible = "", "All", False
    study.header_actions = {"Pose": {"search", "sort", "copy"}}
    study.open, study._panel_context = {"Pose": True}, ctx
    study.menus = Menus(study, ())
    controls, copied, children = {}, [], {}

    def record_control(method, prefix):
        original = getattr(imgui, method)

        def draw(identifier, *args, **kwargs):
            result = original(identifier, *args, **kwargs)
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            controls[prefix + identifier] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            return result

        monkeypatch.setattr(imgui, method, draw)

    for method, prefix in (
        ("input_text_with_hint", "input-"),
        ("menu_item", "menu-"),
        ("checkbox", "checkbox-"),
        ("begin_menu", "submenu-"),
    ):
        record_control(method, prefix)
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    original_child = imgui.begin_child

    def begin_child(identifier, *args, **kwargs):
        result = original_child(identifier, *args, **kwargs)
        children[identifier] = imgui.internal.get_current_window()
        return result

    monkeypatch.setattr(imgui, "begin_child", begin_child)
    state = SimpleNamespace(
        nav=Navigation(study),
        study=study,
        session=session,
        ctx=ctx,
        app=app,
        mode="scene",
        node_id=None,
        window_size=None,
        show_pose_header=False,
        controls=controls,
        copied=copied,
        children=children,
        panel_name="Pose",
    )
    study.navigation = state.nav

    def frame(count=1):
        for _ in range(count):
            session.tick(FrameNeeds(qpos=True, joint_frames=True), wall_dt=0)
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(state.window_size or io.display_size)
            ui.hits.clear()
            controls.clear()
            children.clear()
            if state.mode == "panel":
                study.panel(state.panel_name, ctx)
            else:
                imgui.begin("Navigation", flags=imgui.WindowFlags_.no_title_bar)
                pos = imgui.get_cursor_screen_pos()
                size = imgui.get_content_region_avail()
                if state.mode == "scene":
                    state.nav.draw_scene(ctx, pos.x, pos.y, size.x, size.y)
                else:
                    y = pos.y
                    if state.show_pose_header:
                        study.panel_header("Pose", pos.x, y, size.x)
                        y += 38 * study.s
                    state.nav.draw_pose(ctx, pos.x, y, size.x, node_id=state.node_id)
                imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def point(key):
        x, y, w, h = ui.hits[key] if key in ui.hits else controls[key]
        return x + w / 2, y + h / 2

    def click(key):
        event("add_mouse_pos_event", *point(key))
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame(3)

    def double_click(key):
        event("add_mouse_pos_event", *point(key))
        for _ in range(2):
            event("add_mouse_button_event", 0, True)
            event("add_mouse_button_event", 0, False)
        frame(3)

    state.frame, state.event, state.click, state.double_click = frame, event, click, double_click
    frame(3)
    try:
        yield state
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.mark.physics
def test_scene_geometry_modes_use_renderer_preset_flags(navigation):
    for view in GEOMETRY_VIEWS:
        navigation.click("scene-geometry-" + view.value)
        assert geometry_view(navigation.ctx.backend) is view


@pytest.mark.physics
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_geometry_presets_preserve_source_groups_and_joint_state(navigation, language, monkeypatch):
    state, ui = navigation, navigation.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    active = []
    original = ui.segmented

    def observe(*args, **kwargs):
        active.append(args[5])
        return original(*args, **kwargs)

    monkeypatch.setattr(ui, "segmented", observe)
    assert state.session.submit(cmd.SetVisualGroup("geom", 0, False)).ok
    state.frame(3)
    assert active[-1] == 0
    assert geometry_view(state.ctx.backend) is GeometryView.VISUAL
    source = state.session.source
    groups, qpos = source.geom_group_visible.copy(), state.session.frame.qpos.copy()
    for index, view in enumerate(GEOMETRY_VIEWS):
        state.click("scene-geometry-" + view.value)
        assert active[-1] == index
        assert state.session.source is source
        np.testing.assert_array_equal(source.geom_group_visible, groups)
        np.testing.assert_array_equal(state.session.frame.qpos, qpos)


@pytest.mark.physics
@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (390, 252, 140))
def test_scene_geometry_segments_keep_labels_and_dispatch_keyboard_presets(
    navigation, width, language, monkeypatch
):
    state, ui = navigation, navigation.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    captions = {}
    original = ui._segment_label

    def observe(draw, lo, hi, color, scale, icon, label):
        captions[icon] = label
        return original(draw, lo, hi, color, scale, icon, label)

    monkeypatch.setattr(ui, "_segment_label", observe)
    imgui.get_io().display_size = (width * ui.s, 760 * ui.s)
    state.frame(3)
    assert ui.hits["scene-geometry"][2] == pytest.approx((width - 24) * ui.s, abs=0.5)
    assert ui.hits["scene-geometry"][3] == pytest.approx(28 * ui.s)
    assert "scene-geometry-default" not in ui.hits
    rects = [ui.hits["scene-geometry-" + view.value] for view in GEOMETRY_VIEWS]
    for index, (x, y, w, h) in enumerate(rects):
        label = state.ctx.tr(GEOMETRY_VIEWS[index].value.title())
        icon = GEOMETRY_ICONS[index]
        assert captions[icon] == (label if width != 140 else "")
        if captions[icon]:
            assert ui.measure(label) + (18 + 5 + 12) * ui.s <= w + 0.5
        assert h == pytest.approx(28 * ui.s)
        if index:
            px, py, pw, _ = rects[index - 1]
            assert (x, y) == pytest.approx((px + pw, py))
    state.click("scene-geometry-visual")
    state.event("add_mouse_pos_event", -100, -100)
    state.event("add_key_event", imgui.Key.right_arrow, True)
    state.event("add_key_event", imgui.Key.right_arrow, False)
    state.frame(2)
    state.event("add_key_event", imgui.Key.space, True)
    state.event("add_key_event", imgui.Key.space, False)
    state.frame(2)
    assert geometry_view(state.ctx.backend) is GeometryView.COLLISION
    state.click("scene-geometry-both")
    assert geometry_view(state.ctx.backend) is GeometryView.BOTH


@pytest.mark.physics
def test_narrow_scene_modes_and_type_capsules_remain_inside_the_panel(navigation):
    state = navigation
    imgui.get_io().display_size = (180 * state.study.s, 760 * state.study.s)
    state.frame(3)
    for key, (x, _y, width, _height) in state.study.ui.hits.items():
        if key.startswith(("scene-geometry-", "scene-filter-", "entity-", "visible-", "expand-")):
            assert x >= 0
            assert x + width <= imgui.get_io().display_size.x


@pytest.mark.physics
def test_scene_type_capsules_filter_real_nodes_and_keep_geometry(navigation):
    state = navigation
    state.click("scene-filter-geom")
    rows = state.nav.scene_rows(state.ctx)
    assert rows
    assert {node.type for node, _, _ in rows} == {NodeType.GEOM}
    assert len(rows) == sum(node.type is NodeType.GEOM for node in state.session.nodes)
    state.click("scene-filter-all")
    rows = state.nav.scene_rows(state.ctx)
    assert any(node.type is NodeType.GEOM for node, _, _ in rows)
    assert any(node.type is NodeType.ROBOT for node, _, _ in rows)


@pytest.mark.physics
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_scene_type_capsules_draw_semantic_icons_counts_and_full_tooltips(
    navigation, language, monkeypatch
):
    state, ui = navigation, navigation.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    icons, text, hints = [], [], []
    draw_icon, draw_text, draw_tooltip = ui.role_icon, ImguiDraw2D.text, ui.tooltip

    def icon(*args, **kwargs):
        icons.append(args)
        return draw_icon(*args, **kwargs)

    def caption(draw, position, color, value, **kwargs):
        text.append(value)
        return draw_text(draw, position, color, value, **kwargs)

    def tooltip(value, **kwargs):
        hints.append(value)
        return draw_tooltip(value, **kwargs)

    monkeypatch.setattr(ui, "role_icon", icon)
    monkeypatch.setattr(ImguiDraw2D, "text", caption)
    monkeypatch.setattr(ui, "tooltip", tooltip)
    state.frame(3)
    count = dict(state.nav.scene_filters(state.ctx))["geom"]
    x, y, width, height = ui.hits["scene-filter-geom"]
    geom_icons = [item for item in icons if item[0] == "geom" and item[2] == y + 6 * ui.s]
    assert geom_icons and geom_icons[-1][1] == pytest.approx(x + 6 * ui.s)
    assert str(count) in text and f"{state.ctx.tr('Geom')} {count}" not in text
    assert "scene-geometry-more" not in ui.hits
    state.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    assert hints[-1] == f"{state.ctx.tr('Geom')}: {count}"
    state.click("scene-filter-geom")
    assert all(node.type is NodeType.GEOM for node, _, _ in state.nav.scene_rows(state.ctx))


@pytest.mark.physics
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_scene_selection_uses_one_row_background_and_keeps_keyboard_activation(
    navigation, language, monkeypatch
):
    state, ui = navigation, navigation.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    node = next(node for node in state.session.nodes if node.name == "hinge_body")
    state.click("scene-filter-link")
    state.click("entity-" + str(node.node_id))
    rows, current = {}, []
    scene_row, rect = state.nav._scene_row, ui.rect

    def row(ctx, item, depth, leaf, width):
        current.append(item.node_id)
        rows[item.node_id] = []
        try:
            return scene_row(ctx, item, depth, leaf, width)
        finally:
            current.pop()

    def background(*args, **kwargs):
        if current:
            rows[current[-1]].append(args)
        return rect(*args, **kwargs)

    monkeypatch.setattr(state.nav, "_scene_row", row)
    monkeypatch.setattr(ui, "rect", background)
    state.frame(3)
    assert len(rows[node.node_id]) == 1
    assert rows[node.node_id][0][3] == pytest.approx(28 * ui.s)
    assert rows[node.node_id][0][2] > 100 * ui.s
    assert state.session.submit(cmd.Select(0)).ok
    state.event("add_mouse_pos_event", -100, -100)
    for key in (imgui.Key.down_arrow, imgui.Key.up_arrow):
        state.event("add_key_event", key, True)
        state.event("add_key_event", key, False)
        state.frame(2)
    state.event("add_key_event", imgui.Key.space, True)
    state.event("add_key_event", imgui.Key.space, False)
    state.frame(3)
    assert state.session.selected_node is node


@pytest.mark.physics
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_source_joint_names_are_literal_in_scene_and_pose(navigation, language, monkeypatch):
    state, ui = navigation, navigation.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    names = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **kwargs):
        names.append(value)
        return original(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    state.click("scene-filter-joint")
    node = next(
        node for node in state.session.nodes if node.type is NodeType.JOINT and node.name == "free"
    )
    assert "free" in names
    names.clear()
    state.mode = "pose"
    state.frame(3)
    assert "free" in names
    assert state.session.node(node.node_id).name == "free"


@pytest.mark.physics
def test_scene_click_selects_and_double_click_requests_real_node_focus(navigation):
    state = navigation
    state.click("scene-filter-link")
    node = next(node for node in state.session.nodes if node.name == "hinge_body")
    key = "entity-" + str(node.node_id)
    state.click(key)
    assert state.session.selected_node is node
    assert state.app._pending_node_focus_id is None
    state.double_click(key)
    assert state.session.selected_node is node
    assert state.app._pending_node_focus_id == node.node_id
    state.session.submit(cmd.Select(0))
    state.app._pending_node_focus_id = None
    state.frame(22)
    state.double_click(key)
    assert state.app._pending_node_focus_id == node.node_id


@pytest.mark.physics
def test_scene_disclosures_and_visibility_write_shared_state(navigation):
    state = navigation
    node = next(node for node in state.session.nodes if node.name == "free_body")
    key = "expand-" + str(node.node_id)
    assert any(child.node_id in node.children for child, _, _ in state.nav.scene_rows(state.ctx))
    state.click(key)
    assert not any(
        child.node_id in node.children for child, _, _ in state.nav.scene_rows(state.ctx)
    )
    state.click(key)
    assert any(child.node_id in node.children for child, _, _ in state.nav.scene_rows(state.ctx))
    state.click("visible-" + str(node.node_id))
    assert not state.session.node(node.node_id).visible
    state.click("visible-" + str(node.node_id))
    assert state.session.node(node.node_id).visible


@pytest.mark.physics
def test_pose_selects_actual_joint_and_focuses_the_joint_gizmo(navigation):
    state = navigation
    state.mode = "pose"
    state.frame(3)
    joint = next(joint for joint in state.session.joints if joint.name == "hinge_limited")
    node = next(
        node
        for node in state.session.nodes
        if node.type is NodeType.JOINT and node.joint_index == joint.joint_id
    )
    key = "pose-joint-" + str(joint.joint_id)
    state.click(key)
    assert state.session.selected_node is node
    assert not state.ctx.gizmo.enabled
    state.double_click(key)
    assert state.session.selected_node is node
    assert state.app._pending_joint_focus_id == joint.joint_id
    assert state.ctx.gizmo.enabled
    assert state.ctx.gizmo.mode == "rotate"
    state.app.refresh_document()
    assert state.app.entity["type"] == "joint"


@pytest.mark.physics
def test_inspector_pose_uses_selected_body_or_joint_and_preserves_hinge_rail(navigation):
    state = navigation
    body = next(node for node in state.session.nodes if node.name == "hinge_body")
    joint = next(joint for joint in state.session.joints if joint.name == "hinge_limited")
    joint_node = next(
        node
        for node in state.session.nodes
        if node.type is NodeType.JOINT and node.joint_index == joint.joint_id
    )
    assert state.nav.pose_joints(state.ctx, body.node_id) == (joint,)
    assert state.nav.pose_joints(state.ctx, joint_node.node_id) == (joint,)
    state.mode, state.node_id = "pose", body.node_id
    state.frame(3)
    assert not any(key.startswith("pose-joint-") for key in state.study.ui.hits)
    assert "##pose-hinge-track" in state.study.ui.hits
    x, y, w, h = state.study.ui.hits["##pose-hinge-track"]
    before = state.session.frame.qpos[joint.qpos_adr]
    state.event("add_mouse_pos_event", x + w * 0.5, y + h * 0.5)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_pos_event", x + w * 0.8, y + h * 0.5)
    state.event("add_mouse_button_event", 0, False)
    assert not np.isclose(state.session.frame.qpos[joint.qpos_adr], before)
    assert state.session.selected_node is joint_node
    assert joint.range[0] <= state.session.frame.qpos[joint.qpos_adr] <= joint.range[1]


@pytest.mark.physics
@pytest.mark.parametrize("width,control", ((320, "track"), (320, "value"), (140, "value")))
def test_pose_value_activation_respects_window_clipping(navigation, width, control):
    state = navigation
    state.mode = "pose"
    state.window_size = (width * state.study.s, 210 * state.study.s)
    state.frame(3)
    joint = next(joint for joint in state.session.joints if joint.name == "hinge_limited")
    key = "##pose-hinge-" + control
    x, y, w, h = state.study.ui.hits[key]
    assert y >= state.window_size[1]
    before = state.session.frame.qpos.copy()
    state.session.submit(cmd.Select(0))
    state.click(key)
    assert state.session.selected_node is None
    np.testing.assert_array_equal(state.session.frame.qpos, before)
    state.frame(22)

    state.window_size = (width * state.study.s, 760 * state.study.s)
    state.frame(3)
    state.click(key)
    assert state.session.selected_node.type is NodeType.JOINT
    assert state.session.selected_node.joint_index == joint.joint_id
    state.frame(22)
    x, y, w, h = state.study.ui.hits[key]
    state.event("add_mouse_pos_event", x + w * 0.5, y + h * 0.5)
    state.event("add_mouse_button_event", 0, True)
    state.event("add_mouse_pos_event", x + w * 0.8, y + h * 0.5)
    state.event("add_mouse_button_event", 0, False)
    assert not np.isclose(state.session.frame.qpos[joint.qpos_adr], before[joint.qpos_adr])


@pytest.mark.physics
def test_pose_header_search_sort_and_copy_use_real_joint_state(navigation):
    state = navigation
    state.mode, state.show_pose_header = "pose", True
    state.study.query = "scene filter"
    state.frame(3)
    assert all("header-Pose-" + key in state.study.ui.hits for key in ("search", "sort", "copy"))
    assert not any(key in state.study.ui.hits for key in ("pose-search", "pose-sort", "pose-copy"))
    assert "input-##pose-search" not in state.controls
    state.click("header-Pose-search")
    assert state.nav.pose_action_active("search")
    state.click("input-##pose-search")
    state.event("add_input_characters_utf8", "hinge_limited")
    assert [joint.name for joint in state.nav.pose_joints(state.ctx)] == ["hinge_limited"]
    state.frame(3)
    state.click("header-Pose-search")
    assert not state.nav.pose_search_visible and not state.nav.joints._search
    assert state.study.query == "scene filter"
    state.click("header-Pose-sort")
    assert state.nav.pose_action_active("sort")
    names = [joint.name for joint in state.nav.pose_joints(state.ctx)]
    assert names == sorted(names)
    state.click("header-Pose-copy")
    assert "menu-Copy qpos" in state.controls
    state.click("menu-Copy qpos")
    assert state.copied[-1] == state_vector_text(state.session.frame.qpos)
    state.click("header-Pose-copy")
    state.click("menu-Copy qvel")
    assert state.copied[-1] == state_vector_text(state.session.adapter.capture_state().qvel)


@pytest.mark.physics
def test_pose_more_keeps_actions_and_customizes_the_header(navigation):
    state = navigation
    state.mode, state.show_pose_header = "pose", True
    state.frame(3)
    state.click("header-Pose-more")
    assert all(
        "menu-" + label in state.controls
        for label in ("Search joints", "Sort by name", "Copy state")
    )
    rect = state.controls["submenu-Header actions"]
    state.event("add_mouse_pos_event", rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
    state.frame(20)
    state.click("checkbox-Sort by name")
    assert "sort" not in state.study.header_actions["Pose"]
    state.event("add_key_event", imgui.Key.escape, True)
    state.event("add_key_event", imgui.Key.escape, False)
    state.frame(3)
    assert "header-Pose-sort" not in state.study.ui.hits
    state.click("header-Pose-more")
    state.click("menu-Sort by name")
    assert state.nav.joints._sort_by_name
    state.click("header-Pose-more")
    state.click("menu-Copy state")
    assert "menu-Copy qpos" in state.controls
    state.click("menu-Copy qpos")
    assert state.copied[-1] == state_vector_text(state.session.frame.qpos)


@pytest.mark.physics
def test_narrow_pose_header_keeps_all_actions_in_more(navigation):
    state = navigation
    state.mode, state.show_pose_header = "pose", True
    state.window_size = (140 * state.study.s, 760 * state.study.s)
    state.frame(3)
    assert not any(
        "header-Pose-" + key in state.study.ui.hits for key in ("search", "sort", "copy")
    )
    x, _y, width, _h = state.study.ui.hits["header-Pose-more"]
    assert 0 <= x < x + width <= state.window_size[0]
    state.click("header-Pose-more")
    state.click("menu-Search joints")
    assert "input-##pose-search" in state.controls
    state.click("header-Pose-more")
    state.click("menu-Copy state")
    assert "menu-Copy qpos" in state.controls
    state.click("menu-Copy qpos")
    assert state.copied[-1] == state_vector_text(state.session.frame.qpos)


def _use_real_sidebar(state, name, width, height):
    study = state.study
    state.app.panels = ReferencePanels()
    state.app.refresh_document()
    state.nav.hierarchy = state.app.panels.get("hierarchy")
    state.nav.joints = state.app.panels.get("joints")
    study.open = {panel.name: panel.name == name for panel in state.app.panels.panels}
    study.header_actions["Scene"] = {"search", "frame", "add"}
    study.header_add_pending = False
    study.focus_next = study.bottom_focus = None
    study.widths, study.bottom_active = {}, "Timeline"
    state.mode, state.panel_name = "panel", name
    state.window_size = (width * study.s, height * study.s)
    state.frame(4)


@pytest.mark.physics
@pytest.mark.parametrize("navigation", (1.0, 2.0), indirect=True)
def test_pose_sidebar_header_stays_fixed_while_child_scrolls(navigation):
    state = navigation
    _use_real_sidebar(state, "Pose", 280, 230)
    before = state.study.ui.hits["header-Pose-more"]
    assert state.children["Panel content"].scroll_max.y > 0
    state.event("add_mouse_pos_event", 120 * state.study.s, 180 * state.study.s)
    state.event("add_mouse_wheel_event", 0, -5)
    state.frame(4)
    assert state.children["Panel content"].scroll.y > 0
    assert imgui.internal.find_window_by_name("Pose").scroll.y == 0
    assert state.study.ui.hits["header-Pose-more"] == before
    state.click("header-Pose-more")
    state.click("menu-Copy state")
    assert "menu-Copy qpos" in state.controls
    state.click("menu-Copy qpos")
    assert state.copied[-1] == state_vector_text(state.session.frame.qpos)


@pytest.mark.physics
@pytest.mark.parametrize("navigation", (1.0, 2.0), indirect=True)
def test_short_scene_sidebar_scrolls_type_filters_without_moving_the_header(navigation):
    state = navigation
    _use_real_sidebar(state, "Scene", 140, 150)
    before = state.study.ui.hits["header-Scene-more"]
    assert state.children["Panel content"].scroll_max.y > 0
    state.event("add_mouse_pos_event", 70 * state.study.s, 120 * state.study.s)
    for _ in range(8):
        _x, y, _w, height = state.study.ui.hits["scene-filter-robot"]
        clip = state.children["Panel content"].inner_clip_rect
        if clip.min.y <= y and y + height <= clip.max.y:
            break
        direction = -1 if y + height > clip.max.y else 1
        state.event("add_mouse_wheel_event", 0, direction * state.study.s)
        state.frame(4)
    assert state.children["Panel content"].scroll.y > 0
    assert imgui.internal.find_window_by_name("Scene").scroll.y == 0
    assert state.study.ui.hits["header-Scene-more"] == before
    _x, y, _w, height = state.study.ui.hits["scene-filter-robot"]
    clip = state.children["Panel content"].inner_clip_rect
    assert clip.min.y <= y < y + height <= clip.max.y
    state.click("scene-filter-robot")
    assert {node.type for node, _, _ in state.nav.scene_rows(state.ctx)} == {NodeType.ROBOT}


@pytest.mark.physics
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_scene_search_round_trip_preserves_bottom_scroll_and_row_position(navigation, language):
    state = navigation
    _use_real_sidebar(state, "Scene", 252, 500)
    selected = state.nav.scene_rows(state.ctx)[-5][0]
    assert state.session.submit(cmd.SelectNode(selected.node_id)).ok
    state.frame(3)
    child = state.children["Scene entities"]
    state.event(
        "add_mouse_pos_event", child.pos.x + child.size.x / 2, child.pos.y + child.size.y / 2
    )
    state.event("add_mouse_wheel_event", 0, -100)
    state.frame(3)
    child = state.children["Scene entities"]
    assert child.scroll.y > 0
    assert child.scroll.y == pytest.approx(child.scroll_max.y, abs=state.study.s)
    row_key = "entity-" + str(selected.node_id)
    row_y = state.study.ui.hits[row_key][1]
    content_height, scroll_y = child.content_size.y, child.scroll.y
    state.click("header-Scene-search")
    assert state.study.search_visible
    assert state.children["Scene entities"].content_size.y == pytest.approx(content_height)
    state.click("header-Scene-search")
    assert not state.study.search_visible
    assert state.children["Scene entities"].content_size.y == pytest.approx(content_height)
    assert state.children["Scene entities"].scroll.y == pytest.approx(scroll_y, abs=state.study.s)
    assert state.study.ui.hits[row_key][1] == pytest.approx(row_y, abs=state.study.s)
    assert state.session.selected_node is selected


@pytest.mark.physics
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_scene_external_selection_follows_the_visible_list_viewport(navigation, language):
    state = navigation
    _use_real_sidebar(state, "Scene", 252, 500)
    selected = state.nav.scene_rows(state.ctx)[-8][0]
    assert state.session.submit(cmd.SelectNode(selected.node_id)).ok
    state.frame(4)
    child = state.children["Scene entities"]
    _x, y, _width, height = state.study.ui.hits["entity-" + str(selected.node_id)]
    assert child.inner_clip_rect.min.y <= y < y + height <= child.inner_clip_rect.max.y
    assert state.session.selected_node is selected
