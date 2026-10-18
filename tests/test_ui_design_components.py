"""Native component tabs use the registered Inspector and real scene commands."""

from dataclasses import fields as dataclass_fields
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from examples.ui_design.native.docking_study import Study
from examples.ui_design.native.scene_preview import ReferencePanels
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import (
    FrameNeeds,
    GeometryProperties,
    NodeType,
    SceneModelInfo,
    SceneNode,
)
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.scene.assets import resolve
from mojive.session import Session
from mojive.types import CameraView, Light
from mojive.ui.app import ViewerApp
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.design import BODY_SIZE, DESIGN
from mojive.ui.workspace.inspector_components import draw_component
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1.0, 2.25))
def components(request, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    scale = request.param
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * scale, 1600 * scale)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    language = request.node.callspec.params.get("language")
    if language is not None:
        report = SimpleNamespace(cjk_path=None)
        if language == "zh_CN":
            from mojive.text.sources import resolve_font_sources

            sources = resolve_font_sources(allow_download=False)
            if len(sources) < 2:
                pytest.skip("CJK font is unavailable")
            report.cjk_path, report.cjk_index = sources[1].path, sources[1].index
        ui = UI(SimpleNamespace(style_scale=scale, font_report=report))
        imgui.get_style().window_min_size = (32 * scale, 32 * scale)
    else:
        ui = UI(SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)))
    ui.s, ui.hits, ui.numeric_edits = scale, {}, {}
    ui.translate = str
    style = imgui.get_style()
    style.font_size_base = BODY_SIZE * scale
    style.frame_padding = (8 * scale, 7 * scale)
    style.item_spacing = (6 * scale, 8 * scale)
    scene = Scene()
    scene.box(name="box")
    scene.add_camera("camera", CameraView())
    scene.add_light("light", Light())
    model = SceneNode(
        max(node.node_id for node in scene.source.nodes) + 1,
        "model",
        NodeType.MODEL,
        model_id=7,
    )

    class Adapter(StaticSceneAdapter):
        def nodes(self):
            return [*super().nodes(), model]

        def scene_models(self):
            return (SceneModelInfo(model.model_id, model.name, tmp_path / "model", True),)

        def geometry_properties(self, node_id):
            node = next((node for node in self.nodes() if node.node_id == node_id), None)
            if node is not None and node.type is NodeType.GEOM:
                return GeometryProperties(node_id, (1.0, 0.005, 0.0001), 1, 1, 3, 0, 0, 0, 1)
            return None

    session = Session(Adapter(scene))
    session.tick(FrameNeeds(poses=True), wall_dt=0)
    app = ViewerApp(session, NullBackend())
    study = Study.__new__(Study)
    study.s, study.ui = scale, ui
    study.preview = SimpleNamespace(panels=ReferencePanels())
    ctx = PanelContext(
        session,
        app.backend,
        camera=app.camera,
        gizmo=app.gizmo,
        view_cube=app.view_cube,
        style_scale=scale,
    )
    if language is not None:
        ctx.translate = Localizer(Language(language)).text
        ui.translate = ctx.tr
    study._panel_context = ctx
    inspector = study.preview.panels.get("inspector")
    inspector.open = True
    controls, fields, labels, rendered, rendered_rects, rendered_items = (
        {},
        {},
        set(),
        set(),
        {},
        [],
    )
    text = ui.text

    def observe_text(x, y, value, *args, **kwargs):
        labels.add(value)
        return text(x, y, value, *args, **kwargs)

    monkeypatch.setattr(ui, "text", observe_text)
    draw_text = ImguiDraw2D.text

    def observe_draw_text(draw, position, color, value, **kwargs):
        rendered.add(value)
        rendered_rects[value] = (*position, draw.text_size(value)[0], imgui.get_font_size())
        rendered_items.append((value, rendered_rects[value]))
        return draw_text(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", observe_draw_text)
    for method in ("drag_float", "slider_float", "button", "combo"):
        original = getattr(imgui, method)

        def observe(identifier, *args, _original=original, _method=method, **kwargs):
            result = _original(identifier, *args, **kwargs)
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            controls[identifier] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            if _method == "drag_float":
                fields[identifier] = (args[0], args[4] if len(args) > 4 else kwargs.get("format"))
            return result

        monkeypatch.setattr(imgui, method, observe)
    state = SimpleNamespace(
        study=study,
        ctx=ctx,
        inspector=inspector,
        io=io,
        controls=controls,
        fields=fields,
        labels=labels,
        rendered=rendered,
        rendered_rects=rendered_rects,
        rendered_items=rendered_items,
    )

    def frame(node, tab="Properties", count=1, width=320):
        entity = {
            "node_id": node.node_id,
            "colorable": node.type in (NodeType.LINK, NodeType.GEOM),
        }
        result = None
        for _ in range(count):
            controls.clear()
            fields.clear()
            labels.clear()
            rendered.clear()
            rendered_rects.clear()
            rendered_items.clear()
            ui.hits.clear()
            ui.numeric_edits.clear()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((width * scale, 1580 * scale))
            imgui.begin("Inspector", flags=imgui.WindowFlags_.no_title_bar)
            imgui.begin_child("Inspector content", (0, 0))
            result = draw_component(study, ctx, entity, tab)
            assert ctx.inspector_presentation is None
            imgui.end_child()
            imgui.end()
            inspector.finish_frame(
                ctx, keep_edit_transaction=study.inspector_presentation.color_edit_open
            )
            imgui.render()
        return result

    state.frame = frame
    try:
        yield state
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.fixture
def model_components(components):
    session = Session(
        WorkspaceAdapter(
            MuJoCoAdapter(resolve("joint_types")), components.ctx.session.adapter.scene
        )
    )
    session.tick(FrameNeeds(poses=True), wall_dt=0)
    components.ctx.session = session
    assert session.submit(cmd.Pause()).ok
    try:
        yield components
    finally:
        session.release()


def _shown(ctx, label, unit=""):
    label = ctx.tr(label)
    return label[:1].upper() + label[1:] + (f" ({unit})" if unit else "")


def _assert_caption_rendered(state, shown, key):
    x, y, width, height = state.study.ui.hits[key]
    s = state.study.s
    lines = [
        value
        for value, (tx, ty, tw, th) in state.rendered_items
        if abs(tx - x) < 1
        and y - 4 * s <= ty < y + height
        and tw <= width + 1
        and th == pytest.approx(BODY_SIZE * s, abs=0.5)
    ]
    assert len(lines) == 1
    assert lines[0] == shown or (
        lines[0].endswith("…") and shown.startswith(lines[0].removesuffix("…"))
    )


def _assert_camera_equal(actual, expected):
    for field in dataclass_fields(expected):
        before, after = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(after, before)
        else:
            assert after == before, field.name


def _click_control(state, node, rect, *, tab="Properties", width=320):
    x, y, hit_width, height = rect
    state.io.add_mouse_pos_event(x + hit_width / 2, y + height / 2)
    state.frame(node, tab, width=width)
    state.io.add_mouse_button_event(0, True)
    state.frame(node, tab, width=width)
    state.io.add_mouse_button_event(0, False)
    state.frame(node, tab, count=3, width=width)


def _replace_numeric_text(state, node, key, value, *, tab="Properties"):
    x, y, width, height = state.controls[key]
    modifier = imgui.Key.mod_super if state.io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    state.io.add_mouse_pos_event(x + width / 2, y + height / 2)
    state.io.add_key_event(modifier, True)
    state.frame(node, tab)
    state.io.add_mouse_button_event(0, True)
    state.frame(node, tab)
    state.io.add_mouse_button_event(0, False)
    state.frame(node, tab)
    state.io.add_key_event(imgui.Key.a, True)
    state.frame(node, tab)
    state.io.add_key_event(imgui.Key.a, False)
    state.io.add_key_event(modifier, False)
    state.frame(node, tab)
    assert state.io.want_text_input
    state.io.add_input_characters_utf8(value)
    state.frame(node, tab)
    state.io.add_key_event(imgui.Key.enter, True)
    state.frame(node, tab)
    state.io.add_key_event(imgui.Key.enter, False)
    state.frame(node, tab, count=3)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (180, 245, 320, 720))
@pytest.mark.parametrize("component", ("camera", "geometry", "body", "joint"))
def test_component_groups_use_one_label_placement_and_control_column(
    model_components, language, width, component
):
    state, ctx, ui, s = (
        model_components,
        model_components.ctx,
        model_components.study.ui,
        model_components.study.s,
    )
    if component == "camera":
        node = next(node for node in ctx.session.nodes if node.type is NodeType.CAMERA)
        rows = tuple(
            (
                f"vector-label-insp_camera_transform{ctx.tr(label)}",
                f"##insp_camera_transform{ctx.tr(label)}_0_{node.node_id}-frame",
            )
            for label in ("position", "target", "up")
        ) + tuple(
            ("property-label-" + ctx.tr(label), "##inspector-camera-" + label)
            for label in ("vertical fov", "near", "far")
        )
        equal_fields = rows[:3]
        tab = "Properties"
    elif component == "geometry":
        node = next(node for node in ctx.session.nodes if node.type is NodeType.GEOM)
        rows = tuple(
            ("property-label-" + label, key)
            for label, key in (
                ("contact margin", "##contact_margin-frame"),
                ("contact gap", "##contact_gap-frame"),
                ("solver mix", "##contact_solver_mix-frame"),
            )
        ) + tuple(
            (
                "vector-label-contact_surface_" + kind + "_velocity",
                f"##contact_surface_{kind}_velocity_0_{node.node_id}-frame",
            )
            for kind in ("linear", "angular")
        )
        equal_fields = rows[:3]
        tab = "Physics"
    elif component == "body":
        node = next(node for node in ctx.session.nodes if node.name == "hinge_body")
        rows = [("property-label-mass", "##body-mass-frame")]
        rows.extend(
            ("vector-label-" + name, f"##{name}_0_{node.node_id}-frame")
            for name in (
                "body_inertial_position",
                "body_inertial_rotation",
                "body_diagonal_inertia",
            )
        )
        equal_fields = rows[1:]
        tab = "Physics"
    else:
        node = next(
            node
            for node in ctx.session.nodes
            if node.type is NodeType.JOINT and ctx.session.joints[node.joint_index].type == "hinge"
        )
        rows = [
            ("property-label-" + label, key)
            for label, key in (
                ("range", "##joint_range-0-frame"),
                ("damping", "##joint_damping-frame"),
                ("stiffness", "##joint_stiffness-frame"),
            )
        ]
        rows.append(("vector-label-joint_axis", f"##joint_axis_0_{node.node_id}-frame"))
        equal_fields = rows[1:3]
        tab = "Physics"
    assert ctx.session.submit(cmd.SelectNode(node.node_id)).ok
    assert state.frame(node, tab, count=3, width=width)
    if component == "body":
        state.inspector._body_property_edit = replace(
            ctx.session.body_properties(node.node_id), inertia_mode="diagonal"
        )
        state.frame(node, tab, count=3, width=width)
    for label_key, control_key in rows:
        lx, ly, _, lh = ui.hits[label_key]
        cx, cy, cw, ch = ui.hits[control_key]
        assert lh == pytest.approx(BODY_SIZE * s)
        assert cx >= 0 and cx + cw <= width * s + 0.5
        if abs(cx - lx) <= 0.5:
            assert cy - (ly + lh) >= 8 * s - 0.5
        else:
            assert cx >= lx + ui.hits[label_key][2] + 12 * s - 0.5
            assert ly + lh / 2 == pytest.approx(cy + 14 * s, abs=0.5)
        assert ch >= 28 * s - 0.5
    assert all(
        ui.hits[key][2] == pytest.approx(ui.hits[equal_fields[0][1]][2], abs=0.5)
        for _, key in equal_fields
    )
    frames = [rect for key, rect in ui.hits.items() if key.endswith("-frame")]
    for index, (x, y, w, h) in enumerate(frames):
        for nx, ny, nw, nh in frames[index + 1 :]:
            assert (
                x + w <= nx + 0.5 or nx + nw <= x + 0.5 or y + h <= ny + 0.5 or ny + nh <= y + 0.5
            )
    if component in ("body", "geometry", "joint"):
        starts = [ui.hits[key][0] for _, key in rows]
        assert max(starts) - min(starts) <= 0.5


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (245, 320, 720))
def test_component_table_content_height_ends_at_the_last_vector_field(components, language, width):
    state, ui, s = components, components.study.ui, components.study.s
    light = next(node for node in state.ctx.session.nodes if node.type is NodeType.LIGHT)
    state.frame(light, count=3, width=width)
    child = next(
        window
        for window in imgui.get_current_context().windows
        if "Inspector content" in str(window.name) and window.active
    )
    _x, y, _w, h = ui.hits[f"##light_direction_2_{light.node_id}-frame"]
    # The table's bottom cell padding and the section's trailing row gap both
    # contribute to the complete measured content extent.
    expected = y + h - child.dc.cursor_start_pos.y + DESIGN.row_gap * 1.5 * s + 0.5
    assert child.content_size.y == pytest.approx(expected, abs=1)
    assert child.scroll_max.x == 0


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (320, 420, 540, 720))
def test_light_transform_reserves_numeric_width_before_clipping_labels(components, language, width):
    state, ui, s = components, components.study.ui, components.study.s
    light = next(node for node in state.ctx.session.nodes if node.type is NodeType.LIGHT)
    state.frame(light, count=3, width=width)
    captions = tuple(
        _shown(state.ctx, label, unit)
        for label, unit in (("position (local)", "m"), ("direction (local)", ""))
    )
    for name in ("light_position", "light_direction"):
        label = ui.hits["vector-label-" + name]
        fields = [ui.hits[f"##{name}_{index}_{light.node_id}-frame"] for index in range(3)]
        inline = fields[0][0] > label[0] + s
        if width == 320:
            assert not inline
        elif width == 720:
            assert inline
        assert len({round(rect[2], 4) for rect in fields}) == 1
        assert fields[-1][0] + fields[-1][2] <= width * s + 0.5
        if inline:
            assert len({round(rect[1], 4) for rect in fields}) == 1
            assert fields[0][0] - label[0] == pytest.approx(label[2] + 12 * s, abs=0.5)
            assert label[1] + label[3] / 2 == pytest.approx(
                fields[0][1] + fields[0][3] / 2, abs=0.5
            )
        else:
            assert fields[0][0] == pytest.approx(label[0], abs=0.5)
            assert fields[0][1] >= label[1] + label[3] + 8 * s - 0.5
    for caption, name in zip(captions, ("light_position", "light_direction"), strict=True):
        _assert_caption_rendered(state, caption, "vector-label-" + name)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_scalar_labels_ellipsize_beside_one_numeric_column(model_components, monkeypatch, language):
    state, ctx, ui, s = (
        model_components,
        model_components.ctx,
        model_components.study.ui,
        model_components.study.s,
    )
    joint = next(node for node in ctx.session.nodes if node.name == "hinge_limited")
    state.frame(joint, "Physics", count=3, width=720)
    section = "section-" + ctx.tr("advanced joint properties").upper()
    _click_control(state, joint, ui.hits[section], tab="Physics", width=720)
    state.frame(joint, "Physics", count=3, width=720)
    starts = []
    clipped = []
    for label, key, unit in (
        ("actuator force range", "##joint_actuator_force_range-0-frame", "N·m"),
        ("friction loss", "##joint_friction_loss-frame", "N·m"),
        ("spring reference", "##joint_spring_reference-frame", "°"),
    ):
        if key not in ui.hits:
            continue
        rect, field = ui.hits["property-label-" + label], ui.hits[key]
        assert rect[3] == pytest.approx(BODY_SIZE * s)
        assert 48 * s <= rect[2] <= 120 * s
        assert field[0] - rect[0] == pytest.approx(rect[2] + 12 * s, abs=0.5)
        assert rect[1] + rect[3] / 2 == pytest.approx(field[1] + field[3] / 2, abs=0.5)
        _assert_caption_rendered(state, _shown(ctx, label, unit), "property-label-" + label)
        starts.append(field[0])
        caption = _shown(ctx, label, unit)
        if ui.measure(caption, translate=False) > rect[2]:
            clipped.append((label, caption))
    assert len(starts) >= 2
    assert starts == pytest.approx([starts[0]] * len(starts), abs=0.5)
    hints = []
    monkeypatch.setattr(ui, "tooltip", lambda value, **_kwargs: hints.append(value))
    label, caption = (
        clipped[0] if clipped else ("spring reference", _shown(ctx, "spring reference", "°"))
    )
    x, y, width, height = ui.hits["property-label-" + label]
    imgui.get_io().add_mouse_pos_event(x + width / 2, y + height / 2)
    state.frame(joint, "Physics", count=3, width=720)
    if clipped:
        assert any(caption in hint for hint in hints)
    else:
        assert not hints


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (180, 245, 320, 720))
def test_native_inertia_axes_display_common_precision_without_ellipsis(
    model_components, language, width
):
    state, ctx, s = model_components, model_components.ctx, model_components.study.s
    body = next(node for node in ctx.session.nodes if node.name == "hinge_body")
    assert ctx.session.submit(cmd.SelectNode(body.node_id)).ok
    initial = ctx.session.body_properties(body.node_id)
    state.frame(body, "Physics", count=3, width=width)
    state.inspector._body_property_edit = replace(
        initial, inertia_mode="diagonal", diagonal_inertia=(0.0041472,) * 3
    )
    state.frame(body, "Physics", count=3, width=width)
    shown = [rect for value, rect in state.rendered_items if value == "0.0041472"]
    assert len(shown) == 3
    for index in range(3):
        key = f"##body_diagonal_inertia_{index}_{body.node_id}"
        x, y, w, h = state.study.ui.hits[key]
        assert state.fields[key][1] == "%.6g"
        assert any(
            x + 8 * s - 1 <= tx and tx + tw <= x + w - 8 * s + 1 and y <= ty < y + h
            for tx, ty, tw, _ in shown
        )
    assert ctx.session.body_properties(body.node_id) == initial
    assert not ctx.session.can_undo


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_camera_labels_own_units_without_changing_precision_or_radian_state(components, language):
    ctx, inspector = components.ctx, components.inspector
    camera = next(node for node in ctx.session.nodes if node.type is NodeType.CAMERA)
    initial = ctx.session.camera_view(camera.camera_index)
    components.frame(camera, count=3)
    for label, unit in (
        ("position", "m"),
        ("target", "m"),
        ("up", ""),
        ("vertical fov", "°"),
        ("near", "m"),
        ("far", "m"),
    ):
        label_key = (
            "vector-label-insp_camera_transform" + ctx.tr(label)
            if label in ("position", "target", "up")
            else "property-label-" + ctx.tr(label)
        )
        _assert_caption_rendered(components, _shown(ctx, label, unit), label_key)
    key = f"##insp_camera_transform{ctx.tr('position')}_0_{camera.node_id}"
    assert components.fields[key][1] == "%.4f"
    fov_key = "##inspector-camera-vertical fov-value"
    assert components.fields[fov_key][1] == "%.2f"
    assert components.fields[fov_key][0] == pytest.approx(np.degrees(initial.fov_y))
    assert not any(
        key.endswith("-unit") or "##inspector-camera-vertical fov-unit" in key
        for key in components.controls
    )

    label_rect = components.study.ui.hits["property-label-" + ctx.tr("vertical fov")]
    _click_control(components, camera, label_rect)
    assert imgui.get_mouse_cursor() == imgui.MouseCursor_.hand
    assert not inspector._camera_angular_degrees
    _assert_caption_rendered(
        components, _shown(ctx, "vertical fov", "rad"), "property-label-" + ctx.tr("vertical fov")
    )
    assert components.fields[fov_key][0] == pytest.approx(initial.fov_y)
    _assert_camera_equal(ctx.session.camera_view(camera.camera_index), initial)
    assert not ctx.session.can_undo
    _replace_numeric_text(components, camera, fov_key, "0.75")
    assert ctx.session.camera_view(camera.camera_index).fov_y == pytest.approx(0.75)
    assert ctx.session.submit(cmd.Undo()).ok
    components.frame(camera, count=3)
    _assert_camera_equal(ctx.session.camera_view(camera.camera_index), initial)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_native_body_inertia_units_keep_source_precision_and_draft_apply_undo(
    model_components, language
):
    state, ctx = model_components, model_components.ctx
    body = next(node for node in ctx.session.nodes if node.name == "hinge_body")
    assert ctx.session.submit(cmd.SelectNode(body.node_id)).ok
    initial = ctx.session.body_properties(body.node_id)
    assert initial is not None
    state.frame(body, "Physics", count=3)
    state.inspector._body_property_edit = replace(initial, inertia_mode="diagonal")
    state.frame(body, "Physics", count=3)
    for label, unit in (
        ("mass", "kg"),
        ("inertial position", "m"),
        ("inertial rotation", "°"),
        ("diagonal inertia", "kg·m²"),
    ):
        key = (
            "property-label-mass"
            if label == "mass"
            else "vector-label-body_" + label.replace(" ", "_")
        )
        _assert_caption_rendered(state, _shown(ctx, label, unit), key)
    assert state.fields["##body-mass"][1] == "%.6g"
    for prefix, fmt in (
        ("body_inertial_position", "%.5f"),
        ("body_inertial_rotation", "%.2f"),
        ("body_diagonal_inertia", "%.6g"),
    ):
        for axis in range(3):
            assert state.fields[f"##{prefix}_{axis}_{body.node_id}"][1] == fmt
    key = f"##body_inertial_position_0_{body.node_id}"
    _replace_numeric_text(state, body, key, "0.01234", tab="Physics")
    assert state.inspector._body_property_edit.inertial_position[0] == pytest.approx(
        0.01234, abs=1e-8
    )
    assert (
        state.inspector._body_property_edit.inertial_position[1:] == initial.inertial_position[1:]
    )
    assert ctx.session.body_properties(body.node_id) == initial
    _click_control(
        state, body, state.study.ui.hits[ctx.tr("Apply") + "##body-properties"], tab="Physics"
    )
    assert ctx.session.body_properties(body.node_id).inertial_position[0] == pytest.approx(
        0.01234, abs=1e-8
    )
    assert ctx.session.submit(cmd.Undo()).ok
    assert ctx.session.body_properties(body.node_id) == initial


@pytest.mark.parametrize(
    "joint_type,position_unit,force_unit,damping_unit,stiffness_unit",
    (
        ("hinge", "°", "N·m", "N·m·s/rad", "N·m/rad"),
        ("slide", "m", "N", "N·s/m", "N/m"),
    ),
)
def test_native_joint_units_follow_real_generalized_coordinates(
    model_components, joint_type, position_unit, force_unit, damping_unit, stiffness_unit
):
    state, ctx = model_components, model_components.ctx
    node = next(
        node
        for node in ctx.session.nodes
        if node.type is NodeType.JOINT and ctx.session.joints[node.joint_index].type == joint_type
    )
    assert ctx.session.submit(cmd.SelectNode(node.node_id)).ok
    state.frame(node, "Physics", count=3)
    for label, unit in (
        ("range", position_unit),
        ("damping", damping_unit),
        ("stiffness", stiffness_unit),
    ):
        _assert_caption_rendered(state, _shown(ctx, label, unit), "property-label-" + label)
    assert state.fields["##joint_range-0"][1] == ("%.2f" if joint_type == "hinge" else "%.4f")
    assert ctx.session.body_properties(node.node_id) is None
    _click_control(
        state, node, state.study.ui.hits["section-ADVANCED JOINT PROPERTIES"], tab="Physics"
    )
    _assert_caption_rendered(
        state,
        _shown(ctx, "actuator force range", force_unit),
        "property-label-actuator force range",
    )
    assert state.fields["##joint_actuator_force_range-0"][1] == "%.6g"


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (180, 245, 320, 720))
@pytest.mark.parametrize("joint_type", ("hinge", "free"))
def test_native_joint_identity_is_compact_and_keeps_the_real_joint_type(
    model_components, language, width, joint_type
):
    state, ctx, ui, s = (
        model_components,
        model_components.ctx,
        model_components.study.ui,
        model_components.study.s,
    )
    node = next(
        node
        for node in ctx.session.nodes
        if node.type is NodeType.JOINT and ctx.session.joints[node.joint_index].type == joint_type
    )
    joint = ctx.session.joints[node.joint_index]
    assert ctx.session.submit(cmd.SelectNode(node.node_id)).ok
    state.frame(node, "Physics", count=3, width=width)
    x, _y, w, h = ui.hits["joint-metadata"]
    assert x + w <= width * s + 0.5
    assert h <= (45 if width == 180 else 24) * s
    assert ctx.tr("Free joint" if joint_type == "free" else "Hinge joint") in state.rendered
    assert f"Qpos {joint.qpos_adr}  ·  DOF {joint.dof}" in state.rendered
    assert "编辑器相机" not in state.rendered
    assert not any(
        "property-label-" + label in ui.hits for label in ("type", "dof", "qpos address")
    )
    assert "##joint_armature" not in state.fields
    assert not ctx.session.can_undo


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (245, 320, 720))
def test_native_solver_numeric_groups_use_the_full_row_and_preserve_requested_precision(
    components, language, width
):
    state, ui, s = components, components.study.ui, components.study.s
    node = next(node for node in state.ctx.session.nodes if node.type is NodeType.GEOM)
    state.frame(node, "Physics", count=3, width=width)
    label_x, label_y, _label_width, label_height = ui.hits[
        "property-label-impedance min / max / width"
    ]
    fields = [ui.hits[f"##contact_impedance_first-{index}-frame"] for index in range(3)]
    if width < 720:
        assert fields[0][0] == pytest.approx(label_x, abs=0.5)
        assert fields[0][1] - (label_y + label_height) >= 8 * s - 0.5
    else:
        assert fields[0][0] > label_x
        assert fields[0][1] + fields[0][3] / 2 == pytest.approx(label_y + label_height / 2, abs=0.5)
    required = 3 * ui.numeric_minimum(axis=False, format="%.5g") + 8 * s
    inline = width * s - 2 * DESIGN.panel_padding_x * s >= required
    assert all(rect[1] == pytest.approx(fields[0][1], abs=0.5) for rect in fields) is inline
    if not inline:
        assert all(rect[0] == pytest.approx(fields[0][0], abs=0.5) for rect in fields)
    assert all(rect[2] == pytest.approx(fields[0][2], abs=0.5) for rect in fields)
    assert all(rect[2] >= 50 * s for rect in fields)
    for index in range(3):
        value, fmt = state.fields[f"##contact_impedance_first-{index}"]
        assert fmt == "%.5g"
        assert fmt % value in state.rendered
    assert fields[-1][0] + fields[-1][2] <= width * s + 0.5
    assert not state.ctx.session.can_undo


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_native_joint_advanced_collapse_preserves_a_real_uncommitted_draft(
    model_components, language
):
    state, ctx, ui = model_components, model_components.ctx, model_components.study.ui
    node = next(
        node
        for node in ctx.session.nodes
        if node.type is NodeType.JOINT and ctx.session.joints[node.joint_index].type == "hinge"
    )
    joint = ctx.session.joints[node.joint_index]
    initial = ctx.session.joint_advanced_properties(joint.joint_id)
    assert ctx.session.submit(cmd.SelectNode(node.node_id)).ok
    state.frame(node, "Physics", count=3)
    section = "section-" + _shown(ctx, "advanced joint properties").upper()
    assert "##joint_armature" not in state.fields
    _click_control(state, node, ui.hits[section], tab="Physics")
    _replace_numeric_text(state, node, "##joint_armature", "0.12345", tab="Physics")
    draft = state.inspector._joint_advanced_edit
    assert draft.armature == pytest.approx(0.12345)
    assert ctx.session.joint_advanced_properties(joint.joint_id) == initial
    _click_control(state, node, ui.hits[section], tab="Physics")
    assert "##joint_armature" not in state.fields
    _click_control(state, node, ui.hits[section], tab="Physics")
    assert state.inspector._joint_advanced_edit == draft
    assert state.fields["##joint_armature"][0] == pytest.approx(draft.armature)
    assert ctx.session.joint_advanced_properties(joint.joint_id) == initial
    assert not ctx.session.can_undo


def test_component_properties_keep_native_transforms_and_use_registered_camera_state(components):
    session = components.ctx.session
    for node in session.nodes:
        if node.type in (NodeType.CAMERA, NodeType.LIGHT, NodeType.ENVIRONMENT):
            assert components.frame(node, count=3)
            assert imgui.get_draw_data().total_vtx_count > 0
        elif node.type in (NodeType.LINK, NodeType.GEOM):
            assert not components.frame(node)
    camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    assert components.inspector._camera_initial_key == (
        session.cameras[camera.camera_index].camera_id,
        session.structure_generation,
    )
    assert components.study.preview.panels.get("inspector") is components.inspector


def test_light_component_drag_is_live_and_one_undo_restores_the_scene(components):
    session, io = components.ctx.session, components.io
    node = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    original = session.source.lights.lights[node.light_index].diffuse.copy()
    components.frame(node, count=3)
    x, y, w, h = components.controls["##light_intensity"]
    io.add_mouse_pos_event(x + w / 2, y + h / 2)
    components.frame(node)
    io.add_mouse_button_event(0, True)
    components.frame(node)
    io.add_mouse_pos_event(x + w / 2 + 24 * components.study.s, y + h / 2)
    components.frame(node)
    assert components.inspector._edit_transaction and session.editing
    assert not np.allclose(session.source.lights.lights[node.light_index].diffuse, original)
    io.add_mouse_button_event(0, False)
    components.frame(node, count=3)
    assert not components.inspector._edit_transaction and not session.editing
    assert session.submit(cmd.Undo()).ok
    np.testing.assert_allclose(session.source.lights.lights[node.light_index].diffuse, original)
    assert not session.can_undo


def test_component_tabs_show_empty_states_and_preserve_native_material_controls(components):
    session = components.ctx.session
    camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    geometry = next(node for node in session.nodes if node.type is NodeType.GEOM)
    for tab in ("Physics", "Material", "Joint"):
        assert components.frame(camera, tab, count=3)
    assert not components.frame(geometry, "Material")


def test_pinned_model_placement_selects_its_owner_once_and_selection_change_cancels(components):
    session, io, gizmo = components.ctx.session, components.io, components.ctx.gizmo
    model = next(node for node in session.nodes if node.type is NodeType.MODEL)
    camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    assert session.submit(cmd.SelectNode(camera.node_id)).ok
    components.frame(model, count=3)
    assert session.selected_node is camera
    assert gizmo.model_placement_model_id == -1

    x, y, w, h = components.study.ui.hits["Edit Placement"]
    io.add_mouse_pos_event(x + w / 2, y + h / 2)
    components.frame(model)
    io.add_mouse_button_event(0, True)
    components.frame(model)
    io.add_mouse_button_event(0, False)
    components.frame(model, count=3)
    assert session.selected_node is model
    assert gizmo.model_placement_active(session, model.model_id)
    assert not session.can_undo

    assert session.submit(cmd.SelectNode(camera.node_id)).ok
    components.frame(model, count=3)
    assert session.selected_node is camera
    assert gizmo.model_placement_model_id == -1


@pytest.mark.parametrize("width", (245, 320))
def test_component_fields_use_native_section_badge_and_numeric_geometry(components, width):
    session, ui, s = components.ctx.session, components.study.ui, components.study.s
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    components.frame(light, count=3, width=width)
    assert "section-LIGHT PROPERTIES" in ui.hits
    assert "section-LIGHT TRANSFORM" in ui.hits
    assert ui.hits["section-LIGHT PROPERTIES"][3] == pytest.approx(30 * s)
    for key in (
        f"##light_position_0_{light.node_id}",
        f"##light_direction_0_{light.node_id}",
        "##light_colorR",
    ):
        assert ui.hits[key][3] == pytest.approx(28 * s, abs=0.5)
        assert ui.hits[key + "-reset"][2] == pytest.approx(18 * s)
    color = ui.hits["##light_colorR"]
    assert ui.hits["##light_intensity"][1] - (color[1] + color[3]) >= 8 * s - 0.5
    for key, (x, _y, w, _h) in ui.hits.items():
        assert x >= 0 and x + w <= width * s + 0.5 * s, key
    fields = [rect for key, rect in ui.hits.items() if key.endswith("-frame")]
    for index, (x, y, w, h) in enumerate(fields):
        for other_x, other_y, other_w, other_h in fields[index + 1 :]:
            assert not (
                x < other_x + other_w - s
                and other_x < x + w - s
                and y < other_y + other_h - s
                and other_y < y + h - s
            ), "Native component numeric fields overlap"

    camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    components.frame(camera, count=3, width=width)
    assert "section-CAMERA TRANSFORM" in ui.hits
    assert "section-CAMERA PROJECTION" in ui.hits
    assert "##inspector-camera-vertical fov" in ui.hits
    assert all(rect[0] + rect[2] <= width * s + 0.5 * s for rect in ui.hits.values())


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (140, 180, 245, 320))
@pytest.mark.parametrize("component", ("camera", "physics"))
def test_native_property_rows_leave_space_between_real_numeric_controls(
    components, language, width, component
):
    session, ui, s = components.ctx.session, components.study.ui, components.study.s
    node_type = NodeType.CAMERA if component == "camera" else NodeType.GEOM
    node = next(node for node in session.nodes if node.type is node_type)
    assert components.frame(
        node, "Properties" if component == "camera" else "Physics", count=3, width=width
    )
    rows = (
        tuple("##inspector-camera-" + name for name in ("vertical fov", "near", "far"))
        if component == "camera"
        else (
            "##contact_friction-0",
            "##contact_priority",
            "##contact_margin",
            "##contact_gap",
            "##contact_solver_mix",
            "##contact_solver_reference-0",
            "##contact_impedance_first-0",
            "##contact_impedance_shape-0",
            "##contact_adhesion",
            f"##contact_surface_linear_velocity_0_{node.node_id}",
            f"##contact_surface_angular_velocity_0_{node.node_id}",
        )
    )
    previous_bottom = None
    for key in rows:
        x, y, w, h = ui.hits[key]
        assert x >= 0 and x + w <= width * s + 0.5, key
        if component == "camera":
            tx, ty, tw, th = components.controls[key]
            fx, fy, fw, fh = components.controls[key + "-value"]
            for a, b, field_width, field_height in ((tx, ty, tw, th), (fx, fy, fw, fh)):
                assert field_height == pytest.approx(28 * s, abs=0.5), key
                assert x <= a < a + field_width <= x + w + 0.5, key
                assert y <= b < b + field_height <= y + h + 0.5, key
            assert min(tx + tw, fx + fw) <= max(tx, fx) or min(ty + th, fy + fh) <= max(ty, fy), key
            if width == 140:
                assert fy > ty, key
            if fy > ty + 0.5:
                assert fy - (ty + th) >= 8 * s - 1, key
                assert h == pytest.approx(fy + fh - ty, abs=1), key
            else:
                assert h == pytest.approx(fh, abs=0.5), key
        else:
            assert h == pytest.approx(28 * s, abs=0.5), key
        if previous_bottom is not None:
            assert y - previous_bottom >= 8 * s - 0.5, key
        previous_bottom = y + h
    if component == "physics":
        for prefix, count in (
            ("##contact_friction", 3),
            ("##contact_solver_reference", 2),
            ("##contact_impedance_first", 3),
            ("##contact_impedance_shape", 2),
        ):
            for index in range(count - 1):
                x, y, w, h = ui.hits[f"{prefix}-{index}"]
                nx, ny, nw, nh = ui.hits[f"{prefix}-{index + 1}"]
                assert (nw, nh) == pytest.approx((w, h), abs=0.5)
                if ny > y + 0.5:
                    assert nx == pytest.approx(x, abs=0.5)
                    assert ny - (y + h) >= 8 * s - 0.5
                else:
                    assert nx - (x + w) >= 4 * s - 0.5
                    assert ny == pytest.approx(y, abs=0.5)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (245, 320))
def test_surface_velocity_labels_render_fully_above_narrow_vector_fields(
    components, monkeypatch, language, width
):
    session, ui, s = components.ctx.session, components.study.ui, components.study.s
    node = next(node for node in session.nodes if node.type is NodeType.GEOM)
    rendered = {}
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **kwargs):
        rendered[value] = (*position, draw.text_size(value)[0])
        return original(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    components.frame(node, "Physics", count=3, width=width)
    for kind in ("linear", "angular"):
        label = components.ctx.tr(f"surface {kind} velocity")
        unit = "m/s" if kind == "linear" else "rad/s"
        shown = label[:1].upper() + label[1:] + f" ({unit})"
        assert shown in rendered, f"The complete {shown!r} label was not drawn"
        label_x, label_y, text_width = rendered[shown]
        key = f"##contact_surface_{kind}_velocity_0_{node.node_id}"
        field_x, field_y, _, _ = ui.hits[key + "-frame"]
        last_x, _, last_width, _ = ui.hits[
            f"##contact_surface_{kind}_velocity_2_{node.node_id}-frame"
        ]
        assert label_x + text_width <= last_x + last_width + 0.5
        assert field_x == pytest.approx(label_x, abs=0.5)
        assert field_y >= label_y + 13 * s - 0.5


@pytest.mark.parametrize("kind", ("color", "position"))
def test_native_light_color_and_vector_drag_commit_once_and_undo(components, kind):
    session, io = components.ctx.session, components.io
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    source = session.source.lights.lights[light.light_index]
    before = (source.diffuse if kind == "color" else source.position).copy()
    key = "##light_colorR" if kind == "color" else f"##light_position_0_{light.node_id}"
    components.frame(light, count=3)
    x, y, w, h = components.study.ui.hits[key]
    io.add_mouse_pos_event(x + w / 2, y + h / 2)
    components.frame(light)
    io.add_mouse_button_event(0, True)
    components.frame(light)
    direction = -1 if kind == "color" else 1
    io.add_mouse_pos_event(x + w / 2 + 24 * components.study.s * direction, y + h / 2)
    components.frame(light)
    source = session.source.lights.lights[light.light_index]
    after = source.diffuse if kind == "color" else source.position
    assert not np.allclose(after, before)
    assert session.editing and components.inspector._edit_transaction
    io.add_mouse_button_event(0, False)
    components.frame(light, count=3)
    assert not session.editing
    assert session.submit(cmd.Undo()).ok
    source = session.source.lights.lights[light.light_index]
    np.testing.assert_allclose(source.diffuse if kind == "color" else source.position, before)
    assert not session.can_undo


def _open_light_color_popup(state, light):
    state.frame(light, count=3)
    _click_control(state, light, state.study.ui.hits["##light_color-picker"])
    popup = f"##light_color-color-editor-{light.node_id}"
    assert state.study.inspector_presentation.color_edit_open
    assert popup + "-paste" in state.study.ui.hits
    return popup


def test_native_color_popup_keeps_one_transaction_across_paste_and_idle_until_done(components):
    state, session, ui = components, components.ctx.session, components.study.ui
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    before = session.source.lights.lights[light.light_index].diffuse.copy()
    popup = _open_light_color_popup(state, light)
    for color in ("#336699", "#99AA66"):
        imgui.set_clipboard_text(color)
        _click_control(state, light, ui.hits[popup + "-paste"])
        state.frame(light, count=4)
        assert session.editing and state.inspector._edit_transaction
        assert not session.can_undo
    assert not np.allclose(session.source.lights.lights[light.light_index].diffuse, before)
    _click_control(state, light, ui.hits[popup + "-done"])
    assert not session.editing and not state.study.inspector_presentation.color_edit_open
    assert session.submit(cmd.Undo()).ok
    np.testing.assert_array_equal(session.source.lights.lights[light.light_index].diffuse, before)
    assert not session.can_undo


@pytest.mark.parametrize("close", ("cancel", "escape"))
def test_native_color_popup_invalid_paste_is_atomic_and_cancel_restores_the_whole_gesture(
    components, close
):
    state, session, ui = components, components.ctx.session, components.study.ui
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    before = session.source.lights.lights[light.light_index].diffuse.copy()
    popup = _open_light_color_popup(state, light)
    imgui.set_clipboard_text("rgba(1, 2, bad)")
    _click_control(state, light, ui.hits[popup + "-paste"])
    np.testing.assert_array_equal(session.source.lights.lights[light.light_index].diffuse, before)
    editor = state.study.inspector_presentation._color_editors[(light.node_id, "##light_color")]
    assert editor.error
    imgui.set_clipboard_text("#336699")
    _click_control(state, light, ui.hits[popup + "-paste"])
    assert not editor.error and session.editing
    assert not np.allclose(session.source.lights.lights[light.light_index].diffuse, before)
    if close == "cancel":
        _click_control(state, light, ui.hits[popup + "-cancel"])
    else:
        state.io.add_key_event(imgui.Key.escape, True)
        state.frame(light)
        state.io.add_key_event(imgui.Key.escape, False)
        state.frame(light, count=3)
    assert not session.editing and not state.study.inspector_presentation.color_edit_open
    np.testing.assert_array_equal(session.source.lights.lights[light.light_index].diffuse, before)
    assert not session.can_undo


def test_native_light_switch_and_section_are_real_controls(components):
    session, io, ui = components.ctx.session, components.io, components.study.ui
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    components.frame(light, count=3)
    initial = session.source.lights.lights[light.light_index].active
    for key in ("##light_enabled", "section-LIGHT TRANSFORM"):
        x, y, w, h = ui.hits[key]
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        components.frame(light)
        io.add_mouse_button_event(0, True)
        components.frame(light)
        io.add_mouse_button_event(0, False)
        components.frame(light, count=3)
    assert session.source.lights.lights[light.light_index].active != initial
    assert f"##light_position_0_{light.node_id}" not in ui.hits
    assert session.submit(cmd.Undo()).ok
    assert session.source.lights.lights[light.light_index].active == initial
    assert not session.can_undo


@pytest.mark.parametrize("width", (245, 320))
def test_native_light_switches_share_their_label_row_and_keep_a_clear_gap(components, width):
    session, ui, s = components.ctx.session, components.study.ui, components.study.s
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    components.frame(light, count=3, width=width)
    for label, key in (("enabled", "##light_enabled"), ("cast shadow", "##light_cast_shadow")):
        x, y, w, h = ui.hits["property-label-" + label]
        switch_x, switch_y, switch_w, switch_h = ui.hits[key]
        assert y + h / 2 == pytest.approx(switch_y + switch_h / 2, abs=0.5)
        assert x + w <= switch_x - 10 * s + 0.5
        section_x, _, section_w, _ = ui.hits["section-LIGHT PROPERTIES"]
        assert switch_x + switch_w <= section_x + section_w + 0.5


def test_native_light_escape_cancels_live_drag_without_history(components):
    session, io, ui = components.ctx.session, components.io, components.study.ui
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    before = session.source.lights.lights[light.light_index].diffuse.copy()
    components.frame(light, count=3)
    x, y, w, h = ui.hits["##light_intensity"]
    io.add_mouse_pos_event(x + w / 2, y + h / 2)
    components.frame(light)
    io.add_mouse_button_event(0, True)
    components.frame(light)
    io.add_mouse_pos_event(x + w / 2 + 24 * components.study.s, y + h / 2)
    components.frame(light)
    assert session.editing
    io.add_key_event(imgui.Key.escape, True)
    components.frame(light)
    io.add_key_event(imgui.Key.escape, False)
    io.add_mouse_pos_event(x + w / 2 + 48 * components.study.s, y + h / 2)
    components.frame(light)
    assert not session.editing
    io.add_mouse_button_event(0, False)
    components.frame(light, count=3)
    np.testing.assert_allclose(session.source.lights.lights[light.light_index].diffuse, before)
    assert not session.editing and not components.inspector._edit_transaction
    assert not session.can_undo


def test_native_light_ctrl_text_commit_and_escape_preserve_values(components):
    session, io, ui = components.ctx.session, components.io, components.study.ui
    light = next(node for node in session.nodes if node.type is NodeType.LIGHT)
    before = session.source.lights.lights[light.light_index].diffuse.copy()
    components.frame(light, count=3)

    def text(value):
        x, y, w, h = ui.hits["##light_intensity"]
        modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
        io.add_mouse_pos_event(x + w / 2, y + h / 2)
        io.add_key_event(modifier, True)
        components.frame(light)
        io.add_mouse_button_event(0, True)
        components.frame(light)
        io.add_mouse_button_event(0, False)
        components.frame(light)
        io.add_key_event(imgui.Key.a, True)
        components.frame(light)
        io.add_key_event(imgui.Key.a, False)
        io.add_key_event(modifier, False)
        components.frame(light)
        assert io.want_text_input
        io.add_input_characters_utf8(value)
        components.frame(light)

    text("0.125")
    io.add_key_event(imgui.Key.enter, True)
    components.frame(light)
    io.add_key_event(imgui.Key.enter, False)
    components.frame(light, count=3)
    assert np.max(session.source.lights.lights[light.light_index].diffuse) == pytest.approx(0.125)
    assert not session.editing
    assert session.submit(cmd.Undo()).ok
    components.frame(light, count=3)
    np.testing.assert_allclose(session.source.lights.lights[light.light_index].diffuse, before)
    assert not session.can_undo

    text("0.375")
    io.add_key_event(imgui.Key.escape, True)
    components.frame(light)
    io.add_key_event(imgui.Key.escape, False)
    components.frame(light, count=3)
    np.testing.assert_allclose(session.source.lights.lights[light.light_index].diffuse, before)
    assert not session.editing and not session.can_undo
