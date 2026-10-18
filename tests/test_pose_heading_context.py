"""Pose headings spend spare width on constraints while preserving names and disclosure."""

from copy import deepcopy
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive.adapters.base import JointInfo
from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.pose_panel import NativePoseEditor, joint_heading_context
from mojive.ui.workspace.widgets import UI


@pytest.fixture(
    params=tuple(product((1, 1.2, 1.25, 2.25), ("en", "zh_CN"), ("system", "portable")))
)
def heading(request, tmp_path, monkeypatch):
    scale, language, family = request.param
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size, io.delta_time = (1000 * scale, 500 * scale), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=family == "portable",
    )
    ui.prepare_fonts(family)
    ui.translate = Localizer(Language(language)).text
    runtime = DesignRuntime(SpecificationStore(tmp_path / "ui.json"))
    spec = deepcopy(runtime.spec)
    spec["tokens"]["font_family"] = family
    runtime.update_spec(spec)
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    session = SimpleNamespace(frame=SimpleNamespace(qpos=np.zeros(24)), selected_node=None)
    ctx = PanelContext(session, None, style_scale=scale, translate=ui.translate)
    editor = NativePoseEditor(SimpleNamespace(ui=ui, s=scale))
    state = SimpleNamespace(
        ui=ui,
        runtime=runtime,
        editor=editor,
        ctx=ctx,
        width=640,
        joint=JointInfo(9, "wrist", "hinge", True, (-np.pi / 2, np.pi / 2), 8, 7, 1),
        context=True,
        owner="Pose",
        painted=[],
        select=[],
    )
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        state.painted.append((value, imgui.get_font_size() / scale, color))
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def frame(count=1):
        for _ in range(count):
            imgui.new_frame()
            ui.design.begin_frame()
            ui.hits.clear()
            state.painted.clear()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Pose context", flags=imgui.WindowFlags_.no_title_bar)
            joint = state.joint
            foldable = joint.type in ("ball", "free")
            with group_scope(state.owner), ui.component_scope("joint"):
                layout = editor._heading_layout(joint, state.width * scale, foldable)
                editor._heading(
                    ctx,
                    joint,
                    None,
                    lambda *args, **kwargs: state.select.append(args),
                    20 * scale,
                    40 * scale,
                    state.width * scale,
                    None,
                    28 * scale,
                    layout,
                    show_context=state.context,
                    expanded=False if foldable else None,
                )
            imgui.end()
            imgui.render()
            state.records = {
                row["key"]: row for row in (*runtime.observed, *ui.design.primitives.observed)
            }

    state.frame = frame
    try:
        yield state
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("kind", ("hinge", "slide", "ball", "free"))
@pytest.mark.parametrize("width", (160, 640))
def test_actual_heading_paints_complete_context_only_when_the_complete_name_fits(
    heading, kind, width
):
    state = heading
    joint = replace(
        state.joint, type=kind, dof={"hinge": 1, "slide": 1, "ball": 3, "free": 6}[kind]
    )
    state.joint, state.width = joint, width
    state.frame(3)
    key = f"pose-joint-{joint.joint_id}"
    group = state.records[key + "-heading"]
    name = state.records[key]
    if group["contextVisible"]:
        caption = group["context"]
        assert caption in [text for text, *_ in state.painted]
        assert joint.name in [text for text, *_ in state.painted]
        assert not name["clippedLabel"]
        nx, ny, nw, nh = group["nameRect"]
        cx, cy, cw, ch = group["contextRect"]
        assert nx + nw < cx < cx + cw <= 20 * state.ui.s + state.width * state.ui.s
        assert ny == cy and nh == ch == 28 * state.ui.s
        if group["disclosureRect"]:
            assert cx + cw < group["disclosureRect"][0]
    elif width == 640:
        pytest.fail("A short name and concise context must fit a wide heading")
    assert group["hit"][3] == 28 * state.ui.s
    baseline_fold = group["disclosureRect"]
    baseline_fields = set(state.ui.hits)
    state.context = False
    state.frame()
    if baseline_fold:
        assert state.ui.hits[key + "-expand"] == tuple(baseline_fold)
    assert baseline_fields - {key + "-context"} == set(state.ui.hits)


def test_long_joint_names_hide_context_and_keep_the_original_disclosure_position(heading):
    state = heading
    state.width = 180
    long_name = (
        "肩关节俯仰控制轴" * 16
        if state.ui.translate("DoF") != "DoF"
        else "very_long_shoulder_joint_" * 8
    )
    state.joint = replace(state.joint, name=long_name, type="free", dof=6)
    state.frame(3)
    key = f"pose-joint-{state.joint.joint_id}"
    group = state.records[key + "-heading"]
    assert not group["contextVisible"] and group["contextRect"] is None
    assert key + "-context" not in state.ui.hits
    before_name, before_fold = state.ui.hits[key], state.ui.hits[key + "-expand"]
    state.context = False
    state.frame()
    assert state.ui.hits[key] == before_name
    assert state.ui.hits[key + "-expand"] == before_fold


def test_context_font_color_and_gap_are_local_and_do_not_increase_heading_height(heading):
    state = heading
    state.joint = replace(state.joint, type="free", dof=6)
    state.frame(2)
    key = f"pose-joint-{state.joint.joint_id}"
    before = state.records[key + "-heading"]
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["pose_control_metadata"] = {"fontSize": 18, "textColor": "#809090"}
    spec["parts"]["pose_joint"] = {"columnGap": 22}
    state.runtime.update_spec(spec)
    state.frame(2)
    after = state.records[key + "-heading"]
    caption = next(row for row in state.painted if row[0] == after["context"])
    assert caption[1] == pytest.approx(18, abs=0.5 / state.ui.s + 1e-6)
    assert caption[2][:3] == pytest.approx((128 / 255, 144 / 255, 144 / 255))
    assert after["contextRect"][0] - after["nameRect"][0] - after["nameRect"][2] == pytest.approx(
        22 * state.ui.s
    )
    assert after["hit"][3] == before["hit"][3]
    state.owner = "Inspector"
    state.frame()
    caption = next(row for row in state.painted if row[0] == after["context"])
    assert caption[1] == pytest.approx(12, abs=0.5 / state.ui.s + 1e-6)


@pytest.mark.parametrize("kind,size,dof", (("hinge", 1, 1), ("ball", 4, 3), ("free", 7, 6)))
def test_hover_details_include_only_a_genuinely_available_qpos_slice(heading, kind, size, dof):
    state = heading
    joint = replace(state.joint, type=kind, dof=dof, qpos_adr=8)
    detail = state.editor._heading_details(state.ctx, joint)
    assert ("qpos[8]" if size == 1 else f"qpos[8:{8 + size}]") in detail
    state.ctx.session.frame.qpos = np.zeros(8 + size - 1)
    assert "qpos[" not in state.editor._heading_details(state.ctx, joint)
    state.ctx.session.frame.qpos = None
    assert "qpos[" not in state.editor._heading_details(state.ctx, joint)
    state.joint = joint
    state.frame(2)
    assert state.records[f"pose-joint-{joint.joint_id}-heading"]["contextVisible"]


def test_joint_context_reports_real_limits_unbounded_state_and_degrees_of_freedom():
    joint = JointInfo(4, "joint", "hinge", True, (-np.pi / 2, np.pi / 2), 0, 0, 1)
    assert joint_heading_context(joint) == "−90…90°"
    assert joint_heading_context(replace(joint, type="slide", range=(-0.25, 0.5))) == "−0.25…0.5 m"
    assert joint_heading_context(replace(joint, limited=False)) == "Unbounded"
    assert joint_heading_context(replace(joint, range=(np.nan, 1))) == "Invalid joint range"
    assert joint_heading_context(replace(joint, range=(1, -1))) == "Invalid joint range"
    assert joint_heading_context(replace(joint, type="ball", dof=3)) == "3 DoF"
    assert joint_heading_context(replace(joint, type="free", dof=6)) == "6 DoF"
