"""Boolean Inspector rows wrap for their own controls, independently of XYZ grids."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mojive import commands as cmd
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from tests.test_ui_design_components import (
    components as components,
)
from tests.test_ui_design_components import (
    model_components as model_components,
)


def _body_physics(state, tmp_path, width, inertia_mode):
    ctx, ui = state.ctx, state.study.ui
    runtime = DesignRuntime(SpecificationStore(tmp_path / "switch-design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    body = next(node for node in ctx.session.nodes if node.name == "hinge_body")
    assert ctx.session.submit(cmd.SelectNode(body.node_id)).ok
    state.frame(body, "Physics", count=3, width=width)
    state.inspector._body_property_edit = replace(
        ctx.session.body_properties(body.node_id), inertia_mode=inertia_mode
    )
    state.frame(body, "Physics", count=3, width=width)
    return body, runtime


def _switch_row(runtime):
    return next(
        record
        for record in reversed(runtime.observed)
        if record["key"] == "body_inertial_properties/row/mocap body"
    )


@pytest.mark.parametrize("components", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (150, 284, 344, 704))
@pytest.mark.parametrize("inertia_mode", ("auto", "diagonal"))
def test_body_mocap_switch_shares_a_row_when_caption_and_switch_fit(
    model_components, tmp_path, language, width, inertia_mode
):
    state, ctx = model_components, model_components.ctx
    ui, scale = state.study.ui, state.study.s
    body, runtime = _body_physics(state, tmp_path, width, inertia_mode)
    row = _switch_row(runtime)
    label = ui.hits["property-label-mocap body"]
    switch = ui.hits["##body-mocap"]
    shown = ctx.tr("mocap body")
    shown = shown[:1].upper() + shown[1:]
    assert shown in state.rendered
    assert label[0] == pytest.approx(ui.hits["property-label-gravity compensation"][0])
    assert switch[0] + switch[2] <= row["hit"][0] + row["hit"][2] + 1
    assert row["layout"] == ("top" if width == 150 else "left"), row
    if width == 150:
        assert label[1] + label[3] <= switch[1]
    else:
        assert label[1] + label[3] / 2 == pytest.approx(switch[1] + switch[3] / 2, abs=0.5)
        assert switch[0] >= label[0] + label[2] + 12 * scale - 0.5
        if inertia_mode == "diagonal":
            vector = ui.hits["vector-label-body_inertial_position"]
            first = ui.hits[f"##body_inertial_position_0_{body.node_id}-frame"]
            assert (first[1] >= vector[1] + vector[3]) is (width < 704)


@pytest.mark.parametrize("components", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("language", ("zh_CN",))
@pytest.mark.parametrize("switch_width", (44, 110))
def test_body_switch_reflows_after_resize_and_keeps_its_real_draft_edit(
    model_components, tmp_path, language, switch_width
):
    state, ctx = model_components, model_components.ctx
    ui, scale = state.study.ui, state.study.s
    body, runtime = _body_physics(state, tmp_path, 344, "diagonal")
    spec = deepcopy(runtime.spec)
    spec["parts"]["switch"] = {"switchWidth": switch_width}
    spec["parts"]["form_label"] = {"fontSize": 24}
    runtime.update_spec(spec)
    for width in (344, 150, 704, 284, 344):
        state.frame(body, "Physics", count=3, width=width)
        row = _switch_row(runtime)
        wrapped = width == 150 or (width == 284 and switch_width == 110)
        assert row["layout"] == ("top" if wrapped else "left")
        rect = ui.hits["##body-mocap"]
        track = next(
            r for r in reversed(ui.design.primitives.observed) if r["key"] == "##body-mocap"
        )
        assert track["switchRect"][2] == pytest.approx(switch_width * scale)
        for expected in (True, False):
            state.io.add_mouse_pos_event(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
            state.frame(body, "Physics", width=width)
            state.io.add_mouse_button_event(0, True)
            state.frame(body, "Physics", width=width)
            state.io.add_mouse_button_event(0, False)
            state.frame(body, "Physics", count=3, width=width)
            assert state.inspector._body_property_edit.mocap is expected
            assert ctx.session.body_properties(body.node_id).mocap is False
