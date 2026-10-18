"""Static numeric budgets preserve authored precision and caller-approved grids."""

from copy import deepcopy

import pytest
from imgui_bundle import imgui

from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.workspace.design import DESIGN
from tests.test_native_workspace_widgets import native_ui as native_ui


@pytest.mark.parametrize("count", (3, 4))
def test_numeric_group_never_introduces_an_unapproved_partial_row(native_ui, tmp_path, count):
    ui = native_ui.ui
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    ui.design = DesignPresentation(ui, runtime)
    spec = deepcopy(runtime.spec)
    spec["parts"]["rgba"] = {"columns": 2}
    runtime.update_spec(spec)
    layout = ui.numeric_layout(
        1000 * ui.s, count, parts="rgba", column_options=(count, 1), minimum=30 * ui.s
    )
    assert layout.columns == 1
    assert len({field[0] for field in layout.fields}) == 1
    assert len({field[1] for field in layout.fields}) == count
    spec["parts"]["rgba"]["columns"] = 4
    runtime.update_spec(spec)
    assert (
        ui.numeric_layout(
            1000 * ui.s, count, parts="rgba", column_options=(count, 1), minimum=30 * ui.s
        ).columns
        == count
    )
    assert (
        ui.numeric_layout(
            50 * ui.s, count, parts="rgba", column_options=(count, 1), minimum=30 * ui.s
        ).columns
        == 1
    )


def test_color_budget_measures_only_its_bounded_display_domain(native_ui):
    ui = native_ui.ui
    normalized = ui.color_channel_minimum(format="%.3f")
    integer = ui.color_channel_minimum(format="%.0f")
    ordinary = ui.numeric_minimum(format="%.3f")
    assert integer < normalized < ordinary
    style = ui.component_style(("input_number", "axis_input"))
    imgui.push_font(
        ui.resolve_font(style, mono=True), style.get("fontSize", DESIGN.mono_size) * ui.s
    )
    try:
        text_width = max(imgui.calc_text_size(value).x for value in ("0.000", "1.000"))
    finally:
        imgui.pop_font()
    padding = style.edges("padding", ui.s, (8, 0, 8, 0))
    assert normalized == pytest.approx(
        text_width + padding[0] + padding[2] + style.get("axisWidth", 18) * ui.s + 1
    )
