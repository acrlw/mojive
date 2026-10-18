"""Color specimens allocate the production editor's insets exactly once."""

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from mojive.design.native.specimens import ComponentSpecimens
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("asymmetric", (False, True))
def test_color_editor_specimen_keeps_one_inset_budget_and_complete_rgba_row(appearance, asymmetric):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    if asymmetric:
        update(
            {
                "color_editor": {
                    "paddingLeft": 5,
                    "paddingRight": 11,
                    "paddingTop": 7,
                    "paddingBottom": 13,
                    "marginLeft": 3,
                    "marginRight": 4,
                    "marginTop": 6,
                    "marginBottom": 9,
                }
            }
        )
    specimens = ComponentSpecimens(runtime)
    bounds = Rect(20 * ui.s, 30 * ui.s, 320 * ui.s, 620 * ui.s)

    def paint():
        record = specimens.paint(ui, "color_editor", bounds)
        imgui.dummy((0, 0))
        return record

    frame(paint)
    editor = specimens.color_editors["color_editor"]
    editor.mode = 2
    record = frame(paint)
    content = Rect(*record["content"])
    assert editor.content_origin == pytest.approx((content.x, content.y))
    assert editor.content_width == pytest.approx(content.width)
    fields = [ui.hits["reference-color_editor" + axis + "-frame"] for axis in "RGBA"]
    assert len({round(field[1], 3) for field in fields}) == 1
    for x, _y, width, _height in fields:
        assert content.x <= x and x + width <= content.right + 1
    editor_record = next(
        item
        for item in reversed(runtime.observed)
        if item["key"] == "reference-color_editor-editor"
    )
    assert specimens.color_heights["color_editor", bounds.width] == pytest.approx(
        editor_record["hit"][3]
    )
    assert imgui.get_current_context().active_id == 0
