"""Parent corner constraints reach real controls without mutating authored styles."""

import pytest

from mojive.ui.workspace.color_editor import color_popup_geometry
from mojive.ui.workspace.form_popup import measure_form_popup
from mojive.ui.workspace.menu_geometry import native_unit, popup_radius
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)


@pytest.mark.parametrize("authored", (False, True))
def test_parent_corner_scope_overrides_authored_children_and_restores_cached_styles(
    appearance, authored
):
    ui, _runtime, update, _frame = appearance
    if authored:
        update(
            {
                "button": {"radius": 19},
                "icon_button": {"radius": 20},
                "input_number": {"radius": 21},
            }
        )
    baseline = ui.component_style("input_number")
    with ui.corner_scope(5.2, ("button", "icon_button", "input_number")):
        assert ui.component_style("input_number").get("radius") * ui.s == pytest.approx(5.2)
        assert ui.component_style("input_number").get("radius", 99) * ui.s == pytest.approx(5.2)
        assert ui.component_style("icon_button").get("radius") * ui.s == pytest.approx(5.2)
        with ui.corner_scope(0, "input_number"):
            assert ui.component_style("input_number").get("radius") == 0
            assert ui.component_style("button").get("radius") * ui.s == pytest.approx(5.2)
        assert ui.component_style("input_number").get("radius") * ui.s == pytest.approx(5.2)
    assert ui.component_style("input_number") is baseline
    assert baseline.get("radius") == (21 if authored else 6)
    with pytest.raises(RuntimeError), ui.corner_scope(0, "input_number"):
        raise RuntimeError("Scope restoration")
    assert ui.component_style("input_number") is baseline


@pytest.mark.parametrize("designed", (False, True))
def test_effective_corner_reaches_button_and_numeric_painters(appearance, monkeypatch, designed):
    ui, _runtime, update, frame = appearance
    update(
        {"button": {"radius": 19}, "icon_button": {"radius": 20}, "input_number": {"radius": 21}}
    )
    if not designed:
        ui.design = None
    seen = []
    original = ui.rect

    def rectangle(*args, **options):
        if len(args) >= 4:
            seen.append((args[:4], options.get("r")))
        return original(*args, **options)

    monkeypatch.setattr(ui, "rect", rectangle)

    def paint():
        with ui.corner_scope(5.2, ("button", "icon_button", "input_number")):
            ui.button(
                "corner-button", 30 * ui.s, 40 * ui.s, 120, label="Button", filled=True, radius=25
            )
            ui.input("corner-number", 30 * ui.s, 90 * ui.s, 120 * ui.s, 1.25)

    frame(paint)
    button = next(radius for bounds, radius in seen if bounds[1] == 40 * ui.s)
    assert button * ui.s == pytest.approx(5.2)
    assert ui.component_style("input_number").get("radius", 6) == (6 if not designed else 21)


def test_popup_roots_measure_their_own_control_corners_independently(appearance):
    ui, _runtime, _update, frame = appearance

    def paint():
        with ui.corner_scope(2 * ui.s, ("input_number", "button", "select_option")):
            form = measure_form_popup(ui, ("Field",), field_width=100 * ui.s, row_height=28 * ui.s)
            color = color_popup_geometry(ui)
            choice = popup_radius(ui, "select_popup", (native_unit(8 * ui.s),) * 4)
        assert form.inner_radius == pytest.approx(4 * ui.s)
        assert color.inner_radius == pytest.approx(6 * ui.s)
        assert choice == pytest.approx(6 * ui.s + native_unit(8 * ui.s))

    frame(paint)
