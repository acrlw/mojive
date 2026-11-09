"""Value and unit captions share a baseline and a visible typographic space."""

import pytest
from imgui_bundle import imgui

from mojive.design.native.geometry import Rect
from tests.test_component_appearance import appearance as appearance

pytestmark = pytest.mark.parametrize("appearance", (1, 1.25, 2), indirect=True)


@pytest.mark.parametrize("unit", ("m", "rad", "°"))
@pytest.mark.parametrize("compact", (False, True))
def test_inline_units_reserve_a_space_without_moving_the_value_baseline(appearance, unit, compact):
    ui, runtime, _update, frame = appearance
    runtime.observe_components = True
    painter = ui.design.primitives

    def paint():
        result = painter.number_shell(
            "unit-value",
            Rect(40 * ui.s, 50 * ui.s, 120 * ui.s, 28 * ui.s),
            -12.345,
            unit,
            compact_caption=compact,
        )
        style = painter.style("input_number")
        imgui.push_font(ui.resolve_font(style, mono=True), style.get("fontSize", 14) * ui.s)
        space = imgui.calc_text_size(" ").x
        imgui.pop_font()
        return result, space

    for _ in range(3):
        result, space = frame(paint)
    record = next(row for row in reversed(painter.observed) if row["key"] == "unit-value-shell")
    text, suffix = record["textRect"], record["unitRect"]
    assert suffix[0] - text[0] - text[2] == pytest.approx(space, abs=0.01)
    assert record["valueTextPosition"][1] == pytest.approx(record["unitTextPosition"][1])
    assert text[0] >= result.box.content.x
    assert suffix[0] + suffix[2] == pytest.approx(result.box.content.right, abs=0.01)
    assert not result.changed and result.value == pytest.approx(-12.345)


@pytest.mark.parametrize("axis", (None, "X"))
@pytest.mark.parametrize("value", (1.0, -123.45))
def test_numeric_paint_shares_the_padded_right_edge(appearance, monkeypatch, axis, value):
    ui, _runtime, _update, frame = appearance
    ends = []
    paint_text = ui.text

    def record(x, y, text, color, size, **options):
        if text == f"{value:.3f}":
            ends.append(
                x
                + ui.measure(
                    text,
                    size,
                    mono=True,
                    translate=False,
                    font_resolved=True,
                    style=options["style"],
                )
            )
        return paint_text(x, y, text, color, size, **options)

    monkeypatch.setattr(ui, "text", record)
    for _ in range(3):
        result = frame(
            lambda: ui.design.primitives.input_number(
                "aligned-value",
                Rect(40 * ui.s, 50 * ui.s, 160 * ui.s, 28 * ui.s),
                value,
                axis=axis,
            )
        )
    assert ends
    assert ends[-1] == pytest.approx(result.box.content.right, abs=0.01)
