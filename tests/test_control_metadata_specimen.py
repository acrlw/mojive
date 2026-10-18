"""Control metadata references use the production caption and its own type size."""

import pytest

from mojive.design.native.geometry import Rect
from mojive.design.native.specimens import ComponentSpecimens
from mojive.ui.workspace import scene_panels
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("font_size", (12, 21))
@pytest.mark.parametrize("width", (180, 480))
def test_control_metadata_reference_uses_production_geometry(
    appearance, monkeypatch, font_size, width
):
    ui, runtime, update, frame = appearance
    update({"control_metadata": {"fontSize": font_size}})
    specimens, calls = ComponentSpecimens(runtime), []
    original = scene_panels._control_detail

    def observe(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(scene_panels, "_control_detail", observe)

    def paint():
        size = specimens.measure(ui, "control_metadata", width * ui.s)
        assert size == pytest.approx((width * ui.s, (font_size * 1.4 + 6) * ui.s))
        result = specimens.paint(ui, "control_metadata", Rect(20 * ui.s, 40 * ui.s, *size))
        assert result["part"] == "control_metadata"
        assert result["frame"] == pytest.approx((20 * ui.s, 40 * ui.s, *size))
        assert result["children"] == [ui.hits["reference-control_metadata"]]
        assert len(calls) == 1 and calls[0][-1] == "hinge_motor · -1 … 1"

    frame(paint)
