"""Timeline references retain the same rounded content boundary as the live child."""

import pytest

from mojive.design.native.geometry import Rect
from mojive.design.native.specimens import ComponentSpecimens
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit
from mojive.ui.workspace.timeline_body import body_geometry
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25), indirect=True)
@pytest.mark.parametrize("inset", (8, 14))
def test_timeline_body_reference_keeps_controls_inside_native_insets(appearance, inset):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"timeline_body": {"paddingX": inset, "paddingY": inset}})
    specimen = ComponentSpecimens(runtime)

    def paint():
        width, height = specimen.measure(ui, "timeline_body", 480 * ui.s)
        result = specimen.paint(ui, "timeline_body", Rect(24 * ui.s, 32 * ui.s, width, height))
        geometry = body_geometry(ui)
        assert geometry.padding == (native_unit(inset * ui.s),) * 2
        x, y, w, h = result["frame"]
        px, py = geometry.padding
        assert abs(geometry.inner_radius - max(0, geometry.radius - px)) <= native_epsilon(
            geometry.radius, px
        )
        assert result["content"] == pytest.approx((x + px, y + py, w - 2 * px, h - 2 * py))
        assert len(result["children"]) >= 5
        for left, top, cw, ch in result["children"]:
            assert left >= x + px and top >= y + py
            assert left + cw <= x + w - px
            assert top + ch <= y + h - py
        controls = [
            record
            for record in result["controls"]
            if any(part in record["parts"] for part in ("select", "icon_button", "lane_label"))
        ]
        assert len(controls) >= 5
        for control in controls:
            radius = control["fieldValues"]["radius"] * ui.s
            assert abs(radius - geometry.inner_radius) <= native_epsilon(
                radius, geometry.inner_radius
            )

    frame(paint)
