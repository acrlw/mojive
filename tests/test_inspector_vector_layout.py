"""Inspector headings fit before the shared XYZ layout switches to inline rows."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.localization import Language, Localizer
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 2.25))
def inspector_vectors(request, tmp_path, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (900 * request.param, 900 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    ui.design = DesignPresentation(ui, runtime)
    painted = []
    original = ImguiDraw2D.text

    def text(draw, position, color, value, **options):
        painted.append(value)
        return original(draw, position, color, value, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)

    def frame(width, language, values=(0.85, 0, 0.55)):
        ui.translate = Localizer(Language(language)).text
        painted.clear()
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Inspector vector captions", flags=imgui.WindowFlags_.no_title_bar)
        rows = (("Position", "m"), ("Rotation", "°"), ("Scale", ""))
        try:
            with group_scope("Inspector"):
                caption_width = max(
                    ui.vector_caption_width(ui.translate(label), unit) for label, unit in rows
                )
                layout = ui.vector_layout(
                    width * ui.s,
                    format="%.3g",
                    label_minimum=caption_width,
                    key="transform",
                )
                for index, (label, unit) in enumerate(rows):
                    ui.vector(
                        "transform/" + label,
                        30 * ui.s,
                        40 * ui.s + index * (layout.height + 8 * ui.s),
                        width * ui.s,
                        label,
                        values,
                        unit,
                        format="%.3g",
                        layout=layout,
                    )
                return layout, caption_width, tuple(painted)
        finally:
            imgui.end()
            imgui.render()

    try:
        yield ui, runtime, frame
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (180, 296, 640))
def test_transform_captions_and_units_fit_without_changing_xyz_widths(
    inspector_vectors, language, width
):
    ui, _runtime, frame = inspector_vectors
    layout, caption_width, painted = frame(width, language)
    assert layout.label_width >= caption_width
    if width != 296:
        assert layout.label_above is (width < 640)
    assert len({field[2] for field in layout.fields}) == 1
    for label in ("Position", "Rotation", "Scale"):
        assert ui.translate(label) in painted
    assert "(m)" in painted and "(°)" in painted
    assert all(not text.endswith("…") for text in painted)
    changed_values = frame(width, language, (-9999, 0.00001, 111.1))[0]
    assert changed_values == layout


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_caption_font_and_margins_participate_in_shared_reflow(inspector_vectors, language):
    ui, runtime, frame = inspector_vectors
    before = frame(296, language)
    spec = deepcopy(runtime.spec)
    spec["parts"]["form_label"] = {"fontSize": 20, "marginLeft": 8, "marginRight": 12}
    runtime.update_spec(spec)
    layout, caption_width, painted = frame(296, language)
    assert caption_width > before[1] + 20 * ui.s
    assert layout.label_above and layout.label_width >= caption_width
    assert "(m)" in painted and "(°)" in painted
    assert frame(640, language)[0].label_above is False
