"""Layer sections keep native visibility switches and publisher masks independent."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import ViewportLayers
from mojive.ui.layers import LAYER_LABELS
from mojive.ui.workspace.scene_panels import layers_panel
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=(1, 2))
def layers(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (420 * request.param, 1000 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(
        SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    items = (
        SimpleNamespace(name="trajectory", primitives=(object(),)),
        SimpleNamespace(name="canvas2d:labels", primitives=(object(),)),
        SimpleNamespace(name="ui.selection", primitives=(object(),)),
    )
    app = SimpleNamespace(viewport_layers=ViewportLayers(hidden_debug_layers=("retained",)))
    writes = []

    def update(value):
        writes.append(value)
        app.viewport_layers = value

    ctx = SimpleNamespace(
        tr=str,
        set_viewport_layers=update,
        backend=SimpleNamespace(debug=SimpleNamespace(layers=lambda: items)),
    )
    workspace = SimpleNamespace(ui=ui, s=ui.s, app=app, _panel_context=ctx)

    def paint():
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size(io.display_size)
        imgui.begin("Layers", flags=imgui.WindowFlags_.no_title_bar)
        ui.hits.clear()
        layers_panel(workspace, 20 * ui.s, 20 * ui.s, 360 * ui.s)
        imgui.end()
        imgui.render()

    def click(key):
        x, y, width, height = ui.hits[key]
        io.add_mouse_pos_event(x + width / 2, y + height / 2)
        paint()
        io.add_mouse_button_event(0, True)
        paint()
        io.add_mouse_button_event(0, False)
        paint()
        paint()

    try:
        for _ in range(3):
            paint()
        yield SimpleNamespace(app=app, ui=ui, writes=writes, paint=paint, click=click)
    finally:
        imgui.destroy_context(context)


@pytest.mark.parametrize("name", [name for name, _ in LAYER_LABELS])
def test_grouped_visibility_switch_changes_only_its_existing_flag(layers, name):
    original = layers.app.viewport_layers
    layers.click("layer-" + name)
    assert layers.app.viewport_layers == replace(original, **{name: False})
    assert len(layers.writes) == 1


def test_collapsing_debug_section_preserves_parent_and_named_visibility(layers):
    original = layers.app.viewport_layers
    assert "drawing-layer-trajectory" in layers.ui.hits
    assert "drawing-layer-ui.selection" not in layers.ui.hits
    layers.click("section-Debug display")
    assert "layer-debug_3d" not in layers.ui.hits
    assert "drawing-layer-trajectory" not in layers.ui.hits
    assert layers.app.viewport_layers == original
    assert layers.writes == []
    layers.click("section-Debug display")
    layers.click("drawing-layer-trajectory")
    assert layers.app.viewport_layers == replace(
        original, hidden_debug_layers=("retained", "trajectory")
    )
    layers.click("layer-debug_3d")
    state = layers.app.viewport_layers
    layers.click("drawing-layer-trajectory")
    assert layers.app.viewport_layers == state
    assert state.hidden_debug_layers == ("retained", "trajectory")


def test_editing_and_selection_sections_have_separate_visibility_state(layers):
    original = layers.app.viewport_layers
    layers.click("section-Editing aids")
    assert "layer-gizmos" not in layers.ui.hits
    assert "layer-selection" in layers.ui.hits
    layers.click("layer-selection")
    assert layers.app.viewport_layers == replace(original, selection=False)
    layers.click("section-Editing aids")
    assert all("layer-" + name in layers.ui.hits for name, _ in LAYER_LABELS)
