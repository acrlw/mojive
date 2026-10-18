"""Production audit fixtures open actual Timeline settings through native input."""

from itertools import product
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive.config import RecordingConfig
from mojive.design.groups import group_scope
from mojive.design.help import get_part_help
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.design.support import native_support
from mojive.tools.design_audit_scenes import ProductionScenes
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.timeline_panel import NativeTimelineToolbar
from mojive.ui.workspace.widgets import UI


@pytest.fixture(params=tuple(product((1, 1.2, 1.25, 2.25), ("form_popup", "timeline_form_popup"))))
def timeline_audit(request, tmp_path):
    scale, part = request.param
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size, io.delta_time = (1000 * scale, 700 * scale), 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    ui = UI(
        SimpleNamespace(style_scale=scale, font_report=SimpleNamespace(cjk_path=None)),
        portable_fonts=True,
    )
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.observe_components = True
    ui.design = DesignPresentation(ui, runtime)
    workspace = SimpleNamespace(
        ui=ui, s=scale, menus=SimpleNamespace(requested_path=(), dialog_pending=None)
    )
    session = SimpleNamespace(state_take_pause_at_end=True)
    ctx = PanelContext(session, None, style_scale=scale, recording_config=RecordingConfig())
    toolbar = NativeTimelineToolbar(workspace)

    def frames(count=1):
        for _ in range(count):
            imgui.new_frame()
            runtime.begin_frame()
            ui.design.begin_frame()
            ui.hits.clear()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Timeline", flags=imgui.WindowFlags_.no_title_bar)
            imgui.set_cursor_screen_pos((40 * scale, 50 * scale))
            with group_scope("Timeline"):
                toolbar._draw_settings(ctx)
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frames()

    def key(value):
        event("add_key_event", value, True)
        event("add_key_event", value, False)

    audit = SimpleNamespace(
        viewer=SimpleNamespace(app=SimpleNamespace(workspace=workspace), session=session),
        input=SimpleNamespace(frames=frames, event=event, key=key),
        bench=SimpleNamespace(part=part, navigate=lambda _workspace: None),
        production_raw_records=lambda: list(ui.design.primitives.observed) + list(runtime.observed),
    )
    scenes = ProductionScenes(audit, tmp_path / "scene.xml")
    frames(3)
    try:
        yield scenes, part
    finally:
        imgui.destroy_context(context)


def test_form_fixture_clicks_production_button_and_reopens_after_editor(timeline_audit):
    scenes, part = timeline_audit
    scenes.navigate(part)
    audit = scenes.audit
    assert scenes.form_popup_key == "timeline-settings"
    assert audit.scene_state["state"] == "expanded production Timeline settings form"
    record = next(
        row for row in audit.production_raw_records() if row["key"] == "timeline-settings-popup"
    )
    stack = imgui.get_current_context().open_popup_stack
    popup = stack[len(stack) - 1].window
    assert record["windowId"] == popup.id_
    assert popup.flags & imgui.WindowFlags_.popup
    assert part in record["parts"]
    assert all(not row["key"].startswith("reference-") for row in audit.production_raw_records())
    audit.input.before_editor()
    assert len(imgui.get_current_context().open_popup_stack) == 0
    assert not any(
        row["key"] == "timeline-settings-popup" for row in audit.production_raw_records()
    )
    audit.input.after_editor()
    assert imgui.get_current_context().open_popup_stack
    assert any(row["key"] == "timeline-settings-popup" for row in audit.production_raw_records())


@pytest.mark.parametrize("part", ("form_popup", "timeline_form_popup"))
def test_form_popup_help_and_geometry_bind_to_the_actual_production_form(part):
    assert get_part_help(part)["preview"]["area"] == "时间轴"
    fields = native_support()[part]["fields"]
    for field in (
        "paddingX",
        "paddingY",
        "paddingLeft",
        "paddingRight",
        "paddingTop",
        "paddingBottom",
        "radius",
        "rowGap",
        "columnGap",
        "groupGap",
    ):
        assert fields[field]["status"] == "applied"
        assert fields[field]["consumers"] == [
            {"source": "ui/workspace/form_popup.py", "description": "生产组件尺寸与排版"}
        ]
