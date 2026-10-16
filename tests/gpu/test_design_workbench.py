"""Exercise the native parameter editor on the production workspace."""

from copy import deepcopy

import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.editor import SpecificationEditor
from mojive.design.native.workbench import NativeWorkbench
from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore, read_spec

pytestmark = pytest.mark.gpu


@pytest.fixture(params=(1, 2))
def workbench(request, backend_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(request.param))
    store = SpecificationStore(tmp_path / "current-spec.json")
    store.write(DEFAULT, save=True)
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(store.saved_path))
    with build_workspace(
        resolve("joint_types"),
        renderer=backend_name,
        width=1920 if request.param == 1 else 3200,
        height=1080 if request.param == 1 else 1800,
        show_window=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), threaded_physics=False, debug_server=False
        ),
    ) as viewer:
        runtime = viewer.app.design
        runtime.observe_components = True
        editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
        bench = runtime.workbench = NativeWorkbench(editor, runtime, runtime)
        viewer.app.set_language("zh_CN")
        events = []
        process = viewer.window._input.process_inputs

        def inputs():
            process()
            for method, args in events:
                getattr(imgui.get_io(), method)(*args)
            events.clear()
            imgui.get_io().delta_time = 1 / 60

        monkeypatch.setattr(viewer.window._input, "process_inputs", inputs)

        def frame(count=3):
            for _ in range(count):
                viewer.sync()

        def event(method, *args):
            events.append((method, args))
            frame(1)

        def click(key):
            x, y, w, h = bench.hits[key]
            event("add_mouse_pos_event", x + w / 2, y + h / 2)
            event("add_mouse_button_event", 0, True)
            event("add_mouse_button_event", 0, False)
            frame()

        def enter(key, value):
            click(key)
            for method, args in (
                ("add_key_event", (imgui.Key.mod_ctrl, True)),
                ("add_key_event", (imgui.Key.a, True)),
                ("add_key_event", (imgui.Key.a, False)),
                ("add_key_event", (imgui.Key.mod_ctrl, False)),
                ("add_input_characters_utf8", (str(value),)),
                ("add_key_event", (imgui.Key.enter, True)),
                ("add_key_event", (imgui.Key.enter, False)),
            ):
                event(method, *args)
            frame()

        viewer.frame, viewer.event, viewer.click, viewer.enter = frame, event, click, enter
        frame(8)
        yield viewer, bench, runtime, store


def test_all_component_navigation_and_native_references(workbench):
    viewer, bench, runtime, _ = workbench
    bench.page = "设计规范"
    for part in runtime.catalog:
        bench.part = part
        bench.navigate(viewer.app.workspace, part)
        viewer.frame(2)
        # Every type uses a live native reference, including types not mounted
        # in the current production panel (for example the future agent panel).
        assert part in bench.specimens.last
        assert all(value == value for value in bench.specimens.last[part]["frame"])


def test_native_parameter_input_reset_and_shared_production_instances(workbench):
    viewer, bench, runtime, store = workbench
    bench.part = "input_number"
    bench.field_search = "高度"
    bench.navigate(viewer.app.workspace)
    viewer.frame(4)
    key = "value/parts/input_number/height"
    viewer.click(key)
    viewer.event("add_key_event", imgui.Key.mod_ctrl, True)
    viewer.event("add_key_event", imgui.Key.a, True)
    viewer.event("add_key_event", imgui.Key.a, False)
    viewer.event("add_key_event", imgui.Key.mod_ctrl, False)
    viewer.event("add_input_characters_utf8", "40")
    viewer.event("add_key_event", imgui.Key.enter, True)
    viewer.event("add_key_event", imgui.Key.enter, False)
    viewer.frame(3)
    assert runtime.resolved_part("input_number", "height") == 40
    assert store.latest()["spec"]["parts"]["input_number"]["height"] == 40
    numbers = [
        item
        for item in viewer.app.workspace.ui.design_primitives.observed
        if "input_number" in item["parts"]
    ]
    assert len(numbers) > 3
    assert all(item["hit"][3] == 40 * viewer.window.style_scale for item in numbers)
    viewer.click("reset/parts/input_number/height")
    assert runtime.resolved_part("input_number", "height") == 28
    viewer.click("undo")
    assert runtime.resolved_part("input_number", "height") == 40
    viewer.click("redo")
    assert runtime.resolved_part("input_number", "height") == 28


def test_external_live_edit_and_workbench_capture(workbench):
    viewer, bench, runtime, store = workbench
    spec = deepcopy(DEFAULT)
    spec["parts"]["tree_row"] = {"height": 36, "rowGap": 6, "radius": 10}
    store.write(spec)
    runtime.poll(force=True)
    bench.part = "tree_row"
    bench.navigate(viewer.app.workspace)
    viewer.frame(5)
    rows = [item for item in runtime.observed if "tree_row" in item["parts"]]
    assert len(rows) > 5
    assert all(item["hit"][3] == 36 * viewer.window.style_scale for item in rows)
    store.write(DEFAULT)
    runtime.poll(force=True)
    viewer.frame(3)
    assert all(
        item["hit"][3] == 28 * viewer.window.style_scale
        for item in runtime.observed
        if "tree_row" in item["parts"]
    )
    from pathlib import Path

    target = Path("output/design-workbench")
    target.mkdir(parents=True, exist_ok=True)
    for page in ("交互原型", "设计规范"):
        bench.page = page
        viewer.frame(4)
        future = viewer.app.request_capture_async(surface="window", memory=True)
        viewer.sync()
        Image.fromarray(future.result(timeout=3)["image"]).save(
            target / f"native-{page}-{viewer.window.style_scale:g}x.png"
        )


@pytest.mark.parametrize("scale", (1, 2))
def test_detached_editor_first_frame_and_mode_switch(scale, backend_name, monkeypatch, tmp_path):
    """A second window may own the context before the scene's first upload."""
    import numpy as np

    from mojive.design.native.window import WorkbenchWindow, run_workbench

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    store = SpecificationStore(tmp_path / "design.json")
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(store.saved_path))
    store.write(DEFAULT, save=True)
    with build_workspace(
        resolve("joint_types"),
        renderer=backend_name,
        show_window=False,
        width=1600 * scale,
        height=900 * scale,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), threaded_physics=False, debug_server=False
        ),
    ) as viewer:
        runtime = viewer.app.design
        editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
        bench = runtime.workbench = NativeWorkbench(editor, runtime, runtime)
        with WorkbenchWindow(
            bench, height=900 * scale, scale=scale, visible=False, renderer=backend_name
        ) as window:
            run_workbench(viewer, window, max_frames=8)
            assert np.asarray(viewer.window.read_frame()).std() > 10
            assert np.asarray(window.frame(readback=True)).std() > 10
            assert bench.detached
            events = []
            process = window.window._input.process_inputs

            def inputs():
                process()
                for method, args in events:
                    getattr(imgui.get_io(), method)(*args)
                events.clear()

            monkeypatch.setattr(window.window._input, "process_inputs", inputs)
            x, y, w, h = bench.hits["window-mode"]
            for method, args in (
                ("add_mouse_pos_event", (x + w / 2, y + h / 2)),
                ("add_mouse_button_event", (0, True)),
                ("add_mouse_button_event", (0, False)),
            ):
                events.append((method, args))
                run_workbench(viewer, window, max_frames=1)
            run_workbench(viewer, window, max_frames=5)
            assert not bench.detached
            assert viewer.app.preferences.get("workbench_detached") is False
            assert bench._viewport_rect is not None
            bench.requested_detached = True
            run_workbench(viewer, window, max_frames=5)
            assert bench.detached
            assert viewer.app.preferences.get("workbench_detached") is True
            assert viewer.app.session is viewer.session
            assert np.asarray(viewer.window.read_frame()).std() > 10


def test_alt_pick_model_group_and_full_popup_caption(workbench):
    viewer, bench, runtime, _ = workbench
    bench.part = "timeline_model_group"
    bench.navigate(viewer.app.workspace)
    viewer.frame(4)
    ui = viewer.app.workspace.ui
    x, y, w, h = ui.hits["timeline-model"]
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_key_event", imgui.Key.mod_alt, True)
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.event("add_key_event", imgui.Key.mod_alt, False)
    viewer.frame()
    assert "timeline_model_group" in bench.selection_parts
    viewer.event("add_key_event", imgui.Key.escape, True)
    viewer.event("add_key_event", imgui.Key.escape, False)
    viewer.frame()
    viewer.click("parent/timeline_model_group")
    bench.field_search = "列距"
    viewer.frame()
    viewer.click("label/parts/timeline_model_group/columnGap")
    assert bench.field_highlights
    assert all(r["rect"][2] == pytest.approx(8 * ui.s) for r in bench.field_highlights)
    x, y, w, h = ui.hits["timeline-model"]
    viewer.event("add_mouse_pos_event", x + w / 2, y + h / 2)
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.frame()
    choices = [
        r
        for r in runtime.observed
        if r["key"].startswith("timeline-model-choice-") and "select_option" in r["parts"]
    ]
    assert choices
    assert all(not r["clippedLabel"] for r in choices), choices
    assert any(r.get("label") == "joint_types" for r in choices)


def test_alt_pick_owns_button_press_through_release(workbench):
    viewer, bench, _runtime, _ = workbench
    ui = viewer.app.workspace.ui
    x, y, width, height = ui.hits["play"]
    viewer.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    assert not viewer.app.workspace.playing
    viewer.event("add_key_event", imgui.Key.mod_alt, True)
    viewer.event("add_mouse_button_event", 0, True)
    assert "icon_button" in bench.selection_parts
    # Releasing Alt before the mouse must still leave this gesture owned by the editor.
    viewer.event("add_key_event", imgui.Key.mod_alt, False)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.frame()
    assert not viewer.app.workspace.playing
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    assert viewer.app.workspace.playing


def test_live_group_margins_keep_children_inside_their_frames(workbench):
    viewer, bench, runtime, _ = workbench
    for part in ("viewport_toolbar", "transport"):
        bench.editor.change(("parts", part, "marginLeft"), 16)
        bench.editor.change(("parts", part, "marginRight"), 12)
    runtime.poll(force=True)
    viewer.frame(4)
    for part in ("viewport_toolbar", "transport"):
        row = next(record for record in runtime.observed if part in record["parts"])
        x, y, width, height = row["hit"]
        assert row["fields"]
        for fx, fy, fw, fh in row["fields"]:
            assert x <= fx <= fx + fw <= x + width + 0.5
            assert y <= fy <= fy + fh <= y + height + 0.5


def test_reset_click_remains_valid_when_the_selected_slider_changes(workbench):
    from mojive.design.editor import MISSING, path_value

    viewer, bench, runtime, _ = workbench
    bench.part, bench.field_search = "lane_label", "gap"
    bench.navigate(viewer.app.workspace)
    viewer.frame(4)
    viewer.click("reset/parts/lane_label/columnGap")
    assert bench.selected_field == ("parts", "lane_label", "columnGap")
    viewer.click("reset/parts/lane_label/gap")
    assert path_value(bench.editor.spec, ("parts", "lane_label", "gap")) is MISSING
    assert runtime.resolved_part("lane_label", "gap") == 14


def test_shared_and_layout_fields_select_and_highlight_their_consumers(workbench):
    viewer, bench, _runtime, _ = workbench
    bench.mode = "共享规范"
    viewer.frame(4)
    viewer.click("label/tokens/body_font_size")
    assert bench.selected_field == ("tokens", "body_font_size")
    assert bench.field_highlights
    assert all(r["field"] in ("fontSize", "lineHeight") for r in bench.field_highlights)
    bench.mode = "整体布局"
    viewer.frame(4)
    viewer.click("label/layout/inspector_width")
    assert bench.selected_field == ("layout", "inspector_width")
    assert bench.field_highlights
    assert all(r["key"] == "panel-Inspector" for r in bench.field_highlights)


@pytest.mark.parametrize("detached", (False, True))
def test_statusbar_alt_pick_and_live_geometry(workbench, backend_name, detached, monkeypatch):
    from pathlib import Path

    from mojive.design.native.window import WorkbenchWindow

    viewer, bench, runtime, _ = workbench
    scale = viewer.window.style_scale
    with WorkbenchWindow(
        bench,
        height=round(900 * scale),
        scale=scale,
        visible=False,
        detached=detached,
        renderer=backend_name,
    ) as window:
        viewer.frame(4)

        def alt_click(x, y):
            viewer.event("add_mouse_pos_event", x, y)
            viewer.event("add_key_event", imgui.Key.mod_alt, True)
            viewer.event("add_mouse_button_event", 0, True)
            viewer.event("add_mouse_button_event", 0, False)
            viewer.event("add_key_event", imgui.Key.mod_alt, False)
            viewer.frame()

        status = imgui.internal.find_window_by_name("Status###application_status")
        alt_click(status.pos.x + status.size.x / 2, status.pos.y + scale)
        assert bench.part == "statusbar"

        def record(part):
            return next(
                item
                for item in runtime.observed
                if part in item["parts"] and item["windowId"] == status.id_
            )

        context = record("status_context")
        state_x, state_y, state_width, state_height = context["fields"][0]
        alt_click(state_x + state_width / 2, state_y + state_height / 2)
        assert bench.part == "status_context"
        assert "statusbar" in bench.selection_parts
        hints_before = record("status_hints")["hit"]
        metrics_before = record("status_metrics")["hit"]
        bench.editor.change(("parts", "status_context", "marginLeft"), 20)
        runtime.poll(force=True)
        viewer.frame(4)
        assert record("status_context")["fields"][0][0] == pytest.approx(state_x + 20 * scale)
        assert record("status_hints")["hit"][0] == pytest.approx(hints_before[0] + 20 * scale)
        assert record("status_metrics")["hit"] == pytest.approx(metrics_before)
        for part in ("status_hints", "status_metrics", "status_separator", "badge"):
            x, y, width, height = record(part)["hit"]
            alt_click(x + width / 2, y + height / 2)
            assert part in bench.selection_parts
            assert "statusbar" in bench.selection_parts
            if part == "status_hints":
                assert "status_context" in bench.selection_parts

        metrics_left = record("status_metrics")["hit"][0]
        metric = next(
            item
            for item in runtime.observed
            if "badge" in item["parts"]
            and item["windowId"] == status.id_
            and item["hit"][0] >= metrics_left
        )
        x, y, width, height = metric["hit"]
        mode = viewer.app._status_metric_mode
        alt_click(x + width / 2, y + height / 2)
        assert viewer.app._status_metric_mode == mode
        assert "status_metrics" in bench.selection_parts
        viewer.event("add_mouse_button_event", 0, True)
        viewer.event("add_mouse_button_event", 0, False)
        assert viewer.app._status_metric_mode != mode

        bench.part, bench.field_search = "statusbar", "高度"
        viewer.frame()
        editor_event = viewer.event
        if detached:
            events = []
            process = window.window._input.process_inputs

            def inputs():
                process()
                for method, args in events:
                    getattr(imgui.get_io(), method)(*args)
                events.clear()

            monkeypatch.setattr(window.window._input, "process_inputs", inputs)

            def editor_event(method, *args):
                events.append((method, args))
                window.frame()

            window.frame()
        x, y, width, height = bench.hits["value/parts/statusbar/height"]
        for method, args in (
            ("add_mouse_pos_event", (x + width / 2, y + height / 2)),
            ("add_mouse_button_event", (0, True)),
            ("add_mouse_button_event", (0, False)),
            ("add_key_event", (imgui.Key.mod_ctrl, True)),
            ("add_key_event", (imgui.Key.a, True)),
            ("add_key_event", (imgui.Key.a, False)),
            ("add_key_event", (imgui.Key.mod_ctrl, False)),
            ("add_input_characters_utf8", ("56",)),
            ("add_key_event", (imgui.Key.enter, True)),
            ("add_key_event", (imgui.Key.enter, False)),
        ):
            editor_event(method, *args)
        assert bench.editor.spec["parts"]["statusbar"]["height"] == 56
        bench.editor.change(("parts", "statusbar", "paddingLeft"), 24)
        bench.editor.change(("parts", "badge", "fontSize"), 18)
        runtime.poll(force=True)
        viewer.frame(4)
        bar = record("statusbar")
        assert bar["hit"][3] == pytest.approx(56 * scale)
        assert bar["content"][0] - bar["hit"][0] == pytest.approx(24 * scale)
        assert record("badge")["hit"][3] >= 18 * scale

        state_before = record("status_context")["fields"][0]
        hints_before = record("status_hints")["hit"]
        metrics_before = record("status_metrics")["hit"]
        bench.editor.change(("parts", "status_context", "marginBottom"), 8)
        runtime.poll(force=True)
        viewer.frame(4)
        assert record("status_context")["fields"][0][1] == pytest.approx(
            state_before[1] - 4 * scale
        )
        assert record("status_hints")["hit"][1] == pytest.approx(hints_before[1] - 4 * scale)
        assert record("status_metrics")["hit"] == pytest.approx(metrics_before)

        state_x, state_y, state_width, state_height = record("status_context")["fields"][0]
        alt_click(state_x + state_width / 2, state_y + state_height / 2)
        assert bench.part == "status_context"
        bench.selected_field = ("parts", "status_context", "marginLeft")
        viewer.frame()
        assert bench.field_highlights
        target = Path("output/ui_design/native") / backend_name / "statusbar"
        target.mkdir(parents=True, exist_ok=True)
        suffix = f"{'detached' if detached else 'embedded'}-{scale:g}x.png"
        future = viewer.app.request_capture_async(surface="window", memory=True)
        viewer.sync()
        Image.fromarray(future.result(timeout=3)["image"]).save(target / suffix)
        if detached:
            Image.fromarray(window.frame(readback=True)[::-1]).save(target / ("editor-" + suffix))
