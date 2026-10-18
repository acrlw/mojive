"""Exercise the native parameter editor on the production workspace."""

from copy import deepcopy
from itertools import pairwise

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


def shortcut_modifier():
    return imgui.Key.mod_super if imgui.get_io().config_mac_osx_behaviors else imgui.Key.mod_ctrl


def assert_auto_status_phrases(runtime, window, scale):
    """Auto columns reserve the displayed ink advances and equal divider padding."""
    parts = (
        "status_backend",
        "status_steps",
        "status_time",
        "status_delta",
        "status_physics",
        "status_render",
    )
    rows = [
        row
        for row in runtime.observed
        if row["windowId"] == window.id_ and row["key"] in {"application-" + part for part in parts}
    ]
    assert len(rows) == 5
    for row in rows:
        assert row["compact"]
        x, _y, width, _height = row["hit"]
        label, value, unit = row["labelRect"], row["valueTextRect"], row["unitRect"]
        assert label[0] - x == pytest.approx(2 * scale)
        assert row["valueRect"][2] == pytest.approx(value[2])
        assert row["valueRect"][2] <= row["formatBudgetWidth"] + 0.01
        if row["key"] != "application-status_backend":
            assert value[0] - label[0] - label[2] == pytest.approx(4 * scale)
            assert unit[0] - value[0] - value[2] == pytest.approx(row["unitGap"])
        assert x + width - row["unitRight"] == pytest.approx(2 * scale)
    for first, second in pairwise(rows):
        expected = first["rightGap"] + scale + second["leftGap"]
        assert second["hit"][0] - first["hit"][0] - first["hit"][2] == pytest.approx(expected)
        assert first["rightGap"] == second["leftGap"] == pytest.approx(6 * scale)


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
                ("add_key_event", (shortcut_modifier(), True)),
                ("add_key_event", (imgui.Key.a, True)),
                ("add_key_event", (imgui.Key.a, False)),
                ("add_key_event", (shortcut_modifier(), False)),
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


def test_panel_local_groups_and_backend_readout_are_directly_pickable(workbench, backend_name):
    viewer, bench, runtime, _store = workbench
    shell = viewer.app.workspace
    for part, key in (
        ("cameras_button_row", "camera-Front"),
        ("control_button_row", "reset-actuators"),
        ("inspector_tabs", "tab-Properties"),
        ("scene_type_filters", "scene-filter-all"),
        ("status_backend", None),
    ):
        bench.part = part
        bench.navigate(shell, part)
        viewer.frame(6)
        if key is None:
            rect = next(
                row["hit"] for row in runtime.observed if row["key"] == "application-" + part
            )
        else:
            rect = shell.ui.hits[key]
        previous_tab = shell.inspector_tab
        x, y, width, height = rect
        viewer.event("add_mouse_pos_event", x + width / 2, y + height / 2)
        viewer.event("add_key_event", imgui.Key.mod_alt, True)
        viewer.event("add_mouse_button_event", 0, True)
        viewer.event("add_mouse_button_event", 0, False)
        viewer.event("add_key_event", imgui.Key.mod_alt, False)
        viewer.frame(3)
        expected = {
            "inspector_tabs": "inspector_tabs_tab",
            "cameras_button_row": "cameras_button_row_button",
            "control_button_row": "control_button_row_button",
            "scene_type_filters": "scene_type_filters_filter",
        }.get(part, part)
        assert bench.part == expected
        assert part in bench.selection_parts
        assert bench.native_instances
        assert all(expected in record["parts"] for record in bench.native_instances)
        if part == "inspector_tabs":
            assert shell.inspector_tab == previous_tab
    from pathlib import Path

    bench.field_group = "指标布局"
    bench.selected_field = ("parts", "status_backend", "rightGap")
    viewer.frame(4)
    target = Path("output/panel-groups") / backend_name / f"{shell.s:g}x" / "workbench-backend.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(viewer.window.read_frame()[::-1]).save(target)


def test_native_parameter_input_reset_and_shared_production_instances(workbench):
    viewer, bench, runtime, store = workbench
    bench.part = "input_number"
    bench.field_search = "高度"
    bench.navigate(viewer.app.workspace)
    viewer.frame(4)
    key = "value/parts/input_number/height"
    viewer.click(key)
    viewer.event("add_key_event", shortcut_modifier(), True)
    viewer.event("add_key_event", imgui.Key.a, True)
    viewer.event("add_key_event", imgui.Key.a, False)
    viewer.event("add_key_event", shortcut_modifier(), False)
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
    assert {item["styleFamily"] for item in numbers} >= {"inspector", "timeline"}
    assert {item["key"] for item in numbers} >= {
        "##body-gravity-compensation",
        "##timeline-range-start",
        "##timeline-range-end",
    }
    assert all(item["hit"][3] == 40 * viewer.window.style_scale for item in numbers)
    viewer.click("reset/parts/input_number/height")
    assert runtime.resolved_part("input_number", "height") == 28
    viewer.click("undo")
    assert runtime.resolved_part("input_number", "height") == 40
    viewer.click("redo")
    assert runtime.resolved_part("input_number", "height") == 28


def test_status_unit_inset_and_divider_sides_edit_separately(workbench, backend_name):
    from pathlib import Path

    viewer, bench, runtime, _store = workbench
    shell = viewer.app.workspace
    bench.part, bench.field_group = "status_physics", "指标布局"
    bench.navigate(shell)
    viewer.frame(4)
    assert "value/parts/status_physics/paddingRight" not in bench.hits
    viewer.click("field-advanced/指标布局")
    for field, value in (("paddingRight", 3), ("rightGap", 9), ("leftGap", 2)):
        viewer.enter(f"value/parts/status_physics/{field}", value)
        assert runtime.resolved_part("status_physics", field) == value
    bench.part = "status_render"
    viewer.frame(4)
    viewer.enter("value/parts/status_render/leftGap", 5)
    assert runtime.resolved_part("status_physics", "rightGap") == 9
    assert runtime.resolved_part("status_render", "leftGap") == 5
    bench.part = "status_physics"
    bench.selected_field = ("parts", "status_physics", "paddingRight")
    viewer.frame(4)
    target = Path("output/status-spacing") / backend_name / f"{shell.s:g}x" / "workbench.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(viewer.window.read_frame()[::-1]).save(target)


def test_semantic_icon_groups_and_text_colors_edit_independently(workbench):
    from pathlib import Path

    from mojive.design.editor import MISSING

    viewer, bench, runtime, _ = workbench
    ui = viewer.app.workspace.ui

    def icons():
        return {
            record["key"]: tuple(record["iconRect"])
            for record in ui.design_primitives.observed
            if record.get("iconRect")
        }

    before = icons()
    assert {"rail-Scene", "tool-move", "projection-button"} <= before.keys()
    bench.part, bench.field_search = "rail", "iconSize"
    viewer.frame()
    viewer.enter("value/parts/rail/iconSize", 28)
    after = icons()
    assert after["rail-Scene"][2] == 28 * ui.s
    assert after["tool-move"][2:] == before["tool-move"][2:]
    assert after["projection-button"][2:] == before["projection-button"][2:]

    bench.part, bench.field_search = "toolbox", "iconSize"
    viewer.frame()
    viewer.enter("value/parts/toolbox/iconSize", 14)
    after = icons()
    assert after["tool-move"][2] == 14 * ui.s
    assert after["rail-Scene"][2] == 28 * ui.s

    bench.part, bench.field_search = "menu_trigger", "textColor"
    viewer.frame()
    viewer.enter("value/parts/menu_trigger/textColor", "#68b5b7ff")
    assert runtime.resolved_part("menu_trigger", "textColor") == "#68b5b7ff"
    assert runtime.resolved_part("panel_header", "textColor") == "inherit"
    # A second family is installed once; unrelated color edits preserve it.
    bench.field_search = "fontFamily"
    viewer.frame()
    viewer.click("value/parts/menu_trigger/fontFamily")
    stack = imgui.get_current_context().open_popup_stack
    assert len(stack)
    popup = stack[len(stack) - 1].window
    choices = runtime.catalog["menu_trigger"]["fields"]["fontFamily"]["choices"]
    pitch = (popup.dc.cursor_pos.y - popup.dc.cursor_start_pos.y) / len(choices)
    viewer.event(
        "add_mouse_pos_event",
        (popup.work_rect.min.x + popup.work_rect.max.x) / 2,
        popup.dc.cursor_start_pos.y + (choices.index("portable") + 0.4) * pitch,
    )
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.frame(5)
    assert runtime.resolved_part("menu_trigger", "fontFamily") == "portable"
    faces = len(imgui.get_io().fonts.fonts)
    portable = ui._font_families["portable"][1][0]
    assert ui.resolve_font(ui.component_style("menu_trigger")) is portable
    bench.field_search = "textColor"
    viewer.frame()
    viewer.enter("value/parts/menu_trigger/textColor", "#e8b04fff")
    assert len(imgui.get_io().fonts.fonts) == faces

    target = Path("output/ui-customization/groups") / viewer.backend.caps.name
    target.mkdir(parents=True, exist_ok=True)
    future = viewer.app.request_capture_async(surface="window", memory=True)
    viewer.sync()
    Image.fromarray(future.result(timeout=3)["image"]).save(
        target / f"independent-icons-and-menu-{ui.s:g}x.png"
    )
    for part, field in (
        ("rail", "iconSize"),
        ("toolbox", "iconSize"),
        ("menu_trigger", "textColor"),
        ("menu_trigger", "fontFamily"),
    ):
        bench.editor.change(("parts", part, field), MISSING)
    runtime.poll(force=True)
    viewer.frame()
    restored = icons()
    assert restored["rail-Scene"][2:] == before["rail-Scene"][2:]
    assert restored["tool-move"][2:] == before["tool-move"][2:]


def test_settings_navigation_pick_and_appearance_stay_in_their_group(
    workbench, backend_name, monkeypatch
):
    from pathlib import Path

    from mojive.ui.workspace.settings_panel import CATEGORIES

    viewer, bench, runtime, store = workbench
    workspace, scale = viewer.app.workspace, viewer.window.style_scale
    ui = workspace.ui
    page_lines, page_text = [], []
    original_line, original_text = ui.line, ui.text

    def in_page():
        window = imgui.get_current_context().current_window
        return window is not None and "Reference settings content" in window.name

    def line(x, y, width):
        if in_page():
            page_lines.append((x, y, width))
        return original_line(x, y, width)

    def text(*args, **options):
        if in_page():
            caption = args[2] if len(args) > 2 else options["text"]
            size = args[4] if len(args) > 4 else options.get("size", 14)
            page_text.append((str(caption), size))
        return original_text(*args, **options)

    monkeypatch.setattr(ui, "line", line)
    monkeypatch.setattr(ui, "text", text)
    workspace.activate("Settings")
    workspace.settings.show_category("Camera")
    viewer.frame(5)

    def click_control(key, *, alt=False):
        x, y, width, height = ui.hits[key]
        viewer.event("add_mouse_pos_event", x + width / 2, y + height / 2)
        if alt:
            viewer.event("add_key_event", imgui.Key.mod_alt, True)
        viewer.event("add_mouse_button_event", 0, True)
        viewer.event("add_mouse_button_event", 0, False)
        if alt:
            viewer.event("add_key_event", imgui.Key.mod_alt, False)
        viewer.frame(3)

    def appearance():
        return {
            record["key"]: (record.get("fontSize"), record.get("iconSize"))
            for record in ui.design_primitives.observed
            if record["key"].startswith(("menu-", "entity-", "tool-", "timeline-"))
        }

    def navigation():
        return {
            record["key"]: record
            for record in ui.design_primitives.observed
            if record["key"] in {"settings-" + category for category in CATEGORIES}
        }

    target = Path("output/settings-groups") / backend_name / f"{scale:g}x"
    target.mkdir(parents=True, exist_ok=True)

    def capture(name, *, bounds=False):
        show_bounds, selected = bench.show_bounds, bench.selected_field
        bench.show_bounds, bench.selected_field = bounds, None
        viewer.frame(3)
        future = viewer.app.request_capture_async(surface="window", memory=True)
        viewer.sync()
        Image.fromarray(future.result(timeout=3)["image"]).save(target / f"{name}.png")
        bench.show_bounds, bench.selected_field = show_bounds, selected
        viewer.frame(2)

    page_lines.clear()
    page_text.clear()
    viewer.frame(1)
    focus = next(
        record for record in runtime.observed if record["key"] == "section-" + ui.translate("Focus")
    )
    zoom = next(
        record for record in runtime.observed if record["key"] == "section-" + ui.translate("Zoom")
    )
    # Headings and whitespace group Settings content; the window header owns its only rule.
    assert not page_lines
    assert not focus["divider"] and not zoom["divider"]
    focus_top, focus_height = focus["hit"][1], focus["hit"][3]
    zoom_top, zoom_height = zoom["hit"][1], zoom["hit"][3]
    first_focus = ui.hits["##camera_focus_margin"]
    last_focus = ui.hits["##camera_focus_easing"]
    first_zoom = ui.hits["##camera_zoom_mode"]
    assert focus_top + focus_height + 4 * scale <= first_focus[1]
    assert last_focus[1] + last_focus[3] + 8 * scale <= zoom_top
    assert zoom_top + zoom_height + 4 * scale <= first_zoom[1]
    original_page_text = tuple(page_text)
    unrelated = appearance()
    assert any(key.startswith("menu-") for key in unrelated)
    assert any(key.startswith("entity-") for key in unrelated)
    capture("camera-page")

    click_control("settings-Camera", alt=True)
    assert bench.part == "settings_nav_button"
    assert "button" in bench.selection_parts
    assert bench.native_instances
    assert all("settings_nav_button" in record["parts"] for record in bench.native_instances)
    assert not any(
        record["key"].startswith(("menu-", "entity-", "inspector-"))
        for record in bench.native_instances
    )
    assert workspace.settings._category == "Camera"
    capture("navigation-selected", bounds=True)

    for field, value in (("fontSize", 20), ("iconSize", 24)):
        bench.field_search = runtime.catalog["settings_nav_button"]["fields"][field]["label"]
        viewer.frame(3)
        viewer.enter(f"value/parts/settings_nav_button/{field}", value)
        assert runtime.resolved_part("settings_nav_button", field) == value
        assert store.latest()["spec"]["parts"]["settings_nav_button"][field] == value
    page_text.clear()
    viewer.frame(1)
    assert tuple(page_text) == original_page_text
    nav = navigation()
    assert len(nav) == len(CATEGORIES)
    assert all(record["fontSize"] == 20 for record in nav.values())
    assert all(record["iconRect"][2:] == (24 * scale, 24 * scale) for record in nav.values())
    assert all(not record["clippedLabel"] for record in nav.values())
    assert appearance() == unrelated
    capture("navigation-custom-appearance")

    # Shared button settings remain reachable through an explicit hierarchy choice.
    bench.field_search = ""
    viewer.frame(3)
    viewer.click("shared-classes")
    viewer.click("parent/button")
    assert bench.part == "button"
    assert any("settings_nav" not in record["parts"] for record in bench.native_instances)


def test_settings_window_and_navigation_widths_edit_save_and_reset_independently(
    workbench, backend_name
):
    from pathlib import Path

    viewer, bench, runtime, store = workbench
    shell, scale = viewer.app.workspace, viewer.window.style_scale
    shell.activate("Settings")
    viewer.frame(5)
    window = imgui.internal.find_window_by_name("Settings")
    imgui.set_window_size("Settings", (920 * scale, 560 * scale))
    viewer.frame(3)
    original = tuple(window.size)

    def navigation():
        return next(row for row in runtime.observed if row["key"] == "settings-navigation")

    bench.part, bench.field_search = "settings_nav", "导航栏宽度"
    viewer.frame(3)
    for width in (180, 280):
        viewer.enter("value/parts/settings_nav/navWidth", width)
        assert navigation()["outer"][2] == pytest.approx(width * scale)
        assert tuple(window.size) == original
        assert store.latest()["spec"]["parts"]["settings_nav"]["navWidth"] == width
    bench.selected_field = ("parts", "settings_nav", "navWidth")
    viewer.frame(3)
    directory = Path("output/settings-width") / backend_name / f"{scale:g}x"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(viewer.window.read_frame()[::-1]).save(directory / "navigation.png")

    bench.part, bench.field_search = "settings_layout", "设置窗口宽度"
    viewer.frame(3)
    for width in (1100, 480, 2400):
        viewer.enter("value/parts/settings_layout/windowWidth", width)
        viewport = imgui.get_main_viewport()
        expected = min(width * scale, viewport.work_size.x - 32 * scale)
        assert window.size.x == pytest.approx(expected, abs=1)
        assert window.size.y == original[1]
        assert window.pos.x + window.size.x <= viewport.work_pos.x + viewport.work_size.x + 1
        assert navigation()["outer"][2] == pytest.approx(280 * scale)
    viewer.click("reset/parts/settings_layout/windowWidth")
    assert tuple(window.size) == original
    viewer.click("undo")
    assert runtime.resolved_part("settings_layout", "windowWidth") == 2400
    viewer.click("redo")
    assert tuple(window.size) == original
    bench.selected_field = ("parts", "settings_layout", "windowWidth")
    viewer.frame(3)
    Image.fromarray(viewer.window.read_frame()[::-1]).save(directory / "window.png")


@pytest.mark.parametrize(
    ("part", "key"),
    (("rail", "rail-Scene"), ("toolbox", "tool-move"), ("timeline_toolbar", "timeline-first")),
)
def test_alt_pick_prefers_the_semantic_group_for_production_controls(workbench, part, key):
    viewer, bench, _runtime, _store = workbench
    ui = viewer.app.workspace.ui
    viewer.frame(3)
    x, y, width, height = ui.hits[key]
    viewer.event("add_mouse_pos_event", x + width / 2, y + height / 2)
    viewer.event("add_key_event", imgui.Key.mod_alt, True)
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.event("add_key_event", imgui.Key.mod_alt, False)
    viewer.frame(3)
    assert bench.part == part
    assert bench.native_instances
    assert all(part in record["parts"] for record in bench.native_instances)


def test_module_appearance_follows_docked_panels_and_application_sidebars(workbench, monkeypatch):
    from mojive.ui.imgui_draw import ImguiDraw2D

    viewer, _bench, runtime, store = workbench
    viewer.app.set_language("en")
    value = deepcopy(DEFAULT)
    value["components"] = {
        "navigation": {"font_family": "portable", "text_color": "#b5d8a0"},
        "inspector": {"font_family": "portable"},
        "chrome": {"font_family": "portable", "text_color": "#e8ba72"},
    }
    value["parts"]["menu_trigger"] = {"textColor": "#ed92bd"}
    store.write(value)
    runtime.poll(force=True)
    painted = {}
    original = ImguiDraw2D.text

    def text(draw, position, tint, caption, **options):
        if caption in ("world", "File", "Scene"):
            painted[caption] = (imgui.get_font().font_id, tint)
        return original(draw, position, tint, caption, **options)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    viewer.frame(5)
    ui = viewer.app.workspace.ui
    portable = ui._font_families["portable"][1][0].font_id
    portable_bold = ui._font_families["portable"][1][1].font_id
    assert {name: painted[name][0] for name in ("world", "File", "Scene")} == {
        "world": portable,
        "File": portable,
        "Scene": portable_bold,
    }
    assert painted["Scene"][1] == pytest.approx((181 / 255, 216 / 255, 160 / 255, 1))
    assert painted["File"][1] == pytest.approx((237 / 255, 146 / 255, 189 / 255, 1))
    scene = next(row for row in runtime.observed if row["key"] == "panel-header-Scene")
    assert scene["styleFamily"] == "navigation"
    assert any(
        row.get("styleFamily") == "inspector" and "inspector_header" in row["parts"]
        for row in runtime.observed
    )


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


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_metrics_digits_units_and_exact_value_in_the_production_bar(
    workbench, backend_name, monkeypatch, language
):
    from pathlib import Path

    import numpy as np

    from mojive.ui.app import status as status_module
    from mojive.ui.imgui_draw import ImguiDraw2D
    from mojive.ui.viewport_widgets.status_metrics import _StatusNumbers

    viewer, bench, runtime, _store = workbench
    viewer.app.set_language(language)
    viewer.app._status_metric_mode = "steps"
    monkeypatch.setattr(viewer.app._frame_rate, "update", lambda elapsed: None)
    values = {"step": 12345, "fps": 59.0, "dt": 0.002, "physics_hz": 0}
    rendered, tooltips, copied, layout = [], [], [], []
    original_status, original_text = status_module.draw_status, ImguiDraw2D.text
    original_paint, original_tooltip = _StatusNumbers.paint, status_module.status_tooltip

    def status(*args, **kwargs):
        kwargs.update(values)
        result = original_status(*args, **kwargs)
        layout[:] = [result]
        return result

    def text(draw, position, tint, caption, **options):
        current = imgui.get_current_context().current_window
        if current and current.name.endswith("###application_status"):
            rendered.append((caption, tuple(position), imgui.calc_text_size(caption).x))
        return original_text(draw, position, tint, caption, **options)

    render_frames = []

    def paint(numbers, field, center_y, tint):
        start = len(rendered)
        original_paint(numbers, field, center_y, tint)
        if field.key == "render":
            render_frames.append((field, rendered[start:]))

    def tooltip(ui, first_line, *action_lines, **options):
        tooltips.append("\n".join((first_line, *action_lines)))
        return original_tooltip(ui, first_line, *action_lines, **options)

    monkeypatch.setattr(status_module, "draw_status", status)
    monkeypatch.setattr(ImguiDraw2D, "text", text)
    monkeypatch.setattr(_StatusNumbers, "paint", paint)
    monkeypatch.setattr(status_module, "status_tooltip", tooltip)
    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    target = (
        Path("output/status-metrics") / backend_name / f"{language}-{viewer.window.style_scale:g}x"
    )
    target.mkdir(parents=True, exist_ok=True)
    bench.show_bounds = False
    bench.selected_field = None
    expected = None
    metric_bounds = []
    strips = []
    for fps in (59.0, 51.1, 1.1, 999.9):
        values["fps"] = fps
        viewer.frame(3)
        _field, commands = render_frames[-1]
        decimal = next(pos for caption, pos, _width in commands if caption == ".")
        unit = next(pos for caption, pos, _width in commands if caption == "FPS")
        digits = [
            (pos[0] + width / 2, pos[1]) for caption, pos, width in commands if caption.isdigit()
        ]
        anchors = (decimal, unit, digits[-2], digits[-1])
        if expected is None:
            expected = anchors
        assert anchors == expected
        status_window = imgui.internal.find_window_by_name("Status###application_status")
        assert_auto_status_phrases(runtime, status_window, viewer.window.style_scale)
        metric_bounds.append(layout[0].metric_rect)
        if fps in (59.0, 51.1):
            record = next(
                row for row in runtime.observed if row["key"] == "application-status_metrics"
            )
            x, y, width, height = record["hit"]
            ratio = viewer.window.pixel_scale
            pixels = np.asarray(viewer.window.read_frame())[::-1]
            strip = Image.fromarray(pixels).crop(
                (
                    round((x - 3) * ratio),
                    round((y - 3) * ratio),
                    round((x + width + 3) * ratio),
                    round((y + height + 3) * ratio),
                )
            )
            strip.save(target / f"fps-{fps:.1f}.png")
            strips.append(strip)
    strips[0].save(
        target / "stable-fps.gif", save_all=True, append_images=strips[1:], duration=750, loop=0
    )
    assert metric_bounds[0] == metric_bounds[1]
    assert metric_bounds[2] != metric_bounds[0] != metric_bounds[3]

    # Positive authoring budgets retain decimal/unit anchors and all neighbor slots.
    bench.editor.change(("parts", "status_metrics", "stepsValueWidth"), 64)
    bench.editor.change(("parts", "status_metrics", "renderValueWidth"), 48)
    runtime.poll(force=True)
    fixed = None
    for fps in (59.0, 51.1, 1.1, 999.9):
        values["fps"] = fps
        viewer.frame(3)
        field, commands = render_frames[-1]
        assert not field.compact
        assert field.value_width == pytest.approx(48 * viewer.window.style_scale)
        anchors = (
            next(pos for caption, pos, _width in commands if caption == "."),
            next(pos for caption, pos, _width in commands if caption == "FPS"),
            layout[0].metric_rect,
        )
        if fixed is None:
            fixed = anchors
        assert anchors == fixed
    rect = layout[0].metric_rect
    for steps in (0, 999, 1000, 999_999, 123_456_789):
        values["step"] = steps
        viewer.frame(2)
        assert layout[0].metric_rect == rect
    x0, y0, x1, y1 = rect
    viewer.event("add_mouse_pos_event", (x0 + x1) / 2, (y0 + y1) / 2)
    viewer.frame(3)
    assert not any("123456789" in caption for caption in tooltips)
    viewer.frame(int(imgui.get_style().hover_delay_normal * 60) + 3)
    assert any("123456789" in caption for caption in tooltips)
    viewer.event("add_mouse_button_event", 1, True)
    viewer.event("add_mouse_button_event", 1, False)
    assert copied == ["123456789"]
    assert viewer.app._status_metric_mode == "steps"
    Image.fromarray(np.asarray(viewer.window.read_frame())[::-1]).save(
        target / "exact-value-hover.png"
    )
    for key in ("delta", "physics", "render", "backend"):
        metric = next(field for field in layout[0].metrics if field.key == key)
        x0, y0, x1, y1 = metric.rect
        viewer.event("add_mouse_pos_event", (x0 + x1) / 2, (y0 + y1) / 2)
        viewer.frame(int(imgui.get_style().hover_delay_normal * 60) + 3)
        assert tooltips[-1].startswith(f"{metric.label} {metric.exact}\n")
        viewer.event("add_mouse_button_event", 1, True)
        viewer.event("add_mouse_button_event", 1, False)
        assert copied[-1] == metric.exact
        viewer.event("add_mouse_button_event", 0, True)
        viewer.event("add_mouse_button_event", 0, False)
        assert viewer.app._status_metric_mode == "steps"


def test_status_metric_spacing_controls_edit_live_geometry(workbench, backend_name, monkeypatch):
    from pathlib import Path

    import numpy as np

    from mojive.ui.app import status as status_module
    from mojive.ui.viewport_widgets.status_metrics import _StatusNumbers

    viewer, bench, runtime, store = workbench
    scale = viewer.window.style_scale
    values = {"dt": 0.002, "fps": 59.0, "physics_hz": 0, "step": 12345, "sim_time": 2.5}
    painted = {}
    original_status, original_paint = status_module.draw_status, _StatusNumbers.paint

    def status(*args, **kwargs):
        kwargs.update(values)
        return original_status(*args, **kwargs)

    def paint(numbers, field, center_y, tint):
        painted[field.key] = (field, numbers.draw.text_size(field.label)[0])
        return original_paint(numbers, field, center_y, tint)

    monkeypatch.setattr(status_module, "draw_status", status)
    monkeypatch.setattr(_StatusNumbers, "paint", paint)
    monkeypatch.setattr(viewer.app._frame_rate, "update", lambda elapsed: None)
    viewer.app._status_metric_mode = "steps"
    bench.part = "status_metrics"
    viewer.frame(3)
    assert "field-section/数字排版" in bench.hits
    assert "field-section/悬停提示" in bench.hits
    target = Path("output/status-metrics/readability") / backend_name / f"{scale:g}x"
    target.mkdir(parents=True, exist_ok=True)

    def capture(name):
        show_bounds, selected = bench.show_bounds, bench.selected_field
        bench.show_bounds, bench.selected_field = False, None
        viewer.frame(3)
        record = next(row for row in runtime.observed if row["key"] == "application-status_metrics")
        x, y, width, height = record["hit"]
        ratio = viewer.window.pixel_scale
        pixels = np.asarray(viewer.window.read_frame())[::-1]
        Image.fromarray(pixels).crop(
            (
                round((x - 3 * scale) * ratio),
                round((y - 3 * scale) * ratio),
                round((x + width + 3 * scale) * ratio),
                round((y + height + 3 * scale) * ratio),
            )
        ).save(target / f"{name}.png")
        bench.show_bounds, bench.selected_field = show_bounds, selected
        viewer.frame(2)

    def fields():
        assert {"delta", "physics", "render"} <= painted.keys()
        return {key: value[0] for key, value in painted.items()}

    def gaps(label, unit):
        for key in ("delta", "physics", "render"):
            field, label_width = painted[key]
            assert field.value_x - field.content_x - label_width == pytest.approx(label * scale)
            assert field.unit_x - field.value_x - field.value_width == pytest.approx(unit * scale)

    def edit(field, value):
        bench.field_search = runtime.catalog["status_metrics"]["fields"][field]["label"]
        viewer.frame(3)
        viewer.enter(f"value/parts/status_metrics/{field}", value)
        assert runtime.resolved_part("status_metrics", field) == value
        assert store.latest()["spec"]["parts"]["status_metrics"][field] == value
        return fields()

    def anchors(field):
        return field.value_x + field.value_width, field.unit_x

    def same_right_neighbors(before, after, changed):
        order = ("backend", "metric", "delta", "physics", "render")
        for key in order[order.index(changed) + 1 :]:
            assert after[key] == before[key]
        assert after[changed].x + after[changed].width == pytest.approx(
            before[changed].x + before[changed].width
        )
        assert after[changed].unit_x - anchors(after[changed])[0] == pytest.approx(
            after[changed].unit_gap
        )

    initial = fields()
    assert initial["delta"].value == "2.0"
    gaps(4, 2)
    assert initial["physics"].value_width < initial["delta"].value_width
    capture("default-bar")
    edit("labelGap", 0)
    gaps(0, 2)
    edit("unitGap", 0)
    gaps(0, 0)
    before_width = fields()
    compact_delta = edit("deltaValueWidth", 18)
    assert compact_delta["delta"].value_width == pytest.approx(18 * scale)
    assert compact_delta["physics"] == before_width["physics"]
    assert compact_delta["render"] == before_width["render"]
    compact = edit("physicsValueWidth", 12)
    assert compact["physics"].value_width == pytest.approx(12 * scale)
    assert compact["delta"].value_width == compact_delta["delta"].value_width
    assert compact["render"] == compact_delta["render"]
    capture("compact")

    wider_delta = edit("deltaValueWidth", 48)
    assert wider_delta["delta"].value_width == pytest.approx(48 * scale)
    assert wider_delta["physics"] == compact["physics"]
    assert wider_delta["render"] == compact["render"]
    wider = edit("physicsValueWidth", 40)
    assert wider["physics"].value_width == pytest.approx(40 * scale)
    assert wider["delta"].value_width == wider_delta["delta"].value_width
    assert wider["render"] == wider_delta["render"]
    edit("labelGap", 4)
    edit("unitGap", 2)
    gaps(4, 2)
    capture("wider")

    edit("deltaValueWidth", 0)
    restored = edit("physicsValueWidth", 0)
    assert restored == initial

    steps = edit("stepsValueWidth", 24)
    assert steps["metric"].value_width == pytest.approx(24 * scale)
    same_right_neighbors(initial, steps, "metric")
    viewer.app._status_metric_mode = "time"
    viewer.frame(3)
    time_before = fields()
    timed = edit("timeValueWidth", 80)
    assert timed["metric"].value_width == pytest.approx(80 * scale)
    same_right_neighbors(time_before, timed, "metric")
    viewer.app._status_metric_mode = "steps"
    viewer.frame(3)
    assert fields() == steps

    rendered = edit("renderValueWidth", 30)
    assert rendered["render"].value_width == pytest.approx(30 * scale)
    same_right_neighbors(steps, rendered, "render")
    assert rendered["metric"].value_width == steps["metric"].value_width
    capture("custom-steps-and-render")
    for field in ("stepsValueWidth", "timeValueWidth", "renderValueWidth"):
        edit(field, 0)
    assert fields() == initial
    viewer.app._status_metric_mode = "time"
    viewer.frame(3)
    assert fields() == time_before


def test_status_metric_tooltip_spacing_controls_edit_the_live_popup(
    workbench, backend_name, monkeypatch
):
    from pathlib import Path

    import numpy as np

    from mojive.ui.app import status as status_module

    viewer, bench, runtime, store = workbench
    scale = viewer.window.style_scale
    original_status, original_tooltip = status_module.draw_status, status_module.status_tooltip
    original_text = imgui.text_unformatted
    layout, rows, frames = [], [], []
    collecting = False

    def status(*args, **kwargs):
        kwargs.update(dt=0.002, fps=59.0, physics_hz=0, step=12345)
        result = original_status(*args, **kwargs)
        layout[:] = [result]
        return result

    def text(caption, *args, **kwargs):
        if collecting:
            window = imgui.get_current_context().current_window
            position = imgui.get_cursor_screen_pos()
            padding = imgui.get_style().window_padding
            rows.append(
                (
                    caption,
                    (position.x, position.y),
                    (padding.x, padding.y),
                    imgui.get_text_line_height(),
                    window,
                )
            )
        return original_text(caption, *args, **kwargs)

    def tooltip(*args, **kwargs):
        nonlocal collecting
        rows.clear()
        collecting = True
        try:
            result = original_tooltip(*args, **kwargs)
        finally:
            collecting = False
        frames.append(tuple(rows))
        return result

    monkeypatch.setattr(status_module, "draw_status", status)
    monkeypatch.setattr(status_module, "status_tooltip", tooltip)
    monkeypatch.setattr(imgui, "text_unformatted", text)
    monkeypatch.setattr(viewer.app._frame_rate, "update", lambda elapsed: None)
    viewer.app._status_metric_mode = "steps"
    bench.part = "status_metrics"
    viewer.frame(3)
    target = Path("output/status-metrics/readability") / backend_name / f"{scale:g}x"
    target.mkdir(parents=True, exist_ok=True)

    def hover(key):
        rect = (
            layout[0].metric_rect
            if key == "metric"
            else next(metric.rect for metric in layout[0].metrics if metric.key == key)
        )
        x0, y0, x1, y1 = rect
        viewer.event("add_mouse_pos_event", (x0 + x1) / 2, (y0 + y1) / 2)
        viewer.frame(int(imgui.get_style().hover_delay_normal * 60) + 4)
        assert frames and frames[-1]
        return frames[-1]

    def check(rows, padding, gap, count):
        assert len(rows) == count
        assert all(
            row[2] == pytest.approx(tuple(value * scale for value in padding)) for row in rows
        )
        for previous, current in pairwise(rows):
            assert current[1][1] - previous[1][1] == pytest.approx(previous[3] + gap * scale)
        window = rows[0][4]
        assert rows[0][1][0] - window.pos.x == pytest.approx(padding[0] * scale)
        assert rows[0][1][1] - window.pos.y == pytest.approx(padding[1] * scale)
        return window.size.x, window.size.y

    def capture(name):
        show_bounds, selected = bench.show_bounds, bench.selected_field
        bench.show_bounds, bench.selected_field = False, None
        viewer.frame(3)
        pixels = Image.fromarray(np.asarray(viewer.window.read_frame())[::-1])
        pixels.save(target / f"{name}.png")
        window = frames[-1][0][4]
        ratio = viewer.window.pixel_scale
        pixels.crop(
            (
                round((window.pos.x - 2 * scale) * ratio),
                round((window.pos.y - 2 * scale) * ratio),
                round((window.pos.x + window.size.x + 2 * scale) * ratio),
                round((window.pos.y + window.size.y + 2 * scale) * ratio),
            )
        ).save(target / f"{name}-detail.png")
        bench.show_bounds, bench.selected_field = show_bounds, selected
        viewer.frame(2)

    default_size = check(hover("delta"), (10, 8), 6, 2)
    capture("tooltip-default")
    check(hover("metric"), (10, 8), 6, 3)
    for field, value in (("tooltipPaddingX", 18), ("tooltipPaddingY", 12), ("tooltipRowGap", 10)):
        bench.field_search = runtime.catalog["status_metrics"]["fields"][field]["label"]
        viewer.frame(3)
        viewer.enter(f"value/parts/status_metrics/{field}", value)
        assert runtime.resolved_part("status_metrics", field) == value
        assert store.latest()["spec"]["parts"]["status_metrics"][field] == value
    adjusted_size = check(hover("delta"), (18, 12), 10, 2)
    assert adjusted_size[0] > default_size[0]
    assert adjusted_size[1] > default_size[1]
    capture("tooltip-adjusted")
    check(hover("metric"), (18, 12), 10, 3)


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


@pytest.mark.parametrize("editor_closed", (False, True))
def test_preview_release_activates_its_context_after_a_detached_editor(
    require_opengl, backend_name, monkeypatch, tmp_path, editor_closed
):
    import ctypes

    import glfw

    from mojive.design.native.window import WorkbenchWindow, run_workbench

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    store = SpecificationStore(tmp_path / "spec.json")
    store.write(DEFAULT, save=True)
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(store.saved_path))
    contexts = []
    with build_workspace(
        resolve("joint_types"),
        renderer=backend_name,
        show_window=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), threaded_physics=False, debug_server=False
        ),
    ) as viewer:
        runtime = viewer.app.design
        editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
        bench = runtime.workbench = NativeWorkbench(editor, runtime, runtime)
        owner = ctypes.cast(viewer.window._window, ctypes.c_void_p).value
        release = viewer.backend.release

        def release_with_context():
            contexts.append(ctypes.cast(glfw.get_current_context(), ctypes.c_void_p).value)
            release()

        monkeypatch.setattr(viewer.backend, "release", release_with_context)
        with WorkbenchWindow(bench, visible=False, renderer=backend_name) as window:
            run_workbench(viewer, window, max_frames=2)
            window.frame()
            if editor_closed:
                window.close()
            viewer.release()
    assert contexts == [owner]


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
    assert bench.part == "timeline_model_group"
    viewer.event("add_key_event", imgui.Key.escape, True)
    viewer.event("add_key_event", imgui.Key.escape, False)
    viewer.frame()
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
    # Auto phrases stay stable for equal digit counts and reflow only by their measured width.
    monkeypatch.setattr(viewer.app._frame_rate, "update", lambda elapsed: None)
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
        viewer.app._frame_rate.value = 59.0
        viewer.frame(2)
        same_digits = record("status_metrics")
        for fps in (51.1, 99.9):
            viewer.app._frame_rate.value = fps
            viewer.frame(2)
            assert record("status_metrics")["hit"] == pytest.approx(same_digits["hit"])
            assert record("status_metrics")["fields"] == same_digits["fields"]
        metrics_hit = record("status_metrics")["hit"]
        for fps in (1.1, 9999.9, 60.0):
            viewer.app._frame_rate.value = fps
            viewer.frame(2)
            current = record("status_metrics")["hit"]
            assert current[0] + current[2] == pytest.approx(metrics_hit[0] + metrics_hit[2])
            assert_auto_status_phrases(runtime, status, scale)

        bench.editor.change(("parts", "status_metrics", "renderValueWidth"), 64)
        runtime.poll(force=True)
        viewer.frame(2)
        metric_bounds = tuple(tuple(field) for field in record("status_metrics")["fields"])
        metrics_hit = record("status_metrics")["hit"]
        for fps in (1.1, 9999.9, 60.0):
            viewer.app._frame_rate.value = fps
            viewer.frame(2)
            assert record("status_metrics")["hit"] == pytest.approx(metrics_hit)
            assert (
                tuple(tuple(field) for field in record("status_metrics")["fields"]) == metric_bounds
            )
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

        x, y, width, height = record("status_metrics")["fields"][1]
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
        child = next(
            item
            for item in imgui.get_current_context().windows
            if item.active and "参数逐项列表" in item.name
        )
        assert child.clip_rect.min.y <= y < y + height <= child.clip_rect.max.y
        for method, args in (
            ("add_mouse_pos_event", (x + width / 2, y + height / 2)),
            ("add_mouse_button_event", (0, True)),
            ("add_mouse_button_event", (0, False)),
            ("add_key_event", (shortcut_modifier(), True)),
            ("add_key_event", (imgui.Key.a, True)),
            ("add_key_event", (imgui.Key.a, False)),
            ("add_key_event", (shortcut_modifier(), False)),
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


def test_group_button_geometry_edits_only_the_selected_group(workbench, backend_name):
    from pathlib import Path

    viewer, bench, runtime, store = workbench
    shell, scale = viewer.app.workspace, viewer.window.style_scale
    bench.select_sample_body(shell)
    shell.inspector_tab = "Properties"
    bench.part = "settings_nav_button"
    bench.navigate(shell)
    viewer.frame(5)
    ui = shell.ui

    def controls():
        return {
            row["key"]: row for row in ui.design_primitives.observed if row.get("kind") == "button"
        }

    def geometry(row):
        hit, content = row["hit"], row["content"]
        return (
            hit[2:],
            row["fontSize"],
            row["iconSize"],
            content[0] - hit[0],
            content[1] - hit[1],
            hit[0] + hit[2] - content[0] - content[2],
            hit[1] + hit[3] - content[1] - content[3],
        )

    def unrelated(rows):
        return {
            key: geometry(row)
            for key, row in rows.items()
            if key.startswith(("entity-", "tab-", "menu-"))
        }

    before = unrelated(controls())
    assert any(key.startswith("entity-") for key in before)
    assert any(key.startswith("tab-") for key in before)
    assert any(key.startswith("menu-") for key in before)
    rect = ui.hits["settings-Camera"]
    viewer.event("add_mouse_pos_event", rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
    viewer.event("add_key_event", imgui.Key.mod_alt, True)
    viewer.event("add_mouse_button_event", 0, True)
    viewer.event("add_mouse_button_event", 0, False)
    viewer.event("add_key_event", imgui.Key.mod_alt, False)
    viewer.frame(3)
    assert bench.part == "settings_nav_button"
    assert len(bench.native_instances) == 6
    assert all("settings_nav_button" in row["parts"] for row in bench.native_instances)
    assert "parent/button" not in bench.hits

    bench.editor.change(("parts", "settings_nav_button", "widthMode"), "fixed")
    viewer.frame(3)
    for field, value in (
        ("width", 196),
        ("height", 40),
        ("paddingLeft", 12),
        ("paddingRight", 16),
        ("paddingTop", 6),
        ("paddingBottom", 6),
    ):
        bench.field_search = field
        viewer.frame(3)
        viewer.enter("value/parts/settings_nav_button/" + field, value)
    rows = controls()
    navigation = [
        row for row in rows.values() if row["key"].startswith("settings-") and row.get("label")
    ]
    assert len(navigation) == 6
    for row in navigation:
        assert row["hit"][2:] == pytest.approx((196 * scale, 40 * scale))
        assert geometry(row)[3:] == pytest.approx((12 * scale, 6 * scale, 16 * scale, 6 * scale))
    assert unrelated(rows) == before
    assert runtime.spec["parts"].get("button") == DEFAULT["parts"].get("button")
    bench.editor.save()
    assert read_spec(store.saved_path)["parts"]["settings_nav_button"]["height"] == 40
    bench.field_search = ""
    bench.selected_field = ("parts", "settings_nav_button", "paddingLeft")
    viewer.frame(3)
    directory = Path("output/button-groups") / backend_name / f"{scale:g}x"
    directory.mkdir(parents=True, exist_ok=True)
    viewer.event("add_mouse_pos_event", 0, 0)
    viewer.frame(3)
    Image.fromarray(viewer.window.read_frame()[::-1]).save(directory / "settings-navigation.png")
    bench.field_group, bench.selected_field = "尺寸与形状", None
    viewer.frame(3)
    Image.fromarray(viewer.window.read_frame()[::-1]).save(
        directory / "settings-navigation-size.png"
    )
    bench.field_group = "全部参数"
    viewer.click("undo")
    assert runtime.spec["parts"]["settings_nav_button"].get("paddingBottom", 0) == 0
    viewer.click("redo")
    assert runtime.spec["parts"]["settings_nav_button"]["paddingBottom"] == 6
    viewer.event("add_mouse_pos_event", 0, 0)

    cases = (
        ("scene_tree_row_tree_label", "entity-", 36, 9),
        ("scene_segment_button", "scene-geometry-", 40, 8),
        ("inspector_tabs_tab", "tab-", 38, 10),
        ("settings_camera_settings_page_button", "property-button/", 42, 11),
    )
    for part, prefix, height, padding in cases:
        bench.part, bench.field_search = part, ""
        bench.navigate(shell)
        assert "settings_nav_button" not in bench.selection_parts
        viewer.frame(4)
        baseline = controls()
        spec = deepcopy(runtime.spec)
        spec["parts"][part] = {"height": height, "paddingLeft": padding, "paddingRight": padding}
        runtime.update_spec(spec)
        viewer.frame(4)
        changed = controls()
        targets = [row for row in changed.values() if part in row["parts"]]
        assert targets and all(row["key"].startswith(prefix) for row in targets)
        assert all(row["hit"][3] == pytest.approx(height * scale) for row in targets)
        assert all(geometry(row)[3] == pytest.approx(padding * scale) for row in targets)
        for key, row in changed.items():
            if key in baseline and part not in row["parts"] and not key.startswith("expand-"):
                assert geometry(row) == geometry(baseline[key]), (part, key)
        Image.fromarray(viewer.window.read_frame()[::-1]).save(directory / (part + ".png"))


def test_scene_count_glyphs_survive_live_monospace_switch(workbench, backend_name, monkeypatch):
    from pathlib import Path

    from mojive.ui.imgui_draw import ImguiDraw2D

    viewer, bench, runtime, _store = workbench
    shell, ui = viewer.app.workspace, viewer.app.workspace.ui
    bench.part = "scene_type_filters_filter"
    bench.navigate(shell)
    bench.show_bounds = False
    viewer.frame(4)
    expected = {
        "scene-filter-" + kind: str(count)
        for kind, count in shell.navigation.scene_filters(shell._panel_context)
    }
    glyphs = {}
    original = ImguiDraw2D.text

    def text(draw, pos, ink, value, **options):
        start = len(draw._dl.vtx_buffer)
        original(draw, pos, ink, value, **options)
        item = imgui.get_item_id()
        for key in expected:
            if item == imgui.get_id(key):
                glyphs[key] = (value, imgui.get_font().font_id, len(draw._dl.vtx_buffer) - start)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    for family in ("mono", "system", "mono"):
        bench.editor.change(("parts", bench.part, "fontFamily"), family)
        bench.editor.change(("parts", bench.part, "fontSize"), 12)
        runtime.poll(force=True)
        assert runtime.spec["parts"][bench.part]["fontFamily"] == family
        glyphs.clear()
        viewer.frame(4)
        assert set(glyphs) == set(expected)
        for key, value in expected.items():
            assert glyphs[key][0] == value
            assert glyphs[key][2] > 0
            if family == "mono":
                assert glyphs[key][1] == ui.mono.font_id
    directory = Path("output/ui-polish") / backend_name / f"{shell.s:g}x"
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(viewer.window.read_frame()[::-1]).save(directory / "scene-counts.png")


def test_material_spacing_and_toolbar_dividers_are_group_local(workbench, backend_name):
    from pathlib import Path

    import numpy as np

    viewer, bench, runtime, store = workbench
    shell, scale = viewer.app.workspace, viewer.window.style_scale
    bench.part, bench.show_bounds = "assets_material", False
    bench.navigate(shell)
    viewer.app.selected = "free_box"
    viewer.frame(4)
    directory = Path("output/ui-polish") / backend_name / f"{scale:g}x"
    directory.mkdir(parents=True, exist_ok=True)

    def records():
        return {row["key"]: row for row in runtime.observed}

    baseline = records()
    assert "parent/assets_material_list" in bench.hits
    for index in range(4):
        card = baseline[f"material-card-{index}"]
        assert card["content"][0] - card["hit"][0] == pytest.approx(12 * scale)
        assert card["content"][1] - card["hit"][1] == pytest.approx(8 * scale)
    assert baseline["viewport-toolbox"]["separators"]
    assert baseline["viewport-transport"]["separators"]
    Image.fromarray(viewer.window.read_frame()[::-1]).save(directory / "materials-default.png")
    edits = {
        "assets_material": {
            "paddingLeft": 20,
            "paddingRight": 20,
            "paddingTop": 10,
            "paddingBottom": 10,
            "columnGap": 16,
        },
        "assets_material_list": {
            "paddingLeft": 8,
            "paddingRight": 8,
            "paddingTop": 4,
            "paddingBottom": 4,
            "rowGap": 12,
            "actionGap": 16,
        },
        "toolbox": {
            "separatorThickness": 2,
            "separatorGap": 10,
            "separatorInset": 3,
            "separatorColor": "#E62132",
        },
    }
    for part, fields in edits.items():
        for field, value in fields.items():
            bench.editor.change(("parts", part, field), value)
    runtime.poll(force=True)
    viewer.frame(4)
    changed = records()
    assert (
        changed["viewport-transport"]["separators"] == baseline["viewport-transport"]["separators"]
    )
    for index in range(4):
        card = changed[f"material-card-{index}"]
        assert card["content"][0] - card["hit"][0] == pytest.approx(20 * scale)
        assert card["content"][1] - card["hit"][1] == pytest.approx(10 * scale)
        if index:
            previous = changed[f"material-card-{index - 1}"]["hit"]
            assert card["hit"][1] - previous[1] - previous[3] == pytest.approx(12 * scale)
    bench.part = "assets_material_list"
    viewer.event("add_mouse_pos_event", 0, 0)
    viewer.frame(3)
    assert "parent/assets_material" in bench.hits
    Image.fromarray(viewer.window.read_frame()[::-1]).save(directory / "materials-spacing.png")
    tool_divider = changed["viewport-toolbox"]["separators"]
    bench.part, bench.field_group = "transport", "分隔线"
    for field, value in {
        "separatorThickness": 2,
        "separatorGap": 8,
        "separatorInset": 3,
        "separatorColor": "#2FBDFF",
    }.items():
        bench.editor.change(("parts", "transport", field), value)
    runtime.poll(force=True)
    viewer.frame(4)
    changed = records()
    assert changed["viewport-toolbox"]["separators"] == tool_divider
    pixels = np.asarray(viewer.window.read_frame())[::-1]
    display = imgui.get_io().display_size
    for key, expected in (
        ("viewport-toolbox", (230, 33, 50)),
        ("viewport-transport", (47, 189, 255)),
    ):
        x, y, width, height = changed[key]["separators"][0]
        px = round((x + width / 2) * pixels.shape[1] / display.x)
        py = round((y + height / 2) * pixels.shape[0] / display.y)
        assert pixels[py, px, :3] == pytest.approx(expected, abs=4), (key, pixels[py, px])
    Image.fromarray(pixels).save(directory / "toolbar-dividers.png")
    bench.editor.save()
    saved = read_spec(store.saved_path)
    assert saved["parts"]["toolbox"]["separatorColor"].lower() == "#e62132"
    assert saved["parts"]["assets_material_list"]["actionGap"] == 16
    for part in ("toolbox", "transport"):
        bench.editor.change(("parts", part, "separatorThickness"), 0)
    runtime.poll(force=True)
    viewer.frame(4)
    assert not records()["viewport-toolbox"]["separators"]
    assert not records()["viewport-transport"]["separators"]
