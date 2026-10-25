"""Native guide navigation and local appearance edits through actual window input."""

import json
from contextlib import closing
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from examples.ui_design.native import design_guide
from examples.ui_design.native.design_guide import DesignGuide
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.app.ui.window import create_window
from mojive.design.store import read_spec
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("fail_footer", (False, True))
def test_timeline_pending_edit_before_viewport_status_at_fractional_scale(
    backend_name, tmp_path, monkeypatch, fail_footer
):
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1.25")
    with (
        create_window(
            WindowConfig(width=1440, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path)) as guide,
    ):
        guide.animate = False
        guide.page = "Timeline"
        guide.frame()
        app = guide.sample.app
        assert app._status_notice_bounds is None
        original = app.session.keyframes
        node = app.session.selected_node
        assert node is not None and node.type is NodeType.LINK and node.model_id >= 0
        assert app.model_edits.stage(cmd.AddModelKeyframe(node.model_id, "Pending guide state")).ok
        assert app.model_edits.active
        if fail_footer:

            def fail():
                raise RuntimeError("Pending-edit footer failed")

            with monkeypatch.context() as patch:
                patch.setattr(app, "_draw_pending_model_edits", fail)
                with pytest.raises(RuntimeError, match="Pending-edit footer failed"):
                    guide.frame()
                # Every owned child and ID scope must unwind before the original
                # exception reaches the caller, which can then end the frame.
                assert len(imgui.get_current_context().current_window_stack) == 1
                imgui.end_frame()
        for _ in range(3):
            pixels = guide.frame(readback=True)
        notice = imgui.internal.find_window_by_name("Pending model edits###pending_model_edits")
        assert notice is not None and notice.active
        assert app.session.keyframes == original
        if not fail_footer:
            target = Path("output/design-guide") / backend_name
            target.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels[::-1]).save(target / "pending-edits-1.25x.png")
        app.model_edits.clear()
        guide.frame()


@pytest.mark.parametrize("legacy_surfaces", (False, True))
def test_interactive_entrypoint_shows_a_completed_page(
    backend_name, tmp_path, monkeypatch, legacy_surfaces
):
    from mojive.design.specification import DEFAULT
    from mojive.ui import window as platform

    specification = deepcopy(DEFAULT)
    if legacy_surfaces:
        del specification["tokens"]["floating_color"]
        del specification["tokens"]["popup_color"]
    saved = tmp_path / "current-spec.json"
    saved.write_text(json.dumps(specification))
    monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(saved))
    shown = []

    def create(config, renderer):
        window = create_window(config, renderer)
        show = window.show

        def present():
            assert window.frame_index >= 3
            show()
            shown.append(platform.glfw.get_window_attrib(window._window, platform.glfw.VISIBLE))

        monkeypatch.setattr(window, "show", present)
        monkeypatch.setattr(window, "should_close", lambda: window.frame_index >= 5)
        return window

    monkeypatch.setattr(design_guide, "create_window", create)
    assert (
        design_guide.main(
            ["--backend", backend_name, "--output", str(tmp_path), "--page", "Typography"]
        )
        == 0
    )
    assert shown == [platform.glfw.TRUE]


def test_guide_navigation_and_local_dimensions_do_not_change_neighbor(
    backend_name, tmp_path, monkeypatch
):
    with (
        create_window(
            WindowConfig(width=1440, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path)) as guide,
    ):
        guide.animate = False
        queued = []
        process = window._input.process_inputs

        def inputs():
            process()
            io = imgui.get_io()
            for method, args in queued:
                getattr(io, method)(*args)
            queued.clear()
            io.delta_time = 1 / 60

        monkeypatch.setattr(window._input, "process_inputs", inputs)

        def event(method, *args):
            queued.append((method, args))
            guide.frame()

        def click(rect, fraction=0.5):
            x, y, width, height = rect
            event("add_mouse_pos_event", x + width * fraction, y + height / 2)
            event("add_mouse_button_event", 0, True)
            event("add_mouse_button_event", 0, False)
            guide.frame()

        for _ in range(4):
            guide.frame()
        nav = guide.hits["page/Overview"]
        content = guide.hits["page/Overview/content"]
        assert (content[0] - nav[0]) / guide.s == pytest.approx(12)
        assert (content[1] - nav[1]) / guide.s == pytest.approx(6)
        header_inset = guide.hits["export/content"][0] - guide.hits["export"][0]
        original = deepcopy(guide.spec)
        click(guide.hits["page/Viewer layout"])
        click(guide.hits["viewer/paddingLeft"], 0.75)
        nav, content = guide.hits["page/Overview"], guide.hits["page/Overview/content"]
        assert content[0] - nav[0] > 12 * guide.s
        assert guide.hits["export/content"][0] - guide.hits["export"][0] == pytest.approx(
            header_inset
        )
        assert guide.spec == original
        click(guide.hits["page/Group laboratory"])
        assert guide.page == "Group laboratory"
        neighbor = guide.ui.hits["sample/Scene tree/0"]
        old_height = guide.ui.hits["sample/Settings navigation/0"][3]
        original = deepcopy(guide.spec)
        click(guide.hits["edit/height"], 0.55)
        assert guide.ui.hits["sample/Settings navigation/0"][3] > old_height
        assert guide.ui.hits["sample/Scene tree/0"][2:] == pytest.approx(neighbor[2:])
        assert guide.spec["parts"].get("scene_tree_row_tree_label", {}) == original["parts"].get(
            "scene_tree_row_tree_label", {}
        )
        neighbor = guide.ui.hits["sample/Scene tree/0"]
        family_y = guide.hits["edit/fontFamily"][1]
        x, y, width, height = guide.hits["edit/height"]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_wheel_event", 0, -5)
        for _ in range(4):
            guide.frame()
        assert guide.hits["edit/fontFamily"][1] < family_y
        assert guide.ui.hits["sample/Scene tree/0"] == pytest.approx(neighbor)
        guide.set_group("fontFamily", "portable")
        guide.frame()
        assert "portable" in guide.ui._font_families
        click(guide.hits["group-selector"])
        click(guide.ui.hits["group-selector/choice/5"])
        assert guide.group == "Material cards"
        assert "material-Mist blue ceramic" in guide.ui.hits
        click(guide.hits["grayscale"])
        assert guide.gray
        click(guide.hits["export"])
        assert read_spec(tmp_path / "proposed-spec.json") == guide.spec
        assert json.loads((tmp_path / "viewer-layout.json").read_text()) == guide.viewer_layout
        assert guide.runtime.spec != guide.spec


@pytest.mark.parametrize("language", ("zh_CN", "en"))
def test_guide_shared_choices_switches_and_capsules(backend_name, tmp_path, monkeypatch, language):
    from mojive.ui.localization import Language

    with (
        create_window(
            WindowConfig(width=1440, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path, language=language)) as guide,
    ):
        guide.animate = False
        guide.runtime.observe_components = True
        queued = []
        process = window._input.process_inputs

        def inputs():
            process()
            for method, args in queued:
                getattr(imgui.get_io(), method)(*args)
            queued.clear()

        monkeypatch.setattr(window._input, "process_inputs", inputs)

        def event(method, *args):
            queued.append((method, args))
            guide.frame()

        def click(rect):
            x, y, width, height = rect
            event("add_mouse_pos_event", x + width / 2, y + height / 2)
            event("add_mouse_button_event", 0, True)
            event("add_mouse_button_event", 0, False)
            guide.frame()

        guide.page = "Viewer layout"
        for _ in range(4):
            guide.frame()
        click(guide.hits["viewer-group"])
        row = next(
            item for item in guide.runtime.observed if item["key"] == "viewer-group/choice/0"
        )
        assert row["hit"][3] / guide.s == pytest.approx(28)
        assert row["padding"][0] / guide.s == pytest.approx(12)
        label = row["labelRect"]
        assert label[1] + label[3] / 2 == pytest.approx(row["hit"][1] + row["hit"][3] / 2)
        click(guide.ui.hits["viewer-group/choice/1"])
        assert guide.viewer_group == "Header actions"
        click(guide.hits["language"])
        assert guide.localizer.language is Language("en" if language == "zh_CN" else "zh_CN")
        guide.page = "Controls"
        guide.frame()
        assert guide.values["guide-switch-0"]
        click(guide.ui.hits["guide-switch-0"])
        assert not guide.values["guide-switch-0"]
        guide.page, guide.group = "Group laboratory", "Scene filters"
        guide.set_group("height", 56)
        for _ in range(3):
            guide.frame()
        pill = next(
            item for item in guide.runtime.observed if item["key"] == "sample/Scene filters/0"
        )
        assert pill["radius"] * guide.s == pytest.approx(pill["hit"][3] / 2)


@pytest.mark.parametrize("language", ("zh_CN", "en"))
@pytest.mark.parametrize("width", (1000, 1440))
def test_guide_catalogue_popup_insets_and_production_slices(
    backend_name, tmp_path, monkeypatch, language, width
):
    with (
        create_window(
            WindowConfig(width=width, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path, language=language)) as guide,
    ):
        guide.animate = False
        guide.runtime.observe_components = True
        queued = []
        process = window._input.process_inputs

        def inputs():
            process()
            for method, args in queued:
                getattr(imgui.get_io(), method)(*args)
            queued.clear()
            imgui.get_io().delta_time = 1 / 60

        monkeypatch.setattr(window._input, "process_inputs", inputs)

        def event(method, *args):
            queued.append((method, args))
            guide.frame()

        def settle():
            for _ in range(4):
                guide.frame()

        def click(rect):
            x, y, w, h = rect
            event("add_mouse_pos_event", x + w / 2, y + h / 2)
            event("add_mouse_button_event", 0, True)
            event("add_mouse_button_event", 0, False)
            settle()

        guide.page = "Icons"
        guide.values["icon-family"] = "Viewport tools"
        settle()
        assert guide.values["icon-selected"] == "tool-move"
        assert len(guide.icon_cells) == 6
        for _name, center, left, caption_width in guide.icon_cells:
            assert left + caption_width / 2 == pytest.approx(center)
        click(guide.hits["icon/Viewport tools/tool-rotate"])
        assert guide.values["icon-selected"] == "tool-rotate"

        guide.page = "Inspector"
        settle()
        app = guide.sample.app
        assert app.window is guide.window
        assert app.preferences.path is None
        app.selected = "free_body"
        modes = []
        for size in ("Narrow", "Medium", "Wide"):
            guide.values["inspector-size"] = size
            settle()
            entity = app.entity
            key = "##" + entity["id"] + "position"
            before = [guide.ui.hits[key + axis] for axis in "XYZ"]
            label = guide.ui.hits[key + "-label"]
            modes.append((len({rect[1] for rect in before}), label[1] < before[0][1]))
            assert len({rect[2] for rect in before}) == 1
            assert all(rect[2] > 0 and rect[3] > 0 for rect in before)
            assert modes[-1][0] in (1, 3)
            assert app.session.submit(
                cmd.SetPose(entity["node_id"], np.array([1e8, 0, 1]), np.eye(3))
            ).ok
            guide.frame()
            assert [guide.ui.hits[key + axis] for axis in "XYZ"] == before
            assert app.session.submit(
                cmd.SetPose(entity["node_id"], np.array([-1.6, 0, 0.55]), np.eye(3))
            ).ok
        assert modes[0] == (3, True)
        assert modes[-1] == (1, False)

        guide.page = "Viewer layout"
        guide.viewer_group = "Header actions"
        settle()
        click(guide.hits["viewer-group"])
        x, y, w, h = guide.ui.hits["viewer-group/choice/4"]
        event("add_mouse_pos_event", x + w / 2, y + h / 2)
        settle()
        observed = {item["key"]: item for item in guide.runtime.observed}
        popup = observed["local-menu-viewer-group"]["hit"]
        rows = [observed[f"viewer-group/choice/{i}"] for i in range(5)]
        assert rows[-1]["hovered"]
        assert rows[0]["hit"][1] - popup[1] == pytest.approx(4 * guide.s)
        assert rows[-1]["hit"][1] + rows[-1]["hit"][3] <= popup[1] + popup[3] - 4 * guide.s + 0.1
        for row in rows:
            rx, _ry, rw, _rh = row["hit"]
            assert rx - popup[0] == pytest.approx(4 * guide.s)
            assert popup[0] + popup[2] - rx - rw == pytest.approx(4 * guide.s)
        event("add_key_event", imgui.Key.escape, True)
        event("add_key_event", imgui.Key.escape, False)

        guide.page = "Viewport"
        settle()
        click(guide.hits["playback-toggle"])
        assert guide.values["playback-running"]
        click(guide.hits["playback-toggle"])
        assert not guide.values["playback-running"]
        click(guide.hits["playback-next"])
        assert guide.values["playback-time"] == pytest.approx(0.002)
        click(guide.hits["playback-reset"])
        assert not guide.values["playback-running"]
        assert guide.values["playback-time"] == 0

        guide.page = "Output"
        settle()
        assert len(guide.sample_history.entries()) == 3
        click(guide.ui.hits["output-level-info"])
        assert guide.sample_output._levels == {"warning", "error"}
        click(guide.ui.hits["output-search"])
        assert guide.sample_output.show_search
        click(guide.ui.hits["output-clear"])
        assert not guide.sample_history.entries()
        click(guide.hits["restore-output"])
        assert len(guide.sample_history.entries()) == 3

        guide.page = "Status"
        settle()
        positions = [metric.rect for metric in guide.status_layout.metrics]
        for time in (0.1, 1.2, 3.1):
            guide.clock = time
            guide.frame()
            assert [metric.rect for metric in guide.status_layout.metrics] == positions
        physics = next(
            item for item in guide.runtime.observed if item["key"] == "application-status_physics"
        )
        assert physics["unitRight"] + physics["trailingReserve"] == pytest.approx(
            physics["content"][0] + physics["content"][2]
        )
        assert physics["unitX"] - physics["valueTextRect"][0] - physics["valueTextRect"][2] == (
            pytest.approx(physics["unitGap"])
        )
        assert guide.status_layout.metrics[-2].label == (
            "物理" if language == "zh_CN" else "Physics"
        )


def test_embedded_inspector_matches_native_module_overrides(backend_name, tmp_path, monkeypatch):
    from mojive.ui.workspace.design import DESIGN
    from mojive.ui.workspace.panel_host import panel
    from mojive.ui.workspace.widgets import TEXT

    with (
        create_window(
            WindowConfig(width=1440, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path)) as guide,
    ):
        guide.spec["components"]["inspector"] = {
            "body_font_size": 20,
            "mono_font_size": 18,
            "control_height": 40,
            "frame_padding_x": 14,
            "text_color": "#abcdef",
        }
        guide._dirty = True
        guide.page, guide.animate = "Inspector", False
        seen = []
        original = guide.ui.input

        def number(*args, **kwargs):
            style = guide.ui.component_style("input_number")
            result = original(*args, **kwargs)
            seen.append(
                (
                    style.family,
                    style.get("fontSize"),
                    style.edges("padding"),
                    guide.ui.ink(TEXT, style),
                )
            )
            return result

        monkeypatch.setattr(guide.ui, "input", number)
        for _ in range(4):
            guide.frame()
        embedded = tuple(seen[-9:])
        assert len(embedded) == 9
        assert all(
            row[0] == "inspector" and row[1] == 18 and row[3] == (171, 205, 239, 255)
            for row in embedded
        )
        key = "##" + guide.sample.app.entity["id"] + "positionX"
        assert guide.ui.hits[key][3] == pytest.approx(40 * guide.s)

        def native(*args):
            imgui.set_next_window_pos((100, 100))
            imgui.set_next_window_size((640 * guide.s, 700 * guide.s))
            imgui.push_style_var(
                imgui.StyleVar_.window_padding,
                (DESIGN.panel_padding_x * guide.s, DESIGN.panel_padding_y * guide.s),
            )
            try:
                guide.sample._drawn.add("Inspector")
                panel(guide.sample.app.workspace, "Inspector", guide.sample.context)
            finally:
                imgui.pop_style_var()

        monkeypatch.setattr(guide, "draw_contents", native)
        seen.clear()
        for _ in range(3):
            guide.frame()
        assert tuple(seen[-9:]) == embedded
        assert guide.ui.hits[key][3] == pytest.approx(40 * guide.s)
        assert guide.ui.design.family() is None


def test_local_panel_callbacks_complete_and_page_switch_releases_edit(backend_name, tmp_path):
    with (
        create_window(
            WindowConfig(width=1440, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path)) as guide,
    ):
        guide.page, guide.animate = "Inspector", False
        for _ in range(3):
            guide.frame()
        sample, app = guide.sample, guide.sample.app
        sample.context.request_texture_import(0, "diffuse")
        assert app._texture_dialog is None
        assert guide.message
        app.live_model_updates = True
        results = []
        node_id = app.session.selected_node.node_id
        sample.context.submit_model_edit(
            cmd.RenameModelElement(node_id, "example_body"), results.append
        )
        assert not results
        guide.frame()
        assert len(results) == 1 and results[0].ok
        assert app.session.node(node_id).name == "example_body"
        assert not app._model_load_queue and app._model_load_future is None
        assert app.session.submit(cmd.BeginEditTransaction("Inspector edit")).ok
        app.panels.get("Inspector")._edit_transaction = True
        guide.page = "Typography"
        guide.frame()
        assert not app.session.editing
        assert not app.panels.get("Inspector")._edit_transaction


@pytest.mark.parametrize("language", ("zh_CN", "en"))
@pytest.mark.parametrize("font_size", (14, 20))
def test_guide_menu_and_full_navigation_specimens_remain_visible(
    backend_name, tmp_path, language, font_size
):
    from mojive.ui.workspace.menu_catalog import TOP_MENUS
    from mojive.ui.workspace.settings_panel import CATEGORIES

    with (
        create_window(
            WindowConfig(width=1440, height=1000, docking=False, ini_path="", show_on_start=False),
            backend_name,
        ) as window,
        closing(DesignGuide(window, tmp_path, language=language)) as guide,
    ):
        guide.animate = False
        guide.spec["parts"].setdefault("menu_trigger", {}).update(fontSize=font_size, height=40)
        guide.spec["parts"].setdefault("settings_nav_button", {}).update(fontSize=18, height=44)
        guide._dirty = True
        guide.runtime.observe_components = True
        guide.page = "Menus"
        for _ in range(4):
            pixels = guide.frame(readback=True)
        observed = {row["key"]: row for row in guide.ui.design.primitives.observed}
        framebuffer = pixels[::-1]
        pixel_scale = framebuffer.shape[1] / imgui.get_io().display_size.x
        for name in TOP_MENUS:
            row = observed["menu-" + name]
            x, y, width, height = row["textRect"]
            left, top, right, bottom = row["clip"]
            assert top <= y and y + height <= bottom, (name, row)
            assert left <= x and x + width <= right
            x0, y0, x1, y1 = [round(v * pixel_scale) for v in (x, y, x + width, y + height)]
            ink = framebuffer[y0:y1, x0:x1, :3].max(axis=2) > 120
            painted_rows = np.flatnonzero(ink.any(axis=1))
            assert len(painted_rows) >= font_size * pixel_scale * 0.55, name

        guide.page, guide.group = "Group laboratory", "Settings navigation"
        guide.selected["Settings navigation"] = len(CATEGORIES) - 1
        for _ in range(5):
            guide.frame()
        observed = {row["key"]: row for row in guide.ui.design.primitives.observed}
        for name in CATEGORIES:
            row = observed["settings-" + name]
            x, y, width, height = row["hit"]
            left, top, right, bottom = row["clip"]
            assert top <= y and y + height <= bottom, (name, row)
            assert left <= x and x + width <= right
        child = next(
            window
            for window in imgui.get_current_context().windows
            if "Group comparison" in window.name and "Reference settings navigation" in window.name
        )
        assert child.scroll_max.y == 0
