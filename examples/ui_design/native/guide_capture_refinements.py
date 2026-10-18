"""Exercise accepted UI refinements through the production guide presenters."""

import json
import math
from copy import deepcopy

from imgui_bundle import imgui as g
from PIL import Image

from mojive import commands as cmd
from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.settings_panel import CATEGORIES


def capture_refinements(v):
    """Capture imports, recoverable empty states, pose summaries and compact time."""
    app, session = v.sample.app, v.sample.app.session
    values, selected, spec = deepcopy(v.values), deepcopy(v.selected), deepcopy(v.spec)
    original_selection = (
        session.selected_node.node_id if session.selected_node is not None else None
    )
    geometry = app.backend.get_geometry_style()
    process, queued = v.window._input.process_inputs, []
    observed = v.runtime.observe_components
    report = {}

    def inputs():
        process()
        for method, args in queued:
            getattr(g.get_io(), method)(*args)
        queued.clear()
        g.get_io().delta_time = 1 / 60

    def event(method, *args):
        queued.append((method, args))
        v.frame()

    def click(key):
        x, y, width, height = v.ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        for _ in range(3):
            v.frame()

    def save(name):
        for _ in range(4):
            pixels = v.frame(readback=True)
        Image.fromarray(pixels[::-1]).save(v.output / (name + ".png"))

    def page(name):
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
        v.page = name
        event("add_mouse_pos_event", 0, 0)
        for _ in range(4):
            v.frame()

    v.window._input.process_inputs = inputs
    v.runtime.observe_components = True
    try:
        page("Assets")
        if "asset-import" not in v.ui.hits:
            click("section-" + v.tr("Model assets"))
        click("asset-import")
        save("assets-import-menu")
        assert all("asset-import/choice/" + str(index) in v.ui.hits for index in range(3))
        click("asset-import/choice/0")
        assert v.message == v.tr("Preview action: {action}", action=v.tr("Import model asset"))
        page("Assets")
        if "##texture-import-type-frame" not in v.ui.hits:
            click("section-" + v.tr("Import options"))
        save("assets-import-options")
        report["imports"] = ["mesh", "hfield", "texture"]

        page("Output")
        output = app.panels.get("Output")
        app.output.clear()
        output._reset_filters()
        save("output-no-messages")
        assert "output-empty-state" in v.ui.hits and "output-reset-filters" not in v.ui.hits
        app.output.write("Scene loaded", level="info", timestamp="15:42:17")
        output._filter_text, output._levels = "no-match", {"error"}
        output.show_search = True
        save("output-no-matches")
        assert "output-reset-filters" in v.ui.hits
        click("output-reset-filters")
        save("output-filters-restored")
        assert not output._filter_text and len(output._levels) == 3
        assert "output-empty-state" not in v.ui.hits
        report["output"] = "Empty history and filtered results recover without deleting messages"

        page("Pose")
        result = session.submit(cmd.Select(0))
        assert result.ok, result.message
        v.frame()
        joint = next(item for item in session.joints if item.type == "free")
        summary = f"pose-summary-{joint.joint_id}-position-X"
        expand = f"pose-joint-{joint.joint_id}-expand"
        if summary not in v.ui.hits:
            click(expand)
        save("pose-summary")
        assert summary in v.ui.hits
        click(expand)
        save("pose-expanded")
        assert summary not in v.ui.hits
        fields = tuple(
            v.ui.hits[f"##pose-joint-{joint.joint_id}-position{axis}-frame"] for axis in "XYZ"
        )
        assert len({round(rect[1], 4) for rect in fields}) == 1
        v.values["pose-size"] = "Narrow"
        save("pose-expanded-narrow")
        fields = tuple(
            v.ui.hits[f"##pose-joint-{joint.joint_id}-position{axis}-frame"] for axis in "XYZ"
        )
        assert fields[0][1] < fields[1][1] < fields[2][1]
        v.values["pose-size"] = "Wide"
        v.frame()
        pose_style = deepcopy(v.spec["parts"].get("pose_joint"))
        v.spec["parts"].setdefault("pose_joint", {})["columnGap"] = 16
        v.refresh()
        save("pose-gap-custom")
        if pose_style is None:
            v.spec["parts"].pop("pose_joint", None)
        else:
            v.spec["parts"]["pose_joint"] = pose_style
        v.refresh()
        report["pose"] = {
            "interaction": "Free-joint summary expands through its actual header action",
            "wide_columns": 3,
            "narrow_columns": 1,
        }

        page("Status")
        v.values["status-readout"] = "time"
        for display, value, name in (
            ("seconds", 12.125, "status-time-seconds"),
            ("clock", 3661.125, "status-time-clock"),
            ("seconds", 1e9, "status-time-long"),
        ):
            v.values["playback-time"] = value
            v.spec["parts"].setdefault("status_time", {})["displayFormat"] = display
            v.refresh()
            save(name)
        report["time"] = ["seconds", "clock", "large value"]

        page("Settings")
        v.selected["Settings navigation"] = CATEGORIES.index("Rendering")
        v.frame()
        click("ground-appearance-1")
        assert app.backend.get_geometry_style().quiet_ground
        save("settings-quiet-ground")
        click("ground-appearance-0")
        assert not app.backend.get_geometry_style().quiet_ground
        report["ground"] = "Display preference toggles through the production setting"

        page("Settings")
        v.selected["Settings navigation"] = CATEGORIES.index("MuJoCo Visuals")
        v.frame()
        save("settings-navigation-inset")
        click("##collision_color-swatch")
        click("##collision_color-format-2")
        channels = tuple(v.ui.hits["##collision_color" + axis + "-frame"] for axis in "RGB")
        assert len({round(rect[1], 4) for rect in channels}) == 1
        save("color-rgb1-row")
        popups = g.get_current_context().open_popup_stack
        popup = popups[len(popups) - 1].window
        header = v.ui.hits["##collision_color-header"]
        done = v.ui.hits["##collision_color-done"]
        container = next(
            row
            for row in v.runtime.observed
            if row["kind"] == "color_popup" and row["windowId"] == popup.id_
        )
        inner_radius = container["innerRadius"]
        insets = (
            header[0] - popup.pos.x,
            header[1] - popup.pos.y,
            popup.pos.x + popup.size.x - done[0] - done[2],
            popup.pos.y + popup.size.y - done[1] - done[3],
        )
        epsilon = native_epsilon(*header, *done, *popup.pos, *popup.size)
        assert all(
            math.isclose(actual, expected, rel_tol=0, abs_tol=epsilon)
            for actual, expected in zip(insets, container["padding"], strict=True)
        ), (insets, container["padding"])
        if not container["radiusAuthored"]:
            assert math.isclose(
                popup.window_rounding,
                inner_radius * v.s + container["padding"][0],
                rel_tol=0,
                abs_tol=epsilon,
            )
        (v.output / "color-corner-geometry.json").write_text(
            json.dumps(
                {
                    "scale": v.s,
                    "outer_radius": popup.window_rounding / v.s,
                    "footer_radius": inner_radius,
                    "insets_left_top_right_bottom": [inset / v.s for inset in insets],
                    "native_insets_left_top_right_bottom": insets,
                    "native_expected_insets": container["padding"],
                    "native_epsilon": epsilon,
                    "popup": [popup.pos.x, popup.pos.y, popup.size.x, popup.size.y],
                    "footer_button": done,
                },
                indent=2,
            )
            + "\n"
        )
        rgba_style = deepcopy(v.spec["parts"].get("rgba"))
        v.spec["parts"].setdefault("rgba", {})["columns"] = 1
        v.refresh()
        save("color-rgb1-stacked")
        channels = tuple(v.ui.hits["##collision_color" + axis + "-frame"] for axis in "RGB")
        assert channels[0][1] < channels[1][1] < channels[2][1]
        if rgba_style is None:
            v.spec["parts"].pop("rgba", None)
        else:
            v.spec["parts"]["rgba"] = rgba_style
        v.refresh()
        event("add_key_event", g.Key.escape, True)
        event("add_key_event", g.Key.escape, False)
        report["color"] = "Complete RGB row and an explicit single-column layout"

        page("Timeline")
        editor = app.panels.get("Timeline").editor
        original_snapshot = editor.selected_snapshot
        snapshots = {item.snapshot_id for item in session.scene_snapshots}
        click("timeline-snapshot")
        captured = tuple(
            item.snapshot_id
            for item in session.scene_snapshots
            if item.snapshot_id not in snapshots
        )
        assert len(captured) == 1 and editor.selected_snapshot == captured[0]
        try:
            save("timeline-state-snapshot")
        finally:
            assert session.submit(cmd.RemoveSceneSnapshot(captured[0])).ok
            editor.selected_snapshot = original_snapshot
        report["snapshot"] = "Save and select a scene state through the production toolbar"
        original_view = (
            editor.view_start,
            editor.view_end,
            editor.view_needs_fit,
            editor.manual_view_range,
            editor.follow_mode,
            editor.locked_fraction,
            editor.last_followed_playhead,
        )
        try:
            editor.set_follow_mode("off")
            editor.manual_view_range = None
            editor.view_start, editor.view_end, editor.view_needs_fit = -4.99e7, 9.63e7, False
            v.frame()
            layout = app.workspace.timeline_panel.last_layout
            event(
                "add_mouse_pos_event",
                sum(layout.time_bounds) / 2,
                layout.lo[1] + layout.ruler_height / 2,
            )
            for _ in range(25):
                event("add_mouse_wheel_event", 0, -100000)
            assert math.isclose(editor.view_end - editor.view_start, 60)
            save("timeline-boundaries")
            broad = [editor.view_start, editor.view_end]
            for _ in range(35):
                event("add_mouse_wheel_event", 0, 100000)
            assert math.isclose(editor.view_end - editor.view_start, 0.001, abs_tol=1e-9)
            save("timeline-boundaries-minimum")
            assert editor.set_view_range(0, 7200)
            save("timeline-long-range")
            assert editor.view_start == 0 and editor.view_end == 7200
            report["timeline_navigation"] = {
                "wheel_maximum": broad,
                "wheel_minimum_span": 0.001,
                "manual_long_range": [editor.view_start, editor.view_end],
            }
        finally:
            (
                editor.view_start,
                editor.view_end,
                editor.view_needs_fit,
                editor.manual_view_range,
                editor.follow_mode,
                editor.locked_fraction,
                editor.last_followed_playhead,
            ) = original_view
        from .guide_capture_spacing import capture_spacing

        capture_spacing(v, page, click, save, event)
        (v.output / "refinement-interactions.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        v.runtime.observe_components = observed
        v.window._input.process_inputs = process
        app.backend.set_geometry_style(geometry)
        result = session.submit(
            cmd.SelectNode(original_selection) if original_selection is not None else cmd.Select(0)
        )
        assert result.ok, result.message
        v.sample.restore_output()
        v.values, v.selected, v.spec = values, selected, spec
        v.refresh()
