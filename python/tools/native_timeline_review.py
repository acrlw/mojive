"""Capture recorded sample identity in the actual production workspace timeline."""

import json
from copy import deepcopy
from math import isclose

from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.design.groups import group_scope
from mojive.tools.keyframe_timeline import show_timeline
from mojive.tools.native_workspace import capture
from mojive.ui.workspace.menu_geometry import native_epsilon
from mojive.ui.workspace.panel_geometry import floating_panel_geometry
from mojive.ui.workspace.timeline_body import body_geometry


def capture_timeline_regressions(viewer, output):
    """Append reproducible sample, loop-range and tooltip evidence to a review viewer."""
    output.mkdir(parents=True, exist_ok=True)
    session, workspace = viewer.session, viewer.app.workspace
    runtime, ui = viewer.app.design, workspace.ui
    observing = runtime.observe_components
    process, pointer = viewer.window._input.process_inputs, [0.0, 0.0]
    report = {"scale": viewer.window.style_scale, "captures": {}}

    def inputs():
        process()
        imgui.get_io().add_mouse_pos_event(*pointer)
        imgui.get_io().delta_time = 1 / 60

    def submit(command):
        result = session.submit(command)
        assert result.ok, result.message
        return result

    def sync(count=4):
        for _ in range(count):
            viewer.sync()

    def save(name, *, tooltip=False):
        capture(viewer, output / (name + ".png"))
        records = deepcopy(
            [
                record
                for record in runtime.observed
                if record["kind"]
                in (
                    "timeline_tick",
                    "take_sample",
                    "timeline_playhead",
                    "timeline_range",
                    "timeline_loop_range",
                    "tooltip",
                )
                or record["key"] in ("panel-Timeline", "timeline-body", "timeline-model-group")
                or record["key"].startswith("timeline-lane-label-")
            ]
        )
        body = next(record for record in records if record["key"] == "timeline-body")
        owner = next(record for record in records if record["key"] == "panel-Timeline")
        assert not owner["docked"], "The review must expose the actual floating Timeline frame"
        assert len(set(owner["nativeWindowPadding"])) == 1
        with group_scope("Timeline"), ui.design.family_scope("timeline"):
            outer_geometry = floating_panel_geometry(ui, "Timeline")
            body_style = body_geometry(ui)
        assert abs(owner["nativeWindowRadius"] - outer_geometry.radius) <= native_epsilon(
            owner["nativeWindowRadius"], outer_geometry.radius
        )
        assert owner["nativeWindowPadding"] == [outer_geometry.padding] * 2
        frame, content = body["hit"], body["content"]
        padding = body["padding"]
        epsilon = native_epsilon(*frame, *content)
        assert abs(content[0] + body["scroll"][0] - frame[0] - padding[0]) <= epsilon
        assert abs(content[1] + body["scroll"][1] - frame[1] - padding[1]) <= epsilon
        assert (
            abs(
                frame[0]
                + frame[2]
                - content[0]
                - content[2]
                - body["scrollbarGutter"][0]
                - padding[2]
            )
            <= epsilon
        )
        assert abs(body["radius"] - body_style.radius) <= native_epsilon(
            body["radius"], body_style.radius
        )
        assert padding == [*body_style.padding, *body_style.padding]
        assert (
            abs(
                body["radius"]
                - max(0, owner["nativeWindowRadius"] - owner["nativeWindowPadding"][0])
            )
            <= epsilon
        )
        assert abs(body["innerRadius"] - max(0, body["radius"] - padding[0])) <= epsilon
        selector = ui.hits["timeline-model"]
        assert selector[0] >= content[0] - epsilon and selector[1] >= content[1] - epsilon
        assert selector[0] - frame[0] >= padding[0] - epsilon
        assert selector[1] - frame[1] >= padding[1] - epsilon
        for record in records:
            if record["key"].startswith("timeline-lane-label-"):
                assert record["hit"][0] - frame[0] >= padding[0] - epsilon
                assert abs(record["radius"] - body["innerRadius"]) <= epsilon
        playhead = next(record for record in records if record["kind"] == "timeline_playhead")
        sample = next(
            record for record in records if record["kind"] == "take_sample" and record["active"]
        )
        assert playhead["time"] == sample["time"] == session.state_take_times[4]
        assert abs(playhead["playheadX"] - sample["sampleX"]) <= native_epsilon(
            playhead["playheadX"], sample["sampleX"]
        )
        tips = [record for record in records if record["kind"] == "tooltip"]
        if tooltip:
            assert any(
                "5/44" in tip["text"]
                and f"{times[4]:.6f} s" in tip["text"]
                and tip["fontSize"] == 12
                for tip in tips
            ), "The production sample tooltip must identify the actual frame and use quiet type"
        report["captures"][name] = {
            "image": name + ".png",
            "cursor": session.state_take_cursor,
            "viewRange": [editor.view_start, editor.view_end],
            "selectorRect": list(selector),
            "records": records,
        }
        (output / "timeline-sample-geometry.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )

    runtime.observe_components = True
    try:
        submit(cmd.Pause())
        submit(cmd.Reset())
        submit(cmd.SetSpeed(1))
        submit(cmd.StartStateTakeRecording())
        for _ in range(43):
            session.tick(FrameNeeds.none(), wall_dt=0.008)
        submit(cmd.StopStateTakeRecording())
        times = tuple(session.state_take_times)
        assert len(times) == 44
        assert all(
            isclose(times[index], expected, rel_tol=0, abs_tol=1e-12)
            for index, expected in ((4, 0.032), (23, 0.184), (43, 0.344))
        )
        report["sampleTimes"] = list(times)
        report["sampleCells"] = list(session.state_take_cells)
        snapshots = []
        for index in (4, 23, 43):
            submit(cmd.SeekStateTake(index))
            result = submit(cmd.CaptureSceneSnapshot(f"Sample {index + 1}"))
            snapshots.append({"id": result.entity_id, "time": times[index], "frameIndex": index})
        report["snapshots"] = snapshots
        submit(cmd.SetStateTakeLoop(23, 43))
        submit(cmd.SeekStateTake(4))
        workspace.menus.preview(())
        show_timeline(viewer)
        viewer.window._input.process_inputs = inputs
        editor = workspace.timeline_panel.editor
        editor.set_follow_mode("off")
        editor.take_selection = (23, 43)
        assert editor.set_view_range(0, 0.112)
        sync()
        save("timeline-sample-zoom")
        frame = ui.hits["timeline-take-frame-4"]
        pointer[:] = frame[0] + frame[2] / 2, workspace.timeline_panel.last_layout.lane_y(1)
        sync()
        save("timeline-sample-hover", tooltip=True)
        pointer[:] = (0.0, 0.0)
        editor.view_needs_fit = True
        sync()
        save("timeline-sample-fit")
        return report
    finally:
        viewer.window._input.process_inputs = process
        runtime.observe_components = observing
