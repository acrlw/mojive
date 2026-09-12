"""Capture responsive native layouts across widths, languages, and UI scales."""

from __future__ import annotations

import argparse
import json
import os
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
from imgui_bundle import imgui

from mojive.application.composition import build
from mojive.interaction.gizmo import GizmoHandle
from mojive.scene.assets import resolve

from .. import commands as cmd
from ..types import CameraView
from .gizmo_gallery import _save as _save_gizmo_crop
from .ui_runtime import (
    _activate_panel,
    _capture_dock_tab_without_nav_cursor,
    _click,
    _dismiss_popup,
    _item_center,
    _item_rect,
    _open_main_menu,
    _park_cursor,
    _save_active_popup_crop,
    _save_window_crop,
    _settle,
)


def _capture_button_focus(viewer, label, target, output):
    native = imgui.button
    focused = None

    def focus_button(item_label, *args, **kwargs):
        nonlocal focused
        if item_label == label:
            imgui.set_keyboard_focus_here()
            imgui.get_current_context().nav_cursor_visible = True
        result = native(item_label, *args, **kwargs)
        if item_label == label:
            focused = imgui.get_item_id()
        return result

    with patch.object(imgui, "button", focus_button):
        imgui.get_io().add_key_event(imgui.Key.tab, True)
        viewer.sync()
        imgui.get_io().add_key_event(imgui.Key.tab, False)
        _settle(viewer, 3)
        assert focused is not None and imgui.get_current_context().nav_id == focused
        assert imgui.get_current_context().nav_cursor_visible
        _save_window_crop(viewer, target, output, padding=0)


def _capture_interaction_chrome(viewer, folder: Path) -> None:
    """Exercise the same row, title, and focus surfaces used by the editor."""
    root = next(node for node in viewer.session.nodes if node.parent < 0)
    lo, hi = _item_rect(viewer, "invisible_button", f"##hierarchy-node-{root.node_id}")
    imgui.get_io().add_mouse_pos_event((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
    _settle(viewer, 3)
    _save_window_crop(viewer, "Hierarchy", folder / "hierarchy-hover.png", padding=0)
    viewer.session.submit(cmd.SelectNode(root.node_id))
    _park_cursor(viewer)
    _settle(viewer, 3)
    _save_window_crop(viewer, "Hierarchy", folder / "hierarchy-selected.png", padding=0)
    node = next(node for node in viewer.session.nodes if node.name == "05_multi_joint")
    viewer.set_gizmo_mode("translate")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    _settle(viewer, 4)
    _save_window_crop(viewer, "###viewport_joint_gizmo", folder / "joint-picker.png")
    original_name = node.name
    try:
        node.name = "left_hand_with_a_long_body_name"
        _settle(viewer, 3)
        picker = imgui.internal.find_window_by_name("###viewport_joint_gizmo")
        assert (
            picker.size.x
            >= imgui.calc_text_size(node.name).x + 2 * imgui.get_style().window_padding.x
        )
        _save_window_crop(viewer, "###viewport_joint_gizmo", folder / "joint-picker-long-title.png")
    finally:
        node.name = original_name
    _capture_dock_tab_without_nav_cursor(viewer, folder)
    viewer.app.gizmo.enabled = False
    viewer.session.submit(
        cmd.SelectNode(next(n.node_id for n in viewer.session.nodes if n.name == "04_free"))
    )
    _park_cursor(viewer)
    _settle(viewer, 3)
    playback = viewer.app._playback_widget_rect
    tools = viewer.app._tool_widget_rect
    assert playback is not None and tools is not None
    assert abs((playback[3] - playback[1]) - (tools[2] - tools[0])) < 0.01
    _save_window_crop(viewer, "Viewport", folder / "viewport-selection-tools-off.png", padding=0)
    from PIL import Image

    Image.fromarray(viewer.capture_array(surface="window")).save(folder / "editor-overview.png")
    _click(
        viewer, _item_center(viewer, "invisible_button", "##viewport-playback-recording-options")
    )
    _settle(viewer, 3)
    _save_active_popup_crop(viewer, folder / "recording-options.png")
    _dismiss_popup(viewer)
    viewer.start_recording(folder / "cancelled.mp4", countdown=60)
    _settle(viewer, 3)
    countdown = imgui.internal.find_window_by_name("##recording_countdown")
    assert countdown.pos.y >= playback[3]
    Image.fromarray(viewer.capture_array(surface="window")).save(folder / "recording-countdown.png")
    viewer.stop_recording()
    _activate_panel(viewer, "Output")
    _click(viewer, _item_center(viewer, "button", "##output-collapse"))
    _settle(viewer, 3)
    _save_window_crop(viewer, "##output-summary", folder / "output-collapsed.png", padding=0)
    _click(viewer, _item_center(viewer, "button", "##output-expand"))
    _settle(viewer, 3)
    _activate_panel(viewer, "Hierarchy")
    _click(viewer, _item_center(viewer, "button", "##hierarchy-type-joint"))
    _settle(viewer, 3)
    _save_window_crop(viewer, "Hierarchy", folder / "hierarchy-filter-joint.png", padding=0)
    _click(viewer, _item_center(viewer, "button", "##hierarchy-type-all"))
    _settle(viewer, 3)


def capture(output: Path, scale: float, language: str) -> list[dict]:
    """Capture real production widgets and assert their horizontal containment."""
    folder = output / f"{language}-{scale:g}x"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    environment = {
        "MOJIVE_UI_SCALE": str(scale),
        "MOJIVE_LANGUAGE": language,
        "MOJIVE_SETTINGS": str(folder / "settings.json"),
    }
    with (
        patch.dict(os.environ, environment),
        build(
            resolve("joint_gizmo"),
            paused=True,
            vsync=False,
            width=round(1600 * max(1.0, scale / 1.5)),
            height=round(1100 * max(1.0, scale / 1.5)),
            show_window=False,
        ) as viewer,
    ):
        _settle(viewer, 8)
        _capture_interaction_chrome(viewer, folder)
        from PIL import Image

        from mojive.session.model_edits import model_edit_scope

        pending_node = next(
            n for n in viewer.session.nodes if n.source_editable and n.geom_index >= 0
        )
        with model_edit_scope(viewer.session, viewer.app._intercept_model_edit):
            assert viewer.session.submit(
                cmd.RenameModelElement(pending_node.node_id, "pending_entity")
            ).ok
        _settle(viewer, 4)
        Image.fromarray(viewer.capture_array(surface="window")).save(
            folder / "pending-model-edits.png"
        )
        _click(
            viewer,
            _item_center(
                viewer, "button", viewer.app.localizer.text("Discard") + "##discard_model_edits"
            ),
        )
        _settle(viewer, 2)
        assert not viewer.app.model_edits.active
        manager = viewer.panels
        begin = manager._begin_panel_window
        target, width = "", 480.0

        def place(panel, *args, **kwargs):
            if panel.name == target:
                imgui.set_next_window_pos((80.0, 75.0))
                imgui.set_next_window_size((width * scale, 900.0))
            return begin(panel, *args, **kwargs)

        manager._begin_panel_window = place
        link = next(node for node in viewer.session.nodes if node.name == "03_ball_anchor")
        viewer.session.submit(cmd.SelectNode(link.node_id))
        translate = viewer.app.localizer.text
        menus = [
            _item_rect(viewer, "begin_menu", translate(name))
            for name in ("File", "Edit", "Entity", "View", "Window", "Help")
        ]
        for previous, current in pairwise(menus):
            assert current[0][0] - previous[1][0] <= 18.0 * scale
        assert menus[0][0][0] <= 10.0 * scale
        _save_window_crop(viewer, "##MainMenuBar", folder / "menu-bar.png", padding=0)
        _save_window_crop(viewer, "Status###application_status", folder / "status.png", padding=0)
        _open_main_menu(viewer, translate("Window"))
        _save_active_popup_crop(viewer, folder / "window-menu.png")
        _dismiss_popup(viewer)
        assert viewer.session.selected_node is link
        hinge = next(node for node in viewer.session.nodes if node.name == "01_revolute")
        viewer.session.submit(cmd.SelectNode(hinge.node_id))
        viewer.app.gizmo.set_mode("rotate")
        _settle(viewer, 3)
        camera = viewer.app._camera_view()
        for azimuth in (-135, 45):
            viewer.app.camera.look_from(azimuth, 25, viewer.app.camera_out, animate=False)
            _settle(viewer, 3)
            assert viewer.app.gizmo._hinge_axis and not viewer.app.gizmo.using
            _save_gizmo_crop(viewer, hinge, folder / f"hinge-axis-idle-{azimuth}.png")
        origin = viewer.app.gizmo._frame.position.copy()
        basis = viewer.app.gizmo._frame.rotation.copy()
        for angle in (0, 10, 45, 80, 90, 135):
            radians = np.radians(angle)
            offset = basis[:, 0] * np.sin(radians) + basis[:, 2] * np.cos(radians)
            viewer.set_camera(CameraView(eye=origin + offset * 2, target=origin, up=basis[:, 1]))
            _settle(viewer, 3)
            _save_gizmo_crop(viewer, hinge, folder / f"hinge-axis-angle-{angle}.png")
        viewer.set_camera(camera)
        _settle(viewer, 3)
        viewer.app.gizmo._hovered = GizmoHandle.ROTATE_Z
        edit = viewer.app.gizmo.precise_input(viewer.session)
        assert edit is not None
        viewer.app._precise_gizmo_angle_unit = "radians"
        viewer.app._begin_precise_gizmo_input(edit)
        _settle(viewer, 3)
        _save_active_popup_crop(viewer, folder / "precise-input.png")
        for index in range(5):
            imgui.get_io().add_key_event(imgui.Key.tab, True)
            viewer.sync()
            imgui.get_io().add_key_event(imgui.Key.tab, False)
            _settle(viewer, 3)
            _save_active_popup_crop(viewer, folder / f"precise-input-tab-{index}.png")
        _dismiss_popup(viewer)
        viewer.session.submit(cmd.SelectNode(link.node_id))

        for level in ("info", "warning", "error"):
            viewer.app.output.write(f"Layout audit: {level} message", level=level)
        model_id = viewer.session.scene_models[0].model_id
        for index in range(3):
            assert viewer.session.submit(cmd.Step(20))
            viewer.sync()
            assert viewer.session.submit(cmd.AddModelKeyframe(model_id, f"Pose {index + 1}"))
            _settle(viewer, 2)
        for index in range(2):
            assert viewer.session.submit(cmd.Step(20))
            viewer.sync()
            assert viewer.session.submit(cmd.CaptureSceneSnapshot(f"Snapshot {index + 1}"))
        link = next(node for node in viewer.session.nodes if node.name == "03_ball_anchor")
        viewer.session.submit(cmd.SelectNode(link.node_id))
        viewer.track_node(link.node_id)
        cases = [("Camera", w, "") for w in (140, 180, 240, 360)]
        cases += [
            ("Settings", w, category)
            for category in ("General", "Interaction", "Rendering")
            for w in (360, 720, 960)
        ]
        cases += [("Inspector", w, "") for w in (180, 230, 320, 480)]
        cases += [("Keyframes", w, "") for w in (230, 480, 820, 1000)]
        cases += [("Output", w, "") for w in (180, 320)]
        cases += [("Hierarchy", w, "") for w in (180, 320)]
        cases += [("Joints", w, "") for w in (180, 320)]
        for target, width, category in cases:
            panel = manager.get(target)
            manager.open_panel(target)
            if category:
                panel._category = category
            _settle(viewer, 2)
            window = imgui.internal.find_window_by_name(target)
            if window.dock_node is not None:
                imgui.internal.dock_context_process_undock_window(
                    imgui.get_current_context(), window, True
                )
            imgui.internal.focus_window(window)
            imgui.internal.set_scroll_y(window, 0.0)
            imgui.get_io().add_mouse_pos_event(-1000, -1000)
            _settle(viewer, 4)
            filename = f"{target.lower()}-{width}-{category.lower() or 'layout'}.png"
            _save_window_crop(viewer, target, folder / filename, padding=3.0)
            if target == "Joints" and width == 320:
                for label, suffix in (
                    (translate("Copy qpos"), "copy"),
                    ("rad####joint-qpos-0-unit", "unit"),
                ):
                    _capture_button_focus(
                        viewer, label, target, folder / f"joints-focus-{suffix}.png"
                    )
            if target == "Keyframes":
                # The take range and three follow choices require about 900 pt in
                # English even with icon-only commands; 820 pt exercises two rows.
                if width >= 1000:
                    record = _item_rect(viewer, "invisible_button", "##take-record")
                    options = _item_rect(viewer, "invisible_button", "##timeline-options")
                    assert abs(record[0][1] - options[0][1]) < 1, (width, record, options)
                panel._selected_id = viewer.session.keyframes[0].keyframe_id
                panel._selection_generation = -1
                _settle(viewer, 3)
                _save_window_crop(
                    viewer, target, folder / f"keyframes-{width}-snapshot.png", padding=3.0
                )
                if width == 1000:
                    _click(viewer, _item_center(viewer, "invisible_button", "##timeline-options"))
                    _settle(viewer, 3)
                    _save_active_popup_crop(viewer, folder / "keyframes-recording-settings.png")
                    _dismiss_popup(viewer)
            if target == "Inspector" and width == 320:
                point = _item_center(viewer, "invisible_button", "##entity_name_label")
                _click(viewer, point)
                _click(viewer, point)
                _settle(viewer, 2)
                assert panel._renaming
                _save_window_crop(viewer, target, folder / "inspector-name-edit.png", padding=0)
                imgui.get_io().add_key_event(imgui.Key.escape, True)
                viewer.sync()
                imgui.get_io().add_key_event(imgui.Key.escape, False)
                _settle(viewer, 2)
            if target == "Camera":
                labels = (
                    "##camera-projection-0",
                    "##camera-projection-1",
                    "X-Y##tracking-axes-0",
                    "X-Y-Z##tracking-axes-1",
                )
            elif category == "Rendering":
                labels = tuple(
                    f"{translate(label)}##shadow-quality-{i}"
                    for i, label in enumerate(("Performance", "Balanced", "High"))
                )
            elif target == "Inspector":
                labels = tuple(
                    f"{axis}##{translate('rotation')}_{i}_{link.node_id}"
                    for i, axis in enumerate("XYZ")
                )
            else:
                labels = ()
            rectangles = [_item_rect(viewer, "button", label) for label in labels]
            if target == "Output":
                rectangles = [
                    _item_rect(viewer, "input_text_with_hint", "##output-filter"),
                    *(
                        _item_rect(viewer, "button", f"##output-level-{level}")
                        for level in ("info", "warning", "error")
                    ),
                ]
            for lo, hi in rectangles:
                assert lo[0] >= window.inner_clip_rect.min.x, (filename, lo)
                assert hi[0] <= window.inner_clip_rect.max.x, (filename, hi)
            if category == "Rendering":
                for label, (lo, hi) in zip(labels, rectangles, strict=True):
                    text_width = imgui.calc_text_size(label.partition("##")[0]).x
                    assert hi[0] - lo[0] >= text_width + 2 * imgui.get_style().frame_padding.x
            if target == "Inspector":
                section, _ = _item_rect(viewer, "collapsing_header", translate("velocity"))
                assert max(hi[1] for _, hi in rectangles) <= section[1]
            results.append({"image": str(folder / filename), "rectangles": rectangles})
            if target == "Camera" and width == 360:
                names = (
                    "yaw",
                    "pitch",
                    "distance",
                    "fov_y_deg",
                    "far",
                    "projection",
                    "Target",
                    "Axes",
                    "Smoothing",
                )
                labels = {
                    name: _item_rect(viewer, "text_disabled", translate(name)) for name in names
                }
                assert (
                    max(rect[0][0] for rect in labels.values())
                    - min(rect[0][0] for rect in labels.values())
                    < 1
                )
                gaps = [
                    labels[b][0][1] - labels[a][0][1]
                    for a, b in (
                        ("yaw", "pitch"),
                        ("pitch", "distance"),
                        ("fov_y_deg", "far"),
                        ("far", "projection"),
                        ("Target", "Axes"),
                        ("Axes", "Smoothing"),
                    )
                ]
                assert max(gaps) - min(gaps) < 1, gaps
            if target == "Camera":
                imgui.internal.set_scroll_y(window, window.scroll_max.y)
                _settle(viewer, 3)
                _save_window_crop(
                    viewer, target, folder / f"camera-{width}-tracking.png", padding=0
                )
            panel.open = False

        viewer.track_node(None)
        target = "Inspector"
        manager.open_panel(target)
        _settle(viewer, 3)
        window = imgui.internal.find_window_by_name(target)
        imgui.internal.focus_window(window)
        measurements = []
        for width in range(280, 461, 4):
            _settle(viewer, 2)
            rects = [
                _item_rect(viewer, "drag_float", f"##{translate('position')}_{axis}_{link.node_id}")
                for axis in range(3)
            ]
            inline_axes = abs(rects[0][0][1] - rects[2][0][1]) < 1
            field_width = rects[0][1][0] - rects[0][0][0]
            for lo, hi in rects:
                assert lo[0] >= window.inner_clip_rect.min.x, (width, lo)
                assert hi[0] <= window.inner_clip_rect.max.x, (width, hi)
            if inline_axes:
                assert abs((rects[2][1][0] - rects[2][0][0]) - field_width) <= 1.01

            measurements.append(
                {"panel_width": width, "field_width": field_width, "inline_axes": inline_axes}
            )
        results.append({"transform_width_sweep": measurements})
        manager.get(target).open = False
        camera_asset = folder / "inspector-camera.xml"
        camera_asset.write_text(
            '<mujoco><worldbody><geom type="sphere" size=".1"/>'
            '<camera name="inspection_camera" pos="0 -3 1"/></worldbody></mujoco>'
        )
        assert viewer.session.submit(cmd.LoadAsset(camera_asset))
        camera_node = next(n for n in viewer.session.nodes if n.name == "inspection_camera")
        viewer.session.submit(cmd.SelectNode(camera_node.node_id))
        for width in (180, 320, 480):
            manager.open_panel(target)
            _settle(viewer, 4)
            imgui.internal.focus_window(window)
            imgui.internal.set_scroll_y(window, 0)
            _settle(viewer, 2)
            filename = folder / f"inspector-camera-{width}.png"
            _save_window_crop(viewer, target, filename, padding=0)
            if width == 320:
                label_rect = _item_rect(viewer, "text_disabled", translate("projection"))
                button_rect = _item_rect(
                    viewer, "button", f"##camera-inspector-projection-{camera_node.node_id}-0"
                )
                assert label_rect[1][0] < button_rect[0][0]
                imgui.internal.set_scroll_y(
                    window, max(0, label_rect[0][1] - window.pos.y - 90 * scale)
                )
                _settle(viewer, 3)
                _save_window_crop(
                    viewer, target, folder / "inspector-camera-projection.png", padding=0
                )
            results.append({"image": str(filename)})
            manager.get(target).open = False

        assert viewer.session.submit(cmd.LoadAsset(resolve("actuator_visuals")))
        for width in (180, 320, 480):
            target = "Control"
            manager.open_panel(target)
            _settle(viewer, 3)
            window = imgui.internal.find_window_by_name(target)
            if window.dock_node is not None:
                imgui.internal.dock_context_process_undock_window(
                    imgui.get_current_context(), window, True
                )
            imgui.internal.focus_window(window)
            _settle(viewer, 4)
            filename = folder / f"control-{width}-layout.png"
            _save_window_crop(viewer, target, filename, padding=0)
            panel = manager.get(target)
            rectangles = [
                _item_rect(viewer, "invisible_button", f"##actuator-select-{a.ctrl_address}")
                for a, component in panel._row_cache[:2]
                if component == 0
            ]
            for lo, hi in rectangles:
                assert lo[0] >= window.inner_clip_rect.min.x
                assert hi[0] <= window.inner_clip_rect.max.x
            results.append({"image": str(filename), "rectangles": rectangles})
            panel.open = False
        searches = (
            ("Hierarchy", "filter"),
            ("Joints", "joint_search"),
            ("Control", "actuator_search"),
            ("Assets", "asset-filter"),
            ("Settings", "settings_search"),
            ("Output", "output-filter"),
            ("Camera", "tracking-filter"),
        )
        for target, field in searches:
            width = 360
            manager.open_panel(target)
            _settle(viewer, 3)
            window = imgui.internal.find_window_by_name(target)
            if window.dock_node is not None:
                imgui.internal.dock_context_process_undock_window(
                    imgui.get_current_context(), window, True
                )
            imgui.internal.focus_window(window)
            _settle(viewer, 4)
            if target == "Camera":
                imgui.internal.set_scroll_y(window, window.scroll_max.y)
                _settle(viewer, 3)
                _click(viewer, _item_center(viewer, "begin_combo", "##tracking-target"))
            _click(viewer, _item_center(viewer, "input_text_with_hint", f"##{field}"))
            imgui.get_io().add_input_characters_utf8("123")
            _settle(viewer, 3)
            filename = folder / f"search-{target.lower()}.png"
            if target == "Camera":
                _save_active_popup_crop(viewer, filename)
            else:
                _save_window_crop(viewer, target, filename, padding=0)
            lo, hi = _item_rect(viewer, "input_text_with_hint", f"##{field}")
            clear_lo, clear_hi = _item_rect(viewer, "invisible_button", f"##clear_{field}")
            assert clear_lo[0] >= hi[0]
            _click(viewer, ((clear_lo[0] + clear_hi[0]) / 2, (clear_lo[1] + clear_hi[1]) / 2))
            _dismiss_popup(viewer)
            results.append({"image": str(filename), "rectangles": [(lo, hi), (clear_lo, clear_hi)]})
            manager.get(target).open = False
        manager._begin_panel_window = begin
    return results


def main(argv: list[str] | None = None) -> int:
    """Run responsive layout acceptance and write an inspectable capture index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/ui-layout-audit"))
    parser.add_argument("--scales", default="1,1.5")
    parser.add_argument("--languages", default="en,zh_CN")
    args = parser.parse_args(argv)
    results = []
    for scale in map(float, args.scales.split(",")):
        for language in args.languages.split(","):
            results.extend(capture(args.output, scale, language))
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
