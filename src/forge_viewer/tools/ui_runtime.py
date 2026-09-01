"""Capture the production viewer chrome in paused, running, and Settings states."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from .. import commands as cmd
from ..adapters.base import NodeType
from ..assets import resolve
from ..composition import build, build_editor
from ..gizmo import RING_RADIUS, SIZE_PT, TRACKBALL_RADIUS, GizmoHandle, project, world_scale
from ..types import CameraView


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/ui-runtime"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "d2-tools-disabled-closeup.png").unlink(missing_ok=True)
    (args.output / "joint-slide-external-arrows.png").unlink(missing_ok=True)
    (args.output / "joint-slide-single-arrow.png").unlink(missing_ok=True)

    _capture_empty_workspace(args.output)

    width, height = _capture_size(1920, 1080)
    viewer = build(
        resolve("gizmo"),
        paused=True,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    )
    try:
        _settle(viewer)
        _capture_dock_tab_without_nav_cursor(viewer, args.output)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, args.output / "d2-default-paused.png")
        _save_window_crop(
            viewer,
            "Status###application_status",
            args.output / "camera-hint-closeup.png",
            padding=0.0,
        )

        selected = next(node for node in viewer.session.nodes if node.posable)
        viewer.session.submit(cmd.Select(selected.object_id))
        _settle(viewer)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, args.output / "paused.png")
        _save(viewer, args.output / "d3-selected-edit.png")
        _capture_position_snap_guide(viewer, selected, args.output)
        _save_window_crop(
            viewer,
            "Playback###viewport_playback",
            args.output / "paused-playback-closeup.png",
        )
        _save_window_crop(
            viewer,
            "Status###application_status",
            args.output / "ready-hint-closeup.png",
            padding=0.0,
        )
        if selected.model_id >= 0:
            viewer.app.request_model_rename(selected.node_id)
        else:
            viewer.app.request_rename(selected.object_id)
        # The first frame measures an always-auto-resize popup while hidden;
        # keep the cursor stationary and capture only after its visible frame
        # has reached the readback buffer.
        _settle(viewer, 3)
        _save(viewer, args.output / "rename-popover.png")
        _dismiss_popup(viewer)

        viewer.session.submit(cmd.Play())
        _settle(viewer, 4)
        _save(viewer, args.output / "running.png")
        position, _rotation = viewer.app._node_pose(selected)
        viewer.app.perturb.begin(
            viewer.session,
            viewer.app._camera_view(),
            selected,
            position,
            "translate",
        )
        viewer.session.perturb.target_pos = np.asarray(position, np.float32) + np.array(
            (0.22, 0.10, 0.08), np.float32
        )
        poll_perturb = viewer.app._poll_perturb
        viewer.app._poll_perturb = lambda _state: None
        _settle(viewer, 2)
        _save(viewer, args.output / "d4-running-perturb.png")
        viewer.app._poll_perturb = poll_perturb
        viewer.app.perturb.end(viewer.session)
        _save_window_crop(
            viewer,
            "Playback###viewport_playback",
            args.output / "running-playback-closeup.png",
        )

        viewer.session.submit(cmd.Pause())
        viewer.app.gizmo.set_mode("rotate")
        _capture_trackball(viewer, selected, args.output)
        viewer.app._snap_latched = True
        _settle(viewer, 8)
        _save(viewer, args.output / "rotate-snap.png")
        _save_window_crop(
            viewer,
            "Tools###viewport_tools",
            args.output / "rotate-snap-tools-closeup.png",
        )
        viewer.app.gizmo.set_space("world")
        _settle(viewer, 2)
        _save_window_crop(
            viewer,
            "Tools###viewport_tools",
            args.output / "rotate-snap-tools-world-closeup.png",
        )
        viewer.app.gizmo.set_space("body")
        _settle(viewer, 2)

        viewer.app.panels.open_panel("Settings")
        settings = viewer.app.panels.get("Settings")
        if settings is not None:
            settings._category = "Interaction"
        _settle(viewer, 3)
        _activate_panel(viewer, "Settings")
        _settle(viewer, 2)
        _save(viewer, args.output / "settings.png")
        _save(viewer, args.output / "d6-settings-interaction.png")
        _save_window_crop(
            viewer,
            "Settings",
            args.output / "settings-interaction-closeup.png",
            padding=8.0,
            max_height=620.0,
        )
        if settings is not None:
            settings._search = "mujoco"
            _settle(viewer, 2)
            _save_window_crop(
                viewer,
                "Settings",
                args.output / "settings-search-clear.png",
                padding=8.0,
                max_height=460.0,
            )
            _click(viewer, _item_center(viewer, "invisible_button", "##clear_settings_search"))
            assert settings._search == ""
            for category, filename in (
                ("General", "settings-general-closeup.png"),
                ("Rendering", "settings-rendering-closeup.png"),
                ("MuJoCo Visuals", "settings-mujoco-closeup.png"),
            ):
                settings._category = category
                _settle(viewer, 2)
                _save_window_crop(
                    viewer,
                    "Settings",
                    args.output / filename,
                    padding=8.0,
                    max_height=1040.0 if category == "MuJoCo Visuals" else 760.0,
                )
            settings._category = "Interaction"
            _settle(viewer, 2)

        hierarchy = viewer.app.panels.get("Hierarchy")
        if hierarchy is not None:
            hierarchy._filter = "revolute"
            _activate_panel(viewer, "Hierarchy")
            _settle(viewer, 2)
            _save_window_crop(
                viewer,
                "Hierarchy",
                args.output / "hierarchy-search-clear.png",
                padding=8.0,
                max_height=620.0,
            )
            _click(viewer, _item_center(viewer, "invisible_button", "##clear_filter"))
            assert hierarchy._filter == ""

        for panel_name, filename in (
            ("Hierarchy", "hierarchy-closeup.png"),
            ("Assets", "assets-closeup.png"),
            ("Inspector", "inspector-transform-closeup.png"),
            ("Camera", "camera-closeup.png"),
            ("Stats", "stats-closeup.png"),
            ("Output", "output-closeup.png"),
        ):
            viewer.app.panels.open_panel(panel_name)
            _settle(viewer, 2)
            _activate_panel(viewer, panel_name)
            _settle(viewer, 2)
            _save_window_crop(
                viewer,
                panel_name,
                args.output / filename,
                padding=8.0,
                max_height=620.0,
            )
        _open_output_context_menu(viewer)
        _save(viewer, args.output / "output-context-menu.png")
        _dismiss_popup(viewer)
        for index in range(500):
            viewer.app.output.write(
                f"/very/long/workspace/component/{index:04d}/mesh/resource.bin "
                f"rebuilt diagnostic payload {index}",
                level="warning" if index % 17 == 0 else "info",
                timestamp="08:15:00",
            )
        _settle(viewer, 3)
        _activate_panel(viewer, "Output")
        _save_window_crop(
            viewer,
            "Output",
            args.output / "output-high-count-closeup.png",
            padding=8.0,
            max_height=360.0,
        )
        for label in ("File", "Edit", "Entity", "View", "Window", "Help"):
            _open_main_menu(viewer, label)
            _save(viewer, args.output / f"{label.lower()}-menu.png")
            _dismiss_popup(viewer)
    finally:
        viewer.release()

    _capture_control(args.output, "actuator_visuals", "control-actuators-closeup.png")
    _capture_control(args.output, "mocap_equality", "control-equality-closeup.png")
    _capture_keyframes(args.output)
    _capture_sensors(args.output)
    _capture_joint_gizmos(args.output)
    _capture_joint_gizmo_scene(args.output)
    _capture_light_helpers(args.output)

    for path in sorted(args.output.glob("*.png")):
        print(path.resolve())
    return 0


def _capture_empty_workspace(output: Path) -> None:
    width, height = _capture_size(1600, 1000)
    viewer = build_editor(vsync=False, width=width, height=height, show_window=False)
    try:
        _settle(viewer, 8)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, output / "d1-empty-workspace.png")
    finally:
        viewer.release()


def _settle(viewer, frames: int = 7) -> None:
    for _ in range(frames):
        viewer.sync()


def _capture_size(width: int, height: int) -> tuple[int, int]:
    """Keep the gallery workspace usable under extreme explicit UI scales."""

    try:
        ui_scale = float(os.environ.get("FORGE_VIEWER_UI_SCALE", "1"))
    except ValueError:
        ui_scale = 1.0
    factor = max(1.0, ui_scale / 2.0)
    return round(width * factor), round(height * factor)


def _click(viewer, point: tuple[float, float]) -> None:
    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()


def _right_click(viewer, point: tuple[float, float]) -> None:
    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    io.add_mouse_button_event(1, True)
    viewer.sync()
    io.add_mouse_button_event(1, False)
    viewer.sync()


def _hover_gizmo_candidate(
    viewer,
    candidates,
    handle: GizmoHandle,
) -> tuple[int, np.ndarray]:
    """Return the first candidate owned by the real viewport input pipeline."""

    io = imgui.get_io()
    for index, candidate in enumerate(candidates):
        point = np.asarray(candidate, np.float64)
        io.add_mouse_pos_event(float(point[0]), float(point[1]))
        viewer.sync()
        if viewer.app.gizmo.hovered_handle is handle:
            return index, point
    raise AssertionError(f"no interactive {handle.name} gizmo candidate")


def _item_rect(
    viewer,
    function_name: str,
    label: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    original = getattr(imgui, function_name)
    found: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def spy(item_label, *args, **kwargs):
        result = original(item_label, *args, **kwargs)
        if item_label == label:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            found.append(((lo.x, lo.y), (hi.x, hi.y)))
        return result

    setattr(imgui, function_name, spy)
    try:
        viewer.sync()
    finally:
        setattr(imgui, function_name, original)
    assert found
    return found[-1]


def _item_center(viewer, function_name: str, label: str) -> tuple[float, float]:
    lo, hi = _item_rect(viewer, function_name, label)
    return ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5)


def _open_collapsing_header(viewer, label: str) -> None:
    original = imgui.collapsing_header
    found: list[bool] = []

    def force_open(item_label, *args, **kwargs):
        if item_label == label:
            imgui.set_next_item_open(True, imgui.Cond_.always)
            found.append(True)
        return original(item_label, *args, **kwargs)

    imgui.collapsing_header = force_open
    try:
        viewer.sync()
    finally:
        imgui.collapsing_header = original
    assert found


def _park_cursor(viewer) -> None:
    x, y, width, height = viewer.app._viewport_rect
    imgui.get_io().add_mouse_pos_event(x + width * 0.68, y + height * 0.32)


def _capture_dock_tab_without_nav_cursor(viewer, output: Path) -> None:
    """Exercise the keyboard-nav tab state while accepting no tab outline."""

    window = imgui.internal.find_window_by_name("Viewport")
    if window is None or window.dock_node is None or window.tab_id == 0:
        return
    context = imgui.get_current_context()
    context.nav_window = window
    context.nav_id = window.tab_id
    context.nav_cursor_visible = True
    viewer.sync()
    assert not context.nav_cursor_visible
    _save_window_crop(
        viewer,
        "Viewport",
        output / "viewport-tab-no-nav-outline.png",
        padding=0.0,
        max_height=72.0,
    )
    context.nav_id = 0


def _activate_panel(viewer, name: str) -> None:
    panel = imgui.internal.find_window_by_name(name)
    if panel is None or panel.dock_node is None or panel.dock_node.tab_bar is None:
        return
    node = panel.dock_node
    if node.selected_tab_id == panel.tab_id:
        return
    tab = next(tab for tab in node.tab_bar.tabs if tab.id_ == panel.tab_id)
    bar = node.tab_bar.bar_rect
    io = imgui.get_io()
    io.add_mouse_pos_event(bar.min.x + tab.offset + tab.width * 0.5, (bar.min.y + bar.max.y) * 0.5)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    _park_cursor(viewer)


def _scroll_panel(viewer, name: str, wheel_y: float) -> None:
    panel = imgui.internal.find_window_by_name(name)
    if panel is None:
        return
    io = imgui.get_io()
    io.add_mouse_pos_event(panel.pos.x + panel.size.x * 0.5, panel.pos.y + panel.size.y * 0.7)
    io.add_mouse_wheel_event(0.0, wheel_y)
    _settle(viewer, 3)


def _open_output_context_menu(viewer) -> None:
    _activate_panel(viewer, "Output")
    panel = imgui.internal.find_window_by_name("Output")
    if panel is None:
        return
    io = imgui.get_io()
    io.add_mouse_pos_event(panel.pos.x + 180.0, panel.pos.y + 104.0)
    viewer.sync()
    io.add_mouse_button_event(1, True)
    viewer.sync()
    io.add_mouse_button_event(1, False)
    viewer.sync()
    _settle(viewer, 2)


def _dismiss_popup(viewer) -> None:
    io = imgui.get_io()
    io.add_key_event(imgui.Key.escape, True)
    viewer.sync()
    io.add_key_event(imgui.Key.escape, False)
    viewer.sync()


def _open_main_menu(viewer, label: str) -> None:
    viewport = imgui.get_main_viewport()
    io = imgui.get_io()
    labels = ("File", "Edit", "Entity", "View", "Window", "Help")
    slot = labels.index(label)
    padding = float(imgui.get_style().frame_padding.x)
    x = viewport.pos.x + padding
    x += sum(float(imgui.calc_text_size(item).x) + padding * 2.0 for item in labels[:slot])
    x += imgui.calc_text_size(label).x * 0.5
    y = viewport.pos.y + imgui.get_frame_height() * 0.5
    io.add_mouse_pos_event(x, y)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    _settle(viewer, 2)


def _save(viewer, path: Path) -> None:
    pixels = viewer.window.read_frame()[::-1, :, :3]
    Image.fromarray(pixels, "RGB").save(path)


def _save_window_crop(
    viewer,
    window_name: str,
    path: Path,
    *,
    padding: float = 12.0,
    max_height: float = 0.0,
) -> None:
    panel = imgui.internal.find_window_by_name(window_name)
    if panel is None:
        return
    _save_panel_crop(viewer, panel, path, padding=padding, max_height=max_height)


def _save_active_popup_crop(
    viewer,
    path: Path,
    *,
    padding: float = 12.0,
) -> None:
    """Crop the topmost active ImGui popup, whose internal name is generated."""

    panel = next(
        (
            window
            for window in reversed(tuple(imgui.get_current_context().windows))
            if window.active and str(window.name).startswith("##Popup_")
        ),
        None,
    )
    if panel is None:
        return
    _save_panel_crop(viewer, panel, path, padding=padding)


def _save_panel_crop(
    viewer,
    panel,
    path: Path,
    *,
    padding: float,
    max_height: float = 0.0,
) -> None:
    pixels = viewer.window.read_frame()[::-1, :, :3]
    image = Image.fromarray(pixels, "RGB")
    point_width, point_height = viewer.window.size_points
    scale_x = image.width / max(point_width, 1)
    scale_y = image.height / max(point_height, 1)
    left = max(0, round((panel.pos.x - padding) * scale_x))
    top = max(0, round((panel.pos.y - padding) * scale_y))
    right = min(image.width, round((panel.pos.x + panel.size.x + padding) * scale_x))
    point_bottom = panel.pos.y + panel.size.y + padding
    if max_height > 0.0:
        point_bottom = min(point_bottom, panel.pos.y + max_height)
    bottom = min(image.height, round(point_bottom * scale_y))
    if right > left and bottom > top:
        image.crop((left, top, right, bottom)).save(path)


def _save_point_crop(
    viewer,
    path: Path,
    center: tuple[float, float],
    *,
    width: float,
    height: float,
) -> None:
    pixels = viewer.window.read_frame()[::-1, :, :3]
    image = Image.fromarray(pixels, "RGB")
    point_width, point_height = viewer.window.size_points
    scale_x = image.width / max(point_width, 1)
    scale_y = image.height / max(point_height, 1)
    left = max(0, round((center[0] - width * 0.5) * scale_x))
    top = max(0, round((center[1] - height * 0.5) * scale_y))
    right = min(image.width, round((center[0] + width * 0.5) * scale_x))
    bottom = min(image.height, round((center[1] + height * 0.5) * scale_y))
    if right > left and bottom > top:
        image.crop((left, top, right, bottom)).save(path)


def _capture_control(output: Path, asset: str, filename: str) -> None:
    width, height = _capture_size(1280, 900)
    viewer = build(
        resolve(asset),
        paused=True,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    )
    try:
        viewer.app.panels.open_panel("Control")
        _settle(viewer, 5)
        _activate_panel(viewer, "Control")
        _settle(viewer, 2)
        _save_window_crop(
            viewer,
            "Control",
            output / filename,
            padding=8.0,
            max_height=520.0,
        )
        panel = viewer.app.panels.get("Control")
        if panel is not None and viewer.session.actuators:
            panel._search = viewer.session.actuators[0].name or "act0"
            _settle(viewer, 2)
            _save_window_crop(
                viewer,
                "Control",
                output / "control-actuators-search-clear.png",
                padding=8.0,
                max_height=520.0,
            )
            _click(
                viewer,
                _item_center(viewer, "invisible_button", "##clear_actuator_search"),
            )
            assert panel._search == ""
    finally:
        viewer.release()


def _capture_keyframes(output: Path) -> None:
    width, height = _capture_size(1600, 1000)
    viewer = build(
        resolve("deformables"),
        paused=True,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    )
    try:
        viewer.app.panels.open_panel("Keyframes")
        _settle(viewer, 4)
        _activate_panel(viewer, "Keyframes")
        _settle(viewer, 3)
        _save_window_crop(
            viewer,
            "Keyframes",
            output / "keyframes-idle-closeup.png",
            padding=8.0,
            max_height=560.0,
        )
        timeline_lo, timeline_hi = _item_rect(
            viewer,
            "invisible_button",
            "##keyframe-dope-sheet",
        )
        panel_window = imgui.internal.find_window_by_name("Keyframes")
        assert panel_window is not None
        scale = viewer.window.style_scale
        timeline = (
            timeline_lo[0] + (timeline_hi[0] - timeline_lo[0]) * 0.65,
            min(
                timeline_lo[1] + (timeline_hi[1] - timeline_lo[1]) * 0.25,
                float(panel_window.pos.y + panel_window.size.y) - 4.0 * scale,
            ),
        )
        imgui.get_io().add_mouse_pos_event(*timeline)
        _settle(viewer, 2)
        _save_window_crop(
            viewer,
            "Status###application_status",
            output / "keyframes-timeline-status-closeup.png",
            padding=0.0,
        )

        viewer.session.submit(cmd.StartStateTakeRecording())
        _settle(viewer, 10)
        _save_window_crop(
            viewer,
            "Keyframes",
            output / "keyframes-recording-closeup.png",
            padding=8.0,
            max_height=560.0,
        )
        viewer.session.submit(cmd.StopStateTakeRecording())
        _settle(viewer, 3)
        if viewer.session.state_take_times:
            viewer.session.submit(cmd.PlayStateTake())
            _settle(viewer, 2)
            _save_window_crop(
                viewer,
                "Keyframes",
                output / "keyframes-replay-closeup.png",
                padding=8.0,
                max_height=560.0,
            )
            viewer.session.submit(cmd.PauseStateTake())

        keyframes = viewer.app.panels.get("Keyframes")
        model_ids = tuple(model.model_id for model in viewer.session.scene_models)
        if keyframes is not None and model_ids:
            result = viewer.session.submit(cmd.AddModelKeyframe(model_ids[0], "pose1"))
            if result.ok:
                keyframes._selected_id = result.entity_id
                keyframes._selection_generation = -1
        _settle(viewer, 3)
        _save_window_crop(
            viewer,
            "Keyframes",
            output / "keyframes-closeup.png",
            padding=8.0,
            max_height=560.0,
        )
        _scroll_panel(viewer, "Keyframes", -8.0)
        _save_window_crop(
            viewer,
            "Keyframes",
            output / "keyframes-selected-closeup.png",
            padding=8.0,
            max_height=680.0,
        )
    finally:
        viewer.release()


def _capture_sensors(output: Path) -> None:
    width, height = _capture_size(1440, 900)
    viewer = build(
        resolve("rangefinder"),
        paused=True,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    )
    try:
        viewer.app.panels.open_panel("Sensors")
        _settle(viewer, 5)
        sensors = viewer.app.panels.get("Sensors")
        if sensors is not None:
            sensors.sensor_index = max(0, len(viewer.session.sensor_infos) - 1)
        _activate_panel(viewer, "Sensors")
        _settle(viewer, 3)
        _save_window_crop(
            viewer,
            "Sensors",
            output / "sensors-multiline-closeup.png",
            padding=8.0,
            max_height=520.0,
        )
        plot = viewer.app.panels.get("Plot")
        focus = getattr(plot, "focus_sensor", None)
        if callable(focus):
            focus(sensors.sensor_index if sensors is not None else 0)
        viewer.app.panels.open_panel("Plot")
        _settle(viewer, 4)
        _activate_panel(viewer, "Plot")
        _settle(viewer, 2)
        _save_window_crop(
            viewer,
            "Plot",
            output / "plot-sensor-closeup.png",
            padding=8.0,
            max_height=520.0,
        )
    finally:
        viewer.release()


def _capture_joint_gizmos(output: Path) -> None:
    width, height = _capture_size(1440, 900)
    viewer = build(
        resolve("joint_types"),
        paused=True,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    )
    try:
        viewer.app.gizmo.set_style("2d")
        viewer.app.panels.open_panel("Joints")
        _settle(viewer, 3)
        _activate_panel(viewer, "Joints")
        joints = viewer.app.panels.get("Joints")
        if joints is not None:
            joints._search = "slide"
            _settle(viewer, 2)
            _save_window_crop(
                viewer,
                "Joints",
                output / "joints-search-clear.png",
                padding=8.0,
                max_height=520.0,
            )
            _click(
                viewer,
                _item_center(viewer, "invisible_button", "##clear_joint_search"),
            )
            assert joints._search == ""
            _right_click(
                viewer,
                _item_center(viewer, "button", "##sort_order_##joint_search"),
            )
            assert joints._sort_by_name
            _settle(viewer, 45)
            _save(viewer, output / "joints-sort-tooltip.png")
            _save_window_crop(
                viewer,
                "Joints",
                output / "joints-name-order.png",
                padding=8.0,
                max_height=520.0,
            )
            _right_click(
                viewer,
                _item_center(viewer, "button", "##sort_order_##joint_search"),
            )
            assert not joints._sort_by_name
            slide_joint = next(joint for joint in viewer.session.joints if joint.name == "slide")
            slide_joint_node = next(
                node
                for node in viewer.session.nodes
                if node.type is NodeType.JOINT and node.joint_index == slide_joint.joint_id
            )
            _click(
                viewer,
                _item_center(
                    viewer,
                    "selectable",
                    f"slide##joint-select-{slide_joint.joint_id}",
                ),
            )
            assert viewer.session.selected_node is slide_joint_node
            _settle(viewer, 3)
            assert viewer.app.gizmo.visible
            _save(viewer, output / "joints-viewport-selection.png")
            _save_window_crop(
                viewer,
                "Joints",
                output / "joints-selected.png",
                padding=8.0,
                max_height=520.0,
            )
            _activate_panel(viewer, "Inspector")
            _settle(viewer, 2)
            _scroll_panel(viewer, "Inspector", -2.0)
            _save_window_crop(
                viewer,
                "Inspector",
                output / "joint-inspector-properties.png",
                padding=8.0,
                max_height=880.0,
            )
            _scroll_panel(viewer, "Inspector", -24.0)
            _save_window_crop(
                viewer,
                "Inspector",
                output / "joint-inspector-advanced.png",
                padding=8.0,
                max_height=880.0,
            )
        for body_name, filename in (
            ("hinge_body", "joint-hinge-closeup.png"),
            ("slide_body", "joint-slide-closeup.png"),
        ):
            node = next(item for item in viewer.session.nodes if item.name == body_name)
            viewer.session.submit(cmd.Select(node.object_id))
            _settle(viewer, 6)
            _park_cursor(viewer)
            viewer.sync()
            _save_window_crop(
                viewer,
                "Viewport",
                output / filename,
                padding=0.0,
                max_height=700.0,
            )
            if body_name == "hinge_body":
                _activate_panel(viewer, "Inspector")
                _open_collapsing_header(
                    viewer,
                    viewer.app.localizer.text("body inertial and dynamics"),
                )
                _scroll_panel(viewer, "Inspector", -24.0)
                _save_window_crop(
                    viewer,
                    "Inspector",
                    output / "body-inertial-inspector.png",
                    padding=8.0,
                    max_height=880.0,
                )
                _save(viewer, output / "d7-joint-pose.png")
                _capture_hinge_held_at_limit(viewer, node, output)
                viewer.session.submit(cmd.Reset())
                viewer.session.submit(cmd.Select(node.object_id))
                _settle(viewer, 5)
            if body_name == "hinge_body" and viewer.app.gizmo.joint_limit_hits:
                hit = viewer.app.gizmo.joint_limit_hits[0]
                x0, y0, x1, y1 = hit.rect
                io = imgui.get_io()
                io.add_mouse_pos_event((x0 + x1) * 0.5, (y0 + y1) * 0.5)
                _settle(viewer, 2)
                _save_window_crop(
                    viewer,
                    "Viewport",
                    output / "joint-limit-hover.png",
                    padding=0.0,
                    max_height=700.0,
                )
                io.add_mouse_button_event(0, True)
                viewer.sync()
                _save_window_crop(
                    viewer,
                    "Viewport",
                    output / "joint-limit-pressed.png",
                    padding=0.0,
                    max_height=700.0,
                )
                io.add_mouse_button_event(0, False)
                _settle(viewer, 2)
            if body_name == "slide_body":
                _capture_slide_held_at_limit(viewer, node, output)
                viewer.session.submit(cmd.Reset())
                viewer.session.submit(cmd.Select(node.object_id))
                _settle(viewer, 5)
                _capture_slide_drag_guide(viewer, node, output)
                viewer.session.submit(cmd.Reset())
                viewer.session.submit(cmd.Select(node.object_id))
                _settle(viewer, 5)

        node = next(item for item in viewer.session.nodes if item.name == "hinge_body")
        viewer.session.submit(cmd.Select(node.object_id))
        _settle(viewer, 4)
        viewer.app.gizmo._hovered = GizmoHandle.ROTATE_Z
        edit = viewer.app.gizmo.precise_input(viewer.session)
        if edit is not None:
            viewer.app._precise_gizmo_angle_unit = "degrees"
            viewer.app._begin_precise_gizmo_input(edit)
            _settle(viewer, 3)
            _click(
                viewer,
                _item_center(viewer, "input_double", "##precise_gizmo_value"),
            )
            assert viewer.app._precise_gizmo_edit is not None
            _save(viewer, output / "joint-type-value.png")
            _save(viewer, output / "d5-precise-input.png")
            _save_active_popup_crop(
                viewer,
                output / "joint-type-value-closeup.png",
                padding=8.0,
            )
            _save_window_crop(
                viewer,
                "Status###application_status",
                output / "joint-type-value-status-closeup.png",
                padding=0.0,
            )
            viewer.app.set_language("zh_CN")
            _settle(viewer, 3)
            _save_active_popup_crop(
                viewer,
                output / "joint-type-value-zh-closeup.png",
                padding=8.0,
            )
            _save_window_crop(
                viewer,
                "Status###application_status",
                output / "joint-type-value-zh-status-closeup.png",
                padding=0.0,
            )
            viewer.app.set_language("en")
            io = imgui.get_io()
            io.add_key_event(imgui.Key.u, True)
            io.add_input_character(ord("u"))
            viewer.sync()
            io.add_key_event(imgui.Key.u, False)
            _settle(viewer, 2)
            assert viewer.app._precise_gizmo_angle_unit == "radians"
            _save(viewer, output / "joint-type-value-unit-shortcut.png")
            _save_active_popup_crop(
                viewer,
                output / "joint-type-value-unit-shortcut-closeup.png",
                padding=8.0,
            )
    finally:
        viewer.release()


def _capture_hinge_held_at_limit(viewer, node, output: Path) -> None:
    """Exercise clamp, held feedback, immediate return, and release in the real UI."""

    target, reason = viewer.app.gizmo._joint_target(viewer.session, node)
    assert target is not None, reason
    pose = viewer.app.gizmo._target_pose(viewer.session, node, target)
    assert pose is not None
    position, basis = pose
    cam = viewer.app._camera_view()
    rect = viewer.app._viewport_rect
    style_scale = viewer.window.style_scale
    scale = world_scale(cam, position, rect[3], SIZE_PT * style_scale)

    def cursor(angle: float) -> np.ndarray:
        world = (
            position
            + (basis[:, 0] * np.cos(angle) + basis[:, 1] * np.sin(angle)) * scale * RING_RADIUS
        )
        return project(cam, (world,), rect)[0, :2]

    start_angle = next(
        angle
        for angle in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        if viewer.app.gizmo.update_hover(
            viewer.session,
            cam,
            rect,
            tuple(cursor(angle)),
            style_scale=style_scale,
        )
        is GizmoHandle.ROTATE_Z
    )
    io = imgui.get_io()
    viewer.app._snap_latched = True
    io.add_mouse_pos_event(*cursor(start_angle))
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    assert viewer.app.gizmo.using

    qpos = viewer.session.frame.qpos
    assert qpos is not None
    origin = float(qpos[target.joint.qpos_adr])
    lower = float(target.joint.range[0])
    overtravel = lower - origin - 0.35
    for delta in np.linspace(0.0, overtravel, 40)[1:]:
        io.add_mouse_pos_event(*cursor(start_angle + delta))
        viewer.sync()
    viewer.sync()
    assert viewer.app.gizmo.using
    assert viewer.session.frame.qpos is not None
    assert np.isclose(viewer.session.frame.qpos[target.joint.qpos_adr], lower, atol=1e-5)
    viewer.sync()
    _save_window_crop(
        viewer,
        "Viewport",
        output / "joint-hinge-held-at-min.png",
        padding=0.0,
        max_height=700.0,
    )
    projected_center = project(cam, (position,), rect)[0, :2]
    _save_point_crop(
        viewer,
        output / "joint-hinge-held-at-min-closeup.png",
        tuple(projected_center),
        width=620.0,
        height=480.0,
    )
    viewer.app._snap_latched = False

    io.add_mouse_pos_event(*cursor(start_angle + overtravel + 0.05))
    viewer.sync()
    viewer.sync()
    assert viewer.session.frame.qpos is not None
    returned = float(viewer.session.frame.qpos[target.joint.qpos_adr])
    assert lower + 0.03 < returned < lower + 0.07
    assert viewer.app.gizmo.using

    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not viewer.app.gizmo.using


def _capture_position_snap_guide(viewer, node, output: Path) -> None:
    """Capture the final-overlay snap link above a regular position gizmo."""

    viewer.app.gizmo.set_mode("translate")
    viewer.app.gizmo.set_style("2d")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    _settle(viewer, 3)
    position, rotation = viewer.app._node_pose(node)
    cam = viewer.app._camera_view()
    rect = viewer.app._viewport_rect
    style_scale = viewer.window.style_scale
    scale = world_scale(cam, position, rect[3], SIZE_PT * style_scale)
    axis = np.asarray(rotation, np.float64).reshape(3, 3)[:, 2]
    fractions = np.linspace(0.15, 0.95, 33)
    candidates = tuple(
        project(cam, (position + axis * scale * fraction,), rect)[0, :2] for fraction in fractions
    )
    _candidate_index, start = _hover_gizmo_candidate(viewer, candidates, GizmoHandle.Z)
    projected_axis = project(cam, (position, position + axis * scale), rect)[:, :2]
    direction = projected_axis[1] - projected_axis[0]
    direction /= np.linalg.norm(direction)
    io = imgui.get_io()
    viewer.app._snap_latched = True
    io.add_mouse_button_event(0, True)
    viewer.sync()
    assert viewer.app.gizmo.using
    io.add_mouse_pos_event(*(start + direction * 52.0 * style_scale))
    _settle(viewer, 2)
    assert viewer.app.gizmo.snapping
    anchor = project(cam, (viewer.app.gizmo._drag_origin_pos,), rect)[0, :2]
    _save_point_crop(
        viewer,
        output / "position-snap-guide-closeup.png",
        tuple(anchor),
        width=560.0,
        height=420.0,
    )
    io.add_mouse_button_event(0, False)
    viewer.sync()
    viewer.app._snap_latched = False
    viewer.session.submit(cmd.Reset())
    viewer.session.submit(cmd.SelectNode(node.node_id))
    _settle(viewer, 3)


def _capture_slide_held_at_limit(viewer, node, output: Path) -> None:
    """Keep the slide guide active while its value is clamped at MIN."""

    target, reason = viewer.app.gizmo._joint_target(viewer.session, node)
    assert target is not None, reason
    pose = viewer.app.gizmo._target_pose(viewer.session, node, target)
    assert pose is not None
    position, basis = pose
    cam = viewer.app._camera_view()
    rect = viewer.app._viewport_rect
    state = viewer.app.gizmo._joint_range_state(viewer.session, target)
    assert state is not None
    slide = viewer.app.gizmo._slide_range_projection(
        cam,
        rect,
        viewer.window.style_scale,
        state,
        position,
        basis,
    )
    assert slide is not None
    start = next(
        center
        for center in reversed(
            viewer.app.gizmo._slide_arrow_targets(slide, viewer.window.style_scale)
        )
        if viewer.app.gizmo.update_hover(
            viewer.session,
            cam,
            rect,
            tuple(center),
            style_scale=viewer.window.style_scale,
        )
        is GizmoHandle.Z
    )
    io = imgui.get_io()
    viewer.app._snap_latched = True
    io.add_mouse_pos_event(*start)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    assert viewer.app.gizmo.using

    qpos = viewer.session.frame.qpos
    assert qpos is not None
    origin = float(qpos[target.joint.qpos_adr])
    lower = float(target.joint.range[0])
    overtravel = lower - origin - 0.12
    end = start + viewer.app.gizmo._axis_screen * (overtravel / viewer.app.gizmo._world_per_pt)
    for cursor in np.linspace(start, end, 32)[1:]:
        io.add_mouse_pos_event(*cursor)
        viewer.sync()
    viewer.sync()
    assert viewer.app.gizmo.using
    assert viewer.session.frame.qpos is not None
    assert np.isclose(viewer.session.frame.qpos[target.joint.qpos_adr], lower, atol=1e-5)
    projected_center = project(cam, (position,), rect)[0, :2]
    _save_point_crop(
        viewer,
        output / "joint-slide-held-at-min-closeup.png",
        tuple(projected_center),
        width=660.0,
        height=420.0,
    )
    _save_point_crop(
        viewer,
        output / "joint-slide-snap-guide-closeup.png",
        tuple(projected_center),
        width=660.0,
        height=420.0,
    )
    viewer.app._snap_latched = False

    farther = end - viewer.app.gizmo._axis_screen * (0.08 / viewer.app.gizmo._world_per_pt)
    io.add_mouse_pos_event(*farther)
    viewer.sync()
    # Keep the return motion at least three screen pixels. Hidden GLFW test
    # windows quantize synthetic cursor positions to integer pixels, which is
    # otherwise larger than a 0.02 m move for this HiDPI camera.
    return_distance = max(0.02, abs(viewer.app.gizmo._world_per_pt) * 3.0)
    inward = farther + viewer.app.gizmo._axis_screen * (
        return_distance / viewer.app.gizmo._world_per_pt
    )
    io.add_mouse_pos_event(*inward)
    viewer.sync()
    viewer.sync()
    assert viewer.session.frame.qpos is not None
    returned = float(viewer.session.frame.qpos[target.joint.qpos_adr])
    tolerance = abs(viewer.app.gizmo._world_per_pt) * 0.75
    assert return_distance - tolerance < returned - lower < return_distance + tolerance
    assert viewer.app.gizmo.using

    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not viewer.app.gizmo.using


def _capture_slide_drag_guide(viewer, node, output: Path) -> None:
    """Capture the non-snap slide segment and its two tick endpoints."""

    target, reason = viewer.app.gizmo._joint_target(viewer.session, node)
    assert target is not None, reason
    pose = viewer.app.gizmo._target_pose(viewer.session, node, target)
    assert pose is not None
    position, basis = pose
    cam = viewer.app._camera_view()
    rect = viewer.app._viewport_rect
    state = viewer.app.gizmo._joint_range_state(viewer.session, target)
    assert state is not None
    slide = viewer.app.gizmo._slide_range_projection(
        cam,
        rect,
        viewer.window.style_scale,
        state,
        position,
        basis,
    )
    assert slide is not None
    start = next(
        center
        for center in viewer.app.gizmo._slide_arrow_targets(slide, viewer.window.style_scale)
        if viewer.app.gizmo.update_hover(
            viewer.session,
            cam,
            rect,
            tuple(center),
            style_scale=viewer.window.style_scale,
        )
        is GizmoHandle.Z
    )
    io = imgui.get_io()
    viewer.app._snap_latched = False
    io.add_mouse_pos_event(*start)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    assert viewer.app.gizmo.using
    end = start + viewer.app.gizmo._axis_screen * (0.12 / viewer.app.gizmo._world_per_pt)
    io.add_mouse_pos_event(*end)
    _settle(viewer, 2)
    assert viewer.app.gizmo.using and not viewer.app.gizmo.snapping
    projected_center = project(cam, (position,), rect)[0, :2]
    _save_point_crop(
        viewer,
        output / "joint-slide-drag-guide-closeup.png",
        tuple(projected_center),
        width=660.0,
        height=420.0,
    )
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not viewer.app.gizmo.using


def _capture_trackball(viewer, node, output: Path) -> None:
    """Capture the inner background handle in hover and active states."""

    position, _rotation = viewer.app._node_pose(node)
    cam = viewer.app._camera_view()
    rect = viewer.app._viewport_rect
    center = project(cam, (position,), rect)[0, :2]
    radius = TRACKBALL_RADIUS * SIZE_PT * viewer.window.style_scale
    offsets = (
        (0.0, 0.0),
        *(
            (x, y)
            for y in np.linspace(-0.65, 0.65, 14)
            for x in np.linspace(-0.65, 0.65, 14)
            if x * x + y * y <= 0.65**2
        ),
    )
    candidates = tuple(center + np.asarray(offset) * radius for offset in offsets)
    io = imgui.get_io()
    viewer.app._snap_latched = False
    _candidate_index, start = _hover_gizmo_candidate(
        viewer,
        candidates,
        GizmoHandle.ROTATE_TRACKBALL,
    )
    _save_point_crop(
        viewer,
        output / "rotate-trackball-hover-closeup.png",
        tuple(center),
        width=520.0,
        height=420.0,
    )
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_pos_event(
        start[0] + 34.0 * viewer.window.style_scale,
        start[1] - 22.0 * viewer.window.style_scale,
    )
    _settle(viewer, 2)
    assert viewer.app.gizmo.active_handle is GizmoHandle.ROTATE_TRACKBALL
    _save_point_crop(
        viewer,
        output / "rotate-trackball-active-closeup.png",
        tuple(center),
        width=520.0,
        height=420.0,
    )
    io.add_mouse_button_event(0, False)
    viewer.sync()
    viewer.session.submit(cmd.Reset())
    viewer.session.submit(cmd.Select(node.object_id))
    _settle(viewer, 3)


def _capture_joint_gizmo_scene(output: Path) -> None:
    """Capture the production multi-joint chooser and ball precise-input state."""

    width, height = _capture_size(1920, 1080)
    viewer = build(
        resolve("joint_gizmo"),
        paused=True,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    )
    try:
        viewer.app.gizmo.set_style("2d")
        _settle(viewer, 8)

        slide = next(item for item in viewer.session.nodes if item.name == "02_prismatic")
        viewer.session.submit(cmd.Select(slide.object_id))
        _settle(viewer, 6)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, output / "joint-gizmo-page.png")
        _save_window_crop(
            viewer,
            "Viewport",
            output / "joint-slide-handles.png",
            padding=0.0,
        )
        viewer.app.set_language("zh_CN")
        _settle(viewer, 3)
        _save_window_crop(
            viewer,
            "Playback###viewport_playback",
            output / "playback-zh-closeup.png",
        )
        _save_window_crop(
            viewer,
            "Tools###viewport_tools",
            output / "tool-column-zh-closeup.png",
        )
        _save_window_crop(
            viewer,
            "Status###application_status",
            output / "tool-hint-zh-closeup.png",
            padding=0.0,
        )
        tool_window = imgui.internal.find_window_by_name("Tools###viewport_tools")
        if tool_window is not None and tool_window.active:
            imgui.get_io().add_mouse_pos_event(
                *_item_center(viewer, "invisible_button", "##viewport-tool-snap")
            )
            _settle(viewer, 2)
            _save(viewer, output / "tool-column-tooltip-zh.png")
        _park_cursor(viewer)
        viewer.app.set_language("en")
        _settle(viewer, 2)
        viewer.session.report_message("02_prismatic · +0.236 m", duration=5.0)
        _settle(viewer, 4)
        _save(viewer, output / "status-action.png")
        _save_window_crop(
            viewer,
            "Status###application_status",
            output / "status-action-closeup.png",
            padding=0.0,
        )
        viewer.app._status_metric_mode = "time"
        _click(
            viewer,
            _item_center(viewer, "invisible_button", "##status_simulation_metric"),
        )
        assert viewer.app._status_metric_mode == "steps"
        _settle(viewer, 3)
        _save_window_crop(
            viewer,
            "Status###application_status",
            output / "status-steps-closeup.png",
            padding=0.0,
        )

        viewer.app._pending_document_action = ("new_scene", None)
        io = imgui.get_io()
        io.add_key_event(imgui.Key.left_ctrl, True)
        _settle(viewer, 3)
        assert viewer.app._state.blocked
        hint = imgui.internal.find_window_by_name("Hints###viewport_hints")
        assert hint is None or not hint.active
        _save(viewer, output / "unsaved-modal-input-blocked.png")
        _save_window_crop(
            viewer,
            "Unsaved changes",
            output / "unsaved-modal-equal-actions.png",
            padding=12.0,
        )
        io.add_key_event(imgui.Key.left_ctrl, False)
        _dismiss_popup(viewer)

        ball = next(item for item in viewer.session.nodes if item.name == "03_ball")
        viewer.session.submit(cmd.Select(ball.object_id))
        _settle(viewer, 5)
        _capture_ball_rotation_axis(viewer, ball, output)
        viewer.session.submit(cmd.Reset())
        viewer.session.submit(cmd.Select(ball.object_id))
        _settle(viewer, 4)
        viewer.app.gizmo._hovered = GizmoHandle.ROTATE_Z
        edit = viewer.app.gizmo.precise_input(viewer.session)
        if edit is not None:
            viewer.app._begin_precise_gizmo_input(edit)
            _settle(viewer, 3)
            _save(viewer, output / "joint-ball-type-value.png")
            _save_active_popup_crop(
                viewer,
                output / "joint-ball-type-value-closeup.png",
                padding=8.0,
            )
            _dismiss_popup(viewer)

        multi = next(item for item in viewer.session.nodes if item.name == "05_multi_joint")
        world = np.asarray(viewer.session.frame.body_xpos[multi.body_index], np.float64)
        screen = project(viewer.app._camera_view(), (world,), viewer.app._viewport_rect)[0]
        imgui.get_io().add_mouse_pos_event(float(screen[0]), float(screen[1]))
        viewer.session.submit(cmd.Select(multi.object_id))
        _settle(viewer, 5)
        _save(viewer, output / "joint-multi-picker.png")
        _save_window_crop(
            viewer,
            "Joint gizmo###viewport_joint_gizmo",
            output / "joint-multi-picker-closeup.png",
            padding=8.0,
        )

        camera = next(item for item in viewer.session.nodes if item.name == "joint_gizmos")
        model_view = viewer.session.camera_view(camera.camera_index)
        assert model_view is not None
        camera_anchor = np.asarray(model_view.eye, np.float32)
        viewer.app.camera.adopt(
            CameraView(
                eye=camera_anchor + np.array((2.2, -2.2, 1.4), np.float32),
                target=camera_anchor,
                far=200.0,
                aspect=viewer.app._camera_view().aspect,
            )
        )
        viewer.app.camera.publish(viewer.app.camera_out)
        viewer.session.submit(cmd.Select(slide.object_id))
        _settle(viewer, 5)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, output / "camera-helper-icon.png")
        viewer.session.submit(cmd.Select(camera.object_id))
        _settle(viewer, 5)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, output / "camera-helper-selected.png")
        _save_window_crop(
            viewer,
            "Inspector",
            output / "camera-inspector-closeup.png",
            padding=8.0,
            max_height=880.0,
        )
        _scroll_panel(viewer, "Inspector", -10.0)
        _save_window_crop(
            viewer,
            "Inspector",
            output / "camera-inspector-fields.png",
            padding=8.0,
            max_height=880.0,
        )
        viewer.app.camera_preview.set_enabled(True)
        _settle(viewer, 4)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, output / "camera-helper-preview.png")
        viewer.app.camera_preview.set_enabled(False)
    finally:
        viewer.release()


def _capture_ball_rotation_axis(viewer, node, output: Path) -> None:
    """Keep a ball-joint axis rotation active for visual acceptance."""

    target, reason = viewer.app.gizmo._joint_target(viewer.session, node)
    assert target is not None, reason
    pose = viewer.app.gizmo._target_pose(viewer.session, node, target)
    assert pose is not None
    position, basis = pose
    cam = viewer.app._camera_view()
    rect = viewer.app._viewport_rect
    style_scale = viewer.window.style_scale
    scale = world_scale(cam, position, rect[3], SIZE_PT * style_scale)

    def cursor(angle: float) -> np.ndarray:
        world = (
            position
            + (basis[:, 0] * np.cos(angle) + basis[:, 1] * np.sin(angle)) * scale * RING_RADIUS
        )
        return project(cam, (world,), rect)[0, :2]

    start_angle = next(
        angle
        for angle in np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        if viewer.app.gizmo.update_hover(
            viewer.session,
            cam,
            rect,
            tuple(cursor(angle)),
            style_scale=style_scale,
        )
        is GizmoHandle.ROTATE_Z
    )
    io = imgui.get_io()
    io.add_mouse_pos_event(*cursor(start_angle))
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    assert viewer.app.gizmo.using
    io.add_mouse_pos_event(*cursor(start_angle + np.radians(38.0)))
    viewer.sync()
    viewer.sync()
    assert viewer.app.gizmo.using
    _save_window_crop(
        viewer,
        "Viewport",
        output / "joint-ball-rotation-axis.png",
        padding=0.0,
        max_height=700.0,
    )
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not viewer.app.gizmo.using


def _capture_light_helpers(output: Path) -> None:
    width, height = _capture_size(1600, 1000)
    viewer = build(
        resolve("many_lights"),
        paused=True,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    )
    try:
        node = next(item for item in viewer.session.nodes if item.name == "spot_narrow")
        viewer.session.submit(cmd.Select(node.object_id))
        _settle(viewer, 8)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, output / "d8-light-edit.png")
    finally:
        viewer.release()


if __name__ == "__main__":
    raise SystemExit(main())
