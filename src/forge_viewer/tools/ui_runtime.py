"""Capture the production viewer chrome in paused, running, and Settings states."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from .. import commands as cmd
from ..assets import resolve
from ..composition import build, build_editor
from ..gizmo import RING_RADIUS, SIZE_PT, GizmoHandle, project, world_scale
from ..types import CameraView


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/ui-runtime"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "d2-tools-disabled-closeup.png").unlink(missing_ok=True)
    (args.output / "joint-slide-external-arrows.png").unlink(missing_ok=True)

    _capture_empty_workspace(args.output)

    viewer = build(
        resolve("gizmo"),
        paused=True,
        vsync=False,
        width=1920,
        height=1080,
        show_window=False,
    )
    try:
        _settle(viewer)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, args.output / "d2-default-paused.png")
        _save_window_crop(
            viewer,
            "Hints###viewport_hints",
            args.output / "camera-hint-closeup.png",
        )

        selected = next(node for node in viewer.session.nodes if node.posable)
        viewer.session.submit(cmd.Select(selected.object_id))
        _settle(viewer)
        _park_cursor(viewer)
        viewer.sync()
        _save(viewer, args.output / "paused.png")
        _save(viewer, args.output / "d3-selected-edit.png")
        _save_window_crop(
            viewer,
            "Playback###viewport_playback",
            args.output / "paused-playback-closeup.png",
        )
        _save_window_crop(
            viewer,
            "Hints###viewport_hints",
            args.output / "ready-hint-closeup.png",
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
    viewer = build_editor(vsync=False, width=1600, height=1000, show_window=False)
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


def _click(viewer, point: tuple[float, float]) -> None:
    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()


def _item_center(viewer, function_name: str, label: str) -> tuple[float, float]:
    original = getattr(imgui, function_name)
    found: list[tuple[float, float]] = []

    def spy(item_label, *args, **kwargs):
        result = original(item_label, *args, **kwargs)
        if item_label == label:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            found.append(((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5))
        return result

    setattr(imgui, function_name, spy)
    try:
        viewer.sync()
    finally:
        setattr(imgui, function_name, original)
    assert found
    return found[-1]


def _park_cursor(viewer) -> None:
    x, y, width, height = viewer.app._viewport_rect
    imgui.get_io().add_mouse_pos_event(x + width * 0.68, y + height * 0.32)


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
    viewer = build(
        resolve(asset),
        paused=True,
        vsync=False,
        width=1280,
        height=900,
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
    viewer = build(
        resolve("deformables"),
        paused=True,
        vsync=False,
        width=1600,
        height=1000,
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
    viewer = build(
        resolve("rangefinder"),
        paused=True,
        vsync=False,
        width=1440,
        height=900,
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
    viewer = build(
        resolve("joint_types"),
        paused=True,
        vsync=False,
        width=1440,
        height=900,
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

        node = next(item for item in viewer.session.nodes if item.name == "hinge_body")
        viewer.session.submit(cmd.Select(node.object_id))
        _settle(viewer, 4)
        viewer.app.gizmo._hovered = GizmoHandle.ROTATE_Z
        edit = viewer.app.gizmo.precise_input(viewer.session)
        if edit is not None:
            viewer.app._begin_precise_gizmo_input(edit)
            _settle(viewer, 3)
            _click(
                viewer,
                _item_center(viewer, "input_double", "##precise_gizmo_value"),
            )
            assert viewer.app._precise_gizmo_edit is not None
            _save(viewer, output / "joint-type-value.png")
            _save(viewer, output / "d5-precise-input.png")
            _save_window_crop(
                viewer,
                "Type value###precise_gizmo_input",
                output / "joint-type-value-closeup.png",
                padding=8.0,
            )
            viewer.app.set_language("zh_CN")
            _settle(viewer, 3)
            _save_window_crop(
                viewer,
                "输入数值###precise_gizmo_input",
                output / "joint-type-value-zh-closeup.png",
                padding=8.0,
            )
            viewer.app.set_language("en")
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
    scale = world_scale(cam, position, rect[3], SIZE_PT)

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
    arrow = viewer.app.gizmo._slide_arrow_polygon(slide, viewer.window.style_scale)
    start = np.mean(arrow, axis=0)
    io = imgui.get_io()
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

    farther = end - viewer.app.gizmo._axis_screen * (0.08 / viewer.app.gizmo._world_per_pt)
    io.add_mouse_pos_event(*farther)
    viewer.sync()
    inward = farther + viewer.app.gizmo._axis_screen * (0.02 / viewer.app.gizmo._world_per_pt)
    io.add_mouse_pos_event(*inward)
    viewer.sync()
    viewer.sync()
    assert viewer.session.frame.qpos is not None
    returned = float(viewer.session.frame.qpos[target.joint.qpos_adr])
    assert lower + 0.01 < returned < lower + 0.025
    assert viewer.app.gizmo.using

    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert not viewer.app.gizmo.using


def _capture_joint_gizmo_scene(output: Path) -> None:
    """Capture the production multi-joint chooser and ball precise-input state."""

    viewer = build(
        resolve("joint_gizmo"),
        paused=True,
        vsync=False,
        width=1920,
        height=1080,
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
            output / "joint-slide-single-arrow.png",
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
            "Hints###viewport_hints",
            output / "tool-hint-zh-closeup.png",
        )
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
        viewer.app.gizmo._hovered = GizmoHandle.ROTATE_Z
        edit = viewer.app.gizmo.precise_input(viewer.session)
        if edit is not None:
            viewer.app._begin_precise_gizmo_input(edit)
            _settle(viewer, 3)
            _save(viewer, output / "joint-ball-type-value.png")
            _save_window_crop(
                viewer,
                "Type value###precise_gizmo_input",
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
    finally:
        viewer.release()


def _capture_light_helpers(output: Path) -> None:
    viewer = build(
        resolve("many_lights"),
        paused=True,
        vsync=False,
        width=1600,
        height=1000,
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
