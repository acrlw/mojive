"""Capture camera tracking controls and a 60 FPS comparison of a moving target."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from .. import CameraTrackingConfig, ViewerConfig, ViewportLayers, build_scene
from .. import commands as cmd
from ..capture import CaptureSurface
from ..recording import VideoRecorder
from ..scene import Scene
from ..types import CameraView, Light, LightSet, LightType
from .ui_runtime import _item_center, _item_rect, _save_window_crop


def click(viewer, point) -> None:
    """Activate a real widget through native ImGui pointer events."""
    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    viewer.sync()


def exercise_controls(viewer, node_id: int, output: Path | None = None) -> None:
    """Select a target and change axes using production camera-panel widgets."""
    viewer.panels.open_panel("Camera")
    viewer.session.submit(cmd.SelectNode(node_id))
    for _ in range(5):
        viewer.sync()
    window = imgui.internal.find_window_by_name("Camera")
    imgui.internal.focus_window(window)
    # Tracking is below the view/lens groups; exercise it at the same scroll
    # position a user reaches in a short dock or a larger UI scale.
    imgui.internal.set_scroll_y(window, window.scroll_max.y)
    viewer.sync()
    viewer.sync()
    click(viewer, _item_center(viewer, "begin_combo", "##tracking-target"))
    node = viewer.session.node(node_id)
    label = viewer.app.localizer.text("Use selection") + f": {node.name}##track-selected"
    click(viewer, _item_center(viewer, "selectable", label))
    assert viewer.tracking_node_id == node_id
    assert viewer.app.camera_tracker.config.axes == "xy"
    if output is not None:
        _save_window_crop(viewer, "Camera", output / "camera-tracking-xy.png", padding=0)
    click(viewer, _item_center(viewer, "button", "X-Y-Z##tracking-axes-1"))
    assert viewer.app.camera_tracker.config.axes == "xyz"
    if output is not None:
        _save_window_crop(viewer, "Camera", output / "camera-tracking-xyz.png", padding=0)
    click(viewer, _item_center(viewer, "button", "X-Y##tracking-axes-0"))
    assert viewer.app.camera_tracker.config.axes == "xy"
    lo, hi = _item_rect(viewer, "slider_float", "##tracking-smoothing")
    click(viewer, (lo[0] + (hi[0] - lo[0]) * 0.75, (lo[1] + hi[1]) * 0.5))
    assert viewer.app.camera_tracker.config.smoothing > 1
    io = imgui.get_io()
    io.add_mouse_button_event(1, True)
    viewer.sync()
    io.add_mouse_button_event(1, False)
    viewer.sync()
    assert viewer.app.camera_tracker.config.smoothing == 0.25
    viewer.configure_tracking(replace(viewer.app.camera_tracker.config, smoothing=0))
    viewer.sync()
    pose = viewer.session.camera
    click(viewer, _item_center(viewer, "invisible_button", "##clear_tracking-target"))
    assert np.allclose(viewer.session.camera.eye, pose.eye)
    assert np.allclose(viewer.session.camera.target, pose.target)
    assert not imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id)
    assert viewer.tracking_node_id is None
    click(viewer, _item_center(viewer, "begin_combo", "##tracking-target"))
    node = viewer.session.node(node_id)
    click(viewer, _item_center(viewer, "selectable", f"{node.name}##tracking-{node_id}"))
    assert viewer.tracking_node_id == node_id


def _scene():
    sun = Light(
        type=LightType.DIRECTIONAL,
        direction=np.array((-0.4, 0.5, -1), np.float32),
        diffuse=np.full(3, 0.9, np.float32),
    )
    scene = Scene(lights=LightSet(lights=(sun,), ambient=np.full(3, 0.3, np.float32)))
    scene.plane(size=(20, 20, 1), color=(0.16, 0.19, 0.24, 1))
    parts = [
        (scene.box(name="root", size=(0.23, 0.13, 0.3), color=(0.14, 0.63, 0.83, 1)), (0, 0, 0)),
        (
            scene.sphere(name="head", size=(0.17, 0.17, 0.17), color=(0.8, 0.85, 0.9, 1)),
            (0, 0, 0.5),
        ),
    ]
    for side in (-1, 1):
        for name, offset, size in (
            ("arm", (side * 0.4, 0, 0.05), (0.1, 0.1, 0.32)),
            ("leg", (side * 0.15, 0, -0.63), (0.1, 0.12, 0.32)),
        ):
            part = scene.box(name=f"{name}_{side}", size=size, color=(0.48, 0.59, 0.67, 1))
            parts.append((part, offset))
    for x in range(-6, 7):
        scene.box(name=f"floor_marker_{x}", position=(x, 1.5, 0.04), size=(0.06, 0.06, 0.04))
    return scene, parts


def _pose(parts, t):
    root = np.array((1.2 * np.sin(t * 1.4), 0.35 * np.sin(t * 2), 1.35 + 0.3 * np.sin(t * 18)))
    angle = 0.4 * np.sin(t * 8)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))
    for part, offset in parts:
        part.set_pose(root + rotation @ offset, rotation)
    return root


def main(argv: list[str] | None = None) -> int:
    """Write real UI captures and direct/XYZ/XY tracking movies under output/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/camera-tracking"))
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    scene, parts = _scene()
    _pose(parts, 0)
    with build_scene(
        scene, config=ViewerConfig(), vsync=False, width=1440, height=1000, show_window=False
    ) as viewer:
        viewer.sync()
        node = viewer.session.node_by_object_id(parts[0][0].object_id)
        view = CameraView(eye=np.array((3.6, -5.5, 3.1)), target=np.array((0, 0, 1.35)))
        viewer.set_camera(view)
        exercise_controls(viewer, node.node_id, args.output)
        Image.fromarray(viewer.capture_array(surface=CaptureSurface.WINDOW)).save(
            args.output / "tracking-window.png"
        )
        viewer.session.submit(cmd.Select(0))
        viewer.configure_layers(ViewportLayers(viewport_ui=False))
        advance = viewer.app._sync_camera_tracking
        viewer.app._sync_camera_tracking = lambda _dt: advance(1 / 60)
        report = {}
        try:
            for name, axes, smoothing in (
                ("direct-xyz", "xyz", 0),
                ("smooth-xyz", "xyz", 0.25),
                ("smooth-xy", "xy", 0.25),
            ):
                _pose(parts, 0)
                viewer.set_camera(view)
                viewer.configure_tracking(CameraTrackingConfig(axes=axes, smoothing=smoothing))
                viewer.track_node(node.node_id)
                samples = []
                recorder = None
                try:
                    for frame in range(args.frames):
                        root = _pose(parts, frame / 60)
                        image = viewer.capture_array(surface=CaptureSurface.VIEWPORT)
                        if recorder is None:
                            recorder = VideoRecorder(
                                args.output / f"{name}.mp4",
                                (image.shape[1], image.shape[0]),
                                fps=60,
                            )
                        recorder.append(image)
                        samples.append(
                            {
                                "target": root.tolist(),
                                "camera_target": viewer.session.camera.target.tolist(),
                            }
                        )
                        if frame == args.frames // 2:
                            Image.fromarray(image).save(args.output / f"{name}.png")
                finally:
                    if recorder is not None:
                        recorder.close()
                report[name] = samples
        finally:
            viewer.app._sync_camera_tracking = advance
        (args.output / "motion.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
