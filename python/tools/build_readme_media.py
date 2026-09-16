"""Capture the production editor and renderer at one README image size."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive import CameraView, ViewerConfig, build
from mojive import commands as cmd
from mojive.config import LayoutConfig, ViewportLayers
from mojive.scene.assets import resolve

from ._harness import OffscreenHarness
from .keyframe_timeline import populate_take
from .showcase import _add_debug_layers
from .ui_runtime import _activate_panel, _park_cursor, _save, _settle


def _build_layout(viewer) -> None:
    """Build docking nodes inside the window's active ImGui frame."""
    root = viewer.window.dockspace_id
    dock = imgui.internal
    dock.dock_builder_remove_node(root)
    dock.dock_builder_add_node(root, dock.DockNodeFlagsPrivate_.dock_space)
    dock.dock_builder_set_node_size(root, imgui.get_main_viewport().work_size)
    left_share, right_share = 0.205, 0.235
    _, right, rest = dock.dock_builder_split_node_py(root, imgui.Dir.right, right_share)
    _, bottom, upper = dock.dock_builder_split_node_py(rest, imgui.Dir.down, 0.24)
    _, left, center = dock.dock_builder_split_node_py(
        upper, imgui.Dir.left, left_share / (1.0 - right_share)
    )
    _, inspector, joints = dock.dock_builder_split_node_py(right, imgui.Dir.down, 0.48)
    for name, node in {
        "Hierarchy": left,
        "Assets": left,
        "Viewport": center,
        "Keyframes": bottom,
        "Joints": joints,
        "Inspector": inspector,
    }.items():
        dock.dock_builder_dock_window(name, node)
    dock.dock_builder_finish(root)


def _layout(viewer) -> None:
    """Keep the real panels docked, with Output reduced to its status strip."""
    for panel in viewer.panels:
        panel.open = panel.name in {"Hierarchy", "Joints", "Inspector", "Keyframes", "Output"}
    original = viewer.window._build_default_layout
    viewer.window._build_default_layout = lambda: _build_layout(viewer)
    try:
        viewer.sync()
    finally:
        viewer.window._build_default_layout = original
    viewer.panels.get("Output").collapsed = True
    _settle(viewer)
    for name in ("Hierarchy", "Joints", "Inspector", "Keyframes"):
        _activate_panel(viewer, name)


def _capture_editor(output: Path, width: int, height: int) -> dict:
    config = ViewerConfig(
        layout=LayoutConfig(persistence=False),
        layers=ViewportLayers(helpers=False),
        threaded_physics=False,
    )
    with build(
        resolve("joint_types"),
        config=config,
        renderer="opengl",
        paused=True,
        vsync=False,
        width=width,
        height=height,
        show_window=False,
    ) as viewer:
        _settle(viewer)
        # GLFW window units can differ from framebuffer pixels on HiDPI displays.
        import glfw

        pixel_width, pixel_height = viewer.window.size_pixels
        logical_width, logical_height = glfw.get_window_size(viewer.window._window)
        # Keep the same physical font and control sizes across display densities.
        viewer.window._scale_override = 1.3 * logical_width / pixel_width
        glfw.set_window_size(
            viewer.window._window,
            round(logical_width * width / pixel_width),
            round(logical_height * height / pixel_height),
        )
        viewer.app.set_language("en")
        _settle(viewer)
        _layout(viewer)
        session = viewer.session
        joints = {joint.name: joint for joint in session.joints}
        for name, value in (("hinge_limited", -0.55), ("chain_0", 0.4), ("chain_1", -0.7)):
            if not session.submit(cmd.SetQpos(joints[name].qpos_adr, value)):
                raise RuntimeError(f"Could not pose {name}")
        populate_take(viewer, count=240)
        model_id = session.scene_models[0].model_id
        for index, name in ((0, "Start"), (60, "Reach"), (120, "Hold"), (210, "Return")):
            if not session.submit(cmd.SeekStateTake(index)) or not session.submit(
                cmd.AddModelKeyframe(model_id, name)
            ):
                raise RuntimeError(f"Could not capture keyframe {name}")
        session.submit(cmd.SeekStateTake(120))
        timeline = viewer.panels.get("Keyframes")
        timeline.editor.view_start, timeline.editor.view_end = -0.5, 9.0
        timeline.editor.set_follow_mode("off")
        viewer.panels.get("Joints")._angular_degrees = True
        selected = next(node for node in session.nodes if node.name == "hinge_body")
        session.submit(cmd.SelectNode(selected.node_id))
        session.submit(cmd.CaptureSceneSnapshot())
        hierarchy = viewer.panels.get("Hierarchy")
        hierarchy._open_state = {
            node.node_id: node.name in {"world", "hinge_anchor"} for node in session.nodes
        }
        viewer.set_gizmo_mode("rotate")
        viewer.set_camera(
            CameraView(
                eye=np.array([2.0, -4.5, 3.0]),
                target=np.array([0.1, 0.35, 0.48]),
                up=np.array([0.0, 0.0, 1.0]),
            )
        )
        _park_cursor(viewer)
        _settle(viewer)
        layout = _layout_report(viewer)
        _save(viewer, output / "hero.png")
        viewer.set_camera(
            CameraView(
                eye=np.array([2.2, -2.7, 1.9]),
                target=np.array([0.9, 0.2, 0.52]),
                up=np.array([0.0, 0.0, 1.0]),
            )
        )
        _settle(viewer)
        _save(viewer, output / "joint-authoring.png")
        return {
            "scene": "joint_types",
            "joint_count": len(session.joints),
            "visible_joint_filter": "all",
            "selected_node": selected.name,
            "keyframes": ["Start", "Reach", "Hold", "Return"],
            "output_collapsed": viewer.panels.get("Output").collapsed,
            "logical_ui_scale": viewer.window.style_scale,
            "physical_ui_scale": viewer.window.ui_scale,
            "layout": layout,
        }


def _layout_report(viewer) -> dict:
    """Reject hidden panels or clipped capsules before publishing a capture."""
    names = ("Hierarchy", "Viewport", "Joints", "Inspector", "Keyframes")
    bounds = {}
    for name in (*names, "Playback###viewport_playback", "Tools###viewport_tools"):
        window = imgui.internal.find_window_by_name(name)
        if window is None or not window.active:
            raise RuntimeError(f"Required screenshot panel is not visible: {name}")
        bounds[name] = [window.pos.x, window.pos.y, window.size.x, window.size.y]
    x, y, width, height = viewer.app._viewport_rect
    for name in ("Playback###viewport_playback", "Tools###viewport_tools"):
        left, top, w, h = bounds[name]
        if left < x or top < y or left + w > x + width or top + h > y + height:
            raise RuntimeError(f"Screenshot capsule is clipped: {name}")
    return bounds


def _capture_rendering(output: Path, width: int, height: int) -> None:
    """Frame the showcase around its objects instead of the infinite ground."""
    with OffscreenHarness(resolve("showcase"), width, height) as harness:
        harness.backend.set_camera(
            CameraView(
                eye=np.array([0.0, -10.0, 8.5]),
                target=np.array([0.0, -0.4, 0.3]),
                up=np.array([0.0, 0.0, 1.0]),
                fov_y=np.deg2rad(40.0),
            )
        )
        harness.warmup(4)
        _add_debug_layers(harness)
        harness.step_and_render(20)
        harness.save_png(output / "rendering.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/readme-media"))
    parser.add_argument(
        "--publish", type=Path, help="Copy verified captures to documentation media"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    # A capture must not read or overwrite the user's persisted desktop settings.
    with tempfile.TemporaryDirectory(prefix="settings-", dir=args.output) as settings:
        os.environ["MOJIVE_SETTINGS"] = str(Path(settings) / "settings.json")
        os.environ["MOJIVE_UI_SCALE"] = "0.6"
        os.environ["MOJIVE_RENDERER"] = "opengl"
        report = _capture_editor(args.output, args.width, args.height)
        _capture_rendering(args.output, args.width, args.height)
    names = ("hero.png", "joint-authoring.png", "rendering.png")
    for name in names:
        source = args.output / name
        with Image.open(source) as image:
            if image.size != (args.width, args.height):
                raise RuntimeError(f"Unexpected capture size for {source}: {image.size}")
        if args.publish:
            args.publish.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, args.publish / name)
        print(source.resolve())
    report["image_size"] = [args.width, args.height]
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
