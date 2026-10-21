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

from mojive import CameraView, InputClaim, ViewerConfig
from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.config import LayoutConfig, ViewportLayers
from mojive.design.store import default_spec_path
from mojive.scene.assets import resolve

from ._harness import OffscreenHarness
from .keyframe_timeline import populate_take
from .showcase import _add_debug_layers
from .ui_runtime import _park_cursor, _save, _settle


def _capture_editor(
    output: Path, width: int, height: int, *, interactive=False, ui_scale=1.2
) -> dict:
    config = ViewerConfig(
        layout=LayoutConfig(persistence=False),
        layers=ViewportLayers(helpers=False),
        threaded_physics=False,
    )
    with build_workspace(
        resolve("joint_types"),
        config=config,
        renderer="opengl",
        paused=True,
        vsync=interactive,
        width=width,
        height=height,
        show_window=False,
    ) as viewer:
        if interactive:
            viewer.set_input_handler(lambda _context: InputClaim(keys=frozenset({"f8"})))
        _settle(viewer)
        # GLFW window units can differ from framebuffer pixels on HiDPI displays.
        import glfw

        pixel_width, pixel_height = viewer.window.size_pixels
        logical_width, logical_height = glfw.get_window_size(viewer.window._window)
        # Keep the same physical font and control sizes across display densities.
        viewer.window._scale_override = ui_scale * logical_width / pixel_width
        glfw.set_window_size(
            viewer.window._window,
            round(logical_width * width / pixel_width),
            round(logical_height * height / pixel_height),
        )
        viewer.app.set_language("en")
        _settle(viewer)
        workspace = viewer.app.workspace
        workspace.needs_layout = True
        workspace.activate("Scene")
        workspace.activate("Timeline")
        _settle(viewer)
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
        viewer.app.output.active_status(float("inf"))
        _settle(viewer)
        _fit_timeline(viewer)
        if interactive:
            viewer.window.show()
            _wait_for_capture(viewer, "hero.png")
        layout = _layout_report(viewer)
        _save(viewer, output / "hero.png")
        workspace.activate("Pose")
        workspace.inspector_tab = "Properties"
        viewer.set_camera(
            CameraView(
                eye=np.array([2.2, -2.7, 1.9]),
                target=np.array([0.9, 0.2, 0.52]),
                up=np.array([0.0, 0.0, 1.0]),
            )
        )
        _settle(viewer)
        if interactive:
            _wait_for_capture(viewer, "joint-authoring.png")
        _layout_report(viewer, sidebar="Pose")
        _save(viewer, output / "joint-authoring.png")
        return {
            "scene": "joint_types",
            "joint_count": len(session.joints),
            "visible_joint_filter": "all",
            "selected_node": selected.name,
            "keyframes": ["Start", "Reach", "Hold", "Return"],
            "logical_ui_scale": viewer.window.style_scale,
            "physical_ui_scale": viewer.window.ui_scale,
            "framebuffer_scale": list(imgui.get_io().display_framebuffer_scale),
            "font_scale_dpi": imgui.get_style().font_scale_dpi,
            "layout": layout,
        }


def _fit_timeline(viewer) -> None:
    """Resize the capture's dock split to include all three production tracks."""
    workspace = viewer.app.workspace
    timeline = imgui.internal.find_window_by_name("Timeline")
    viewport = imgui.internal.find_window_by_name("Viewport")
    track = workspace.timeline_panel.last_layout
    extra = track.hi[1] + 12 * workspace.s - timeline.inner_rect.max.y
    if extra > 0:
        for window, delta in ((timeline, extra), (viewport, -extra)):
            node = window.dock_node
            imgui.internal.dock_builder_set_node_size(
                node.id_, imgui.ImVec2(node.size.x, node.size.y + delta)
            )
        _settle(viewer)


def _layout_report(viewer, *, sidebar="Scene") -> dict:
    """Verify production panels and viewport overlays before publication."""
    names = (sidebar, "Viewport", "Inspector", "Timeline")
    bounds = {}
    for name in names:
        window = imgui.internal.find_window_by_name(name)
        if window is None or not window.active or window.hidden:
            raise RuntimeError(f"Required screenshot panel is not visible: {name}")
        bounds[name] = [window.pos.x, window.pos.y, window.size.x, window.size.y]
    workspace = viewer.app.workspace
    x, y, width, height = workspace.rects["viewport_content"]
    for left, top, w, h in workspace.rects["overlays"]:
        if left < x - 1 or top < y - 1 or left + w > x + width + 1 or top + h > y + height + 1:
            raise RuntimeError("A production viewport overlay is clipped")
    timeline = imgui.internal.find_window_by_name("Timeline")
    if workspace.timeline_panel.last_layout.hi[1] > timeline.inner_rect.max.y + 1:
        raise RuntimeError("The production Timeline tracks are clipped")
    return bounds


def _wait_for_capture(viewer, name):
    print(f"Adjust the layout and camera for {name}; press F8 to capture and continue.", flush=True)
    while viewer.is_running():
        viewer.sync()
        if imgui.is_key_pressed(imgui.Key.f8, False):
            _park_cursor(viewer)
            _settle(viewer)
            return
    raise SystemExit("Capture cancelled; README images were not published.")


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
    parser.add_argument("--ui-scale", type=float, default=1.2)
    parser.add_argument("--spec", type=Path, help="Use a saved UI design specification")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Adjust each editor view, then press F8 to capture",
    )
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or not 0.5 <= args.ui_scale <= 4:
        parser.error("Image dimensions must be positive; UI scale must be between 0.5 and 4")
    if args.spec and not args.spec.is_file():
        parser.error(f"Design specification does not exist: {args.spec}")
    # Resolve the user's design before isolating the capture's editor preferences.
    os.environ["MOJIVE_DESIGN_SPEC"] = str(
        args.spec.resolve() if args.spec else default_spec_path()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep capture settings separate while reading the user's saved design specification.
    with tempfile.TemporaryDirectory(prefix="settings-", dir=args.output) as settings:
        os.environ["MOJIVE_SETTINGS"] = str(Path(settings) / "settings.json")
        os.environ["MOJIVE_UI_SCALE"] = "1"
        os.environ["MOJIVE_RENDERER"] = "opengl"
        report = _capture_editor(
            args.output,
            args.width,
            args.height,
            interactive=args.interactive,
            ui_scale=args.ui_scale,
        )
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
