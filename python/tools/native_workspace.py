"""Run or capture the production workspace with a representative editable scene."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.types import CameraView
from mojive.ui.workspace.constants import PANELS
from mojive.ui.workspace.design import DEFAULT_DESIGN


def capture(viewer, path: Path):
    """Capture after the render target has caught up with the docked viewport."""
    for _ in range(6):
        viewer.sync()
    deadline = time.perf_counter() + viewer.window.latch.settle_seconds + 2
    while time.perf_counter() < deadline:
        target = tuple(
            max(1, int(v)) for v in viewer.window.points_to_pixels(viewer.app.viewport_surface.size)
        )
        if viewer.window.latch.committed == target:
            break
        time.sleep(0.01)
        viewer.sync()
    else:
        raise RuntimeError("Viewport render target did not settle before capture")
    viewer.sync()
    Image.fromarray(np.asarray(viewer.window.read_frame())[::-1]).save(path)
    return viewer.app.workspace.snapshot()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--interactive", action="store_true", help="Open the production workspace (default)"
    )
    mode.add_argument("--capture", action="store_true", help="Capture panels and menus, then exit")
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--asset", default="joint_types")
    parser.add_argument("--output", type=Path, default=Path("output/native-workspace"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--language", choices=("en", "zh_CN"), default="zh_CN")
    parser.add_argument("--panel", choices=(*PANELS, "Inspector", "Timeline", "Plot"))
    args = parser.parse_args(argv)
    if args.capture:
        args.output.mkdir(parents=True, exist_ok=True)
        os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    viewer = build_workspace(
        resolve(args.asset),
        renderer=args.backend,
        config=ViewerConfig(layout=LayoutConfig(persistence=False), debug_server=False),
        width=args.width,
        height=args.height,
        show_window=not args.capture,
    )
    try:
        viewer.app.set_language(args.language)
        if args.asset == "joint_types":
            # Keep design acceptance comparable without changing application camera defaults.
            viewer.app.camera.adopt(
                CameraView(
                    eye=np.array([4.3, -5.9, 4.2]),
                    target=np.array([0, 0, 0.55]),
                    fov_y=np.radians(42),
                    near=0.05,
                    far=100,
                )
            )
            viewer.app._initial_camera_set = True
            viewer.app.tool = "rotate"
            node = next((n for n in viewer.session.nodes if n.name == "hinge_body"), None)
            if node is not None:
                viewer.session.submit(cmd.SelectNode(node.node_id))
        if not args.capture:
            if args.panel is not None:
                viewer.app.workspace.activate(args.panel)
            viewer.run()
            return 0
        args.output.mkdir(parents=True, exist_ok=True)
        workspace = viewer.app.workspace
        report = {
            "backend": args.backend,
            "size": [args.width, args.height],
            "scale": viewer.window.style_scale,
            "design": asdict(viewer.app.design.dimensions(DEFAULT_DESIGN)),
            "states": {},
        }
        states = report["states"]
        states["workspace"] = capture(viewer, args.output / "workspace.png")
        for name in ("Control", "Assets", "Pose", "Cameras", "Sensors", "Layers", "Settings"):
            workspace.activate(name)
            states[name] = capture(viewer, args.output / (name.lower() + ".png"))
            if name == "Cameras":
                from mojive.tools.camera_tracking import click

                def click_widget(key):
                    x, y, width, height = workspace.ui.hits[key]
                    click(viewer, (x + width / 2, y + height / 2))

                direction = "section-" + viewer.app._panel_context().tr("VIEW DIRECTION")
                click_widget(direction)
                states["Camera parameters"] = capture(viewer, args.output / "camera-parameters.png")
                click_widget("##camera-yaw-unit")
                states["Camera radians"] = capture(viewer, args.output / "camera-radians.png")
                click_widget("##camera-yaw-unit")
                if viewer.session.cameras:
                    viewer.app.select_model_camera(viewer.session.cameras[0].camera_id)
                    states["Camera source"] = capture(
                        viewer, args.output / "camera-model-source.png"
                    )
                    click_widget("camera-editor")
                click_widget(direction)
            if name == "Settings":
                from mojive.ui.workspace.settings_panel import CATEGORIES

                for category in CATEGORIES:
                    workspace.settings.show_category(category)
                    states["Settings " + category] = capture(
                        viewer,
                        args.output / ("settings-" + category.lower().replace(" ", "-") + ".png"),
                    )
                    from imgui_bundle import imgui

                    content = next(
                        child
                        for child in imgui.get_current_context().windows
                        if child.active and "Reference settings content" in child.name
                    )
                    if content.scroll_max.y > 0:
                        imgui.internal.set_scroll_y(content, content.scroll_max.y)
                        states["Settings bottom " + category] = capture(
                            viewer,
                            args.output
                            / ("settings-" + category.lower().replace(" ", "-") + "-bottom.png"),
                        )
                        imgui.internal.set_scroll_y(content, 0)
                workspace.open[name] = False
        workspace.activate("Assets")
        assets = viewer.panels.get("assets")
        model = viewer.session.scene_models[0] if viewer.session.scene_models else None
        if model is not None:
            inventory = viewer.session.model_assets(model.model_id)
            material = next((item for item in inventory if item.type == "material"), None)
            if material is not None:
                assets.focus(model.model_id, material.type, material.name)
                states["Model assets"] = capture(viewer, args.output / "assets-inventory.png")
                from imgui_bundle import imgui

                content = next(
                    child
                    for child in imgui.get_current_context().windows
                    if child.active
                    and child.parent_window is not None
                    and child.parent_window.id_ == imgui.internal.find_window_by_name("Assets").id_
                    and "Panel content" in child.name
                )
                imgui.internal.set_scroll_y(content, content.scroll_max.y)
                states["Asset material"] = capture(viewer, args.output / "assets-material.png")
        viewer.panels.open_panel("Plot")
        states["Plot joints"] = capture(viewer, args.output / "plot-joints.png")
        viewer.panels.get("plot").source = "sensor"
        states["Plot sensors"] = capture(viewer, args.output / "plot-sensors.png")
        viewer.panels.get("plot").open = False
        workspace.activate("Scene")
        for tab in ("Properties", "Physics", "Material", "Joint"):
            workspace.inspector_tab = tab
            states[tab] = capture(viewer, args.output / ("inspector-" + tab.lower() + ".png"))
        workspace.activate("Timeline")
        states["Timeline"] = capture(viewer, args.output / "timeline.png")
        for menu in ("File", "Edit", "Add", "View", "Simulate", "Window", "Help"):
            workspace.menus.preview((menu,))
            states[menu] = capture(viewer, args.output / ("menu-" + menu.lower() + ".png"))
        geometry = next((n for n in viewer.session.nodes if n.name == "free_box"), None)
        if geometry is not None:
            viewer.session.submit(cmd.SelectNode(geometry.node_id))
            workspace.inspector_tab = "Material"
            workspace.menus.color_editor.mode = 2
            workspace.menus.preview(("Color",))
            states["Color"] = capture(viewer, args.output / "color-rgba.png")
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Captured production workspace to {args.output}")
    finally:
        viewer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
