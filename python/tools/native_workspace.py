"""Run or capture the production workspace with a representative editable scene."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
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
    """Capture a prepared workspace with current viewport geometry."""
    for _ in range(6):
        viewer.sync()
    target = tuple(
        max(1, int(v)) for v in viewer.window.points_to_pixels(viewer.app.viewport_surface.size)
    )
    if viewer.app._current_viewport_render_size() != target:
        raise RuntimeError("Viewport render target does not match the current content size")
    viewer.sync()
    Image.fromarray(np.asarray(viewer.window.read_frame())[::-1]).save(path)
    return viewer.app.workspace.snapshot()


def float_panel(viewer, name, *, with_panel=None):
    """Float existing production panels inside the actual native work viewport."""
    from imgui_bundle import imgui

    workspace = viewer.app.workspace
    names = (name,) if with_panel is None else (name, with_panel)
    for panel_name in names:
        workspace.activate(panel_name)
    for panel_name in names:
        workspace.open[panel_name] = True
    for _ in range(4):
        viewer.sync()
    windows = [imgui.internal.find_window_by_name(panel_name) for panel_name in names]
    if any(window is None for window in windows):
        raise RuntimeError(f"Production panels did not initialize: {', '.join(names)}")
    viewport, scale = imgui.get_main_viewport(), viewer.window.style_scale
    margin = min(24 * scale, viewport.work_size.x / 4, viewport.work_size.y / 4)
    preferred = (1000, 560) if "Timeline" in names else (560, 640)
    size = tuple(
        min(value * scale, available - margin * 2)
        for value, available in zip(preferred, viewport.work_size, strict=True)
    )
    position = (
        viewport.work_pos.x + (viewport.work_size.x - size[0]) / 2,
        viewport.work_pos.y + (viewport.work_size.y - size[1]) / 2,
    )
    if with_panel is None:
        imgui.internal.dock_context_queue_undock_window(imgui.get_current_context(), windows[0])
        viewer.sync()
        imgui.internal.set_window_pos(windows[0], position)
        imgui.internal.set_window_size(windows[0], size)
        imgui.internal.set_window_collapsed(windows[0], False)
    else:
        node_id = imgui.internal.dock_builder_add_node()
        imgui.internal.dock_builder_set_node_pos(node_id, position)
        imgui.internal.dock_builder_set_node_size(node_id, size)
        for window in windows:
            imgui.internal.dock_builder_dock_window(window.name, node_id)
        imgui.internal.dock_builder_finish(node_id)
    workspace.focus_next = name
    for _ in range(4):
        viewer.sync()


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
    parser.add_argument(
        "--float-panel",
        choices=(*PANELS, "Inspector", "Timeline", "Plot"),
        help="Undock a production panel for interactive native-window inspection",
    )
    parser.add_argument(
        "--float-with",
        choices=(*PANELS, "Inspector", "Timeline", "Plot"),
        help="Group another existing panel with --float-panel in a floating dock",
    )
    parser.add_argument(
        "--review", action="store_true", help="Capture production UI regression scenarios"
    )
    args = parser.parse_args(argv)
    if args.review and not args.capture:
        parser.error("--review requires --capture")
    if args.float_panel and args.capture:
        parser.error("--float-panel requires interactive mode")
    if args.float_with and not args.float_panel:
        parser.error("--float-with requires --float-panel")
    if args.float_with and args.float_with == args.float_panel:
        parser.error("--float-with must name a different panel")
    if args.float_with and "Settings" in (args.float_panel, args.float_with):
        parser.error("Settings is a standalone window and cannot join a floating dock")
    if args.capture:
        args.output.mkdir(parents=True, exist_ok=True)
        os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    viewer = build_workspace(
        resolve(args.asset),
        renderer=args.backend,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False),
            debug_server=False,
            threaded_physics=not args.review,
        ),
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
            if args.float_panel is not None:
                float_panel(viewer, args.float_panel, with_panel=args.float_with)
            viewer.run()
            return 0
        args.output.mkdir(parents=True, exist_ok=True)
        if args.review:
            from mojive.tools.native_ui_review import capture_regressions

            capture_regressions(viewer, args.output)
            print(f"Captured production UI review to {args.output}")
            return 0
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
        ground_style = viewer.app.backend.get_geometry_style()
        try:
            viewer.app.set_geometry_style(replace(ground_style, quiet_ground=True), persist=False)
            states["Quiet ground"] = capture(viewer, args.output / "workspace-quiet-ground.png")
        finally:
            viewer.app.set_geometry_style(ground_style, persist=False)
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
        workspace.menus.preview(("Window", "Panels"))
        states["Panels"] = capture(viewer, args.output / "menu-window-panels.png")
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
