"""Capture the selected object's compact Inspector transform controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from imgui_bundle import imgui
from PIL import Image

from .. import commands as cmd
from ..assets import resolve
from ..composition import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/inspector/transform.png"),
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    viewer = build(resolve("gizmo"), paused=True, vsync=False, width=1280, height=800)
    try:
        node = next(node for node in viewer.session.nodes if node.posable)
        viewer.session.submit(cmd.Select(node.object_id))
        for _ in range(6):
            viewer.sync()
        _activate_panel(viewer, "Inspector")
        for _ in range(3):
            viewer.sync()
        _capture_panel(viewer, "Inspector", args.output)
    finally:
        viewer.release()

    print(args.output.resolve())
    return 0


def _activate_panel(viewer, name: str) -> None:
    panel = imgui.internal.find_window_by_name(name)
    if panel is None or panel.dock_node is None:
        return
    node = panel.dock_node
    if node.selected_tab_id == panel.tab_id:
        return
    tab = next(tab for tab in node.tab_bar.tabs if tab.id_ == panel.tab_id)
    bar = node.tab_bar.bar_rect
    io = imgui.get_io()
    io.add_mouse_pos_event(
        bar.min.x + tab.offset + tab.width * 0.5,
        (bar.min.y + bar.max.y) * 0.5,
    )
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()


def _capture_panel(viewer, name: str, output: Path) -> None:
    panel = imgui.internal.find_window_by_name(name)
    if panel is None:
        raise RuntimeError(f"Panel is unavailable: {name}")
    pixels = viewer.window.read_frame()[::-1, :, :3]
    scale = viewer.window.pixel_scale
    x0 = max(0, round(panel.pos.x * scale))
    y0 = max(0, round(panel.pos.y * scale))
    x1 = min(pixels.shape[1], round((panel.pos.x + panel.size.x) * scale))
    y1 = min(pixels.shape[0], round((panel.pos.y + panel.size.y) * scale))
    Image.fromarray(pixels[y0:y1, x0:x1], "RGB").save(output)


if __name__ == "__main__":
    raise SystemExit(main())
