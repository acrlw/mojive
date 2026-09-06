"""Capture responsive native layouts across widths, languages, and UI scales."""

from __future__ import annotations

import argparse
import json
import os
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

from imgui_bundle import imgui

from .. import commands as cmd
from ..assets import resolve
from ..composition import build
from ..gizmo import GizmoHandle
from .ui_runtime import (
    _dismiss_popup,
    _item_rect,
    _open_main_menu,
    _park_cursor,
    _save_active_popup_crop,
    _save_window_crop,
    _settle,
)


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
            width=1600,
            height=1100,
            show_window=False,
        ) as viewer,
    ):
        _settle(viewer, 8)
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
        free = next(node for node in viewer.session.nodes if node.name == "04_free")
        viewer.session.submit(cmd.SelectNode(free.node_id))
        viewer.app.gizmo.set_mode("rotate")
        _settle(viewer, 3)
        viewer.app.gizmo._hovered = GizmoHandle.ROTATE_Z
        edit = viewer.app.gizmo.precise_input(viewer.session)
        assert edit is not None
        viewer.app._begin_precise_gizmo_input(edit)
        _settle(viewer, 3)
        _save_active_popup_crop(viewer, folder / "precise-input.png")
        _dismiss_popup(viewer)
        viewer.session.submit(cmd.SelectNode(link.node_id))

        cases = [("Camera", w, "") for w in (140, 180, 240, 360)]
        cases += [
            ("Settings", w, category)
            for category in ("General", "Interaction", "Rendering")
            for w in (360, 720, 960)
        ]
        cases += [("Inspector", w, "") for w in (180, 230, 320, 480)]
        cases += [("Keyframes", w, "") for w in (230, 480)]
        cases += [("Output", w, "") for w in (180, 320)]
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
            _park_cursor(viewer)
            _settle(viewer, 4)
            filename = f"{target.lower()}-{width}-{category.lower() or 'layout'}.png"
            _save_window_crop(viewer, target, folder / filename, padding=3.0)
            if target == "Camera":
                labels = ("##camera-projection-0", "##camera-projection-1")
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
                    _item_rect(viewer, "combo", "##output-level"),
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
            panel.open = False
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
