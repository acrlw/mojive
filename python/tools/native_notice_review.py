"""Capture the actual production notice family without publishing capture receipts.

Use ``make ui-notice-review`` with ``BACKEND`` and ``ARGS`` for renderer,
language and output selection. ``MOJIVE_UI_SCALE`` selects the actual UI scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

from imgui_bundle import imgui
from PIL import Image

from mojive.app.composition import build_workspace
from mojive.config import LayoutConfig, ViewerConfig
from mojive.scene.assets import resolve
from mojive.tools.native_workspace import capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--language", choices=("en", "zh_CN"), default="en")
    parser.add_argument("--output", type=Path, default=Path("output/ui_design/notices"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    report = {"backend": args.backend, "language": args.language, "states": {}}
    source = Path(__file__).resolve().parents[1] / "ui/workspace/notices.py"
    report["sourceSha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    with build_workspace(
        resolve("joint_gizmo"),
        renderer=args.backend,
        paused=True,
        vsync=False,
        show_window=False,
        width=1600,
        height=1000,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    ) as viewer:
        app, workspace = viewer.app, viewer.app.workspace
        app.set_language(args.language)
        workspace.collapse_bottom()
        workspace.open["Settings"] = False
        app.design.observe_components = True
        ui = workspace.ui
        report["scale"] = ui.s
        report["windowPoints"] = list(viewer.window.size_points)
        text_calls = []
        original_text = ui.text

        def text(x, y, value, tint, size=14, **options):
            text_calls.append(
                {
                    "text": value,
                    "position": [x, y],
                    "fontSize": size,
                    "frame": imgui.get_frame_count(),
                }
            )
            return original_text(x, y, value, tint, size, **options)

        ui.text = text

        def save(name):
            for _ in range(5):
                viewer.sync()
            text_calls.clear()
            full = args.output / (name + ".png")
            capture(viewer, full)
            records = deepcopy(app.design.observed)
            selected = [
                record
                for record in records
                if record["key"] in ("operation-notice", "file-receipt")
            ]
            assert len(selected) == 1, (name, selected)
            record = selected[0]
            assert app._status_notice_bounds is not None
            x, y, right, bottom = app._status_notice_bounds
            vx, vy, width, height = app._viewport_rect
            assert vx <= x < right <= vx + width and vy <= y < bottom <= vy + height
            with Image.open(full) as image:
                density_x = image.width / imgui.get_io().display_size.x
                density_y = image.height / imgui.get_io().display_size.y
                crop = (
                    max(0, int((x - 6) * density_x)),
                    max(0, int((y - 6) * density_y)),
                    min(image.width, int((right + 6) * density_x + 1)),
                    min(image.height, int((bottom + 6) * density_y + 1)),
                )
                image.crop(crop).save(args.output / (name + "-detail.png"))
            report["states"][name] = {
                "full": full.name,
                "detail": name + "-detail.png",
                "record": record,
                "textCalls": [
                    item
                    for item in text_calls
                    if item["frame"] == record["submittedFrame"]
                    and x <= item["position"][0] < right
                    and y <= item["position"][1] < bottom
                ],
                "actionHits": {
                    key: ui.hits[key]
                    for key in ("receipt-open-folder", "receipt-copy-path")
                    if key in ui.hits
                },
            }
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

        for _ in range(24):
            viewer.sync()
        for level in ("info", "warning", "error"):
            caption = {
                "info": "Image copied to clipboard",
                "warning": "Clipboard copy failed",
                "error": "Capture failed",
            }[level]
            app.output.publish(app.localizer.text(caption), level=level, duration=None)
            save(level)
        for name, filename in (
            ("receipt-inline", "shot.png"),
            ("receipt-stacked", "long-recording-name-" * 8 + ".png"),
        ):
            path = args.output / filename
            # File notices deliberately require a real path before offering actions.
            Image.new("RGB", (1, 1)).save(path)
            app.output.publish(
                app.localizer.text("Saved capture to") + " " + str(path),
                duration=None,
                copy_text=str(path),
            )
            save(name)
            record = report["states"][name]["record"]
            assert record["inline"] == (name == "receipt-inline"), record
        ui.text = original_text
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
