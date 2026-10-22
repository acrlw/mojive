"""Capture right-aligned pending-edit actions in production viewport layouts."""

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import glfw
from imgui_bundle import imgui
from PIL import Image

from mojive.app.composition import build_workspace
from mojive.commands import SetGeometrySize
from mojive.config import LayoutConfig, ViewerConfig
from mojive.scene.assets import resolve
from mojive.tools.native_workspace import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--language", choices=("en", "zh_CN"), default="en")
    parser.add_argument("--output", type=Path, default=Path("output/ui-pending-review"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    os.environ["MOJIVE_UI_SCALE"] = str(args.scale)
    report = {
        "states": {},
        "scale": args.scale,
        "sourceSha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "ui/workspace/viewport.py").read_bytes()
        ).hexdigest(),
    }
    with build_workspace(
        resolve("joint_types"),
        renderer=args.backend,
        paused=True,
        show_window=False,
        vsync=False,
        width=round(1600 * args.scale),
        height=round(1000 * args.scale),
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    ) as v:
        app, shell = v.app, v.app.workspace
        app.set_language(args.language)
        app.design.observe_components = True
        for key in shell.open:
            shell.open[key] = False
        shell.needs_layout = True
        node = next(n for n in v.session.nodes if n.name == "free_box")
        glfw.set_window_size_limits(
            v.window._window, glfw.DONT_CARE, glfw.DONT_CARE, glfw.DONT_CARE, glfw.DONT_CARE
        )
        for name, width, height in (
            ("wide", 850, 700),
            ("stacked", 350, 550),
            ("compact", 360, 240),
        ):
            glfw.set_window_size(
                v.window._window, round(width * args.scale), round(height * args.scale)
            )
            shell.needs_layout = True
            for _ in range(15):
                v.sync()
            assert app.model_edits.stage(SetGeometrySize(node.node_id, (0.14, 0.14, 0.14))).ok
            for _ in range(5):
                v.sync()
            image = args.output / f"{name}.png"
            capture(v, image)
            records = deepcopy(app.design.observed)
            parent = next(row for row in records if row["key"] == "pending-model-edits")
            content = parent["content"]
            hits = {
                key: list(shell.ui.hits[key])
                for key in ("apply-model-edits", "discard-model-edits")
            }
            assert (
                abs(
                    hits["discard-model-edits"][0]
                    + hits["discard-model-edits"][2]
                    - content[0]
                    - content[2]
                )
                < 0.05
            )
            assert hits["apply-model-edits"][0] >= content[0]
            x, y, w, h = parent["hit"]
            with Image.open(image) as pixels:
                density = pixels.width / imgui.get_io().display_size.x
                pixels.crop(
                    (
                        int(x * density),
                        int(y * density),
                        int((x + w) * density + 1),
                        int((y + h) * density + 1),
                    )
                ).save(args.output / f"{name}-detail.png")
            report["states"][name] = {"record": parent, "actionHits": hits}
            app.model_edits.clear()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"3 pending-edit layouts passed at {args.scale:g}x")


if __name__ == "__main__":
    main()
