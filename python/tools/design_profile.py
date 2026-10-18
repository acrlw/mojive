"""Compare warmed production UI frames with default and component appearance settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.specification import DEFAULT
from mojive.design.store import SpecificationStore


def appearance_spec():
    """Exercise independent groups and a second family without changing global defaults."""
    value = deepcopy(DEFAULT)
    edits = {
        "rail": {"iconSize": 28, "iconColor": "#9cbf8d"},
        "toolbox": {
            "iconSize": 18,
            "iconColor": "#a9bfe0",
            "separatorThickness": 1.5,
            "separatorGap": 8,
            "separatorInset": 5,
        },
        "viewport_toolbar": {"iconSize": 14, "fontSize": 13, "textColor": "#a9bfe0"},
        "transport": {
            "iconSize": 20,
            "fontFamily": "mono",
            "fontSize": 13,
            "separatorThickness": 1.5,
            "separatorGap": 8,
        },
        "timeline_toolbar": {"iconSize": 16, "fontSize": 13},
        "panel_header": {"fontFamily": "portable", "fontSize": 15, "textColor": "#c3d5b8"},
        "menu_trigger": {"fontFamily": "portable", "fontSize": 13},
        "tree_row": {"fontSize": 13, "iconSize": 14},
        "section_header": {"fontSize": 15, "iconSize": 14},
        "form_label": {"fontFamily": "portable", "fontSize": 13},
        "input_number": {"fontFamily": "system", "fontSize": 13},
        "axis_input": {"fontFamily": "system", "fontSize": 13},
        "scene_panel_header": {
            "fontFamily": "portable",
            "fontSize": 16,
            "iconSize": 23,
            "columnGap": 7,
        },
        "inspector_tabs": {"fontSize": 13, "iconSize": 16, "columnGap": 5},
        "inspector_tabs_tab": {"height": 30, "paddingLeft": 9, "paddingRight": 9},
        "scene_tree_row_tree_label": {"height": 28, "paddingLeft": 8, "paddingRight": 10},
        "scene_segment_button": {"height": 30, "paddingLeft": 7, "paddingRight": 9},
        "scene_type_filters_filter": {"fontFamily": "mono", "fontSize": 12},
        "assets_material": {
            "paddingLeft": 16,
            "paddingRight": 16,
            "paddingTop": 10,
            "paddingBottom": 10,
            "columnGap": 14,
        },
        "assets_material_list": {"rowGap": 12, "actionGap": 16},
        "timeline_toolbar_button": {"height": 30, "paddingLeft": 9, "paddingRight": 11},
        "dock_tabs": {"fontSize": 13, "paddingX": 8, "gap": 5},
        "status_backend": {"fontFamily": "portable", "valueWidth": 52, "rightGap": 5},
        "status_steps": {"valueWidth": 24, "rightGap": 4},
        "status_delta": {"valueWidth": 18, "rightGap": 5},
        "status_physics": {"valueWidth": 12, "rightGap": 4},
        "status_render": {"fontFamily": "system", "valueWidth": 30, "rightGap": 3},
        "status_metrics": {
            "labelGap": 0,
            "unitGap": 1,
            "deltaValueWidth": 18,
            "physicsValueWidth": 12,
            "stepsValueWidth": 24,
            "timeValueWidth": 80,
            "renderValueWidth": 30,
            "tooltipPaddingX": 18,
            "tooltipPaddingY": 12,
            "tooltipRowGap": 10,
        },
    }
    for part, fields in edits.items():
        value["parts"].setdefault(part, {}).update(fields)
    value["components"] = {
        "navigation": {"muted_color": "#a8b6a2"},
        "inspector": {"muted_color": "#b7bdc7"},
    }
    return value


def summary(samples):
    return {name: float(np.percentile(samples, p)) for name, p in (("median", 50), ("p95", 95))}


def font_atlas_state():
    from imgui_bundle import imgui

    atlas = imgui.get_io().fonts
    return len(atlas.fonts), tuple(
        (
            texture.width,
            texture.height,
            texture.bytes_per_pixel,
            hashlib.blake2b(texture.get_pixels_array(), digest_size=16).hexdigest(),
        )
        for texture in atlas.tex_list
    )


def run(args):
    from imgui_bundle import imgui

    from mojive.tools.native_workspace import capture

    store = SpecificationStore(args.output / "design.json")
    store.write(DEFAULT, save=True)
    with (
        patch.dict(
            os.environ,
            {
                "MOJIVE_SETTINGS": str(args.output / "settings.json"),
                "MOJIVE_DESIGN_SPEC": str(store.saved_path),
            },
        ),
        build_workspace(
            resolve("joint_types"),
            renderer=args.backend,
            width=1600,
            height=1000,
            show_window=False,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), threaded_physics=False, debug_server=False
            ),
        ) as viewer,
    ):
        viewer.app.set_language(args.language)
        body = next(node for node in viewer.session.nodes if node.name == "hinge_body")
        viewer.session.submit(cmd.SelectNode(body.node_id))
        viewer.app.tool = "rotate"
        viewer.app.workspace.activate("Timeline")
        runtime = viewer.app.design
        specs = {"default": DEFAULT, "customized": appearance_spec()}
        measurements = {name: {"wall_ms": [], "thread_cpu_ms": []} for name in specs}
        report = {
            "backend": args.backend,
            "platform": platform.platform(),
            "window_points": viewer.window.size_points,
            "framebuffer_scale": viewer.window.pixel_scale,
            "frames_per_round": args.frames,
            "rounds": args.rounds,
            "scope": "Hidden production workspace, paused physics, VSync disabled. Warmed sync() "
            "CPU/wall samples exclude captures and specification changes. Cases alternate to "
            "reduce order bias; these are diagnostic measurements, not display latency. "
            "First application compares reapplying the warm default with the first custom "
            "application, not symmetric cold starts.",
            "cases": {},
        }
        for _ in range(24):
            viewer.sync()
        first_apply = {}
        # Keep initial application and font loading separate from steady frames.
        for name, spec in specs.items():
            wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
            runtime.update_spec(spec)
            viewer.sync()
            first_apply[name] = {
                "wall_ms": (time.perf_counter_ns() - wall) / 1e6,
                "thread_cpu_ms": (time.thread_time_ns() - cpu) / 1e6,
            }
            for _ in range(30):
                viewer.sync()
        for round_index in range(args.rounds):
            order = tuple(specs) if round_index % 2 == 0 else tuple(reversed(specs))
            for name in order:
                runtime.update_spec(specs[name])
                for _ in range(12):
                    viewer.sync()
                atlas_before = font_atlas_state()
                for _ in range(args.frames):
                    wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
                    viewer.sync()
                    measurements[name]["thread_cpu_ms"].append((time.thread_time_ns() - cpu) / 1e6)
                    measurements[name]["wall_ms"].append((time.perf_counter_ns() - wall) / 1e6)
                if font_atlas_state() != atlas_before:
                    raise RuntimeError("A warmed appearance run changed the font atlas")
        for name, spec in specs.items():
            report["cases"][name] = {
                metric: summary(values) for metric, values in measurements[name].items()
            }
            report["cases"][name]["first_apply"] = first_apply[name]
            runtime.update_spec(spec)
            capture(viewer, args.output / f"{name}.png")
            (args.output / f"{name}-spec.json").write_text(
                json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
            )
        report["font_faces_after_warmup"] = len(imgui.get_io().fonts.fonts)
        report["warmed_atlas_pixels_stable"] = True
        report["customized_to_default_cpu_ratio"] = (
            report["cases"]["customized"]["thread_cpu_ms"]["median"]
            / report["cases"]["default"]["thread_cpu_ms"]["median"]
        )
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--language", choices=("en", "zh_CN"), default="zh_CN")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path("output/design-profile") / args.backend
    if args.frames < 12 or args.rounds < 2:
        parser.error("Use at least 12 frames and two alternating rounds")
    args.output.mkdir(parents=True, exist_ok=True)
    report = run(args)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
