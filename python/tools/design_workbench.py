"""Edit the production workspace's native geometry and component parameters."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.editor import SpecificationEditor
from mojive.design.native.window import WorkbenchWindow, run_workbench
from mojive.design.native.workbench import NativeWorkbench
from mojive.design.specification import DEFAULT, component_catalog
from mojive.design.store import SpecificationStore, read_spec
from mojive.tools.native_workspace import capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--asset", default="joint_types")
    scale = float(os.environ.get("MOJIVE_UI_SCALE", "1"))
    parser.add_argument("--width", type=int, default=round(1600 * scale))
    parser.add_argument("--height", type=int, default=round(900 * scale))
    parser.add_argument("--frames", type=int)
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument(
        "--capture", type=Path, help="Save the workspace and parameter editor on exit"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--embedded", dest="detached", action="store_false")
    mode.add_argument("--detached", dest="detached", action="store_true")
    parser.set_defaults(detached=None)
    parser.add_argument("--workbench-width", type=float, default=460)
    parser.add_argument("--part", choices=tuple(component_catalog()), default="timeline_toolbar")
    parser.add_argument("--workbench-page", choices=("交互原型", "设计规范"), default="交互原型")
    parser.add_argument("--language", choices=("zh_CN", "en"), default="zh_CN")
    args = parser.parse_args(argv)
    store = SpecificationStore(args.spec)
    if not store.saved_path.exists() and not store.live_path.exists():
        store.write(DEFAULT, save=True)
    os.environ["MOJIVE_DESIGN_SPEC"] = str(store.saved_path)
    with build_workspace(
        resolve(args.asset),
        renderer=args.backend,
        width=args.width,
        height=args.height,
        show_window=not args.hidden,
        config=ViewerConfig(layout=LayoutConfig(persistence=False), debug_server=False),
    ) as viewer:
        viewer.app.set_language(args.language)
        runtime = viewer.app.design
        editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
        bench = runtime.workbench = NativeWorkbench(
            editor, runtime, runtime, width=args.workbench_width
        )
        bench.page, bench.part = args.workbench_page, args.part
        bench.ready_path = store.saved_path.parent / "workbench-ready.json"
        node = next((node for node in viewer.session.nodes if node.name == "hinge_body"), None)
        if node is not None:
            viewer.session.submit(cmd.SelectNode(node.node_id))
            viewer.app.tool = "rotate"
        detached = args.detached
        if detached is None:
            detached = viewer.app.preferences.get("workbench_detached", True) is not False
        with WorkbenchWindow(
            bench,
            height=args.height,
            scale=viewer.window.style_scale,
            visible=not args.hidden,
            detached=detached,
            renderer=args.backend,
        ) as window:
            run_workbench(viewer, window, max_frames=args.frames)
            if args.capture:
                args.capture.mkdir(parents=True, exist_ok=True)
                state = capture(viewer, args.capture / "workspace.png")
                if bench.detached:
                    Image.fromarray(window.frame(readback=True)[::-1]).save(
                        args.capture / "workbench.png"
                    )
                (args.capture / "geometry.json").write_text(
                    json.dumps(state, ensure_ascii=False, indent=2)
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
