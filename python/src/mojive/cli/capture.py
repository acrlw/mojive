"""Cli: capture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mojive.render.backend import RenderFlag

from .common import _resolve


def cmd_capture(args: argparse.Namespace) -> int:
    if bool(args.width) != bool(args.height):
        raise ValueError("Capture width and height must be provided together")
    from mojive.application.composition import capture

    size = (args.width, args.height) if args.width and args.height else None
    output = Path(args.output).expanduser().resolve()
    ok = capture(
        _resolve(args.asset),
        output,
        args.backend,
        include_ui=args.include_ui,
        size=size,
        render_flags=tuple(args.enable_render),
        camera_name=args.camera,
    )
    if not ok:
        print("Capture failed", file=sys.stderr)
        return 1
    print(output)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from mojive.application.composition import build

    viewer = build(
        _resolve(args.asset),
        args.backend,
        vsync=False,
        width=args.width,
        height=args.height,
        title="Mojive recording",
    )
    try:
        for name in args.enable_render:
            viewer.backend.set_flag(RenderFlag(name), True)
        viewer.record(
            Path(args.output),
            frames=args.frames,
            fps=args.fps,
            size=(args.width, args.height),
        )
    finally:
        viewer.release()
    print(args.output)
    return 0


def cmd_keyframes(args: argparse.Namespace) -> int:
    from mojive import commands as cmd
    from mojive.application.composition import build

    viewer = build(
        _resolve(args.asset),
        args.backend,
        paused=True,
        vsync=False,
        width=args.width,
        height=args.height,
        title="Mojive keyframes",
    )
    try:
        for name in args.enable_render:
            viewer.backend.set_flag(RenderFlag(name), True)
        count = len(viewer.session.keyframes)
        if not count:
            print("Model has no keyframes", file=sys.stderr)
            return 1
        viewer.session.submit(cmd.LoadKeyframe(0))
        viewer.app.set_fixed_render_size(args.width, args.height)
        if args.camera:
            camera = next(
                (item for item in viewer.session.cameras if item.name == args.camera), None
            )
            if camera is None:
                raise ValueError(f"model camera {args.camera!r} is unavailable")
            viewer.app.select_model_camera(camera.camera_id)
        viewer.sync()  # compile passes and frame the loaded pose; not part of the video
        if not args.camera:
            viewer.app.camera.distance *= args.camera_distance_scale

        def load(index, current):
            result = current.session.submit(cmd.LoadKeyframe(index))
            if not result.ok:
                raise RuntimeError(result.message)

        viewer.record(
            Path(args.output),
            frames=count,
            fps=args.fps,
            before_frame=load,
            size=(args.width, args.height),
        )
    finally:
        viewer.release()
    print(args.output)
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    import subprocess

    return subprocess.call([sys.executable, "-m", "mojive.tools.probe_gl"])
